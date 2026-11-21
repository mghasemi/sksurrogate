"""Temporal baseline bundles (plan phase P4.2).

Before this path existed, ``POST /api/bundles/{task}/train-baseline`` on a
temporal task produced a *tabular* bundle: a scaler + regressor fitted on the
feature columns with the time column dropped. Such a bundle could not answer
``predict_forecast`` at all, and no naive baseline was ever recorded next to a
forecast metric.

Contract under test:

* a temporal task's baseline bundle is a ``Pipeline`` whose first step is the
  ``ForecastFeatureBuilder`` and whose record carries the serving geometry;
* ``baseline_scores`` (naive, seasonal naive, drift, mean on the *same* origins)
  is stored on the bundle and echoed by the route — plan P4.2's acceptance line;
* the same bundle serves a multi-step forecast through
  ``POST /api/inference/{task}/forecast``;
* a tabular task's baseline bundle is unchanged (no ``temporal`` block).
"""

import importlib.util
import io
import json

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from SKSurrogate import load_bundle  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402

TASK = "temporal-baseline"
TABULAR_TASK = "tabular-baseline"
HORIZON = 3


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    return TestClient(app)


def _frame(rows=90, season=4):
    t = np.arange(1, rows + 1, dtype=float)
    # Trend + a deterministic seasonal component: seasonal naive is a real
    # competitor here, so a baseline ranking that ignored it would be wrong.
    y = 0.4 * t + 3.0 * np.sin(2.0 * np.pi * t / season)
    return pd.DataFrame({"time": t, "y": y})


def _csv(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _register(client, frame, task=TASK, temporal=True):
    data = {"target": "y", "partition": "train"}
    if temporal:
        data["temporal"] = json.dumps(
            {"time_column": "time", "horizon": HORIZON}
        )
    response = client.post(
        f"/api/datasets/{task}/register",
        data=data,
        files={"file": ("train.csv", _csv(frame), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_temporal_baseline_bundle_carries_geometry_and_baselines(client):
    _register(client, _frame())
    response = client.post(
        f"/api/bundles/{TASK}/train-baseline",
        json={"estimator": "linear_regression", "seasonality": 4},
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    temporal = payload["temporal"]
    assert isinstance(temporal, dict)
    assert temporal["strategy"] == "direct"
    assert temporal["spec"]["time_column"] == "time"
    assert temporal["spec"]["horizon"] == HORIZON
    assert temporal["feature_spec"]  # builder configuration recorded

    baselines = payload["baseline_scores"]
    for method in ("naive", "seasonal_naive", "drift", "mean"):
        assert method in baselines, method
        assert baselines[method]["rows"] > 0
    # The chosen model's own metrics come from the same origins.
    for metric in ("mae", "rmse", "mase"):
        assert metric in payload["metrics"], metric

    bundle = load_bundle(payload["bundle_path"])
    assert bundle.temporal == temporal
    assert (bundle.metadata or {})["baseline_scores"] == baselines
    assert bundle.metadata["backtest"]["available"] is True


def test_temporal_baseline_bundle_serves_a_forecast(client):
    frame = _frame()
    _register(client, frame)
    response = client.post(
        f"/api/bundles/{TASK}/train-baseline",
        json={"estimator": "linear_regression", "seasonality": 4},
    )
    assert response.status_code == 200, response.text
    version = response.json()["model_version"]

    history = [
        {"time": float(row.time), "y": float(row.y)}
        for row in frame.itertuples(index=False)
    ]
    forecast = client.post(
        f"/api/inference/{TASK}/forecast",
        json={"model_version": version, "history": history, "horizon": HORIZON},
    )
    assert forecast.status_code == 200, forecast.text
    rows = forecast.json()
    assert rows["strategy"] == "direct"
    assert rows["horizon"] == HORIZON
    assert len(rows["predictions"]) == HORIZON
    assert [float(time) for time in rows["times"]] == pytest.approx(
        [float(frame["time"].iloc[-1]) + step for step in range(1, HORIZON + 1)]
    )
    # Intervals appear only when the bundle carries a calibration; a plain
    # baseline does not claim one.
    assert all(value is not None for value in rows["predictions"])
    assert rows["lower_bounds"] is None and rows["upper_bounds"] is None


def test_recursive_temporal_baseline_records_the_strategy(client):
    _register(client, _frame())
    response = client.post(
        f"/api/bundles/{TASK}/train-baseline",
        json={"estimator": "linear_regression", "strategy": "recursive", "seasonality": 4},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["temporal"]["strategy"] == "recursive"
    assert payload["baseline_scores"]["naive"]["rows"] > 0


def test_unknown_strategy_is_refused(client):
    _register(client, _frame())
    response = client.post(
        f"/api/bundles/{TASK}/train-baseline", json={"strategy": "two-step"}
    )
    assert response.status_code == 400


def test_tabular_baseline_bundle_is_unchanged(client):
    frame = pd.DataFrame({"a": np.arange(60, dtype=float), "y": np.arange(60) * 2.0})
    _register(client, frame, task=TABULAR_TASK, temporal=False)
    response = client.post(
        f"/api/bundles/{TABULAR_TASK}/train-baseline", json={}
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["temporal"] is None
    assert payload["baseline_scores"] == {}
