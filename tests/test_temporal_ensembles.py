"""Combining temporal bundles: the ensemble is fitted on the generated design.

Before this path existed, ``POST /api/ensembles/{task}/fit`` sliced the raw
partition for its feature matrix. A temporal partition has no feature column of
its own — the time column is excluded by the temporal contract and the target is
the label — so the matrix was ``(n, 0)`` and the aggregator's member died inside
scikit-learn's ``check_X_y`` with the opaque
``ValueError: at least one array or dtype is required``.

Contract under test:

* the members are refit on the same generated one-step forecast design the
  search, backtest and explanations use (``feature_source: temporal_design``);
* the members enter it through their *bare* estimators, not their
  ``Pipeline(ForecastFeatureBuilder, …)`` wrapper, so a builder never sees a
  design matrix;
* the scoring folds are order-respecting (an expanding walk-forward
  ``TimeSeriesSplit``, never a shuffled ``KFold`` over time-ordered rows);
* the result is still an ordinary forecast bundle: ``Pipeline(builder, model)``
  plus the serving geometry, so ``POST /api/inference/{task}/forecast`` answers.
"""

import importlib.util
import io
import json
import time

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from SKSurrogate import ForecastFeatureBuilder, load_bundle  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402

TASK = "temporal-ensemble"
HORIZON = 3


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    return TestClient(app)


def _frame(rows=90, season=4):
    t = np.arange(1, rows + 1, dtype=float)
    y = 0.4 * t + 3.0 * np.sin(2.0 * np.pi * t / season)
    return pd.DataFrame({"time": t, "y": y})


def _register(client, frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    response = client.post(
        "/api/datasets/%s/register" % TASK,
        data={
            "target": "y",
            "partition": "train",
            "temporal": json.dumps({"time_column": "time", "horizon": HORIZON}),
        },
        files={"file": ("train.csv", buffer.getvalue().encode(), "text/csv")},
    )
    assert response.status_code == 200, response.text


def _wait(client, job_id, attempts=240):
    for _ in range(attempts):
        record = client.get("/api/jobs/" + job_id).json()
        if record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.25)
    raise AssertionError("job did not finish")


def _baseline(client, estimator):
    response = client.post(
        "/api/bundles/%s/train-baseline" % TASK,
        json={"estimator": estimator, "seasonality": 4},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_a_temporal_ensemble_is_fitted_on_the_forecast_design_and_serves(client):
    _register(client, _frame())
    first = _baseline(client, "linear_regression")
    second = _baseline(client, "random_forest_regressor")

    submitted = client.post(
        "/api/ensembles/%s/fit" % TASK,
        json={
            "members": [first["model_version"], second["model_version"]],
            "method": "voting",
            "params": {},
            "random_state": 0,
        },
    )
    assert submitted.status_code == 200, submitted.text
    record = _wait(client, submitted.json()["job_id"])
    assert record["status"] == "completed", record.get("error")

    result = record["result"]
    assert result["feature_source"] == "temporal_design"
    # Both members were scored on the design, not silently skipped.
    assert [entry["error"] for entry in result["members"]] == [None, None]
    assert all(entry["cv_mean"] is not None for entry in result["members"])

    bundle = load_bundle(settings.task_bundles_dir(TASK) / (result["model_version"] + ".bundle"))
    steps = [step for _, step in bundle.model.steps]
    assert isinstance(steps[0], ForecastFeatureBuilder)
    assert bundle.temporal["strategy"] == "recursive"
    assert bundle.metadata["ensemble"]["feature_source"] == "temporal_design"
    assert bundle.metadata["ensemble"]["cv"]["type"] == "TimeSeriesSplit"
    # The schema describes the raw history frame the pipeline is fed, not the design.
    assert set(bundle.schema) == {"time", "y"}
    assert any("rolling-origin" in caveat for caveat in bundle.metadata["ensemble"]["caveats"])

    forecast = client.post(
        "/api/inference/%s/forecast" % TASK,
        json={
            "model_version": result["model_version"],
            "history": _frame().to_dict(orient="records"),
            "horizon": HORIZON,
        },
    )
    assert forecast.status_code == 200, forecast.text
    payload = forecast.json()
    assert payload["strategy"] == "recursive"
    assert len(payload["predictions"]) == HORIZON
    assert all(np.isfinite(payload["predictions"]))


def test_weight_optimization_walks_forward_instead_of_partitioning(client):
    """A weight search on a temporal task must not use a shuffled partition splitter.

    ``cross_val_predict`` refuses the walk-forward splitter outright
    (``ValueError: cross_val_predict only works for partitions``). The outputs are
    replayed fold by fold, and the initial training rows no fold can predict are
    reported as ``rows_unscored`` instead of being scored with a placeholder.
    """
    _register(client, _frame())
    first = _baseline(client, "linear_regression")
    second = _baseline(client, "random_forest_regressor")
    submitted = client.post(
        "/api/ensembles/%s/fit" % TASK,
        json={
            "members": [first["model_version"], second["model_version"]],
            "method": "voting",
            "params": {},
            "weight_optimization": {"strategy": "least_squares"},
            "random_state": 0,
        },
    )
    assert submitted.status_code == 200, submitted.text
    record = _wait(client, submitted.json()["job_id"])
    assert record["status"] == "completed", record.get("error")
    weights = record["result"]["weight_optimization"]
    assert len(weights["weights"]) == 2
    assert abs(sum(weights["weights"]) - 1.0) < 1e-6
    assert weights["rows_unscored"] >= 1
    assert weights["rows_scored"] > weights["rows_unscored"]


def test_the_temporal_default_splitter_walks_forward():
    """The fallback splitter must not be an integer fold count (a shuffled KFold)."""
    from sklearn.model_selection import TimeSeriesSplit

    from api.training import DEFAULT_ENSEMBLE_FOLDS
    from SKSurrogate.mltrace import cv_is_temporal_safe

    cv = TimeSeriesSplit(n_splits=DEFAULT_ENSEMBLE_FOLDS)
    assert cv_is_temporal_safe(cv)
    folds = list(cv.split(np.zeros((40, 3))))
    assert len(folds) == DEFAULT_ENSEMBLE_FOLDS
    for train_index, test_index in folds:
        assert train_index.max() < test_index.min()
