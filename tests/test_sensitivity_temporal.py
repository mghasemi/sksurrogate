"""Sensitivity/correlation on temporal tasks (the generated-design path).

A time-series partition usually has no raw feature column: ``time`` (and a panel
id) are excluded by the temporal contract, and the target is the label. The
numeric projection used by sensitivity and Pearson pruning therefore found an
empty feature set and answered a misleading 400 — "encode or drop those columns"
when there were no such columns to drop.

Contract under test:

* a temporal task runs sensitivity over its *generated* design
  (``sales__lag_h``, rollings, calendar fields), with the provenance reported;
* calendar fields work from a CSV-loaded ISO date column (object dtype), which the
  builder must parse exactly as ``validate_temporal_frame`` does;
* the correlation matrix/threshold endpoints use the same design, and a
  single-feature design is an all-kept answer rather than a crash;
* a tabular task keeps its numeric-column projection unchanged, and a partition
  with no feature column left now says so instead of blaming non-numeric data.
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

from api.config import settings  # noqa: E402
from api.jobs import job_manager  # noqa: E402
from api.main import app  # noqa: E402

TASK = "sensitivity-temporal"
TABULAR = "sensitivity-tabular"
HORIZON = 4
JOB_TIMEOUT_SECONDS = 60.0


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    return TestClient(app)


def _temporal_frame(rows=90):
    """Date column as CSV writes it (ISO strings → object dtype) plus the target."""
    t = pd.date_range("2026-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {"date": t.strftime("%Y-%m-%d"), "sales": 10.0 + 0.5 * np.arange(rows)}
    )


def _csv(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _register(client, frame, task, temporal=None):
    data = {"target": frame.columns[-1], "partition": "train"}
    if temporal is not None:
        data["temporal"] = json.dumps(temporal)
    response = client.post(
        f"/api/datasets/{task}/register",
        data=data,
        files={"file": ("train.csv", _csv(frame), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _wait(job_id, timeout=JOB_TIMEOUT_SECONDS):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = job_manager.get(job_id)
        if record is not None and record["status"] in {"completed", "failed"}:
            return record
        time.sleep(0.05)
    raise AssertionError("sensitivity job did not finish within %.0fs" % timeout)


def test_sensitivity_runs_over_the_generated_design(client):
    _register(
        client,
        _temporal_frame(),
        TASK,
        {"time_column": "date", "horizon": HORIZON, "freq": "D"},
    )
    response = client.post(
        f"/api/sensitivity/{TASK}/run", json={"method": "sobol", "n_features_to_select": 1}
    )
    assert response.status_code == 200, response.text
    record = _wait(response.json()["job_id"])
    assert record["status"] == "completed", record.get("error")

    result = record["result"]
    protocol = result["evaluation_protocol"]
    assert protocol["feature_source"] == "temporal_design"
    # Default design: one lag at the declared horizon, built from history only.
    assert protocol["feature_columns"] == ["sales__lag_%d" % HORIZON]
    temporal = protocol["temporal"]
    assert temporal["temporal_spec"]["horizon"] == HORIZON
    assert temporal["feature_specs"][0]["op"] == "lag"
    assert temporal["feature_specs"][0]["shift"] == HORIZON
    assert result["top_feature_names"] == ["sales__lag_4"]
    # The honest caveat is present, not implied.
    limitations = " ".join(result["interpretation"]["limitations"])
    assert "generated from the declared temporal spec" in limitations


def test_calendar_features_parse_the_object_date_column(client):
    """A CSV date column is object dtype; calendar fields must still be derived."""
    _register(
        client,
        _temporal_frame(),
        TASK,
        {"time_column": "date", "horizon": HORIZON, "freq": "D"},
    )
    response = client.post(
        f"/api/sensitivity/{TASK}/run",
        json={
            "method": "sobol",
            "n_features_to_select": 2,
            "features": {
                "lags": [HORIZON],
                "rolling": [[HORIZON, ["mean"]]],
                "calendar": ["dow", "is_weekend"],
            },
        },
    )
    assert response.status_code == 200, response.text
    record = _wait(response.json()["job_id"])
    assert record["status"] == "completed", record.get("error")

    protocol = record["result"]["evaluation_protocol"]
    columns = protocol["feature_columns"]
    assert "sales__lag_4" in columns
    assert "sales__roll_mean_4" in columns
    assert "cal__dow" in columns and "cal__is_weekend" in columns
    ops = {entry["name"]: entry["op"] for entry in protocol["temporal"]["feature_specs"]}
    assert ops["cal__dow"] == "calendar"


def test_correlation_endpoints_use_the_design_and_survive_one_feature(client):
    _register(
        client,
        _temporal_frame(),
        TASK,
        {"time_column": "date", "horizon": HORIZON, "freq": "D"},
    )
    matrix = client.get(f"/api/sensitivity/{TASK}/correlation-matrix")
    assert matrix.status_code == 200, matrix.text
    payload = matrix.json()
    assert payload["feature_source"] == "temporal_design"
    assert payload["columns"] == ["sales__lag_%d" % HORIZON]

    # One feature: numpy.corrcoef returns a scalar, so this used to raise
    # IndexError inside CorrelationThreshold. Nothing to prune ⇒ keep it.
    pruned = client.post(
        f"/api/sensitivity/{TASK}/correlation-threshold", params={"threshold": 0.9}
    )
    assert pruned.status_code == 200, pruned.text
    body = pruned.json()
    assert body["kept_feature_names"] == ["sales__lag_%d" % HORIZON]
    assert body["dropped_feature_names"] == []


def test_tabular_task_keeps_the_numeric_column_projection(client):
    frame = pd.DataFrame(
        {
            "x1": np.linspace(0.0, 1.0, 60),
            "x2": np.linspace(1.0, 0.0, 60),
            "label": np.linspace(0.0, 2.0, 60),
        }
    )
    _register(client, frame, TABULAR)
    response = client.post(
        f"/api/sensitivity/{TABULAR}/run", json={"method": "morris", "n_features_to_select": 1}
    )
    assert response.status_code == 200, response.text
    record = _wait(response.json()["job_id"])
    assert record["status"] == "completed", record.get("error")
    protocol = record["result"]["evaluation_protocol"]
    assert protocol["feature_source"] == "partition_columns"
    assert protocol["feature_columns"] == ["x1", "x2"]
    assert protocol["temporal"] is None


def test_empty_projection_names_the_real_cause(client):
    """Ignoring the only feature leaves no column — not a dtype problem."""
    frame = pd.DataFrame(
        {"x": np.arange(60.0), "y": np.arange(60.0) * 2.0}
    )
    _register(client, frame, TABULAR)
    ignored = client.put(
        f"/api/datasets/{TABULAR}/ignored-columns", json={"columns": ["x"]}
    )
    assert ignored.status_code == 200, ignored.text

    response = client.post(f"/api/sensitivity/{TABULAR}/run", json={})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "no feature columns left" in detail
    assert "ignored" in detail
    # The old message pointed at non-numeric columns, which do not exist here.
    assert "Encode or drop those columns" not in detail
