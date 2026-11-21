"""Partition defaulting: a request that names no partition uses the task's data.

Request models used to default to the literal partition name ``"train"``. A task
whose data lives elsewhere — a single ``main`` partition, or one named like the
task — answered ``404 No dataset partition 'train' stored`` for a request that never
asked for a partition at all.

The rule: an explicit name always wins (and a wrong explicit name still 404s); an
omitted name resolves to ``train`` when it exists, otherwise to the task's only or
first registered partition, and the resolved name is echoed back.
"""

import importlib.util
import io
import time

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402
from api.partitions import resolve_optional_partition, resolve_partition  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    return TestClient(app, raise_server_exceptions=False)


def _frame(rows=30):
    return pd.DataFrame(
        {"x": np.arange(rows, dtype=float), "y": 2.0 * np.arange(rows, dtype=float)}
    )


def _register(client, task, partition):
    buffer = io.StringIO()
    _frame().to_csv(buffer, index=False)
    response = client.post(
        f"/api/datasets/{task}/register",
        data={"target": "y", "partition": partition},
        files={"file": ("train.csv", buffer.getvalue().encode(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_resolve_prefers_train_when_present(client):
    _register(client, "pd-train", "train")
    assert resolve_partition("pd-train", None) == "train"


def test_resolve_falls_back_to_the_registered_partition(client):
    _register(client, "pd-main", "main")
    assert resolve_partition("pd-main", None) == "main"


def test_resolve_keeps_an_explicit_name(client):
    _register(client, "pd-main", "main")
    # An explicit name is never replaced, even when it does not exist.
    assert resolve_partition("pd-main", "holdout") == "holdout"


def test_resolve_optional_partition_is_none_when_absent(client):
    _register(client, "pd-main", "main")
    assert resolve_optional_partition("pd-main", None) is None
    _register(client, "pd-val", "validation")
    assert resolve_optional_partition("pd-val", None) == "validation"


def test_unknown_task_keeps_the_historical_default(client):
    # No tracking database: the caller's 404 message keeps saying 'train'.
    assert resolve_partition("never-registered", None) == "train"


def test_train_baseline_uses_the_registered_partition(client):
    _register(client, "pd-main", "main")
    response = client.post("/api/bundles/pd-main/train-baseline", json={})
    assert response.status_code == 200, response.text
    assert response.json()["train_partition"] == "main"


def test_train_baseline_honours_an_explicit_partition(client):
    _register(client, "pd-main", "main")
    response = client.post(
        "/api/bundles/pd-main/train-baseline", json={"train_partition": "main"}
    )
    assert response.status_code == 200, response.text


def test_train_baseline_rejects_a_wrong_explicit_partition(client):
    _register(client, "pd-main", "main")
    response = client.post(
        "/api/bundles/pd-main/train-baseline", json={"train_partition": "holdout"}
    )
    assert response.status_code == 404
    assert "'holdout'" in response.json()["detail"]


def test_synthetic_uses_the_same_default(client):
    _register(client, "pd-main", "main")
    response = client.post("/api/datasets/pd-main/synthetic", json={"num": 5})
    assert response.status_code == 200, response.text
    assert response.json()["partition"] == "main"


def test_preflight_uses_the_registered_partition(client):
    _register(client, "pd-main", "main")
    response = client.get("/api/datasets/pd-main/preflight")
    assert response.status_code == 200, response.text
    assert response.json()["partition"] == "main"


def test_sensitivity_reads_the_registered_partition(client):
    _register(client, "pd-main", "main")
    response = client.get("/api/sensitivity/pd-main/correlation-matrix")
    assert response.status_code == 200, response.text
    assert response.json()["partition"] == "main"


def test_sensitivity_explicitly_named_partition_still_errors(client):
    _register(client, "pd-main", "main")
    response = client.get("/api/sensitivity/pd-main/correlation-matrix?train_partition=holdout")
    assert response.status_code == 404
    assert "'holdout'" in response.json()["detail"]


def _wait(client, job_id, attempts=240):
    for _ in range(attempts):
        record = client.get("/api/jobs/" + job_id).json()
        if record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def test_experiment_runs_on_the_registered_partition(client):
    """The reported bug: a task stored as `main` must not need a `train` partition."""
    _register(client, "pd-main", "main")
    body = {"preset": "regression", "strategy": "exhaustive", "length": 1, "random_state": 0}
    response = client.post("/api/experiments/pd-main/run", json=body)
    assert response.status_code == 200, response.text
    record = _wait(client, response.json()["job_id"])
    assert record["status"] == "completed", record
    assert record["result"]["model_version"]


def test_optimize_pipeline_runs_on_the_registered_partition(client):
    _register(client, "pd-main", "main")
    body = {"seq": ["sklearn.tree.DecisionTreeRegressor"], "scoring": "neg_root_mean_squared_error"}
    response = client.post("/api/experiments/pd-main/optimize-pipeline", json=body)
    assert response.status_code == 200, response.text
    record = _wait(client, response.json()["job_id"])
    assert record["status"] == "completed", record


def test_retraining_defaults_to_the_registered_partition(client):
    _register(client, "pd-main", "main")
    body = {"trainer": {"kind": "baseline", "estimator": "linear_regression"}}
    response = client.post("/api/retraining/pd-main/run", json=body)
    assert response.status_code == 200, response.text
    record = _wait(client, response.json()["job_id"])
    assert record["status"] == "completed", record
