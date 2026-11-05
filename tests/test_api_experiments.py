"""API tests for the quick-start experiment options (preset / strategy)."""

import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    return TestClient(app)


def _register(client, task):
    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 3))
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    frame["y"] = (X[:, 0] + 0.3 * rng.normal(size=60) > 0).astype(int)
    response = client.post(
        "/api/datasets/%s/register" % task,
        data={"target": "y", "partition": "train"},
        files={"file": ("t.csv", frame.to_csv(index=False), "text/csv")},
    )
    assert response.status_code == 200, response.text


def _wait(client, job_id):
    for _ in range(240):
        record = client.get("/api/jobs/" + job_id).json()
        if record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def test_presets_listed(client):
    values = [p["value"] for p in client.get("/api/experiments/presets").json()["presets"]]
    assert values == ["classification", "regression"]


def test_config_and_preset_are_exclusive(client):
    body = {"preset": "classification", "config": {"sklearn.tree.DecisionTreeClassifier": {}}}
    assert client.post("/api/experiments/t/run", json=body).status_code == 422
    assert client.post("/api/experiments/t/run", json={}).status_code == 422


def test_preset_run_produces_bundle(client):
    _register(client, "quick")
    body = {"preset": "classification", "strategy": "surrogate", "length": 1, "random_state": 0}
    job = client.post("/api/experiments/quick/run", json=body).json()["job_id"]
    record = _wait(client, job)
    assert record["status"] == "completed", record
    assert record["result"]["model_version"]


def test_legacy_config_eoa_still_works(client):
    _register(client, "legacy")
    body = {
        "config": {
            "sklearn.tree.DecisionTreeClassifier": {"max_depth": {"type": "integer", "low": 2, "high": 5}},
            "sklearn.linear_model.LogisticRegression": {"C": {"type": "real", "low": 0.1, "high": 5}},
        },
        "length": 1,
        "max_generations": 1,
        "num_parents": 2,
        "random_state": 0,
    }
    job = client.post("/api/experiments/legacy/run", json=body).json()["job_id"]
    record = _wait(client, job)
    assert record["status"] == "completed", record


def test_spa_fallback(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<div id='root'></div>")
    monkeypatch.setenv("SKSURROGATE_WEBUI_DIST", str(dist))
    import importlib

    import api.main as main

    reloaded = importlib.reload(main)
    try:
        c = TestClient(reloaded.app)
        assert "root" in c.get("/experiments").text
        assert c.get("/api/health").json()["status"] == "ok"
        assert c.get("/api/health").json()["version"]
        assert c.get("/api/missing").status_code == 404
    finally:
        monkeypatch.delenv("SKSURROGATE_WEBUI_DIST")
        importlib.reload(main)


def _register_regression(client, task):
    rng = np.random.RandomState(1)
    X = rng.normal(size=(120, 3))
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    frame["y"] = 3 * X[:, 0] + 0.2 * rng.normal(size=120)
    client.post(
        "/api/datasets/%s/register" % task,
        data={"target": "y", "partition": "train"},
        files={"file": ("t.csv", frame.to_csv(index=False), "text/csv")},
    )


def _quick_bundle(client, task, preset):
    body = {"preset": preset, "strategy": "surrogate", "length": 1, "random_state": 0}
    record = _wait(client, client.post("/api/experiments/%s/run" % task, json=body).json()["job_id"])
    assert record["status"] == "completed", record
    return record["result"]["model_version"], record["job_id"]


def test_explain_intervals_and_project_summary(client):
    _register_regression(client, "reg")
    version, job_id = _quick_bundle(client, "reg", "regression")
    explained = client.post("/api/evaluation/reg/%s/explain" % version, json={"max_rows": 100, "n_repeats": 2})
    assert explained.status_code == 200, explained.text
    data = explained.json()
    assert data["importance"][0]["feature"] == "a"
    assert data["partial_dependence"] and data["partial_dependence"][0]["grid"]

    intervals = client.post("/api/evaluation/reg/%s/intervals" % version, json={"alpha": 0.2})
    assert intervals.status_code == 200, intervals.text
    assert 0.0 <= intervals.json()["empirical_coverage"] <= 1.0

    project = client.post(
        "/api/projects",
        json={
            "name": "p",
            "artifacts": [
                {"kind": "dataset_partition", "task_name": "reg", "artifact_id": "train"},
                {"kind": "job", "task_name": "reg", "artifact_id": job_id},
                {"kind": "model_bundle", "task_name": "reg", "artifact_id": version},
            ],
        },
    ).json()
    summary = client.get("/api/projects/%s/summary" % project["project_id"]).json()
    assert all(stage["done"] for stage in summary["stages"])


def test_intervals_reject_classifier(client):
    _register(client, "cls")
    version, _ = _quick_bundle(client, "cls", "classification")
    assert client.post("/api/evaluation/cls/%s/intervals" % version, json={}).status_code == 422


def test_design_endpoints(client):
    variables = [{"name": "x", "low": 0, "high": 1}, {"name": "t", "low": 10, "high": 20}]
    lhs = client.post("/api/design/latin-hypercube", json={"variables": variables, "n_samples": 5}).json()
    assert lhs["columns"] == ["x", "t"] and len(lhs["points"]) == 5
    samples = lhs["points"]
    values = [p[0] ** 2 + p[1] / 100 for p in samples]
    nxt = client.post("/api/design/suggest-next", json={"variables": variables, "samples": samples, "values": values})
    assert nxt.status_code == 200, nxt.text
    bad = client.post("/api/design/latin-hypercube", json={"variables": [{"name": "x", "low": 2, "high": 1}]})
    assert bad.status_code == 400
