"""API tests for the lightweight job list used by the activity indicator.

``GET /api/jobs/active`` exists so the top bar can poll without dragging whole
experiment results (and their evaluation histories) through every request; the
tests pin that contract, plus the per-trial ``score``/``params`` fields the
live activity feed renders.
"""

import importlib.util
import threading
import time

import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api import logs  # noqa: E402
from api.config import settings  # noqa: E402
from api.jobs import job_manager  # noqa: E402
from api.main import app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    logs.store.clear()
    return TestClient(app)


def _wait(job_id, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = job_manager.get(job_id)
        if record and record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_active_lists_only_unfinished_jobs(client):
    release = threading.Event()

    def slow_factory(job_id):
        def run():
            release.wait(30)
            return {"ok": True}

        return run

    running_id = job_manager.submit("active-task", "experiment", slow_factory)
    deadline = time.time() + 10
    while time.time() < deadline and (job_manager.get(running_id) or {}).get("status") != "running":
        time.sleep(0.02)

    response = client.get("/api/jobs/active")
    assert response.status_code == 200
    payload = response.json()
    assert [job["job_id"] for job in payload["jobs"]] == [running_id]
    assert payload["count"] == 1
    # Summaries must not carry the heavy fields the full list returns.
    summary = payload["jobs"][0]
    assert "result" not in summary and "resume" not in summary
    assert summary["kind"] == "experiment"
    assert summary["status"] == "running"

    release.set()
    _wait(running_id)
    # A finished job drops out of the active list on the next poll.
    assert client.get("/api/jobs/active").json() == {"jobs": [], "count": 0}


def test_active_honours_task_filter(client):
    release = threading.Event()

    def slow_factory(job_id):
        def run():
            release.wait(30)
            return {"ok": True}

        return run

    first = job_manager.submit("task-one", "sensitivity", slow_factory)
    second = job_manager.submit("task-two", "sensitivity", slow_factory)
    try:
        payload = client.get("/api/jobs/active", params={"task_name": "task-one"}).json()
        assert [job["job_id"] for job in payload["jobs"]] == [first]
        assert second not in [job["job_id"] for job in payload["jobs"]]
    finally:
        release.set()
        _wait(first)
        _wait(second)


def test_trial_callback_keeps_score_and_params_fields(client):
    def factory(job_id):
        def run():
            logs.trial_callback(
                {
                    "trial": 4,
                    "phase": "explore",
                    "status": "completed",
                    "score": 0.8123,
                    "params": {"max_depth": 3, "criterion": "gini"},
                }
            )
            return {"ok": True}

        return run

    job_id = job_manager.submit("telemetry-task", "experiment", factory)
    _wait(job_id)

    entries = client.get(f"/api/logs/{job_id}", params={"level": "INFO"}).json()["entries"]
    trial_entries = [entry for entry in entries if entry.get("trial") == 4]
    assert len(trial_entries) == 1
    entry = trial_entries[0]
    assert entry["status"] == "completed"
    assert entry["score"] == pytest.approx(0.8123)
    assert entry["params"] == {"max_depth": 3, "criterion": "gini"}
    # The human-readable message stays the same one the Logs page renders.
    assert "trial 4 (explore) completed score=0.8123" in entry["message"]
