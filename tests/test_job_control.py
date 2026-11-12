"""Tests for pausing/stopping jobs, resuming them, and liveness truth.

The API cannot kill a worker thread, so a stop is cooperative: the search checks
``stop_check`` between candidate evaluations and raises ``SearchStopped``. These
tests pin that core hook, the job-manager lifecycle, and the rule that
``/api/jobs/active`` and ``GET /api/jobs/{id}`` trust the live registry rather
than the persisted status.
"""

import importlib.util
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api import control  # noqa: E402
from api.config import settings  # noqa: E402
from api.jobs import job_manager  # noqa: E402
from api.main import app  # noqa: E402
from SKSurrogate import Integer  # noqa: E402
from SKSurrogate.structsearch import SearchStopped, SurrogateRandomCV  # noqa: E402

ACTIVE = ("queued", "running")
FINISHED = ("completed", "failed", "paused", "cancelled", "interrupted")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    job_manager.reconcile()
    return TestClient(app)


def _wait(job_id, timeout=30.0, states=FINISHED):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = job_manager.get(job_id)
        if record and record["status"] in states:
            return record
        time.sleep(0.02)
    raise AssertionError("job did not reach %s: %s" % (states, job_manager.get(job_id)))


def _wait_live(job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if job_manager.is_live(job_id):
            return True
        time.sleep(0.02)
    raise AssertionError("job never became live")


def _stoppable(seconds=10.0):
    """A job body that keeps working until the stop flag is observed."""

    def factory(_job_id):
        def run():
            deadline = time.time() + seconds
            while time.time() < deadline:
                if control.stop_requested():
                    raise SearchStopped("stopped by request")
                time.sleep(0.005)
            return {"ok": True}

        return run

    return factory


def _resume_metadata(job_id):
    return {
        "resume": {
            "request": {"config": {}, "backend": "local"},
            "checkpoint_job_id": job_id,
            "run_id": job_id,
        }
    }


# --------------------------------------------------------------------------- #
# Core hook
# --------------------------------------------------------------------------- #


def test_search_stops_at_a_candidate_boundary():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 2

    search = SurrogateRandomCV(
        DecisionTreeClassifier(random_state=0),
        {"max_depth": Integer(1, 6)},
        cv=2,
        n_jobs=1,
        max_iter=20,
        min_evals=1,
        refit=False,
        random_state=0,
        stop_check=stop,
    )
    with pytest.raises(SearchStopped):
        search.fit(X, y)
    # The check runs before evaluating a candidate, so it fires on the third
    # candidate rather than after the whole budget.
    assert calls["n"] == 3


def test_search_without_stop_check_is_unchanged():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    search = SurrogateRandomCV(
        DecisionTreeClassifier(random_state=0),
        {"max_depth": Integer(1, 6)},
        cv=2,
        n_jobs=1,
        max_iter=2,
        min_evals=1,
        refit=False,
        random_state=0,
    )
    search.fit(X, y)
    assert search.evaluation_history_


# --------------------------------------------------------------------------- #
# Pause / cancel / resume
# --------------------------------------------------------------------------- #


def test_pause_marks_the_job_paused_and_keeps_it_resumable(client):
    job_id = job_manager.submit(
        "pause-task", "experiment", _stoppable(), metadata_factory=_resume_metadata
    )
    _wait_live(job_id)
    time.sleep(0.1)
    response = client.post("/api/jobs/%s/stop" % job_id, json={"mode": "pause"})
    assert response.status_code == 200, response.text
    assert response.json()["stop_mode"] == "pause"

    record = _wait(job_id)
    assert record["status"] == "paused"
    assert record["stop"]["mode"] == "pause"
    assert record["resumable"] is True
    assert not job_manager.is_live(job_id)
    # A paused job is no longer advertised as active.
    assert client.get("/api/jobs/active").json()["count"] == 0


def test_resume_dispatches_to_the_original_checkpoint(client):
    job_id = job_manager.submit(
        "pause-task", "experiment", _stoppable(), metadata_factory=_resume_metadata
    )
    _wait_live(job_id)
    client.post("/api/jobs/%s/stop" % job_id, json={"mode": "pause"})
    _wait(job_id)

    checkpoint = settings.task_checkpoint_dir("pause-task", job_id)
    checkpoint.mkdir(parents=True)
    (checkpoint / "search.eoa").write_bytes(b"checkpoint")

    with patch(
        "api.routers.experiments.submit_experiment",
        return_value={"job_id": "resumed-job", "status": "queued"},
    ) as submitted:
        response = client.post("/api/jobs/%s/resume" % job_id)
    assert response.status_code == 200, response.text
    assert response.json()["job_id"] == "resumed-job"
    assert submitted.call_args.kwargs["checkpoint_job_id"] == job_id


def test_interrupted_job_resumes_from_the_surviving_checkpoint(client):
    """A crash leaves ``interrupted``; the checkpoint must still be resumable."""
    job_id = job_manager.submit(
        "crash-task", "experiment", _stoppable(seconds=0.1), metadata_factory=_resume_metadata
    )
    _wait(job_id)
    # Simulate what start-up reconciliation writes for a job the process lost.
    record = job_manager.get(job_id)
    record["status"] = "interrupted"
    record["interrupted_at"] = "2026-01-01T00:00:00+00:00"
    record["error"] = "The API process that owned this job exited before it finished"
    job_manager._write(record)

    checkpoint = settings.task_checkpoint_dir("crash-task", job_id)
    checkpoint.mkdir(parents=True)
    (checkpoint / "EOA.eoa").write_bytes(b"checkpoint")

    with patch(
        "api.routers.experiments.submit_experiment",
        return_value={"job_id": "resumed-after-crash", "status": "queued"},
    ) as submitted:
        response = client.post("/api/jobs/%s/resume" % job_id)
    assert response.status_code == 200, response.text
    assert response.json()["job_id"] == "resumed-after-crash"
    assert submitted.call_args.kwargs["checkpoint_job_id"] == job_id


def test_interrupted_job_without_a_checkpoint_cannot_resume(client):
    job_id = job_manager.submit(
        "crash-task", "experiment", _stoppable(seconds=0.1), metadata_factory=_resume_metadata
    )
    _wait(job_id)
    record = job_manager.get(job_id)
    record["status"] = "interrupted"
    job_manager._write(record)
    response = client.post("/api/jobs/%s/resume" % job_id)
    assert response.status_code == 409
    assert "No EOA checkpoint" in response.json()["detail"]


def test_cancel_marks_the_job_cancelled_and_blocks_resume(client):
    job_id = job_manager.submit(
        "cancel-task", "experiment", _stoppable(), metadata_factory=_resume_metadata
    )
    _wait_live(job_id)
    response = client.post("/api/jobs/%s/stop" % job_id, json={"mode": "cancel"})
    assert response.status_code == 200, response.text

    record = _wait(job_id)
    assert record["status"] == "cancelled"
    assert record["stop"]["mode"] == "cancel"

    checkpoint = settings.task_checkpoint_dir("cancel-task", job_id)
    checkpoint.mkdir(parents=True)
    (checkpoint / "search.eoa").write_bytes(b"checkpoint")
    assert client.post("/api/jobs/%s/resume" % job_id).status_code == 409


def test_pause_is_rejected_without_resume_metadata(client):
    job_id = job_manager.submit("plain-task", "sensitivity", _stoppable(seconds=5))
    _wait_live(job_id)
    try:
        response = client.post("/api/jobs/%s/stop" % job_id, json={"mode": "pause"})
        assert response.status_code == 409
        assert "cannot be paused" in response.json()["detail"]
    finally:
        client.post("/api/jobs/%s/stop" % job_id, json={"mode": "cancel"})
        _wait(job_id)


def test_stop_rejects_an_unknown_mode(client):
    job_id = job_manager.submit("mode-task", "sensitivity", _stoppable(seconds=5))
    _wait_live(job_id)
    try:
        response = client.post("/api/jobs/%s/stop" % job_id, json={"mode": "freeze"})
        assert response.status_code == 422
    finally:
        client.post("/api/jobs/%s/stop" % job_id, json={"mode": "cancel"})
        _wait(job_id)


def test_stop_on_a_finished_job_is_rejected(client):
    job_id = job_manager.submit("done-task", "sensitivity", lambda _jid: (lambda: {"ok": True}))
    _wait(job_id)
    response = client.post("/api/jobs/%s/stop" % job_id, json={"mode": "cancel"})
    assert response.status_code == 409


def test_queued_job_honours_a_stop_before_it_starts(client, monkeypatch):
    """A stop that arrives while the job is queued must not run the body."""
    ran = {"value": False}

    def factory(_job_id):
        def run():
            ran["value"] = True
            return {}

        return run

    # Occupy the single job slot with a stoppable job so the second one queues.
    monkeypatch.setattr("api.compute.current", lambda: {"job_workers": 1, "job_threads": 1, "blas_threads": 1,
                                                        "dask_workers": 1, "dask_threads": 1})
    blocker = job_manager.submit("slot-task", "sensitivity", _stoppable(seconds=10))
    _wait_live(blocker)
    try:
        queued = job_manager.submit("queued-task", "sensitivity", factory)
        time.sleep(0.05)
        assert job_manager.get(queued)["status"] == "queued"
        response = client.post("/api/jobs/%s/stop" % queued, json={"mode": "cancel"})
        assert response.status_code == 200, response.text
        record = _wait(queued)
        assert record["status"] == "cancelled"
        assert ran["value"] is False
    finally:
        client.post("/api/jobs/%s/stop" % blocker, json={"mode": "cancel"})
        _wait(blocker)


# --------------------------------------------------------------------------- #
# Liveness and stale records
# --------------------------------------------------------------------------- #


def _seed_record(job_id, status="running", **extra):
    record = {
        "job_id": job_id,
        "task_name": "stale-task",
        "kind": "experiment",
        "status": status,
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
        "result": None,
        "error": None,
    }
    record.update(extra)
    job_manager._write(record)
    return record


def test_reconcile_marks_stale_records_interrupted(client):
    _seed_record("stale-running")
    _seed_record("stale-queued", status="queued")
    corrected = job_manager.reconcile()
    assert set(corrected) == {"stale-running", "stale-queued"}
    for job_id in ("stale-running", "stale-queued"):
        record = job_manager.get(job_id)
        assert record["status"] == "interrupted"
        assert "not running" in record["error"]
    # Idempotent: nothing left to correct.
    assert job_manager.reconcile() == []


def test_active_excludes_stale_records_and_get_self_heals(client):
    _seed_record("stale-running")
    assert client.get("/api/jobs/active").json() == {"jobs": [], "count": 0}
    record = client.get("/api/jobs/stale-running").json()
    assert record["status"] == "interrupted"
    assert job_manager.get("stale-running")["status"] == "interrupted"


def test_active_lists_live_jobs_only(client):
    live = job_manager.submit("live-task", "experiment", _stoppable(seconds=10))
    _wait_live(live)
    try:
        _seed_record("stale-running-2")
        payload = client.get("/api/jobs/active").json()
        ids = [job["job_id"] for job in payload["jobs"]]
        assert live in ids
        assert "stale-running-2" not in ids
        summary = next(job for job in payload["jobs"] if job["job_id"] == live)
        assert set(summary) == {
            "job_id",
            "task_name",
            "display_name",
            "kind",
            "backend",
            "status",
            "created_at",
            "updated_at",
            "heartbeat_at",
            "log_counts",
        }
    finally:
        client.post("/api/jobs/%s/stop" % live, json={"mode": "cancel"})
        _wait(live)


def test_heartbeat_is_refreshed_while_running(client, monkeypatch):
    monkeypatch.setattr(control, "HEARTBEAT_SECONDS", 0.05)
    job_id = job_manager.submit("beat-task", "sensitivity", _stoppable(seconds=5))
    _wait_live(job_id)
    try:
        first = None
        deadline = time.time() + 5
        while time.time() < deadline:
            record = job_manager.get(job_id)
            if record.get("heartbeat_at") and first is None:
                first = record["heartbeat_at"]
            elif record.get("heartbeat_at") and record["heartbeat_at"] != first:
                break
            time.sleep(0.02)
        assert first is not None
        assert job_manager.get(job_id)["heartbeat_at"] != first
    finally:
        client.post("/api/jobs/%s/stop" % job_id, json={"mode": "cancel"})
        _wait(job_id)


def test_optimize_pipeline_jobs_are_resumable(client):
    from types import SimpleNamespace

    from api.routers import experiments as experiments_router

    bundle = SimpleNamespace(model_version="stub-model", metrics={"train_score": 0.5})
    with patch.object(
        experiments_router, "fit_pipeline_bundle", return_value=bundle
    ), patch.object(experiments_router, "save_bundle", return_value=Path("stub.bundle")):
        response = client.post(
            "/api/experiments/pipeline-task/optimize-pipeline",
            json={"seq": ["sklearn.tree.DecisionTreeClassifier"]},
        )
        assert response.status_code == 200, response.text
        job_id = response.json()["job_id"]
        # Keep the patch alive: the job runs on another thread.
        record = _wait(job_id)
    assert record["status"] == "completed", record["error"]
    assert record["resumable"] is True
    assert record["resume"]["checkpoint_job_id"] == job_id
    assert json.loads(settings.job_record_path(job_id).read_text())["resumable"] is True
