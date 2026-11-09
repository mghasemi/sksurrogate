"""API tests for the per-job log capture and the /api/logs endpoints."""

import json
import logging
import time

import pandas as pd
import pytest

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
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def _submit(client, task, body, messages=None, fail=False):
    """Submit a job that optionally logs core-namespace messages then ends."""

    def fn_factory(job_id):
        def run():
            logger = logging.getLogger("SKSurrogate")
            for level_name, text in messages or []:
                logger.log(getattr(logging, level_name), text)
            if fail:
                raise ValueError("intentional job failure")
            return {"ok": True}

        return run

    return job_manager.submit(task, "test", fn_factory)


def test_lifecycle_and_core_messages_are_captured(client):
    job_id = _submit(
        client,
        "logs-task",
        None,
        messages=[("INFO", "core info line"), ("DEBUG", "core debug line")],
    )
    record = _wait(job_id)

    response = client.get("/api/logs", params={"job_id": job_id})
    assert response.status_code == 200
    payload = response.json()
    messages = [entry["message"] for entry in payload["entries"]]
    assert "Job accepted (kind=test, backend=local)" in messages
    assert "Job started (kind=test)" in messages
    assert "core info line" in messages
    assert "core debug line" in messages
    assert "Job completed (kind=test)" in messages
    assert payload["counts"]["INFO"] >= 3
    assert payload["counts"]["DEBUG"] >= 1
    # Counts are denormalised onto the job record for the list view.
    assert record["log_counts"]["INFO"] >= 3


def test_severity_filters(client):
    messages = [
        ("DEBUG", "d-line"),
        ("INFO", "i-line"),
        ("WARNING", "w-line"),
        ("ERROR", "e-line"),
        ("CRITICAL", "c-line"),
    ]
    job_id = _submit(client, "levels", None, messages=messages)
    _wait(job_id)

    only_warnings = client.get(
        "/api/logs", params={"job_id": job_id, "level": "WARNING"}
    ).json()
    levels = {entry["level"] for entry in only_warnings["entries"]}
    assert levels == {"WARNING", "ERROR", "CRITICAL"}
    assert "i-line" not in [entry["message"] for entry in only_warnings["entries"]]

    explicit = client.get(
        "/api/logs", params={"job_id": job_id, "levels": "INFO,ERROR"}
    ).json()
    assert {entry["level"] for entry in explicit["entries"]} == {"INFO", "ERROR"}
    assert "i-line" in [entry["message"] for entry in explicit["entries"]]

    everything = client.get("/api/logs", params={"job_id": job_id}).json()
    assert {"d-line", "c-line"} <= {entry["message"] for entry in everything["entries"]}


def test_invalid_level_is_rejected(client):
    response = client.get("/api/logs", params={"level": "LOUD"})
    assert response.status_code == 422
    assert "Unknown log level" in response.json()["detail"]


def test_failed_job_logs_error_with_traceback_detail(client):
    job_id = _submit(client, "failing", None, fail=True)
    record = _wait(job_id)
    assert record["status"] == "failed"

    response = client.get("/api/logs", params={"job_id": job_id, "level": "ERROR"})
    entries = response.json()["entries"]
    assert entries
    failure = next(entry for entry in entries if "Job failed" in entry["message"])
    assert failure["level"] == "ERROR"
    assert "intentional job failure" in failure["message"]
    assert "ValueError" in failure["detail"]
    assert record["log_counts"]["ERROR"] >= 1


def test_concurrent_jobs_do_not_share_entries(client):
    first = _submit(client, "concurrent-a", None, messages=[("INFO", "a-only")])
    second = _submit(client, "concurrent-b", None, messages=[("INFO", "b-only")])
    _wait(first)
    _wait(second)

    first_messages = [
        entry["message"]
        for entry in client.get("/api/logs", params={"job_id": first}).json()["entries"]
    ]
    second_messages = [
        entry["message"]
        for entry in client.get("/api/logs", params={"job_id": second}).json()["entries"]
    ]
    assert "a-only" in first_messages and "b-only" not in first_messages
    assert "b-only" in second_messages and "a-only" not in second_messages


def test_jsonl_survives_a_restart(client, tmp_path):
    job_id = _submit(client, "durable", None, messages=[("INFO", "persisted line")])
    _wait(job_id)

    fresh_store = logs.JobLogStore()
    result = fresh_store.query(job_id=job_id)
    messages = [entry["message"] for entry in result["entries"]]
    assert "persisted line" in messages
    assert result["counts"]["INFO"] >= 2
    log_file = settings.logs_dir / (job_id + ".jsonl")
    assert log_file.exists()
    assert "persisted line" in log_file.read_text(encoding="utf-8")


def test_free_text_and_limit_filters(client):
    job_id = _submit(
        client, "searchable", None, messages=[("INFO", "alpha marker"), ("INFO", "beta")]
    )
    _wait(job_id)

    result = client.get("/api/logs", params={"job_id": job_id, "q": "marker"}).json()
    assert [entry["message"] for entry in result["entries"]] == ["alpha marker"]
    assert result["total"] == 1

    limited = client.get("/api/logs", params={"job_id": job_id, "limit": 1}).json()
    assert len(limited["entries"]) == 1
    assert limited["total"] > 1


def test_repeated_trial_failures_are_capped_at_warning(client):
    job = {"job_id": "cap-test", "task_name": "cap", "kind": "test"}
    token = logs.set_current_job(job)
    try:
        for index in range(8):
            logs.trial_callback(
                {"trial": index, "phase": "explore", "status": "failed", "score": None}
            )
    finally:
        logs.reset_current_job(token)

    entries = logs.store.query(job_id="cap-test", level="DEBUG")["entries"]
    levels = [entry["level"] for entry in entries if entry.get("trial") is not None]
    assert levels.count("WARNING") == 5
    assert levels.count("DEBUG") == 3


def _register(client, task):
    import numpy as np

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


def test_experiment_jobs_capture_trial_events(client):
    """AML forwards the callbacks hook, so trials appear as log entries."""
    _register(client, "trial-logs")
    body = {"preset": "classification", "strategy": "surrogate", "length": 1, "random_state": 0}
    job_id = client.post("/api/experiments/trial-logs/run", json=body).json()["job_id"]
    record = _wait(job_id)
    assert record["status"] == "completed"

    entries = client.get("/api/logs", params={"job_id": job_id, "limit": 2000}).json()["entries"]
    trial_entries = [entry for entry in entries if entry.get("trial") is not None]
    assert trial_entries, "no trial events were captured"
    assert {"phase", "status"} <= set(trial_entries[0])
    assert all(entry["logger"] == "sksurrogate.api" for entry in trial_entries)
