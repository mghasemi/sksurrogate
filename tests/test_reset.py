"""Per-user data reset: scope, safety checks and cleanup.

The reset is destructive, so the tests pin the three things that make it safe:
the typed confirmation, ownership scoping (including shared resources and
unowned legacy data), and the live-job refusal. Both the open single-user mode
and the authenticated per-account mode are covered.
"""

import importlib.util
import io
import json
import threading
import time

import pandas as pd
import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402

from SKSurrogate import ModelBundle, ModelRegistry  # noqa: E402

from api import acl, audit, logs, reset  # noqa: E402
from api.auth import Principal  # noqa: E402
from api.config import settings  # noqa: E402
from api.jobs import job_manager  # noqa: E402
from api.main import app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    monkeypatch.setattr(settings, "require_auth", False)
    monkeypatch.setattr(settings, "api_key", None)
    settings.ensure()
    acl.reset_store()
    logs.store.clear()
    return TestClient(app)


def _csv_bytes():
    frame = pd.DataFrame({"x": range(20), "y": [2 * i for i in range(20)]})
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _register(client, task):
    response = client.post(
        f"/api/datasets/{task}/register",
        data={"target": "y", "partition": "train"},
        files={"file": ("train.csv", _csv_bytes(), "text/csv")},
    )
    assert response.status_code == 200, response.text


def _task_paths_exist(task):
    return any(
        path.exists()
        for path in (
            settings.task_dataset_dir(task),
            settings.mltrace_db_path(task),
            settings.task_bundles_dir(task),
            settings.task_predictions_dir(task),
            settings.monitoring_dir / task,
            settings.checkpoints_dir / task,
            settings.registry_dir / task,
        )
    )


def _wait(job_id, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = job_manager.get(job_id)
        if record and record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_open_mode_reset_requires_the_exact_confirmation(client):
    _register(client, "keep-task")
    response = client.post("/api/settings/reset", json={"confirm": "reset"})
    assert response.status_code == 400
    assert "reset everything" in response.json()["detail"]
    assert _task_paths_exist("keep-task")


def test_open_mode_reset_wipes_tasks_projects_jobs_and_logs(client):
    _register(client, "alpha")
    _register(client, "beta")
    project = client.post(
        "/api/projects", json={"name": "study", "research_question": "does it work?"}
    )
    assert project.status_code == 200, project.text
    project_id = project.json()["project_id"]

    job_id = job_manager.submit("alpha", "experiment", lambda _id: lambda: {"ok": True})
    _wait(job_id)
    assert settings.job_record_path(job_id).exists()

    response = client.post(
        "/api/settings/reset", json={"confirm": "reset everything"}
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert sorted(payload["tasks"]) == ["alpha", "beta"]
    assert project_id in payload["projects"]
    assert payload["jobs"] >= 1
    assert payload["open_mode"] is True

    for task in ("alpha", "beta"):
        assert not _task_paths_exist(task)
    assert not (settings.projects_dir / (project_id + ".json")).exists()
    assert not settings.job_record_path(job_id).exists()
    assert not (settings.logs_dir / (job_id + ".jsonl")).exists()

    # The registry index no longer advertises the deleted tasks.
    index = json.loads((settings.registry_dir / "registry.json").read_text())
    assert "alpha" not in index["models"] and "beta" not in index["models"]
    # The task dropdown is empty again.
    assert client.get("/api/tasks").json() == {"tasks": []}

    events = [entry["event"] for entry in audit.read(20)]
    assert "account.reset" in events


def _seed_task(task):
    settings.task_dataset_dir(task).mkdir(parents=True, exist_ok=True)
    (settings.task_dataset_dir(task) / "train.csv").write_text("x,y\n1,2\n")
    settings.task_bundles_dir(task).mkdir(parents=True, exist_ok=True)
    (settings.task_bundles_dir(task) / "model.bundle").write_bytes(b"bundle")
    settings.task_predictions_dir(task).mkdir(parents=True, exist_ok=True)
    (settings.task_predictions_dir(task) / "pred.csv").write_text("prediction\n1\n")
    settings.monitor_alerts_path(task).parent.mkdir(parents=True, exist_ok=True)
    settings.monitor_alerts_path(task).write_text("{}\n")
    settings.task_checkpoint_dir(task, "job-1").mkdir(parents=True, exist_ok=True)
    settings.task_checkpoint_dir(task, "job-1").joinpath("run.eoa").write_text("checkpoint")
    settings.mltrace_db_path(task).write_bytes(b"")


def _seed_registry(task):
    registry = ModelRegistry(settings.registry_dir)
    registry.register(
        ModelBundle(  # noqa: F841 - registration is the side effect
            DummyClassifier(strategy="most_frequent"),
            task_name=task,
            schema={"x": {"dtype": "int64"}},
        )
    )


def test_authenticated_reset_respects_ownership_and_sharing(client, monkeypatch):
    monkeypatch.setattr(settings, "require_auth", True)
    acl.reset_store()
    alice = Principal(1, "alice", False, "session")
    alice_id, bob_id = 1, 2

    for task in ("mine", "bobs", "shared", "unowned"):
        _seed_task(task)
        _seed_registry(task)

    acl.store().set_role(acl.TASK, "mine", alice_id, "owner")
    acl.store().set_role(acl.TASK, "bobs", bob_id, "owner")
    acl.store().set_role(acl.TASK, "shared", alice_id, "owner")
    acl.store().set_role(acl.TASK, "shared", bob_id, "editor")

    result = reset.reset_user_data(alice)
    assert result["tasks"] == ["mine"]
    assert result["skipped_shared"]["tasks"] == ["shared"]
    assert not _task_paths_exist("mine")
    for kept in ("bobs", "shared", "unowned"):
        assert _task_paths_exist(kept)
    # Sharing survives for resources that were not deleted.
    assert acl.store().role_of(acl.TASK, "shared", bob_id) == "editor"

    result = reset.reset_user_data(alice, include_shared=True)
    assert result["tasks"] == ["shared"]
    assert not _task_paths_exist("shared")
    assert acl.store().role_of(acl.TASK, "shared", bob_id) is None
    assert _task_paths_exist("bobs") and _task_paths_exist("unowned")


def test_reset_refuses_while_a_scoped_job_is_live(client, monkeypatch):
    monkeypatch.setattr(settings, "require_auth", True)
    acl.reset_store()
    alice = Principal(1, "alice", False, "session")
    _seed_task("busy")
    acl.store().set_role(acl.TASK, "busy", 1, "owner")

    release = threading.Event()
    job_id = job_manager.submit(
        "busy",
        "experiment",
        lambda _id: lambda: (release.wait(20), {"ok": True})[1],
    )
    deadline = time.time() + 10
    while time.time() < deadline and (job_manager.get(job_id) or {}).get("status") != "running":
        time.sleep(0.02)
    try:
        with pytest.raises(reset.LiveJobsError) as excinfo:
            reset.reset_user_data(alice)
        assert excinfo.value.jobs[0]["job_id"] == job_id
        assert _task_paths_exist("busy")
    finally:
        release.set()
        _wait(job_id)


def test_authenticated_endpoint_requires_an_account_not_the_static_key(client, monkeypatch):
    monkeypatch.setattr(settings, "require_auth", True)
    acl.reset_store()
    # The static break-glass key has no user id, so there is no "my data" scope.
    with pytest.raises(PermissionError):
        reset.reset_user_data(Principal(None, "api-key", True, "static"))
