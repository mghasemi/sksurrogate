"""Tests for human-friendly names: job labels and user-chosen model aliases.

Both features add a *display* identity on top of an immutable machine identity:
``job_id`` stays a uuid4 hex and ``model_version`` stays the bundle's version id.
These tests pin that separation — the alias is stored, resolvable, renamable and
audited, while every route keeps working on the canonical id.
"""

import importlib.util
import threading
import time

import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from SKSurrogate import ModelBundle, ModelRegistry  # noqa: E402
from SKSurrogate.modelbundle import LIFECYCLE_ALIAS_NAMES  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402

from api import logs  # noqa: E402
from api.config import settings  # noqa: E402
from api.deps import bundle_summary  # noqa: E402
from api.jobs import job_manager  # noqa: E402
from api.main import app  # noqa: E402
from api.routers.experiments import OptimizePipelineRequest, RunExperimentRequest  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    logs.store.clear()
    return TestClient(app)


def _wait(job_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = job_manager.get(job_id)
        if record and record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _instant_factory(_job_id):
    return lambda: {"ok": True}


# --------------------------------------------------------------------------- #
# Job labels / display names
# --------------------------------------------------------------------------- #

def test_job_without_label_gets_readable_auto_name(client):
    job_id = job_manager.submit("credit-card", "experiment", _instant_factory)
    _wait(job_id)

    record = client.get(f"/api/jobs/{job_id}").json()
    assert record["label"] is None
    name = record["display_name"]
    # The auto name is human readable and never the machine id.
    assert name.startswith("experiment · credit-card · ")
    assert job_id not in name
    assert len(job_id) == 32 and all(c in "0123456789abcdef" for c in job_id)


def test_job_label_round_trips_and_rename_clears_it(client):
    job_id = job_manager.submit(
        "credit-card", "experiment", _instant_factory, label="  baseline EOA  "
    )
    _wait(job_id)

    record = client.get(f"/api/jobs/{job_id}").json()
    assert record["label"] == "baseline EOA"  # trimmed on write
    assert record["display_name"] == "baseline EOA"

    renamed = client.patch(f"/api/jobs/{job_id}", json={"label": "credit-card v2"}).json()
    assert renamed["display_name"] == "credit-card v2"
    assert renamed["label"] == "credit-card v2"
    assert renamed["aliased_at"] and "aliased_by" in renamed

    cleared = client.patch(f"/api/jobs/{job_id}", json={"label": None}).json()
    assert cleared["label"] is None
    assert cleared["display_name"].startswith("experiment · credit-card · ")


def test_job_label_validation_and_unknown_job(client):
    job_id = job_manager.submit("credit-card", "experiment", _instant_factory)
    _wait(job_id)

    assert client.patch(f"/api/jobs/{job_id}", json={"label": "bad\x00name"}).status_code == 400
    assert client.patch(f"/api/jobs/{job_id}", json={"label": "x" * 81}).status_code == 400
    assert client.patch("/api/jobs/does-not-exist", json={"label": "x"}).status_code == 404


def test_active_summaries_carry_display_name(client):
    release = threading.Event()

    def slow_factory(_job_id):
        def run():
            release.wait(20)
            return {"ok": True}

        return run

    job_id = job_manager.submit("active-task", "sensitivity", slow_factory)
    deadline = time.time() + 10
    while time.time() < deadline and (job_manager.get(job_id) or {}).get("status") != "running":
        time.sleep(0.02)
    try:
        summaries = client.get("/api/jobs/active").json()["jobs"]
        assert [job["job_id"] for job in summaries] == [job_id]
        assert summaries[0]["display_name"].startswith("sensitivity · active-task · ")
    finally:
        release.set()
        _wait(job_id)


def test_experiment_requests_accept_a_label():
    run = RunExperimentRequest(config={"sklearn.linear_model.Ridge": {}}, label="ridge-search")
    assert run.label == "ridge-search"
    optimize = OptimizePipelineRequest(
        seq=["sklearn.linear_model.Ridge"], label="ridge-pipeline"
    )
    assert optimize.label == "ridge-pipeline"


# --------------------------------------------------------------------------- #
# Bundle labels
# --------------------------------------------------------------------------- #

def test_bundle_summary_exposes_metadata_label():
    bundle = ModelBundle(
        DummyClassifier(strategy="most_frequent"),
        task_name="credit-card",
        metadata={"reproducibility": {}, "label": "baseline"},
    )
    summary = bundle_summary(bundle)
    assert summary["label"] == "baseline"
    assert summary["model_version"] == bundle.model_version


# --------------------------------------------------------------------------- #
# Registry user-chosen aliases
# --------------------------------------------------------------------------- #

def _registered(registry, task_name, count, **metadata):
    versions = []
    for _ in range(count):
        bundle = ModelBundle(
            DummyClassifier(strategy="most_frequent"),
            task_name=task_name,
            metadata=metadata or None,
        )
        registry.register(bundle)
        versions.append(bundle.model_version)
    return versions


def test_named_alias_resolves_to_exact_version(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    versions = _registered(registry, "credit-card", 2)

    entry = registry.set_named_alias("credit-card", "best-f1", versions[1], actor="mehdi")
    assert entry["model_version"] == versions[1]
    assert entry["created_by"] == "mehdi"

    assert registry.resolve_version("credit-card", "best-f1") == versions[1]
    assert registry.load_version("credit-card", "best-f1").model_version == versions[1]
    assert registry.load("credit-card", alias="best-f1").model_version == versions[1]
    # A lifecycle alias still resolves exactly as before.
    assert registry.load("credit-card", alias="latest").model_version == versions[1]
    # The summary exposes user aliases separately from lifecycle pointers.
    summary = registry.summary("credit-card")
    assert summary["named_aliases"]["best-f1"]["model_version"] == versions[1]
    assert "best-f1" not in summary["aliases"]


def test_named_alias_rejects_reserved_or_illegal_names_and_unknown_versions(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    versions = _registered(registry, "credit-card", 1)

    for bad in ("Production", "bad name", "-leading", "", "a" * 65):
        with pytest.raises(ValueError):
            registry.set_named_alias("credit-card", bad, versions[0])
    for reserved in sorted(LIFECYCLE_ALIAS_NAMES):
        with pytest.raises(ValueError):
            registry.set_named_alias("credit-card", reserved, versions[0])
    with pytest.raises(KeyError):
        registry.set_named_alias("credit-card", "ok-name", "no-such-version")
    with pytest.raises(KeyError):
        registry.resolve_version("credit-card", "missing-alias")


def test_alias_mutations_are_audited_and_deletable(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    versions = _registered(registry, "credit-card", 2)
    registry.set_named_alias("credit-card", "baseline", versions[0], actor="mehdi")
    registry.set_named_alias("credit-card", "baseline", versions[1], actor="mehdi")
    removed = registry.delete_named_alias("credit-card", "baseline")
    assert removed["model_version"] == versions[1]

    audit = registry.audit_log("credit-card")
    kinds = [event["event_type"] for event in audit]
    assert "alias_set" in kinds and "alias_deleted" in kinds
    renames = [event for event in audit if event["event_type"] == "alias_set"]
    assert renames[-1]["previous_model_version"] == versions[0]

    with pytest.raises(KeyError):
        registry.delete_named_alias("credit-card", "baseline")


def test_registry_alias_endpoints(client, tmp_path):
    registry = ModelRegistry(settings.registry_dir)
    versions = _registered(registry, "credit-card", 1)

    created = client.post(
        "/api/registry/credit-card/aliases",
        json={"name": "baseline", "model_version": versions[0]},
    )
    assert created.status_code == 200, created.text
    assert created.json()["model_version"] == versions[0]

    listed = client.get("/api/registry/credit-card/aliases").json()
    assert listed["aliases"]["baseline"]["model_version"] == versions[0]

    summary = client.get("/api/registry/credit-card/summary").json()
    assert "baseline" in summary["named_aliases"]

    deleted = client.delete("/api/registry/credit-card/aliases/baseline")
    assert deleted.status_code == 200
    assert client.get("/api/registry/credit-card/aliases").json()["aliases"] == {}

    bad = client.post(
        "/api/registry/credit-card/aliases",
        json={"name": "production", "model_version": versions[0]},
    )
    assert bad.status_code == 400
    missing = client.post(
        "/api/registry/credit-card/aliases",
        json={"name": "fine", "model_version": "nope"},
    )
    assert missing.status_code == 404
