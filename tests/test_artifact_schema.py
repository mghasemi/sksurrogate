"""Versioned artifacts: upgrades on read, refusal of newer versions.

Frozen version-1 fixtures are the point: each document kind must load with its
old shape and come back stamped, with every original key preserved.
"""

import json

import pytest

from api.config import settings
from api.jobs import job_manager
from SKSurrogate.artifacts import (
    ArtifactVersionError,
    CURRENT_VERSIONS,
    read_artifact,
    schema_version_of,
    stamp,
)
from SKSurrogate.modelbundle import ModelRegistry


def test_an_unstamped_job_record_is_version_one_and_upgrades():
    legacy = {
        "job_id": "abc123",
        "task_name": "demo",
        "kind": "experiment",
        "status": "completed",
        "created_at": "2026-01-01T00:00:00+00:00",
        "result": {"model_version": "v1"},
        "error": None,
        "custom_field": "kept",
    }
    assert schema_version_of("job", legacy) == 1
    upgraded = read_artifact("job", legacy)
    assert upgraded["schema_version"] == CURRENT_VERSIONS["job"] == 2
    assert upgraded["custom_field"] == "kept"
    assert upgraded["label"] is None
    assert upgraded["log_counts"] == {}
    assert upgraded["compute"] == {}
    assert upgraded["resumable"] is False
    assert upgraded["result"] == {"model_version": "v1"}


def test_resumable_is_recomputed_from_the_legacy_resume_block():
    legacy = {"job_id": "j", "resume": {"request": {}, "checkpoint_job_id": "j"}}
    assert read_artifact("job", legacy)["resumable"] is True


def test_an_unstamped_serving_pin_upgrades_with_an_empty_history():
    legacy = {
        "task_name": "demo",
        "active": True,
        "resolve_mode": "version",
        "requested": "v1",
        "model_version": "v1",
        "started_at": "2026-01-01T00:00:00+00:00",
    }
    upgraded = read_artifact("serving_pin", legacy)
    assert upgraded["schema_version"] == 2
    assert upgraded["history"] == []
    assert upgraded["requested"] == "v1"


def test_an_unstamped_registry_index_upgrades():
    upgraded = read_artifact("registry_index", {"models": {"demo": {"versions": {}}}})
    assert upgraded["schema_version"] == 2
    assert "demo" in upgraded["models"]
    # A malformed models value is replaced, not trusted.
    assert read_artifact("registry_index", {"models": []})["models"] == {}


def test_a_newer_artifact_is_refused_rather_than_guessed():
    with pytest.raises(ArtifactVersionError, match="newer than the supported"):
        read_artifact("job", {"job_id": "j", "schema_version": 99})
    with pytest.raises(ArtifactVersionError, match="unknown artifact kind"):
        read_artifact("not_a_kind", {})


def test_stamp_does_not_mutate_the_input():
    original = {"job_id": "j"}
    stamped = stamp("job", original)
    assert stamped["schema_version"] == 2
    assert "schema_version" not in original


def test_job_manager_reads_a_frozen_v1_record_and_writes_v2(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    job_id = "frozenv1job"
    legacy = {
        "job_id": job_id,
        "task_name": "legacy",
        "kind": "experiment",
        "status": "completed",
        "created_at": "2026-01-01T00:00:00+00:00",
        "result": {"ok": True},
        "error": None,
    }
    settings.job_record_path(job_id).write_text(json.dumps(legacy), encoding="utf-8")

    record = job_manager.get(job_id)
    assert record["schema_version"] == 2
    assert record["label"] is None
    assert record["result"] == {"ok": True}

    # A fresh write stamps the current version on disk.
    job_manager._write(record)
    raw = json.loads(settings.job_record_path(job_id).read_text(encoding="utf-8"))
    assert raw["schema_version"] == 2


def test_registry_index_is_stamped_and_upgrades_a_v1_document(tmp_path):
    root = tmp_path / "registry"
    registry = ModelRegistry(root)
    registry._write_index({"models": {"demo": {"versions": {}}}})
    raw = json.loads((root / "registry.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == 2

    legacy = {"models": {"demo": {"versions": {"v1": {"path": "demo/v1.bundle"}}}}}
    (root / "registry.json").write_text(json.dumps(legacy), encoding="utf-8")
    index = registry._read_index()
    assert index["schema_version"] == 2
    assert index["models"]["demo"]["versions"]["v1"]["path"] == "demo/v1.bundle"
