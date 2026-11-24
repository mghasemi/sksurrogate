"""Schema-v2 experiment manifests and bundle checksum sidecars.

The manifest is the artifact that answers "what produced this model?" — dataset
partition fingerprint, preprocessing, estimator/optimizer configuration, seeds
with their generator-stream provenance, environment, hardware, git revision,
evaluation-protocol and artifact checksums. These tests pin:

* a round trip preserves every field and a stable digest;
* a v1 ``reproducibility`` record is upgraded instead of rejected;
* ``save_bundle`` writes an integrity sidecar and ``verify_bundle`` rejects one
  corrupted byte while still loading bundles that predate checksums;
* the bundle carries the manifest record additively.
"""

import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from SKSurrogate import (
    BundleIntegrityError,
    ExperimentManifest,
    MANIFEST_SCHEMA_VERSION,
    ModelBundle,
    load_bundle,
    save_bundle,
    verify_bundle,
)
from SKSurrogate.manifest import (
    canonical_json,
    git_revision,
    hardware_snapshot,
    payload_digest,
    upgrade_manifest,
)


def _manifest(**overrides):
    payload = {
        "task_name": "manifest-task",
        "data": {
            "train_partition": "train",
            "train_fingerprint": "f" * 64,
            "rows_used": 120,
        },
        "preprocessing": [{"class": "sklearn.preprocessing.StandardScaler", "parameters": {}}],
        "configuration": {"kind": "automl_experiment", "random_state": 7},
        "evaluation_protocol": {"scoring": "accuracy", "cv": {"folds": 3}},
        "seeds": {
            "random_state": 7,
            "provenance": {"sampling": "SeedSequence.spawn(2)[1]"},
        },
        "search_space": {"sklearn.ensemble.RandomForestClassifier": {"n_estimators": {"low": 10, "high": 50}}},
        "evaluation_history_digest": "a" * 64,
        "artifacts": [{"kind": "dataset_partition", "id": "train", "fingerprint": "f" * 64}],
        "known_gaps": ["target construction is not recorded"],
        "leakage_warnings": {"checked": True, "duplicate_rows": {"duplicate_rows": 0}},
    }
    payload.update(overrides)
    return ExperimentManifest(**payload)


def _bundle(manifest=None):
    model = LinearRegression().fit(np.array([[0.0], [1.0]]), np.array([0.0, 2.0]))
    bundle = ModelBundle(
        model,
        task_name="manifest-task",
        metrics={"r2": 1.0},
        dataset_fingerprint="f" * 64,
        manifest=manifest,
    )
    return bundle


# --------------------------------------------------------------------------- #
# Manifest format
# --------------------------------------------------------------------------- #


def test_manifest_round_trip_preserves_fields_and_digest():
    manifest = _manifest()
    record = manifest.to_dict()
    assert record["schema_version"] == MANIFEST_SCHEMA_VERSION == 2
    assert record["seeds"]["random_state"] == 7
    assert record["manifest_digest"] == manifest.digest()

    restored = ExperimentManifest.from_dict(json.loads(json.dumps(record)))
    assert restored.to_dict() == record
    assert restored.digest() == record["manifest_digest"]


def test_digest_is_canonical_json_and_changes_with_content():
    left = _manifest()
    right = _manifest(configuration={"kind": "automl_experiment", "random_state": 8})
    assert left.digest() != right.digest()
    assert payload_digest(left.to_dict(include_digest=False)) == left.digest()
    # Key order must not matter (the digest field itself is excluded).
    reordered = dict(reversed(list(left.to_dict().items())))
    reordered.pop("manifest_digest", None)
    assert payload_digest(reordered) == left.digest()


def test_v1_record_is_upgraded_not_rejected():
    legacy = {
        "schema_version": 1,
        "task_name": "legacy",
        "data": {"train_fingerprint": "b" * 64},
        "configuration": {"kind": "automl_experiment"},
        "transformations": [{"class": "sklearn.preprocessing.StandardScaler"}],
        "evaluation_protocol": {"scoring": "accuracy"},
        "environment": {"python_version": "3.11.0", "numpy_version": "1.26.0"},
        "known_gaps": ["anything"],
    }
    manifest = ExperimentManifest.from_dict(legacy)
    assert manifest.schema_version == 2
    assert manifest.preprocessing == legacy["transformations"]
    # The v1 flat versions are preserved under ``registered`` and the live
    # packages block is added, so both readings stay available.
    assert manifest.environment["registered"]["numpy_version"] == "1.26.0"
    assert "packages" in manifest.environment
    assert manifest.data["train_fingerprint"] == "b" * 64


def test_future_manifest_schema_is_refused():
    with pytest.raises(ValueError, match="Unsupported manifest schema"):
        ExperimentManifest.from_dict({"schema_version": MANIFEST_SCHEMA_VERSION + 1})


def test_upgrade_is_lossless_for_unknown_keys():
    upgraded = upgrade_manifest(
        {"schema_version": 1, "task_name": "x", "custom_field": {"kept": True}}
    )
    assert upgraded["custom_field"] == {"kept": True}


def test_environment_and_git_helpers_are_serializable():
    hardware = hardware_snapshot()
    for key in ("python_version", "platform", "machine", "cpu_count", "blas"):
        assert key in hardware
    json.dumps(hardware)
    revision = git_revision()
    assert set(revision) == {"revision", "dirty", "source"}
    json.dumps(revision)


def test_manifest_writes_atomically(tmp_path):
    destination = tmp_path / "manifest.json"
    _manifest().write(destination)
    payload = json.loads(destination.read_text())
    assert payload["schema_version"] == 2
    assert not (tmp_path / "manifest.json.tmp").exists()


# --------------------------------------------------------------------------- #
# Bundle checksums
# --------------------------------------------------------------------------- #


def test_save_bundle_writes_sidecar_and_verify_passes(tmp_path):
    bundle = _bundle(_manifest().to_dict())
    path = save_bundle(bundle, tmp_path / "model.bundle")
    sidecar = tmp_path / "model.bundle.manifest.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["kind"] == "model_bundle"
    assert payload["artifact"]["path"] == "model.bundle"
    assert payload["artifact"]["sha256"]
    assert payload["manifest"]["schema_version"] == 2

    verdict = verify_bundle(path)
    assert verdict["verified"] is True
    assert verdict["sha256"] == payload["artifact"]["sha256"]


def test_verify_bundle_rejects_a_single_corrupted_byte(tmp_path):
    path = save_bundle(_bundle(_manifest().to_dict()), tmp_path / "model.bundle")
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0x01
    path.write_bytes(bytes(raw))
    with pytest.raises(BundleIntegrityError, match="checksum mismatch"):
        verify_bundle(path)
    with pytest.raises(BundleIntegrityError, match="checksum mismatch"):
        load_bundle(path, verify=True)


def test_legacy_bundle_without_sidecar_still_loads(tmp_path):
    path = tmp_path / "legacy.bundle"
    joblib.dump(_bundle().to_record(), path)
    verdict = verify_bundle(path)
    assert verdict["verified"] is False
    assert "sidecar missing" in verdict["reason"]
    with pytest.raises(BundleIntegrityError, match="strict mode"):
        verify_bundle(path, strict=True)
    assert load_bundle(path).task_name == "manifest-task"


def test_bundle_record_carries_manifest_additively(tmp_path):
    manifest = _manifest().to_dict()
    path = save_bundle(_bundle(manifest), tmp_path / "model.bundle")
    loaded = load_bundle(path)
    assert loaded.manifest == manifest
    # Old readers keep finding the record at its historical location when the
    # caller stored it there too; the new location is additive.
    assert loaded.to_record()["manifest"]["task_name"] == "manifest-task"


def test_manifest_recovers_legacy_reproducibility_record():
    bundle = ModelBundle(
        LinearRegression(),
        task_name="legacy",
        metadata={"reproducibility": {"schema_version": 1, "task_name": "legacy"}},
    )
    manifest = ExperimentManifest.from_bundle(bundle)
    assert manifest.schema_version == 2
    assert manifest.task_name == "legacy"
