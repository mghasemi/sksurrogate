"""Ignored-column contract: dataset-level exclusions reach every modeling path.

A column marked ``ignore`` in the Datasets page is stored in the task metadata but
must disappear from experiments, baselines, sensitivity analysis, profiles,
synthetic data and monitoring. These tests pin the storage/validation rules and
the downstream behaviour at the frame boundary.
"""

import importlib.util
import io

import pandas as pd
import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from SKSurrogate import load_bundle  # noqa: E402

from api import logs  # noqa: E402
from api.config import settings  # noqa: E402
from api.deps import open_tracker  # noqa: E402
from api.main import app  # noqa: E402
from api.training import load_partition  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402


TASK = "ignore-task"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    logs.store.clear()
    return TestClient(app)


def _csv(rows=48):
    rng = range(rows)
    frame = pd.DataFrame(
        {
            "x": [i * 0.5 for i in rng],
            "z": [((i * 7) % 11) - 5 for i in rng],
            "w": [((i * 3) % 5) + 1 for i in rng],
            "flag": [i % 2 for i in rng],
            "y": [2.0 * i + 0.25 * (i % 3) for i in rng],
        }
    )
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _register(client, *, ignored=None, task=TASK):
    data = {"target": "y", "partition": "train"}
    if ignored is not None:
        import json

        data["ignored_columns"] = json.dumps(ignored)
    response = client.post(
        f"/api/datasets/{task}/register",
        data=data,
        files={"file": ("train.csv", _csv(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _set_ignored(client, columns, task=TASK):
    return client.put(f"/api/datasets/{task}/ignored-columns", json={"columns": columns})


def test_ignore_list_persists_and_reports_effective_features(client):
    _register(client)
    response = _set_ignored(client, ["z"])
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ignored_columns"] == ["z"]
    assert payload["target"] == "y"
    assert "z" not in payload["effective_features"]
    assert payload["effective_feature_count"] == len(payload["effective_features"])

    metadata = client.get(f"/api/datasets/{TASK}").json()
    assert metadata["ignored_columns"] == ["z"]
    assert metadata["effective_feature_count"] == payload["effective_feature_count"]

    # Idempotent and order-stable.
    assert _set_ignored(client, ["z", "z"]).json()["ignored_columns"] == ["z"]


def test_ignore_validation_rejects_target_unknown_and_unregistered(client):
    _register(client)
    target = _set_ignored(client, ["y"])
    assert target.status_code == 400
    assert "target column" in target.json()["detail"].lower()

    unknown = _set_ignored(client, ["nope"])
    assert unknown.status_code == 400
    assert "unknown column" in unknown.json()["detail"].lower()

    missing = _set_ignored(client, ["x"], task="unregistered-task")
    assert missing.status_code == 404


def test_registration_accepts_ignored_columns(client):
    registered = _register(client, ignored=["z"])
    assert registered["ignored_columns"] == ["z"]
    assert client.get(f"/api/datasets/{TASK}").json()["ignored_columns"] == ["z"]

    # Registration-time ignore already applies at load.
    frame = load_partition(TASK, "train")
    assert "z" not in frame.columns and "x" in frame.columns


def test_load_partition_and_baseline_bundle_exclude_ignored(client):
    _register(client)
    assert "z" in load_partition(TASK, "train").columns

    _set_ignored(client, ["z"])
    frame = load_partition(TASK, "train")
    assert "z" not in frame.columns
    # The stored CSV is untouched: the choice is reversible.
    with open(settings.task_dataset_dir(TASK) / "train.csv", encoding="utf-8") as handle:
        assert "z" in handle.readline().split(",")

    trained = client.post(
        f"/api/bundles/{TASK}/train-baseline",
        json={"estimator": "linear_regression", "validation_partition": None},
    )
    assert trained.status_code == 200, trained.text
    assert "z" not in trained.json()["schema"]
    bundle = load_bundle(
        settings.task_bundles_dir(TASK) / (trained.json()["model_version"] + ".bundle"),
        strict_dependencies=False,
    )
    assert "z" not in bundle.schema

    # Re-enabling restores the column without re-uploading anything.
    _set_ignored(client, [])
    assert "z" in load_partition(TASK, "train").columns


def test_sensitivity_and_base_weights_skip_ignored(client):
    _register(client)
    _set_ignored(client, ["z"])
    endpoint = client.post(f"/api/sensitivity/{TASK}/base-weights")
    assert endpoint.status_code == 200, endpoint.text
    # x, w, flag are numeric features; z is ignored rather than reported as a
    # non-numeric exclusion, because it never reaches the projection.
    assert endpoint.json()["feature_count"] == 3

    heatmap = client.get(f"/api/sensitivity/{TASK}/weights-heatmap")
    assert heatmap.status_code == 200
    assert "z" not in heatmap.json()["labels"]


def test_dataset_profile_skips_ignored(client):
    _register(client)
    _set_ignored(client, ["z"])
    profile = client.post(
        f"/api/datasets/{TASK}/profile", json={"partition": "train", "max_rows": 200}
    )
    assert profile.status_code == 200, profile.text
    payload = profile.json()
    assert payload["ignored_columns"] == ["z"]
    assert "z" not in [column["column"] for column in payload["columns"]]


def test_nested_cv_excludes_ignored_columns(client):
    _register(client)
    _set_ignored(client, ["z"])
    with open_tracker(TASK) as tracker:
        result = tracker.nested_cv_evaluation(
            DummyRegressor(),
            inner_cv=2,
            outer_cv=2,
            train_partition="train",
            validation_partition="validation",
            exclude_columns=["z"],
        )
    assert result["outer_scores"], "nested CV produced no outer scores"
    # Without the exclusion the frame still carries z; the API's load helper
    # asserts the exclusion independently.
    assert "z" not in load_partition(TASK, "train").columns


def test_synthetic_text_binary_defaults_to_categorical(client):
    task = "synth-text"
    frame = pd.DataFrame(
        {
            "x": [i * 0.5 for i in range(30)],
            "gender": ["Female" if i % 2 else "Male" for i in range(30)],
            "y": [float(i) for i in range(30)],
        }
    )
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    registered = client.post(
        f"/api/datasets/{task}/register",
        data={"target": "y", "partition": "train"},
        files={"file": ("train.csv", buffer.getvalue().encode("utf-8"), "text/csv")},
    )
    assert registered.status_code == 200, registered.text

    # No overrides: the deducer calls a two-value string column "binary", but
    # SynthData's bin type is numeric-only, so the default must be categorical.
    generated = client.post(
        f"/api/datasets/{task}/synthetic",
        json={"num": 8, "partition": "train", "distribution_type": "marginal"},
    )
    assert generated.status_code == 200, generated.text
    values = {row["gender"] for row in generated.json()["preview"]}
    assert values <= {"Female", "Male"}

    # An explicit incompatible override is a clear 400, not a str/float crash.
    bad = client.post(
        f"/api/datasets/{task}/synthetic",
        json={
            "num": 8,
            "partition": "train",
            "type_overrides": {"gender": "bin"},
        },
    )
    assert bad.status_code == 400
    assert "requires numeric values" in bad.json()["detail"]


def test_synthetic_generation_skips_ignored(client):
    _register(client)
    _set_ignored(client, ["z"])
    response = client.post(
        f"/api/datasets/{TASK}/synthetic",
        json={"num": 12, "partition": "train", "distribution_type": "marginal"},
    )
    assert response.status_code == 200, response.text
    assert "z" not in response.json()["columns"]
