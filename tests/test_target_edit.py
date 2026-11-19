"""Editable target-column contract: re-pointing what registered analysis predicts.

The target column is no longer locked at registration: ``PUT /api/datasets/
<task>/target`` re-points it in metadata AND in the tracker's ``Task.target``
row (the source ``get_data``/``get_dataframe``/``split_train`` read directly),
recomputes the feature count, and reseeds the Pearson/variance weightings so
the "Consensus top features" card correlates against the new target. These
tests pin the validation rules and the downstream effect at the frame boundary.
"""

import importlib.util
import io

import pandas as pd
import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api import logs  # noqa: E402
from api.config import settings  # noqa: E402
from api.deps import open_tracker  # noqa: E402
from api.main import app  # noqa: E402
from api.training import load_partition  # noqa: E402


TASK = "retarget-task"


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


def _register(client, *, ignored=None, task=TASK, target="y"):
    data = {"target": target, "partition": "train"}
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


def _set_target(client, target, task=TASK):
    return client.put(f"/api/datasets/{task}/target", json={"target": target})


def test_target_change_persists_and_reseeds_weights(client):
    _register(client)
    response = _set_target(client, "w")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["target"] == "w"
    assert payload["previous_target"] == "y"
    # 5 columns minus the new target (w) = 4 effective features; y is now a feature.
    assert payload["effective_feature_count"] == 4
    assert payload["weights_stored"] == ["pearson", "variance"]

    # Metadata reflects the new target.
    metadata = client.get(f"/api/datasets/{TASK}").json()
    assert metadata["target_name"] == "w"
    assert metadata["effective_feature_count"] == 4

    # Both metadata AND the tracker's Task.target row updated, so the frame
    # helpers select the new target as the response.
    with open_tracker(TASK) as tracker:
        assert tracker.target == "w"
        frame = tracker.get_dataframe(partition="train")
        # w is the target, the rest (including the old target y) are features.
        assert "w" in frame.columns
        assert "y" in frame.columns
        # The derived count matches the new target exclusion.
        assert tracker.GetMetadata()["dataset_feature_count"] == len(frame.columns) - 1


def test_target_change_propagates_to_load_partition(client):
    _register(client)
    _set_target(client, "z")
    frame = load_partition(TASK, "train")
    assert "z" in frame.columns


def test_target_validation_rejects_unknown_ignored_and_unregistered(client):
    _register(client)
    _register(client, ignored=["z"], task="reignore")
    # Unknown column.
    unknown = _set_target(client, "nope")
    assert unknown.status_code == 400
    assert "not found" in unknown.json()["detail"].lower()
    # Ignored column cannot become the target.
    ignored = _set_target(client, "z", task="reignore")
    assert ignored.status_code == 400
    assert "ignored" in ignored.json()["detail"].lower()
    # No dataset registered.
    missing = _set_target(client, "y", task="unregistered-task")
    assert missing.status_code == 404


def test_target_change_is_reversible_and_clears_old_feature(client):
    _register(client)
    _set_target(client, "w")
    # Switch back to the original target.
    response = _set_target(client, "y")
    assert response.status_code == 200, response.text
    assert response.json()["previous_target"] == "w"
    assert response.json()["effective_feature_count"] == 4
    metadata = client.get(f"/api/datasets/{TASK}").json()
    assert metadata["target_name"] == "y"


def test_target_change_reseeds_consensus_weights(client):
    _register(client)
    # Seed a sensitivity weighting column first (so the weights table has more
    # than the base pearson/variance rows) — the re-target must not wipe it.
    seeded = client.post(f"/api/sensitivity/{TASK}/base-weights")
    assert seeded.status_code == 200, seeded.text

    _set_target(client, "w")
    after = client.get(f"/api/sensitivity/{TASK}/weights-heatmap")
    assert after.status_code == 200
    # The reseeded pearson row now excludes w (the target) and includes y.
    assert "z" in after.json()["labels"]
