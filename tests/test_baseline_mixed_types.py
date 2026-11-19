"""Baselines on mixed-type partitions (regression).

``POST /{task}/train-baseline`` used to hardwire ``StandardScaler`` in front of
the baseline estimator, so any non-numeric feature column (object/category/
string/datetime) crashed inside ``check_array`` and surfaced as an opaque 500.
Mixed-type partitions now get the same fold-local preprocessor the experiment
path installs; purely numeric ones keep the historical scaler-first pipeline.
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
from api.main import app  # noqa: E402


MIXED_TASK = "baseline-mixed"
NUMERIC_TASK = "baseline-numeric"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    logs.store.clear()
    return TestClient(app)


def _mixed_frame(rows=96):
    rng = range(rows)
    return pd.DataFrame(
        {
            "x_num": [i * 0.5 for i in rng],
            "x_int": [(i * 7) % 11 - 5 for i in rng],
            "group": ["a" if i % 3 else ("b" if i % 2 else "c") for i in rng],
            # A category level that also carries missing values: the fold-local
            # preprocessor imputes it as a sentinel before one-hotting.
            "flag_cat": [None if i % 7 == 0 else ("on" if i % 2 else "off") for i in rng],
            "y": [float(i) + (i % 5) for i in rng],
        }
    )


def _numeric_frame(rows=96):
    rng = range(rows)
    return pd.DataFrame(
        {
            "x_num": [i * 0.5 for i in rng],
            "x_int": [(i * 7) % 11 - 5 for i in rng],
            "y": [float(i) + (i % 5) for i in rng],
        }
    )


def _csv(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _register(client, task, partition, frame):
    response = client.post(
        f"/api/datasets/{task}/register",
        data={"target": "y", "partition": partition},
        files={"file": (f"{partition}.csv", _csv(frame), "text/csv")},
    )
    assert response.status_code == 200, response.text


def test_train_baseline_on_mixed_types_succeeds(client):
    frame = _mixed_frame()
    _register(client, MIXED_TASK, "train", frame)
    _register(client, MIXED_TASK, "validation", frame)

    trained = client.post(f"/api/bundles/{MIXED_TASK}/train-baseline", json={})
    assert trained.status_code == 200, trained.text
    payload = trained.json()
    # The categorical features are part of the bundle's feature schema...
    for column in ("group", "flag_cat"):
        assert column in payload["schema"]
    # ...and the validation score was computed through the same preprocessor.
    assert isinstance(payload["metrics"].get("score"), float)

    bundle = load_bundle(
        settings.task_bundles_dir(MIXED_TASK) / (payload["model_version"] + ".bundle"),
        strict_dependencies=False,
    )
    step_classes = [step.__class__.__name__ for _, step in bundle.model.steps]
    assert "FoldLocalDataPreprocessor" in step_classes

    # Provenance records the encoder actually fitted, not a hardcoded scaler.
    transformations = bundle.metadata["reproducibility"]["transformations"]
    assert transformations[0]["class"].endswith("FoldLocalDataPreprocessor")

    # Unknown categories at prediction time are tolerated (handle_unknown=ignore).
    fresh = frame.drop(columns=["y"]).copy()
    fresh.loc[fresh.index[:3], "group"] = "never-seen-before"
    predictions = bundle.model.predict(fresh)
    assert len(predictions) == len(fresh)


def test_train_baseline_mixed_types_tree_estimator(client):
    frame = _mixed_frame(rows=128)
    _register(client, MIXED_TASK, "train", frame)

    trained = client.post(
        f"/api/bundles/{MIXED_TASK}/train-baseline",
        json={"estimator": "random_forest_regressor"},
    )
    assert trained.status_code == 200, trained.text


def test_train_baseline_numeric_keeps_scaler_pipeline(client):
    frame = _numeric_frame()
    _register(client, NUMERIC_TASK, "train", frame)

    trained = client.post(f"/api/bundles/{NUMERIC_TASK}/train-baseline", json={})
    assert trained.status_code == 200, trained.text
    bundle = load_bundle(
        settings.task_bundles_dir(NUMERIC_TASK) / (trained.json()["model_version"] + ".bundle"),
        strict_dependencies=False,
    )
    # Purely numeric partitions keep the historical scaler-first shape.
    assert bundle.model.steps[0][1].__class__.__name__ == "StandardScaler"
    transformations = bundle.metadata["reproducibility"]["transformations"]
    assert transformations[0]["class"].endswith("StandardScaler")
