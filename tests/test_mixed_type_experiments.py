"""Mixed-type partitions: experiments must search them, and serve them back.

`AML` only installs its fold-local preprocessor when it is handed a
``DataFrame``. The API used to pass ``X.values`` (an object array), so every
candidate pipeline died on the first categorical value:

    ValueError: All candidate pipelines failed to produce a finite CV score:
    pipeline preprocessing failed before evaluation:
    ValueError: could not convert string to float: 'MacDonald'

These tests pin the three parts of the fix: the frame is what reaches ``AML``,
the resulting bundle survives a save/load round trip, and the bundle keeps the
pre-existing contract that explanations and learning curves can predict from
positional arrays once it starts with a DataFrame-only preprocessor.
"""

import importlib.util
import json
import pickle

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402
from SKSurrogate import load_bundle  # noqa: E402
from SKSurrogate.DataProcess import (  # noqa: E402
    ArrayAsFrame,
    FoldLocalDataPreprocessor,
    frame_tolerant,
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    return TestClient(app)


def _register_mixed(client, task, n=90):
    """Register a partition with one free-text column, like Churn's Surname."""
    rng = np.random.RandomState(0)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "surname": rng.choice(["MacDonald", "Hargrave", "Smith"], size=n),
        }
    )
    frame["y"] = ((frame["a"] + 0.3 * rng.normal(size=n)) > 0).astype(int)
    response = client.post(
        "/api/datasets/%s/register" % task,
        data={"target": "y", "partition": "train"},
        files={"file": ("t.csv", frame.to_csv(index=False), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return frame


def _wait(client, job_id, attempts=600):
    import time

    for _ in range(attempts):
        record = client.get("/api/jobs/" + job_id).json()
        if record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.25)
    raise AssertionError("job did not finish")


# --------------------------------------------------------------------------- #
# ArrayAsFrame / frame_tolerant
# --------------------------------------------------------------------------- #


def test_array_as_frame_labels_arrays_and_passes_frames_through():
    adapter = ArrayAsFrame(["a", "b"]).fit(np.zeros((4, 2)))
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert adapter.transform(frame) is frame

    rebuilt = adapter.transform(np.array([[1.0, 3.0], [2.0, 4.0]]))
    assert list(rebuilt.columns) == ["a", "b"]
    assert rebuilt["b"].tolist() == [3.0, 4.0]


def test_array_as_frame_rejects_the_wrong_width():
    adapter = ArrayAsFrame(["a", "b"]).fit(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="expected 2 feature column"):
        adapter.transform(np.zeros((3, 5)))


def test_frame_tolerant_wraps_only_dataframe_first_pipelines():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    plain = Pipeline([("clf", LogisticRegression())])
    assert frame_tolerant(plain) is plain

    gated = Pipeline([("preprocess", FoldLocalDataPreprocessor()), ("clf", LogisticRegression())])
    wrapped = frame_tolerant(gated, columns=["a", "b"])
    assert [name for name, _ in wrapped.steps] == ["as_frame", "preprocess", "clf"]
    # Idempotent: wrapping an already tolerant pipeline changes nothing.
    assert frame_tolerant(wrapped) is wrapped
    # The wrapper must survive bundle serialization.
    assert pickle.loads(pickle.dumps(wrapped)).steps[0][0] == "as_frame"


# --------------------------------------------------------------------------- #
# Experiment search on a mixed-type partition
# --------------------------------------------------------------------------- #


def test_experiment_with_a_text_column_completes_and_serves(client):
    frame = _register_mixed(client, "mixed")
    body = {
        "preset": "classification",
        "strategy": "eoa",
        "length": 1,
        "max_generations": 1,
        "num_parents": 2,
        "scoring": "accuracy",
        "random_state": 0,
    }
    job = client.post("/api/experiments/mixed/run", json=body).json()["job_id"]
    record = _wait(client, job)
    assert record["status"] == "completed", record.get("error")
    version = record["result"]["model_version"]

    bundle = load_bundle(settings.task_bundles_dir("mixed") / (version + ".bundle"))
    steps = [name for name, _ in bundle.model.steps]
    # The frame-tolerant adapter fronts the fold-local preprocessor, which is
    # what makes the categorical column searchable at all.
    assert steps[0] == "as_frame"
    assert steps[1] == "preprocess"

    features = frame[[column for column in frame.columns if column != "y"]]
    from_frame = bundle.model.predict(features)
    from_array = bundle.model.predict(features.to_numpy())
    assert np.array_equal(from_frame, from_array)

    # Serving path: inline JSON rows carrying the same categorical values.
    response = client.post(
        "/api/inference/mixed/predict",
        json={"model_version": version, "rows": json.loads(features.head(3).to_json(orient="records"))},
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["predictions"]) == 3


def test_advanced_search_space_encodes_text_columns(client):
    """An explicit config has no preset preprocessing, so AML needs cat_cols.

    This is the Advanced tab: the search space is supplied by the caller and
    AML only installs an encoder when it is told which columns are categorical.
    """
    frame = _register_mixed(client, "advanced")
    body = {
        "config": {
            "sklearn.tree.DecisionTreeClassifier": {
                "min_samples_split": {"type": "integer", "low": 2, "high": 6},
                "min_samples_leaf": {"type": "integer", "low": 1, "high": 4},
            },
            "sklearn.linear_model.LogisticRegression": {
                "C": {"type": "real", "low": 0.1, "high": 5.0},
            },
        },
        # The surrogate engine exercises the same pipeline construction as EOA
        # without EOA's per-generation Gaussian-process fits, which dominate the
        # runtime regardless of dataset size (388 s on 90 rows for one
        # generation). The contract under test is the encoding, not the search.
        "strategy": "surrogate",
        "surrogate_mode": True,
        "surrogate_itrs": 4,
        "length": 1,
        "scoring": "accuracy",
        "random_state": 0,
    }
    job = client.post("/api/experiments/advanced/run", json=body).json()["job_id"]
    record = _wait(client, job)
    assert record["status"] == "completed", record.get("error")

    result = record["result"]
    bundle = load_bundle(settings.task_bundles_dir("advanced") / (result["model_version"] + ".bundle"))
    # The reproducibility record lives with the bundle, not in the job payload.
    record_metadata = bundle.metadata["reproducibility"]
    assert record_metadata["configuration"]["cat_cols"] == ["surname"]
    steps = [entry["class"] for entry in record_metadata["transformations"]]
    assert any(step.endswith("OneHotEncoder") for step in steps)
    features = frame[[column for column in frame.columns if column != "y"]]
    assert np.array_equal(bundle.model.predict(features), bundle.model.predict(features.to_numpy()))
