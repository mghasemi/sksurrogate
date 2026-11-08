"""API tests for accumulated local effects and reference-subset stability.

Covers the Stage C contract in ``docs/mltrace.rst``:

* numeric ALE recovers a known linear effect and is mean-centered by bin weights;
* the deterministic Phase 5 perfect-correlation fixture shows ALE staying near
  the observed support where marginal PDP leaves it;
* nominal categorical features are skipped with a recorded reason;
* binary classifier curves use the labelled positive-class probability and
  multiclass bundles are rejected;
* reference-subset stability is deterministic, descriptive, and absent by
  default for both PDP and ALE.
"""

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api.config import settings  # noqa: E402
from api.explanations import (  # noqa: E402
    accumulated_local_effects,
    numeric_bin_edges,
    response_callable,
)
from api.main import app  # noqa: E402


class FirstColumnRegressor(RegressorMixin, BaseEstimator):
    """Deterministic 2x response on column ``a``; ignores other columns.

    Defined at module scope so :func:`SKSurrogate.save_bundle` (pickle) can
    resolve it when the API loads the bundle back.
    """

    def fit(self, X, y):
        self.n_features_in_ = np.asarray(X, dtype=object).shape[1]
        return self

    def predict(self, X):
        rows = np.asarray(X, dtype=object)
        return 2.0 * np.asarray([float(row[0]) for row in rows])


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("api-explanations")
    patcher = pytest.MonkeyPatch()
    patcher.setattr(settings, "root", root)
    settings.ensure()
    yield TestClient(app)
    patcher.undo()


def _wait(client, job_id):
    for _ in range(240):
        record = client.get("/api/jobs/" + job_id).json()
        if record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def _register(client, task, frame, target="y"):
    response = client.post(
        "/api/datasets/%s/register" % task,
        data={"target": target, "partition": "train"},
        files={"file": ("t.csv", frame.to_csv(index=False), "text/csv")},
    )
    assert response.status_code == 200, response.text


def _bundle(client, task, preset):
    body = {"preset": preset, "strategy": "surrogate", "length": 1, "random_state": 0}
    job = client.post("/api/experiments/%s/run" % task, json=body).json()["job_id"]
    record = _wait(client, job)
    assert record["status"] == "completed", record
    return record["result"]["model_version"]


def _regression_frame(n=160, seed=3):
    rng = np.random.RandomState(seed)
    features = rng.normal(size=(n, 3))
    frame = pd.DataFrame(features, columns=["a", "b", "c"])
    frame["y"] = 3.0 * features[:, 0] + 0.4 * features[:, 1] + 0.2 * rng.normal(size=n)
    return frame


def _classification_frame(n=160, seed=4, classes=2):
    rng = np.random.RandomState(seed)
    features = rng.normal(size=(n, 3))
    frame = pd.DataFrame(features, columns=["a", "b", "c"])
    if classes == 2:
        frame["y"] = (features[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    else:
        frame["y"] = np.digitize(features[:, 0], [-0.6, 0.6])
    return frame


@pytest.fixture(scope="module")
def regression_bundle(client):
    _register(client, "ale-reg", _regression_frame())
    return _bundle(client, "ale-reg", "regression")


@pytest.fixture(scope="module")
def classification_bundle(client):
    _register(client, "ale-clf", _classification_frame())
    return _bundle(client, "ale-clf", "classification")


def test_ale_recovers_known_linear_effect(client, regression_bundle):
    response = client.post(
        "/api/evaluation/ale-reg/%s/ale" % regression_bundle,
        json={"features": ["a", "b", "c"], "n_bins": 10, "max_rows": 500},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["feature_selection"] == "explicit"
    assert data["response"] == {"family": "Regression", "output": "predict", "causal_effect": False}
    assert data["interpretation"]["accumulated_local_effects"]["category"] == "fitted_model_response_profile"
    assert data["interpretation"]["accumulated_local_effects"]["causal_effect"] is False
    assert data["skipped"] == [] and data["errors"] == []
    assert {curve["feature"] for curve in data["curves"]} == {"a", "b", "c"}

    curve = next(item for item in data["curves"] if item["feature"] == "a")
    grid = np.asarray(curve["grid"], dtype=float)
    accumulated = np.asarray(curve["accumulated"], dtype=float)
    counts = np.asarray(curve["counts"], dtype=float)
    assert grid.size == accumulated.size == counts.size + 1
    assert counts.sum() == data["reference_sample"]["used_rows"]
    # Centered by bin weights: the weighted mean of the interior curve is zero.
    assert abs(float(np.average(accumulated[1:], weights=counts))) < 1e-9
    # The fitted linear model's ALE slope on "a" recovers the true coefficient.
    slope = (accumulated[-1] - accumulated[0]) / (grid[-1] - grid[0])
    assert abs(slope - 3.0) < 0.75, slope
    # "c" is independent noise: its curve stays small in magnitude.
    noise_curve = next(item for item in data["curves"] if item["feature"] == "c")
    assert float(np.max(np.abs(noise_curve["accumulated"]))) < 0.75

    repeated = client.post(
        "/api/evaluation/ale-reg/%s/ale" % regression_bundle,
        json={"features": ["a", "b", "c"], "n_bins": 10, "max_rows": 500},
    ).json()
    assert repeated["curves"] == data["curves"]


def test_ale_stays_on_support_where_pdp_leaves_it():
    """Phase 5 perfect-correlation fixture, evaluated through the ALE kernel."""
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.inspection import partial_dependence

    class DifferenceRegressor(RegressorMixin, BaseEstimator):
        def fit(self, X, y):
            self.n_features_in_ = np.asarray(X).shape[1]
            return self

        def predict(self, X):
            values = np.asarray(X, dtype=float)
            return np.square(values[:, 0] - values[:, 1])

    values = np.linspace(-1.0, 1.0, 101)
    reference = np.column_stack([values, values])
    model = DifferenceRegressor().fit(reference, np.zeros(len(reference)))
    response, _ = response_callable(model, "Regression")

    edges, _ = numeric_bin_edges(reference[:, 0], n_bins=9)
    accumulated, counts, excluded = accumulated_local_effects(response, reference, 0, edges)
    profile = partial_dependence(model, reference, [0], method="brute", grid_resolution=9)

    assert excluded == 0
    assert float(np.max(np.abs(model.predict(reference)))) == 0.0
    assert float(np.max(profile["average"])) > 1.0  # marginal PDP invents off-support response
    assert float(np.max(np.abs(accumulated))) < 0.05  # ALE accumulates observed-row differences only


def test_ale_skips_nominal_categorical_features(client):
    """A nominal categorical feature is skipped while numeric features still draw."""
    from SKSurrogate import ModelBundle, save_bundle

    from api.deps import open_tracker

    task = "ale-cat"
    rng = np.random.RandomState(7)
    frame = _regression_frame(n=120, seed=7)
    frame["kind"] = np.where(rng.normal(size=120) > 0, "left", "right")
    dataset_dir = settings.task_dataset_dir(task)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dataset_dir / "train.csv", index=False)
    with open_tracker(task) as tracker:
        tracker.RegisterData(frame, "y", partition="train")
        fingerprint = tracker.GetMetadata()["dataset_fingerprint"]

    model = FirstColumnRegressor().fit(frame.drop(columns=["y"]), frame["y"])
    bundle = ModelBundle(
        model,
        task_name=task,
        model_version="ale-cat-v1",
        metrics={"score": 1.0},
        dataset_fingerprint=fingerprint,
        schema={column: {"dtype": str(frame[column].dtype)} for column in frame.columns if column != "y"},
        dependencies={},
    )
    save_bundle(bundle, settings.task_bundles_dir(task) / "ale-cat-v1.bundle")

    response = client.post(
        "/api/evaluation/ale-cat/ale-cat-v1/ale",
        json={"features": ["kind"], "n_bins": 8},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["curves"] == []
    assert [item["reason"] for item in data["skipped"]] == ["nominal_categorical_ale_undefined"]
    assert "Partial dependence is still available" in data["skipped"][0]["detail"]

    # The same request with a numeric feature still draws a curve (with a slope of 2).
    numeric = client.post(
        "/api/evaluation/ale-cat/ale-cat-v1/ale",
        json={"features": ["a"], "n_bins": 8},
    ).json()
    assert [curve["feature"] for curve in numeric["curves"]] == ["a"]
    curve = numeric["curves"][0]
    slope = (curve["accumulated"][-1] - curve["accumulated"][0]) / (curve["grid"][-1] - curve["grid"][0])
    assert abs(slope - 2.0) < 1e-6


def test_ale_classifier_probability_scale(client, classification_bundle):
    response = client.post(
        "/api/evaluation/ale-clf/%s/ale" % classification_bundle,
        json={"features": ["a"], "n_bins": 10},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["response"]["family"] == "Classification"
    assert data["response"]["output"].startswith("predict_proba[class=")
    assert data["response"]["positive_class"] == 1
    assert data["interpretation"]["accumulated_local_effects"]["response_output"] == data["response"]["output"]
    curve = data["curves"][0]
    assert len(curve["grid"]) == len(curve["accumulated"])
    assert all(value is not None for value in curve["accumulated"])


def test_ale_rejects_multiclass_bundles(client):
    _register(client, "ale-multi", _classification_frame(classes=3))
    version = _bundle(client, "ale-multi", "classification")
    response = client.post(
        "/api/evaluation/ale-multi/%s/ale" % version,
        json={"features": ["a"]},
    )
    assert response.status_code == 422, response.text
    assert "3-class classifier" in response.json()["detail"]


def test_ale_rejects_unknown_and_excess_features(client, regression_bundle):
    unknown = client.post(
        "/api/evaluation/ale-reg/%s/ale" % regression_bundle,
        json={"features": ["nope"]},
    )
    assert unknown.status_code == 400
    assert "Unknown feature" in unknown.json()["detail"]

    too_many = client.post(
        "/api/evaluation/ale-reg/%s/ale" % regression_bundle,
        json={"features": ["a", "b", "c", "a", "b", "c", "a"]},
    )
    assert too_many.status_code == 400
    assert "At most six features" in too_many.json()["detail"]


def test_ale_stability_block_is_deterministic_and_descriptive(client, regression_bundle):
    body = {"features": ["a"], "n_bins": 8, "max_rows": 100, "stability_samples": 3, "random_state": 0}
    response = client.post("/api/evaluation/ale-reg/%s/ale" % regression_bundle, json=body)
    assert response.status_code == 200, response.text
    stability = response.json()["curves"][0]["stability"]
    assert stability["sample_count"] == 3
    assert stability["sample_size"] == 100
    assert stability["sampling"] == "random_without_replacement"
    grid_size = len(response.json()["curves"][0]["grid"])
    assert len(stability["mean"]) == len(stability["sample_std"]) == grid_size
    assert all(value >= 0.0 for value in stability["sample_std"])
    assert all(
        low <= mean <= high
        for low, mean, high in zip(stability["min"], stability["mean"], stability["max"])
    )
    assert "not a confidence band" in stability["note"]

    repeated = client.post("/api/evaluation/ale-reg/%s/ale" % regression_bundle, json=body).json()
    assert repeated["curves"][0]["stability"] == stability


def test_ale_merges_sparse_bins_on_tied_values():
    rng = np.random.RandomState(11)
    column = np.concatenate([np.zeros(300), rng.normal(scale=0.05, size=20), np.ones(5) * 3.0])
    edges, counts = numeric_bin_edges(column, n_bins=20, min_bin_fraction=0.01)
    assert edges.size < 21  # duplicate quantiles collapsed and sparse bins merged
    assert counts.min() >= 3  # ceil(0.01 * 325) == 4 -> floor satisfied without empty bins
    assert counts.sum() == column.size
    assert np.all(np.diff(edges) > 0)


def test_explain_pdp_stability_block(client, regression_bundle):
    body = {"max_rows": 100, "n_repeats": 2, "stability_samples": 3, "random_state": 0}
    response = client.post("/api/evaluation/ale-reg/%s/explain" % regression_bundle, json=body)
    assert response.status_code == 200, response.text
    data = response.json()
    stability = data["partial_dependence_stability"]
    assert stability["sample_count"] == 3
    assert stability["sample_size"] == 100
    assert stability["sampling"] == "random_without_replacement"
    assert stability["grid_source"] == "primary_curve_grid"
    assert stability["response"]["output"] == "predict"
    assert len(stability["curves"]) == len(data["partial_dependence"])
    for curve, primary in zip(stability["curves"], data["partial_dependence"]):
        assert curve["feature"] == primary["feature"]
        assert curve["grid"] == primary["grid"]
        assert len(curve["mean"]) == len(curve["sample_std"]) == len(primary["grid"])
        assert len(curve["sample_means"]) == 3
        assert all(value >= 0.0 for value in curve["sample_std"])

    repeated = client.post("/api/evaluation/ale-reg/%s/explain" % regression_bundle, json=body).json()
    assert repeated["partial_dependence_stability"] == stability

    default = client.post(
        "/api/evaluation/ale-reg/%s/explain" % regression_bundle,
        json={"max_rows": 100, "n_repeats": 2},
    ).json()
    assert default["partial_dependence_stability"] is None
    assert any("not a confidence band" in item for item in data["interpretation"]["limitations"])
