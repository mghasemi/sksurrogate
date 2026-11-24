"""Explanation guarantees: grouped importance, OOD flags, provenance, stability.

The helper tests use hand-built matrices; the endpoint tests reuse the same
direct-bundle pattern as the comparison tests so the assertions target the
contract rather than a search result.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api.config import settings  # noqa: E402
from api.explanations import (  # noqa: E402
    correlated_groups,
    grouped_permutation_importance,
    out_of_distribution,
    sign_consistency,
)
from api.main import app  # noqa: E402
from SKSurrogate.modelbundle import ModelBundle, save_bundle  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("api-explanation-guarantees")
    patcher = pytest.MonkeyPatch()
    patcher.setattr(settings, "root", root)
    settings.ensure()
    yield TestClient(app)
    patcher.undo()


def _register(client, task, frame, target="y"):
    response = client.post(
        "/api/datasets/%s/register" % task,
        data={"target": target, "partition": "train"},
        files={"file": ("t.csv", frame.to_csv(index=False), "text/csv")},
    )
    assert response.status_code == 200, response.text


def _write_bundle(task, version, estimator, X, y):
    estimator.fit(X, y)
    bundle = ModelBundle(
        estimator,
        task_name=task,
        schema={column: "float64" for column in X.columns},
        model_version=version,
        dataset_fingerprint="deadbeefcafe",
    )
    directory = settings.task_bundles_dir(task)
    directory.mkdir(parents=True, exist_ok=True)
    save_bundle(bundle, directory / ("%s.bundle" % version))


def _regression_frame(n=200, seed=11):
    rng = np.random.RandomState(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = 0.98 * a + 0.2 * rng.normal(size=n)  # near-duplicate of a
    frame = pd.DataFrame({"a": a, "b": b, "c": c})
    frame["y"] = 2.0 * a + 0.3 * b + 0.1 * rng.normal(size=n)
    return frame


@pytest.fixture(scope="module")
def regression_task(client):
    frame = _regression_frame()
    _register(client, "guar-reg", frame)
    X, y = frame[["a", "b", "c"]], frame["y"]
    _write_bundle("guar-reg", "linear", LinearRegression(), X, y)
    return "guar-reg"


@pytest.fixture(scope="module")
def classification_task(client):
    frame = _regression_frame(seed=12)
    frame["y"] = (frame["a"] + frame["b"] > 0).astype(int)
    _register(client, "guar-clf", frame)
    X, y = frame[["a", "b", "c"]], frame["y"]
    _write_bundle("guar-clf", "logit", LogisticRegression(max_iter=500), X, y)
    return "guar-clf"


def test_correlated_groups_merge_a_near_duplicate():
    rng = np.random.RandomState(0)
    a = rng.normal(size=100)
    values = np.column_stack([a, 0.99 * a + 0.01 * rng.normal(size=100), rng.normal(size=100)])
    groups = correlated_groups(values, ["a", "a_clone", "noise"])
    merged = [group for group in groups if group["grouped"]]
    assert len(merged) == 1
    assert merged[0]["features"] == ["a", "a_clone"]
    assert merged[0]["max_abs_correlation"] > 0.9
    assert [group for group in groups if not group["grouped"]][0]["features"] == ["noise"]


def test_grouped_permutation_importance_attributes_the_pair_jointly():
    rng = np.random.RandomState(1)
    a = rng.normal(size=160)
    values = np.column_stack([a, a + 0.01 * rng.normal(size=160), rng.normal(size=160)])
    y = 3.0 * a
    model = LinearRegression().fit(values, y)
    groups = correlated_groups(values, ["a", "a_clone", "noise"])
    for group in groups:
        group["indices"] = [
            ["a", "a_clone", "noise"].index(feature) for feature in group["features"]
        ]
    results = grouped_permutation_importance(model, values, y, groups, n_repeats=3)
    by_feature = {tuple(item["features"]): item for item in results}
    pair = by_feature[("a", "a_clone")]
    noise = by_feature[("noise",)]
    assert pair["grouped"] is True
    assert pair["mean"] > 0.5
    assert abs(noise["mean"]) < 0.1


def test_out_of_distribution_flags_an_extreme_row_and_ignores_a_constant_column():
    rng = np.random.RandomState(2)
    reference = np.column_stack([rng.normal(size=400), rng.normal(size=400), np.ones(400)])
    query = np.array([[0.0, 0.0, 1.0], [40.0, 40.0, 1.0]])
    result = out_of_distribution(reference, query)
    assert result["method"] == "robust_mahalanobis"
    assert result["features_used"] == 2
    assert result["flagged"] == [False, True]
    assert result["distances"][1] > result["distances"][0]


def test_out_of_distribution_degrades_to_one_dimension():
    rng = np.random.RandomState(3)
    reference = np.column_stack([rng.normal(size=200), np.zeros(200)])
    result = out_of_distribution(reference, np.array([[0.0, 0.0], [10.0, 0.0]]))
    assert result["method"] == "robust_max_abs_z"
    assert result["flagged"] == [False, True]


def test_sign_consistency_marks_a_flipping_feature_unstable():
    samples = np.array(
        [
            [0.4, 0.2, 0.0],
            [0.3, -0.2, 0.0],
            [0.5, 0.2, 1e-9],
            [0.2, -0.2, -1e-9],
        ]
    )
    report = {entry["feature"]: entry for entry in sign_consistency(samples, ["stable", "flip", "ties"])}
    assert report["stable"]["sign_consistency"] == 1.0
    assert report["stable"]["unstable"] is False
    assert report["flip"]["sign_consistency"] == pytest.approx(0.5)
    assert report["flip"]["unstable"] is True
    assert report["ties"]["n_fits"] == 4


def test_explain_adds_grouped_importance_and_fold_stability(client, regression_task):
    response = client.post(
        "/api/evaluation/%s/linear/explain" % regression_task,
        json={
            "max_rows": 150,
            "n_repeats": 3,
            "top_k": 2,
            "stability_samples": 1,
            "stability_folds": 3,
            "random_state": 0,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    grouped = body["grouped_importance"]
    assert grouped["skipped_reason"] is None
    assert any(group["grouped"] for group in grouped["groups"])
    assert any(item["grouped"] for item in grouped["importances"])
    stability = body["fold_importance_stability"]
    assert stability["available"] is True
    assert stability["n_folds"] == 3
    assert stability["refit_per_fold"] is False
    assert {entry["feature"] for entry in stability["features"]} == {"a", "b", "c"}
    assert body["provenance"]["method"] == "permutation_importance"
    assert body["provenance"]["model_version"] == "linear"
    assert body["provenance"]["dataset_fingerprint"] == "deadbeefcafe"


def test_ale_exposes_the_explained_quantity_and_provenance(client, classification_task):
    response = client.post(
        "/api/evaluation/%s/logit/ale" % classification_task,
        json={"max_rows": 150, "n_bins": 8, "top_k": 2, "stability_samples": 1},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["explained_quantity"] == "probability"
    assert body["provenance"]["method"] == "accumulated_local_effects"
    assert body["provenance"]["settings"]["n_bins"] == 8


def test_row_explanation_reports_reconstruction_status_and_ood(client, regression_task):
    response = client.post(
        "/api/evaluation/%s/linear/explain/row" % regression_task,
        json={"row_index": 3, "orders": 2, "reference_max_rows": 150},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["explained_quantity"] == "prediction"
    assert body["reconstruction"]["status"] == "exact"
    assert body["reconstruction"]["reliable"] is True
    assert body["reconstruction"]["max_abs_residual"] <= body["reconstruction"]["tolerance"]
    assert body["out_of_distribution"]["available"] is True
    assert body["out_of_distribution"]["row_flagged"] is False
    assert body["provenance"]["method"] == "ordered_occlusion_chain"
