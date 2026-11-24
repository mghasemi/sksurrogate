"""API tests for paired model comparison (``POST /api/comparison/{task}``).

The fixtures write bundles directly instead of running a search: the point is a
*known* paired delta, so the compared estimators are deliberately fixed
(informative linear model versus a mean/prior baseline) and the endpoint must
detect the ordering, refuse degenerate requests, and keep the exploratory
selection caveat in the payload.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

if importlib.util.find_spec("httpx") is None:  # httpx2 ships the same TestClient
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402
from SKSurrogate.modelbundle import ModelBundle, save_bundle  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("api-comparison")
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
    )
    directory = settings.task_bundles_dir(task)
    directory.mkdir(parents=True, exist_ok=True)
    save_bundle(bundle, directory / ("%s.bundle" % version))
    return version


def _regression_frame(n=200, seed=5):
    rng = np.random.RandomState(seed)
    features = rng.normal(size=(n, 3))
    frame = pd.DataFrame(features, columns=["a", "b", "c"])
    frame["y"] = 2.0 * features[:, 0] + 0.5 * features[:, 1] + 0.1 * rng.normal(size=n)
    return frame


def _classification_frame(n=200, seed=6):
    rng = np.random.RandomState(seed)
    features = rng.normal(size=(n, 3))
    frame = pd.DataFrame(features, columns=["a", "b", "c"])
    frame["y"] = (features[:, 0] + features[:, 1] > 0).astype(int)
    return frame


@pytest.fixture(scope="module")
def regression_task(client):
    frame = _regression_frame()
    _register(client, "cmp-reg", frame)
    X, y = frame[["a", "b", "c"]], frame["y"]
    _write_bundle("cmp-reg", "linear", LinearRegression(), X, y)
    _write_bundle("cmp-reg", "mean", DummyRegressor(strategy="mean"), X, y)
    _write_bundle("cmp-reg", "linear-copy", LinearRegression(), X, y)
    return "cmp-reg"


@pytest.fixture(scope="module")
def classification_task(client):
    frame = _classification_frame()
    _register(client, "cmp-clf", frame)
    X, y = frame[["a", "b", "c"]], frame["y"]
    _write_bundle("cmp-clf", "logit", LogisticRegression(max_iter=500), X, y)
    _write_bundle("cmp-clf", "prior", DummyClassifier(strategy="prior"), X, y)
    return "cmp-clf"


def _compare(client, task, **body):
    payload = {"n_splits": 6, "max_rows": 200, "n_resamples": 400, **body}
    return client.post("/api/comparison/%s" % task, json=payload)


def test_detects_the_known_paired_improvement(client, regression_task):
    response = _compare(
        client,
        regression_task,
        model_versions=["mean", "linear"],
        scoring="r2",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["family"] == "Regression"
    assert body["higher_is_better"] is True
    assert body["cv"]["folds"] == 6
    (comparison,) = body["statistics"]["comparisons"]
    assert comparison["treatment"] == "linear"
    assert comparison["control"] == "mean"
    assert comparison["mean_delta"] > 0.5
    assert comparison["wins"] == 6
    assert comparison["ci_low"] > 0.0
    assert comparison["distinguishable"] is True
    # The linear model is both better and on the Pareto front.
    assert "linear" in body["trade_off"]["pareto_frontier"]
    assert body["trade_off"]["simpler_within_tolerance"]["best"] == "linear"
    assert body["score_provenance"]["exploratory"] is True
    assert any("exploratory" in warning for warning in body["warnings"])
    for record in body["models"]:
        assert record["scored_folds"] == 6
        assert record["fit_time_s"] is not None
        assert record["parameters"] is not None


def test_subsampling_the_comparison_rows_still_measures_latency(client, regression_task):
    """A capped ``max_rows`` samples rows; the latency probe must not treat the
    positional index as column labels (a live 500 that a full-frame test missed)."""
    response = _compare(
        client,
        regression_task,
        model_versions=["linear", "mean"],
        scoring="r2",
        max_rows=150,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rows"] == 150
    assert all(record["predict_time_ms"] is not None for record in body["models"])


def test_identical_models_are_not_distinguishable(client, regression_task):
    response = _compare(
        client,
        regression_task,
        model_versions=["linear", "linear-copy"],
        scoring="r2",
    )
    assert response.status_code == 200, response.text
    (comparison,) = response.json()["statistics"]["comparisons"]
    assert comparison["mean_delta"] == pytest.approx(0.0)
    assert comparison["ties"] == 6
    assert comparison["wilcoxon_p"] == 1.0
    assert comparison["distinguishable"] is False


def test_lower_is_better_metric_orients_the_delta(client, regression_task):
    response = _compare(
        client,
        regression_task,
        model_versions=["linear", "mean"],
        scoring="neg_root_mean_squared_error",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["statistics"]["metric_direction"] == "lower_is_better"
    (comparison,) = body["statistics"]["comparisons"]
    # Oriented deltas are positive when the treatment is better, even though
    # the raw scorer is negated (a less negative RMSE is a smaller error).
    assert comparison["treatment"] == "mean"
    assert comparison["mean_delta"] > 0.0
    assert comparison["distinguishable"] is True


def test_classifier_comparison_reports_calibration(client, classification_task):
    response = _compare(
        client,
        classification_task,
        model_versions=["prior", "logit"],
        scoring="accuracy",
    )
    assert response.status_code == 200, response.text
    models = {record["model_version"]: record for record in response.json()["models"]}
    calibration = models["logit"]["calibration"]
    assert calibration["available"] is True
    assert 0.0 <= calibration["brier_score"] <= 2.0
    assert 0.0 <= calibration["expected_calibration_error"] <= 1.0
    assert models["prior"]["calibration"]["available"] is True


def test_a_single_model_is_rejected_with_guidance(client, regression_task):
    response = _compare(client, regression_task, model_versions=["linear"])
    assert response.status_code == 400
    assert "at least two" in response.json()["detail"]


def test_a_control_outside_the_comparison_is_rejected(client, regression_task):
    response = _compare(
        client, regression_task, model_versions=["linear", "mean"], control="elsewhere"
    )
    assert response.status_code == 400
    assert "control" in response.json()["detail"]


def test_an_unknown_version_is_a_404(client, regression_task):
    response = _compare(client, regression_task, model_versions=["linear", "nope"])
    assert response.status_code == 404


def test_mixed_problem_families_are_rejected(client, regression_task):
    # Save a classifier bundle onto the regression task so the family check
    # itself is what rejects the request (a bundle from another task is a 404).
    frame = _regression_frame()
    X, y = frame[["a", "b", "c"]], (frame["y"] > frame["y"].median()).astype(int)
    _write_bundle("cmp-reg", "classifier", LogisticRegression(max_iter=500), X, y)
    response = _compare(client, "cmp-reg", model_versions=["linear", "classifier"])
    assert response.status_code == 400
    assert "same problem family" in response.json()["detail"]


def test_an_unknown_metric_is_a_400(client, regression_task):
    response = _compare(
        client,
        regression_task,
        model_versions=["linear", "mean"],
        scoring="not_a_metric",
    )
    assert response.status_code == 400


def test_repeated_folds_skip_calibration_with_a_reason(client, classification_task):
    response = _compare(
        client,
        classification_task,
        model_versions=["prior", "logit"],
        scoring="accuracy",
        repeats=2,
    )
    assert response.status_code == 200, response.text
    models = {record["model_version"]: record for record in response.json()["models"]}
    calibration = models["logit"]["calibration"]
    assert calibration["available"] is False
    assert calibration["reason"] == "repeated_folds_not_supported"
