"""Ensemble endpoints: combining the task's trained models, and evaluating the result.

The point of these tests is not only that ``/api/ensembles`` works, but that its output
is an ordinary bundle: the same nested-CV job, learning curves, diagnostic curves and
run report that evaluate any other model must evaluate an ensemble without special
cases. `test_the_result_is_evaluated_by_the_existing_endpoints` is that claim.

Member bundles are built directly (fitted pipeline + ``save_bundle``) instead of by
running experiments: an EOA search on this fixture burns minutes of Gaussian-process
fits regardless of dataset size, and the contract under test is the combining, not the
search (the same reasoning as tests/test_mixed_type_experiments.py).
"""

import importlib.util
import json
import time

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402
from api.training import load_partition  # noqa: E402
from SKSurrogate import ENSEMBLE_METHODS, load_bundle, save_bundle  # noqa: E402
from SKSurrogate.DataProcess import (  # noqa: E402
    FoldLocalDataPreprocessor,
    frame_tolerant,
)
from SKSurrogate.modelbundle import ModelBundle  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    return TestClient(app)


def _register(client, task, n=120):
    """Register a mixed-type classification partition with one free-text column."""
    rng = np.random.RandomState(0)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "surname": rng.choice(["MacDonald", "Hargrave", "Smith"], size=n),
        }
    )
    frame["y"] = ((frame["a"] + 0.4 * rng.normal(size=n)) > 0).astype(int)
    response = client.post(
        "/api/datasets/%s/register" % task,
        data={"target": "y", "partition": "train"},
        files={"file": ("t.csv", frame.to_csv(index=False), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return frame


def _wait(client, job_id, attempts=900):
    for _ in range(attempts):
        record = client.get("/api/jobs/" + job_id).json()
        if record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def _store_member(task, estimator, *, version, label=None, fit=True):
    """Fit ``estimator`` on the task's train partition and store it as a bundle version."""
    from sklearn.pipeline import Pipeline

    frame = load_partition(task, "train")
    features = frame.drop(columns=["y"])
    model = frame_tolerant(
        Pipeline([("preprocess", FoldLocalDataPreprocessor()), ("model", estimator)]),
        columns=list(features.columns),
    )
    if fit:
        model.fit(features, frame["y"])
    bundle = ModelBundle(
        model,
        task_name=task,
        schema={column: {"dtype": str(features[column].dtype)} for column in features.columns},
        metadata={"label": label} if label else {},
        metrics={"train_score": 0.5},
        model_version=version,
    )
    path = settings.task_bundles_dir(task) / (version + ".bundle")
    save_bundle(bundle, path)
    return path


def _two_members(task, client):
    """Two members of different kinds, so a combination actually has something to do."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    _register(client, task)
    _store_member(task, LogisticRegression(max_iter=500), version="member-lr", label="Logistic")
    _store_member(task, DecisionTreeClassifier(max_depth=3, random_state=0), version="member-dt")


def _fit(client, task, *, members, method, params=None, **extra):
    body = {"members": members, "method": method, "params": params or {}, **extra}
    response = client.post("/api/ensembles/%s/fit" % task, json=body)
    assert response.status_code == 200, response.text
    record = _wait(client, response.json()["job_id"])
    assert record["status"] == "completed", record.get("error")
    return record


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def test_methods_and_candidates_describe_what_can_be_combined(client):
    _two_members("churn", client)

    methods = client.get("/api/ensembles/churn/methods").json()
    assert methods["family"] == "Classification"
    assert [entry["value"] for entry in methods["methods"]] == list(ENSEMBLE_METHODS)
    assert methods["caveats"]  # the comparison caveat travels with the capability list
    assert methods["max_members"] == 12
    stacking = next(entry for entry in methods["methods"] if entry["value"] == "stacking")
    assert [spec["name"] for spec in stacking["parameters"]] == ["final_estimator", "cv", "passthrough"]

    candidates = client.get("/api/ensembles/churn/candidates").json()["candidates"]
    versions = {entry["model_version"] for entry in candidates}
    assert {"member-lr", "member-dt"} <= versions
    entry = next(item for item in candidates if item["model_version"] == "member-lr")
    assert entry["family"] == "Classification"
    assert entry["is_ensemble"] is False
    assert entry["label"] == "Logistic"
    assert "model" not in entry  # never the fitted object


def test_methods_on_a_task_with_no_models_reports_no_family(client):
    _register(client, "empty")
    methods = client.get("/api/ensembles/empty/methods").json()
    assert methods["family"] is None
    assert client.get("/api/ensembles/empty/candidates").json()["candidates"] == []


# --------------------------------------------------------------------------- #
# Fitting every method
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ENSEMBLE_METHODS)
def test_each_method_produces_an_ensemble_bundle(client, method):
    _two_members("churn", client)
    params = {
        "bagging": {"n_estimators": 3},
        "boosting": {"n_estimators": 3},
        "stacking": {"cv": 2},
    }.get(method, {})

    record = _fit(
        client, "churn", members=["member-lr", "member-dt"], method=method, params=params
    )
    result = record["result"]
    version = result["model_version"]
    assert result["method"] == method
    assert result["family"] == "Classification"
    # The honest score is the out-of-fold one; the refit training score is there too.
    assert result["metrics"]["cv_score"] is not None
    assert result["metrics"]["train_score"] is not None
    assert [entry["model_version"] for entry in result["members"]] == ["member-lr", "member-dt"]
    assert all(entry["cv_mean"] is not None for entry in result["members"])

    stored = load_bundle(
        settings.task_bundles_dir("churn") / (version + ".bundle"), strict_dependencies=False
    )
    metadata = stored.metadata["ensemble"]
    assert metadata["schema"] == "sksurrogate.ensemble.v1"
    assert metadata["method"] == method
    assert metadata["family"] == "Classification"
    assert len(metadata["members"]) == 2
    assert metadata["scores"]["cv_score"] is not None
    assert metadata["caveats"]
    assert stored.metrics["cv_score"] is not None

    listed = client.get("/api/ensembles/churn").json()["ensembles"]
    entry = next(item for item in listed if item["model_version"] == version)
    assert entry["method"] == method
    assert entry["member_count"] == 2

    detail = client.get("/api/ensembles/churn/%s" % version).json()
    assert detail["model_version"] == version
    assert detail["ensemble"]["method"] == method
    assert "model" not in detail
    # The produced bundle is itself offered as a member of a later ensemble.
    candidates = client.get("/api/ensembles/churn/candidates").json()["candidates"]
    created = next(item for item in candidates if item["model_version"] == version)
    assert created["is_ensemble"] is True
    assert created["ensemble"]["method"] == method


def test_member_and_ensemble_scores_share_the_stored_splitter(client):
    _two_members("churn", client)
    response = client.put(
        "/api/datasets/churn/cv", json={"spec": {"type": "KFold", "n_splits": 3}}
    )
    assert response.status_code == 200, response.text

    record = _fit(client, "churn", members=["member-lr", "member-dt"], method="voting")
    stored = load_bundle(
        settings.task_bundles_dir("churn") / (record["result"]["model_version"] + ".bundle"),
        strict_dependencies=False,
    )
    metadata = stored.metadata["ensemble"]
    assert metadata["cv"]["type"] == "KFold"
    # Every row in the comparison is computed on the same three folds.
    assert len(metadata["scores"]["cv_folds"]) == 3
    assert all(len(entry["cv_folds"]) == 3 for entry in metadata["members"])


def test_weights_are_applied_positionally(client):
    _two_members("churn", client)
    unweighted = _fit(client, "churn", members=["member-lr", "member-dt"], method="averaging")
    weighted = _fit(
        client,
        "churn",
        members=["member-lr", "member-dt"],
        method="averaging",
        params={"weights": [3.0, 1.0]},
    )
    assert unweighted["result"]["model_version"] != weighted["result"]["model_version"]
    stored = load_bundle(
        settings.task_bundles_dir("churn") / (weighted["result"]["model_version"] + ".bundle"),
        strict_dependencies=False,
    )
    assert stored.metadata["ensemble"]["parameters"]["weights"] == [3.0, 1.0]


def test_register_on_fit_puts_the_ensemble_in_the_registry(client):
    _two_members("churn", client)
    record = _fit(
        client, "churn", members=["member-lr", "member-dt"], method="voting", register=True
    )
    version = record["result"]["model_version"]
    assert record["result"]["registered"] is True

    summary = client.get("/api/registry/churn/summary").json()
    assert summary["versions"][version] == "candidate"
    assert summary["aliases"]["latest"] == version


def test_without_register_the_ensemble_stays_an_unregistered_bundle(client):
    _two_members("churn", client)
    record = _fit(client, "churn", members=["member-lr", "member-dt"], method="voting")
    version = record["result"]["model_version"]

    assert record["result"]["registered"] is False
    # 404 from the registry, while the bundle itself is served normally.
    assert client.get("/api/registry/churn/summary").status_code in (200, 404)
    response = client.get("/api/registry/churn/summary").json()
    assert version not in (response.get("versions") or {})
    assert client.get("/api/ensembles/churn/%s" % version).status_code == 200


# --------------------------------------------------------------------------- #
# The result is evaluated by the endpoints that already exist
# --------------------------------------------------------------------------- #


def test_the_result_is_evaluated_by_the_existing_endpoints(client):
    """An ensemble is a bundle: nested CV, curves and the run report must just work.

    The diagnostic-curve path is the interesting one — it used to add its own encoder
    in front of any estimator that did not look frame-preprocessing, which for an
    ensemble would hand every member an already-encoded matrix.
    """
    _two_members("churn", client)
    record = _fit(
        client, "churn", members=["member-lr", "member-dt"], method="stacking", params={"cv": 2}
    )
    version = record["result"]["model_version"]

    # 1. Learning curves — predict from positional arrays through the ensemble.
    curves = client.get(
        "/api/evaluation/churn/learning-curves",
        params={"scoring": "accuracy", "n_points": 3, "model_versions": version},
    )
    assert curves.status_code == 200, curves.text
    payload = curves.json()
    assert payload["errors"] == [], payload["errors"]
    assert len(payload["series"]) == 1
    assert all(score is not None for score in payload["series"][0]["validation_scores_mean"])

    # 2. Diagnostic curves — classification bundle, computed server-side.
    diagnostics = client.get("/api/evaluation/churn/%s/curves" % version)
    assert diagnostics.status_code == 200, diagnostics.text
    assert diagnostics.json()["errors"] == [], diagnostics.json()["errors"]
    assert diagnostics.json()["roc"]["auc"] is not None

    # 3. Nested CV — the out-of-fold estimate that accounts for member selection.
    response = client.post(
        "/api/evaluation/churn/nested-cv",
        json={"model_version": version, "inner_cv": 2, "outer_cv": 2, "scoring": "accuracy", "repeats": 2},
    )
    assert response.status_code == 200, response.text
    nested = _wait(client, response.json()["job_id"])
    assert nested["status"] == "completed", nested.get("error")
    assert nested["result"]["mean_outer_score"] is not None

    # 4. Run report — the provenance record names the ensemble members.
    report = client.get("/api/bundles/churn/%s/report.md" % version)
    assert report.status_code == 200, report.text
    assert "ensemble" in report.text
    assert "member-lr" in report.text


def test_the_ensemble_bundle_serves_predictions(client):
    """Serving must work through the control plane like any other bundle."""
    _two_members("churn", client)
    record = _fit(client, "churn", members=["member-lr", "member-dt"], method="voting")
    version = record["result"]["model_version"]
    frame = load_partition("churn", "train")
    rows = json.loads(frame.drop(columns=["y"]).head(3).to_json(orient="records"))

    response = client.post(
        "/api/inference/churn/predict",
        json={"model_version": version, "rows": rows},
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["predictions"]) == 3


def _store_cat_cols_member(task, estimator, *, version):
    """Store a member in the shape ``AML``'s ``cat_cols`` search builds.

    The encoder comes first and resolves its columns *by name*, which is what the
    ensemble aggregators break when they validate the frame down to an array.
    """
    from category_encoders.one_hot import OneHotEncoder
    from imblearn.pipeline import Pipeline as ImbPipeline

    frame = load_partition(task, "train")
    features = frame.drop(columns=["y"])
    model = ImbPipeline(
        [
            (
                "categorical_encoder",
                OneHotEncoder(cols=["surname"], return_df=False, handle_unknown="value"),
            ),
            ("model", estimator),
        ]
    ).fit(features, frame["y"])
    bundle = ModelBundle(
        model,
        task_name=task,
        schema={column: {"dtype": str(features[column].dtype)} for column in features.columns},
        metadata={"label": version},
        metrics={"train_score": 0.5},
        model_version=version,
    )
    save_bundle(bundle, settings.task_bundles_dir(task) / (version + ".bundle"))


def test_cat_cols_members_combine_under_bagging(client):
    """Regression for the measured SPRA job: two ``cat_cols`` bundles, bagging + weights.

    Both members start with ``OneHotEncoder(cols=['surname'])``. ``BaggingClassifier``
    validates the frame into a positional matrix before fitting its base estimator, so
    the encoder lost its column names and the job failed with ``X does not contain the
    columns listed in cols ... non-DataFrame input at fit`` — first inside the weight
    search, then in the final fit. The combination must complete, and the stored
    ensemble must still serve predictions for frame rows.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    _register(client, "churn")
    _store_cat_cols_member("churn", LogisticRegression(max_iter=300), version="cat-lr")
    _store_cat_cols_member(
        "churn", DecisionTreeClassifier(max_depth=3, random_state=0), version="cat-dt"
    )

    record = _fit(
        client,
        "churn",
        members=["cat-lr", "cat-dt"],
        method="bagging",
        params={"n_estimators": 4},
        weight_optimization={"strategy": "least_squares"},
    )
    result = record["result"]
    assert len(result["weight_optimization"]["weights"]) == 2

    rows = json.loads(
        load_partition("churn", "train").drop(columns=["y"]).head(3).to_json(orient="records")
    )
    response = client.post(
        "/api/inference/churn/predict",
        json={"model_version": result["model_version"], "rows": rows},
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["predictions"]) == 3


def test_diagnostic_curves_use_vote_fractions_of_a_hard_voting_ensemble(client):
    """A hard-voting ensemble has no probabilities; its vote share is the score.

    Measured live on task SPRA: ``GET /api/evaluation/SPRA/<version>/curves?bins=10``
    answered 500 with ``AttributeError: 'VotingClassifier' object has no attribute
    'decision_function'`` — the gain/lift fallback called the method unguarded, and
    ``VotingClassifier(voting="hard")`` exposes neither ``predict_proba`` nor
    ``decision_function``. The curves are now drawn from the members' vote fractions
    (their argmax *is* the hard prediction) and the provenance says so, because a
    two-member vote score is coarse and is not a calibrated probability.
    """
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    _register(client, "churn")
    frame = load_partition("churn", "train")
    features = frame.drop(columns=["y"])
    columns = list(features.columns)

    def member(estimator):
        return frame_tolerant(
            Pipeline([("preprocess", FoldLocalDataPreprocessor()), ("model", estimator)]),
            columns=columns,
        )

    hard = VotingClassifier(
        [
            ("lr", member(LogisticRegression(max_iter=400))),
            ("dt", member(DecisionTreeClassifier(max_depth=3, random_state=0))),
        ],
        voting="hard",
    )
    hard.fit(features, frame["y"])
    # The two methods the endpoint used to assume are both absent.
    assert not hasattr(hard, "decision_function")
    assert not hasattr(hard, "predict_proba")
    save_bundle(
        ModelBundle(
            hard,
            task_name="churn",
            model_version="hard-vote",
            schema={column: str(features[column].dtype) for column in features.columns},
            metadata={},
            metrics={"train_score": 0.5},
        ),
        settings.task_bundles_dir("churn") / "hard-vote.bundle",
    )

    response = client.get("/api/evaluation/churn/hard-vote/curves?bins=10")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["errors"] == [], payload["errors"]
    assert payload["evaluation_protocol"]["score_source"] == "vote_fraction"
    assert payload["roc"]["auc"] is not None
    assert set(payload) >= {"roc", "calibration", "cumulative_gain", "lift"}
    # A two-member vote fraction takes 0, 0.5 or 1 — the ROC has only those thresholds.
    assert len(payload["roc"]["fpr"]) <= 4
    assert payload["n_test_samples"] > 0


def test_a_model_with_no_scores_at_all_is_refused_with_a_remedy():
    """No probabilities, no decision function, no members: say why, don't crash."""
    from api.routers import evaluation as evaluation_router

    class _VoteOnly:
        def predict(self, X):  # noqa: ANN001, N802 - sklearn-shaped stub
            raise AssertionError("no per-row score exists, so nothing should be predicted")

    with pytest.raises(ValueError) as ctx:
        evaluation_router._class_scores(_VoteOnly(), np.zeros((4, 2)), [0, 1])
    message = str(ctx.value)
    assert "_VoteOnly" in message
    assert "predict_proba" in message and "soft voting" in message


# --------------------------------------------------------------------------- #
# Weight optimization
# --------------------------------------------------------------------------- #


def test_weights_can_be_optimized_and_are_reported(client):
    _two_members("churn", client)
    record = _fit(
        client,
        "churn",
        members=["member-lr", "member-dt"],
        method="voting",
        params={"voting": "soft"},
        weight_optimization={"strategy": "least_squares"},
    )
    result = record["result"]
    optimized = result["weight_optimization"]
    assert optimized["strategy"] == "least_squares"
    assert len(optimized["weights"]) == 2
    assert abs(sum(optimized["weights"]) - 1.0) < 1e-6
    assert optimized["objective"] is not None
    # The vector is recorded on the bundle, on the member rows, and as the parameters.
    stored = load_bundle(
        settings.task_bundles_dir("churn") / (result["model_version"] + ".bundle"),
        strict_dependencies=False,
    )
    metadata = stored.metadata["ensemble"]
    assert metadata["weight_optimization"]["weights"] == optimized["weights"]
    assert metadata["parameters"]["weights"] == optimized["weights"]
    assert [entry["weight"] for entry in metadata["members"]] == optimized["weights"]
    # A weight-search caveat is added, quoting the measured optimism.
    assert any("weight" in caveat.lower() and "optimistic" in caveat.lower() for caveat in result["caveats"])


def test_each_weight_strategy_runs_and_reports_its_budget(client):
    _two_members("churn", client)
    for strategy, extra in (
        ("least_squares", {}),
        ("dirichlet", {"samples": 20}),
        ("greedy", {"rounds": 4}),
    ):
        record = _fit(
            client,
            "churn",
            members=["member-lr", "member-dt"],
            method="averaging",
            params={"weights": None},
            weight_optimization={"strategy": strategy, **extra},
        )
        optimized = record["result"]["weight_optimization"]
        assert optimized["strategy"] == strategy
        assert optimized["objective"] is not None
        if strategy == "least_squares":
            assert optimized["budget"] is None
        else:
            # A scorer-based search must say which budget it spent.
            assert optimized["budget"]
            assert optimized["objective_name"]


def test_weight_optimization_and_explicit_weights_conflict(client):
    _two_members("churn", client)
    response = _reject(
        client,
        "churn",
        params={"weights": [1.0, 1.0]},
        weight_optimization={"strategy": "least_squares"},
    )
    assert response.status_code == 400
    assert "not both" in response.json()["detail"]


def test_weight_optimization_is_refused_for_stacking(client):
    _two_members("churn", client)
    response = _reject(
        client, "churn", method="stacking", weight_optimization={"strategy": "least_squares"}
    )
    assert response.status_code == 400
    assert "Stacking learns its own combination" in response.json()["detail"]


def test_unknown_weight_strategy_is_refused(client):
    _two_members("churn", client)
    response = _reject(
        client, "churn", weight_optimization={"strategy": "bayesian"}
    )
    assert response.status_code == 400
    assert "Unknown weight strategy" in response.json()["detail"]


def test_bagging_and_boosting_accept_weights(client):
    _two_members("churn", client)
    for method, params in (
        ("bagging", {"n_estimators": 3, "weights": [0.0, 1.0]}),
        ("boosting", {"n_estimators": 3, "weights": [1.0, 0.0]}),
    ):
        record = _fit(client, "churn", members=["member-lr", "member-dt"], method=method, params=params)
        stored = load_bundle(
            settings.task_bundles_dir("churn") / (record["result"]["model_version"] + ".bundle"),
            strict_dependencies=False,
        )
        assert stored.metadata["ensemble"]["parameters"]["weights"] == params["weights"]


def test_zero_weight_is_reported_as_a_dropped_member(client):
    """A strategy that collapses a member to zero says so instead of hiding it."""
    _two_members("churn", client)
    record = _fit(
        client,
        "churn",
        members=["member-lr", "member-dt"],
        method="voting",
        params={"voting": "soft"},
        weight_optimization={"strategy": "greedy", "rounds": 4},
    )
    optimized = record["result"]["weight_optimization"]
    zeroed = [index for index, weight in enumerate(optimized["weights"]) if weight == 0.0]
    assert set(optimized["dropped"]) == {
        ["member-lr", "member-dt"][index] for index in zeroed
    }


# --------------------------------------------------------------------------- #
# Per-member calibration
# --------------------------------------------------------------------------- #


def test_calibration_is_reported_with_before_and_after_probability_scores(client):
    _two_members("churn", client)
    record = _fit(
        client,
        "churn",
        members=["member-lr", "member-dt"],
        method="averaging",
        calibration={"method": "sigmoid", "cv": 3},
    )
    result = record["result"]
    assert result["calibration"]["method"] == "sigmoid"
    report = {entry["member"]: entry for entry in result["calibration"]["members"]}
    assert set(report) == {"member-lr", "member-dt"}
    for entry in report.values():
        assert entry["brier_before"] is not None and entry["brier_after"] is not None
        assert entry["log_loss_before"] is not None and entry["log_loss_after"] is not None
    # The member rows record that they were scored as calibrated members.
    assert all(entry["calibrated"] is True for entry in result["members"])
    assert any("calibrat" in caveat.lower() for caveat in result["caveats"])
    stored = load_bundle(
        settings.task_bundles_dir("churn") / (result["model_version"] + ".bundle"),
        strict_dependencies=False,
    )
    assert stored.metadata["ensemble"]["calibration"]["method"] == "sigmoid"


def test_isotonic_calibration_is_available(client):
    _two_members("churn", client)
    record = _fit(
        client,
        "churn",
        members=["member-lr", "member-dt"],
        method="averaging",
        calibration={"method": "isotonic", "cv": 3, "ensemble": "single"},
    )
    assert record["result"]["calibration"]["method"] == "isotonic"


def test_calibration_is_refused_for_regression_members(client):
    from sklearn.linear_model import LinearRegression

    _two_members("churn", client)
    _store_member("churn", LinearRegression(), version="member-reg")
    _store_member("churn", LinearRegression(fit_intercept=False), version="member-reg2")
    response = _reject(
        client,
        "churn",
        members=["member-reg", "member-reg2"],
        method="averaging",
        calibration={"method": "sigmoid"},
    )
    assert response.status_code == 400
    assert "classification members only" in response.json()["detail"]


def test_unknown_calibration_method_and_token_are_refused(client):
    _two_members("churn", client)
    response = _reject(client, "churn", calibration={"method": "platt"})
    assert response.status_code == 400
    assert "Unknown calibration method" in response.json()["detail"]
    response = _reject(client, "churn", calibration={"method": "sigmoid", "ensemble": "sometimes"})
    assert response.status_code == 400
    assert "ensemble" in response.json()["detail"]


def test_calibration_plus_bagging_emits_no_sample_weight_warning(client):
    """Calibration must sit outside the resampling wrapper, or sklearn warns.

    Nested inside, ``CalibratedClassifierCV`` receives the bootstrap weights
    (``BaggingClassifier`` forwards them to any base whose ``fit`` declares them) and
    warns that the calibration is "likely to be incorrect". The fit is driven through the
    core so the warning filter sees it.
    """
    import warnings

    from SKSurrogate import fit_ensemble
    from SKSurrogate.DataProcess import FoldLocalDataPreprocessor
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    _two_members("churn", client)
    frame = load_partition("churn", "train")
    features = frame.drop(columns=["y"])
    columns = list(features.columns)

    def member(estimator):
        from SKSurrogate.DataProcess import frame_tolerant

        return frame_tolerant(
            Pipeline([("preprocess", FoldLocalDataPreprocessor()), ("model", estimator)]),
            columns=columns,
        )

    members = [
        ("a", member(LogisticRegression(max_iter=300))),
        ("b", member(LogisticRegression(max_iter=300, C=0.5))),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_ensemble(
            "bagging",
            members,
            features,
            frame["y"].to_numpy(),
            params={"n_estimators": 3},
            calibration={"method": "sigmoid", "cv": 3},
            columns=columns,
            random_state=0,
            n_jobs=1,
        )
    assert [str(item.message) for item in caught if "sample_weight" in str(item.message)] == []


def test_methods_endpoint_publishes_the_new_capabilities(client):
    _two_members("churn", client)
    methods = client.get("/api/ensembles/churn/methods").json()

    assert methods["calibration"]["available"] is True
    assert {entry["value"] for entry in methods["calibration"]["methods"]} == {"sigmoid", "isotonic"}
    assert {spec["name"] for spec in methods["calibration"]["parameters"]} == {"method", "cv", "ensemble"}
    assert methods["weight_optimization"]["available"] is True
    assert {entry["value"] for entry in methods["weight_optimization"]["strategies"]} == {
        "least_squares",
        "dirichlet",
        "greedy",
    }
    accepts = {entry["value"]: entry["accepts_weights"] for entry in methods["methods"]}
    assert accepts == {
        "voting": True,
        "averaging": True,
        "stacking": False,
        "bagging": True,
        "boosting": True,
    }


# --------------------------------------------------------------------------- #
# The honest nested estimate
# --------------------------------------------------------------------------- #


def test_honest_cv_prices_the_weight_search(client):
    """The nested estimate must exist next to the optimistic one, with its own caveat."""
    _two_members("churn", client)
    record = _fit(
        client,
        "churn",
        members=["member-lr", "member-dt"],
        method="voting",
        params={"voting": "soft"},
        weight_optimization={"strategy": "least_squares"},
        honest_cv={"folds": 3, "inner_folds": 2},
    )
    result = record["result"]
    nested = result["nested"]
    assert nested["mean"] is not None
    assert nested["std"] is not None
    assert len(nested["folds"]) == 3
    assert nested["errors"] == []
    # Both numbers are stored so the UI can show the gap.
    assert result["metrics"]["cv_score"] is not None
    assert nested["mean"] != result["metrics"]["cv_score"] or nested["std"] > 0
    stored = load_bundle(
        settings.task_bundles_dir("churn") / (result["model_version"] + ".bundle"),
        strict_dependencies=False,
    )
    assert stored.metadata["ensemble"]["scores"]["nested"]["mean"] == nested["mean"]
    assert "a weight vector was fitted on those folds" in (
        stored.metadata["reproducibility"]["evaluation_protocol"]["metric_scope"]
    )


def test_honest_cv_is_optional_and_absent_by_default(client):
    _two_members("churn", client)
    record = _fit(client, "churn", members=["member-lr", "member-dt"], method="voting")
    assert record["result"]["nested"] is None
    stored = load_bundle(
        settings.task_bundles_dir("churn") / (record["result"]["model_version"] + ".bundle"),
        strict_dependencies=False,
    )
    assert stored.metadata["ensemble"]["scores"]["nested"] is None


def test_malformed_honest_cv_is_refused(client):
    _two_members("churn", client)
    response = _reject(client, "churn", honest_cv={"folds": 1})
    assert response.status_code == 400
    assert "honest_cv.folds" in response.json()["detail"]
    response = _reject(client, "churn", honest_cv={"folds": "three"})
    assert response.status_code == 400
    assert "honest_cv.folds" in response.json()["detail"]


def test_a_failed_honest_estimate_does_not_lose_the_ensemble(client):
    """The model is the artifact; the honest estimate is a measurement of it.

    Driven through the API function rather than the route, because the route caps the
    fold count at 20 and this fixture can fill 20 folds; 500 folds cannot be filled.
    """
    from api.training import fit_ensemble_bundle

    _two_members("churn", client)
    bundle, summary = fit_ensemble_bundle(
        "churn",
        members=["member-lr", "member-dt"],
        method="voting",
        params={"voting": "soft"},
        honest_cv={"folds": 500, "inner_folds": 2},
        scoring="accuracy",
        random_state=0,
    )
    assert summary["nested"]["mean"] is None
    assert summary["nested"]["errors"][0]["fold"] == -1
    assert "Could not build" in summary["nested"]["errors"][0]["detail"]
    # The ensemble itself is complete: fitted, scored and stored with the failure recorded.
    assert summary["metrics"]["cv_score"] is not None
    assert summary["model_version"] == bundle.model_version
    assert bundle.metadata["ensemble"]["scores"]["nested"]["mean"] is None
    assert bundle.metadata["ensemble"]["scores"]["cv_score"] is not None


def test_candidates_report_calibrated_and_weighted_flags(client):
    _two_members("churn", client)
    record = _fit(
        client,
        "churn",
        members=["member-lr", "member-dt"],
        method="voting",
        params={"voting": "soft"},
        calibration={"method": "sigmoid", "cv": 3},
        weight_optimization={"strategy": "least_squares"},
    )
    version = record["result"]["model_version"]
    candidates = client.get("/api/ensembles/churn/candidates").json()["candidates"]
    entry = next(item for item in candidates if item["model_version"] == version)
    assert entry["ensemble"]["calibrated"] is True
    assert entry["ensemble"]["weighted"] is True
    # And the listed ensemble row carries them too, so the table can badge them.
    listed = client.get("/api/ensembles/churn").json()["ensembles"]
    row = next(item for item in listed if item["model_version"] == version)
    assert row["calibrated"] is True and row["weighted"] is True


# --------------------------------------------------------------------------- #
# Rejections: at submit time, not inside the job
# --------------------------------------------------------------------------- #


def _reject(client, task, **body):
    body.setdefault("members", ["member-lr", "member-dt"])
    body.setdefault("method", "voting")
    response = client.post("/api/ensembles/%s/fit" % task, json=body)
    assert response.status_code in (400, 404), response.text
    return response


def test_unknown_member_is_a_404(client):
    _two_members("churn", client)
    response = _reject(client, "churn", members=["member-lr", "does-not-exist"])
    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


def test_duplicate_members_are_refused(client):
    _two_members("churn", client)
    response = _reject(client, "churn", members=["member-lr", "member-lr"])
    assert response.status_code == 400
    assert "twice" in response.json()["detail"]


def test_single_member_voting_is_refused(client):
    _two_members("churn", client)
    response = _reject(client, "churn", members=["member-lr"])
    assert response.status_code == 400
    assert "at least 2 member" in response.json()["detail"]


def test_unknown_parameter_is_refused(client):
    _two_members("churn", client)
    response = _reject(client, "churn", params={"n_estimators": 4})
    assert response.status_code == 400
    assert "does not take" in response.json()["detail"]


def test_mixed_family_members_are_refused(client):
    from sklearn.linear_model import LinearRegression

    _two_members("churn", client)
    _store_member("churn", LinearRegression(), version="member-reg", fit=False)
    response = _reject(client, "churn", members=["member-lr", "member-reg"])
    assert response.status_code == 400
    assert "family" in response.json()["detail"]


def test_soft_voting_without_probability_support_is_refused(client):
    from sklearn.svm import LinearSVC

    _two_members("churn", client)
    _store_member("churn", LinearSVC(), version="member-svc")
    response = _reject(
        client, "churn", members=["member-svc", "member-lr"], method="averaging"
    )
    assert response.status_code == 400
    assert "predict_proba" in response.json()["detail"]


def test_boosted_classification_member_without_sample_weight_is_refused(client):
    from sklearn.neighbors import KNeighborsClassifier

    _two_members("churn", client)
    _store_member("churn", KNeighborsClassifier(), version="member-knn")
    response = _reject(
        client, "churn", members=["member-knn", "member-dt"], method="boosting"
    )
    assert response.status_code == 400
    assert "sample_weight" in response.json()["detail"]


def test_unknown_method_is_refused(client):
    _two_members("churn", client)
    response = _reject(client, "churn", method="blend")
    assert response.status_code == 400
    assert "Unknown ensemble method" in response.json()["detail"]


def test_more_members_than_the_cap_are_refused(client):
    _two_members("churn", client)
    response = _reject(client, "churn", members=["member-%d" % index for index in range(13)])
    assert response.status_code == 400
    assert "At most" in response.json()["detail"]


def test_detail_of_a_non_ensemble_bundle_is_refused(client):
    _two_members("churn", client)
    response = client.get("/api/ensembles/churn/member-lr")
    assert response.status_code == 400
    assert "not an ensemble" in response.json()["detail"]


def test_detail_of_an_unknown_bundle_is_a_404(client):
    _two_members("churn", client)
    assert client.get("/api/ensembles/churn/nope").status_code == 404


def test_fit_on_an_unregistered_task_is_a_404(client):
    response = client.post(
        "/api/ensembles/ghost/fit", json={"members": ["a"], "method": "voting"}
    )
    assert response.status_code == 404
