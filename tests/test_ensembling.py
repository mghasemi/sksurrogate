"""Combining trained models with scikit-learn's meta-estimators.

Three measured properties drive these tests, because each one silently produces a
*wrong* model rather than an error:

``ArrayAsFrame`` must restore dtypes
    Bagging hands its base estimator a positional (mixed-type → object) array. A
    frame-first preprocessor decides per column whether it is numeric or categorical
    from the dtype it sees, so without dtype inference on the rebuild every numeric
    column is one-hot encoded as a categorical: measured on a 3-column frame with 300
    rows, the encoder expanded to 423 columns and the combined model scored near
    random (AUC 0.55 against 0.95 for the same member fitted on the frame).

A member must be array-safe to survive bagging
    Bagging indexes ``X`` positionally, so a member whose encoder selects columns *by
    name* fails outright unless it starts with the frame adapter, which is what
    ``frame_tolerant`` installs.

Boosting needs an explicitly named ``sample_weight``
    ``AdaBoost*`` gates its base estimator on
    ``has_fit_parameter(estimator_, "sample_weight")``; ``Pipeline`` never has that
    parameter, so a plain member pipeline is refused and
    ``SampleWeightRouter`` exists to route the weights to the pipeline's last step.
"""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from SKSurrogate import (
    ENSEMBLE_METHODS,
    EnsembleError,
    SampleWeightRouter,
    build_ensemble,
    calibrate_member,
    calibrate_members,
    calibration_report,
    cross_validated_scores,
    fit_ensemble,
    member_forms,
    member_scores,
    method_catalogue,
    nested_ensemble_cv,
    optimize_weights,
    out_of_fold_outputs,
    problem_family_of,
    resolve_family,
    supports_sample_weight,
    validate_params,
    validate_weight_optimization,
)
from SKSurrogate.DataProcess import (
    ArrayAsFrame,
    FoldLocalDataPreprocessor,
    frame_tolerant,
    preprocesses_frames,
)
from SKSurrogate.modelbundle import ModelBundle, load_bundle, save_bundle

COLUMNS = ["x1", "x2", "team"]
NUMERIC = ["x1", "x2"]


def _frame(n=160, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "team": rng.choice(["a", "b", "c"], size=n),
        }
    )
    frame["y_cls"] = (
        frame[["x1", "x2"]].sum(axis=1) * 1.2
        + (frame["team"] == "a") * 0.8
        + rng.normal(scale=0.5, size=n)
        > 0
    ).astype(int)
    frame["y_reg"] = (
        2.0 * frame["x1"] - frame["x2"] + (frame["team"] == "b") + rng.normal(scale=0.4, size=n)
    )
    return frame


def _member(estimator):
    """A member in the shape a preset bundle stores: frame-tolerant, fold-local encoder."""
    return frame_tolerant(
        Pipeline([("preprocess", FoldLocalDataPreprocessor()), ("model", estimator)]),
        columns=COLUMNS,
    )


def _cls_members():
    return [
        ("lr", _member(LogisticRegression(max_iter=500))),
        ("dt", _member(DecisionTreeClassifier(max_depth=3, random_state=0))),
    ]


def _reg_members():
    return [
        ("lin", _member(LinearRegression())),
        ("tree", _member(DecisionTreeRegressor(max_depth=3, random_state=0))),
    ]


def _small(method):
    """Parameters that keep the test's fits light without changing the combination."""
    if method == "bagging":
        return {"n_estimators": 3}
    if method == "boosting":
        return {"n_estimators": 3}
    if method == "stacking":
        return {"cv": 2}
    return {}


# --------------------------------------------------------------------------- #
# ArrayAsFrame dtype restoration (the fix bagging depends on)
# --------------------------------------------------------------------------- #


def test_array_as_frame_restores_numeric_dtypes_from_a_mixed_matrix():
    adapter = ArrayAsFrame(COLUMNS).fit(pd.DataFrame({name: [1.0] for name in COLUMNS}))

    rebuilt = adapter.transform(np.array([[1.0, 2.0, "a"], [3.0, 4.0, "b"]], dtype=object))

    dtypes = [str(dtype) for dtype in rebuilt.dtypes]
    # The two numeric columns stay float64 and the text column stays text. Under
    # pandas 3 a text column gets the new ``str`` dtype (PDEP-14) where pandas 2
    # reports ``object`` — either is "text, not numbers".
    assert dtypes[:2] == ["float64", "float64"]
    assert dtypes[2] in ("object", "str", "string")
    # A text column must stay text, not be coerced into numbers.
    assert rebuilt["team"].tolist() == ["a", "b"]


def test_array_as_frame_leaves_a_dataframe_untouched():
    adapter = ArrayAsFrame(COLUMNS).fit(pd.DataFrame({name: [1.0] for name in COLUMNS}))
    frame = pd.DataFrame({"x1": [1.0], "x2": [2.0], "team": ["a"]})
    assert adapter.transform(frame) is frame


def test_fitting_on_an_object_matrix_matches_fitting_on_the_frame():
    """The contract bagging and the learning-curve helper rely on."""
    frame = _frame()
    features = frame[COLUMNS]

    on_frame = _member(LogisticRegression(max_iter=500)).fit(features, frame["y_cls"])
    on_array = _member(LogisticRegression(max_iter=500)).fit(features.to_numpy(), frame["y_cls"])

    def dense(matrix):
        # The fold-local preprocessor returns a sparse matrix for an encoded frame and
        # a frame for a pass-through one, so normalise before comparing.
        if hasattr(matrix, "todense"):
            matrix = matrix.todense()
        return np.asarray(matrix, dtype=float)

    encoder_on_frame = on_frame.named_steps["preprocess"]
    encoder_on_array = on_array.named_steps["preprocess"]
    # Equal encodings, not merely equal shapes: a dtype-blind rebuild one-hot encodes
    # the numeric columns and the two "models" disagree on the same input.
    assert dense(encoder_on_frame.transform(features)).shape == dense(
        encoder_on_array.transform(features)
    ).shape
    assert np.array_equal(
        dense(encoder_on_frame.transform(features)), dense(encoder_on_array.transform(features))
    )
    assert np.array_equal(on_frame.predict(features), on_array.predict(features))


# --------------------------------------------------------------------------- #
# preprocesses_frames through meta-estimators
# --------------------------------------------------------------------------- #


def test_preprocesses_frames_sees_through_an_ensemble():
    """Otherwise nested CV stacks a second encoder in front of the members' own."""
    member = _member(LogisticRegression(max_iter=200))
    assert preprocesses_frames(member) is True

    for method in ("voting", "averaging", "stacking"):
        ensemble = build_ensemble(method, _cls_members(), columns=COLUMNS, params=_small(method))
        assert preprocesses_frames(ensemble) is True, method

    # A member that needs no encoding cannot make the whole ensemble frame-first.
    plain = Pipeline([("model", LogisticRegression(max_iter=200))])
    assert preprocesses_frames(plain) is False
    assert preprocesses_frames(build_ensemble("voting", [("a", plain), ("b", plain)])) is False


def test_preprocesses_frames_sees_through_a_sample_weight_router():
    """Boosting wraps members, so the wrapper must not hide the member's encoder."""
    member = _member(DecisionTreeClassifier(max_depth=1, random_state=0))
    assert preprocesses_frames(member) is True
    assert preprocesses_frames(SampleWeightRouter(member)) is True
    assert preprocesses_frames(SampleWeightRouter(Pipeline([("model", LogisticRegression())]))) is False


# --------------------------------------------------------------------------- #
# Every method, both families
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ENSEMBLE_METHODS)
def test_every_method_fits_and_scores_better_than_nothing(method):
    frame = _frame()
    features = frame[COLUMNS]
    members = _cls_members()

    estimator = build_ensemble(
        method,
        members,
        family="Classification",
        params=_small(method),
        columns=COLUMNS,
        random_state=0,
        n_jobs=1,
    )
    estimator.fit(features, frame["y_cls"])

    assert problem_family_of(estimator) == "Classification"
    accuracy = estimator.score(features, frame["y_cls"])
    # A naive majority baseline on this frame is ~0.5; anything at or below that means
    # the combination is broken, not merely weak (measured range 0.85-0.95).
    assert accuracy > 0.7, accuracy
    # Frame and positional-array predictions must agree for every member-based path.
    assert np.array_equal(
        estimator.predict(features), estimator.predict(features.to_numpy())
    )


@pytest.mark.parametrize("method", ENSEMBLE_METHODS)
def test_every_method_fits_a_regression_task(method):
    frame = _frame()
    features = frame[COLUMNS]

    estimator = build_ensemble(
        method,
        _reg_members(),
        family="Regression",
        params=_small(method),
        columns=COLUMNS,
        random_state=0,
        n_jobs=1,
    )
    estimator.fit(features, frame["y_reg"])

    assert problem_family_of(estimator) == "Regression"
    assert estimator.score(features, frame["y_reg"]) > 0.5
    assert np.array_equal(
        estimator.predict(features), estimator.predict(features.to_numpy())
    )


def test_regression_voting_and_averaging_are_the_same_operator():
    """scikit-learn has no label-voting regressor; that identity is documented, not hidden."""
    voting = build_ensemble("voting", _reg_members(), columns=COLUMNS)
    averaging = build_ensemble("averaging", _reg_members(), columns=COLUMNS)
    assert type(voting) is type(averaging)


def test_ensemble_survives_the_bundle_round_trip(tmp_path):
    frame = _frame()
    features = frame[COLUMNS]
    estimator = build_ensemble("stacking", _cls_members(), params={"cv": 2}, columns=COLUMNS, n_jobs=1)
    estimator.fit(features, frame["y_cls"])
    bundle = ModelBundle(
        estimator,
        task_name="roundtrip",
        schema={column: {"dtype": str(features[column].dtype)} for column in COLUMNS},
    )

    path = save_bundle(bundle, tmp_path / "ensemble.bundle")
    reloaded = load_bundle(path, strict_dependencies=False)

    assert np.array_equal(reloaded.predict(features), estimator.predict(features))
    assert np.array_equal(reloaded.predict(features.to_numpy()), estimator.predict(features))


def test_sample_weight_router_forwards_weights_to_the_last_step():
    """The weights must reach the member's model, not be silently dropped."""
    frame = _frame()
    features = frame[COLUMNS]
    weights = np.zeros(len(frame))
    weights[:20] = 1.0  # only the first 20 rows carry weight

    router = SampleWeightRouter(_member(LogisticRegression(max_iter=500)))
    router.fit(features, frame["y_cls"], sample_weight=weights)
    weighted = router.estimator_.named_steps["model"]

    unweighted = _member(LogisticRegression(max_iter=500)).fit(features, frame["y_cls"])
    assert not np.allclose(weighted.coef_, unweighted.named_steps["model"].coef_)


def test_adaptive_boosting_accepts_the_router_over_a_member_pipeline():
    """The exact gate: AdaBoostClassifier needs ``sample_weight`` in the base's ``fit``.

    ``Pipeline.fit`` is ``fit(self, X, y=None, **params)``, so ``has_fit_parameter``
    answers False for the member itself while its *last step* (the model that would
    receive the weights) does support them — that difference is precisely why
    :class:`SampleWeightRouter` exists.
    """
    from sklearn.utils.validation import has_fit_parameter

    frame = _frame()
    features = frame[COLUMNS]
    base = _member(DecisionTreeClassifier(max_depth=1, random_state=0))

    assert has_fit_parameter(base, "sample_weight") is False
    assert supports_sample_weight(base) is True
    assert supports_sample_weight(base.named_steps["model"]) is True
    with pytest.raises(ValueError, match="doesn't support sample_weight"):
        AdaBoostClassifier(estimator=base, n_estimators=3, random_state=0).fit(
            features, frame["y_cls"]
        )

    booster = AdaBoostClassifier(
        estimator=SampleWeightRouter(_member(DecisionTreeClassifier(max_depth=1, random_state=0))),
        n_estimators=3,
        random_state=0,
    )
    booster.fit(features, frame["y_cls"])
    assert len(booster.estimators_) == 3


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_mixed_family_members_are_refused():
    with pytest.raises(EnsembleError, match="share one problem family"):
        build_ensemble(
            "voting",
            [("cls", _cls_members()[0][1]), ("reg", _reg_members()[0][1])],
            columns=COLUMNS,
        )


def test_a_requested_family_must_match_the_members():
    with pytest.raises(EnsembleError, match="must share one family"):
        build_ensemble("voting", _cls_members(), family="Regression", columns=COLUMNS)


def test_unknown_method_is_refused():
    with pytest.raises(EnsembleError, match="Unknown ensemble method"):
        build_ensemble("blend", _cls_members(), columns=COLUMNS)


def test_aggregating_methods_need_two_members():
    with pytest.raises(EnsembleError, match="at least 2 member"):
        build_ensemble("voting", _cls_members()[:1], columns=COLUMNS)
    # Bagging and boosting *are* the ensembling mechanism, so one member is allowed.
    single = _member(DecisionTreeClassifier(max_depth=3, random_state=0))
    assert build_ensemble("bagging", [("only", single)], params={"n_estimators": 2}, columns=COLUMNS)


def test_duplicate_member_names_are_refused():
    with pytest.raises(EnsembleError, match="unique"):
        build_ensemble(
            "voting", [("same", _cls_members()[0][1]), ("same", _cls_members()[1][1])], columns=COLUMNS
        )


def test_soft_voting_requires_probability_support():
    """LinearSVC has a decision function but no predict_proba."""
    svc = Pipeline([("preprocess", FoldLocalDataPreprocessor()), ("model", LinearSVC())])
    tolerant = frame_tolerant(svc, columns=COLUMNS)
    with pytest.raises(EnsembleError, match="cannot predict them"):
        build_ensemble(
            "averaging", [("svc", tolerant), ("lr", _cls_members()[0][1])], columns=COLUMNS
        )
    # Hard voting does not need probabilities and is therefore accepted.
    assert build_ensemble("voting", [("svc", tolerant), ("lr", _cls_members()[0][1])], columns=COLUMNS)


def test_boosting_refuses_a_classification_member_without_sample_weight():
    """AdaBoostClassifier forwards the round weights, so a member must accept them."""
    knn = _member(KNeighborsClassifier())
    tree = _member(DecisionTreeClassifier(max_depth=3, random_state=0))
    with pytest.raises(EnsembleError, match="does not accept sample_weight"):
        build_ensemble("boosting", [("knn", knn), ("tree", tree)], columns=COLUMNS)


def test_regression_boosting_accepts_a_member_without_sample_weight():
    """AdaBoostRegressor reweights by resampling, so support is not required.

    This asymmetry is the reason the check is scoped to classification: applying it to
    regression would refuse members scikit-learn itself never asks for weights
    (measured in ``AdaBoostRegressor._boost``: ``estimator.fit(X_, y_)``).
    """
    knn = _member(KNeighborsRegressor())
    tree = _member(DecisionTreeRegressor(max_depth=3, random_state=0))
    assert supports_sample_weight(knn) is False

    frame = _frame()
    estimator = build_ensemble(
        "boosting", [("knn", knn), ("tree", tree)], params={"n_estimators": 3}, columns=COLUMNS, n_jobs=1
    )
    estimator.fit(frame[COLUMNS], frame["y_reg"])
    assert estimator.score(frame[COLUMNS], frame["y_reg"]) > 0.3


def test_unknown_parameter_is_refused_rather_than_ignored():
    with pytest.raises(EnsembleError, match="does not take"):
        validate_params("voting", {"n_estimators": 4}, family="Classification", member_count=2)


def test_weight_vector_length_is_checked():
    with pytest.raises(EnsembleError, match="exactly 3 values"):
        validate_params("voting", {"weights": [1.0, 2.0]}, family="Classification", member_count=3)
    with pytest.raises(EnsembleError, match="non-negative"):
        validate_params("voting", {"weights": [1.0, -1.0]}, family="Classification", member_count=2)


def test_stacking_final_estimator_must_belong_to_the_family():
    with pytest.raises(EnsembleError, match="Unknown stacking final estimator"):
        build_ensemble(
            "stacking", _cls_members(), params={"final_estimator": "ridge"}, columns=COLUMNS
        )
    estimator = build_ensemble(
        "stacking",
        _cls_members(),
        params={"final_estimator": "random_forest_classifier", "cv": 2},
        columns=COLUMNS,
    )
    assert getattr(estimator, "final_estimator").__class__.__name__ == "RandomForestClassifier"


def test_resolve_family_rejects_a_non_estimator_member():
    with pytest.raises(EnsembleError, match="neither a scikit-learn classifier"):
        resolve_family([("weird", StandardScaler())])


# --------------------------------------------------------------------------- #
# Scores and the catalogue
# --------------------------------------------------------------------------- #


def test_member_scores_use_the_same_folds_as_the_ensemble():
    frame = _frame()
    features = frame[COLUMNS]
    members = _cls_members()
    ensemble = build_ensemble("voting", members, columns=COLUMNS)

    rows = member_scores(members, features, frame["y_cls"], cv=3, scoring="accuracy", n_jobs=1)
    mean, std, folds = cross_validated_scores(
        ensemble, features, frame["y_cls"], cv=3, scoring="accuracy", n_jobs=1
    )

    assert [row["member"] for row in rows] == ["lr", "dt"]
    assert all(len(row["folds"]) == 3 for row in rows)
    assert len(folds) == 3
    assert mean is not None and 0.0 <= mean <= 1.0
    assert std is not None and std >= 0.0


def test_member_scores_report_a_failure_instead_of_raising():
    """A member that cannot be scored must not sink the comparison table."""
    frame = _frame()
    rows = member_scores(
        [("bad", StandardScaler())], frame[COLUMNS], frame["y_cls"], cv=2, scoring="accuracy", n_jobs=1
    )
    assert rows[0]["member"] == "bad"
    assert "error" in rows[0]
    assert rows[0].get("mean") is None


def test_cross_validated_scores_rejects_an_impossible_scorer():
    frame = _frame()
    with pytest.raises(EnsembleError):
        cross_validated_scores(
            _cls_members()[0][1], frame[COLUMNS], frame["y_cls"], cv=2, scoring="nonsense"
        )


def test_method_catalogue_filters_family_specific_parameters():
    classification = {entry["value"]: entry for entry in method_catalogue("Classification")["methods"]}
    regression = {entry["value"]: entry for entry in method_catalogue("Regression")["methods"]}

    assert set(classification) == set(ENSEMBLE_METHODS)
    assert [spec["name"] for spec in classification["stacking"]["parameters"]] == [
        "final_estimator",
        "cv",
        "passthrough",
    ]
    # Soft voting is classification-only and the update loss regression-only.
    assert any(spec["name"] == "voting" for spec in classification["voting"]["parameters"])
    assert not any(spec["name"] == "voting" for spec in regression["voting"]["parameters"])
    assert any(spec["name"] == "loss" for spec in regression["boosting"]["parameters"])
    assert not any(spec["name"] == "loss" for spec in classification["boosting"]["parameters"])
    assert classification["voting"]["minimum_members"] == 2
    assert classification["bagging"]["minimum_members"] == 1
    assert all(entry["aggregation"] for entry in classification.values())


# --------------------------------------------------------------------------- #
# Portability: the adapters must travel inside a bundle
# --------------------------------------------------------------------------- #


def test_public_surface_is_picklable():
    """The adapter lives in the core package, so a downloaded bundle can be loaded."""
    instance = SampleWeightRouter(LinearRegression())
    assert pickle.loads(pickle.dumps(instance)) is not instance


def test_base_estimator_subclass_contract_holds():
    """``clone`` must round-trip the adapter, or AdaBoost cannot refit it."""
    router = SampleWeightRouter(_member(DecisionTreeClassifier(max_depth=1)))
    cloned = clone(router)
    assert isinstance(cloned, SampleWeightRouter)
    assert not hasattr(cloned, "estimator_")


# --------------------------------------------------------------------------- #
# Per-member calibration
# --------------------------------------------------------------------------- #


def _overconfident_member():
    """A fully grown tree: probabilities are 0/1, so log loss is catastrophic."""
    return _member(DecisionTreeClassifier(max_depth=12, min_samples_leaf=1, random_state=0))


def test_calibration_repairs_an_overconfident_member():
    """The evidence the feature exists for: probabilities, not decisions, get repaired."""
    rng = np.random.default_rng(7)
    n = 400
    frame = pd.DataFrame(
        {"x1": rng.normal(size=n), "x2": rng.normal(size=n), "team": rng.choice(["a", "b"], size=n)}
    )
    frame["y"] = (
        frame["x1"] * 1.5 + (frame["team"] == "a") + rng.normal(scale=0.8, size=n) > 0
    ).astype(int)
    features = frame[COLUMNS[:2] + ["team"]]
    members = [("tree", _overconfident_member())]

    report = calibration_report(members, features, frame["y"], cv=5, method="sigmoid")
    entry = report[0]

    assert entry["log_loss_before"] > 1.0  # an overconfident member is heavily penalised
    assert entry["log_loss_after"] < entry["log_loss_before"] / 2
    assert entry["brier_after"] < entry["brier_before"]
    # Calibration does not have to change a single hard decision, and it is not asked to.
    assert abs(entry["brier_after"] - entry["brier_before"]) < 0.2


def test_calibration_leaves_a_well_calibrated_member_about_unchanged():
    frame = _frame()
    report = calibration_report(
        [("lr", _member(LogisticRegression(max_iter=500)))],
        frame[COLUMNS],
        frame["y_cls"],
        cv=5,
        method="sigmoid",
    )
    entry = report[0]
    assert abs(entry["brier_after"] - entry["brier_before"]) < 0.02
    assert abs(entry["log_loss_after"] - entry["log_loss_before"]) < 0.05


def test_calibration_reports_a_member_failure_instead_of_raising():
    """A member that cannot be calibrated is reported, not fatal."""
    frame = _frame()
    report = calibration_report(
        [("bare", _ScoresLessClassifier())], frame[COLUMNS], frame["y_cls"], cv=2
    )
    assert "error" in report[0]


def test_calibration_report_explains_a_missing_probability_metric():
    """A decision-function member is calibratable, but Brier/log loss are not available.

    The report must say that, rather than surfacing sklearn's own ``predict_proba``
    AttributeError for a member whose calibration worked.
    """
    frame = _frame()
    report = calibration_report(
        [("svc", _member(LinearSVC()))], frame[COLUMNS], frame["y_cls"], cv=2
    )
    entry = report[0]
    assert "predict_proba" in entry["error"]
    assert "decision_function only" in entry["error"]
    # No half-computed metric is reported alongside it.
    assert "brier_after" not in entry


def test_a_degenerate_weight_search_reports_no_objective_rather_than_infinity(monkeypatch):
    """±inf (or NaN) in the job result would be invalid JSON, so the objective is dropped.

    The reproducible case is a scorer that cannot score any candidate — scikit-learn's own
    scorers return NaN on a degenerate fold — which leaves the uniform vector in place and no
    usable objective.
    """
    import json

    import sklearn.metrics

    frame = _weighting_frame()

    class _Unscorable:
        """A scorer that never produces a usable value, as a degenerate fold's would not."""

        def __call__(self, estimator, X, y):  # noqa: ANN001, ARG002 - test double
            return float("nan")

    # ``optimize_weights`` resolves ``get_scorer`` from this module at call time.
    monkeypatch.setattr(sklearn.metrics, "get_scorer", lambda name: _Unscorable())

    for strategy, extra in (("greedy", {"rounds": 4}), ("dirichlet", {"samples": 10})):
        result = optimize_weights(
            _cls_members(), frame[COLUMNS], frame["y_cls"], cv=4, strategy=strategy,
            scoring="accuracy", **extra,
        )
        assert result["objective"] is None
        # The uniform vector is the honest answer when nothing could be scored.
        assert abs(sum(result["weights"]) - 1.0) < 1e-6
        assert result["dropped"] == []
        # And the payload a job would carry is valid JSON, not `-Infinity`/`NaN`.
        assert json.loads(json.dumps(result, allow_nan=False))["objective"] is None


class _ScoresLessClassifier(ClassifierMixin, BaseEstimator):
    """A classifier with neither probabilities nor a decision function.

    Exists so the calibration eligibility check can be pinned: such a member is refused by
    name rather than failing somewhere inside ``CalibratedClassifierCV``.
    """

    def fit(self, X, y=None):
        self.classes_ = np.unique(np.asarray(y))
        self.majority_ = self.classes_[0]
        return self

    def predict(self, X):
        return np.full(len(np.asarray(X)), self.majority_)


def test_calibration_members_refuses_regression_and_unscorable_members():
    with pytest.raises(EnsembleError, match="classification members only"):
        calibrate_members(_reg_members(), method="sigmoid", cv=3)
    # A classifier that exposes neither predict_proba nor decision_function has nothing
    # to calibrate; the refusal names it instead of failing inside the fit.
    with pytest.raises(EnsembleError, match="neither predict_proba nor decision_function"):
        calibrate_members([("bare", _ScoresLessClassifier())], method="sigmoid", cv=3)
    # A classifier *with* a decision function is calibratable even without probabilities.
    svc = _member(LinearSVC())
    assert calibrate_members([("svc", svc)], method="sigmoid", cv=3)


def test_calibration_method_and_folds_are_validated():
    from SKSurrogate.ensembling import _installed_calibration_ensemble_default

    member = _member(LogisticRegression(max_iter=200))
    with pytest.raises(EnsembleError, match="Unknown calibration method"):
        calibrate_member(member, method="platt")
    with pytest.raises(EnsembleError, match="at least 2"):
        calibrate_member(member, cv=1)
    with pytest.raises(EnsembleError, match="must be one of"):
        calibrate_member(member, ensemble="sometimes")
    # The three tokens resolve to scikit-learn's own values, and "auto" resolves against
    # the *installed* build: scikit-learn 1.5 accepts only a boolean here, so hard-coding
    # the literal "auto" would make calibration unusable at the declared dependency floor.
    assert calibrate_member(member, ensemble="per_fold").ensemble is True
    assert calibrate_member(member, ensemble="single").ensemble is False
    assert calibrate_member(member, ensemble=False).ensemble is False
    assert calibrate_member(member, ensemble="auto").ensemble == _installed_calibration_ensemble_default()
    # And whatever it resolved to is a value this build actually accepts.
    frame = _frame()
    calibrated = calibrate_member(member, ensemble="auto")
    calibrated.fit(frame[COLUMNS], frame["y_cls"])
    assert calibrated.predict_proba(frame[COLUMNS].head(3)).shape == (3, 2)


def test_auto_calibration_ensemble_survives_a_boolean_only_scikit_learn(monkeypatch):
    """A boolean-only build (scikit-learn 1.5) must still be served.

    The resolution reads the installed signature instead of comparing versions, so pinning
    the signature to what 1.5 declares is enough to exercise that path.
    """
    import inspect as _inspect

    from sklearn.calibration import CalibratedClassifierCV as _CalibratedClassifierCV

    from SKSurrogate import ensembling
    from SKSurrogate.ensembling import _installed_calibration_ensemble_default

    real_signature = _inspect.signature

    def fake_signature(callable_):  # noqa: ANN001 - test double
        signature = real_signature(callable_)
        if callable_ is _CalibratedClassifierCV.__init__:
            parameters = [
                parameter.replace(default=True) if parameter.name == "ensemble" else parameter
                for parameter in signature.parameters.values()
            ]
            return signature.replace(parameters=parameters)
        return signature

    monkeypatch.setattr(ensembling.inspect, "signature", fake_signature)
    assert _installed_calibration_ensemble_default() is True
    assert ensembling._resolve_calibration_ensemble("auto") is True
    # The other tokens are unaffected: they were booleans before and after.
    assert ensembling._resolve_calibration_ensemble("per_fold") is True
    assert ensembling._resolve_calibration_ensemble("single") is False
    monkeypatch.undo()
    assert ensembling._resolve_calibration_ensemble("auto") == _installed_calibration_ensemble_default()


def test_calibration_is_wrapped_outside_the_resampling_wrapper():
    """The nesting that keeps the calibration honest (and sklearn quiet).

    Inside, ``BaggingClassifier`` forwards the bootstrap weights to the calibrator and
    scikit-learn warns that the calibration is "likely to be incorrect"; outside, the
    calibrator sees the fold's own rows. This pins the structure, and
    tests/test_ensemble_endpoints.py pins the absence of the warning end to end.
    """
    from sklearn.calibration import CalibratedClassifierCV

    frame = _frame()
    features = frame[COLUMNS]
    members = _cls_members()

    for method in ("bagging", "boosting"):
        forms, _family, _cleaned = member_forms(
            method,
            members,
            params={"n_estimators": 3},
            calibration={"method": "sigmoid", "cv": 3},
            columns=COLUMNS,
            random_state=0,
        )
        for _name, form in forms:
            # Outermost is the calibrator, and what it calibrates is the wrapper.
            assert isinstance(form, CalibratedClassifierCV), method
            inner = form.estimator
            assert type(inner).__name__ in ("BaggingClassifier", "AdaBoostClassifier"), method

    # The aggregating methods calibrate the member itself instead.
    forms, _family, _cleaned = member_forms(
        "voting", members, calibration={"method": "sigmoid", "cv": 3}, columns=COLUMNS
    )
    assert all(isinstance(form, CalibratedClassifierCV) for _name, form in forms)


def test_already_calibrated_members_are_not_double_wrapped():
    frame = _frame()
    features = frame[COLUMNS]
    members = [(name, calibrate_member(est, cv=3)) for name, est in _cls_members()]

    estimator = build_ensemble("averaging", members, columns=COLUMNS, n_jobs=1)
    estimator.fit(features, frame["y_cls"])

    # build_ensemble leaves an already-calibrated member alone rather than nesting again.
    from sklearn.calibration import CalibratedClassifierCV

    assert all(
        isinstance(est, CalibratedClassifierCV) for _name, est in estimator.estimators
    )


# --------------------------------------------------------------------------- #
# Weight optimization
# --------------------------------------------------------------------------- #


def _weighting_frame(n=500, seed=3):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {"x1": rng.normal(size=n), "x2": rng.normal(size=n), "team": rng.choice(["a", "b", "c"], size=n)}
    )
    frame["y_cls"] = (
        frame["x1"] * 1.6
        + 0.9 * (frame["team"] == "a")
        + 0.7 * frame["x1"] * frame["x2"]
        + rng.normal(scale=1.0, size=n)
        > 0
    ).astype(int)
    frame["y_reg"] = 2 * frame["x1"] - frame["x2"] + (frame["team"] == "b") + rng.normal(scale=0.4, size=n)
    return frame


def test_out_of_fold_outputs_are_not_in_sample():
    """Each member's stored output must come from folds it was not trained on."""
    frame = _weighting_frame()
    features = frame[COLUMNS]
    member = _member(LogisticRegression(max_iter=500))
    outputs = out_of_fold_outputs([("lr", member)], features, frame["y_cls"], 4, None)

    oof_prediction = (outputs[0][:, 1] > 0.5).astype(int)
    oof_accuracy = float((oof_prediction == frame["y_cls"]).mean())
    in_sample = float(clone(member).fit(features, frame["y_cls"]).score(features, frame["y_cls"]))
    assert oof_accuracy < in_sample


def test_least_squares_weights_live_on_the_simplex_and_beat_equal_weights():
    frame = _weighting_frame()
    features = frame[COLUMNS]
    members = _cls_members()
    result = optimize_weights(members, features, frame["y_cls"], cv=4, strategy="least_squares")

    weights = np.asarray(result["weights"])
    assert np.all(weights >= 0.0)
    assert abs(float(weights.sum()) - 1.0) < 1e-6
    assert result["objective_name"].startswith("mean squared error")
    assert result["budget"] is None
    # `dropped` must name exactly the zero entries, positionally.
    assert set(result["dropped"]) == {
        members[index][0] for index in range(len(members)) if weights[index] <= 1e-12
    }
    # The fit is not a no-op: it must do at least as well as equal weights on its own
    # objective (that is the whole point of solving it).
    outputs = out_of_fold_outputs(members, features, frame["y_cls"], 4, None)
    equal = np.full(len(members), 1.0 / len(members))
    onehot = np.zeros((len(frame), 2))
    onehot[np.arange(len(frame)), frame["y_cls"].to_numpy()] = 1.0
    equal_objective = float(np.mean((sum(w * o for w, o in zip(equal, outputs)) - onehot) ** 2))
    assert result["objective"] <= equal_objective


def test_scorer_strategies_use_the_task_metric_and_report_their_budget():
    frame = _weighting_frame()
    features = frame[COLUMNS]
    members = _cls_members()

    for strategy, extra, key in (
        ("dirichlet", {"samples": 40}, "samples"),
        ("greedy", {"rounds": 6}, "rounds"),
    ):
        result = optimize_weights(
            members, features, frame["y_cls"], cv=4, strategy=strategy, scoring="accuracy", **extra
        )
        assert result["strategy"] == strategy
        assert result["budget"][key] == extra[key]
        assert "accuracy" in result["objective_name"]
        assert 0.0 < float(np.asarray(result["weights"]).sum()) <= 1.0 + 1e-9
    # A scorer-based search is meaningless without the metric.
    with pytest.raises(EnsembleError, match="scoring metric is required"):
        optimize_weights(members, features, frame["y_cls"], cv=4, strategy="dirichlet")


def test_optimize_weights_validates_strategy_and_budget():
    with pytest.raises(EnsembleError, match="Unknown weight strategy"):
        validate_weight_optimization({"strategy": "bayesian"})
    with pytest.raises(EnsembleError, match="weight_optimization.samples"):
        validate_weight_optimization({"strategy": "dirichlet", "samples": 3})
    with pytest.raises(EnsembleError, match="weight_optimization.rounds"):
        validate_weight_optimization({"strategy": "greedy", "rounds": 0})
    cleaned = validate_weight_optimization({})
    assert cleaned == {"strategy": "least_squares", "samples": 200, "rounds": 20}


def test_weights_are_optimized_for_regression_too():
    frame = _weighting_frame()
    features = frame[COLUMNS]
    result = optimize_weights(
        _reg_members(), features, frame["y_reg"], cv=4, strategy="least_squares"
    )
    assert abs(sum(result["weights"]) - 1.0) < 1e-6
    assert result["objective"] < 1.0  # an MSE well below the target's own variance


def test_optimized_weights_reach_the_bagging_and_boosting_aggregation():
    """Weights apply wherever the merge is a weighted vote, not only to voting/averaging."""
    frame = _weighting_frame()
    features = frame[COLUMNS]

    for method in ("bagging", "boosting"):
        estimator = build_ensemble(
            method,
            _cls_members(),
            params={"n_estimators": 3, "weights": [0.0, 1.0]},
            columns=COLUMNS,
            random_state=0,
            n_jobs=1,
        )
        estimator.fit(features, frame["y_cls"])
        assert list(estimator.weights) == [0.0, 1.0]
        # The dropped member's estimator is still fitted, but contributes nothing.
        assert len(estimator.estimators_) == 2


def test_fit_ensemble_records_what_it_did():
    frame = _weighting_frame()
    features = frame[COLUMNS]
    estimator, provenance = fit_ensemble(
        "voting",
        _cls_members(),
        features,
        frame["y_cls"],
        params={"voting": "soft"},
        calibration={"method": "sigmoid", "cv": 3},
        weight_optimization={"strategy": "least_squares"},
        columns=COLUMNS,
        cv=4,
        random_state=0,
        n_jobs=1,
    )
    assert provenance["calibration"]["method"] == "sigmoid"
    assert provenance["weights"]["strategy"] == "least_squares"
    assert provenance["parameters"]["weights"] == provenance["weights"]["weights"]
    # The forms reported are the ones that were combined (calibrated, wrappers included).
    from sklearn.calibration import CalibratedClassifierCV

    assert all(isinstance(form, CalibratedClassifierCV) for _name, form in provenance["forms"])
    assert estimator.score(features, frame["y_cls"]) > 0.7


def test_fit_ensemble_refuses_conflicting_and_impossible_combinations():
    frame = _weighting_frame()
    features = frame[COLUMNS]
    with pytest.raises(EnsembleError, match="not both"):
        fit_ensemble(
            "voting",
            _cls_members(),
            features,
            frame["y_cls"],
            params={"weights": [1.0, 1.0]},
            weight_optimization={"strategy": "least_squares"},
            columns=COLUMNS,
            n_jobs=1,
        )
    with pytest.raises(EnsembleError, match="Stacking learns its own combination"):
        fit_ensemble(
            "stacking",
            _cls_members(),
            features,
            frame["y_cls"],
            weight_optimization={"strategy": "least_squares"},
            columns=COLUMNS,
            n_jobs=1,
        )
    with pytest.raises(EnsembleError, match="Unknown weight strategy"):
        fit_ensemble(
            "voting",
            _cls_members(),
            features,
            frame["y_cls"],
            weight_optimization={"strategy": "bayesian"},
            columns=COLUMNS,
            n_jobs=1,
        )
    with pytest.raises(EnsembleError, match="Unknown calibration method"):
        fit_ensemble(
            "voting",
            _cls_members(),
            features,
            frame["y_cls"],
            calibration={"method": "platt"},
            columns=COLUMNS,
            n_jobs=1,
        )


def test_nested_ensemble_cv_prices_the_weight_search():
    """The honest procedure must be available next to the optimistic one.

    ``nested_ensemble_cv`` refits the weights inside every outer training fold, so its
    number is not the one the weight search was fitted against. On this fixture the
    in-sample-optimized score is the higher of the two — that gap *is* the finding, so
    the assertion is deliberately one-sided rather than an equality.
    """
    frame = _weighting_frame()
    features = frame[COLUMNS]
    members = _cls_members()

    nested = nested_ensemble_cv(
        "voting",
        members,
        features,
        frame["y_cls"],
        params={"voting": "soft"},
        weight_optimization={"strategy": "least_squares"},
        columns=COLUMNS,
        outer_cv=4,
        inner_cv=3,
        scoring="accuracy",
        random_state=0,
        n_jobs=1,
    )
    assert nested["mean"] is not None and nested["std"] is not None
    assert len(nested["folds"]) == 4
    assert nested["errors"] == []
    assert nested["scoring"] == "accuracy"
    # Each outer fold fitted its own weight vector, and each is a distribution.
    assert len(nested["per_fold_weights"]) == 4
    for entry in nested["per_fold_weights"]:
        assert abs(sum(entry["weights"]) - 1.0) < 1e-6
        assert all(weight >= 0.0 for weight in entry["weights"])

    # The optimistic counterpart: weights fitted on the folds that score them.
    estimator, _provenance = fit_ensemble(
        "voting",
        members,
        features,
        frame["y_cls"],
        params={"voting": "soft"},
        weight_optimization={"strategy": "least_squares"},
        columns=COLUMNS,
        cv=4,
        scoring="accuracy",
        random_state=0,
        n_jobs=1,
    )
    from SKSurrogate import cross_validated_scores

    optimistic, _std, _folds = cross_validated_scores(
        estimator, features, frame["y_cls"], cv=4, scoring="accuracy", n_jobs=1
    )
    assert nested["mean"] <= optimistic + 0.02


def test_nested_ensemble_cv_without_weight_optimization_scores_the_plain_ensemble():
    frame = _weighting_frame()
    nested = nested_ensemble_cv(
        "averaging",
        _cls_members(),
        frame[COLUMNS],
        frame["y_cls"],
        columns=COLUMNS,
        outer_cv=3,
        inner_cv=2,
        scoring="accuracy",
        random_state=0,
        n_jobs=1,
    )
    assert nested["mean"] is not None
    assert len(nested["folds"]) == 3


def test_nested_ensemble_cv_fails_loudly_on_a_bad_request():
    """A request-level mistake must raise, not become one error per fold."""
    frame = _weighting_frame()
    with pytest.raises(EnsembleError, match="does not take"):
        nested_ensemble_cv(
            "averaging",
            _cls_members(),
            frame[COLUMNS],
            frame["y_cls"],
            params={"voting": "soft"},  # averaging has no `voting` parameter
            columns=COLUMNS,
            outer_cv=3,
            inner_cv=2,
            scoring="accuracy",
            n_jobs=1,
        )


def test_a_fold_count_the_data_cannot_support_is_a_clear_error():
    """Folds the partition cannot support must not become one error per fold."""
    frame = _weighting_frame(n=60)
    with pytest.raises(EnsembleError, match="Could not build"):
        nested_ensemble_cv(
            "voting",
            _cls_members(),
            frame[COLUMNS],
            frame["y_cls"],
            params={"voting": "soft"},
            columns=COLUMNS,
            outer_cv=40,  # far more folds than the smallest class can fill
            inner_cv=2,
            scoring="accuracy",
            n_jobs=1,
        )


def test_nested_ensemble_cv_reports_a_fold_failure_instead_of_raising():
    """One degenerate split must not lose the estimate."""
    frame = _weighting_frame(n=60)
    nested = nested_ensemble_cv(
        "voting",
        _cls_members(),
        frame[COLUMNS],
        frame["y_cls"],
        params={"voting": "soft"},
        columns=COLUMNS,
        outer_cv=2,
        inner_cv=6,  # more inner folds than the smallest class can support
        scoring="accuracy",
        random_state=0,
        n_jobs=1,
    )
    assert nested["errors"] or nested["mean"] is not None
