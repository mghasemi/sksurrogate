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
    cross_validated_scores,
    member_scores,
    method_catalogue,
    problem_family_of,
    resolve_family,
    supports_sample_weight,
    validate_params,
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

    assert [str(dtype) for dtype in rebuilt.dtypes] == ["float64", "float64", "object"]
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
    from sklearn.base import clone

    router = SampleWeightRouter(_member(DecisionTreeClassifier(max_depth=1)))
    cloned = clone(router)
    assert isinstance(cloned, SampleWeightRouter)
    assert not hasattr(cloned, "estimator_")
