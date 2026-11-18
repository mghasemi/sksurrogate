"""Combine several trained SKSurrogate models into a single scikit-learn ensemble.

Every model this project stores is a fitted estimator (usually a pipeline, see
:mod:`SKSurrogate.modelbundle`). This module turns a *selection* of such models into
one scikit-learn meta-estimator, so the combination is itself an ordinary estimator
and can be wrapped into the same :class:`~SKSurrogate.modelbundle.ModelBundle` as any
other model — and therefore evaluated, registered and served by machinery that knows
nothing about ensembles.

Methods and what they mean here
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 12 44 44

   * - method
     - classification
     - regression
   * - ``voting``
     - ``VotingClassifier``; majority label vote unless soft voting is requested
     - ``VotingRegressor``; unweighted mean of predictions
   * - ``averaging``
     - ``VotingClassifier(voting="soft")``; mean of the members' class probabilities
     - ``VotingRegressor``; unweighted mean of predictions
   * - ``stacking``
     - ``StackingClassifier`` with a final estimator refitted on out-of-fold member
       predictions
     - ``StackingRegressor`` with a final estimator refitted on out-of-fold member
       predictions
   * - ``bagging``
     - one ``BaggingClassifier`` per member, then soft voting over the bagged members
     - one ``BaggingRegressor`` per member, then averaging
   * - ``boosting``
     - one ``AdaBoostClassifier`` per member, then soft voting over the boosted members
     - one ``AdaBoostRegressor`` per member, then averaging

Three properties of scikit-learn are worth stating, because they decide the design
and are not obvious from the class names:

1. ``VotingClassifier``/``VotingRegressor``/``Stacking*`` do consume a *heterogeneous*
   set of member pipelines — but only when they are handed a ``DataFrame``. Every
   array-input path (learning curves and legacy trackers predict from positional
   arrays) therefore relies on the members being array-safe, which is what
   :func:`SKSurrogate.DataProcess.frame_tolerant` provides. :func:`build_ensemble`
   normalises each member through it.
2. scikit-learn's bagging and boosting replicate **one** base estimator; they do not
   combine a set. To combine a set, this module gives every member its own bagging
   (or boosting) wrapper and then aggregates the wrappers — so each selected model
   genuinely contributes. Bagging indexes ``X`` positionally before fitting the base
   (``X[indices]``), which is why a member whose encoder selects columns *by name*
   must be array-safe: without ``ArrayAsFrame`` it fails with
   ``Specifying the columns using strings is only supported for dataframes``.
3. ``AdaBoost*`` will not take a pipeline as its base estimator, and the two AdaBoost
   variants differ in *how*:

   * ``AdaBoostClassifier`` gates on
     ``sklearn.utils.validation.has_fit_parameter(estimator_, "sample_weight")``
     (``sklearn/ensemble/_weight_boosting.py``) and then passes the round weights to
     ``fit``. ``Pipeline.fit`` has no such named parameter, and a ``**fit_params``-style
     adapter does not satisfy the check either, so :class:`SampleWeightRouter` names the
     parameter explicitly and forwards it to the pipeline's last step
     (``<step>__sample_weight``). A classification member whose final estimator cannot
     take weights is refused rather than boosted into identical rounds.
   * ``AdaBoostRegressor`` does neither: ``_boost`` reweights by *resampling*
     (``random_state.choice(..., p=sample_weight)``) and calls ``estimator.fit(X_, y_)``
     with no weights at all. Regression members therefore need no adapter and are not
     refused for lacking sample-weight support.

Regression members may also be combined with ``voting``; for regression that is the
same operator as ``averaging`` (scikit-learn has no label-voting regressor) and is
documented rather than rejected.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, is_classifier, is_regressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import has_fit_parameter

from .DataProcess import frame_tolerant

CLASSIFICATION = "Classification"
REGRESSION = "Regression"

#: The combiners this module can build, in the order the UI should offer them.
ENSEMBLE_METHODS = ("voting", "averaging", "stacking", "bagging", "boosting")

#: Minimum member count per method. Bagging and boosting *are* the ensembling
#: mechanism (each member gets its own resampled/boosted wrapper), so a single
#: member is a legitimate — if degenerate — request; the aggregating methods need
#: at least two members or they are a relabelling of the member itself.
MINIMUM_MEMBERS = {
    "voting": 2,
    "averaging": 2,
    "stacking": 2,
    "bagging": 1,
    "boosting": 1,
}

#: Final estimators offered for stacking. The meta-learner sees member predictions
#: (numeric columns), so plain estimators are used here — never the frame-tolerant
#: member pipelines.
STACKING_FINAL_ESTIMATORS = {
    CLASSIFICATION: {
        "logistic_regression": LogisticRegression,
        "random_forest_classifier": RandomForestClassifier,
        "decision_tree_classifier": DecisionTreeClassifier,
        "gradient_boosting_classifier": GradientBoostingClassifier,
    },
    REGRESSION: {
        "linear_regression": LinearRegression,
        "ridge": Ridge,
        "random_forest_regressor": RandomForestRegressor,
        "decision_tree_regressor": DecisionTreeRegressor,
    },
}
DEFAULT_STACKING_FINAL_ESTIMATOR = {
    CLASSIFICATION: "logistic_regression",
    REGRESSION: "linear_regression",
}

#: Parameter schema per method, consumed by the control plane's ``/methods``
#: endpoint and validated by :func:`validate_params` so the UI and the API agree.
METHOD_PARAMETERS = {
    "voting": (
        {"name": "weights", "type": "floats", "default": None, "label": "Weights",
         "help": "One non-negative weight per member, in the order given; omitted means equal weights."},
        {"name": "voting", "type": "choice", "choices": ["hard", "soft"], "default": "hard",
         "label": "Vote", "family": CLASSIFICATION,
         "help": "Hard counts labels, soft averages class probabilities (requires every member to predict probabilities)."},
    ),
    "averaging": (
        {"name": "weights", "type": "floats", "default": None, "label": "Weights",
         "help": "One non-negative weight per member, in the order given; omitted means equal weights."},
    ),
    "stacking": (
        {"name": "final_estimator", "type": "string", "default": None, "label": "Final estimator",
         "help": "Meta-learner fitted on out-of-fold member predictions; a family-appropriate default is used when omitted."},
        {"name": "cv", "type": "integer", "default": 5, "min": 2, "max": 20, "label": "Folds",
         "help": "Inner folds used to build the meta-features (out-of-fold member predictions)."},
        {"name": "passthrough", "type": "boolean", "default": False, "label": "Pass through features",
         "help": "Also hand the original feature matrix to the final estimator."},
    ),
    "bagging": (
        {"name": "n_estimators", "type": "integer", "default": 10, "min": 1, "max": 500, "label": "Resamples",
         "help": "Bootstrap replicas fitted per member."},
        {"name": "max_samples", "type": "float", "default": None, "min": 0.01, "max": 1.0, "label": "Max samples",
         "help": "Fraction of rows per replica; omitted means all rows with replacement."},
        {"name": "max_features", "type": "float", "default": None, "min": 0.01, "max": 1.0, "label": "Max features",
         "help": "Fraction of features considered per replica; omitted means all features."},
        {"name": "bootstrap", "type": "boolean", "default": True, "label": "Bootstrap",
         "help": "Sample rows with replacement; without it bagging degenerates to member averaging."},
    ),
    "boosting": (
        {"name": "n_estimators", "type": "integer", "default": 20, "min": 1, "max": 500, "label": "Rounds",
         "help": "Boosting rounds per member. Every member's final estimator must accept sample weights."},
        {"name": "learning_rate", "type": "float", "default": 1.0, "min": 0.0001, "max": 10.0, "label": "Learning rate",
         "help": "Shrinks each round's contribution."},
        {"name": "loss", "type": "choice", "choices": ["linear", "square", "exponential"], "default": "linear",
         "label": "Loss", "family": REGRESSION,
         "help": "AdaBoost.R2 loss updated after each round (regression only)."},
    ),
}


class EnsembleError(ValueError):
    """A member set or parameter combination that cannot be combined as requested."""


def _parameters_for(method, family):
    return tuple(
        spec
        for spec in METHOD_PARAMETERS.get(method, ())
        if spec.get("family") in (None, family)
    )


def method_catalogue(family):
    """Describe the combiners available for a problem family, for API/UI consumption.

    Returns one entry per method with its label, a prose description, the minimum
    member count and the parameter schema that applies to ``family``. Methods that
    only work for the other family (soft voting, the regression loss) are filtered
    out of the parameter list rather than hidden entirely.
    """
    descriptions = {
        "voting": "Each member votes; the majority label (or the weighted mean prediction) wins.",
        "averaging": "Mean of the members' class probabilities (classification) or predictions (regression).",
        "stacking": "A meta-learner is fitted on out-of-fold member predictions.",
        "bagging": "Each member is bagged over bootstrap replicas; the bagged members are then aggregated.",
        "boosting": "Each member is boosted round by round with reweighted samples; the boosted members are aggregated.",
    }
    entries = []
    for method in ENSEMBLE_METHODS:
        parameters = list(_parameters_for(method, family))
        if method == "stacking" and family in STACKING_FINAL_ESTIMATORS:
            # A free-text parameter the caller can only get wrong: publish the
            # family's accepted names (and its default) so a UI renders a dropdown.
            default = DEFAULT_STACKING_FINAL_ESTIMATOR[family]
            parameters = [
                {**spec, "choices": sorted(STACKING_FINAL_ESTIMATORS[family]), "default": default}
                if spec["name"] == "final_estimator"
                else spec
                for spec in parameters
            ]
        applies = True
        if method == "stacking":
            applies = family in STACKING_FINAL_ESTIMATORS
        entries.append(
            {
                "value": method,
                "label": method.capitalize(),
                "description": descriptions[method],
                "minimum_members": MINIMUM_MEMBERS[method],
                "parameters": [dict(spec) for spec in parameters],
                "aggregation": aggregation_label(method, family),
                "available": applies,
            }
        )
    return {"family": family, "methods": entries}


def aggregation_label(method, family):
    """Plain-language summary of how member outputs are merged, for the UI."""
    if family == REGRESSION:
        if method == "stacking":
            return "final estimator over out-of-fold member predictions"
        return "unweighted mean of member predictions"
    if method in ("voting",):
        return "majority vote (hard) unless soft voting is selected"
    return "mean of member class probabilities"


def problem_family_of(estimator):
    """``"Classification"`` / ``"Regression"`` for a fitted estimator, else ``None``."""
    if is_classifier(estimator):
        return CLASSIFICATION
    if is_regressor(estimator):
        return REGRESSION
    return None


def resolve_family(members):
    """Assert that ``members`` share one problem family and return it.

    ``members`` is a sequence of ``(name, estimator)`` pairs. Mixing a classifier
    with a regressor is refused rather than silently coerced, because every combiner
    below either averages probabilities or averages numbers — not both.
    """
    families = {}
    for name, estimator in members:
        family = problem_family_of(estimator)
        if family is None:
            raise EnsembleError(
                "Member %r is neither a scikit-learn classifier nor a regressor "
                "(%s), so it cannot be combined." % (name, type(estimator).__name__)
            )
        families.setdefault(family, []).append(name)
    if len(families) > 1:
        detail = "; ".join(
            "%s: %s" % (family, ", ".join(names)) for family, names in sorted(families.items())
        )
        raise EnsembleError("All members must share one problem family (%s)." % detail)
    return next(iter(families))


def _final_estimator_of(estimator):
    """The estimator that actually consumes ``X`` (a pipeline's last step)."""
    steps = getattr(estimator, "steps", None)
    if steps:
        return steps[-1][1]
    return estimator


def supports_sample_weight(estimator):
    """Whether ``estimator`` can be fitted with sample weights.

    For a pipeline this is the *last* step's capability, because that is where
    :class:`SampleWeightRouter` forwards the weights.
    """
    target = _final_estimator_of(estimator)
    return bool(has_fit_parameter(target, "sample_weight"))


def _fit_routed(estimator, X, y, sample_weight):
    """Fit a fresh clone of ``estimator``, routing ``sample_weight`` to the last step.

    ``Pipeline.fit`` only forwards a *step-prefixed* fit parameter
    (``<step>__sample_weight``) to that step, so routing is the whole point of the
    adapters below; a pipeline whose last step cannot take weights raises here, which
    is why :func:`supports_sample_weight` is checked before building a booster.
    """
    fitted: Any = clone(estimator)
    if sample_weight is None:
        fitted.fit(X, y)
        return fitted
    steps = getattr(fitted, "steps", None)
    if steps:
        fitted.fit(X, y, **{steps[-1][0] + "__sample_weight": sample_weight})
    else:
        fitted.fit(X, y, sample_weight=sample_weight)
    return fitted


class SampleWeightRouter(BaseEstimator, ClassifierMixin):
    """Fit a member (pipeline or bare estimator) with an explicitly named ``sample_weight``.

    ``AdaBoostClassifier``/``AdaBoostRegressor`` only accept a base estimator whose
    ``fit`` exposes a parameter literally named ``sample_weight``
    (``has_fit_parameter``, see ``sklearn/ensemble/_weight_boosting.py``). A
    ``Pipeline`` never does, and an adapter that swallows the weights through
    ``**fit_params`` does not satisfy the check either — so this class declares the
    parameter and forwards it to the pipeline's last step as
    ``<step>__sample_weight``, which is the only form ``Pipeline.fit`` routes
    onwards. A bare estimator is fitted directly with the weights.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y=None, sample_weight=None):
        fitted = _fit_routed(self.estimator, X, y, sample_weight)
        self.estimator_ = fitted
        self.classes_ = getattr(fitted, "classes_", None)
        self.n_features_in_ = getattr(fitted, "n_features_in_", None)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def decision_function(self, X):
        return self.estimator_.decision_function(X)


def validate_params(method, params, *, family, member_count):
    """Validate and normalise a method's parameters; return the cleaned mapping.

    Every rejection names the offending value so a caller can act on it without
    reading this module. Unknown parameters are refused instead of ignored, so a
    typo (``n_estimators`` on ``voting``) never silently produces a different
    ensemble than the one requested.
    """
    if method not in ENSEMBLE_METHODS:
        raise EnsembleError(
            "Unknown ensemble method %r; choose one of %s." % (method, ", ".join(ENSEMBLE_METHODS))
        )
    if member_count < MINIMUM_MEMBERS[method]:
        raise EnsembleError(
            "Method %r needs at least %d member(s); got %d."
            % (method, MINIMUM_MEMBERS[method], member_count)
        )
    allowed = {spec["name"]: spec for spec in _parameters_for(method, family)}
    unknown = sorted(set(params or {}) - set(allowed))
    if unknown:
        raise EnsembleError(
            "Method %r does not take %s (accepted: %s)."
            % (method, ", ".join(repr(name) for name in unknown), ", ".join(sorted(allowed)) or "no parameters")
        )

    cleaned = {name: spec["default"] for name, spec in allowed.items()}
    for name, value in (params or {}).items():
        spec = allowed[name]
        if value is None:
            cleaned[name] = spec["default"]
            continue
        kind = spec["type"]
        if kind == "boolean":
            if not isinstance(value, bool):
                raise EnsembleError("Parameter %r must be true or false." % name)
        elif kind == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise EnsembleError("Parameter %r must be an integer." % name)
            if value < spec["min"] or value > spec["max"]:
                raise EnsembleError(
                    "Parameter %r must be between %s and %s." % (name, spec["min"], spec["max"])
                )
        elif kind == "float":
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise EnsembleError("Parameter %r must be a number." % name) from None
            if not math.isfinite(value) or value < spec["min"] or value > spec["max"]:
                raise EnsembleError(
                    "Parameter %r must be a finite number between %s and %s."
                    % (name, spec["min"], spec["max"])
                )
        elif kind == "choice":
            if value not in spec["choices"]:
                raise EnsembleError(
                    "Parameter %r must be one of %s." % (name, ", ".join(repr(item) for item in spec["choices"]))
                )
        elif kind == "floats":
            if not isinstance(value, (list, tuple)):
                raise EnsembleError("Parameter %r must be a list of numbers, one per member." % name)
            if len(value) != member_count:
                raise EnsembleError(
                    "Parameter %r must hold exactly %d values (one per member); got %d."
                    % (name, member_count, len(value))
                )
            try:
                value = [float(item) for item in value]
            except (TypeError, ValueError):
                raise EnsembleError("Parameter %r must contain only numbers." % name) from None
            if any(not math.isfinite(item) or item < 0 for item in value):
                raise EnsembleError("Parameter %r must contain finite, non-negative numbers." % name)
            if not any(item > 0 for item in value):
                raise EnsembleError("Parameter %r must contain at least one positive weight." % name)
        elif kind == "string":
            if not isinstance(value, str) or not value.strip():
                raise EnsembleError("Parameter %r must be a non-empty string." % name)
        cleaned[name] = value
    if method == "stacking":
        _validate_stacking_final_estimator(cleaned["final_estimator"], family)
    return cleaned


def _resolve_stacking_final_estimator(name, family):
    """Turn a final-estimator name into an unfitted estimator for ``family``."""
    available = STACKING_FINAL_ESTIMATORS.get(family)
    if not available:
        raise EnsembleError("Stacking is not available for %s." % (family or "this family"))
    if name is None:
        name = DEFAULT_STACKING_FINAL_ESTIMATOR[family]
    if name not in available:
        raise EnsembleError(
            "Unknown stacking final estimator %r for %s; choose one of %s."
            % (name, family, ", ".join(sorted(available)))
        )
    return available[name](), name


def _validate_stacking_final_estimator(name, family):
    _resolve_stacking_final_estimator(name, family)


def _check_probability_support(members, method):
    """Refuse a probability-averaging combiner when a member cannot produce them.

    Only soft voting needs probabilities: hard voting counts labels and works with any
    classifier (``LinearSVC`` has a decision function and no ``predict_proba``), so the
    check is scoped to what scikit-learn will actually require at fit time.
    """
    if method != "soft":
        return
    missing = [name for name, estimator in members if not hasattr(estimator, "predict_proba")]
    if missing:
        raise EnsembleError(
            "Soft voting averages class probabilities, but %s cannot predict them "
            "(no predict_proba); use hard voting or drop those members."
            % ", ".join(repr(name) for name in missing)
        )


def _check_sample_weight_support(members):
    """Refuse *classification* boosting when a member cannot be reweighted.

    ``AdaBoostClassifier`` gates its base estimator on
    ``has_fit_parameter(estimator_, "sample_weight")`` and then passes the round weights
    to ``fit``, so a member that cannot receive them would train an identical learner in
    every round. ``AdaBoostRegressor`` has neither behaviour (it reweights by
    resampling), which is why this check is not applied to regression members.
    """
    unsupported = [name for name, estimator in members if not supports_sample_weight(estimator)]
    if unsupported:
        raise EnsembleError(
            "Boosting reweights the samples after every round, but the final estimator of %s "
            "does not accept sample_weight; use bagging or voting for those members."
            % ", ".join(repr(name) for name in unsupported)
        )


def build_ensemble(
    method,
    members,
    *,
    family=None,
    params=None,
    columns=None,
    random_state=None,
    n_jobs=None,
):
    """Build the (unfitted) scikit-learn meta-estimator combining ``members``.

    :param method: one of :data:`ENSEMBLE_METHODS`.
    :param members: sequence of ``(name, estimator)`` pairs — the *trained* member
        models, in the order the caller wants them aggregated (weights are applied
        positionally, so the order is part of the request). Names must be unique.
    :param family: ``"Classification"``/``"Regression"``; inferred when omitted.
    :param params: method-specific parameters, validated by :func:`validate_params`.
    :param columns: feature names in training order. Each member is passed through
        :func:`~SKSurrogate.DataProcess.frame_tolerant` with them, so a member whose
        encoder selects columns by name survives bagging's positional
        ``X[indices]`` and the array-input prediction paths.
    :param random_state: seeds every stochastic component (bagging resampling).
    :param n_jobs: parallelism handed to scikit-learn; the caller is responsible for
        keeping it inside the process's compute budget.
    :returns: an unfitted estimator implementing ``fit``/``predict`` (and
        ``predict_proba`` for classification).
    """
    members = [(str(name), estimator) for name, estimator in members]
    names = [name for name, _ in members]
    if len(set(names)) != len(names):
        raise EnsembleError("Member names must be unique; got %s." % ", ".join(names))
    if not members:
        raise EnsembleError("At least one member model is required.")

    resolved_family = family or resolve_family(members)
    if family is not None:
        # Trust the caller's label only if it matches the members themselves.
        inferred = resolve_family(members)
        if inferred != family:
            raise EnsembleError(
                "Members are %s but %s was requested; all members must share one family."
                % (inferred, family)
            )
    cleaned = validate_params(method, params, family=resolved_family, member_count=len(members))
    normalised = [
        (name, frame_tolerant(estimator, columns=list(columns) if columns else None))
        for name, estimator in members
    ]

    if method in ("voting", "averaging"):
        return _build_vote(method, normalised, resolved_family, cleaned, n_jobs)
    if method == "stacking":
        return _build_stacking(normalised, resolved_family, cleaned, n_jobs)
    if method == "bagging":
        return _build_bagged(normalised, resolved_family, cleaned, n_jobs, random_state)
    if method == "boosting":
        return _build_boosted(normalised, resolved_family, cleaned, n_jobs, random_state)
    raise EnsembleError("Unknown ensemble method %r." % (method,))  # pragma: no cover - guarded above


def _aggregate(normalised, family, weights, n_jobs, *, soft=True, force_soft=False):
    """Aggregate already-wrapped members with the averaging family's combiner."""
    if family == REGRESSION:
        return VotingRegressor(normalised, weights=weights, n_jobs=n_jobs)
    voting = "soft" if (soft or force_soft) else "hard"
    _check_probability_support(normalised, voting)
    return VotingClassifier(normalised, voting=voting, weights=weights, n_jobs=n_jobs)


def _build_vote(method, normalised, family, params, n_jobs):
    weights = params.get("weights")
    if family == REGRESSION:
        # scikit-learn has no label-voting regressor, so `voting` and `averaging`
        # are the same operator; that identity is documented, not hidden.
        return VotingRegressor(normalised, weights=weights, n_jobs=n_jobs)
    voting = params.get("voting", "hard") if method == "voting" else "soft"
    _check_probability_support(normalised, voting)
    return VotingClassifier(normalised, voting=voting, weights=weights, n_jobs=n_jobs)


def _build_stacking(normalised, family, params, n_jobs):
    final_estimator, _ = _resolve_stacking_final_estimator(params.get("final_estimator"), family)
    cls = StackingClassifier if family == CLASSIFICATION else StackingRegressor
    return cls(
        normalised,
        final_estimator=final_estimator,
        cv=params.get("cv", 5),
        passthrough=bool(params.get("passthrough", False)),
        n_jobs=n_jobs,
    )


def _build_bagged(normalised, family, params, n_jobs, random_state):
    cls = BaggingClassifier if family == CLASSIFICATION else BaggingRegressor
    # scikit-learn's parameter constraints reject an explicit ``None`` for
    # ``max_samples``/``max_features``; the "all rows/all features" behaviour is the
    # omitted-parameter default, so unset values are left out rather than passed on.
    limits = {
        name: params[name]
        for name in ("max_samples", "max_features")
        if params.get(name) is not None
    }
    wrapped = [
        (
            name,
            cls(
                estimator=estimator,
                n_estimators=params.get("n_estimators", 10),
                bootstrap=bool(params.get("bootstrap", True)),
                random_state=random_state,
                n_jobs=n_jobs,
                **limits,
            ),
        )
        for name, estimator in normalised
    ]
    return _aggregate(wrapped, family, None, n_jobs)


def _build_boosted(normalised, family, params, n_jobs, random_state):
    if family == CLASSIFICATION:
        # AdaBoostClassifier refuses a base estimator without a literal
        # ``sample_weight`` parameter and forwards the round weights to it, so every
        # member has to be wrapped (and one that cannot be reweighted must be refused,
        # or every round would train an identical learner).
        _check_sample_weight_support(normalised)
        wrapped = [
            (
                name,
                AdaBoostClassifier(
                    estimator=SampleWeightRouter(estimator),
                    n_estimators=params.get("n_estimators", 20),
                    learning_rate=params.get("learning_rate", 1.0),
                    random_state=random_state,
                ),
            )
            for name, estimator in normalised
        ]
    else:
        # AdaBoostRegressor does *not* forward sample weights: ``_boost`` reweights by
        # resampling (``random_state.choice(..., p=sample_weight)``) and then calls
        # ``estimator.fit(X_, y_)``. Regression members therefore need no adapter and
        # are not refused for lacking sample-weight support.
        wrapped = [
            (
                name,
                AdaBoostRegressor(
                    estimator=estimator,
                    n_estimators=params.get("n_estimators", 20),
                    learning_rate=params.get("learning_rate", 1.0),
                    loss=params.get("loss", "linear"),
                    random_state=random_state,
                ),
            )
            for name, estimator in normalised
        ]
    return _aggregate(wrapped, family, None, n_jobs)


def cross_validated_scores(estimator, X, y, *, cv=5, scoring=None, n_jobs=None):
    """Mean/std of an estimator's cross-validated score, with non-finite values dropped.

    The members' scores and the ensemble's score are meant to be compared, so every
    call in one comparison must use the *same* ``cv`` object or the same integer fold
    count — an integer makes scikit-learn build one deterministic splitter, which is
    what keeps the comparison honest. Returns ``(mean, std, per_fold)`` with ``None``
    for any statistic that could not be computed (a fold that failed to fit), never
    ``nan``/``inf``, so the result stays JSON-serialisable.
    """

    def _finite(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return None if not math.isfinite(value) else value

    try:
        scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
    except Exception as exc:  # noqa: BLE001 - the caller reports the reason per estimator
        raise EnsembleError("%s: %s" % (type(exc).__name__, exc)) from exc
    finite = [_finite(score) for score in np.asarray(scores).ravel()]
    usable = [score for score in finite if score is not None]
    if not usable:
        return None, None, finite
    array = np.asarray(usable, dtype=float)
    return float(array.mean()), float(array.std()), finite


def member_scores(members, X, y, *, cv=5, scoring=None, n_jobs=None):
    """Per-member cross-validated scores, for the comparison table next to the ensemble.

    Each member is *cloned* inside every fold (scikit-learn's own behaviour in
    ``cross_val_score``), so no fitted state leaks between folds. Members that cannot
    be scored are reported with an ``error`` string instead of failing the run.
    """
    results = []
    for name, estimator in members:
        entry = {"member": name}
        try:
            mean, std, per_fold = cross_validated_scores(
                estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs
            )
            entry.update({"mean": mean, "std": std, "folds": per_fold})
        except EnsembleError as exc:
            entry["error"] = str(exc)
        results.append(entry)
    return results
