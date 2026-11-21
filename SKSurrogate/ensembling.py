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

import inspect
import math
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, is_classifier, is_regressor
from sklearn.calibration import CalibratedClassifierCV
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
from .scoring import check_scoring_direction

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

#: Post-hoc probability calibration applied to each member before it is combined.
CALIBRATION_METHODS = ("sigmoid", "isotonic")

#: How ``CalibratedClassifierCV`` builds its calibrators, as user-facing tokens.
#: scikit-learn's own default is exposed as ``"auto"`` rather than left implicit, and it is
#: resolved against the *installed* build (a boolean ``True`` before scikit-learn 1.6, the
#: literal ``"auto"`` from 1.6 on) instead of being hard-coded.
CALIBRATION_ENSEMBLE_TOKENS = {"auto": "auto", "per_fold": True, "single": False}

#: How an optimized weight vector is found. ``least_squares`` minimises the squared error
#: between the weighted blend and the target (the Brier score for classification), which
#: is a small convex problem solved on the simplex; ``dirichlet`` samples the simplex at
#: random and keeps the best task-scorer value; ``greedy`` is Caruana-style forward
#: selection with replacement, which produces integer weights and can therefore *drop* a
#: member visibly.
WEIGHT_STRATEGIES = ("least_squares", "dirichlet", "greedy")

#: Strategies that need a scikit-learn scorer: the task's own metric decides, so they
#: work for label metrics (accuracy, ROC AUC) and for error metrics alike.
SCORER_WEIGHT_STRATEGIES = ("dirichlet", "greedy")

#: A shared help string: weights are positional, and an optimized vector may contain a
#: zero, which is how a strategy reports that it dropped a member.
_WEIGHTS_HELP = (
    "One non-negative weight per member, in the order given; omitted means equal weights. "
    "A zero weight drops that member."
)

#: Parameter schema per method, consumed by the control plane's ``/methods``
#: endpoint and validated by :func:`validate_params` so the UI and the API agree.
METHOD_PARAMETERS = {
    "voting": (
        {"name": "weights", "type": "floats", "default": None, "label": "Weights",
         "help": _WEIGHTS_HELP},
        {"name": "voting", "type": "choice", "choices": ["hard", "soft"], "default": "hard",
         "label": "Vote", "family": CLASSIFICATION,
         "help": "Hard counts labels, soft averages class probabilities (requires every member to predict probabilities)."},
    ),
    "averaging": (
        {"name": "weights", "type": "floats", "default": None, "label": "Weights",
         "help": _WEIGHTS_HELP},
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
        {"name": "weights", "type": "floats", "default": None, "label": "Weights",
         "help": "Applied to the aggregation of the bagged members. " + _WEIGHTS_HELP},
    ),
    "boosting": (
        {"name": "n_estimators", "type": "integer", "default": 20, "min": 1, "max": 500, "label": "Rounds",
         "help": "Boosting rounds per member. Every member's final estimator must accept sample weights."},
        {"name": "learning_rate", "type": "float", "default": 1.0, "min": 0.0001, "max": 10.0, "label": "Learning rate",
         "help": "Shrinks each round's contribution."},
        {"name": "loss", "type": "choice", "choices": ["linear", "square", "exponential"], "default": "linear",
         "label": "Loss", "family": REGRESSION,
         "help": "AdaBoost.R2 loss updated after each round (regression only)."},
        {"name": "weights", "type": "floats", "default": None, "label": "Weights",
         "help": "Applied to the aggregation of the boosted members. " + _WEIGHTS_HELP},
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
        # Weights apply wherever the members are merged by a weighted vote/average;
        # stacking learns its own combination in the final estimator instead.
        accepts_weights = method != "stacking"
        entries.append(
            {
                "value": method,
                "label": method.capitalize(),
                "description": descriptions[method],
                "minimum_members": MINIMUM_MEMBERS[method],
                "parameters": [dict(spec) for spec in parameters],
                "aggregation": aggregation_label(method, family),
                "available": applies,
                "accepts_weights": accepts_weights,
            }
        )
    return {
        "family": family,
        "methods": entries,
        **calibration_and_weight_options(family),
    }


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


def _per_member_forms(method, normalised, family, params, n_jobs, random_state):
    """The per-member estimator that will be handed to the aggregator.

    Bagging and boosting give every member its own resampling/reweighting wrapper, so the
    "member" being aggregated is that wrapper; the other methods aggregate the member
    itself. Keeping this in one place is what makes calibration land in the right spot:
    calling :func:`calibrate_members` on *these* forms puts the calibrator outside the
    wrapper, which is the only nesting that does not refit it on bootstrap resamples.
    """
    if method in ("voting", "averaging", "stacking"):
        return list(normalised)
    if method == "bagging":
        cls = BaggingClassifier if family == CLASSIFICATION else BaggingRegressor
        # scikit-learn's parameter constraints reject an explicit ``None`` for
        # ``max_samples``/``max_features``; the "all rows/all features" behaviour is the
        # omitted-parameter default, so unset values are left out rather than passed on.
        limits = {
            name: params[name]
            for name in ("max_samples", "max_features")
            if params.get(name) is not None
        }
        return [
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
    if method == "boosting":
        if family == CLASSIFICATION:
            # AdaBoostClassifier refuses a base estimator without a literal
            # ``sample_weight`` parameter and forwards the round weights to it, so every
            # member has to be wrapped (and one that cannot be reweighted must be refused,
            # or every round would train an identical learner).
            _check_sample_weight_support(normalised)
            return [
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
        # AdaBoostRegressor does *not* forward sample weights: ``_boost`` reweights by
        # resampling (``random_state.choice(..., p=sample_weight)``) and then calls
        # ``estimator.fit(X_, y_)``. Regression members therefore need no adapter and are
        # not refused for lacking sample-weight support.
        return [
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
    raise EnsembleError("Unknown ensemble method %r." % (method,))  # pragma: no cover - guarded


def aggregate_forms(method, forms, family, params, n_jobs):
    """Aggregate per-member forms into the unfitted meta-estimator (no member wrapping)."""
    if method in ("voting", "averaging"):
        return _build_vote(method, forms, family, params, n_jobs)
    if method == "stacking":
        return _build_stacking(forms, family, params, n_jobs)
    if method in ("bagging", "boosting"):
        return _aggregate(forms, family, params.get("weights"), n_jobs)
    raise EnsembleError("Unknown ensemble method %r." % (method,))  # pragma: no cover - guarded


def member_forms(
    method,
    members,
    *,
    family=None,
    params=None,
    calibration=None,
    columns=None,
    random_state=None,
    n_jobs=None,
):
    """Resolve, validate and prepare the per-member estimators of one ensemble.

    Returns ``(forms, family, cleaned_params)``. ``forms`` are the estimators that will
    actually be combined — array-safe, with a bagging/boosting wrapper where the method
    calls for one, and with calibration wrapped **outermost** when requested. Weight
    optimization uses the same ``forms``, so a weight vector always describes exactly the
    members that end up in the ensemble.
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
    forms = _per_member_forms(method, normalised, resolved_family, cleaned, n_jobs, random_state)
    if calibration:
        forms = calibrate_members(
            forms,
            method=calibration.get("method", "sigmoid"),
            cv=calibration.get("cv", 5),
            ensemble=calibration.get("ensemble", "auto"),
            n_jobs=calibration.get("n_jobs"),
        )
    return forms, resolved_family, cleaned


def build_ensemble(
    method,
    members,
    *,
    family=None,
    params=None,
    calibration=None,
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
    :param calibration: optional ``{"method", "cv", "ensemble"}`` applied to each member's
        outermost form (see :func:`calibrate_member`), which is what repairs an
        overconfident member's probabilities before they are averaged.
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
    forms, resolved_family, cleaned = member_forms(
        method,
        members,
        family=family,
        params=params,
        calibration=calibration,
        columns=columns,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return aggregate_forms(method, forms, resolved_family, cleaned, n_jobs)


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


# --------------------------------------------------------------------------- #
# Per-member probability calibration
# --------------------------------------------------------------------------- #


def _installed_calibration_ensemble_default():
    """``CalibratedClassifierCV``'s own default for ``ensemble`` in the *installed* build.

    The value is not a constant across the versions this package supports: scikit-learn 1.5
    accepts only a boolean (default ``True``), while 1.6 added ``"auto"`` and made it the
    default. Reading the signature keeps the ``"auto"`` token meaning "whatever this build
    does by default" instead of a literal that a 1.5 install refuses
    (``InvalidParameterError: The 'ensemble' parameter … must be an instance of 'bool'``).
    """
    try:
        default = inspect.signature(CalibratedClassifierCV.__init__).parameters["ensemble"].default
    except (KeyError, TypeError, ValueError):  # pragma: no cover - defensive
        return True
    return True if default is inspect.Parameter.empty else default


def _resolve_calibration_ensemble(value) -> Any:
    """Turn an ``ensemble`` token (``auto``/``per_fold``/``single``) into sklearn's value.

    ``auto`` resolves against the installed build rather than a version string, so the same
    request works on a boolean-only scikit-learn and on one that names the default. The
    return type is deliberately unannotated beyond ``Any``: the accepted type of this
    argument changed from ``bool`` to ``str`` between scikit-learn versions.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return _installed_calibration_ensemble_default()
    token = str(value).strip().lower()
    if token not in CALIBRATION_ENSEMBLE_TOKENS:
        raise EnsembleError(
            "Calibration 'ensemble' must be one of %s."
            % ", ".join(sorted(CALIBRATION_ENSEMBLE_TOKENS))
        )
    resolved = CALIBRATION_ENSEMBLE_TOKENS[token]
    return _installed_calibration_ensemble_default() if resolved == "auto" else resolved


def calibrate_member(estimator, *, method="sigmoid", cv=5, ensemble="auto", n_jobs=None):
    """Wrap one member in a probability calibrator.

    Calibration is a post-hoc map from a member's scores to probabilities, so it changes
    what the *aggregation* sees without touching the member's own decisions. It is
    applied to the **outermost per-member form** — after any bagging or boosting wrapper
    — and never inside one: scikit-learn's resampling wrappers forward ``sample_weight``
    to any base whose ``fit`` declares it (``CalibratedClassifierCV`` does), which makes
    it warn *"Since Pipeline does not appear to accept sample_weight … the result of the
    calibration is likely to be incorrect"* and fits the calibrator on bootstrap
    resamples instead of the fold's held-out rows. Measured on a 3-replica bagging
    member: three such warnings nested inside, none nested outside.

    :param method: ``"sigmoid"`` (Platt scaling, the default: safe on few rows) or
        ``"isotonic"`` (more flexible, needs enough samples per fold to avoid step noise).
    :param cv: folds used to produce the calibrator's held-out scores (at least 2).
    :param ensemble: ``"auto"`` (scikit-learn's own default *for the installed build*: one
        calibrator per fold for an ordinary estimator), ``"per_fold"`` to force that, or
        ``"single"`` to fit one calibrator on the cross-validated scores (the classic
        Platt/isotonic construction). ``True``/``False`` are accepted as synonyms of the
        last two. ``"auto"`` is resolved at call time, so a scikit-learn that only accepts
        a boolean is served correctly.
    """
    if method not in CALIBRATION_METHODS:
        raise EnsembleError(
            "Unknown calibration method %r; choose one of %s."
            % (method, ", ".join(CALIBRATION_METHODS))
        )
    if isinstance(cv, bool) or not isinstance(cv, int) or cv < 2:
        raise EnsembleError("Calibration folds must be an integer of at least 2.")
    return CalibratedClassifierCV(
        estimator,
        method=method,
        cv=cv,
        ensemble=_resolve_calibration_ensemble(ensemble),
        n_jobs=n_jobs,
    )


def calibrate_members(members, *, method="sigmoid", cv=5, ensemble="auto", n_jobs=None):
    """Calibrate every member, refusing what cannot be calibrated.

    Only classification members have probabilities to calibrate, and a member needs a
    ``predict_proba`` or a ``decision_function`` for a calibrator to be fitted at all;
    both are checked here rather than left to a confusing failure inside the fit.
    """
    unsupported = [
        name for name, estimator in members if problem_family_of(estimator) != CLASSIFICATION
    ]
    if unsupported:
        raise EnsembleError(
            "Calibration is available for classification members only; %s is not a "
            "classifier." % ", ".join(repr(name) for name in unsupported)
        )
    uncalibratable = [
        name
        for name, estimator in members
        if not (hasattr(estimator, "predict_proba") or hasattr(estimator, "decision_function"))
    ]
    if uncalibratable:
        raise EnsembleError(
            "Calibration needs a member that can predict probabilities or scores; %s "
            "exposes neither predict_proba nor decision_function."
            % ", ".join(repr(name) for name in uncalibratable)
        )
    return [
        (name, calibrate_member(estimator, method=method, cv=cv, ensemble=ensemble, n_jobs=n_jobs))
        for name, estimator in members
    ]


def _brier_and_log_loss(probabilities, y, classes):
    """``(brier, log_loss)`` for a probability matrix, or ``(None, None)`` when undefined."""
    from sklearn.metrics import brier_score_loss, log_loss

    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(classes)
    if len(labels) == 2:
        positive = (np.asarray(y) == labels[1]).astype(float)
        brier = float(brier_score_loss(positive, probabilities[:, 1]))
    else:
        # Multiclass Brier: mean squared error against the one-hot target.
        onehot = np.zeros_like(probabilities)
        onehot[np.arange(len(y)), np.searchsorted(labels, y)] = 1.0
        brier = float(np.mean((probabilities - onehot) ** 2))
    try:
        loss = float(log_loss(y, probabilities, labels=list(labels)))
    except ValueError:
        loss = None
    return brier, loss


def calibration_report(members, X, y, *, cv=5, method="sigmoid", ensemble="auto", n_jobs=None):
    """Per-member Brier score and log loss before and after calibration, on the same folds.

    This is the evidence that calibration did anything: on an already well-calibrated
    member both are ~unchanged, while an overconfident member improves sharply (measured
    on a fully grown decision tree: Brier 0.2000 → 0.1385 and log loss 7.2087 → 0.4440).
    Each member is cloned per fold, so the "before" and "after" numbers describe the same
    rows.
    """
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_predict

    report = []
    for name, estimator in members:
        entry = {"member": name, "method": method, "cv": cv}
        try:
            if not hasattr(estimator, "predict_proba"):
                # A member with only a decision function is calibratable, but Brier score and
                # log loss are probability metrics: substituting a decision-function scale
                # would report a different quantity under the same column name.
                entry["error"] = (
                    "Brier score and log loss need a probability response "
                    "(predict_proba); this member exposes decision_function only, so it is "
                    "calibrated without a before/after metric."
                )
                report.append(entry)
                continue
            raw = cross_val_predict(clone(estimator), X, y, cv=cv, method="predict_proba")
            calibrated = cross_val_predict(
                calibrate_member(estimator, method=method, cv=cv, ensemble=ensemble, n_jobs=n_jobs),
                X,
                y,
                cv=cv,
                method="predict_proba",
            )
            classes = np.unique(np.asarray(y))
            brier_before, loss_before = _brier_and_log_loss(raw, y, classes)
            brier_after, loss_after = _brier_and_log_loss(calibrated, y, classes)
            entry.update(
                {
                    "brier_before": brier_before,
                    "brier_after": brier_after,
                    "log_loss_before": loss_before,
                    "log_loss_after": loss_after,
                }
            )
        except Exception as exc:  # noqa: BLE001 - one member must not fail the report
            entry["error"] = "%s: %s" % (type(exc).__name__, exc)
        report.append(entry)
    return report


# --------------------------------------------------------------------------- #
# Weight optimization
# --------------------------------------------------------------------------- #


class _WeightedBlend(BaseEstimator):
    """Apply a weight vector to precomputed member outputs, so a scorer can judge it.

    A weight *search* has to score thousands of candidate vectors; refitting the members
    for each is impossible, so the members' out-of-fold outputs are computed once and this
    adapter exposes them under the response method the scorer asks for. It never touches
    data — ``fit`` is a no-op — which is why the search can only ever see out-of-fold
    member output.
    """

    def __init__(self, outputs=None, weights=None, classes=None, family=None):
        self.outputs = outputs
        self.weights = weights
        self.classes = classes
        self.family = family

    def fit(self, X=None, y=None, **kwargs):
        return self

    def _blend(self):
        weights = np.asarray(self.weights, dtype=float)
        total = weights.sum()
        if total <= 0:
            raise EnsembleError("Weights must sum to a positive value.")
        weights = weights / total
        if self.family == CLASSIFICATION:
            return sum(weight * output for weight, output in zip(weights, self.outputs))
        # ``outputs`` is one 1-D prediction vector per member, so the stacked array is
        # (members, rows): transpose before applying the per-member weights. Without
        # this, a scorer-based search (greedy/dirichlet) failed with "matmul: ...
        # (size <members> is different from <rows>)" as soon as members != rows.
        return np.asarray(self.outputs).T @ weights

    def predict_proba(self, X=None):
        return self._blend()

    def predict(self, X=None):
        if self.family == CLASSIFICATION:
            return np.asarray(self.classes)[np.argmax(self._blend(), axis=1)]
        return self._blend()

    def decision_function(self, X=None):
        blended = self._blend()
        if self.family == CLASSIFICATION:
            return blended[:, 1] if blended.shape[1] == 2 else blended
        return blended


def _take_rows(data, index):
    """Positional rows of ``X``/``y`` without assuming numpy or pandas."""
    if hasattr(data, "iloc"):
        return data.iloc[index]
    return data[index]


def _folds_partition(folds, n_samples):
    """True when the folds' test sets cover every sample exactly once."""
    covered = np.zeros(int(n_samples), dtype=int)
    for _, test_index in folds:
        covered[np.asarray(test_index, dtype=int)] += 1
    return bool(covered.size and (covered == 1).all())


def _walk_forward_outputs(members, X, y, folds, mask):
    """Out-of-fold member outputs replayed fold by fold for a non-partition splitter.

    ``cross_val_predict`` refuses a splitter whose test sets do not partition the
    samples (``only works for partitions``): a rolling-origin splitter leaves the
    initial training window uncovered and its origins may hold gaps. Fitting each
    fold here reproduces exactly what ``cross_val_predict`` does for a partition —
    clone, fit on the fold's training rows, predict its test rows — so every value
    still comes from a member that never saw that row.
    """
    from sklearn.base import clone

    y_array = np.asarray(y)
    classes = np.unique(y_array)
    outputs = []
    for _, estimator in members:
        family = problem_family_of(estimator)
        if family == CLASSIFICATION:
            values = np.full((len(y_array), len(classes)), np.nan, dtype=float)
        else:
            values = np.full(len(y_array), np.nan, dtype=float)
        for train_index, test_index in folds:
            train_index = np.asarray(train_index, dtype=int)
            test_index = np.asarray(test_index, dtype=int)
            fitted = clone(estimator).fit(_take_rows(X, train_index), y_array[train_index])
            if family == CLASSIFICATION:
                probability = np.asarray(
                    fitted.predict_proba(_take_rows(X, test_index)), dtype=float
                )
                positions = np.searchsorted(classes, np.asarray(fitted.classes_))
                values[np.ix_(test_index, positions)] = probability
            else:
                values[test_index] = np.asarray(
                    fitted.predict(_take_rows(X, test_index)), dtype=float
                )
        outputs.append(values[mask])
    return outputs


def out_of_fold_outputs(members, X, y, cv, n_jobs, *, return_mask=False):
    """Out-of-fold member outputs: probability matrices (classification) or predictions.

    One pass per member, on the same folds, so every weight the optimizer considers is
    judged on rows no member was trained on. This is the only place a weight search is
    allowed to look.

    A splitter whose folds are not a partition of the rows — a walk-forward
    ``RollingOriginSplit``, which leaves the initial training window and any gap
    uncovered — is refused by ``cross_val_predict`` with
    ``ValueError: cross_val_predict only works for partitions``. Those folds are
    replayed fold by fold instead, and the uncovered rows are dropped from every
    member through the same mask, so the members are still compared on identical,
    never-trained-on rows. With ``return_mask=True`` the mask is returned too, so
    the caller can align ``y`` (``None`` means every row was used).
    """
    from sklearn.model_selection import cross_val_predict

    folds = None
    if not isinstance(cv, (int, np.integer)) and hasattr(cv, "split"):
        candidate = list(cv.split(X, y))
        if candidate and not _folds_partition(candidate, len(np.asarray(y))):
            folds = candidate
    if folds is not None:
        mask = np.zeros(len(np.asarray(y)), dtype=bool)
        for _, test_index in folds:
            mask[np.asarray(test_index, dtype=int)] = True
        outputs = _walk_forward_outputs(members, X, y, folds, mask)
        return (outputs, mask) if return_mask else outputs

    outputs = []
    for _, estimator in members:
        if problem_family_of(estimator) == CLASSIFICATION:
            outputs.append(
                np.asarray(
                    cross_val_predict(estimator, X, y, cv=cv, method="predict_proba", n_jobs=n_jobs)
                )
            )
        else:
            outputs.append(
                np.asarray(cross_val_predict(estimator, X, y, cv=cv, method="predict", n_jobs=n_jobs))
            )
    return (outputs, None) if return_mask else outputs


def _simplex_least_squares(outputs, y, family, classes):
    """Weights minimizing squared error between the blend and the target, on the simplex.

    For classification the target is the one-hot label matrix, so this minimizes the
    Brier score of the blended probabilities; for regression it is the plain MSE. The
    problem is convex in the weights with a simplex constraint (non-negative, summing to
    one), which is what keeps a member interpretable as "dropped" when its weight is
    exactly zero — measured on a three-member set, the optimizer put 0.902 on the best
    member and 0.000 on a useless one.
    """
    from scipy.optimize import minimize

    count = len(outputs)
    if family == CLASSIFICATION:
        target = np.zeros((len(y), len(classes)))
        target[np.arange(len(y)), np.searchsorted(classes, y)] = 1.0
    else:
        target = np.asarray(y, dtype=float)

    def objective(weights):
        blended = (
            sum(weight * output for weight, output in zip(weights, outputs))
            if family == CLASSIFICATION
            else np.asarray(outputs).T @ weights
        )
        return float(np.mean((blended - target) ** 2))

    result = minimize(
        objective,
        np.full(count, 1.0 / count),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints=[{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    weights = np.asarray(result.x, dtype=float)
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total <= 0:
        weights = np.full(count, 1.0 / count)
    else:
        weights = weights / total
    return weights, objective(weights)


def _scorer_value(scorer, outputs, weights, y, family, classes):
    """The task scorer's value for one candidate weight vector, on the stored outputs."""
    blend = _WeightedBlend(outputs, weights, classes, family)
    return float(scorer(blend, None, y))


def _dirichlet_search(outputs, weights_shape, y, family, classes, *, scorer, samples, random_state):
    generator = np.random.default_rng(random_state)
    count = weights_shape
    best_weights = np.full(count, 1.0 / count)
    best_value = _scorer_value(scorer, outputs, best_weights, y, family, classes)
    for _ in range(int(samples)):
        candidate = generator.dirichlet(np.ones(count))
        value = _scorer_value(scorer, outputs, candidate, y, family, classes)
        if value > best_value:
            best_weights, best_value = candidate, value
    return best_weights, best_value


def _greedy_selection(outputs, weights_shape, y, family, classes, *, scorer, rounds):
    """Caruana-style forward selection with replacement: integer weights, visible drops."""
    counts = np.zeros(weights_shape)
    best_value = -np.inf
    for _ in range(int(rounds)):
        chosen_index, chosen_value = None, best_value
        for index in range(weights_shape):
            candidate = counts.copy()
            candidate[index] += 1
            value = _scorer_value(scorer, outputs, candidate / candidate.sum(), y, family, classes)
            if value > chosen_value:
                chosen_index, chosen_value = index, value
        if chosen_index is None:
            break
        counts[chosen_index] += 1
        best_value = chosen_value
    if counts.sum() <= 0:
        return np.full(weights_shape, 1.0 / weights_shape), best_value
    return counts / counts.sum(), best_value


def optimize_weights(
    members,
    X,
    y,
    *,
    cv=5,
    strategy="least_squares",
    scoring=None,
    samples=200,
    rounds=20,
    random_state=None,
    n_jobs=None,
):
    """Find member weights that maximize agreement with the target, on out-of-fold output.

    The weights are fitted on **out-of-fold** member output, so no member scores a row it
    was trained on; that is the minimum requirement for the vector to mean anything. It is
    still an in-sample fit of the *weights*, which is why the score of the weighted
    ensemble is optimistic unless the whole procedure is nested (see
    :func:`nested_ensemble_cv`). How much optimism is bought depends on whether the search
    has anything to fit: measured across ten configurations (200-560 rows, two to three
    members, least-squares and greedy, three to five scoring folds) the gap between the
    in-sample and the nested number ran from 0.0000 — the search collapsed onto one member,
    so the ensemble *is* that member — to 0.0233.

    :returns: ``{"weights", "strategy", "objective", "objective_name", "folds", "budget",
        "dropped"}`` — ``dropped`` names the members whose weight collapsed to zero.
    """
    # Same validation the submit-time path runs, so a bad strategy or budget is refused
    # before any cross-validation is spent and both callers say the same thing.
    validate_weight_optimization({"strategy": strategy, "samples": samples, "rounds": rounds})
    members = list(members)
    if len(members) < 1:
        raise EnsembleError("At least one member is required to optimize weights.")
    family = resolve_family(members)
    outputs, mask = out_of_fold_outputs(members, X, y, cv, n_jobs, return_mask=True)
    if mask is not None:
        # A walk-forward splitter cannot predict its own initial training window.
        # Dropping those rows from the target too keeps the weights fitted on rows
        # the members actually scored out of fold, never on a placeholder.
        y = np.asarray(y)[mask]
        unscored_rows = int(mask.size - int(mask.sum()))
    else:
        unscored_rows = 0
    classes = np.unique(np.asarray(y)) if family == CLASSIFICATION else None

    if strategy == "least_squares":
        weights, objective = _simplex_least_squares(outputs, y, family, classes)
        objective_name = "mean squared error against the target (Brier score for classification)"
        budget = None
    else:
        from sklearn.metrics import get_scorer

        if scoring is None:
            raise EnsembleError(
                "Strategy %r maximizes the task's scorer, so a scoring metric is required." % strategy
            )
        # The weight search keeps the largest value, so a directionless callable
        # would be maximized as-is; strings and scorer objects are higher-is-better.
        try:
            scorer = check_scoring_direction(scoring)
        except TypeError as exc:
            raise EnsembleError(str(exc)) from exc
        try:
            scorer = get_scorer(scorer)
        except ValueError as exc:
            raise EnsembleError(str(exc)) from exc
        if strategy == "dirichlet":
            weights, objective = _dirichlet_search(
                outputs, len(members), y, family, classes,
                scorer=scorer, samples=samples, random_state=random_state,
            )
            objective_name = "scorer %r over %d sampled weight vectors" % (scoring, int(samples))
            budget = {"samples": int(samples)}
        else:
            weights, objective = _greedy_selection(
                outputs, len(members), y, family, classes, scorer=scorer, rounds=rounds
            )
            objective_name = "scorer %r over %d forward-selection rounds" % (scoring, int(rounds))
            budget = {"rounds": int(rounds)}
    return {
        "weights": [float(weight) for weight in weights],
        "strategy": strategy,
        # A degenerate search (every candidate scoring identically, or a scorer returning a
        # non-finite value) leaves the uniform vector in place and no usable objective.
        # Reporting ±inf here would emit `-Infinity`, which is not valid JSON, into the job
        # result — the objective is the one field allowed to be absent instead.
        "objective": objective if math.isfinite(objective) else None,
        "objective_name": objective_name,
        "folds": cv if isinstance(cv, int) else getattr(cv, "get_n_splits", lambda: None)(),
        "budget": budget,
        "rows_scored": int(len(np.asarray(y))),
        # Rows no fold could predict (a walk-forward splitter's initial training
        # window): named rather than silently folded into a smaller sample.
        "rows_unscored": unscored_rows,
        "dropped": [
            name for (name, _), weight in zip(members, weights) if weight <= 1e-12
        ],
    }


# --------------------------------------------------------------------------- #
# Data-dependent orchestration: calibrate, weight, fit, and score honestly
# --------------------------------------------------------------------------- #


def validate_weight_optimization(request):
    """Shape and value checks for a weight-optimization request, before any data is touched.

    Split out so the same rules run at submit time (the control plane refuses a bad
    request before queueing a job) and inside :func:`optimize_weights` (the library
    refuses it before spending a cross-validation), with one wording for both.
    """
    request = request or {}
    strategy = request.get("strategy", "least_squares")
    if strategy not in WEIGHT_STRATEGIES:
        raise EnsembleError(
            "Unknown weight strategy %r; choose one of %s."
            % (strategy, ", ".join(WEIGHT_STRATEGIES))
        )
    cleaned: dict = {"strategy": strategy}
    for name, default, low, high in (("samples", 200, 10, 5000), ("rounds", 20, 1, 500)):
        value = request.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or not (low <= value <= high):
            raise EnsembleError(
                "weight_optimization.%s must be an integer between %d and %d."
                % (name, low, high)
            )
        cleaned[name] = value
    return cleaned


def validate_combination(method, params=None, calibration=None, weight_optimization=None):
    """Refuse the two request-level conflicts that are knowable without any data.

    Both are ambiguous rather than impossible, which is why they are errors instead of
    silent precedence rules: an optimized vector would replace an explicit one, and
    stacking learns its own combination so a member weight vector would be ignored.
    Shared by the library entry point and the control plane so the messages cannot drift.
    """
    if weight_optimization and (params or {}).get("weights"):
        raise EnsembleError(
            "Supply either explicit weights or weight optimization, not both: the "
            "optimized vector would silently replace the one you passed."
        )
    if weight_optimization and method == "stacking":
        raise EnsembleError(
            "Stacking learns its own combination in the final estimator, so member "
            "weights do not apply; optimize weights with voting, averaging, bagging or "
            "boosting instead."
        )
    if weight_optimization:
        validate_weight_optimization(weight_optimization)
    if calibration and calibration.get("method") not in (None, *CALIBRATION_METHODS):
        raise EnsembleError(
            "Unknown calibration method %r; choose one of %s."
            % (calibration.get("method"), ", ".join(CALIBRATION_METHODS))
        )
    return True


def fit_ensemble(
    method,
    members,
    X,
    y,
    *,
    family=None,
    params=None,
    calibration=None,
    weight_optimization=None,
    columns=None,
    cv=5,
    scoring=None,
    random_state=None,
    n_jobs=None,
):
    """Calibrate (optionally), optimize weights (optionally), build and fit one ensemble.

    Returns ``(estimator, provenance)``. ``provenance`` records what was done *and* how
    the numbers should be read: the weight vector with its strategy/objective/folds,
    whether members were calibrated, and the caveat that a weight vector fitted on these
    folds makes the resulting score optimistic (use :func:`nested_ensemble_cv` for the
    honest number).

    Both the weights and the calibration are computed from the members'
    *final per-member forms* — the bagging/boosting wrapper and the calibrator in the
    nesting the ensemble will actually use — so the vector describes exactly what is
    combined.
    """
    members = list(members)
    validate_combination(method, params, calibration, weight_optimization)
    forms, resolved_family, cleaned = member_forms(
        method,
        members,
        family=family,
        params=params,
        calibration=calibration,
        columns=columns,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    provenance = {
        "calibration": dict(calibration) if calibration else None,
        "weights": None,
        # The parameters actually used (optimized weights injected) and the per-member
        # forms that were combined. A caller records the first and may score the second;
        # neither is fitted state, and the *forms* must not be written into a bundle's
        # metadata (they are live estimator objects).
        "parameters": None,
        "forms": None,
    }
    if weight_optimization:
        job_threads = weight_optimization.get("n_jobs") or n_jobs
        optimized = optimize_weights(
            forms,
            X,
            y,
            cv=cv,
            strategy=weight_optimization.get("strategy", "least_squares"),
            scoring=scoring,
            samples=weight_optimization.get("samples", 200),
            rounds=weight_optimization.get("rounds", 20),
            random_state=random_state,
            n_jobs=job_threads,
        )
        provenance["weights"] = optimized
        cleaned["weights"] = optimized["weights"]

    provenance["parameters"] = dict(cleaned)
    provenance["forms"] = forms
    estimator = aggregate_forms(method, forms, resolved_family, cleaned, n_jobs)
    estimator.fit(X, y)
    return estimator, provenance


def nested_ensemble_cv(
    method,
    members,
    X,
    y,
    *,
    family=None,
    params=None,
    calibration=None,
    weight_optimization=None,
    columns=None,
    outer_cv=5,
    inner_cv=3,
    scoring=None,
    random_state=None,
    n_jobs=None,
):
    """Honest estimate for a weight-optimized ensemble: refit everything inside each fold.

    Every outer training fold gets its own calibration, its own out-of-fold member output
    and its own weight vector; the score is read on the held-out fold, which the weight
    search never saw. That is what separates this number from the in-sample one. How far
    apart they are is a property of the search, not a constant: measured across ten
    configurations (200-560 rows, two to three members, least-squares and greedy, three to
    five scoring folds) the gap ran from 0.0000 — the search collapsed onto one member, so
    both protocols score the same estimator — to 0.0233 (three members including a 1-NN
    memorizer, 300 rows, greedy, three scoring folds: 0.8100 in-sample against
    0.7867 ± 0.0340 here). A zero gap is therefore a finding about the members, not a
    failure of the estimate.

    :returns: ``{"scoring", "outer_cv", "inner_cv", "folds", "mean", "std", "errors",
        "per_fold_weights"}``. Fold-level failures are reported per fold rather than
        raising, so a single degenerate split does not lose the estimate.
    """
    from sklearn.model_selection import check_cv

    members = list(members)
    validate_combination(method, params, calibration, weight_optimization)
    # Validate the member set/parameters once up front: a request-level mistake must fail
    # loudly here, not turn into one error per fold and a null estimate.
    member_forms(
        method,
        members,
        family=family,
        params=params,
        calibration=calibration,
        columns=columns,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    outer = check_cv(outer_cv, y, classifier=problem_family_of(members[0][1]) == CLASSIFICATION)
    if isinstance(outer, int):
        # ``check_cv`` can hand back the fold count unchanged (it does so when it cannot
        # decide a classifier/regressor splitter); build the splitter explicitly so the
        # nested loop always has ``.split``.
        from sklearn.model_selection import KFold, StratifiedKFold

        outer = (
            StratifiedKFold(n_splits=outer, shuffle=True, random_state=random_state)
            if problem_family_of(members[0][1]) == CLASSIFICATION
            else KFold(n_splits=outer, shuffle=True, random_state=random_state)
        )
    try:
        # Materialized before the loop: a fold count the data cannot support raises here,
        # where it can be reported as a request problem, rather than as one error per fold.
        splits = list(outer.split(X, y))
    except Exception as exc:  # noqa: BLE001 - re-raised with the splitter's own reason
        raise EnsembleError(
            "Could not build %s outer folds over %d rows: %s: %s"
            % (outer_cv, len(y), type(exc).__name__, exc)
        ) from exc
    folds = []
    errors = []
    per_fold_weights = []
    for index, (train_index, test_index) in enumerate(splits):
        try:
            X_train = X.iloc[train_index] if hasattr(X, "iloc") else X[train_index]
            X_test = X.iloc[test_index] if hasattr(X, "iloc") else X[test_index]
            y_train = np.asarray(y)[train_index]
            y_test = np.asarray(y)[test_index]
            forms, resolved_family, cleaned = member_forms(
                method,
                members,
                family=family,
                params=params,
                calibration=calibration,
                columns=columns,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            if weight_optimization:
                optimized = optimize_weights(
                    forms,
                    X_train,
                    y_train,
                    cv=inner_cv,
                    strategy=weight_optimization.get("strategy", "least_squares"),
                    scoring=scoring,
                    samples=weight_optimization.get("samples", 200),
                    rounds=weight_optimization.get("rounds", 20),
                    random_state=random_state,
                    n_jobs=weight_optimization.get("n_jobs") or n_jobs,
                )
                cleaned["weights"] = optimized["weights"]
                # The point of the nested run: the weight vector is a *decision* made per
                # fold, so recording it shows how stable the search was across folds.
                per_fold_weights.append({"fold": index, "weights": optimized["weights"]})
            estimator = aggregate_forms(method, forms, resolved_family, cleaned, n_jobs)
            estimator.fit(X_train, y_train)
            if scoring:
                from sklearn.metrics import get_scorer

                try:
                    value = float(get_scorer(scoring)(estimator, X_test, y_test))
                except ValueError as exc:
                    raise EnsembleError(str(exc)) from exc
            else:
                value = float(estimator.score(X_test, y_test))
            folds.append(value)
        except Exception as exc:  # noqa: BLE001 - one bad split must not lose the estimate
            errors.append({"fold": index, "detail": "%s: %s" % (type(exc).__name__, exc)})
    usable = [value for value in folds if math.isfinite(value)]
    return {
        "scoring": scoring,
        "outer_cv": outer_cv if isinstance(outer_cv, int) else type(outer_cv).__name__,
        "inner_cv": inner_cv if isinstance(inner_cv, int) else type(inner_cv).__name__,
        "folds": folds,
        "mean": float(np.mean(usable)) if usable else None,
        "std": float(np.std(usable)) if usable else None,
        "errors": errors,
        "per_fold_weights": per_fold_weights or None,
    }


def calibration_and_weight_options(family):
    """The calibration and weight-optimization schema blocks for the API/UI.

    One source of truth for both sides: the control plane forwards these values and the
    page renders controls from them, so a new strategy or calibration method cannot appear
    in one and not the other.
    """
    return {
        "calibration": {
            "available": family == CLASSIFICATION,
            "methods": [
                {
                    "value": "sigmoid",
                    "label": "Sigmoid (Platt)",
                    "help": "Fits a logistic map to the member's scores; safe on few rows.",
                },
                {
                    "value": "isotonic",
                    "label": "Isotonic",
                    "help": "Fits a monotone step function; more flexible, needs enough rows per fold.",
                },
            ],
            "parameters": [
                {"name": "method", "type": "choice", "choices": list(CALIBRATION_METHODS),
                 "default": "sigmoid", "label": "Method"},
                {"name": "cv", "type": "integer", "default": 5, "min": 2, "max": 20,
                 "label": "Folds", "help": "Folds used to fit the calibrator."},
                {"name": "ensemble", "type": "choice",
                 "choices": sorted(CALIBRATION_ENSEMBLE_TOKENS), "default": "auto",
                 "label": "Calibrators",
                 "help": "auto (scikit-learn's own default for the installed build) or per_fold "
                         "average one calibrator per fold; single fits one calibrator on the "
                         "cross-validated scores."},
            ],
            "note": (
                "Calibration is applied to each member's outermost form and refitted inside "
                "every comparison fold; it is refused for regression members."
            ),
        },
        "weight_optimization": {
            "available": True,
            "strategies": [
                {"value": "least_squares", "label": "Least squares (Brier/MSE)",
                 "help": "Convex fit of the weights to the target, on out-of-fold member output; needs no metric."},
                {"value": "dirichlet", "label": "Random simplex search",
                 "help": "Samples weight vectors and keeps the best value of the task's scoring metric."},
                {"value": "greedy", "label": "Greedy selection (Caruana)",
                 "help": "Forward selection with replacement: integer weights, so a dropped member is visible."},
            ],
            "parameters": [
                {"name": "strategy", "type": "choice", "choices": list(WEIGHT_STRATEGIES),
                 "default": "least_squares", "label": "Strategy"},
                {"name": "samples", "type": "integer", "default": 200, "min": 10, "max": 5000,
                 "label": "Samples", "help": "Candidate vectors for the random search."},
                {"name": "rounds", "type": "integer", "default": 20, "min": 1, "max": 500,
                 "label": "Rounds", "help": "Forward-selection rounds for the greedy strategy."},
            ],
            "note": (
                "Weights are fitted on out-of-fold member output; the score of the weighted "
                "ensemble is still optimistic unless the honest nested estimate is requested."
            ),
        },
    }
