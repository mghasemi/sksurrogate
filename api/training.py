"""Shared model-fitting helpers used by the bundles, experiments, and retraining routers."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import get_scorer, get_scorer_names
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from SKSurrogate import AML, Categorical, Integer, ModelBundle, Real

from .config import settings
from .deps import bad_request, finite_or_none, not_found, open_tracker

BASELINE_ESTIMATORS = {
    "linear_regression": LinearRegression,
    "logistic_regression": LogisticRegression,
    "random_forest_classifier": RandomForestClassifier,
    "random_forest_regressor": RandomForestRegressor,
}

# Scorer name fragments whose display label should not be title-cased verbatim.
_SCORING_ACRONYMS = {
    "roc": "ROC",
    "auc": "AUC",
    "f1": "F1",
    "r2": "R²",
    "d2": "D²",
    "ovr": "one-vs-rest",
    "ovo": "one-vs-one",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "msle": "MSLE",
    "rmsle": "RMSLE",
}

# Scorers defined directly in ``sklearn.metrics._scorer`` carry no problem-family
# signal in their module, so pin the ones scikit-learn ships as classification.
_SCORING_FAMILY_OVERRIDES = {
    "positive_likelihood_ratio": "Classification",
    "neg_negative_likelihood_ratio": "Classification",
}


def _scoring_label(name):
    """Human-readable label for a scikit-learn scorer name, e.g. ``r2`` or ``neg_mean_squared_error``."""
    negated = name.startswith("neg_")
    base = name[4:] if negated else name
    words = [_SCORING_ACRONYMS.get(word, word) for word in base.split("_")]
    label = " ".join(words)
    label = label[:1].upper() + label[1:]
    return label + " (negated)" if negated else label


def _scoring_family(name):
    """Classify a scorer name into a problem family.

    Prefers the metric function's module (``sklearn.metrics._regression`` etc.),
    which is not available for scorers defined directly in
    ``sklearn.metrics._scorer`` — those are covered by the explicit override map.
    """
    override = _SCORING_FAMILY_OVERRIDES.get(name)
    if override is not None:
        return override
    module = getattr(getattr(get_scorer(name), "_score_func", None), "__module__", "") or ""
    if ".cluster." in module:
        return "Clustering"
    if module.endswith("_regression"):
        return "Regression"
    if module.endswith(("_classification", "_ranking")):
        return "Classification"
    return "Other"


def standard_scoring_options():
    """Grouped standard scikit-learn scorers usable as the AML/EOA objective.

    Derived from ``sklearn.metrics.get_scorer_names()`` so the catalogue always
    matches the installed scikit-learn, and grouped by problem family so the UI
    can render a tidy dropdown. Every name here is a valid ``scoring`` value for
    :class:`~SKSurrogate.AML`.
    """
    groups = {"Classification": [], "Regression": [], "Clustering": [], "Other": []}
    for name in sorted(get_scorer_names()):
        groups[_scoring_family(name)].append({"value": name, "label": _scoring_label(name)})
    return [{"label": label, "options": options} for label, options in groups.items() if options]


#: Curated short-list of metrics offered as learning-curve tabs on the Evaluation
#: page. `negate` marks `neg_*` scikit-learn scorers whose sign is flipped so the
#: curve plots the metric in its natural units (RMSE rather than "… (negated)").
_LEARNING_CURVE_METRICS = {
    "Classification": [
        {"value": "accuracy", "label": "Accuracy", "negate": False, "higher_is_better": True},
        {"value": "f1", "label": "F1", "negate": False, "higher_is_better": True},
        {"value": "precision", "label": "Precision", "negate": False, "higher_is_better": True},
        {"value": "recall", "label": "Recall", "negate": False, "higher_is_better": True},
        {"value": "roc_auc", "label": "ROC AUC", "negate": False, "higher_is_better": True},
    ],
    "Regression": [
        {"value": "r2", "label": "R²", "negate": False, "higher_is_better": True},
        {"value": "neg_root_mean_squared_error", "label": "RMSE", "negate": True, "higher_is_better": False},
        {"value": "neg_mean_absolute_error", "label": "MAE", "negate": True, "higher_is_better": False},
        {"value": "explained_variance", "label": "Explained Variance", "negate": False, "higher_is_better": True},
    ],
}


def problem_family(estimator):
    """Classify a fitted estimator as ``"Classification"`` / ``"Regression"`` (else ``None``)."""
    from sklearn.base import is_classifier, is_regressor

    if is_classifier(estimator):
        return "Classification"
    if is_regressor(estimator):
        return "Regression"
    return None


def learning_curve_metric_options(family):
    """Relevant learning-curve metrics for a problem family, filtered to the installed scikit-learn."""
    available = set(get_scorer_names())
    entries = _LEARNING_CURVE_METRICS.get(family or "", [])
    metrics = [
        {key: entry[key] for key in ("value", "label", "higher_is_better")}
        for entry in entries
        if entry["value"] in available
    ]
    return {
        "family": family,
        "metrics": metrics,
        "default": metrics[0]["value"] if metrics else None,
    }


def learning_curve_metric_label(scoring):
    """Display label for a metric: the curated label when known, else the derived scorer label."""
    for entries in _LEARNING_CURVE_METRICS.values():
        for entry in entries:
            if entry["value"] == scoring:
                return entry["label"]
    return _scoring_label(scoring)


def _learning_curve_sign(scoring):
    """Sign flip that turns a ``neg_*`` scorer back into the metric it measures."""
    for entries in _LEARNING_CURVE_METRICS.values():
        for entry in entries:
            if entry["value"] == scoring:
                return -1.0 if entry["negate"] else 1.0
    return 1.0


def compute_learning_curve(bundle, *, task_name, scoring="accuracy", train_partition="train", n_points=5):
    """Cross-validated learning curve for one bundle under a scikit-learn scorer.

    Scores are averaged over the folds of the splitter persisted for the task
    (falling back to scikit-learn's default 3-fold convention), for training-set
    fractions spread evenly between 10% and 100%. Non-finite scores (e.g. a fold
    that could not be fitted) become ``None`` so the UI renders a gap.
    """
    import numpy as np
    from sklearn.model_selection import learning_curve

    if scoring not in set(get_scorer_names()):
        raise bad_request("Unknown scoring metric %r" % (scoring,))
    estimator = getattr(bundle, "model", None)
    if estimator is None or not hasattr(estimator, "fit"):
        raise bad_request("Bundle %r carries no fittable estimator" % bundle.model_version)

    target, _ = get_registered_target(task_name)
    frame = load_partition(task_name, train_partition)
    if target not in frame.columns:
        raise bad_request("Partition %r of task %r has no target column %r" % (train_partition, task_name, target))
    feature_columns = [column for column in frame.columns if column != target]
    X = frame[feature_columns].values
    y = frame[target].values

    cv = get_stored_cv(task_name)
    if cv is None:
        cv = 3
    n_points = min(10, max(2, int(n_points)))
    train_sizes = np.linspace(0.1, 1.0, n_points)

    sizes, train_scores, validation_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, n_jobs=1
    )[:3]

    sign = _learning_curve_sign(scoring)
    n_samples = float(len(y))
    return {
        "model_version": bundle.model_version,
        "train_sizes": [int(size) for size in sizes],
        "train_sizes_fraction": [round(float(size) / n_samples, 4) for size in sizes],
        "train_scores_mean": [finite_or_none(sign * float(np.mean(row))) for row in train_scores],
        "train_scores_std": [finite_or_none(float(np.std(row))) for row in train_scores],
        "validation_scores_mean": [finite_or_none(sign * float(np.mean(row))) for row in validation_scores],
        "validation_scores_std": [finite_or_none(float(np.std(row))) for row in validation_scores],
    }


def load_partition(task_name, partition):
    csv_path = settings.task_dataset_dir(task_name) / (partition + ".csv")
    if not csv_path.exists():
        raise not_found("No dataset partition %r stored for task %r" % (partition, task_name))
    return pd.read_csv(csv_path)


def get_registered_target(task_name):
    with open_tracker(task_name) as tracker:
        metadata = tracker.GetMetadata()
    target = metadata.get("target_name")
    if target is None:
        raise not_found("No dataset registered for task %r" % task_name)
    return target, metadata


def get_stored_cv_spec(task_name):
    """Return the raw JSON-serializable CV spec stored for a task, or ``None``.

    This is the audit/display form (``{"type": "ShuffleSplit", "n_splits": 4}``);
    use :func:`get_stored_cv` to obtain something the library can actually fit with.
    """
    with open_tracker(task_name) as tracker:
        return tracker.GetCVSpec()


def get_stored_cv(task_name):
    """Return the CV splitter persisted for a task (via ``SetCV``), or ``None``.

    The stored spec is rebuilt into a real scikit-learn splitter (or an integer
    fold count) so it can be handed straight to the library. Passing the raw
    spec dict instead is a trap: a dict has no ``split`` method, so
    scikit-learn's ``check_cv`` treats it as an iterable of pre-computed
    ``(train, test)`` folds and fails with a confusing unpack error.
    """
    from SKSurrogate import build_cv

    spec = get_stored_cv_spec(task_name)
    if spec is None:
        return None
    try:
        return build_cv(spec)
    except (ValueError, KeyError, TypeError):
        raise bad_request("Stored CV splitter %r for task %r could not be rebuilt" % (spec, task_name))


def fit_baseline_bundle(
    task_name,
    *,
    estimator="linear_regression",
    train_partition="train",
    validation_partition="validation",
    owner=None,
    run_id=None,
):
    """Fit a simple scikit-learn baseline pipeline and wrap it into a ``ModelBundle``."""
    if estimator not in BASELINE_ESTIMATORS:
        raise bad_request("Unknown estimator %r; choose one of %s" % (estimator, sorted(BASELINE_ESTIMATORS)))
    target, metadata = get_registered_target(task_name)
    train_frame = load_partition(task_name, train_partition)
    train_X = train_frame.drop(columns=[target])
    train_y = train_frame[target]

    model = make_pipeline(StandardScaler(), BASELINE_ESTIMATORS[estimator]())
    model.fit(train_X, train_y)

    metrics = {}
    if validation_partition:
        validation_csv = settings.task_dataset_dir(task_name) / (validation_partition + ".csv")
        if validation_csv.exists():
            validation_frame = pd.read_csv(validation_csv)
            validation_X = validation_frame.drop(columns=[target])
            validation_y = validation_frame[target]
            metrics["score"] = float(model.score(validation_X, validation_y))

    schema = {column: {"dtype": str(train_X[column].dtype)} for column in train_X.columns}
    bundle = ModelBundle(
        model,
        task_name=task_name,
        schema=schema,
        metrics=metrics,
        dataset_fingerprint=metadata.get("dataset_fingerprint"),
        owner=owner,
        run_id=run_id,
    )
    bundle.record_audit_event("training", estimator=estimator, metrics=metrics)
    return bundle


def build_search_param(spec):
    """Convert a JSON param spec (``{"type": "real"|"integer"|"categorical", ...}``) into a structsearch range.

    An optional ``depends_on`` key (``{other_param: [allowed values]}``) is
    stripped from the spec — it does not belong on the range object; the caller
    collects those entries into AML's conditional-parameter map instead.
    """
    if not isinstance(spec, dict):
        raise bad_request(
            "Each parameter spec must be an object with a 'type' (real/integer/categorical); got %r" % (spec,)
        )
    if "depends_on" in spec:
        spec = {key: value for key, value in spec.items() if key != "depends_on"}
    if spec["type"] == "real":
        return Real(spec.get("low"), spec.get("high"))
    if spec["type"] == "integer":
        return Integer(int(spec["low"]), int(spec["high"]))
    if spec["type"] == "categorical":
        return Categorical(spec.get("items") or [])
    raise bad_request("param type must be one of real, integer, categorical")


#: Config key reserved for the stacking directive — it configures the automatic
#: ``StackingEstimator`` wrappers rather than naming a pipeline component, so it
#: is pulled out of the search space before the space is parsed.
STACKING_KEY = "stacking"

#: Accepted ``stacking`` directive fields and the AML constructor argument each maps to.
_STACKING_FIELDS = {
    "res": "stack_res",
    "probs": "stack_probs",
    "decision": "stack_decision",
    "cv": "stack_cv",
    "n_jobs": "stack_n_jobs",
}


def extract_stacking(config):
    """Split an optional ``"stacking"`` directive out of a search-space config.

    Returns ``(config, kwargs)`` where ``kwargs`` are AML constructor arguments
    for the automatic :class:`~SKSurrogate.StackingEstimator` wrappers
    (``stack_res``/``stack_probs``/``stack_decision``/``stack_cv``/``stack_n_jobs``).
    Stacking is applied to intermediate estimators only; the directive therefore
    never becomes a pipeline step and is removed from the search space.
    """
    if not isinstance(config, dict) or STACKING_KEY not in config:
        return config, {}
    directive = config.get(STACKING_KEY) or {}
    if not isinstance(directive, dict):
        raise bad_request("The 'stacking' directive must be an object of res/probs/decision/cv/n_jobs")
    kwargs = {}
    for field, target in _STACKING_FIELDS.items():
        if field not in directive or directive[field] is None:
            continue
        value = directive[field]
        if field in {"res", "probs", "decision"}:
            if not isinstance(value, bool):
                raise bad_request("stacking.%s must be true or false" % field)
        elif field == "cv":
            if not isinstance(value, int) or isinstance(value, bool) or value < 2:
                raise bad_request("stacking.cv must be an integer of at least 2")
        else:  # n_jobs
            if not isinstance(value, int) or isinstance(value, bool):
                raise bad_request("stacking.n_jobs must be an integer or null")
        kwargs[target] = value
    remaining = {key: value for key, value in config.items() if key != STACKING_KEY}
    return remaining, kwargs


def build_conditional_map(config):
    """Collect per-param ``depends_on`` entries into AML's conditional map.

    Each entry becomes ``{param_name: {other_param: [allowed values]}}`` — the
    form :class:`~SKSurrogate.structsearch.SurrogateRandomCV` expects, where a
    parameter is dropped from a candidate whenever its rule does not match.
    """
    conditional = {}
    for params in config.values():
        for name, spec in params.items():
            depends_on = spec.get("depends_on") if isinstance(spec, dict) else None
            if depends_on:
                conditional[name] = {other: list(values) for other, values in depends_on.items()}
    return conditional


def build_forbidden_rules(forbidden):
    """Convert ``[["param", "value"], ...]`` pairs into the dict rules SurrogateRandomCV accepts.

    Values may be any JSON scalar (a string, number, boolean or null) so rules
    like ``[["max_depth", 1]]`` match the parameter's actual value.
    """
    if not forbidden:
        return ()
    rules = []
    for pair in forbidden:
        if len(pair) != 2:
            raise bad_request("Each forbidden rule must be a [parameter, value] pair")
        param, value = pair[0], pair[1]
        if not isinstance(param, str):
            raise bad_request("Forbidden rule parameter names must be strings")
        if isinstance(value, (dict, list, tuple)):
            raise bad_request("Forbidden rule values must be scalars (string, number, boolean or null)")
        rules.append({param: value})
    return tuple(rules)


def fit_experiment_bundle(
    task_name,
    *,
    config,
    checkpoint_dir,
    length=2,
    max_generation=3,
    num_parents=4,
    train_partition="train",
    scoring="accuracy",
    random_state=None,
    owner=None,
    run_id=None,
    surrogate_mode=False,
    surrogate_itrs=None,
    forbidden=None,
):
    """Run an AML/EOA pipeline search and wrap the best fitted pipeline into a ``ModelBundle``.

    The cross-validation splitter persisted for the task (see
    ``PUT /api/datasets/{task}/cv``) is used for the search when present;
    otherwise scikit-learn's default fold convention applies.
    """
    target, metadata = get_registered_target(task_name)
    frame = load_partition(task_name, train_partition)
    feature_columns = [column for column in frame.columns if column != target]
    X = frame[feature_columns].values
    y = frame[target].values

    config, stacking = extract_stacking(config)
    parsed_config = {
        estimator: {name: build_search_param(spec) for name, spec in params.items()}
        for estimator, params in config.items()
    }

    conditional = build_conditional_map(config)
    forbidden_rules = build_forbidden_rules(forbidden)

    cv_spec = get_stored_cv_spec(task_name)
    cv = get_stored_cv(task_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    aml = AML(
        config=parsed_config,
        length=length,
        scoring=scoring,
        check_point=str(checkpoint_dir) + "/",
        random_state=random_state,
        cv=cv if cv is not None else 3,
        forbidden=forbidden_rules,
        conditional=conditional,
        **stacking,
    )
    if surrogate_mode:
        # A single lightweight regressor replaces the default KRR+GPR pair; the
        # iteration budget bounds how many candidate points each surrogate pass tries.
        aml.add_surrogate(LinearRegression(), int(surrogate_itrs) if surrogate_itrs else 20)
    aml.eoa_fit(X, y, max_generation=max_generation, num_parents=num_parents)
    # SurrogateRandomCV refits best_estimator_ on the full dataset (refit=True),
    # so it is ready for scoring/serving without an extra fit here. When every
    # candidate was rejected (e.g. the forbidden rules ban the whole space) it
    # leaves an unfitted estimator and score inf; report that as a clear 400
    # instead of an opaque NotFittedError from aml.score below.
    best_estimator = aml.best_estimator_
    try:
        check_is_fitted(best_estimator)
    except NotFittedError:
        # Deliberately not chained: the job record stores the whole traceback and
        # the NotFittedError detail adds nothing to the actionable message here.
        raise bad_request(
            "No search candidate could be fitted: every candidate was rejected. "
            "Check that the forbidden rules (and any conditional parameters) leave "
            "at least one viable configuration."
        ) from None
    train_score = float(aml.score(X, y))

    schema = {column: {"dtype": str(frame[column].dtype)} for column in feature_columns}
    bundle = ModelBundle(
        best_estimator,
        task_name=task_name,
        schema=schema,
        metrics={"train_score": train_score},
        dataset_fingerprint=metadata.get("dataset_fingerprint"),
        owner=owner,
        run_id=run_id,
    )
    bundle.record_audit_event(
        "experiment",
        scoring=scoring,
        cv=cv_spec,
        length=length,
        max_generation=max_generation,
        num_parents=num_parents,
        train_score=train_score,
        surrogate_mode=surrogate_mode,
        surrogate_itrs=int(surrogate_itrs) if surrogate_itrs else None,
        forbidden=[list(list(rule.items())[0]) for rule in forbidden_rules],
        conditional=conditional,
        stacking=stacking or None,
    )
    # SurrogateRandomCV reports the negated objective (it maximizes internally),
    # so scores are flipped back into the scorer's own units for display; the
    # non-finite scores of invalid candidates become null instead of Infinity.
    evaluation_history = []
    for entry in aml.evaluation_history_:
        row = {"pipeline": list(entry["pipeline"]), "score": finite_or_none(-float(entry["score"]))}
        if entry.get("duration") is not None:
            row["duration"] = round(float(entry["duration"]), 4)
        evaluation_history.append(row)
    top_pipelines = [
        {"pipeline": list(seq), "score": finite_or_none(-float(score))} for seq, (_, score) in aml.get_top(5).items()
    ]
    try:
        pareto = _json_safe_frontier(aml.pareto_frontier())
    except Exception:
        pareto = None
    return bundle, evaluation_history, top_pipelines, pareto


def _json_safe_frontier(records):
    """Reduce Pareto frontier records to the JSON-safe fields a job result can carry."""
    safe = []
    for record in records:
        row = {}
        pipeline = record.get("pipeline")
        if pipeline is not None:
            row["pipeline"] = list(pipeline)
        score = record.get("score")
        if score is not None:
            row["score"] = finite_or_none(float(score))
        duration = record.get("duration")
        if duration is not None:
            row["duration"] = finite_or_none(round(float(duration), 4))
        safe.append(row)
    return safe


def fit_pipeline_bundle(
    task_name,
    *,
    seq,
    config=None,
    checkpoint_dir,
    train_partition="train",
    scoring="accuracy",
    random_state=None,
    owner=None,
    run_id=None,
):
    """Optimize one explicit pipeline structure and wrap the result into a ``ModelBundle``.

    Unlike :func:`fit_experiment_bundle` (which searches over structures), this
    fits a single component sequence: every name in ``seq`` is resolved through
    the task's search-space config, tuned with a bounded surrogate search when
    parameters are given, and wrapped into a bundle. Estimators without any
    parameter spec fall back to plain cross-validated fitting.
    """
    if not seq or any(not isinstance(name, str) for name in seq):
        raise bad_request("seq must be a non-empty list of estimator class names")

    target, metadata = get_registered_target(task_name)
    frame = load_partition(task_name, train_partition)
    feature_columns = [column for column in frame.columns if column != target]
    X = frame[feature_columns].values
    y = frame[target].values

    config, stacking = extract_stacking(config)
    parsed_config = {
        name: {param: build_search_param(spec) for param, spec in (config or {}).get(name, {}).items()}
        for name in seq
    }

    cv_spec = get_stored_cv_spec(task_name)
    cv = get_stored_cv(task_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    aml = AML(
        config=parsed_config,
        length=len(seq),
        scoring=scoring,
        check_point=str(checkpoint_dir) + "/",
        random_state=random_state,
        cv=cv if cv is not None else 3,
        **stacking,
    )
    # optimize_pipeline requires at least one surrogate regressor; a plain
    # linear model with a small budget keeps the single-structure run fast.
    aml.add_surrogate(LinearRegression(), 10)
    pipeline, score = aml.optimize_pipeline(tuple(seq), X, y)
    # Mirror fit_experiment_bundle: an all-trials-rejected optimization leaves an
    # unfitted pipeline and an infinite score, which would otherwise be wrapped
    # into a bundle (and reported as a non-JSON `Infinity`).
    try:
        check_is_fitted(pipeline)
    except NotFittedError:
        # See fit_experiment_bundle: keep the recorded error free of the
        # NotFittedError chain.
        raise bad_request(
            "No parameter configuration for this structure could be fitted: every "
            "candidate was rejected. Check that the search space leaves at least "
            "one viable configuration."
        ) from None
    train_score = finite_or_none(float(score))
    if train_score is None:
        raise bad_request(
            "No parameter configuration for this structure scored finitely: every "
            "candidate was rejected."
        )

    schema = {column: {"dtype": str(frame[column].dtype)} for column in feature_columns}
    bundle = ModelBundle(
        pipeline,
        task_name=task_name,
        schema=schema,
        metrics={"train_score": train_score},
        dataset_fingerprint=metadata.get("dataset_fingerprint"),
        owner=owner,
        run_id=run_id,
    )
    bundle.record_audit_event(
        "pipeline_optimization",
        pipeline=list(seq),
        scoring=scoring,
        cv=cv_spec,
        train_score=float(score),
        stacking=stacking or None,
    )
    return bundle
