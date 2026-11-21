"""Shared model-fitting helpers used by the bundles, experiments, and retraining routers."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import get_scorer, get_scorer_names
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from SKSurrogate import (
    AML,
    Categorical,
    EnsembleError,
    Integer,
    ModelBundle,
    ModelRegistry,
    Real,
    aggregate_forms,
    aggregation_label,
    build_ensemble,
    calibration_report,
    cross_validated_scores,
    fit_ensemble,
    member_forms,
    member_scores,
    nested_ensemble_cv,
    problem_family_of,
    resolve_family,
    save_bundle,
    validate_combination,
    validate_params,
)
from SKSurrogate.DataProcess import FoldLocalDataPreprocessor, frame_tolerant

from . import compute, control, logs
from .columns import coerce_declared_datetimes, drop_ignored, read_ignored_columns
from .config import settings
from .safety import safe_segment
from .deps import bad_request, finite_or_none, not_found, open_tracker, resolve_bundle

BASELINE_ESTIMATORS = {
    "linear_regression": LinearRegression,
    "logistic_regression": LogisticRegression,
    "random_forest_classifier": RandomForestClassifier,
    "random_forest_regressor": RandomForestRegressor,
}

#: Surrogate estimator / sampler options, forwarded to the library's
#: single-surrogate builder. Re-exported from ``SKSurrogate`` so the routers can
#: validate experiment requests against the same lists the search consumes.
from SKSurrogate import SURROGATE_ESTIMATORS as SURROGATE_FAMILIES
from SKSurrogate import SURROGATE_SAMPLERS, build_surrogate


def build_surrogate_models(name, itrs, sampling="BoxSample"):
    """Build the single surrogate configured for an experiment run.

    Thin wrapper over :func:`SKSurrogate.build_surrogate`. Returns the one-entry
    list of ``(regressor, itrs, sampling, solver)`` tuples consumed by
    ``AML.optimize_pipeline``, which runs one ``SurrogateRandomCV`` pass on
    every evaluated pipeline and keeps the best stage.
    """
    return list(build_surrogate(name, itrs, sampling))


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
    # Project-defined metrics that scikit-learn does not ship. MASE is not here on
    # purpose: it needs the in-sample history of the fold, so it is a backtest
    # metric (mltrack.backtest), not a (y_true, y_pred) scorer. ``rmse_scaled``
    # needs the same in-sample scale and ``winkler_score`` needs interval bounds,
    # so neither can be expressed as a scorer signature; all three are documented
    # in docs/forecasting.rst ("MASE is not a scikit-learn scorer").
    forecast = [
        {"value": "neg_smape", "label": "sMAPE (%) (negated)"},
    ]
    options = [
        {"label": label, "options": entries}
        for label, entries in groups.items()
        if entries
    ]
    options.append({"label": "Forecast", "options": forecast})
    return options


#: Curated short-list of metrics offered as learning-curve tabs on the Evaluation
#: page. `negate` marks `neg_*` scikit-learn scorers whose sign is flipped so the
#: curve plots the metric in its natural units (RMSE rather than "… (negated)").
#: Scorers this project defines because scikit-learn has no equivalent. They are
#: accepted by the learning-curve and nested-CV paths but are not in
#: ``get_scorer_names()``. MASE is deliberately absent: it scales by the in-sample
#: seasonal difference, which a ``(y_true, y_pred)`` scorer never sees — compute it
#: from a rolling-origin backtest (``mltrack.backtest``), where the training window
#: is known per origin.
_CUSTOM_SCORERS = ("neg_smape",)


def build_custom_scorer(name):
    """Return ``(scorer, higher_is_better)`` for a project-defined metric name."""
    from sklearn.metrics import make_scorer

    from SKSurrogate.forecast_metrics import smape

    if name == "neg_smape":
        return make_scorer(smape, greater_is_better=False), False
    raise ValueError("Unknown custom scorer %r" % (name,))


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
        {
            "value": "neg_mean_pinball_loss",
            "label": "Pinball loss",
            "negate": True,
            "higher_is_better": False,
            "requires_alpha": True,
        },
        {
            "value": "neg_smape",
            "label": "sMAPE (%)",
            "negate": True,
            "higher_is_better": False,
        },
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
        {
            **{
                key: entry[key]
                for key in ("value", "label", "higher_is_better")
            },
            **({"requires_alpha": True} if entry.get("requires_alpha") else {}),
        }
        for entry in entries
        if entry["value"] in available or entry.get("requires_alpha") or entry["value"] in _CUSTOM_SCORERS
    ]
    return {
        "family": family,
        "metrics": metrics,
        "default": metrics[0]["value"] if metrics else None,
    }


def learning_curve_metric_label(scoring, pinball_alpha=None):
    """Display label for a metric: the curated label when known, else the derived scorer label."""
    for entries in _LEARNING_CURVE_METRICS.values():
        for entry in entries:
            if entry["value"] == scoring:
                if entry.get("requires_alpha") and pinball_alpha is not None:
                    return "%s (q=%g)" % (entry["label"], pinball_alpha)
                return entry["label"]
    return _scoring_label(scoring)


def _learning_curve_sign(scoring):
    """Sign flip that turns a ``neg_*`` scorer back into the metric it measures."""
    for entries in _LEARNING_CURVE_METRICS.values():
        for entry in entries:
            if entry["value"] == scoring:
                return -1.0 if entry["negate"] else 1.0
    return 1.0


def compute_learning_curve(
    bundle,
    *,
    task_name,
    scoring="accuracy",
    train_partition="train",
    n_points=5,
    pinball_alpha=None,
):
    """Cross-validated learning curve for one bundle under a scikit-learn scorer.

    Scores are averaged over the folds of the splitter persisted for the task
    (falling back to scikit-learn's default 3-fold convention), for training-set
    fractions spread evenly between 10% and 100%. Non-finite scores (e.g. a fold
    that could not be fitted) become ``None`` so the UI renders a gap.

    Features are sliced to the bundle's recorded schema and passed as a
    DataFrame, so bundles whose first step is a named-column encoder (the
    explicit cat_cols path) keep working; when a schema column no longer exists
    in the partition this raises a clear per-bundle error instead of failing
    every fold.
    """
    import numpy as np
    from sklearn.model_selection import learning_curve

    if scoring == "neg_mean_pinball_loss":
        if pinball_alpha is None:
            raise bad_request(
                "pinball_alpha is required when scoring is 'neg_mean_pinball_loss'"
            )
        from sklearn.metrics import make_scorer, mean_pinball_loss

        scorer = make_scorer(
            mean_pinball_loss,
            alpha=pinball_alpha,
            greater_is_better=False,
        )
    elif scoring in _CUSTOM_SCORERS:
        if pinball_alpha is not None:
            raise bad_request(
                "pinball_alpha is only valid when scoring is 'neg_mean_pinball_loss'"
            )
        scorer = build_custom_scorer(scoring)[0]
    else:
        if pinball_alpha is not None:
            raise bad_request(
                "pinball_alpha is only valid when scoring is 'neg_mean_pinball_loss'"
            )
        if scoring not in set(get_scorer_names()):
            raise bad_request("Unknown scoring metric %r" % (scoring,))
        scorer = scoring
    estimator = getattr(bundle, "model", None)
    if estimator is None or not hasattr(estimator, "fit"):
        raise bad_request("Bundle %r carries no fittable estimator" % bundle.model_version)

    target, _ = get_registered_target(task_name)
    frame = load_partition(task_name, train_partition)
    if target not in frame.columns:
        raise bad_request("Partition %r of task %r has no target column %r" % (train_partition, task_name, target))

    # Fit on the bundle's own feature schema rather than whatever columns the
    # partition currently carries, and keep it a DataFrame: bundles from the
    # explicit cat_cols path start with named-column encoders that cannot be
    # fitted positionally, so an ndarray input fails every fold. When no schema
    # was recorded (legacy bundles) fall back to every non-target column.
    schema_columns = [column for column in (bundle.schema or {}) if column != target]
    if schema_columns:
        missing = [column for column in schema_columns if column not in frame.columns]
        if missing:
            raise bad_request(
                "Bundle %r expects feature columns absent from partition %r: %s"
                % (bundle.model_version, train_partition, ", ".join(missing))
            )
        X = frame[schema_columns]
    else:
        X = frame[[column for column in frame.columns if column != target]]
    y = frame[target].values

    cv = get_stored_cv(task_name)
    if cv is None:
        cv = 3
    n_points = min(10, max(2, int(n_points)))
    train_sizes = np.linspace(0.1, 1.0, n_points)

    sizes, train_scores, validation_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scorer, train_sizes=train_sizes, n_jobs=1
    )[:3]

    sign = _learning_curve_sign(scoring)
    n_samples = float(len(y))
    return {
        "model_version": bundle.model_version,
        "pinball_alpha": pinball_alpha,
        "train_sizes": [int(size) for size in sizes],
        "train_sizes_fraction": [round(float(size) / n_samples, 4) for size in sizes],
        "train_scores_mean": [finite_or_none(sign * float(np.mean(row))) for row in train_scores],
        "train_scores_std": [finite_or_none(float(np.std(row))) for row in train_scores],
        "validation_scores_mean": [finite_or_none(sign * float(np.mean(row))) for row in validation_scores],
        "validation_scores_std": [finite_or_none(float(np.std(row))) for row in validation_scores],
    }


def load_partition(task_name, partition):
    """Load a stored partition without the dataset's ignored feature columns.

    This is the single entry point for every training/analysis path, so declaring
    a column "ignored" on the Datasets page removes it from experiments, pipeline
    optimization, baselines, explanations and bundle previews alike.
    """
    csv_path = settings.task_dataset_dir(task_name) / (safe_segment(partition, "partition") + ".csv")
    if not csv_path.exists():
        raise not_found("No dataset partition %r stored for task %r" % (partition, task_name))
    frame = pd.read_csv(csv_path)
    # A datetime dtype does not survive CSV storage: restore the columns the
    # dataset declares as datetime64 (deduced or overridden) before anything
    # inspects dtypes, otherwise they are one-hot encoded as categories.
    frame = coerce_declared_datetimes(frame, task_name=task_name)
    return drop_ignored(frame, read_ignored_columns(task_name))


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


def _reproducibility_record(
    task_name,
    metadata,
    frame,
    target,
    train_partition,
    *,
    configuration,
    evaluation_protocol,
    transformations,
):
    partitions = metadata.get("dataset_partition_history", {})
    train_data = partitions.get(train_partition, {})
    environment_keys = (
        "python_version",
        "numpy_version",
        "pandas_version",
        "scikit_learn_version",
        "sksurrogate_version",
    )
    return {
        "schema_version": 1,
        "task_name": task_name,
        "data": {
            "train_partition": train_partition,
            "train_fingerprint": train_data.get("fingerprint"),
            "rows_used": int(len(frame)),
            "features": [column for column in frame.columns if column != target],
            "target": {
                "column": target,
                "construction": "not recorded; target is used as registered",
            },
            "excluded_observations": {
                "explicit_api_exclusions": 0,
                "upstream_exclusions": "not recorded",
            },
        },
        "configuration": configuration,
        "transformations": transformations,
        "evaluation_protocol": evaluation_protocol,
        "environment": {
            key: metadata[key] for key in environment_keys if key in metadata
        },
        "known_gaps": [
            "Target construction and upstream row exclusions are not recorded.",
            "Metrics without a repeated-CV evaluation have no uncertainty estimate.",
        ],
    }


def _baseline_needs_preprocessing(frame):
    """Whether a baseline pipeline must start with ``FoldLocalDataPreprocessor``.

    The historical baseline was a bare ``StandardScaler`` in front of the
    estimator, which only fits numeric matrices: on a partition carrying an
    object/category/string or datetime feature column it died inside
    ``check_array`` and surfaced as an opaque 500 from ``train-baseline``.
    Any column that would not land in ``FoldLocalDataPreprocessor``'s plain
    numeric branch (booleans, categoricals, datetimes, declared text) is a
    reason to switch the baseline over to the same fold-local preprocessor
    the experiment path installs — which also brings median imputation for
    numerics. Purely numeric partitions keep their historical scaler-first
    pipeline so existing bundles stay shape-compatible.
    """
    for column in frame.columns:
        dtype = frame[column].dtype
        if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
            return True
    return False


def _temporal_baseline_bundle(
    tracker,
    task_name,
    target,
    metadata,
    train_frame,
    estimator,
    estimator_cls,
    *,
    train_partition="train",
    horizon=None,
    features=None,
    strategy="direct",
    seasonality=1,
    n_splits=5,
    owner=None,
    run_id=None,
):
    """Fit a forecasting baseline for a task that declares a temporal spec.

    A tabular baseline bundle cannot answer ``predict_forecast``: it needs a
    feature builder that turns raw history into a design matrix. This path builds
    that pipeline, stores the serving geometry (``TemporalServeState``) on the
    bundle, and — plan P4.2 — always measures the naive baselines on a
    rolling-origin backtest so the bundle record carries ``baseline_scores``
    (seasonal naive included). Without that comparison a forecast metric cannot
    be read honestly.
    """
    from sklearn.pipeline import Pipeline

    from SKSurrogate import ForecastFeatureBuilder, TemporalServeState, TemporalSpec
    from SKSurrogate.forecast_metrics import BASELINE_METHODS

    spec = TemporalSpec.from_dict(tracker.GetTemporalSpec())
    design_horizon = int(spec.horizon if horizon is None else horizon)
    if design_horizon < 1:
        raise bad_request("horizon must be >= 1")
    design_spec = TemporalSpec(**{**spec.to_dict(), "horizon": design_horizon})
    # ``train_frame`` arrives with the ignored columns dropped, and a temporal
    # declaration puts the time/series columns on that list (decision D4). The
    # builder needs them, so reload the partition and drop only *genuine*
    # ignored features (plan P4.2 path, same rule as ``mltrack.backtest``).
    train_frame = tracker.get_dataframe(train_partition)
    ignored = set(read_ignored_columns(task_name)) - set(design_spec.temporal_columns)
    ignored = {column for column in ignored if column in train_frame.columns}
    if ignored:
        train_frame = train_frame.drop(columns=sorted(ignored))
    builder_kwargs = dict(features or {})
    builder_kwargs.setdefault("lags", [design_horizon])
    builder = ForecastFeatureBuilder(
        spec=design_spec, target=target, **builder_kwargs
    )
    builder.fit(train_frame)
    design = builder.build(
        train_frame, target_horizon=1 if strategy == "recursive" else None
    )
    if not len(design.X):
        raise bad_request(
            "the feature configuration leaves no usable rows in partition %r"
            % train_partition
        )
    model = estimator_cls().fit(design.X, design.y)
    pipeline = Pipeline([("features", builder), ("model", model)])
    state = TemporalServeState.from_parts(
        design_spec, target, builder, strategy=strategy
    )

    metrics: dict = {}
    baseline_scores: dict = {}
    backtest_summary: dict = {"available": False}
    try:
        result = tracker.backtest(
            estimator_cls(),
            horizon=design_horizon,
            n_splits=n_splits,
            refit="first",
            strategy=strategy,
            train_partition=train_partition,
            feature_kwargs=builder_kwargs,
            exclude_columns=sorted(ignored),
            seasonality=seasonality,
            baselines=BASELINE_METHODS,
            store=False,
        )
    except ValueError as exc:
        # Too little history for an honest walk-forward: say so on the record
        # instead of reporting an in-sample number as if it were a forecast.
        backtest_summary = {"available": False, "reason": str(exc)}
    else:
        pooled = result.get("pooled") or {}
        metrics = {
            key: pooled[key]
            for key in ("mae", "rmse", "mase", "smape")
            if pooled.get(key) is not None
        }
        baseline_scores = dict(result.get("baselines") or {})
        backtest_summary = {
            "available": True,
            "artifact_key": result.get("artifact_key"),
            "config": result.get("config"),
        }

    run_record = _reproducibility_record(
        task_name,
        metadata,
        train_frame,
        target,
        train_partition,
        configuration={
            "kind": "temporal-baseline",
            "estimator": estimator,
            "estimator_parameters": estimator_cls().get_params(deep=False),
            "strategy": strategy,
            "horizon": design_horizon,
            "features": builder_kwargs,
            "seasonality": seasonality,
        },
        transformations=[
            {
                "class": "%s.%s"
                % (ForecastFeatureBuilder.__module__, ForecastFeatureBuilder.__qualname__),
                "parameters": builder.to_dict(),
            },
            {
                "class": "%s.%s" % (estimator_cls.__module__, estimator_cls.__qualname__),
                "parameters": estimator_cls().get_params(deep=False),
            },
        ],
        evaluation_protocol={
            "selection": "none; fixed baseline forecaster",
            "metric": "rolling-origin backtest",
            "metric_scope": (
                "walk-forward origins on partition %r" % train_partition
                if backtest_summary["available"]
                else "unavailable: %s" % backtest_summary.get("reason")
            ),
        },
    )
    bundle = ModelBundle(
        pipeline,
        task_name=task_name,
        schema={
            column: {"dtype": str(train_frame[column].dtype)}
            for column in train_frame.columns
        },
        metrics=metrics,
        metadata={
            "reproducibility": run_record,
            "baseline_scores": baseline_scores,
            "backtest": backtest_summary,
        },
        dataset_fingerprint=run_record["data"]["train_fingerprint"],
        temporal=state.to_dict(),
        dependencies={},
        owner=owner,
        run_id=run_id,
    )
    bundle.record_audit_event(
        "training",
        estimator=estimator,
        metrics=metrics,
        baseline_scores=baseline_scores,
    )
    return bundle


def fit_baseline_bundle(
    task_name,
    *,
    estimator="linear_regression",
    train_partition="train",
    validation_partition="validation",
    owner=None,
    run_id=None,
    horizon=None,
    features=None,
    strategy="direct",
    seasonality=1,
    n_splits=5,
):
    """Fit a simple scikit-learn baseline pipeline and wrap it into a ``ModelBundle``.

    Tasks that declare a temporal spec take the forecasting path instead
    (``_temporal_baseline_bundle``): the baseline is fitted on a generated design
    matrix, the bundle carries the serving geometry, and its record stores the
    naive baselines measured on the same rolling origins (plan P4.2).
    """
    if estimator not in BASELINE_ESTIMATORS:
        raise bad_request("Unknown estimator %r; choose one of %s" % (estimator, sorted(BASELINE_ESTIMATORS)))
    target, metadata = get_registered_target(task_name)
    train_frame = load_partition(task_name, train_partition)
    estimator_cls = BASELINE_ESTIMATORS[estimator]

    with open_tracker(task_name) as tracker:
        if tracker.GetTemporalSpec() is not None:
            return _temporal_baseline_bundle(
                tracker,
                task_name,
                target,
                metadata,
                train_frame,
                estimator,
                estimator_cls,
                train_partition=train_partition,
                horizon=horizon,
                features=features,
                strategy=strategy,
                seasonality=seasonality,
                n_splits=n_splits,
                owner=owner,
                run_id=run_id,
            )

    train_X = train_frame.drop(columns=[target])
    train_y = train_frame[target]

    preprocessed = _baseline_needs_preprocessing(train_X)
    if preprocessed:
        # Mixed-type partition: the same fold-local encoder the experiment
        # path installs (median imputation + scaling for numerics, one-hot
        # with ``handle_unknown="ignore"`` for categoricals). A bare
        # StandardScaler cannot fit this frame at all.
        model = make_pipeline(FoldLocalDataPreprocessor(scale_numeric=True), estimator_cls())
    else:
        model = make_pipeline(StandardScaler(), estimator_cls())
    model.fit(train_X, train_y)

    metrics = {}
    if validation_partition:
        validation_csv = settings.task_dataset_dir(task_name) / (safe_segment(validation_partition, "partition") + ".csv")
        if validation_csv.exists():
            # load_partition — not a raw read — so columns ignored on the
            # Datasets page are excluded from the score as well as the fit.
            validation_frame = load_partition(task_name, validation_partition)
            validation_X = validation_frame.drop(columns=[target])
            validation_y = validation_frame[target]
            metrics["score"] = float(model.score(validation_X, validation_y))

    schema = {column: {"dtype": str(train_X[column].dtype)} for column in train_X.columns}
    if preprocessed:
        transformation_entries = [
            {
                "class": "%s.%s"
                % (FoldLocalDataPreprocessor.__module__, FoldLocalDataPreprocessor.__qualname__),
                "parameters": {"scale_numeric": True},
            }
        ]
    else:
        transformation_entries = [{"class": "sklearn.preprocessing.StandardScaler", "parameters": {}}]
    transformation_entries.append(
        {
            "class": "%s.%s" % (estimator_cls.__module__, estimator_cls.__qualname__),
            "parameters": estimator_cls().get_params(deep=False),
        }
    )

    run_record = _reproducibility_record(
        task_name,
        metadata,
        train_frame,
        target,
        train_partition,
        configuration={
            "kind": "baseline",
            "estimator": estimator,
            "estimator_parameters": BASELINE_ESTIMATORS[estimator]().get_params(
                deep=False
            ),
            "validation_partition": validation_partition if "score" in metrics else None,
            "validation_partition_fingerprint": (
                metadata.get("dataset_partition_history", {})
                .get(validation_partition, {})
                .get("fingerprint")
                if "score" in metrics
                else None
            ),
        },
        transformations=transformation_entries,
        evaluation_protocol={
            "selection": "none; fixed baseline estimator",
            "metric": "estimator.score",
            "metric_scope": "validation partition" if "score" in metrics else None,
        },
    )
    bundle = _new_bundle(model, metrics, task_name, schema, run_record, owner, run_id)
    bundle.record_audit_event("training", estimator=estimator, metrics=metrics)
    return bundle


def _new_bundle(model, metrics, task_name, schema, run_record, owner, run_id, label=None):
    """Create a ``ModelBundle`` carrying the reproducibility record shared by all training entry points.

    Bundles are frame-tolerant: an experiment fitted on a mixed-type partition
    starts with ``FoldLocalDataPreprocessor``, which only accepts DataFrames, so
    the pipeline is prefixed with ``ArrayAsFrame`` to keep the pre-existing
    contract that explanations, learning curves and legacy trackers can predict
    from positional arrays.

    ``label`` is the experiment's optional human name. It is stored in the
    bundle metadata (the machine ``model_version`` stays the identity), so a
    friendly name survives with the artifact and is visible in bundle summaries
    even when the registry has no alias for the version.
    """
    metadata = {"reproducibility": run_record}
    if label:
        metadata["label"] = label
    return ModelBundle(
        frame_tolerant(model, columns=list(schema)),
        task_name=task_name,
        schema=schema,
        metrics=metrics,
        metadata=metadata,
        dataset_fingerprint=run_record["data"]["train_fingerprint"],
        owner=owner,
        run_id=run_id,
    )


def categorical_feature_columns(frame, feature_columns):
    """Non-numeric feature columns of ``frame``.

    A preset search installs ``FoldLocalDataPreprocessor`` and encodes whatever
    it finds. An explicit search space does not: ``AML`` only adds an encoder
    when ``cat_cols`` is given, so a config-based run on a dataset with a text
    column failed with ``could not convert string to float``. These are the
    columns to hand to ``cat_cols``.
    """
    return [column for column in feature_columns if not pd.api.types.is_numeric_dtype(frame[column])]


def _transformation_entry(step):
    """Describe one pipeline step for the reproducibility record.

    ``Pipeline.steps`` yields ``(name, estimator)`` pairs; recording the pair
    itself wrote ``builtins.tuple`` for every step of every experiment bundle.
    """
    return {"class": "%s.%s" % (step.__class__.__module__, step.__class__.__qualname__)}


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


def _require_feature_columns(task_name, train_partition, target, frame):
    """The partition's feature columns; a featureless partition is a 400.

    A temporal task legitimately has none of its own columns left — the time and
    series columns are excluded by the temporal contract — which is why
    :func:`fit_experiment_bundle` builds a design matrix for it instead. A
    *tabular* task with none is a data-configuration error, and AML would only
    report it as every candidate failing with the opaque ``at least one array or
    dtype is required``.
    """
    columns = [column for column in frame.columns if column != target]
    if columns:
        return columns
    ignored = sorted(read_ignored_columns(task_name))
    excluded = (
        "the ignored column(s) %s are" % ", ".join(repr(name) for name in ignored)
        if ignored
        else "no other column exists"
    )
    raise bad_request(
        "Partition %r of task %r has no feature columns: %r is the target and %s "
        "excluded from the feature matrix. Re-enable a feature column on the "
        "Datasets page, or declare the task temporal if time is its only driver."
        % (train_partition, task_name, target, excluded)
    )


def _temporal_design(task_name, target, train_partition):
    """The generated one-step forecast design for a temporal task.

    Returns ``(raw_frame, design, builder, spec)``, or ``None`` when the task
    declares no temporal spec. The raw partition keeps its time/series columns
    (the builder needs them) and drops only *genuine* ignores — the same rule
    ``_temporal_baseline_bundle`` follows.

    The labels are one step ahead while the declared horizon still governs the
    minimum feature shift, so the search scores one-step predictions and the
    resulting bundle serves recursively (its own predictions are fed back as
    history). Multi-step accuracy is the rolling-origin backtest's job.
    """
    from SKSurrogate import ForecastFeatureBuilder, TemporalSpec

    with open_tracker(task_name) as tracker:
        state = tracker.GetTemporalSpec()
        if state is None:
            return None
        raw = tracker.get_dataframe(train_partition)
    spec = TemporalSpec.from_dict(state)
    ignored = set(read_ignored_columns(task_name)) - set(spec.temporal_columns)
    present = {column for column in ignored if column in raw.columns}
    if present:
        raw = raw.drop(columns=sorted(present))
    builder = ForecastFeatureBuilder(spec=spec, target=target, lags=[spec.horizon])
    builder.fit(raw)
    design = builder.build(raw, target_horizon=1)
    if not len(design.X):
        raise bad_request(
            "Partition %r of task %r has no usable forecast row: the declared "
            "horizon %d and its lag need more history than the partition holds."
            % (train_partition, task_name, spec.horizon)
        )
    return raw, design, builder, spec


def temporal_design_inputs(task_name, target, partition):
    """``(X, y)`` of the one-step forecast design for a temporal task.

    The same matrix the experiment search, the backtest and the explanation
    endpoints consume. The raw partition keeps its time/series columns (the
    builder needs them); only genuine ignores are dropped — the temporal contract
    already excludes the time columns from every *feature* matrix, so they must
    stay present here.
    """
    from SKSurrogate import temporal_columns

    from .features import temporal_design_xy

    with open_tracker(task_name) as tracker:
        spec = tracker.GetTemporalSpec()
        raw = tracker.get_dataframe(partition)
    ignored = set(read_ignored_columns(task_name)) - set(temporal_columns(spec))
    present = {column for column in ignored if column in raw.columns}
    if present:
        raw = raw.drop(columns=sorted(present))
    projection = temporal_design_xy(raw, spec, target)
    frame = pd.DataFrame(projection.X)
    frame.columns = [str(column) for column in projection.columns]
    return frame, np.asarray(projection.y, dtype=float)


def unwrap_temporal_estimator(bundle):
    """The bare estimator of a temporal bundle: builder removed, frame-tolerant.

    A forecast bundle's pipeline starts with ``ForecastFeatureBuilder``, which
    expects raw history. Consumers that build their own design (backtest,
    explanations, nested CV) refit the estimator behind it instead; the
    ``ArrayAsFrame`` adapter keeps a preset pipeline's
    ``FoldLocalDataPreprocessor`` happy with positional arrays.
    """
    from SKSurrogate import ForecastFeatureBuilder
    from SKSurrogate.DataProcess import frame_tolerant

    steps = getattr(bundle.model, "steps", None) or []
    if any(isinstance(step, ForecastFeatureBuilder) for _, step in steps):
        return frame_tolerant(steps[-1][1])
    return bundle.model


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
    surrogate_estimator="gpr",
    surrogate_itrs=30,
    surrogate_sampling="BoxSample",
    surrogate_min_random_evals=10,
    forbidden=None,
    preset=None,
    strategy="eoa",
    resume=False,
    n_jobs=None,
    label=None,
):
    """Run an AML/EOA pipeline search and wrap the best fitted pipeline into a ``ModelBundle``.

    The cross-validation splitter persisted for the task (see
    ``PUT /api/datasets/{task}/cv``) is used for the search when present;
    otherwise scikit-learn's default fold convention applies.
    """
    target, metadata = get_registered_target(task_name)
    # A temporal task's partition has no raw feature column of its own (the time
    # column is excluded by contract), so the search runs on the generated
    # one-step forecast design instead — the same design sensitivity, baselines
    # and the backtest use.
    temporal_builder: Any = None
    temporal_spec: Any = None
    design: Any = None
    temporal = _temporal_design(task_name, target, train_partition)
    if temporal is not None:
        frame, design, temporal_builder, temporal_spec = temporal
        feature_columns = [column for column in frame.columns if column != target]
        X = design.X
        y = design.y.to_numpy(dtype=float)
    else:
        frame = load_partition(task_name, train_partition)
        feature_columns = _require_feature_columns(task_name, train_partition, target, frame)
        # AML installs FoldLocalDataPreprocessor only when it is given a DataFrame,
        # and that is what makes a mixed-type partition searchable: the encoder is
        # then fitted inside every CV fold instead of leaking across the split.
        # Handing it `X.values` (an object array) skipped preprocessing entirely and
        # every candidate died on the first categorical value.
        X = frame[feature_columns]
        y = frame[target].values

    if preset is not None and config:
        raise bad_request("Send either a preset or a config, not both")
    if preset is None and not config:
        raise bad_request("A search-space config or a preset is required")
    if strategy not in ("eoa", "exhaustive"):
        raise bad_request("strategy must be 'eoa' or 'exhaustive'")
    requested_config = config
    config, stacking = extract_stacking(config or {})
    parsed_config = {
        estimator: {name: build_search_param(spec) for name, spec in params.items()}
        for estimator, params in config.items()
    }

    conditional = build_conditional_map(config)
    forbidden_rules = build_forbidden_rules(forbidden)
    # Fold-level parallelism is shared by every job in this process (joblib
    # keeps one loky pool), so the width comes from the compute budget rather
    # than AML's own n_jobs=-1 default. See api/compute.py.
    n_jobs = int(compute.current()["job_threads"] if n_jobs is None else n_jobs)

    # A preset encodes mixed-type data itself (fold-locally, see
    # _new_bundle); an explicit search space needs the columns named or every
    # candidate dies on the first string value. A temporal design is already
    # numeric, so it never needs an encoder.
    cat_cols = (
        categorical_feature_columns(frame, feature_columns)
        if preset is None and temporal is None
        else []
    )
    cv_spec = get_stored_cv_spec(task_name)
    cv = get_stored_cv(task_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # The single chosen surrogate is applied to every candidate pipeline the
    # search evaluates (one SurrogateRandomCV pass, best stage kept); the
    # iteration budget bounds how many candidate points the pass tries and
    # min_random_evals sets how many random initial points are sampled before
    # the surrogate is exploited. Passing surrogates explicitly also suppresses
    # AML's implicit default KRR+GPR pair, so the estimator is the user's pick.
    surrogates = build_surrogate_models(surrogate_estimator, surrogate_itrs, surrogate_sampling)
    aml = AML(
        config=parsed_config if preset is None else None,
        preset=preset,
        cat_cols=cat_cols or None,
        resume=resume,
        length=length,
        scoring=scoring,
        check_point=str(checkpoint_dir) + "/",
        random_state=random_state,
        cv=cv if cv is not None else 3,
        surrogates=surrogates,
        min_random_evals=surrogate_min_random_evals,
        forbidden=forbidden_rules,
        conditional=conditional,
        callbacks=[logs.trial_callback],
        n_jobs=n_jobs,
        # Cooperative pause/cancel: checked between candidate evaluations.
        stop_check=control.stop_requested,
        **stacking,
    )
    if strategy == "eoa":
        # Evolutionary search over the space of all possible pipelines, instead
        # of exhaustively enumerating the whole space.
        aml.fit(X, y, strategy="eoa", max_generations=max_generation, num_parents=num_parents)
    else:
        # "exhaustive": enumerate every admissible pipeline, surrogate-tuned.
        aml.fit(X, y, strategy="exhaustive")
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

    if temporal is not None:
        # A forecast bundle is fed the *raw* history frame; the builder inside its
        # pipeline is what turns that into the design, so the schema describes the
        # partition columns (time/series included), not the generated features.
        schema = {column: {"dtype": str(frame[column].dtype)} for column in frame.columns}
    else:
        schema = {column: {"dtype": str(frame[column].dtype)} for column in feature_columns}
    transformations: list[dict] = [
        _transformation_entry(step) for _, step in getattr(best_estimator, "steps", [])
    ]
    if temporal is not None:
        transformations.insert(
            0,
            {
                "class": "%s.%s"
                % (temporal_builder.__class__.__module__, temporal_builder.__class__.__qualname__),
                "parameters": temporal_builder.to_dict(),
            },
        )
    temporal_configuration = (
        {
            "feature_source": "temporal_design",
            "temporal_spec": temporal_spec.to_dict(),
            "feature_builder": temporal_builder.to_dict(),
            "serving_strategy": "recursive",
        }
        if temporal is not None
        else {}
    )
    run_record = _reproducibility_record(
        task_name,
        metadata,
        frame,
        target,
        train_partition,
        configuration={
            "kind": "automl_experiment",
            "preset": preset,
            "cat_cols": cat_cols or None,
            "strategy": strategy,
            "search_space": requested_config,
            "length": length,
            "max_generation": max_generation,
            "num_parents": num_parents,
            "random_state": random_state,
            "surrogate": {
                "estimator": surrogate_estimator,
                "itrs": int(surrogate_itrs),
                "sampling": surrogate_sampling,
                "min_random_evals": surrogate_min_random_evals,
            },
            "forbidden": [list(list(rule.items())[0]) for rule in forbidden_rules],
            "conditional": conditional,
            "stacking": stacking or None,
            "n_jobs": n_jobs,
            **temporal_configuration,
        },
        transformations=transformations,
        evaluation_protocol={
            "scoring": scoring,
            "cv": cv_spec if cv_spec is not None else {"folds": 3, "source": "default"},
            "metric_scope": (
                "cross-validation search score on the generated one-step forecast "
                "design; multi-step scores come from the rolling-origin backtest"
                if temporal is not None
                else "cross-validation search score; reported train_score is refit training score"
            ),
        },
    )
    if temporal is not None:
        # The record's feature list and row count must name what the search
        # actually saw: the generated design, not the raw partition with its
        # time column and untrimmed tail.
        run_record["data"]["features"] = list(design.X.columns)
        run_record["data"]["rows_used"] = int(len(design.X))
    if temporal is not None:
        from SKSurrogate import TemporalServeState

        serve_state = TemporalServeState.from_parts(
            temporal_spec, target, temporal_builder, strategy="recursive"
        )
        bundle_metadata = {"reproducibility": run_record}
        if label:
            bundle_metadata["label"] = label
        bundle = ModelBundle(
            Pipeline([("features", temporal_builder), ("model", best_estimator)]),
            task_name=task_name,
            schema=schema,
            metrics={"train_score": train_score},
            metadata=bundle_metadata,
            dataset_fingerprint=run_record["data"]["train_fingerprint"],
            temporal=serve_state.to_dict(),
            owner=owner,
            run_id=run_id,
        )
    else:
        bundle = _new_bundle(best_estimator, {"train_score": train_score}, task_name, schema, run_record, owner, run_id, label=label)
    bundle.record_audit_event(
        "experiment",
        scoring=scoring,
        cv=cv_spec,
        length=length,
        max_generation=max_generation,
        num_parents=num_parents,
        train_score=train_score,
        surrogate_estimator=surrogate_estimator,
        surrogate_itrs=int(surrogate_itrs),
        surrogate_sampling=surrogate_sampling,
        surrogate_min_random_evals=surrogate_min_random_evals,
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
    n_jobs=None,
    resume=False,
    label=None,
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
    if not feature_columns:
        with open_tracker(task_name) as tracker:
            is_temporal = tracker.GetTemporalSpec() is not None
        if is_temporal:
            raise bad_request(
                "Task %r is temporal: pipeline optimization searches the partition's own "
                "feature columns, and its time/series columns are excluded by contract. "
                "Use the Experiments page (the search there runs on the generated "
                "one-step forecast design) or train a forecasting baseline instead."
                % task_name
            )
        feature_columns = _require_feature_columns(task_name, train_partition, target, frame)
    # Same reason as fit_experiment_bundle: pass the frame, not `.values`.
    X = frame[feature_columns]
    y = frame[target].values

    config, stacking = extract_stacking(config)
    parsed_config = {
        name: {param: build_search_param(spec) for param, spec in (config or {}).get(name, {}).items()}
        for name in seq
    }

    cv_spec = get_stored_cv_spec(task_name)
    cv = get_stored_cv(task_name)
    n_jobs = int(compute.current()["job_threads"] if n_jobs is None else n_jobs)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    aml = AML(
        config=parsed_config,
        cat_cols=categorical_feature_columns(frame, feature_columns) or None,
        length=len(seq),
        n_jobs=n_jobs,
        resume=resume,
        scoring=scoring,
        check_point=str(checkpoint_dir) + "/",
        random_state=random_state,
        cv=cv if cv is not None else 3,
        callbacks=[logs.trial_callback],
        stop_check=control.stop_requested,
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
    run_record = _reproducibility_record(
        task_name,
        metadata,
        frame,
        target,
        train_partition,
        configuration={
            "kind": "pipeline_optimization",
            "pipeline": list(seq),
            "search_space": config or {},
            "length": len(seq),
            "random_state": random_state,
            "surrogate": {
                "estimator": "sklearn.linear_model.LinearRegression",
                "iterations": 10,
            },
            "stacking": stacking or None,
            "n_jobs": n_jobs,
        },
        transformations=[_transformation_entry(step) for _, step in getattr(pipeline, "steps", [])],
        evaluation_protocol={
            "scoring": scoring,
            "cv": cv_spec if cv_spec is not None else {"folds": 3, "source": "default"},
            "metric_scope": "cross-validation search score; reported train_score is refit training score",
        },
    )
    bundle = _new_bundle(pipeline, {"train_score": train_score}, task_name, schema, run_record, owner, run_id, label=label)
    bundle.record_audit_event(
        "pipeline_optimization",
        pipeline=list(seq),
        scoring=scoring,
        cv=cv_spec,
        train_score=float(score),
        stacking=stacking or None,
    )
    return bundle


# --------------------------------------------------------------------------- #
# Ensembles: combining several trained bundles of one task                    #
# --------------------------------------------------------------------------- #

#: Cap on how many models one ensemble may combine. Each extra member costs one
#: out-of-fold comparison fit per fold and one refit per bagging replica, so an
#: unbounded selection would turn a page action into an hour-long job.
MAX_ENSEMBLE_MEMBERS = 12

#: Default metric per problem family, matching the Experiments presets.
DEFAULT_ENSEMBLE_SCORING = {
    "Classification": "accuracy",
    "Regression": "neg_root_mean_squared_error",
}

#: Folds used for the member-versus-ensemble comparison when the task has no
#: stored splitter. Fixed so the comparison rows are computed on identical folds.
DEFAULT_ENSEMBLE_FOLDS = 5

ENSEMBLE_CAVEATS = [
    "Member models were fitted, and where searched also tuned, on this task's data, so "
    "the member and ensemble cross-validated scores are jointly optimistic: a member "
    "selected because it scored well here is scored on the same rows again. The "
    "nested-CV evaluation is the estimate that accounts for that selection.",
    "cv_score is out-of-fold under the splitter named in the record; train_score is the "
    "score of the refitted ensemble on all training rows and is optimistic by construction.",
    "Every member is refitted inside each comparison fold (scikit-learn clones the "
    "estimator), so no member's fitted state reaches the score.",
]

#: Added only when the corresponding capability was requested, so the caveat list a caller
#: reads always describes the ensemble in front of them.
WEIGHT_OPTIMIZATION_CAVEAT = (
    "Member weights were fitted on the same out-of-fold member output that this "
    "ensemble's cv_score is computed on, so cv_score is optimistic for a *weighted* "
    "ensemble: it does not pay for the weight search. Request the honest nested estimate "
    "to price it. Measured across ten configurations (200-560 rows, 2-3 members, "
    "least-squares and greedy searches, 3-5 scoring folds) the gap ran from 0.0000 to "
    "0.0233; it is exactly zero whenever the search collapses onto one member, because "
    "then the ensemble *is* that member and both protocols score the same estimator."
)

CALIBRATION_CAVEAT = (
    "Member probabilities were calibrated inside every fold and the reported Brier and "
    "log-loss columns are out-of-fold before/after values. Calibration changes "
    "probabilities, not the members' decisions, so an accuracy-style metric may barely "
    "move while log loss falls sharply on an overconfident member."
)


def load_member_bundle(task_name, model_version):
    """Load one member bundle by version id or alias, and say where it lives.

    Resolution order is the unregistered bundle directory first, then the registry,
    matching every other consuming endpoint; ``source`` distinguishes the two for the
    UI without a second round trip.
    """
    bundle = resolve_bundle(
        task_name, model_version=model_version, strict_dependencies=False
    )
    local_path = settings.task_bundles_dir(task_name) / (
        safe_segment(model_version, "model version") + ".bundle"
    )
    return bundle, ("bundle" if local_path.exists() else "registry")


def resolve_ensemble_members(task_name, versions):
    """Resolve member ids to bundles, refusing empty, over-long and duplicate selections.

    Duplicates are refused rather than collapsed: the member order is part of the
    request (voting weights are positional), so silently deduplicating would score a
    different ensemble than the one asked for.
    """
    versions = list(versions or [])
    if not versions:
        raise bad_request("At least one member model version is required.")
    cleaned = []
    for version in versions:
        if not isinstance(version, str) or not version.strip():
            raise bad_request("Member model versions must be non-empty strings.")
        version = version.strip()
        if version in cleaned:
            raise bad_request("Member model version %r was selected twice." % version)
        cleaned.append(version)
    if len(cleaned) > MAX_ENSEMBLE_MEMBERS:
        raise bad_request(
            "At most %d members can be combined in one ensemble; got %d."
            % (MAX_ENSEMBLE_MEMBERS, len(cleaned))
        )
    resolved = []
    for version in cleaned:
        bundle, source = load_member_bundle(task_name, version)
        resolved.append(
            {
                "model_version": version,
                "bundle": bundle,
                "source": source,
                "label": (bundle.metadata or {}).get("label"),
            }
        )
    return resolved


def ensemble_plan(
    task_name,
    *,
    members,
    method,
    params=None,
    calibration=None,
    weight_optimization=None,
    columns=None,
    random_state=None,
    n_jobs=None,
):
    """Validate an ensemble request and return ``(resolved, family, cleaned_params)``.

    Nothing is fitted here, so a submit-time call answers 400/404 straight away instead of
    letting a bad member set, parameter, calibration or weight request fail inside the
    background job: the per-member forms are *built* (which is what checks soft-voting
    probability support, boosting's sample-weight support, calibration eligibility against
    the members' family, and the stacking meta-learner) but never fitted.
    """
    resolved = resolve_ensemble_members(task_name, members)
    pairs = [(entry["model_version"], entry["bundle"].model) for entry in resolved]
    try:
        validate_combination(method, params, calibration, weight_optimization)
        forms, family, cleaned = member_forms(
            method,
            pairs,
            params=params,
            calibration=calibration,
            columns=columns,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        # The aggregation step is where soft voting's probability requirement and the
        # stacking meta-learner are resolved; building it (never fitting it) is what makes
        # those rejections submit-time errors instead of job failures.
        aggregate_forms(method, forms, family, cleaned, n_jobs)
    except EnsembleError as exc:
        raise bad_request(str(exc)) from None
    return resolved, family, cleaned


def fit_ensemble_bundle(
    task_name,
    *,
    members,
    method,
    params=None,
    train_partition="train",
    scoring=None,
    random_state=None,
    owner=None,
    run_id=None,
    label=None,
    n_jobs=None,
    register=False,
    calibration=None,
    weight_optimization=None,
    honest_cv=None,
):
    """Combine several trained bundles of one task into an ensemble bundle.

    The result is an ordinary :class:`~SKSurrogate.ModelBundle` whose estimator happens
    to be a meta-estimator, so registration, serving, learning curves, diagnostic
    curves and the run report all apply to it unchanged. What is ensemble-specific is
    recorded in ``bundle.metadata["ensemble"]``: the method, the members, the same-fold
    member scores, the ensemble's out-of-fold score, any per-member calibration report,
    the optimized weight vector, the honest nested estimate when it was requested, and
    the caveats that qualify all of it.

    Returns ``(bundle, summary)`` where ``summary`` is the JSON-safe payload for the
    job result.
    """
    target, metadata = get_registered_target(task_name)
    frame = load_partition(task_name, train_partition)
    feature_columns = [column for column in frame.columns if column != target]
    # The frame, not `.values`: members may carry a DataFrame-first encoder, and
    # handing them an object array is what made bagging misread every numeric column.
    X = frame[feature_columns]
    y = frame[target].values

    # The process's compute budget, not scikit-learn's ``n_jobs=-1`` default: the four
    # API job slots share one joblib pool (see api/compute.py).
    n_jobs = int(compute.current()["job_threads"] if n_jobs is None else n_jobs)
    resolved, family, cleaned = ensemble_plan(
        task_name,
        members=members,
        method=method,
        params=params,
        calibration=calibration,
        weight_optimization=weight_optimization,
        columns=feature_columns,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    scoring = scoring or DEFAULT_ENSEMBLE_SCORING.get(family, "accuracy")
    cv_spec = get_stored_cv_spec(task_name)
    cv = get_stored_cv(task_name)
    if cv is None:
        # Fixed fold count so the member rows and the ensemble row share folds.
        cv = DEFAULT_ENSEMBLE_FOLDS

    pairs = [(entry["model_version"], entry["bundle"].model) for entry in resolved]
    try:
        estimator, provenance = fit_ensemble(
            method,
            pairs,
            X,
            y,
            family=family,
            params=params,
            calibration=calibration,
            weight_optimization=weight_optimization,
            columns=feature_columns,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    except EnsembleError as exc:
        raise bad_request(str(exc)) from None
    # ``fit_ensemble`` reports the parameters it actually used (the optimized weights are
    # injected there) and the per-member forms it combined, so the record describes the
    # fitted estimator without re-deriving anything here.
    cleaned = provenance["parameters"]
    comparison_forms = provenance["forms"]
    # The comparison rows describe the members *as they enter the ensemble*: when
    # calibration is on, that is the calibrated member, not the raw bundle.
    member_entries = []
    try:
        member_results = {
            entry["member"]: entry
            for entry in member_scores(
                comparison_forms, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs
            )
        }
    except EnsembleError:
        # A per-member scoring failure must not fail the ensemble that already fitted.
        member_results = {}
    for entry in resolved:
        score = member_results.get(entry["model_version"], {})
        member_entries.append(
            {
                "model_version": entry["model_version"],
                "label": entry["label"],
                "source": entry["source"],
                "family": problem_family_of(entry["bundle"].model),
                "metrics": entry["bundle"].metrics,
                "cv_mean": score.get("mean"),
                "cv_std": score.get("std"),
                "cv_folds": score.get("folds"),
                "error": score.get("error"),
                "calibrated": bool(calibration),
                "weight": (cleaned.get("weights") or [None] * len(resolved))[
                    [item["model_version"] for item in resolved].index(entry["model_version"])
                ],
            }
        )

    try:
        cv_mean, cv_std, cv_folds = cross_validated_scores(
            estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs
        )
        cv_error = None
    except EnsembleError as exc:
        # The combined model exists and is usable; only its honest score is missing.
        cv_mean, cv_std, cv_folds, cv_error = None, None, [], str(exc)
    train_score = finite_or_none(estimator.score(X, y))

    calibration_record = None
    if calibration:
        report = calibration_report(
            comparison_forms,
            X,
            y,
            cv=cv,
            method=calibration.get("method", "sigmoid"),
            ensemble=calibration.get("ensemble", "auto"),
            n_jobs=n_jobs,
        )
        calibration_record = {
            "method": calibration.get("method", "sigmoid"),
            "cv": calibration.get("cv", 5),
            "ensemble": calibration.get("ensemble", "auto"),
            "members": report,
        }

    nested_record = None
    if honest_cv:
        try:
            nested_record = nested_ensemble_cv(
                method,
                pairs,
                X,
                y,
                family=family,
                params=params,
                calibration=calibration,
                weight_optimization=weight_optimization,
                columns=feature_columns,
                outer_cv=honest_cv.get("folds", 5),
                inner_cv=honest_cv.get("inner_folds", 3),
                scoring=scoring,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        except EnsembleError as exc:
            # The ensemble itself fitted and is usable; only the honest estimate is
            # missing, and it is recorded as an error rather than taking the model down.
            nested_record = {
                "scoring": scoring,
                "outer_cv": honest_cv.get("folds", 5),
                "inner_cv": honest_cv.get("inner_folds", 3),
                "folds": [],
                "mean": None,
                "std": None,
                "errors": [{"fold": -1, "detail": str(exc)}],
                "per_fold_weights": None,
            }

    caveats = list(ENSEMBLE_CAVEATS)
    if weight_optimization:
        caveats.append(WEIGHT_OPTIMIZATION_CAVEAT)
    if calibration:
        caveats.append(CALIBRATION_CAVEAT)

    metrics = {"train_score": train_score, "cv_score": cv_mean, "cv_score_std": cv_std}
    run_record = _reproducibility_record(
        task_name,
        metadata,
        frame,
        target,
        train_partition,
        configuration={
            "kind": "ensemble",
            "method": method,
            "parameters": cleaned,
            "members": [
                {"model_version": entry["model_version"], "source": entry["source"]}
                for entry in resolved
            ],
            "calibration": (
                {key: value for key, value in calibration.items() if key != "n_jobs"}
                if calibration
                else None
            ),
            "weight_optimization": (
                {key: value for key, value in weight_optimization.items() if key != "n_jobs"}
                if weight_optimization
                else None
            ),
            "honest_cv": honest_cv or None,
            "scoring": scoring,
            "random_state": random_state,
            "n_jobs": n_jobs,
        },
        transformations=[
            {
                "class": "%s.%s" % (type(estimator).__module__, type(estimator).__qualname__),
                "parameters": {
                    name: value
                    for name, value in cleaned.items()
                    if name != "weights"
                },
            }
        ],
        evaluation_protocol={
            "scoring": scoring,
            "cv": cv_spec if cv_spec is not None else {"folds": DEFAULT_ENSEMBLE_FOLDS, "source": "default"},
            "metric_scope": (
                "member and ensemble scores are out-of-fold on identical folds; "
                "train_score is the refit training score"
                + ("; a weight vector was fitted on those folds" if weight_optimization else "")
                + ("; the nested estimate is the honest number" if honest_cv else "")
            ),
        },
    )
    bundle = _new_bundle(
        estimator, metrics, task_name, {column: {"dtype": str(frame[column].dtype)} for column in feature_columns}, run_record, owner, run_id, label=label
    )
    ensemble_metadata = {
        "schema": "sksurrogate.ensemble.v1",
        "method": method,
        "family": family,
        "parameters": cleaned,
        "aggregation": aggregation_label(method, family),
        "members": member_entries,
        "calibration": calibration_record,
        "weight_optimization": provenance["weights"],
        "scoring": scoring,
        "cv": cv_spec if cv_spec is not None else {"folds": DEFAULT_ENSEMBLE_FOLDS, "source": "default"},
        "scores": {
            "train_score": train_score,
            "cv_score": cv_mean,
            "cv_score_std": cv_std,
            "cv_folds": cv_folds,
            "cv_error": cv_error,
            "nested": nested_record,
        },
        "caveats": caveats,
    }
    bundle.metadata["ensemble"] = ensemble_metadata
    bundle.record_audit_event(
        "ensemble",
        method=method,
        members=[entry["model_version"] for entry in resolved],
        scoring=scoring,
        train_score=train_score,
        cv_score=cv_mean,
        calibrated=bool(calibration),
        weights=(
            [round(float(value), 6) for value in cleaned["weights"]]
            if cleaned.get("weights")
            else None
        ),
    )

    bundles_dir = settings.task_bundles_dir(task_name)
    bundles_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = save_bundle(
        bundle,
        bundles_dir / (safe_segment(bundle.model_version, "model version") + ".bundle"),
    )
    registered = False
    if register:
        ModelRegistry(settings.registry_dir).register(bundle)
        registered = True

    summary = {
        "model_version": bundle.model_version,
        "bundle_path": str(bundle_path),
        "method": method,
        "family": family,
        "members": member_entries,
        "metrics": metrics,
        "calibration": calibration_record,
        "weight_optimization": provenance["weights"],
        "nested": nested_record,
        "caveats": caveats,
        "registered": registered,
    }
    return bundle, summary
