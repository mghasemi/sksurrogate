"""Shared model-fitting helpers used by the bundles, experiments, and retraining routers."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from SKSurrogate import AML, Categorical, Integer, ModelBundle, Real

from .config import settings
from .deps import bad_request, not_found, open_tracker

BASELINE_ESTIMATORS = {
    "linear_regression": LinearRegression,
    "logistic_regression": LogisticRegression,
    "random_forest_classifier": RandomForestClassifier,
    "random_forest_regressor": RandomForestRegressor,
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
    """Convert a JSON param spec (``{"type": "real"|"integer"|"categorical", ...}``) into a structsearch range."""
    if spec["type"] == "real":
        return Real(spec.get("low"), spec.get("high"))
    if spec["type"] == "integer":
        return Integer(int(spec["low"]), int(spec["high"]))
    if spec["type"] == "categorical":
        return Categorical(spec.get("items") or [])
    raise bad_request("param type must be one of real, integer, categorical")


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

    parsed_config = {
        estimator: {name: build_search_param(spec) for name, spec in params.items()}
        for estimator, params in config.items()
    }

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
    )
    aml.eoa_fit(X, y, max_generation=max_generation, num_parents=num_parents)
    # SurrogateRandomCV refits best_estimator_ on the full dataset (refit=True),
    # so it is ready for scoring/serving without an extra fit here.
    best_estimator = aml.best_estimator_
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
    )
    evaluation_history = [
        {"pipeline": list(entry["pipeline"]), "score": float(entry["score"])} for entry in aml.evaluation_history_
    ]
    return bundle, evaluation_history
