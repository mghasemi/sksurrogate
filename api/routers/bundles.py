"""Model bundle creation & inspection endpoints.

Phase A ships a synchronous "train baseline" convenience endpoint so bundles
can be produced end-to-end without the async AutoML job infrastructure
(Phase C, see docs/ui-plan.md section 8). Real AutoML-searched models will
plug into the same ``ModelBundle`` creation step once the Experiments router
lands.
"""

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from SKSurrogate import ModelBundle, load_bundle, save_bundle

from ..config import settings
from ..deps import bad_request, bundle_summary, not_found, open_tracker

router = APIRouter(prefix="/api/bundles", tags=["bundles"])

_ESTIMATORS = {
    "linear_regression": LinearRegression,
    "logistic_regression": LogisticRegression,
    "random_forest_classifier": RandomForestClassifier,
    "random_forest_regressor": RandomForestRegressor,
}


class TrainBaselineRequest(BaseModel):
    estimator: str = "linear_regression"
    train_partition: str = "train"
    validation_partition: str | None = "validation"
    owner: str | None = None
    run_id: str | None = None


def _load_partition(task_name, partition):
    csv_path = settings.task_dataset_dir(task_name) / (partition + ".csv")
    if not csv_path.exists():
        raise not_found("No dataset partition %r stored for task %r" % (partition, task_name))
    return pd.read_csv(csv_path)


@router.post("/{task_name}/train-baseline")
def train_baseline(task_name: str, body: TrainBaselineRequest):
    """Fit a simple scikit-learn baseline pipeline and wrap it into a bundle."""
    if body.estimator not in _ESTIMATORS:
        raise bad_request("Unknown estimator %r; choose one of %s" % (body.estimator, sorted(_ESTIMATORS)))

    with open_tracker(task_name) as tracker:
        metadata = tracker.GetMetadata()
    target = metadata.get("target_name")
    if target is None:
        raise not_found("No dataset registered for task %r" % task_name)

    train_frame = _load_partition(task_name, body.train_partition)
    train_X = train_frame.drop(columns=[target])
    train_y = train_frame[target]

    model = make_pipeline(StandardScaler(), _ESTIMATORS[body.estimator]())
    model.fit(train_X, train_y)

    metrics = {}
    if body.validation_partition:
        validation_csv = settings.task_dataset_dir(task_name) / (body.validation_partition + ".csv")
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
        owner=body.owner,
        run_id=body.run_id,
    )
    bundle.record_audit_event("training", estimator=body.estimator, metrics=metrics)

    bundles_dir = settings.task_bundles_dir(task_name)
    bundles_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = save_bundle(bundle, bundles_dir / (bundle.model_version + ".bundle"))

    return {
        "task_name": task_name,
        "model_version": bundle.model_version,
        "estimator": body.estimator,
        "metrics": metrics,
        "schema": schema,
        "bundle_path": str(bundle_path),
    }


@router.get("/{task_name}")
def list_bundles(task_name: str):
    """List unregistered bundles produced for a task, most recent first."""
    bundles_dir = settings.task_bundles_dir(task_name)
    if not bundles_dir.exists():
        return {"task_name": task_name, "bundles": []}
    entries = sorted(bundles_dir.glob("*.bundle"), key=lambda path: path.stat().st_mtime, reverse=True)
    return {"task_name": task_name, "bundles": [path.stem for path in entries]}


@router.get("/{task_name}/{model_version}")
def get_bundle(task_name: str, model_version: str):
    """Return bundle metadata (schema, metrics, dependencies, audit trail) without the model object."""
    bundle_path = settings.task_bundles_dir(task_name) / (model_version + ".bundle")
    if not bundle_path.exists():
        raise not_found("No bundle %r found for task %r" % (model_version, task_name))
    bundle = load_bundle(bundle_path, strict_dependencies=False)
    return bundle_summary(bundle)
