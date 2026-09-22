"""Model bundle creation & inspection endpoints.

Phase A ships a synchronous "train baseline" convenience endpoint so bundles
can be produced end-to-end without the async AutoML job infrastructure
(Phase C, see docs/ui-plan.md section 8). Fitting logic is shared with the
Experiments and Retraining routers via ``api.training``.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from SKSurrogate import load_bundle, save_bundle

from ..config import settings
from ..deps import bundle_summary, not_found
from ..training import fit_baseline_bundle

router = APIRouter(prefix="/api/bundles", tags=["bundles"])


class TrainBaselineRequest(BaseModel):
    estimator: str = "linear_regression"
    train_partition: str = "train"
    validation_partition: str | None = "validation"
    owner: str | None = None
    run_id: str | None = None


@router.post("/{task_name}/train-baseline")
def train_baseline(task_name: str, body: TrainBaselineRequest):
    """Fit a simple scikit-learn baseline pipeline and wrap it into a bundle."""
    bundle = fit_baseline_bundle(
        task_name,
        estimator=body.estimator,
        train_partition=body.train_partition,
        validation_partition=body.validation_partition,
        owner=body.owner,
        run_id=body.run_id,
    )
    bundles_dir = settings.task_bundles_dir(task_name)
    bundles_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = save_bundle(bundle, bundles_dir / (bundle.model_version + ".bundle"))

    return {
        "task_name": task_name,
        "model_version": bundle.model_version,
        "estimator": body.estimator,
        "metrics": bundle.metrics,
        "schema": bundle.schema,
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
