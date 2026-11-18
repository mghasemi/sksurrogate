"""Bundle quality-gate endpoints, wrapping ``SKSurrogate.ci``."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from SKSurrogate import check_bundle_quality

from ..config import settings
from ..safety import safe_segment
from ..deps import not_found, task_access

router = APIRouter(prefix="/api/quality-gates", tags=["quality-gates"], dependencies=[Depends(task_access)])


class QualityGateRequest(BaseModel):
    metric_thresholds: dict[str, float] = {}
    max_bundle_size_bytes: int | None = None


@router.post("/{task_name}/{model_version}/check")
def check_quality(task_name: str, model_version: str, body: QualityGateRequest):
    """Run CI-style quality checks against a stored (not-yet-registered) bundle."""
    bundle_path = settings.task_bundles_dir(task_name) / (safe_segment(model_version, "model version") + ".bundle")
    if not bundle_path.exists():
        raise not_found("No bundle %r found for task %r" % (model_version, task_name))
    report = check_bundle_quality(
        bundle_path,
        metric_thresholds=body.metric_thresholds,
        max_bundle_size_bytes=body.max_bundle_size_bytes,
    )
    return {"task_name": task_name, "model_version": model_version, **report}
