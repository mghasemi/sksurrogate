"""Cross-stage lineage: dataset -> bundle -> registry -> monitoring.

A single endpoint that walks the metadata already recorded by the other
routers (see docs/ui-plan.md section 3, "traceability rail") so the UI can
render one breadcrumb per model version without stitching multiple calls.
"""

from fastapi import APIRouter

from SKSurrogate import ModelRegistry, load_bundle

from ..config import settings
from ..deps import bundle_summary, load_monitor_records, not_found, open_tracker

router = APIRouter(prefix="/api/lineage", tags=["lineage"])


@router.get("/{task_name}/{model_version}")
def get_lineage(task_name: str, model_version: str):
    """Return the full trail for one model version: dataset -> bundle -> registry -> monitoring."""
    # A bundle may only exist under the unregistered bundles folder, only in the
    # registry (e.g. produced by /api/retraining), or in both.
    bundle_path = settings.task_bundles_dir(task_name) / (model_version + ".bundle")
    if not bundle_path.exists():
        bundle_path = settings.registry_dir / task_name / (model_version + ".bundle")
    if not bundle_path.exists():
        raise not_found("No bundle %r found for task %r" % (model_version, task_name))
    bundle = load_bundle(bundle_path, strict_dependencies=False)
    summary = bundle_summary(bundle)

    dataset = None
    if settings.mltrace_db_path(task_name).exists():
        with open_tracker(task_name) as tracker:
            metadata = tracker.GetMetadata()
        dataset = {
            "dataset_fingerprint": metadata.get("dataset_fingerprint"),
            "target_name": metadata.get("target_name"),
            "partitions": tracker.dataset_splits(),
        }

    registry_state = None
    registry = ModelRegistry(settings.registry_dir)
    try:
        history = registry.history(task_name)
        audit = registry.audit_log(task_name)
        version_events = [event for event in audit if event.get("model_version") == model_version]
        current_state = next(
            (event.get("to") for event in reversed(history) if event.get("model_version") == model_version),
            None,
        )
        registry_state = {"state": current_state, "history": version_events}
    except KeyError:
        registry_state = None

    monitoring_summary = None
    records = load_monitor_records(task_name, model_version)
    if records:
        total_rows = sum(record["rows"] for record in records)
        monitoring_summary = {
            "requests": len(records),
            "rows": total_rows,
            "error_rate": sum(not record.get("success", True) for record in records) / len(records),
        }

    return {
        "task_name": task_name,
        "model_version": model_version,
        "bundle": summary,
        "dataset": dataset,
        "registry": registry_state,
        "monitoring": monitoring_summary,
    }
