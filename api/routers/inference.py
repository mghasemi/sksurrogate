"""Batch/online inference endpoints, wrapping ``SKSurrogate.inference.predict_batch``.

Here the *caller* names the model on every request — an exact ``model_version``
or a lifecycle/registry ``alias``. To declare a model once and score through it
without naming it, see :mod:`api.routers.serving` (``/api/serving/{task}/predict``).

Every call also appends a persisted ``InferenceMonitor`` observation so the
Monitoring router can report latency/throughput history for the model
version used (see docs/ui-plan.md section 4). The prediction core itself —
input-frame resolution, pre-flight validation, prediction and monitoring — is
shared with the serving router in :mod:`api.inference_service`.
"""

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from SKSurrogate import ModelRegistry

from ..config import settings
from ..inference_service import resolve_input_frame, score
from ..safety import safe_segment
from ..deps import open_tracker, resolve_bundle, task_access

router = APIRouter(prefix="/api/inference", tags=["inference"], dependencies=[Depends(task_access)])


class PredictRequest(BaseModel):
    model_version: str | None = None
    alias: str | None = None
    partition: str | None = None
    rows: list[dict] | None = None
    request_id: str | None = None
    # Phase 5.2: when set, run ``validate_prediction_data`` against the registered
    # dataset schema before predicting and surface the schema-difference message
    # instead of a raw prediction failure.
    preflight: bool = False


@router.post("/{task_name}/predict")
def predict(task_name: str, body: PredictRequest):
    """Run predictions in-request and return them as JSON (no file written)."""
    bundle = resolve_bundle(task_name, model_version=body.model_version, alias=body.alias)
    frame, ignored_columns = resolve_input_frame(
        task_name, bundle=bundle, rows=body.rows, partition=body.partition
    )
    result = score(
        task_name, frame, bundle, request_id=body.request_id, preflight=body.preflight
    )
    return {
        "task_name": task_name,
        "model_version": bundle.model_version,
        "request_id": result["request_id"],
        "predictions": result["predictions"],
        "metrics": result["metrics"],
        "ignored_columns": ignored_columns,
    }


@router.post("/{task_name}/predict-batch")
def predict_batch_endpoint(task_name: str, body: PredictRequest):
    """Run predictions and persist them as a CSV artifact under the task's predictions folder."""
    bundle = resolve_bundle(task_name, model_version=body.model_version, alias=body.alias)
    frame, ignored_columns = resolve_input_frame(
        task_name, bundle=bundle, rows=body.rows, partition=body.partition
    )
    request_id = body.request_id or uuid.uuid4().hex
    output_dir = settings.task_predictions_dir(task_name) / safe_segment(
        bundle.model_version, "model version"
    )
    output_path = output_dir / (safe_segment(request_id, "request id") + ".csv")
    result = score(
        task_name,
        frame,
        bundle,
        request_id=request_id,
        preflight=body.preflight,
        output_path=output_path,
    )
    metrics = result["metrics"]

    # Phase 3.4 auto-wiring: register the persisted CSV as a prediction artifact
    # (keyed to the model version + dataset fingerprint) so lineage is complete.
    # Best-effort — never fail a prediction that already succeeded.
    try:
        fingerprint = bundle.dataset_fingerprint
        if fingerprint is None and settings.mltrace_db_path(task_name).exists():
            with open_tracker(task_name) as tracker:
                fingerprint = (tracker.GetMetadata() or {}).get("dataset_fingerprint")
        ModelRegistry(settings.registry_dir).register_prediction(
            task_name,
            bundle.model_version,
            fingerprint or "unregistered",
            output_path,
            prediction_id=request_id,
            metadata={"rows": metrics["rows"], "source": "predict-batch"},
        )
    except Exception:  # pragma: no cover - defensive; registration is auxiliary
        pass

    return {
        "task_name": task_name,
        "model_version": bundle.model_version,
        "request_id": request_id,
        "rows": metrics["rows"],
        "metrics": metrics,
        "output_path": str(output_path),
        "ignored_columns": ignored_columns,
    }
