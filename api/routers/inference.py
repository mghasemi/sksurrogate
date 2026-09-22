"""Batch/online inference endpoints, wrapping ``SKSurrogate.inference.predict_batch``.

Every call also appends a persisted ``InferenceMonitor`` observation so the
Monitoring router can report latency/throughput history for the model
version used (see docs/ui-plan.md section 4).
"""

import uuid

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

from SKSurrogate import SchemaValidationError, predict_batch

from ..config import settings
from ..deps import append_monitor_record, bad_request, not_found, resolve_bundle

router = APIRouter(prefix="/api/inference", tags=["inference"])


class PredictRequest(BaseModel):
    model_version: str | None = None
    alias: str | None = None
    partition: str | None = None
    rows: list[dict] | None = None
    request_id: str | None = None


def _resolve_input_frame(task_name, body: PredictRequest):
    if body.rows is not None:
        return pd.DataFrame(body.rows)
    if body.partition is not None:
        csv_path = settings.task_dataset_dir(task_name) / (body.partition + ".csv")
        if not csv_path.exists():
            raise not_found("No dataset partition %r stored for task %r" % (body.partition, task_name))
        return pd.read_csv(csv_path)
    raise bad_request("Either 'rows' or 'partition' must be provided")


@router.post("/{task_name}/predict")
def predict(task_name: str, body: PredictRequest):
    """Run predictions in-request and return them as JSON (no file written)."""
    bundle = resolve_bundle(task_name, model_version=body.model_version, alias=body.alias)
    frame = _resolve_input_frame(task_name, body)
    try:
        result = predict_batch(bundle, frame, request_id=body.request_id or uuid.uuid4().hex)
    except SchemaValidationError as exc:
        raise bad_request("; ".join(exc.errors))
    metrics = result.attrs["inference_metrics"]
    append_monitor_record(task_name, bundle.model_version, metrics["latency_ms"], metrics["rows"])
    return {
        "task_name": task_name,
        "model_version": bundle.model_version,
        "request_id": result["request_id"].iloc[0],
        "predictions": result["prediction"].tolist(),
        "metrics": metrics,
    }


@router.post("/{task_name}/predict-batch")
def predict_batch_endpoint(task_name: str, body: PredictRequest):
    """Run predictions and persist them as a CSV artifact under the task's predictions folder."""
    bundle = resolve_bundle(task_name, model_version=body.model_version, alias=body.alias)
    frame = _resolve_input_frame(task_name, body)
    request_id = body.request_id or uuid.uuid4().hex
    output_dir = settings.task_predictions_dir(task_name) / bundle.model_version
    output_path = output_dir / (request_id + ".csv")
    try:
        result = predict_batch(bundle, frame, output_path=output_path, request_id=request_id)
    except SchemaValidationError as exc:
        raise bad_request("; ".join(exc.errors))
    metrics = result.attrs["inference_metrics"]
    append_monitor_record(task_name, bundle.model_version, metrics["latency_ms"], metrics["rows"])
    return {
        "task_name": task_name,
        "model_version": bundle.model_version,
        "request_id": request_id,
        "rows": metrics["rows"],
        "metrics": metrics,
        "output_path": str(output_path),
    }
