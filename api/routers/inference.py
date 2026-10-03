"""Batch/online inference endpoints, wrapping ``SKSurrogate.inference.predict_batch``.

Every call also appends a persisted ``InferenceMonitor`` observation so the
Monitoring router can report latency/throughput history for the model
version used (see docs/ui-plan.md section 4).
"""

import uuid
from time import perf_counter

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

from SKSurrogate import ModelRegistry, SchemaValidationError, predict_batch

from ..config import settings
from ..deps import append_monitor_record, bad_request, not_found, open_tracker, resolve_bundle

router = APIRouter(prefix="/api/inference", tags=["inference"])


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


def _preflight_error(task_name, frame):
    """Return a validation error string for ``frame``, or None if it passes."""
    if not settings.mltrace_db_path(task_name).exists():
        return "No registered dataset schema found for task %r; cannot run pre-flight validation." % task_name
    try:
        with open_tracker(task_name) as tracker:
            tracker.validate_prediction_data(frame)
    except ValueError as exc:
        return str(exc)
    return None


def _check_preflight(task_name, frame, bundle, enabled):
    if not enabled:
        return
    started = perf_counter()
    error = _preflight_error(task_name, frame)
    if error is not None:
        latency_ms = (perf_counter() - started) * 1000
        append_monitor_record(
            task_name, bundle.model_version, latency_ms, len(frame), success=False, error=error
        )
        raise bad_request("Pre-flight validation failed: %s" % error)


def _resolve_input_frame(task_name, body: PredictRequest, bundle):
    """Build the input frame, returning ``(frame, ignored_columns)``.

    A stored partition is *registered training data*, so it carries the target
    label (and any other non-feature column). ``predict_batch`` validates the
    input schema strictly — it rejects unknown columns outright — so a partition
    is projected onto the bundle schema first and the dropped columns are
    reported back. Inline ``rows`` are the caller's exact payload (the
    production serving contract) and are passed through untouched.
    """
    if body.rows is not None:
        return pd.DataFrame(body.rows), []
    if body.partition is not None:
        csv_path = settings.task_dataset_dir(task_name) / (body.partition + ".csv")
        if not csv_path.exists():
            raise not_found("No dataset partition %r stored for task %r" % (body.partition, task_name))
        frame = pd.read_csv(csv_path)
        if not bundle.schema:
            return frame, []
        ignored = [column for column in frame.columns if column not in bundle.schema]
        # Schema order is authoritative, so selecting by it also normalizes a
        # partition whose columns happen to be stored in a different order.
        return frame[[column for column in bundle.schema if column in frame.columns]], ignored
    raise bad_request("Either 'rows' or 'partition' must be provided")


@router.post("/{task_name}/predict")
def predict(task_name: str, body: PredictRequest):
    """Run predictions in-request and return them as JSON (no file written)."""
    bundle = resolve_bundle(task_name, model_version=body.model_version, alias=body.alias)
    frame, ignored_columns = _resolve_input_frame(task_name, body, bundle)
    _check_preflight(task_name, frame, bundle, body.preflight)
    started = perf_counter()
    try:
        result = predict_batch(bundle, frame, request_id=body.request_id or uuid.uuid4().hex)
    except SchemaValidationError as exc:
        latency_ms = (perf_counter() - started) * 1000
        append_monitor_record(
            task_name, bundle.model_version, latency_ms, len(frame), success=False, error=exc
        )
        raise bad_request("; ".join(exc.errors))
    metrics = result.attrs["inference_metrics"]
    append_monitor_record(task_name, bundle.model_version, metrics["latency_ms"], metrics["rows"])
    return {
        "task_name": task_name,
        "model_version": bundle.model_version,
        "request_id": result["request_id"].iloc[0],
        "predictions": result["prediction"].tolist(),
        "metrics": metrics,
        "ignored_columns": ignored_columns,
    }


@router.post("/{task_name}/predict-batch")
def predict_batch_endpoint(task_name: str, body: PredictRequest):
    """Run predictions and persist them as a CSV artifact under the task's predictions folder."""
    bundle = resolve_bundle(task_name, model_version=body.model_version, alias=body.alias)
    frame, ignored_columns = _resolve_input_frame(task_name, body, bundle)
    _check_preflight(task_name, frame, bundle, body.preflight)
    request_id = body.request_id or uuid.uuid4().hex
    output_dir = settings.task_predictions_dir(task_name) / bundle.model_version
    output_path = output_dir / (request_id + ".csv")
    started = perf_counter()
    try:
        result = predict_batch(bundle, frame, output_path=output_path, request_id=request_id)
    except SchemaValidationError as exc:
        latency_ms = (perf_counter() - started) * 1000
        append_monitor_record(
            task_name, bundle.model_version, latency_ms, len(frame), success=False, error=exc
        )
        raise bad_request("; ".join(exc.errors))
    metrics = result.attrs["inference_metrics"]
    append_monitor_record(task_name, bundle.model_version, metrics["latency_ms"], metrics["rows"])

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
