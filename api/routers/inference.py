"""Batch/online inference endpoints, wrapping ``SKSurrogate.inference.predict_batch``.

Here the *caller* names the model on every request — an exact ``model_version``
or a lifecycle/registry ``alias``. To declare a model once and score through it
without naming it, see :mod:`api.routers.serving` (``/api/serving/{task}/predict``).

Every call also appends a persisted ``InferenceMonitor`` observation so the
Monitoring router can report latency/throughput history for the model
version used (see dev-notes/ui-plan.md section 4). The prediction core itself —
input-frame resolution, pre-flight validation, prediction and monitoring — is
shared with the serving router in :mod:`api.inference_service`.
"""

import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from starlette.responses import FileResponse

from SKSurrogate import ModelRegistry

from ..config import settings
from ..inference_service import resolve_input_frame, score, score_forecast
from ..safety import safe_segment
from ..deps import bad_request, not_found, open_tracker, resolve_bundle, task_access

router = APIRouter(prefix="/api/inference", tags=["inference"], dependencies=[Depends(task_access)])


class PredictRequest(BaseModel):
    model_version: str | None = None
    alias: str | None = None
    partition: str | None = None
    rows: list[dict] | None = None
    request_id: str | None = None
    # Phase 5.2: when set, validate the frame against the model bundle's feature
    # schema before predicting and surface the schema-difference message
    # instead of a raw prediction failure.
    preflight: bool = False


@router.post("/{task_name}/predict")
def predict(task_name: str, body: PredictRequest):
    """Run predictions in-request and return them as JSON (no file written)."""
    bundle = resolve_bundle(
        task_name, model_version=body.model_version, alias=body.alias, verify=True
    )
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


class ForecastRequest(BaseModel):
    """Multi-step forecast payload for a temporal bundle (plan phase P6).

    Carries *history* rows — one series of past observations — plus optional
    future exogenous rows and horizon, not feature rows: the stored design
    validates them before anything is predicted.
    """

    model_version: str | None = None
    alias: str | None = None
    history: list[dict]
    future_exogenous: list[dict] | None = None
    horizon: int | None = None
    request_id: str | None = None


@router.post("/{task_name}/forecast")
def forecast(task_name: str, body: ForecastRequest):
    """Serve a multi-step forecast from the task's temporal bundle (plan P6).

    Temporal bundles — v2 records with the ``temporal`` block written by plan
    phase P6 in ``TIME_SERIES_PLAN.md`` — resolve like any other model; tabular
    bundles answer 400, since forecasting is refused rather than faked. The
    response carries the strategy and requested horizon alongside one entry per
    forecast step (times, predictions and interval bounds).
    """
    bundle = resolve_bundle(
        task_name, model_version=body.model_version, alias=body.alias, verify=True
    )
    future_frame = (
        pd.DataFrame(body.future_exogenous) if body.future_exogenous is not None else None
    )
    result = score_forecast(
        task_name,
        pd.DataFrame(body.history),
        bundle,
        horizon=body.horizon,
        future_exogenous=future_frame,
        request_id=body.request_id,
    )
    return {
        "task_name": task_name,
        "model_version": bundle.model_version,
        **result,
    }


@router.post("/{task_name}/predict-batch")
def predict_batch_endpoint(task_name: str, body: PredictRequest):
    """Run predictions and persist them as a CSV artifact under the task's predictions folder."""
    bundle = resolve_bundle(
        task_name, model_version=body.model_version, alias=body.alias, verify=True
    )
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


@router.get("/{task_name}/predictions/{model_version}/{filename}")
def download_prediction(task_name: str, model_version: str, filename: str):
    """Stream a persisted batch-prediction CSV artifact back to the caller.

    ``POST /{task_name}/predict-batch`` writes ``<predictions>/<task>/<version>/
    <request_id>.csv``; this is the URL the Inference page's success note links,
    so an operator can retrieve the file without shell access to the server. The
    filename must be a single plain name ending in ``.csv``, and every segment
    passes the same containment guard as other caller-supplied path components
    (400), so nothing outside the task's predictions directory is served; an
    unknown artifact answers 404 rather than failing with a traceback.
    """
    if Path(filename).name != filename or not filename.endswith(".csv"):
        raise bad_request("Invalid prediction artifact filename")
    path = (
        settings.task_predictions_dir(task_name) / safe_segment(model_version, "model version") / filename
    )
    if not path.is_file():
        raise not_found(
            "No batch-prediction file %r stored for model %r under task %r"
            % (filename, model_version, task_name)
        )
    return FileResponse(path, media_type="text/csv", filename=filename)
