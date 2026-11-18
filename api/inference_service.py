"""Shared prediction core for the per-request and serving inference endpoints.

Two routers expose prediction over HTTP and they differ in exactly one thing:
*how the bundle is resolved*. ``api.routers.inference`` takes the model from
each request body (``model_version`` or ``alias``); ``api.routers.serving``
takes it from the persisted serving pin (see :mod:`api.serving`). Everything
after resolution — building the input frame, optional pre-flight validation
against the registered dataset schema, the ``predict_batch`` call, the
monitoring record — is identical and lives here so the two paths cannot drift.
"""

import uuid
from time import perf_counter

import pandas as pd

from SKSurrogate import SchemaValidationError, predict_batch

from .config import settings
from .deps import append_monitor_record, bad_request, not_found, open_tracker
from .safety import safe_segment


def resolve_input_frame(task_name, *, bundle, rows=None, partition=None):
    """Build the input frame, returning ``(frame, ignored_columns)``.

    A stored partition is *registered training data*, so it carries the target
    label (and any other non-feature column). ``predict_batch`` validates the
    input schema strictly — it rejects unknown columns outright — so a partition
    is projected onto the bundle schema first and the dropped columns are
    reported back. Inline ``rows`` are the caller's exact payload (the
    production serving contract) and are passed through untouched.
    """
    if rows is not None:
        return pd.DataFrame(rows), []
    if partition is not None:
        csv_path = settings.task_dataset_dir(task_name) / (safe_segment(partition, "partition") + ".csv")
        if not csv_path.exists():
            raise not_found("No dataset partition %r stored for task %r" % (partition, task_name))
        frame = pd.read_csv(csv_path)
        if not bundle.schema:
            return frame, []
        ignored = [column for column in frame.columns if column not in bundle.schema]
        # Schema order is authoritative, so selecting by it also normalizes a
        # partition whose columns happen to be stored in a different order.
        return frame[[column for column in bundle.schema if column in frame.columns]], ignored
    raise bad_request("Either 'rows' or 'partition' must be provided")


def preflight_error(task_name, frame):
    """Return a validation error string for ``frame``, or None if it passes."""
    if not settings.mltrace_db_path(task_name).exists():
        return "No registered dataset schema found for task %r; cannot run pre-flight validation." % task_name
    try:
        with open_tracker(task_name) as tracker:
            tracker.validate_prediction_data(frame)
    except ValueError as exc:
        return str(exc)
    return None


def check_preflight(task_name, frame, bundle, enabled):
    """Validate ``frame`` against the task's registered schema when ``enabled``.

    A failure is a 400 (not a prediction error) and still records the failed
    request, so the Monitoring page sees pre-flight rejections as errors.
    """
    if not enabled:
        return
    started = perf_counter()
    error = preflight_error(task_name, frame)
    if error is not None:
        latency_ms = (perf_counter() - started) * 1000
        append_monitor_record(
            task_name,
            bundle.model_version,
            latency_ms,
            len(frame),
            success=False,
            error="Pre-flight validation failed",
        )
        raise bad_request("Pre-flight validation failed: %s" % error)


def score(task_name, frame, bundle, *, request_id=None, preflight=False, output_path=None):
    """Predict in-request and record the observation, returning the response core.

    ``output_path`` is forwarded to ``predict_batch`` (a batch run persists a
    CSV); the single-request path passes nothing and only returns JSON. The
    returned dict carries the request id, the predictions and the inference
    metrics — the caller adds the model/lineage fields it resolved itself.
    """
    request_id = request_id or uuid.uuid4().hex
    check_preflight(task_name, frame, bundle, preflight)
    started = perf_counter()
    try:
        result = predict_batch(bundle, frame, output_path=output_path, request_id=request_id)
    except SchemaValidationError as exc:
        latency_ms = (perf_counter() - started) * 1000
        append_monitor_record(
            task_name,
            bundle.model_version,
            latency_ms,
            len(frame),
            success=False,
            error="Schema validation failed",
        )
        raise bad_request("; ".join(exc.errors))
    metrics = result.attrs["inference_metrics"]
    append_monitor_record(task_name, bundle.model_version, metrics["latency_ms"], metrics["rows"])
    return {
        "request_id": result["request_id"].iloc[0],
        "predictions": result["prediction"].tolist(),
        "metrics": metrics,
    }
