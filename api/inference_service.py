"""Shared prediction core for the per-request and serving inference endpoints.

Two routers expose prediction over HTTP and they differ in exactly one thing:
*how the bundle is resolved*. ``api.routers.inference`` takes the model from
each request body (``model_version`` or ``alias``); ``api.routers.serving``
takes it from the persisted serving pin (see :mod:`api.serving`). Everything
after resolution — building the input frame, optional pre-flight validation
against the *model bundle's* feature schema, the ``predict_batch`` call, the
monitoring record — is identical and lives here so the two paths cannot drift.
"""

import uuid
from time import perf_counter

import pandas as pd

from SKSurrogate import SchemaValidationError, predict_batch, predict_batch_forecast

from .config import settings
from .deps import append_monitor_record, bad_request, not_found
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


def preflight_error(frame, bundle):
    """Return a validation error string for ``frame``, or None if it passes.

    Pre-flight checks the input against the *model bundle's* feature schema —
    the columns the estimator actually consumes. Validating against the task's
    full registered dataset schema instead would reject a model trained on a
    curated or downstream feature subset of the raw table (its input is missing
    unrelated columns the model never used), so the bundle schema is the
    authoritative contract a caller must satisfy. ``predict_batch`` remains the
    strict gate for feature presence, ordering and dtypes.
    """
    expected = list(bundle.schema or ())
    if not expected:
        return None
    actual = list(frame.columns)
    if actual != expected:
        return "Prediction schema mismatch: %s" % _schema_difference(expected, actual)
    return None


def _schema_difference(expected_columns, actual_columns):
    """Readable description of a column mismatch in the same shape as ``mltrace``.

    Mirrors ``SKSurrogate.mltrace._format_schema_difference`` so callers and the
    Monitoring page see a consistent message, but computed against the model
    bundle's feature schema rather than the task's dataset schema.
    """
    missing = [column for column in expected_columns if column not in actual_columns]
    extra = [column for column in actual_columns if column not in expected_columns]
    if missing or extra:
        detail = []
        if missing:
            detail.append("missing columns: %s" % ", ".join(missing))
        if extra:
            detail.append("unexpected columns: %s" % ", ".join(extra))
        if list(expected_columns) != list(actual_columns):
            detail.append(
                "expected order: [%s], received: [%s]"
                % (", ".join(map(str, expected_columns)), ", ".join(map(str, actual_columns)))
            )
        return "; ".join(detail)
    return "columns are in a different order"


def check_preflight(task_name, frame, bundle, enabled):
    """Validate ``frame`` against ``bundle``'s feature schema when ``enabled``.

    A failure is a 400 (not a prediction error) and still records the failed
    request, so the Monitoring page sees pre-flight rejections as errors.
    """
    if not enabled:
        return
    started = perf_counter()
    error = preflight_error(frame, bundle)
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


def score_forecast(task_name, frame, bundle, *, horizon=None, future_exogenous=None, request_id=None):
    """Serve a multi-step forecast from a temporal bundle and record it (plan P6).

    The :func:`score` counterpart for time-series models: ``frame`` is the
    *history* series, validated against the stored design rather than treated as
    feature rows — so no pre-flight schema check applies here. Tabular bundles
    are refused with a 400 (forecasting is not faked), and rejected horizon
    requests surface the same ``ValueError`` text that ``serve_forecast`` gives
    programmatic callers, keeping both paths honest. One monitoring observation
    is recorded either way, mirroring :func:`score`.
    """
    request_id = request_id or uuid.uuid4().hex
    started = perf_counter()
    try:
        result = predict_batch_forecast(
            bundle,
            frame,
            horizon=horizon,
            future_exogenous=future_exogenous,
            request_id=request_id,
        )
    except SchemaValidationError as exc:
        latency_ms = (perf_counter() - started) * 1000
        append_monitor_record(
            task_name,
            bundle.model_version,
            latency_ms,
            len(frame),
            success=False,
            error="History validation failed",
        )
        raise bad_request("; ".join(exc.errors))
    except ValueError as exc:
        # Tabular bundles or a horizon the design cannot answer fail at serving.
        latency_ms = (perf_counter() - started) * 1000
        append_monitor_record(
            task_name, bundle.model_version, latency_ms, len(frame), success=False, error="Forecast refused"
        )
        raise bad_request(str(exc))
    metrics = result.attrs["inference_metrics"]
    forecast_meta = result.attrs["forecast"]
    append_monitor_record(task_name, bundle.model_version, metrics["latency_ms"], metrics["rows"])

    def _bounds(column):
        if result[column].isna().all():
            return None
        return [None if pd.isna(value) else value for value in result[column]]

    return {
        "request_id": result["request_id"].iloc[0],
        "strategy": forecast_meta["strategy"],
        "horizon": int(forecast_meta["horizon"]),
        "times": [str(value) for value in result["time"]],
        "predictions": [None if pd.isna(value) else value for value in result["prediction"]],
        "lower_bounds": _bounds("lower"),
        "upper_bounds": _bounds("upper"),
        "interval_note": forecast_meta["interval_note"],
        "metrics": metrics,
    }
