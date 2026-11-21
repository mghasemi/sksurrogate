"""Batch inference helpers for portable model bundles."""

import argparse
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

import pandas as pd

from .modelbundle import ModelBundle, load_bundle


class SchemaValidationError(ValueError):
    """Structured input or output schema validation failure."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("Input schema validation failed: " + "; ".join(self.errors))


def _validate_input(bundle, frame):
    """Validate a batch input against a bundle's declared schema or design.

    Temporal bundles — v2 records carrying the ``temporal`` block written by
    plan phase P6 in ``TIME_SERIES_PLAN.md`` — are validated as *history*
    frames (stored spec, row order and minimum length), not feature tables;
    tabular bundles keep the classic column/dtype/categorical checks.

    :raises SchemaValidationError: when the frame violates the bundle's schema
        or temporal design.
    """
    if getattr(bundle, "temporal", None) is not None:
        return _check_temporal_history(bundle, frame)
    schema = bundle.schema
    if not schema:
        return frame
    expected = list(schema)
    actual = list(frame.columns)
    errors = []
    missing = [column for column in expected if column not in actual]
    extra = [column for column in actual if column not in expected]
    if missing:
        errors.append("missing columns: %s" % ", ".join(missing))
    if extra:
        errors.append("extra columns: %s" % ", ".join(extra))
    if not missing and not extra and actual != expected:
        errors.append("columns are reordered; expected: %s" % ", ".join(expected))
    for column in expected:
        if column not in frame:
            continue
        expected_dtype = schema[column].get("dtype") if isinstance(schema[column], dict) else None
        if expected_dtype is not None and str(frame[column].dtype) != expected_dtype:
            errors.append(
                "column %r has dtype %r; expected %r"
                % (column, str(frame[column].dtype), expected_dtype)
            )
        expected_categories = (
            schema[column].get("categorical_values", [])
            if isinstance(schema[column], dict)
            else []
        )
        # Continuous float columns are not categorical: unseen values are expected.
        if expected_categories and not pd.api.types.is_float_dtype(frame[column].dtype):
            unknown = sorted(
                {str(value) for value in frame[column].dropna().unique()}
                - set(expected_categories)
            )
            if unknown:
                errors.append("column %r has unknown categories: %s" % (column, ", ".join(unknown)))
    if errors:
        raise SchemaValidationError(errors)
    return frame[expected]


def _check_temporal_history(bundle, frame):
    """Validate a history frame against a temporal bundle's stored design.

    Delegates to :meth:`SKSurrogate.temporal.TemporalServeState.validate_history` —
    the same code path ``serve_forecast`` uses at prediction time — so batch and
    online serving reject identical inputs for identical reasons (plan phase P6).

    :raises SchemaValidationError: when history lacks required columns, its
        time axis is not strictly increasing, it mixes series where the spec
        declares one, or it is shorter than the design's minimum length.
    """
    from .temporal import TemporalServeState

    state = TemporalServeState.from_dict(bundle.temporal)
    try:
        checked, _series_label = state.validate_history(frame)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError([str(exc)]) from exc
    return checked


def _read_input(input_data):
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()
    return pd.read_csv(input_data)


def _as_bundle(bundle, strict_dependencies=True):
    """Accept a :class:`ModelBundle` or a path to a saved one."""
    if isinstance(bundle, (str, Path)):
        bundle = load_bundle(bundle, strict_dependencies=strict_dependencies)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("bundle must be a ModelBundle or bundle path")
    return bundle


def _atomic_write_csv(result, output_path):
    """Write ``result`` to ``output_path`` atomically (tmp file + fsync + rename)."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", dir=str(destination.parent), delete=False, encoding="utf-8"
    ) as temporary:
        temporary_path = Path(temporary.name)
        result.to_csv(temporary, index=False)
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(str(temporary_path), str(destination))


def predict_batch(
    bundle,
    input_data,
    output_path=None,
    *,
    request_id=None,
    strict_dependencies=True,
):
    """Run stable batch predictions and optionally write a CSV atomically.

    ``bundle`` may be a :class:`ModelBundle` or a bundle path; ``input_data`` is
    a pandas DataFrame or a CSV path. The returned DataFrame always contains
    ``prediction``, ``model_version``, and ``request_id`` columns.

    Temporal bundles — v2 records with the ``temporal`` block written by plan
    phase P6 in ``TIME_SERIES_PLAN.md`` — treat ``input_data`` as *history*, not a
    feature table: it is validated against the stored design, and one row is
    returned per forecast step at the bundle's design horizon. For multi-horizon
    requests, recursive serving or interval bounds use
    :func:`predict_batch_forecast`.
    """
    bundle = _as_bundle(bundle, strict_dependencies)
    if getattr(bundle, "temporal", None) is not None:
        return predict_batch_forecast(
            bundle,
            input_data,
            output_path,
            request_id=request_id,
            strict_dependencies=strict_dependencies,
        )
    started = time.perf_counter()
    frame = _validate_input(bundle, _read_input(input_data))
    predictions = bundle.predict(frame)
    elapsed = max(time.perf_counter() - started, 0.0)
    result = pd.DataFrame(
        {
            "prediction": predictions,
            "model_version": bundle.model_version,
            "request_id": request_id or uuid.uuid4().hex,
        }
    )
    result.attrs["inference_metrics"] = {
        "latency_ms": elapsed * 1000.0,
        "rows": int(len(frame)),
        "throughput_rows_per_second": len(frame) / elapsed if elapsed else 0.0,
    }
    if output_path is not None:
        _atomic_write_csv(result, output_path)
    return result


def predict_batch_forecast(
    bundle,
    history,
    output_path=None,
    *,
    horizon=None,
    future_exogenous=None,
    request_id=None,
    strict_dependencies=True,
):
    """Serve a multi-step forecast from a temporal model bundle.

    ``bundle`` may be a :class:`ModelBundle` or a path to one and must carry the
    v2 ``temporal`` block (plan phase P6 in ``TIME_SERIES_PLAN.md``); tabular
    bundles raise ``ValueError`` — forecasting is refused, not faked. ``history``
    is one series of past observations as a DataFrame or CSV path; it is
    validated against the stored spec before anything is predicted
    (:class:`SchemaValidationError` on violation).

    The returned DataFrame has one row per forecast step with columns
    ``time``, ``series_id``, ``prediction``, ``lower`` and ``upper`` plus
    ``model_version`` and ``request_id``. ``lower``/``upper`` are NaN unless the
    bundle carries interval calibration; the serving metadata (strategy,
    requested horizon, interval note) is in ``result.attrs["forecast"]`` and
    latency metrics in ``result.attrs["inference_metrics"]``, exactly like
    :func:`predict_batch`. With ``output_path`` the table is written atomically.

    A direct-strategy bundle answers only its design horizon; a recursive one
    accepts any positive ``horizon`` — see ``serve_forecast`` in
    ``SKSurrogate/temporal.py`` for the full contract.
    """
    bundle = _as_bundle(bundle, strict_dependencies)
    frame = _validate_input(bundle, _read_input(history))  # temporal branch
    from .temporal import serve_forecast

    started = time.perf_counter()
    forecast = serve_forecast(
        bundle, frame, horizon=horizon, future_exogenous=future_exogenous
    )
    elapsed = max(time.perf_counter() - started, 0.0)
    result = forecast.to_frame().rename(columns={"mean": "prediction"})[
        ["time", "series_id", "prediction", "lower", "upper"]
    ]
    result["model_version"] = bundle.model_version
    result["request_id"] = request_id or uuid.uuid4().hex
    result.attrs["inference_metrics"] = {
        "latency_ms": elapsed * 1000.0,
        "rows": int(len(result)),
        "throughput_rows_per_second": len(result) / elapsed if elapsed else 0.0,
    }
    result.attrs["forecast"] = {
        "strategy": forecast.strategy,
        "horizon": int(forecast.horizon),
        "interval_note": forecast.interval_note,
    }
    if output_path is not None:
        _atomic_write_csv(result, output_path)
    return result


def main(argv=None):
    """Run batch prediction from the command line."""
    parser = argparse.ArgumentParser(description="Run predictions from a SKSurrogate model bundle.")
    parser.add_argument("--bundle", required=True, help="Path to a model bundle")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output prediction CSV path")
    parser.add_argument("--request-id", help="Request ID included in every output row")
    parser.add_argument(
        "--no-strict-dependencies",
        action="store_true",
        help="Skip runtime dependency compatibility checks",
    )
    arguments = parser.parse_args(argv)
    try:
        predict_batch(
            arguments.bundle,
            arguments.input,
            arguments.output,
            request_id=arguments.request_id,
            strict_dependencies=not arguments.no_strict_dependencies,
        )
    except SchemaValidationError as exc:
        parser.error(json.dumps({"errors": exc.errors}))
    return 0


def create_app(bundle, *, strict_dependencies=True):
    """Create a dependency-free WSGI application for online predictions.

    Routes: ``GET /health``, ``GET /ready|/model`` and ``POST /predict`` (a list
    of feature rows). Temporal bundles additionally expose ``POST /forecast`` —
    payload ``{"history": [...], "horizon"?: int, "future_exogenous"?: [...]}``
    — which serves a multi-step forecast (plan phase P6 in
    ``TIME_SERIES_PLAN.md``; tabular bundles get a 400 there).
    """
    bundle = _as_bundle(bundle, strict_dependencies)

    def response(start_response, status, payload):
        body = json.dumps(payload, default=str).encode("utf-8")
        start_response(
            status,
            [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ],
        )
        return [body]

    def _handle_forecast(environ, start_response):
        """POST /forecast: one multi-step forecast from a temporal bundle."""
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
            payload = json.loads(environ["wsgi.input"].read(length) or b"{}")
            rows = payload.get("history")
            if not isinstance(rows, list):
                raise ValueError("request field 'history' must be a list of row objects")
            future_rows = payload.get("future_exogenous")
            future_frame = (
                pd.DataFrame(future_rows)
                if future_rows is not None and isinstance(future_rows, list)
                else None
            )
            result = predict_batch_forecast(
                bundle,
                pd.DataFrame(rows),
                horizon=payload.get("horizon"),
                future_exogenous=future_frame,
                request_id=payload.get("request_id"),
                strict_dependencies=strict_dependencies,
            )
            forecast_meta = result.attrs["forecast"]

            def _bounds(column):
                if result[column].isna().all():
                    return None
                return json.loads(result[column].to_json(orient="records"))

            return response(
                start_response,
                "200 OK",
                {
                    "model_version": bundle.model_version,
                    "request_id": result["request_id"].iloc[0],
                    "strategy": forecast_meta["strategy"],
                    "horizon": int(forecast_meta["horizon"]),
                    "times": [str(value) for value in result["time"]],
                    "predictions": json.loads(result["prediction"].to_json(orient="records")),
                    "lower_bounds": _bounds("lower"),
                    "upper_bounds": _bounds("upper"),
                    "interval_note": forecast_meta["interval_note"],
                    "metrics": result.attrs["inference_metrics"],
                },
            )
        except SchemaValidationError as exc:
            return response(
                start_response,
                "400 Bad Request",
                {"error": {"type": "SchemaValidationError", "messages": exc.errors}},
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return response(
                start_response,
                "400 Bad Request",
                {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            )

    def application(environ, start_response):
        method = environ.get("REQUEST_METHOD", "GET")
        path = environ.get("PATH_INFO", "/")
        if method == "GET" and path == "/health":
            return response(start_response, "200 OK", {"status": "ok"})
        if method == "GET" and path in {"/ready", "/model"}:
            return response(
                start_response,
                "200 OK",
                {"status": "ready", "model_version": bundle.model_version},
            )
        if method == "POST" and path == "/forecast":
            return _handle_forecast(environ, start_response)
        if method != "POST" or path != "/predict":
            return response(start_response, "404 Not Found", {"error": "not found"})
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
            payload = json.loads(environ["wsgi.input"].read(length) or b"{}")
            rows = payload.get("data")
            if not isinstance(rows, list):
                raise ValueError("request field 'data' must be a list of row objects")
            result = predict_batch(
                bundle,
                pd.DataFrame(rows),
                request_id=payload.get("request_id"),
                strict_dependencies=strict_dependencies,
            )
            return response(
                start_response,
                "200 OK",
                {
                    "model_version": bundle.model_version,
                    "request_id": result["request_id"].iloc[0],
                    "predictions": json.loads(result["prediction"].to_json(orient="records")),
                    "metrics": result.attrs["inference_metrics"],
                },
            )
        except SchemaValidationError as exc:
            return response(
                start_response,
                "400 Bad Request",
                {"error": {"type": "SchemaValidationError", "messages": exc.errors}},
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return response(
                start_response,
                "400 Bad Request",
                {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            )

    return application


def serve(bundle, host="127.0.0.1", port=8000, *, strict_dependencies=True):
    """Serve a bundle with the standard-library WSGI server."""
    from wsgiref.simple_server import make_server

    application = create_app(bundle, strict_dependencies=strict_dependencies)
    with make_server(host, port, application) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()