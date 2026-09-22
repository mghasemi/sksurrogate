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
        if expected_categories:
            unknown = sorted(
                {str(value) for value in frame[column].dropna().unique()}
                - set(expected_categories)
            )
            if unknown:
                errors.append("column %r has unknown categories: %s" % (column, ", ".join(unknown)))
    if errors:
        raise SchemaValidationError(errors)
    return frame[expected]


def _read_input(input_data):
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()
    return pd.read_csv(input_data)


def predict_batch(
    bundle,
    input_data,
    output_path=None,
    *,
    request_id=None,
    strict_dependencies=True,
):
    """Run stable batch predictions and optionally write a CSV atomically.

    ``bundle`` may be a :class:`ModelBundle` or a bundle path. ``input_data`` may
    be a pandas DataFrame or a CSV path. The returned DataFrame always contains
    ``prediction``, ``model_version``, and ``request_id`` columns.
    """
    if isinstance(bundle, (str, Path)):
        bundle = load_bundle(bundle, strict_dependencies=strict_dependencies)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("bundle must be a ModelBundle or bundle path")
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
    """Create a dependency-free WSGI application for online predictions."""
    if isinstance(bundle, (str, Path)):
        bundle = load_bundle(bundle, strict_dependencies=strict_dependencies)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("bundle must be a ModelBundle or bundle path")

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