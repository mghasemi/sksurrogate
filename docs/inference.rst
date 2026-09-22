====================
Batch Inference
====================

The Phase 6 batch API scores tabular data with a portable ``ModelBundle`` and uses
the bundle schema as the serving contract. Input can be a pandas DataFrame or a CSV
file, and the bundle can be passed directly or loaded from a bundle path.

.. code-block:: python

    from SKSurrogate import predict_batch

    predictions = predict_batch(
        "artifacts/customer-churn.bundle",
        "data/customer-churn.csv",
        output_path="predictions/customer-churn.csv",
        request_id="batch-2026-09-22",
    )

The returned DataFrame and optional CSV contain:

* ``prediction``: the fitted model prediction for each input row;
* ``model_version``: the version recorded in the bundle;
* ``request_id``: the supplied request ID or a generated unique ID.

The returned DataFrame also exposes ``inference_metrics`` through ``DataFrame.attrs``
with ``latency_ms``, ``rows``, and ``throughput_rows_per_second``. Online responses
include the same metrics under ``metrics``.

Input validation
================

When the bundle contains a feature schema, ``predict_batch`` checks expected column
names and order, rejects missing or extra columns, compares declared dtypes, and
rejects categorical values not recorded in the schema. Validation failures raise
``SchemaValidationError``. Its ``errors`` attribute contains the individual,
machine-readable error messages.

Batch output is written through a temporary file followed by atomic replacement, so
consumers do not observe a partially written CSV. Bundle loading still performs the
normal format and dependency checks; pass ``strict_dependencies=False`` when dependency
compatibility is managed externally.

Command-line batch prediction
=============================

The optional console command uses the same API and validation path:

.. code-block:: console

    sksurrogate-batch-predict \\
        --bundle artifacts/customer-churn.bundle \\
        --input data/customer-churn.csv \\
        --output predictions/customer-churn.csv \\
        --request-id batch-2026-09-22

Use ``--no-strict-dependencies`` when the runtime dependency check is managed by the
deployment environment. Invalid input exits nonzero and reports the structured schema
errors.

HTTP serving
============

The standard-library WSGI adapter exposes the same validation and prediction path used
by batch inference:

.. code-block:: python

    from SKSurrogate import serve

    serve("artifacts/customer-churn.bundle", host="127.0.0.1", port=8000)

The endpoints are:

* ``GET /health`` returns ``{"status": "ok"}``;
* ``GET /ready`` returns readiness and the loaded ``model_version``;
* ``GET /model`` returns the loaded model version;
* ``POST /predict`` accepts ``{"data": [{...}], "request_id": "..."}`` and returns
    predictions, model version, and request ID.

Schema and malformed-request failures return HTTP 400 with a structured ``error``
object. Unknown routes return HTTP 404. The adapter uses only the Python standard
library in addition to the existing bundle and pandas dependencies.

Latency and throughput measurements are collected for every batch and online request.