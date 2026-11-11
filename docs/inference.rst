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

Control-plane inference
=======================

Inside the SKSurrogate control plane the same ``predict_batch`` path is exposed
over HTTP and recorded in monitoring. Resolve a registered model either by an
exact ``model_version`` or by a lifecycle ``alias`` (for example
``production``); exactly one is required, and alias resolution goes through
``ModelRegistry``:

.. code-block:: console

   curl -s -X POST http://127.0.0.1:8013/api/inference/customer-churn/predict \
     -H "Content-Type: application/json" \
     -d '{"alias": "production", "rows": [{"tenure": 24, "monthly": 71.3}]}'

``POST /api/inference/{task_name}/predict`` returns predictions as JSON
together with the model version, request ID, latency/throughput metrics, and
``ignored_columns``. ``POST /api/inference/{task_name}/predict-batch`` writes a
CSV under the task's predictions directory, returns its ``output_path``, and
best-effort registers it as a prediction artifact so lineage stays complete.

Input is either inline ``rows`` — the serving payload, validated exactly as
sent — or a stored ``partition``, which is registered training data and is
therefore projected onto the bundle schema first (the target and other
non-feature columns are dropped and reported as ``ignored_columns``). Set
``"preflight": true`` to validate the frame against the task's registered
dataset schema and receive the schema-difference message instead of a raw
prediction failure.

Every request appends a monitoring record (latency, rows, success or error)
keyed to the model version, which the Monitoring page summarizes. Note the
dependency-check asymmetry: the control plane loads bundles with
``strict_dependencies=False`` and leaves compatibility to the deployment
environment, while the library, CLI, and WSGI adapter default to strict
checks.