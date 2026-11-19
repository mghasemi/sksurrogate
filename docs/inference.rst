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
``"preflight": true`` to validate the frame against the *model bundle's*
feature schema — the columns the estimator actually consumes — and receive the
schema-difference message instead of a raw prediction failure. The bundle
schema (not the task's full registered dataset schema) is the contract, so a
model trained on a curated feature subset of the raw table scores correctly
against a frame holding just those features.

Every request appends a monitoring record (latency, rows, success or error)
keyed to the model version, which the Monitoring page summarizes. Note the
dependency-check asymmetry: the control plane loads bundles with
``strict_dependencies=False`` and leaves compatibility to the deployment
environment, while the library, CLI, and WSGI adapter default to strict
checks.

Serving a pinned model
======================

The endpoints above make the caller choose a model on every call. A deployment
usually wants the opposite: declare once which model a task serves, then let
clients score without naming anything. That declaration is a **serving pin**,
stored per task under the API storage root; it is also what the web UI's top bar
reports.

``POST /api/serving/{task_name}``
    Body ``{"model_version": "<id>"}`` **or** ``{"alias": "production"}`` —
    exactly one; the other, or neither, answers **400** — plus an optional
    ``"note"``. The model must resolve (unregistered bundle file, registered
    version, or registry alias), so a typo answers **404** at declaration time
    instead of failing later on every scored request. Deciding which model is
    served is an edit on the task, so the acting identity is recorded with the
    pin. The response is the status payload below.

``GET /api/serving/{task_name}``
    ``{"active", "resolve_mode", "requested", "model_version", "note",
    "started_at", "started_by", "updated_at", "resolve_error", "history"}``.
    A task that was never configured answers ``active: false`` rather than 404:
    the UI's question "is this task serving?" is a data question, not an error
    path. ``requested`` is what the operator pinned, ``model_version`` the
    model the pin resolves to *now* (they differ for an alias pin), and
    ``history`` is the append-only sequence of ``start`` / ``replace`` / ``stop``
    events with actor and timestamp.

``GET /api/serving``
    Active pins across every task the caller may see, as
    ``{"serving": [...], "count": n}`` — summaries only. This is what a UI or
    monitor polls to declare that the system is serving a model.

``DELETE /api/serving/{task_name}``
    Stops serving. The pin stays in ``history``; **409** when nothing is being
    served.

``POST /api/serving/{task_name}/predict``
    The serving endpoint. Body ``{"rows" | "partition", "request_id"?,
    "preflight"?}`` — note the absence of ``model_version``/``alias``: the pin
    decides. A task that is not serving answers **409**, so a client can never
    be silently scored by a model it did not ask for. Predictions, input
    projection (``ignored_columns``), pre-flight and monitoring behave exactly
    as in the per-request endpoint above — both routers share one prediction
    core, so they cannot drift. The response adds ``resolve_mode`` and
    ``requested`` beside the resolved ``model_version``, so a caller can tell
    which model answered even when an alias moved underneath it.

.. code-block:: console

   curl -s -X POST http://127.0.0.1:8013/api/serving/customer-churn \\
     -H "Content-Type: application/json" -d '{"alias": "production"}'
   curl -s -X POST http://127.0.0.1:8013/api/serving/customer-churn/predict \\
     -H "Content-Type: application/json" \\
     -d '{"rows": [{"tenure": 24, "monthly": 71.3}]}'

Two resolution modes
--------------------

``resolve_mode`` follows from what you pinned, and the difference is the point:

* ``version`` — the pin names an exact model version. A named alias used this
  way is frozen to the version it pointed at when the pin was created. Later
  registry promotions and rollbacks do **not** change what is served.
* ``alias`` — the pin names a registry pointer (a lifecycle alias such as
  ``production``, or a named alias). Every serving request re-resolves it, so a
  later promotion or rollback **does** change what is served without touching
  the pin. The status read reports the version the alias currently resolves to,
  and ``resolve_error`` reports (while leaving the pin in place) that the
  pointer's target has disappeared.

Every serving request appends the same monitoring record as the per-request
path, keyed to the *served* model version, so Monitoring's serving summary
covers both surfaces. Serving through the control plane inherits its
``strict_dependencies=False`` bundle loading described above; to run a single
bundle with strict dependency checks and no control-plane state, use the
standalone ``SKSurrogate.serve`` WSGI adapter instead.
