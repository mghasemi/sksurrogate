=============================
Web UI and Control-Plane API
=============================

SKSurrogate includes an optional FastAPI control-plane API and a separate
React web application. The API orchestrates the Python package and exposes
REST endpoints (plus WebSocket job progress); the web application calls the
API and does not access models or storage directly.

These components are run from a source checkout. The API and web UI have
dependencies beyond those required by the core Python package.

Local setup
===========

From the repository root, install the Python dependencies and start the API
on the port expected by the web UI's development proxy:

.. code-block:: console

   python -m pip install -r requirements.txt
   uvicorn api.main:app --reload --host 127.0.0.1 --port 8013

The API health check is available at ``http://127.0.0.1:8013/api/health``.
FastAPI's interactive endpoint reference is at
``http://127.0.0.1:8013/docs``; the OpenAPI schema is at
``http://127.0.0.1:8013/openapi.json``.

In another terminal, install and run the web UI:

.. code-block:: console

   cd webui
   npm ci
   npm run dev

Open the local URL printed by Vite. Its development server forwards ``/api``
requests and job-progress WebSockets to ``http://localhost:8013``. Build the
static frontend with ``npm run build`` from ``webui/``.

Docker
======

.. code-block:: console

   docker compose --profile prod up --build -d   # UI + API on :8013
   docker compose --profile dev up --build       # hot reload, UI on :5174

The ``prod`` image serves the built UI and API from one port, runs as a
non-root user, exposes ``/api/health`` as its healthcheck and stores data in
the ``sksurrogate-data`` volume (``/data``). The Compose file forwards
``SKSURROGATE_API_KEY``, ``SKSURROGATE_REQUIRE_AUTH`` and
``SKSURROGATE_API_CORS_ORIGINS`` into the container: set the key for non-local
use, and ``SKSURROGATE_REQUIRE_AUTH=1`` to turn a missing key into a startup
error. Use ``--build-arg EXTRAS="..."`` to add optional packages.

Configuration and storage
=========================

The API stores data beneath ``./var/sksurrogate-api`` relative to its working
directory by default. Set ``SKSURROGATE_API_HOME`` to use another root. The
root contains per-task datasets and bundles, SQLite tracking databases,
registry data, predictions, monitoring records, background-job records, and
checkpoints. Keep this directory backed up according to your retention needs.

Authentication is optional for local development. To require a shared API key
for endpoints under ``/api`` other than ``/api/health``, set
``SKSURROGATE_API_KEY`` before starting the service:

.. code-block:: console

   export SKSURROGATE_API_KEY='replace-with-a-secret'
   uvicorn api.main:app --reload --host 127.0.0.1 --port 8013

Clients can send the key in an ``X-API-Key`` header or as an
``Authorization: Bearer`` token. The web UI's Settings page accepts the API
base URL and key; its requests use the key header, and job WebSocket
connections send it as a query parameter. The UI keeps these settings in
browser local storage, so use a trusted browser profile and avoid sharing it
with untrusted users.

When ``SKSURROGATE_API_KEY`` is unset, the API is open to callers that can
reach it — the server logs a startup warning whenever it starts in this mode.
Do not expose a keyless instance to an untrusted network; for deployments whose
port may be published (the Docker entrypoint binds all interfaces), set
``SKSURROGATE_REQUIRE_AUTH=1`` (``true`` and ``yes`` also work) so the API
refuses to start without a key. For
cross-origin browser clients, set ``SKSURROGATE_API_CORS_ORIGINS`` to a
comma-separated list of explicit origins. Localhost and ``127.0.0.1`` origins
are allowed for development by default.

Optional LLM-assisted project synthesis
=======================================

Projects can draft a cited note from computed artifacts through a
**user-supplied, OpenAI-compatible** chat-completions endpoint. The feature is
off unless configured, adds no package dependency, and is implemented over the
standard library. Set these on the API process and restart it:

.. code-block:: console

   export SKSURROGATE_LLM_BASE_URL='https://api.openai.com/v1'
   export SKSURROGATE_LLM_MODEL='your-model-name'
   export SKSURROGATE_LLM_API_KEY='your-key'          # optional
   export SKSURROGATE_LLM_TIMEOUT='60'                # optional seconds
   export SKSURROGATE_LLM_MAX_TOKENS='700'            # optional output budget

``GET /api/synthesis/status`` reports whether a provider is configured without
echoing credentials. Synthesis sends only aggregated computed facts (dataset
profiles, scalar run metrics, bundle metadata, explanation summaries, stored
alerts) — never raw dataset rows. Drafts must cite the supplied sources and are
not saved automatically; persisting one is an explicit note action that records
the model id, citations, and computed facts. The prompt is not stored, while the
job result and any saved note follow the normal jobs/notes storage rules; the
operator is responsible for the chosen provider's retention behaviour. See
:doc:`research-projects` for the full contract and limitations.

Job logs
========

Background jobs write structured log entries while they run. Entries come from
two sources: records emitted by the ``SKSurrogate`` core through the standard
library ``logging`` module (see :doc:`telemetry`), and lifecycle lines written by
the job manager itself (accepted, started, completed, failed). Each entry is a
JSON-safe dictionary::

    {"seq": 42, "ts": "2026-11-09T03:15:00.512+00:00", "level": "WARNING",
     "logger": "sksurrogate.api", "message": "trial 4 (explore) failed ...",
     "job_id": "f62e8efe...", "task_name": "texty", "kind": "experiment",
     "trial": 4, "phase": "explore", "status": "failed"}

Failure entries carry the full traceback under ``detail``. Levels follow the
standard library ordering: ``DEBUG`` < ``INFO`` < ``WARNING`` < ``ERROR`` <
``CRITICAL``. The core emits INFO for progress, DEBUG for surrogate internals,
WARNING for degraded fits and repeated trial failures (capped at five per job
and status, with the rest kept at DEBUG), and ERROR when a search or a job
fails. The capture handler is attached to the ``SKSurrogate`` and
``sksurrogate.api`` loggers and only forwards records while a job is bound to
the current execution context, so concurrent jobs never mix their lines and
ordinary API request handling is not captured.

Entries are kept in an in-memory ring buffer per job (2 000 entries) and
appended to ``<SKSURROGATE_API_HOME>/logs/<job_id>.jsonl``, so history survives
an API restart. Job records carry a ``log_counts`` summary
(``{DEBUG, INFO, WARNING, ERROR, CRITICAL}``) for list views; the entries
themselves are read through the endpoints below.

``GET /api/logs``
    Browse across jobs. Query parameters: ``job_id``, ``task_name`` (one
    dataset/task), ``project_id`` (one research project), ``kind``, ``q``
    (substring over message/logger/task), ``limit`` (1–2000, default 200), and
    either ``level`` (minimum severity — ``WARNING`` returns WARNING, ERROR and
    CRITICAL) or ``levels`` (explicit comma-separated subset). The response is
    ``{"entries": [...], "counts": {...}, "total": n, "levels": [...],
    "project_id"?, "job_count"?}`` with entries newest first.

    ``project_id`` resolves scope from the project store rather than from the
    entries: the referenced jobs are the project's artifacts of kind ``job``, so
    jobs linked before this filter existed are included, and a job linked to two
    projects appears under both. An unknown or malformed project id returns 404;
    a project with no linked jobs returns an empty result with ``job_count: 0``
    instead of scanning the whole store.

``GET /api/logs/{job_id}``
    Tail one job; accepts the same severity, ``kind``, ``task_name`` and ``q``
    filters plus ``since_seq`` for polling (``project_id`` is only meaningful on
    the cross-job endpoint). Returns 404 for an unknown job.

``WS /api/logs/{job_id}/stream``
    Incremental tail: accepts ``level``/``levels``/``since_seq`` as query
    parameters and pushes ``{"entries": [...], "since_seq": n, "status": ...}``
    frames until the job reaches a terminal state. The existing
    ``/api/jobs/{job_id}/stream`` status stream is unchanged.

Unknown severity names are rejected with HTTP 422. Log files are not pruned
automatically; the API operator owns retention under ``logs/``.

API and UI coverage
===================

The API groups endpoints by workflow: dataset management, synthetic data,
experiments and evaluation, sensitivity analysis, bundles and registry,
inference, quality gates, monitoring, retraining, lineage, research projects,
optional LLM-assisted project synthesis, and background jobs. See the interactive OpenAPI reference at ``/docs`` for request and
response schemas. The web UI provides corresponding workflow pages, including
dashboard, datasets, experiments, evaluation, model bundles, registry and
deployment, inference, monitoring, retraining, and settings.

Normal registry promotion requires a passing bundle-quality report by default;
the default check verifies that the registered bundle can be loaded and does
not apply custom metric thresholds. There is no configured metric-threshold
policy on the normal promotion endpoint. Automated retraining promotion is a
separate workflow: it
requires at least one finite metric threshold and can promote automatically
only to ``staging``. Production promotion remains approval-gated. See
:doc:`phase9` for the retraining trigger and promotion contract.

Experiment and retraining search requests use ``max_generations`` for the
generation budget. Existing clients may continue sending ``max_generation``
during migration; the API accepts that legacy spelling, while OpenAPI and the
web UI use the corrected field. Sending both names in one request is rejected.

The Python library can also be used without running either service. See
:doc:`code` for the package API and the other guides for library workflows.
For a complete UI walkthrough with screenshots and links from each screen to
its corresponding SKSurrogate feature, see :doc:`ui-manual`; for an end-to-end
example with screenshots and API equivalents, see :doc:`walkthrough`.
