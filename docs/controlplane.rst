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
the ``sksurrogate-data`` volume (``/data``). Set ``SKSURROGATE_API_KEY`` for
non-local use and ``--build-arg EXTRAS="..."`` to add optional packages.

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

Clients can send the key in an ``X-API-Key`` header. The web UI's Settings
page accepts the API base URL and key; its requests use the key header, and
job WebSocket connections send it as a query parameter. The UI keeps these
settings in browser local storage, so use a trusted browser profile and avoid
sharing it with untrusted users.

When ``SKSURROGATE_API_KEY`` is unset, the API is open to callers that can
reach it. Do not expose a keyless instance to an untrusted network. For
cross-origin browser clients, set ``SKSURROGATE_API_CORS_ORIGINS`` to a
comma-separated list of explicit origins. Localhost and ``127.0.0.1`` origins
are allowed for development by default.

API and UI coverage
===================

The API groups endpoints by workflow: dataset management, synthetic data,
experiments and evaluation, sensitivity analysis, bundles and registry,
inference, quality gates, monitoring, retraining, lineage, and background
jobs. See the interactive OpenAPI reference at ``/docs`` for request and
response schemas. The web UI provides corresponding workflow pages, including
dashboard, datasets, experiments, evaluation, model bundles, registry and
deployment, inference, monitoring, retraining, and settings.

Normal registry promotion requires a passing bundle-quality report by default;
the current default check is schema compatibility, not a configured metric
threshold policy. Automated retraining promotion is a separate workflow: it
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
