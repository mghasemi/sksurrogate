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
``SKSURROGATE_API_KEY``, ``SKSURROGATE_REQUIRE_AUTH``,
``SKSURROGATE_ADMIN_USERNAME``, ``SKSURROGATE_ADMIN_PASSWORD``,
``SKSURROGATE_SESSION_TTL``, ``SKSURROGATE_SESSION_COOKIE_SECURE`` and
``SKSURROGATE_API_CORS_ORIGINS`` into the container. Set the key for non-local
use; set ``SKSURROGATE_REQUIRE_AUTH=1`` to require an identity — a shared key, a
per-user API key, a browser session, or the one-shot administrator bootstrap —
and to make a deployment with none of those refuse to start. Use
``--build-arg EXTRAS="..."`` to add optional packages.

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

The key above is a single shared secret: it authenticates *a* caller, not an
account. For per-user accounts, browser sessions, per-user API keys and sharing
of datasets/models/projects between accounts, see
:doc:`auth-and-sharing` — the same ``SKSURROGATE_REQUIRE_AUTH`` switch enables
both, and the shared key remains available as a break-glass administrator.

The shared key protects the entire control plane, not just the inference
routes: registry promotion and rollback, artifact deletion, dataset uploads
and retraining are all reachable with the same key. When a key is configured,
the OpenAPI schema and interactive documentation (``/openapi.json``,
``/docs``, ``/redoc``) require it as well, so an unauthenticated client cannot
enumerate the request/response surface. Rejected requests are logged at
``WARNING`` with the HTTP method, path and client host; the supplied
credential is never written to the log.

Caller-supplied names that become filesystem path segments — task names, model
versions, dataset partitions, request IDs and job IDs — are validated before
use. Values containing path separators, control characters or ``..`` are
rejected with HTTP 400, and bundle and partition joins are additionally
resolved and proven to stay inside the storage root. Before this validation a
keyed caller could read arbitrary ``*.csv`` files with a traversal
``partition`` and load any ``*.bundle`` file on the host (``joblib``) with a
traversal ``model_version``.

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

Dataset column exclusions
=========================

A dataset can mark feature columns as **ignored**. The choice lives in the task
metadata (``ignored_columns``), not in the stored CSV, so it is reversible and
dataset fingerprints stay valid.

``GET /api/datasets/{task_name}/{partition}/preprocessing``
    One round-trip for the Data Preprocessing page: row/column counts, per-column
    dtype and deduced type, missing count/percentage, complete-row share, numeric
    quantiles or top categories, bounded profile diagnostics (duplicate rows and
    features, constant/all-missing columns, missingness patterns), a row preview,
    ignored-column flags and the supported imputation methods.

``POST /api/datasets/{task_name}/impute``
    Body ``{"partition", "method", "columns"?, "params"?, "output_partition"?,
    "preview_only"?, "drop_missing_target"?}``. Methods: ``drop_rows``, ``mean``,
    ``median``, ``most_frequent``, ``constant``, ``knn``, ``iterative``;
    numeric-only methods reject text columns with **400**. Target imputation and
    ignored columns are refused. ``preview_only`` returns the before/after report
    without writing; otherwise the result is saved as a **new** partition,
    registered in the partition history and registry lineage, and an
    ``imputation_history`` entry records the recipe. The source partition is
    never modified and an existing output name answers **409**.

``GET /api/datasets/{task_name}/{partition}/encoding-preview``
    Query ``sample_rows`` (default 1000) and ``max_outputs`` (default 40). Fits
    the same ``FoldLocalDataPreprocessor`` used by model pipelines on a bounded
    sample and returns the encoded feature names, a preview matrix, per-column
    kind/cardinality/output names and the fold-local caveats. Read-only.

``POST /api/datasets/{task_name}/clip``
    Body ``{"partition", "method", "columns"?, "params"?, "output_partition"?,
    "preview_only"?}``. Methods: ``iqr`` (``factor``), ``quantile`` (``lower``,
    ``upper``), ``zscore`` (``threshold``). Numeric features only; the target and
    ignored columns are refused with **400**. The report gives bounds, tail
    counts, clipped share and before/after moments. Saving writes a **new**
    partition, a ``clipping_history`` recipe and registry lineage; an existing
    output name answers **409**.

``GET /api/datasets/{task_name}/{partition}/duplicates``
    Query ``max_rows`` (default 20000; ``0`` = full frame). Counts exact
    feature-duplicate rows/groups, full-row duplicates, groups with conflicting
    target values, and the distributional impact of removal (target balance,
    numeric means), plus explicit warnings about sample change, balance shift
    and cross-validation leakage. Read-only.

``POST /api/datasets/{task_name}/deduplicate``
    Body ``{"partition", "columns"?, "keep", "output_partition"?,
    "preview_only"?}``. ``keep`` is ``first``/``last``. Returns the same impact
    report; saving writes a **new** partition with a ``deduplication_history``
    recipe and registry lineage. Source untouched; existing output → **409**.

``PUT /api/datasets/{task_name}/ignored-columns``
    Body ``{"columns": ["Surname", "Geography"]}`` replaces the list and returns
    ``ignored_columns``, ``effective_features`` and ``effective_feature_count``.
    Unknown column names and the target column answer **400**; a task without a
    registered dataset answers **404**. The mutation requires edit access to the
    task.

``POST /api/datasets/{task_name}/register``
    Accepts an optional multipart field ``ignored_columns`` (a JSON array), so
    review-time exclusions are stored in the same registration.

``GET /api/datasets/{task_name}``
    Returns ``ignored_columns`` and ``effective_feature_count`` beside the
    registered schema and partitions.

Ignored columns are dropped wherever a stored partition becomes a feature matrix:
experiments, pipeline optimization, baselines, explanations, sensitivity and
correlation analysis, dataset profiles, nested CV, diagnostic curves, monitoring
drift, synthetic generation and the LLM research-synthesis facts. They are still
returned by the partition preview (so the UI can show and restore them). Bundles
keep the schema they were trained with: retrain after changing the list before
serving through those bundles.

Job status
==========

``GET /api/jobs``
    Full job records, newest first, restricted to tasks the caller can see. The
    optional ``task_name`` parameter narrows the list to one dataset/task.

``GET /api/jobs/active``
    Summaries only — ``job_id``, ``task_name``, ``kind``, ``backend``,
    ``status``, ``created_at``, ``updated_at`` and ``log_counts`` — for jobs in
    the ``queued`` or ``running`` state, as ``{"jobs": [...], "count": n}``.
    This backs the UI's always-on activity badge, which polls it every few
    seconds; results and resume payloads are deliberately excluded because a
    finished experiment carries its whole evaluation history.

``GET /api/jobs/{job_id}``
    One full record (404 when unknown), and ``WS /api/jobs/{job_id}/stream``
    pushes its status until it reaches a terminal state.

Naming jobs and experiments
===========================

A ``job_id`` is a machine identity and never changes; every record also carries a
human name so work can be tracked without quoting ids.

``display_name``
    Present on every full record and on the active summaries. It is the custom
    ``label`` when one is set, otherwise an auto-generated
    ``<kind> · <task> · <local creation time>``. Older records are rendered the
    same way — the name is derived on read, so nothing needs to be migrated.

``label``
    The optional user-chosen name (trimmed, at most 80 characters, control
    characters rejected with HTTP 400; blank or ``null`` clears it). Experiment
    and pipeline-optimization submissions accept ``label`` in the request body,
    and the produced bundle stores the same name in its metadata so the
    friendly name survives with the artifact.

``PATCH /api/jobs/{job_id}``
    Body ``{"label": "credit-card baseline"}`` renames a job; ``{"label": null}``
    clears the custom name and returns it to the auto name. The mutation requires
    edit access to the job's task and records ``aliased_at``/``aliased_by`` on the
    record. Routes, checkpoints, project artifacts and logs keep using
    ``job_id``.

Naming model versions
=====================

Registered bundles already expose *lifecycle* aliases (``candidate``,
``staging``, ``production``, ``latest``) that promotion and rollback rewrite. A
**named alias** is a separate, user-chosen handle pointing at an immutable
``model_version`` — stable across promotions:

``GET /api/registry/{task}/aliases``
    ``{"task_name": ..., "aliases": {name: {model_version, created_at, created_by}}}``.

``POST /api/registry/{task}/aliases``
    Body ``{"name": "baseline", "model_version": "<version id>"}``. Names are
    lowercase slugs (letters, digits, ``.``, ``_``, ``-``; max 64 characters) and
    may not collide with lifecycle names; re-pointing an existing alias records
    the previous version. Unknown versions answer 404.

``DELETE /api/registry/{task}/aliases/{name}``
    Removes the alias; the version it pointed at is untouched.

Aliases resolve wherever a version id is accepted: ``ModelRegistry.load`` /
``load_version`` in the library, registry promotion, and the inference endpoints
(``model_version`` falls back to the registry when no unregistered ``.bundle``
file matches, and ``alias`` also accepts a named alias). Every set/delete is
appended to the registry history and audit log with the acting identity.

Pausing, stopping and resuming
==============================

Long searches checkpoint themselves (EOA writes ``<task>.eoa`` before every
generation; the surrogate stage saves per iteration), so a run can be stopped
and continued later. A job is never killed — Python threads cannot be — it is
asked to return at its next candidate boundary, which keeps the checkpoint
consistent.

``POST /api/jobs/{job_id}/stop``
    Body ``{"mode": "pause" | "cancel"}`` (default ``pause``). Requires edit
    access to the job's task. ``pause`` keeps the checkpoint and marks the job
    ``paused``; ``cancel`` marks it ``cancelled``. The request is recorded on the
    record (``stop_requested_at``/``stop_mode``) so the UI can show *stopping…*
    until the worker observes it. A stop that arrives while the job is still
    ``queued`` is honoured before the job body runs. Jobs that are not live in
    this process return **409**, as do job kinds without a checkpoint when
    ``pause`` is requested (``sensitivity``, ``nested-cv``, ``synthesis`` can
    only be cancelled).

``POST /api/jobs/{job_id}/resume``
    Accepts ``failed``, ``paused`` and ``interrupted`` experiment and
    pipeline-optimization jobs. It re-submits the saved request
    (``resume.request``) against the original checkpoint
    (``resume.checkpoint_job_id``) with ``resume=True`` and returns the **new**
    job id; the original record is left untouched as history. Resume is refused
    when the checkpoint is gone, when another job still owns it (live or
    completed), or when the job is currently live.

The search side is a ``stop_check`` callable threaded from ``AML`` into every
``SurrogateRandomCV`` stage; it raises ``SearchStopped`` between candidate
evaluations. ``SearchStopped`` is deliberately not a ``ValueError`` — the
optimizer treats an objective failure as a bad candidate and continues.

Job statuses beyond ``queued``/``running``/``completed``/``failed``:

``paused``
    Stopped on request with a checkpoint; resume it to continue.
``cancelled``
    Stopped on request without a resume path; final.
``interrupted``
    The API process that owned the job exited before it finished. Written by
    start-up reconciliation and by the read endpoints.

Liveness is process-local
-------------------------

A JSON record is history, not evidence of running work. ``api.jobs.JobManager``
keeps an in-memory registry of the jobs this process owns and:

* ``GET /api/jobs/active`` lists only ids that are **live** — a stale record
  cannot keep the UI's activity badge alive after a crash;
* ``GET /api/jobs/{job_id}`` (and the WebSocket stream) rewrites a stale
  ``queued``/``running`` record to ``interrupted`` on read;
* start-up calls ``JobManager.reconcile()``, marking every leftover
  ``queued``/``running`` record as ``interrupted`` with an explanatory error;
* a heartbeat thread refreshes ``heartbeat_at`` every 10 s for each live job, so
  a stalled worker (thread hung, process alive) is visible in the UI even though
  the job is still registered.

Compute budget
================

Jobs share one process, so their parallelism has to be budgeted explicitly. The
budget is server state (it is what jobs obey), capped by the cores the process
may actually use — ``sched_getaffinity`` on Linux, so a ``taskset``/cpuset
container reports its real allotment rather than the host's core count.

``GET /api/settings/compute``
    ``{"cores": n, "values": {...}, "defaults": {...}, "budget": {...}}``.
    ``budget.summary`` spells the arithmetic out, e.g. ``"4 jobs x 1 BLAS
    thread(s) + 28 fold workers = 32 of 32 cores"``.

``PUT /api/settings/compute``
    Replaces all five values (administrator once authentication is enforced) and
    returns the new snapshot. Anything that does not fit the core count is
    rejected with HTTP 422 and a message naming the offending arithmetic.

.. list-table::
   :header-rows: 1

   * - Key
     - Default
     - Meaning
   * - ``job_workers``
     - ``min(4, cores)``
     - Concurrent background jobs; further submissions stay ``queued``.
   * - ``job_threads``
     - ``cores - job_workers``
     - **Total** CV-fold worker processes, shared by all jobs (joblib keeps one
       loky pool per process).
   * - ``blas_threads``
     - ``1``
     - OpenMP/MKL/OpenBLAS threads per job thread.
   * - ``dask_workers``
     - ``min(4, cores)``
     - Processes in the local Dask cluster a Dask-backend job creates.
   * - ``dask_threads``
     - ``1``
     - Threads per Dask worker.

Two invariants are enforced on save::

    job_workers * blas_threads + job_threads <= cores
    dask_workers * dask_threads              <= cores

Enforcement is not advisory:

* the job thread pool is built with ``job_workers`` workers and rebuilt when the
  setting changes (queued work still drains);
* every job start exports ``OMP_NUM_THREADS``/``MKL_NUM_THREADS``/… set to
  ``blas_threads`` — joblib only substitutes its own default when the variable is
  unset, so loky workers inherit the limit — and applies a process-wide
  ``threadpoolctl`` limit for in-process BLAS;
* ``fit_experiment_bundle`` and ``fit_pipeline_bundle`` pass
  ``n_jobs=job_threads`` to ``AML``, so the fold pool, ``cross_val_score`` and
  stacking all use the same width;
* a Dask-backend job builds its cluster with the configured worker/thread
  counts instead of dask's ``os.cpu_count()`` default, and closes it when the
  job ends. Inside a Dask worker joblib refuses to nest a process pool
  (``daemon=True`` forces ``n_jobs=1``), so folds run sequentially within a
  worker and ``dask_workers`` is the parallelism that matters there.

Each job record carries the budget it ran under (``compute``) and the values it
actually applied (``compute_applied``), and bundles repeat ``n_jobs`` in their
reproducibility record. Settings changes apply to jobs submitted afterwards; a
running job keeps the limits it started with. Values persist at
``<API home>/settings/compute.json``; a file that no longer fits the host (for
example after a resize) is ignored in favour of the defaults.

Dask is installed with the API image (``requirements.txt``) and ships as the
``dask`` extra in ``setup.py``; a source install without it keeps working — the
backend simply reports itself unavailable and job submission returns 503.

Resetting a user's data
=======================

``POST /api/settings/reset``
    Body ``{"confirm": "reset <username>", "include_shared": false}``. Erases
    every task the caller **owns** — ``datasets/``, ``mltrace/<task>.db``,
    ``bundles/``, ``registry/`` (versions and the index entry), ``predictions/``,
    ``monitoring/``, ``checkpoints/`` — plus the task's job records and log
    files, and every research project they own. Returns the removed task/project
    ids and counts (jobs, log files, stored files, bytes) plus
    ``skipped_shared``.

    * ``confirm`` must match exactly: ``reset <username>`` for a signed-in
      account, or ``reset everything`` when authentication is disabled (the
      single anonymous user owns everything). A wrong phrase answers **400**.
    * Ownership comes from ``auth/acl.db``: only an explicit ``owner`` grant is
      deleted. Unowned legacy material is left for an administrator to adopt; a
      user who is merely an editor on somebody else's task keeps their access.
    * Resources shared with other users are skipped and reported unless
      ``include_shared`` is true.
    * A live job on a scoped task answers **409** with its job ids; stop jobs
      first so no worker writes into a directory being removed.
    * Accounts, sessions and API keys are never touched. One ``account.reset``
      event is appended to the audit log with counts only.

In open (no-auth) mode this is a full local-workspace wipe, which is the
sensible meaning of "the logged-in user's data" for a single anonymous user.

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

Per-trial entries also carry ``score`` (the objective value) and ``params`` (the
hyperparameters that trial evaluated) as structured fields, so a UI can render
"what is being analysed" without parsing the message::

    {"seq": 43, "level": "INFO", "message": "trial 5 (explore) completed score=0.842 ...",
     "trial": 5, "phase": "explore", "status": "completed", "score": 0.842,
     "params": {"n_estimators": 120, "max_depth": 5}}

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

Identity events use a separate store, because they happen outside any job:
:mod:`api.audit` appends one JSON line per event to
``<SKSURROGATE_API_HOME>/logs/auth.jsonl`` (``login`` with outcome
``ok``/``denied``/``throttled``, ``logout``, ``password.change``,
``api_key.create``, ``api_key.revoke``, ``user.create``, ``user.update``,
``user.sessions_revoked``). Entries carry the username, client address, key
prefix and affected target, never a password, session token or full key, so the
file is safe to inspect or ship. ``GET /api/auth/audit`` returns the most recent
entries (``limit``, max 500) to administrators; the file itself is append-only
and never rewritten by the API. See :doc:`auth-and-sharing`.

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
