# SKSurrogate Web UI & API Layer — Design Plan

## 1. Goals

- Give SKSurrogate a modern, intuitive web UI that covers the **entire MLOps
  lifecycle end to end**: data → sensitivity/feature selection → AutoML/model
  search → evaluation → bundling → quality gates → approval/deployment →
  inference/serving → monitoring → retraining.
- Keep every stage **traceable**: a user should always be able to go from a
  prediction back to the model version, the bundle, the training run, the
  dataset fingerprint, and the features used — and forward from a dataset to
  everything that was ever produced from it.
- Enforce **total UI/backend separation** via a FastAPI service layer. The UI
  is a pure client of a documented REST/JSON (+ WebSocket for progress) API;
  it never imports SKSurrogate directly. This lets the UI be swapped, and
  lets the API be used by CI pipelines, notebooks, or other clients.
- Reuse the existing, already well-tested Python modules
  (`DataProcess`, `sensapprx`, `aml`, `eoa`, `structsearch`, `mltrace`,
  `modelbundle`, `ci`, `deployment`, `inference`, `monitoring`, `retraining`,
  `execution`) as-is; the API layer is a thin orchestration/adaptation layer,
  not a rewrite.

## 2. High-level architecture

```
┌───────────────────────────────────────────────────────────┐
│  Frontend SPA (React + TypeScript + Vite)                 │
│  - Talks ONLY to the FastAPI service over HTTP/JSON + WS   │
│  - No direct filesystem/db/model access                    │
└───────────────────────────────────────────────────────────┘
                         │ REST + WebSocket (OpenAPI-documented)
                         ▼
┌───────────────────────────────────────────────────────────┐
│  FastAPI service ("sksurrogate-api")                       │
│  - Routers per MLOps stage (see §4)                        │
│  - Pydantic schemas = the contract with the UI              │
│  - Opt-in shared-key auth + configurable CORS origins       │
│  - Job manager for long-running work (AML/EOA searches,     │
│    nested CV, batch drift scans) -> background workers      │
│  - Adapts DataFrames/estimators/bundles <-> JSON/files       │
└───────────────────────────────────────────────────────────┘
                         │ direct Python calls (in-process or worker pool)
                         ▼
┌───────────────────────────────────────────────────────────┐
│  SKSurrogate library (unchanged)                            │
│  DataProcess · sensapprx · aml/eoa/structsearch · mltrace   │
│  modelbundle · ci · deployment · inference · monitoring     │
│  retraining · execution (Local/Dask backends)                │
└───────────────────────────────────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────────┐
│  Artifact & metadata storage                                │
│  - Filesystem/S3: datasets, bundles, registry, checkpoints   │
│  - SQLite (peewee, via mltrace) or Postgres: experiments,    │
│    metrics, audit log                                        │
└───────────────────────────────────────────────────────────┘
```

Long-running operations (AML/EOA search, nested CV, large drift scans) are
**not** run inline in the request/response cycle. They're submitted as jobs
(reusing `SKSurrogate.execution.LocalProcessExecutionBackend` /
`DaskExecutionBackend` for the actual durable execution) and the UI polls or
subscribes over WebSocket for progress, reusing checkpoint/resume support
that already exists in `AML`/`EOA`.

## 3. Information architecture of the UI

A left-hand navigation organizes the app by **pipeline stage**, mirroring the
existing docs structure so users can map UI concepts directly onto the
underlying library and its documentation. Every stage view is scoped to a
**Task** (`task_name`, matching `mltrack`/registry conventions) and every
object within it carries lineage links (dataset fingerprint → run →
model_version → bundle → deployment → monitoring events).

```
Sidebar
 ├─ Dashboard              (cross-task overview, health, recent activity)
 ├─ Datasets               (register/browse datasets, schema, fingerprint)
 ├─ Feature Analysis       (sensitivity: Sobol/Morris/delta-mmnt, correlation pruning)
 ├─ Experiments            (AML/EOA/SurrogateSearch runs, live progress, history)
 ├─ Evaluation             (nested CV results, metrics, comparisons)
 ├─ Model Bundles          (create/inspect bundles, dependency + schema view)
 ├─ Quality Gates          (ci checks, thresholds, pass/fail history)
 ├─ Registry & Deployment  (lifecycle graph, promote/rollback, approvals)
 ├─ Inference              (batch prediction jobs, live /predict console)
 ├─ Monitoring             (drift, fairness, delayed-label perf, latency)
 ├─ Retraining             (jobs, triggers, schedule, run history)
 └─ Settings/Audit         (users, approvers, audit log, API keys)
```

Each item in a stage view supports drill-down breadcrumbs, e.g.:

`Dataset(fp=abc123) → Run(exp-42) → Model(v=7f3a) → Bundle → Registry(production) → Monitoring(drift alerts: 2)`

This breadcrumb bar is present on nearly every detail page ("traceability
rail") and is powered by a single `/lineage/{entity_type}/{id}` endpoint that
walks the metadata graph server-side.

## 4. UI stages ↔ API routers ↔ library mapping

| UI Section | FastAPI router | Wraps | Notes |
|---|---|---|---|
| Datasets | `/api/datasets` | `DataProcess.DataPreprocess`, `mltrace.mltrack.RegisterData/validate_data` | Upload CSV → server infers types (`deduce_types`), returns editable schema preview before commit; register returns `dataset_fingerprint` + JSON schema |
| Feature Analysis | `/api/sensitivity` | `sensapprx.SensAprx`, `CorrelationThreshold` | Submitted as a job (Sobol/Morris can be slow); returns ranked feature importances + suggested `n_features_to_select`; result reusable by Experiments stage |
| Experiments | `/api/experiments` | `aml.AML.eoa_fit/aml_fit`, `eoa.EOA`, `structsearch.SurrogateSearch/SurrogateRandomCV` | Async job; live progress via WebSocket streaming `evaluation_history_`/generation stats; supports resume from checkpoint dir; config editor for pipeline search space (`config` dict) as a form/JSON editor |
| Evaluation | `/api/evaluation` | `mltrace.mltrack.nested_cv_evaluation`, `.LogMetrics` | Compare candidate models side by side; charts for outer/inner fold scores |
| Model Bundles | `/api/bundles` | `modelbundle.ModelBundle`, `save_bundle/load_bundle`, `export_mlflow` | Create bundle from a finished experiment run; view schema contract, dependency pins, audit events; download/export |
| Quality Gates | `/api/quality-gates` | `ci.check_bundle_quality/assert_bundle_quality` | Configurable metric thresholds per task; gate result required before promotion is allowed in UI |
| Registry & Deployment | `/api/registry` | `modelbundle.ModelRegistry`, `deployment.DeploymentApprovalGate` | Visual lifecycle graph (candidate→validated→staging→production→archived); promote requires N approver clicks (RBAC-gated); rollback button with confirmation |
| Inference | `/api/inference` | `inference.predict_batch`, `create_app`/`serve` semantics reused as internal call, not by spawning the WSGI server | Batch: upload CSV or point to a dataset → job → downloadable predictions + `inference_metrics`; Online test console: single-row JSON form calls same code path |
| Monitoring | `/api/monitoring` | `monitoring.drift_report/fairness_report/delayed_label_performance/InferenceMonitor` | Dashboards with PSI/drift charts, fairness gap bars, latency/throughput; alerts feed into Dashboard |
| Retraining | `/api/retraining` | `retraining.RetrainingJob` | Define trigger type (`scheduled`/`on_data`) via form (server-side predicate, not arbitrary code from the browser); manual "run now"; history of runs and resulting model_versions |
| Settings/Audit | `/api/audit`, `/api/auth` | audit events already recorded on bundles/registry | Central audit log viewer, filterable by task/actor/action |

All routers return Pydantic response models; heavy objects never cross the
boundary directly:
- DataFrames ↔ CSV/Parquet file upload or reference to a server-stored
  dataset by fingerprint.
- Fitted estimators/pipelines ↔ always stay server-side inside bundles;
  the UI only ever sees bundle IDs/metadata, never pickled objects.
- Arbitrary Python callables (fitness functions, retraining triggers) are
  **not** accepted from the browser; only a constrained JSON/DSL config
  (already how `AML(config=...)` works) or a small set of predefined trigger
  types.

## 5. Handling long-running work

- Every "start experiment / run sensitivity / run drift scan" endpoint
  returns immediately with a `job_id` (HTTP 202).
- A `JobManager` in the FastAPI service enqueues work onto:
  - a local worker pool (thread/process) reusing
    `execution.LocalProcessExecutionBackend` for durability and resume, or
  - `execution.DaskExecutionBackend` when a Dask cluster is configured.
- `GET /api/jobs/{job_id}` for polling; `WS /api/jobs/{job_id}/stream` for
  live progress (generation number, best score so far, ETA) sourced from
  `AML`/`EOA` checkpoint callbacks.
- Jobs are resumable: if the API restarts, incomplete trials are picked up
  again via `resume_incomplete_trials()`, and the UI simply reconnects.

## 6. Tech stack recommendations

- **Backend**: FastAPI, Pydantic v2, Uvicorn/Gunicorn. **SQLite everywhere**
  for portability: reuse `mltrace`'s existing per-task SQLite databases
  (peewee) as-is, and add one additional SQLite database
  (`api/state.db`, plain `sqlite3`/SQLAlchemy) owned by the API layer for
  job records, registry/audit indexes, and any UI-only state. No external DB
  server required; the whole stack runs from a single folder + `pip install`.
- **Auth**: shipped as **opt-in shared-key auth** (the v1 plan was "none",
  kept as the default). Set `SKSURROGATE_API_KEY` to enable; unset leaves all
  endpoints open for trusted local use. The key is accepted via the
  `X-API-Key` header, an `Authorization: Bearer <key>` token, or — on
  WebSocket routes only, because browsers cannot set custom headers on a WS
  handshake — an `api_key` query parameter. Enforcement lives in a pure-ASGI
  middleware (`ApiKeyMiddleware` in `api/main.py`) covering both HTTP and
  WebSocket scopes; global FastAPI dependencies do not apply to WS routes.
  `/api/health` stays open unconditionally (the UI's connectivity indicator).
  CORS origins are configurable via `SKSURROGATE_API_CORS_ORIGINS`
  (comma-separated); unset defaults to local-only origins
  (`localhost`/`127.0.0.1`, any port) instead of a wildcard.
  Approval actions in Registry & Deployment still record an `approver`
  name/string field (free text, unauthenticated) so the existing
  `DeploymentApprovalGate` multi-approver bookkeeping and audit trail keep
  working — it's just not identity-verified yet. Per-user RBAC/OAuth2 remains
  future work if the tool moves beyond single-user/trusted-network use.
- **Frontend**: React + TypeScript + Vite, a component library such as
  Mantine or shadcn/ui + Tailwind for a clean modern look, TanStack Query for
  data fetching/caching, TanStack Table for result grids, Recharts/Visx for
  drift/metric charts, React Flow (or similar) for the registry lifecycle
  graph and lineage breadcrumb visualization.
- **Docs**: FastAPI auto-generates OpenAPI/Swagger; keep it as the
  single source of truth for the UI/backend contract (generate a typed
  API client for the frontend from the OpenAPI schema).

## 7. Suggested repo layout for the new pieces

```
SKSurrogate/                 # unchanged existing library
api/                         # new FastAPI service (separate package)
  main.py                    # app factory, router registration
  deps.py                    # shared dependencies (db session, auth, job manager)
  schemas/                   # pydantic models per stage
  routers/
    datasets.py
    sensitivity.py
    experiments.py
    evaluation.py
    bundles.py
    quality_gates.py
    registry.py
    inference.py
    monitoring.py
    retraining.py
    audit.py
    jobs.py
  services/                  # thin adapters calling into SKSurrogate.*
  jobs/                      # job manager, worker pool, execution backend glue
webui/                       # new React/Vite SPA
  src/
    pages/                   # one folder per sidebar stage
    components/
    api/                     # generated OpenAPI client + hooks
```

Both `api/` and `webui/` are new, independently deployable packages; the
existing `SKSurrogate/` library stays untouched aside from small ergonomic
additions if a gap is found (e.g., an endpoint needing a helper that doesn't
exist yet).

## 8. Phased delivery plan

1. **Phase A — API skeleton & Datasets/Bundles/Registry** (highest value,
   mostly file/JSON I/O, no long jobs): datasets, bundles, quality gates,
   registry & deployment, audit. Ship the FastAPI app with OpenAPI docs and a
   minimal UI shell (nav + these 5 sections).
2. **Phase B — Inference & Monitoring**: batch predict, online test console,
   drift/fairness/latency dashboards.
3. **Phase C — Job manager + Experiments + Feature Analysis**: async job
   infrastructure, AML/EOA/sensitivity as background jobs with live progress,
   evaluation comparisons.
4. **Phase D — Retraining + RBAC/approvals polish + full lineage
   breadcrumb/dashboard** across all stages.

## 9. Decisions

- **Auth**: opt-in shared-key auth shipped in v1 (see §6) — `SKSURROGATE_API_KEY`
  enables it, unset stays open; CORS origins configurable via
  `SKSURROGATE_API_CORS_ORIGINS` with a local-only default. Per-user RBAC/OAuth2
  remains future work if the tool moves beyond single-user/trusted-network use.
- **Storage**: SQLite only, for both `mltrace`'s existing experiment DBs and
  the new API-owned state DB (jobs, registry index, audit). Datasets/bundles
  remain plain files on the local filesystem (or a mounted volume); no S3
  dependency for v1.
- **Existing WSGI (`inference.create_app`/`serve`)**: kept as-is, unchanged,
  as a separate lightweight/dependency-free serving path. The new FastAPI
  service is the control plane (management, UI, batch jobs) and calls the
  same underlying `predict_batch`/`ModelBundle.predict` code directly for its
  own "online test console" rather than proxying to the WSGI server.
