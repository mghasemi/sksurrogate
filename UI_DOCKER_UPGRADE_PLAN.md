# Docker and Web UI Upgrade Plan

## Purpose and status

Bring the Docker setup and the web UI in line with the library changes
described in [`AI_LAYER_ROADMAP.md`](./AI_LAYER_ROADMAP.md) and
[`API_SIMPLIFICATION_PLAN.md`](./API_SIMPLIFICATION_PLAN.md). The UI must stay
intuitive and cover the essential workflow: data, model search, evaluation,
release, operation. No features are removed; advanced options sit behind
"Advanced" sections. This is a living plan: update task status and the progress
log whenever work starts, changes direction, or completes.

**Status:** Phases 1-3 complete and verified; remaining items are the open boxes below.

Legend: `[ ]` open, `[~]` in progress, `[x]` done.

## Baseline (2026-11-05)

- **Docker:** a single dev image (Python 3.12 plus Node 22) in
  [`Dockerfile`](./Dockerfile). [`docker-entrypoint.sh`](./docker-entrypoint.sh)
  runs uvicorn with `--reload` and the Vite dev server over a repo bind mount
  ([`docker-compose.yml`](./docker-compose.yml)). No production build,
  healthcheck, named data volume, `.dockerignore` or optional extras.
- **API routers:** datasets, synthdata, bundles, quality_gates, registry,
  inference, monitoring, projects, sensitivity, experiments, jobs, retraining,
  lineage, evaluation.
- **UI pages (13):** Dashboard, Datasets, Features, Experiments, Evaluation,
  Bundles, Quality gates, Registry, Inference, Monitoring, Retraining, Jobs,
  Settings.

## Gap table

| Library capability | API | UI |
|---|---|---|
| `AML(preset=...)` (P1.1) | missing | missing |
| `AML.fit(strategy=...)`, `resume`, `stacking` (API plan Phases 3-4) | to verify | missing / to verify |
| Projects (P2.5) | present | no page |
| `latin_hypercube`, `suggest_next_point` (P2.4) | missing | missing |
| Split-conformal intervals (P3) | missing | missing |
| Permutation importance, PDP/ICE (P1.5) | missing | missing |
| Provenance/notebook export (P2.2, P2.3) | to verify | to verify |
| Sensitive-feature detection (P1.6) | to verify | to verify |
| Promotion gates (P0.4), persisted monitoring (P0.5) | present | pages exist |
| Renamed args and sklearn-signed `best_score_`, `rank_test_score` | n/a | verify results display |

## Plan

### Phase 0: Audit
- [x] Compared router endpoints with `client.ts`: no client or page existed for
      presets, projects, explain, intervals or design; sensitive-feature scan
      already lives under Monitoring; promotion gates and monitoring pages exist.
- [x] Result payload names unchanged by the simplification plan (old names are
      aliases), so no UI break. Provenance/notebook export has no UI and is
      deferred (not essential to the workflow).
- [x] Coverage table added to `docs/ui-gap-implementation-plan.md`.

### Phase 1: Docker
- [x] `Dockerfile`: `dev` and `prod` targets (Node stage with `npm ci` builds
      `webui/dist`; FastAPI serves it with SPA fallback on port 8013; non-root
      user `sksurrogate`; healthcheck; `/data` volume).
- [x] `/api/health` already existed; `api/main.py` now serves `webui/dist`
      (override with `SKSURROGATE_WEBUI_DIST`) after all API routes.
- [x] `docker-compose.yml`: `dev` and `prod` profiles (named volume
      `sksurrogate-data`, `restart: unless-stopped`, `SKSURROGATE_PORT`,
      API key/CORS passthrough).
- [x] `EXTRAS` build arg; `.dockerignore`.
- [x] Entrypoint mode switch via `SKSURROGATE_MODE` (`dev` / `prod`).
- [x] CI job `docker-smoke` (build prod, curl `/api/health` and `/`).

### Phase 2: API gaps (thin wrappers over the library)
- [x] Experiments: `preset`, `strategy`, resume via `AML(resume=True)`;
      `GET /api/experiments/presets`; `scoring` defaults to the preset's metric.
      `config` is now optional when a preset is given.
- [x] `api/routers/explain.py`: permutation importance + PDP, split-conformal
      intervals (regression only).
- [x] `api/routers/design.py`: `latin-hypercube`, `suggest-next` (stateless,
      so not scoped to a project).
- [x] `GET /api/projects/{id}/summary` (artifact existence plus stage checklist)
      in place of a separate lineage endpoint.
- [x] Tests: `tests/test_api_experiments.py` (preset, legacy EOA path,
      explain/intervals/summary, design, SPA fallback).

### Phase 3: UI
- [x] Sidebar grouped by stage (Overview, 1. Data, 2. Model, 3. Release,
      4. Operate, System).
- [x] Projects page: create, stage checklist with next-step links, link
      artifacts of the current task, notes. (Active-project context deferred.)
- [x] Experiments: Quick start tab plus Advanced tab (search engine added).
- [x] Explain page (importance, PDP, intervals). Placed as its own page, not an
      Evaluation tab, to keep the 700-line Evaluation page untouched.
- [x] Experiment Design page (variables, LHS plan + CSV, suggest-next).
- [x] Dashboard getting-started card and updated stage cards.
- [x] Lineage rail (the existing `LineageRail`, already on Bundles, Registry,
      Monitoring and post-predict Inference) now also shows on Inference as soon
      as a version is picked, on Explain, and per linked bundle on Projects.
- [x] Settings "Server" card; `/api/health` now returns the library `version`.
      version display.

### Phase 4: Verification and docs
- [x] Full suite: 255 passed; `tests/api_snapshot.py` reports nothing removed.
- [x] `tsc -b` and `vite build` pass; lint shows only pre-existing warnings.
- [x] Browser smoke test against the prod container: quick experiment, Explain
      (importance, PDP, intervals), Projects (create, link, stages), Design.
      Not covered: bundle -> gate -> promote -> predict flow (existing pages,
      unchanged).
- [x] Updated `docs/ui-manual.rst` and `CHANGELOG.rst`.
- [x] Updated `docs/controlplane.rst` (Docker section).
- [x] `docs/ui-plan.md` updated. Source-plan status lines left untouched (they track library work, not UI).

Suggested order: Phase 0, Phase 1 (independent), Phase 2, then UI in the order
Experiments quick start, Projects, Explain/Intervals, Design, navigation
regrouping, then Phase 4.

## Open decisions

- Decided (autopilot, no reply received): production UI is served by FastAPI
  (one container); the Design page is included, labelled as a prototype.

## Progress log

- 2026-11-05: Plan created from review of both roadmaps, the Docker files,
  `api/routers/`, and `webui/src`. No code changed.
- 2026-11-05: Phase 1 done. Added SPA serving to `api/main.py`, multi-target
  `Dockerfile`, `.dockerignore`, mode-switching `docker-entrypoint.sh`, compose
  profiles, CI `docker-smoke` job. Validated: both targets build; prod container
  reports healthy, serves `/` and client routes, `/api/nope` returns JSON 404,
  path traversal falls back to index.html, runs as non-root with writable
  `/data`. Compose profiles and the CI job were not run end to end locally.
- 2026-11-05: Phases 0, 2, 3 done. Experiments accept `preset`/`strategy`;
  new explain, intervals, design and project-summary endpoints; Quick start,
  Explain, Design, Projects pages; regrouped navigation. Fixed on the way: my
  first request validation rejected `config={}` that existing tests use
  (now only a missing config is rejected); regression presets inherited the
  `accuracy` default (scoring now defaults per preset). Tests: 255 pass (one
  unrelated sqlite `ProgrammingError` in a phase6 test appeared once in a
  full run and did not recur in three reruns). Added `httpx2` to the CI test
  install because Starlette's TestClient requires it.
- 2026-11-05: Remaining doc items closed (coverage table, ui-plan, controlplane Docker section). Full suite rerun: one flaky phase6 test failed once, passes alone and on stashed baseline.
- 2026-11-05: Deferred items done: lineage rail on Inference/Explain/Projects, server version in Settings. tsc, build and test_api_experiments pass; prod container verified.
- 2026-11-05: Documentation pass. Added `docs/walkthrough.rst` (diabetes end-to-end tutorial, 20 screenshots in `docs/images/wt-*.png`, REST and Python equivalents, all snippets run against a live container), sample CSVs, cross-links, a readme "Run with Docker" section and changelog entries. Found while capturing: `validate_data` and the inference preflight rejected unseen float values as "unknown categories" (`categorical_values` is stored for every column); now skipped for float dtypes, with a regression test. UI polish: metric grid no longer wraps, Explain labels every feature, Evaluation shows a muted note for classification-only curves. Known quirk (not fixed): job results such as the sensitivity ranking live only in the page session.
