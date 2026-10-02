# SKSurrogate → UI Gap Implementation Plan

Step-by-step plan for implementing every library feature that is currently absent from the
API/UI stack, based on the gap analysis of 2026-09-23. Each phase lists: the library entry
points (with verified signatures), the new API endpoints, the `client.ts` additions, and the
exact UI page/component where the feature lands.

**Architecture reminder**: UI → `webui/src/api/client.ts` → FastAPI router (`api/routers/`) →
shared helpers (`api/deps.py`, `api/training.py`) → `SKSurrogate` library. Every step below
follows this chain; nothing in the UI may call the library directly.

**Conventions used throughout**:
- New endpoints are added to an existing router unless a new router is explicitly stated.
- Long-running work (nested CV, plot computation over large partitions, synthetic generation)
  runs through `api/jobs.py` (`job_manager.submit`) and returns `{job_id}`; cheap calls are
  synchronous.
- Plots: the library's `plot_*` methods return matplotlib figures and pickle them into the
  mltrace `plots` table. For the UI we **compute the underlying data arrays server-side**
  (sklearn `roc_curve`, `calibration_curve`, and the pure-Python
  `mltrack.cumulative_gain_curve`) and render natively with Recharts — no image round-trip.
  The library plot methods are still invoked as a side effect so the mltrace store stays
  consistent for non-UI consumers (optional, see Phase 1 step 4).

---

## Phase 1 — Evaluation deep-dive (highest value; promised by `docs/ui-plan.md` §4)

**UI home: `webui/src/pages/Evaluation.tsx`** (new cards/tabs below the existing learning-curve
section), plus one card on FeatureAnalysis.

### 1.1 Nested CV evaluation — `mltrack.nested_cv_evaluation`
Signature (`SKSurrogate/mltrace.py:895`):
```python
nested_cv_evaluation(estimator, inner_cv=2, outer_cv=2, train_partition="train",
                     validation_partition="validation", scoring=None, groups=None,
                     repeats=1, confidence_level=0.95) -> dict
# returns: outer_scores, inner_scores, mean_outer_score, fold_metrics[], repeat_count,
#          repeated_outer_scores[], confidence_interval{confidence_level, lower, upper},
#          final_validation_score (when a validation partition exists)
```
1. **API** — new router file `api/routers/evaluation.py` additions:
   - `POST /api/evaluation/{task}/nested-cv` with body
     `{model_version | alias, inner_cv=2, outer_cv=2, train_partition="train",
       validation_partition="validation", scoring=null, repeats=1, confidence_level=0.95}`.
   - Load the bundle via `resolve_bundle` (`api/deps.py`), open the tracker with
     `open_tracker(task)`, call `tracker.nested_cv_evaluation(bundle.model, ...)`.
   - The call is expensive (repeats × outer × inner fits) → submit as a background job:
     return `{job_id}`; store the result dict in the job record.
2. **client.ts** — `runNestedCV(task, body): Promise<{job_id}>` + reuse existing
   `getJob`/`subscribeToJob`.
3. **UI (Evaluation.tsx)** — new "Nested CV" card: bundle picker (reuse the version select),
   inner/outer fold steppers, repeats, confidence-level slider; on completion render
   - a box-plot-style bar of `outer_scores` with the CI band (`confidence_interval.lower/upper`)
     as an error bar,
   - per-fold table from `fold_metrics` (inner mean vs outer score),
   - headline numbers: `mean_outer_score`, `final_validation_score`.

### 1.2 Classifier diagnostic plots — ROC / calibration / lift / cumulative gain
Signatures (`mltrace.py`): `plot_roc_curve(mdl, label=None)` (:1622),
`plot_calibration_curve(mdl, name, fig_index=1, bins=10)` (:1568),
`plot_cumulative_gain(mdl, ...)` (:1662), `plot_lift_curve(mdl, ...)` (:1798). All internally
call `split_train(mdl)` (75/25 split of the task's stored X/y) and are **binary-classification
only** (gain/lift raise otherwise).

1. **API** — one endpoint returning all four curve datasets for a bundle:
   - `GET /api/evaluation/{task}/{model_version}/curves?bins=10` →
     ```json
     {
       "roc": {"fpr": [...], "tpr": [...], "auc": 0.93},
       "calibration": {"mean_predicted_value": [...], "fraction_of_positives": [...],
                        "histogram_counts": [...]},
       "cumulative_gain": {"percentages": [...], "gains_class0": [...], "gains_class1": [...]},
       "lift": {"percentages": [...], "lift_class0": [...], "lift_class1": [...]},
       "errors": ["..."]   // e.g. "not a binary classifier" — per-curve, not fatal
     }
     ```
   - Implementation: load bundle + tracker; get `X_test/y_test` via
     `tracker.split_train(bundle.model)` (caches the split); then compute with sklearn
     (`roc_curve`, `calibration_curve`) and the pure function
     `mltrack.cumulative_gain_curve(y_true, y_score, pos_label)` for gain/lift points.
   - Guard: only classification bundles; return 422 with a clear message otherwise.
2. **client.ts** — `getBundleCurves(task, version, bins?): Promise<BundleCurves>`.
3. **UI (Evaluation.tsx)** — new "Diagnostic curves" tab group next to the learning-curve tabs:
   - ROC → Recharts `LineChart` with the diagonal reference line + AUC in the title;
   - Calibration → reliability curve (`LineChart`) + predicted-probability histogram
     (`BarChart`) stacked vertically, mirroring the library's 3-row figure;
   - Cumulative gain & Lift → two-class `LineChart`s with baseline dashed lines.

### 1.3 Target statistics — `mltrack.Stats`
Signature: `Stats() -> dict(df.describe())` on the task target column (`mltrace.py:2143`).
1. **API** — `GET /api/datasets/{task}/target-stats` (datasets router; cheap, synchronous).
2. **client.ts** — `getTargetStats(task)`.
3. **UI (Datasets.tsx)** — small "Target statistics" card in the metadata section
   (count/mean/std/min/25%/50%/75%/max table).

### 1.4 Top features across weight types — `mltrack.TopFeatures`
Signature: `TopFeatures(num=10) -> OrderedDict[feature, count]` (`mltrace.py:2114`) — ranks
features by how often they appear in the top-`num` of **every** stored weight column
(pearson + model weights). Requires a prior sensitivity run (weights table populated).
1. **API** — `GET /api/sensitivity/{task}/top-features?num=10`.
2. **client.ts** — `getTopFeatures(task, num)`.
3. **UI (FeatureAnalysis.tsx)** — "Consensus top features" card above the heatmaps: ranked
   list with a count badge ("appeared in 4 of 5 weightings"); disabled state with hint
   "run a sensitivity analysis first" when no weights exist.

### 1.5 Persisted-plot store (optional, low priority)
`allPlots(mdl_id)` / `LoadPlot(pid)` (`mltrace.py:~1390/1420`) — expose only if non-UI tooling
needs the pickled matplotlib figures; otherwise skip and rely on 1.2's data endpoints.

---

## Phase 2 — Monitoring completion

**UI home: `webui/src/pages/Monitoring.tsx`**.

### 2.1 Drift threshold configuration (API already supports it)
The drift endpoint accepts `psi_threshold`, `categorical_threshold`, `missingness_threshold`,
`range_threshold` (`api/routers/monitoring.py:31-56`) but the UI never sends them.
1. **UI only** — in the existing Drift card, add four number inputs (defaults 0.2 / 0.2 / 0.1 /
   0.1) and pass them through `checkDrift`. No API/client changes needed beyond adding optional
   fields to the request type in `client.ts`.

### 2.2 Prediction distribution drift — `monitoring.prediction_distribution_report`
Signature (`SKSurrogate/monitoring.py:168`):
```python
prediction_distribution_report(reference_predictions, current_predictions, *,
                               model_version=None, threshold=0.2, bins=10) -> dict
# (drift_report over a single "prediction" column; returns alerts filtered to drift types)
```
1. **API** — `POST /api/monitoring/{task}/{version}/prediction-drift` with
   `{reference_source: {partition | inline}, current_source: {...}, threshold=0.2, bins=10}`.
   Reuse the same source-resolution helper as the existing drift endpoint (partitions come from
   stored predictions; inline arrays for ad-hoc checks).
2. **client.ts** — `checkPredictionDrift(task, version, body)`.
3. **UI (Monitoring.tsx)** — new "Prediction distribution" card: two source pickers + threshold
   input → PSI bar per the report's columns and an alert banner when alerts are non-empty.

### 2.3 Subgroup performance & loss reports — `subgroup_performance_report` / `loss_report`
Signatures (`monitoring.py:250`, :289):
```python
subgroup_performance_report(y_true, y_pred, groups, *, metric="accuracy", positive_label=1)
    -> {"groups": {g: {metric, count}}, "metric", "fairness_gap"}
loss_report(y_true, y_pred, groups=None, *, positive_label=1)   # wrapper, metric="loss"
```
1. **API** — `POST /api/monitoring/{task}/{version}/subgroup-performance` with
   `{y_true_source, y_pred_source, group_column | inline_groups, metric ∈ accuracy|precision|recall|f1|loss}`.
2. **client.ts** — `checkSubgroupPerformance(task, version, body)`.
3. **UI (Monitoring.tsx)** — new "Subgroup performance" card: grouped bar chart of the per-group
   metric + a highlighted `fairness_gap` number; metric dropdown drives the request.

### 2.4 Sensitive-feature / PII scan — `monitoring.sensitive_feature_report`
Signature (`monitoring.py:296`): `sensitive_feature_report(frame, *, sensitive_features=None)
-> {pii_columns[], sensitive_columns[], warnings[]}`.
1. **API** — two wirings (both cheap, synchronous):
   - Extend the dataset registration response (Phase 5.1) with a `sensitive_scan` field computed
     on the uploaded frame;
   - Standalone: `POST /api/monitoring/{task}/sensitive-scan` taking an inline column list or a
     partition name, plus optional explicit `sensitive_features`.
2. **client.ts** — `scanSensitiveFeatures(task, body)`.
3. **UI (Monitoring.tsx)** — "PII & sensitive columns" card listing flagged columns with warning
   text; also surfaced inline on the Datasets registration result (Phase 5).

---

## Phase 3 — Model lifecycle: preservation, best-model, export, artifacts

**UI homes: `Bundles.tsx`, `Registry.tsx`**.

### 3.1 Model preservation — `PreserveModel` / `RecoverModel` / `allPreserved`
Signatures (`mltrace.py:1349/1371/1339`): pickle a logged model into the mltrace SQLite
`saved` table; recover returns the fitted estimator.
1. **API** (bundles router, tracker-scoped via `open_tracker`):
   - `POST /api/bundles/{task}/{model_version}/preserve` → logs the bundle's estimator if needed
     (`LogModel`) then `PreserveModel`; returns `{pickle_id}`.
   - `GET  /api/bundles/{task}/preserved` → `allPreserved()` rows (pickle_id, model_id, init_date).
   - `POST /api/bundles/{task}/recover {pickle_id}` → `RecoverModel(pickle_id)`; returns the
     recovered estimator's type + a fresh smoke score on the train partition (do **not** return
     the object itself — it lives server-side; optionally re-save as a new bundle version).
2. **client.ts** — `preserveBundle`, `listPreservedModels`, `recoverModel`.
3. **UI (Bundles.tsx)** — in the bundle detail card: "Preserve snapshot" button + a
   "Preserved snapshots" list with per-row "Recover as new version" action (confirmation dialog,
   shows the resulting new model_version).

### 3.2 Best-model shortcut — `mltrack.getBest`
Signature (`mltrace.py:1301`): `getBest(metric) -> row` of the highest-metric logged model.
1. **API** — `GET /api/bundles/{task}/best?metric=train_score`.
2. **client.ts** — `getBestBundle(task, metric)`.
3. **UI (Bundles.tsx)** — "★ Best" badge on the list row matching the result + a "Jump to best"
   control in the header that preselects it; also used by Inference's model picker as a default
   suggestion chip.

### 3.3 MLflow export & bundle download — `modelbundle.export_mlflow`
Signature (`SKSurrogate/modelbundle.py:191`): `export_mlflow(bundle, path) -> Path` (writes
`MLmodel`, `model.pkl`, `bundle.json`; raises if destination exists).
1. **API** (bundles router):
   - `POST /api/bundles/{task}/{model_version}/export-mlflow` → exports into a temp dir under
     `var/sksurrogate-api/exports/{task}/{version}-<ts>/`, zips it, returns the zip as a
     `FileResponse` download (name `{task}-{version}.mlflow.zip`).
   - `GET /api/bundles/{task}/{model_version}/download` → raw `.bundle` file via `FileResponse`.
2. **client.ts** — `exportMlflowUrl(task, version)` and `bundleDownloadUrl(task, version)`
   (plain URL helpers; the UI uses `<a href>` with the API key query param already used for job
   streams).
3. **UI (Bundles.tsx)** — detail card action row: "Export MLflow" + "Download bundle" buttons.

### 3.4 Registry artifact lineage & cleanup — `ModelRegistry.register_dataset` /
`register_prediction` / `delete_artifacts`
Signatures (`modelbundle.py:379/407/431`).
1. **API**:
   - *Auto-wiring (no UI surface)*: the datasets register endpoint calls
     `registry.register_dataset(task, fingerprint, source_path)` after a successful registration;
     the inference batch endpoint calls `registry.register_prediction(...)` with the output path.
     This makes the existing LineageRail complete without new UI.
   - `DELETE /api/registry/{task}/artifacts?model_version=…&dataset_fingerprint=…` →
     `delete_artifacts`; requires at least one selector (422 otherwise) and records an audit
     event first.
2. **client.ts** — `deleteArtifacts(task, body)`; no client change for the auto-wiring.
3. **UI (Registry.tsx)** — new "Artifact cleanup" card: pick a model version or dataset
   fingerprint from the lineage data, show what will be removed (bundle file + registry entries),
   destructive-action confirmation dialog.

---

## Phase 4 — Experiments enhancements

**UI home: `webui/src/pages/Experiments.tsx`**.

### 4.1 Surrogate-assisted search — `AML.add_surrogate`
Signature (`SKSurrogate/aml.py:591`):
```python
add_surrogate(estimator, itrs, sampling=None /* BoxSample default */, optim="L-BFGS-B")
# appends (estimator, itrs, sampling, optim) to self.surrogates; AML falls back to a
# default KRR+GPR pair when surrogates is None at search time.
```
1. **API** — extend `RunExperimentRequest` (`api/routers/experiments.py`) with:
   ```python
   surrogate_mode: bool = False
   surrogate_itrs: int | None = None        # per-surrogate iteration budget
   ```
   and in `fit_experiment_bundle` (`api/training.py:303`), after constructing the `AML`:
   when `surrogate_mode`, call `aml.add_surrogate(<default regressor>, itrs)` before
   `eoa_fit`. **Verify during implementation** that the EOA path consults
   `self.surrogates` (the `_cast` method does); if only the plain `fit()` path uses them, route
   surrogate runs through that entry point instead. Record `surrogate_mode` in the bundle audit
   event.
2. **client.ts** — extend `runExperiment` body type.
3. **UI (Experiments.tsx)** — "Search strategy" section: radio *Evolutionary (EOA)* /
   *Surrogate-assisted*; when surrogate is selected, show an iterations number input and a hint
   that runtime per generation increases.

### 4.2 Conditional parameters & forbidden rules in the search space
`SurrogateRandomCV` supports conditional params/forbidden rules; `ParamSpec`
(`api/routers/experiments.py`) only models real/integer/categorical, so they are unreachable.
1. **API** — extend `ParamSpec`:
   ```python
   depends_on: dict[str, list] | None = None   # {param: [allowed values]} conditional rule
   ```
   and add to the request: `forbidden: list[list[str]] | None = None`. Thread both through
   `build_search_param` / the search constructor in `api/training.py`.
2. **UI (Experiments.tsx)** — JSON editor gains two optional top-level keys (`"forbidden":
   [["paramA", "value"], ...]`) and per-param `"depends_on"`; add a live validation hint listing
   unknown parameter references before submit.

### 4.3 Pareto frontier of the search — `SurrogateRandomCV.pareto_frontier`
Signature (`SKSurrogate/structsearch.py:1359`): `pareto_frontier(score_key="score",
cost_key="duration")`.
1. **API** — in the experiment job's `_run`, after `fit_experiment_bundle` returns, also return
   `"pareto": aml.cv.pareto_frontier(...)` (guard with try/except; include per-candidate duration
   from the evaluation history when available).
2. **UI (Experiments.tsx)** — in the job result view: Recharts `ScatterChart` of score vs
   duration with frontier points highlighted (different fill) and a small table of frontier rows.

### 4.4 Top pipelines & single-pipeline optimization — `AML.get_top` / `optimize_pipeline`
Signatures (`aml.py:864/933`): `get_top(num=5)` returns the best stored pipeline/score pairs;
`optimize_pipeline(seq, X, y)` fits one explicit component sequence.
1. **API**:
   - Job result gains `"top_pipelines": aml.get_top(5)`.
   - New endpoint `POST /api/experiments/{task}/optimize-pipeline {seq: [str], train_partition}`
     → background job that builds the pipeline from the component names and calls
     `optimize_pipeline`; saves the resulting bundle.
2. **UI (Experiments.tsx)** — result view shows a "Top pipelines" table; each row has an
   "Optimize this structure" button that submits 4.1's endpoint and opens the JobTracker.

### 4.5 Stacking as a selectable component — `StackingEstimator`
Exported from `SKSurrogate/__init__.py`; currently not reachable from the search-space editor.
1. **API** — allow `"stacking"` as an estimator key in the experiment config; map it to
   `StackingEstimator` with params `res`/`probs`/`decision`/`cv`/`n_jobs` in
   `build_search_param`.
2. **UI (Experiments.tsx)** — add "stacking" to the estimator dropdown in the JSON editor's
   helper UI, with its parameter schema pre-filled.

---

## Phase 5 — Datasets completion: schema review, validation, synthetic data

**UI home: `webui/src/pages/Datasets.tsx`**.

### 5.1 Editable schema preview before commit (plan §4 item)
The register endpoint already returns `deduced_types`; the UI ignores them and there is no way
to override types (`DataPreprocess.set_type`, `transform_label_bin` are unexposed).
1. **API** — split registration into two steps:
   - `POST /api/datasets/{task}/inspect` (CSV upload) → runs `deduce_types` +
     `sensitive_feature_report` and returns `{columns, deduced_types, target_candidates,
     sensitive_scan}` **without persisting**.
   - Extend `POST /{task}/register` body with optional
     `type_overrides: dict[str, str]`, `target: str`, `binarize_label: bool`; apply
     `DataPreprocess.set_type(...)` / `transform_label_bin(...)` before `RegisterData`.
2. **client.ts** — `inspectDataset(task, file)`, extended `registerDataset` body.
3. **UI (Datasets.tsx)** — registration becomes a two-step flow: upload → review table with an
   editable type column per row (select of the deduced types), target picker, label-binarize
   toggle, and PII/sensitive flags from 2.4 highlighted in amber → "Commit dataset".

### 5.2 Pre-flight validation — `validate_data` / `validate_prediction_data`
Signatures (`mltrace.py:677/721`): validate a frame against the registered schema; both accept
`missing_columns ∈ raise|ignore|allow` and `unknown_categories`.
1. **API**:
   - `POST /api/datasets/{task}/validate` (CSV upload or partition name) → runs
     `tracker.validate_data(df, target)` and returns `{valid: true}` or a 422 with the library's
     schema-difference message.
   - Extend inference requests with optional `preflight: bool`; when set, run
     `validate_prediction_data` before predicting and return the validation error instead of a
     raw prediction failure.
2. **client.ts** — `validateDataset(task, body)`, `preflight` flag on `predict`/`predictBatch`.
3. **UI**:
   - Datasets: "Validate upload" button in the review step (5.1) and a re-validate action in the
     metadata card;
   - Inference (`Inference.tsx`): "Validate before run" checkbox on both predict and batch forms,
     with the schema-difference message rendered as an error note.

### 5.3 Synthetic data — `synthdat.SynthData` (entire module unexported)
Signatures (`SKSurrogate/synthdat.py`): `SynthData(df, default_rv="uniform",
distribution_type="marginal"|"joint", rv=None)`; `set_type(clmns, typ ∈ bin|int|real|cat|date,
param)` (:323); `transform()` (:337); `generate(num) -> DataFrame` (:381); `where(cns)` DSL
(:432); `filter(df)` (:441); `sample(num)` (:486).
1. **Library** — export `SynthData` from `SKSurrogate/__init__.py`.
2. **API** — new router `api/routers/synthdata.py`:
   - `POST /api/datasets/{task}/synthetic {num, distribution_type="marginal", default_rv=
     "uniform", type_overrides: dict[str,str] | null}` → builds `SynthData` from the task's train
     partition (or a named partition), applies overrides via `set_type`, calls
     `transform()` + `generate(num)`, persists the result as CSV under
     `var/sksurrogate-api/datasets/{task}/synthetic-<ts>.csv`, and registers it through
     `ModelRegistry.register_dataset` (Phase 3.4). Returns `{path, rows, columns, preview}`.
   - `GET /api/datasets/{task}/synthetic?limit=50` → list + preview of generated files.
   - Skip the `where`/`filter` constraint DSL in v1 (niche; revisit if requested).
3. **client.ts** — `generateSynthetic(task, body)`, `listSynthetic(task)`.
4. **UI (Datasets.tsx)** — new "Synthetic data" card: row-count input, distribution-type radio
   (marginal/joint), per-column type override selects pre-filled from the deduced types,
   "Generate" button → result table preview + download link; a small list of previously generated
   files below.

**Implementation details**: `type_overrides` is sent as a JSON multipart field on registration;
the reviewed per-column map is retained as `dataset_deduced_types` task metadata so later
synthetic generation can prefill its type selectors. Generated samples have their own registry
fingerprint and can be downloaded through `GET /api/datasets/{task}/synthetic/{filename}`.
Inference preflight is opt-in and checks the final feature frame against the registered dataset
schema before either single or batch prediction.

---

## Phase 6 — Jobs & execution backends

**UI homes: `webui/src/pages/Jobs.tsx`, `Experiments.tsx`**.

### 6.1 Resume from checkpoint (plan §5 item)
AML/EOA already write checkpoints to `settings.task_checkpoint_dir(task, job_id)`; nothing can
reuse them today.
1. **API** — `POST /api/jobs/{job_id}/resume`:
   - only for failed/cancelled jobs of kind `experiment` (409 otherwise);
   - re-submits the original job function with the **same** checkpoint dir and run_id so
     `AML(check_point=...)`/EOA resume logic picks up where it stopped; returns a new `{job_id}`.
2. **client.ts** — `resumeJob(jobId)`.
3. **UI (Jobs.tsx)** — "Resume" button on failed experiment rows (confirmation dialog noting the
   checkpoint is reused); the tracker then follows the new job id.

### 6.2 Execution backend choice — `DaskExecutionBackend` / `LocalProcessExecutionBackend`
Both are exported but never used by `api/jobs.py`.
1. **API** — add optional `backend: "local" | "dask" = "local"` to experiment/retraining job
   submissions; when `"dask"`, wrap the job callable with `DaskExecutionBackend` (import lazily so
   dask stays an optional dependency; 503 with a clear message if not installed).
2. **UI (Experiments.tsx)** — "Execution backend" select in run settings, disabled with a hint
   when the server reports dask unavailable (extend `/api/health` or scoring-options response
   with `backends: ["local", "dask"]`).

---

## Phase 7 — Dashboard & loose ends

### 7.1 Drift-alerts feed on the Dashboard (plan item)
1. **API** — `GET /api/monitoring/alerts?task=&limit=20`: aggregate recent alert entries from
   persisted monitor logs (`var/sksurrogate-api/monitoring/{task}/…`) and any stored drift-report
   alerts; return `{alerts: [{timestamp, task, version, type, detail}]}`.
2. **client.ts** — `getMonitoringAlerts(task?)`.
3. **UI (Dashboard.tsx)** — new "Recent monitoring alerts" card below the job stats: list with
   severity badges and a link through to `/monitoring` for the affected task/version.

### 7.2 Wire up dead client function `loadRegisteredBundle` (`client.ts:377`)
1. **UI (Registry.tsx / Bundles.tsx)** — add a "Use in Inference" action on registered bundles:
   navigates to `/inference` with the version preselected (store it in the existing task-context
   or a query param read by `Inference.tsx`). This gives the registry-load endpoint its first UI
   consumer without new API surface.

### 7.3 Plan §4 leftovers — already satisfied, no work needed
- `/api/auth`: covered by the opt-in `ApiKeyMiddleware` + Settings page key field.
- `/api/audit`: covered by `GET /api/registry/{task}/audit` rendered on Registry and Settings.

---

## Suggested order & rough effort

| Phase | Content | Effort | Why this order |
|---|---|---|---|
| 1 | Nested CV + diagnostic curves + Stats + TopFeatures | M–L | Only gap explicitly promised by `ui-plan.md`; all on existing pages |
| 2 | Monitoring reports + threshold inputs | S–M | Pure additions to one page; API patterns already exist |
| 3 | Preservation, getBest, MLflow export, artifact lineage | M | Completes the bundle lifecycle story |
| 4 | Surrogate mode, conditional params, pareto, top pipelines, stacking | L | Touches `api/training.py` search plumbing — needs careful testing |
| 5 | Schema review/validation + synthetic data | M–L | Changes the registration flow (user-visible); synthdat is self-contained |
| 6 | Job resume + dask backend | S–M | Small API surface, high ops value |
| 7 | Dashboard alerts + loadRegisteredBundle wiring | S | Polish; depends on Phase 2 data existing |

**Testing notes**: extend `tests/test_optimized_paths.py` patterns for each new endpoint (happy
path + one error path); for nested CV and curves use the small Galaxy3 dataset already under
`data/`; keep matplotlib out of the request path entirely (Phase 1 computes arrays, not figures).
