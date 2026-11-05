# SKSurrogate Scientific Workflow Roadmap

## Purpose and status

Turn the recommendations in [`ailayer.md`](./ailayer.md) into a practical,
prioritized plan for making SKSurrogate more useful to scientists. This is a
living plan: update task status and the progress log whenever implementation
work starts, changes direction, or is completed.

**Status:** Implementation in progress; P0.2 and the scoped P0.3 numerical
hardening/diagnostics are complete, P0.4 promotion checks are implemented,
P1.1 quick-start presets are complete, P1.2 has baseline and component
profiles with optimization still open, P1.3 has corrected EOA aliases, and
P1.4 evaluation diagnostics are complete for the scoped tasks. P1.5 model
explanations are complete for bounded permutation importance and
partial-dependence/ICE plots; optional SHAP integration remains deferred.
P1.6 bounded sensitive-feature detection and P1.7 external triggered
retraining are complete for the scoped integrations. P1.8 was evaluated and
deferred because the bounded ensemble probe showed no consistent useful gain.
The broader recommendation audit is complete.

The source assessments describe possible gaps based largely on documentation,
not a verified code audit. Before implementing any item, check the current
code, tests, and roadmap documents. Mark capabilities that already exist as
done or partial instead of rebuilding them.

## Prioritization principles

- **P0 — Correctness and trust:** prevent evaluation leakage, numerical failures,
  unsafe promotion, or loss of important monitoring data.
- **P1 — Immediate user value:** improve search quality, speed, diagnostics,
  and the first-use experience with bounded changes.
- **P2 — Research workflow:** make analysis reproducible and easy to inspect,
  communicate, and continue.
- **P3 — Research bets:** invest in new scientific problem classes and
  discovery capabilities only after validating demand and technical fit.
- Prefer incremental features that reuse scikit-learn and existing SKSurrogate
  APIs. Keep optional integrations optional and preserve existing APIs where
  practical.
- Keep statistical calculations deterministic and auditable. Explain
  assumptions and do not present predictive explanations or sensitivity
  measures as causal conclusions.
- Do not add a large feature merely because it appears in a source assessment.
  Agree on scope and acceptance criteria before starting it.

## Existing foundations to reuse

The repository already has substantial work in these areas; verify the
implementation and tests before extending them:

- Dataset identity, schema validation, and lineage, plus nested/repeated
  evaluation, are marked complete in
  [`docs/mlops-roadmap.md`](./docs/mlops-roadmap.md).
- The UI/API has a separate gap plan for evaluation and monitoring work in
  [`docs/ui-gap-implementation-plan.md`](./docs/ui-gap-implementation-plan.md).
- Sensitivity analysis, experiment tracking, model bundles, monitoring, and
  retraining modules are part of the current architecture.

This roadmap is not a replacement for the MLOps or UI plans. Coordinate work
with them and update both documents when an implementation changes their
scope or status.

## Prioritized plan

### P0 — Correctness and operational trust

#### P0.1 Verify the recommendations against the current code

- **Status:** Complete for the recommendations and claims in the supplied
  assessments. Recheck individual items before implementation as code evolves.
- **Audit outcome:** the assessments mix stale manual descriptions with
  genuine gaps. Classifications below refer to current code and tests, not just
  the manuals.

| Recommendation area | Status | Current evidence and planned disposition |
| --- | --- | --- |
| Default search presets and API ergonomics | Partial | Bounded classification/regression presets are available (P1.1); legacy `AML(config=None)` behavior and broader naming/deprecation work remain under P1.3. |
| DataFrame names and categorical columns in AML | Existing | The source claim that AML strips DataFrames is stale: inputs and named categorical columns are preserved in current paths; keep compatibility work in P1.3 rather than redoing the feature. |
| Fold-safe preprocessing and column routing | Partial | AML's candidate preprocessing is fold-local, with `FoldLocalDataPreprocessor` available (P0.2). The legacy eager `DataPreprocess.encode()` workflow remains distinct and is documented as unsafe to fit before CV. High-cardinality/text improvements are not included in this completed fix. |
| Search cost and surrogate robustness | Partial | Hilbert numerical guards and surrogate diagnostics are implemented (P0.3). The previous real-workload probe mislabeled the library's default GP arm as Hilbert; the benchmark now injects and records Hilbert explicitly, and corrected real-workload comparisons are documented. Component profiling is complete; optimization remains P1.2. |
| Ensembling, TabPFN, and pruning integrations | Deferred | A bounded holdout-selected ensemble probe (P1.8) showed only a small, inconsistent gain; do not ship a production ensemble API without representative evidence. Do not add heavyweight dependencies by default. |
| EDA and data intelligence | Partial | P2.1 adds bounded mixed-type dataset profiles and registered-partition API access, with exploratory/non-causal caveats. Unit inference and document extraction remain deferred. |
| Evaluation uncertainty and statistical comparison | Partial | Nested/repeated evaluation and interval summaries already exist. P1.4 adds paired repeat-score bootstrap intervals/effect sizes and split-conformal regression; broader inferential tests remain future work. |
| Explanations and diagnostic plots | Partial | Sensitivity analysis and P1.4 diagnostics exist; P1.5 adds bounded permutation importance and partial-dependence/ICE plots. SHAP remains a separately evaluated, deferred optional integration. |
| Additional scientific problem classes | Deferred | Time-series, survival, and other specialized tasks require task-specific protocols; retain under P3 rather than extending generic search without validation. |
| Data provenance and lineage | Partial | P2.2 snapshots training settings, partition identity, environment, and evaluation protocol on newly trained API bundles, with JSON/Markdown exports and persisted nested-CV uncertainty. Target construction and upstream exclusions remain user-supplied provenance gaps. |
| Research projects, notebooks, and reports | Partial | P2.2 adds reproducibility exports; P2.3 adds executable offline scientific and standard-library Python API client notebooks. Research-project/hypothesis management remains P2.5. |
| Promotion quality gates | Existing | The server-side gate is implemented and shared by UI/API promotion flows (P0.4); do not duplicate this recommendation. |
| Monitoring persistence and retention | Partial | Inference measurements persist as JSONL and can be reloaded; automatic expiration is unresolved (P0.5). Avoid imposing a retention duration without a deployment requirement. |
| Automated retraining | Partial | Scheduled/on-data trigger entry points and retraining logic exist, but no automatic scheduler/webhook evaluates and submits due work (P1.7). |
| Sensitive-feature scanning | Partial | Current checks are name-based. Value-pattern scanning remains P1.6 and requires privacy/false-positive constraints before implementation. |
| Notebook-native UI and conversational assistant | Deferred | Preserve the existing operational UI; notebook interoperability is P2.3. A research assistant is a later product direction, not a prerequisite for deterministic scientific tools. |

- [x] Audit the concrete claims in `ailayer.md` against implementation, tests,
  and current documentation.
- [x] For each relevant recommendation, record whether it is **existing**,
  **partial**, **missing**, or **deferred** in this roadmap or the related
  implementation plan.
- [x] Resolve overlap with existing MLOps and UI plans before creating duplicate
  work.

**Done when:** planned work has a verified gap, a bounded scope, and an
acceptance check; existing behavior is not scheduled for reimplementation.

#### P0.2 Make preprocessing safe inside cross-validation

- **Status:** Complete — AML encoding is fold-local, a reusable transformer is
  available, and eager tracking preprocessing warns users about leakage risk.
- **Verified findings:**
  - [`AML.fit`](./SKSurrogate/aml.py) and [`AML.eoa_fit`](./SKSurrogate/aml.py)
    originally fit `category_encoders.OneHotEncoder` on all of `X` before
    evaluating candidate pipelines. This was fixed in this update: each
    candidate pipeline now learns encoding independently per training fold.
  - [`DataPreprocess.encode`](./SKSurrogate/DataProcess.py) calls
    `fit_transform` on the complete frame it receives, including its imputer
    and text vectorizer. If callers encode before splitting/CV, those learned
    transformations include held-out rows.
  - [`mltrack`](./SKSurrogate/mltrace.py) defaults to `encode=False`; this is
    not automatically a leakage path in every tracking workflow.
  - Tests now cover held-out category, numeric statistic, datetime statistic,
    and text vocabulary handling, plus use inside sklearn cross-validation.
- **Implemented:** AML now includes its optional categorical encoder inside
  each candidate pipeline. CV clones therefore fit the encoder on each
  training fold, and the final fitted pipeline accepts raw inputs at predict
  time. Unknown categories are handled with the installed encoder's supported
  `"value"` strategy, which was verified to encode them as all-zero columns.
  Feature-selector bounds use a conservative shape-only upper bound so
  categorical expansion remains possible without reading held-out category
  values.
- **Implemented:** exported `FoldLocalDataPreprocessor` provides a
  scikit-learn-compatible DataFrame transformer. It learns numeric imputations,
  categorical levels, datetime imputation, and per-column TF-IDF vocabulary
  during `fit`, with optional numeric/datetime scaling. Tests verify held-out
  values do not affect learned statistics/vocabularies and that the transformer
  runs inside a cross-validation pipeline. Usage guidance was added to
  `docs/dataproc.rst`. `mltrack` now warns when its legacy eager `encode=True`
  option is used. A related existing test exposed numeric strings being
  ordinal-encoded; type inference now recognizes all-numeric object columns
  and preserves their numeric values.
  The full `test_optimized_paths.py` module passes (69 tests), as do syntax and
  whitespace checks.
- **Compatibility caveat:** `DataPreprocess.encode()` and
  `mltrack.RegisterData(..., encode=True)` remain eager APIs. The latter now
  warns that its preprocessing fits on the complete registered frame. They
  are preserved for compatibility; use the new transformer inside a pipeline
  for CV/model selection.
- [x] Add or expose a pipeline that fits imputers, encoders, datetime
  transforms, and optional feature scaling on training folds only.
- [x] Route numeric, categorical, datetime, and explicitly typed text columns
  through fold-fitted transformations.
- [x] Preserve the existing standalone `DataPreprocess` workflow and document
  the eager-versus-fold-local distinction.
- [x] Cover numeric, categorical/high-cardinality, missing, datetime, and text
  inputs with held-out-data and sklearn-pipeline tests.

**Done when:** preprocessing is fitted independently within each CV training
fold, with no held-out fold statistics used to fit transforms.

#### P0.3 Harden surrogate numerical behavior

- **Status:** Complete for the scoped correctness, diagnostics, and benchmark
  work. The runner now explicitly selects and records each response surface;
  broader repeated/noisy scientific workloads remain future validation.
- [x] Reproduce the reported small-sample/underdetermined Hilbert-regression
  failure in a focused test.
- [x] Reproduce the noisy-evaluation Gaussian-process weakness in a committed,
  repeatable focused benchmark.
- [x] Define and test a safe response for ill-conditioned or underdetermined
  fits (clear error or documented fallback; do not silently return a
  success-shaped result).
- [x] Evaluate noise-aware GP settings and a random-forest alternative on
  repeatable noisy and smooth synthetic cases.
- [x] Confirm these results on multiple representative real workloads before
  changing defaults.
- [x] Document that no fallback is automatic today and expose enough
  diagnostics to explain the selected surrogate and fit/optimization failures.
- [x] Verify each real-workload benchmark arm actually selects the documented
  regressor, record the selected class from runtime diagnostics, and rerun the
  Hilbert/GP comparison.

The corrected real-workload comparisons are recorded in
[`benchmarks/DESIGN.md`](./benchmarks/DESIGN.md). They cover one fixed-fold
Galaxy3 search (seven seeds) and bundled wine and breast-cancer datasets (three
seeds each), all at budget 30. The stock GP remains the default: noise-aware GP
did not show a reliable quality gain. Corrected Hilbert results are separately
reported; the earlier mislabeled “Hilbert” rows are explicitly superseded.
Repeated/noisy evaluations and more varied scientific tasks remain open.

**Done when:** edge cases fail clearly or use a tested fallback, surrogate
selection/failure behavior is observable, and default changes are supported by
benchmark results without regressing existing cases.

#### P0.4 Enforce model-promotion quality gates

- **Status:** Complete for registry/API and automatic retraining promotion.
- [x] Verify the current bundle-quality check and promotion flow end to end.
- [x] Require a passing quality report by default before promotion; provide an
  explicit, auditable override only if product requirements call for one.
- [x] Test failed/absent/passing reports through the shared gate and API; the
  UI uses the same server-guarded endpoint.

**Scope note:** the current default `check_bundle_quality` report checks schema
compatibility. The standalone quality-check endpoint supports metric and size
thresholds, but promotion does not yet have a configured mandatory threshold
policy; define that policy separately before claiming metric-quality gating.

**Done when:** an unqualified or unchecked bundle cannot be promoted through
the normal UI/API flow.

#### P0.5 Persist essential inference monitoring

- **Status:** Partial — required measurements already persist as per-version
  JSONL records and are aggregated on demand. Logs currently have no automatic
  expiration; a bounded retention policy remains a follow-up.
- [x] Verify which serving measurements are required (latency, errors, and row
  throughput) and where they are stored.
- [x] Document the current retention behavior (no automatic expiration).
- [ ] Define and implement a bounded retention policy if required by
  deployment scale or operational policy.
- [x] Verify persisted measurements remain queryable after constructing a new
  monitor from stored records; no SQLite migration is needed to satisfy the
  restart/recovery requirement.
- [x] Keep sensitive prediction inputs and raw personal data out of metric
  records unless a separately reviewed requirement explicitly needs them.

**Done when:** measurements survive a process restart and remain queryable
without changing current inference behavior.

### P1 — Faster, clearer, more accessible model work

#### P1.1 Add safe built-in AutoML presets

- **Status:** Complete for bounded scikit-learn classification and regression
  starting presets; the existing broad legacy default remains unchanged.
- [x] Provide small, documented task-appropriate defaults for common
  classification and regression workflows.
- [x] Keep custom search spaces supported and make preset selection explicit
  and reproducible.
- [x] Test estimator availability, time/evaluation limits, and small datasets.
- [x] Avoid optional heavyweight estimators as mandatory dependencies.

**Done when:** a user can start a bounded search with a minimal API call and
can inspect the exact estimators and search space used. The presets are
baselines, not quality guarantees; budgets apply per candidate pipeline.

#### P1.2 Reduce search overhead using measured benchmarks

- **Status:** Partial — benchmark overhead is measurable across budgets and
  total timing now includes random search. A repeatable component profiler
  shows candidate CV dominates the current GP arm; profile/optimize only if a
  supported workload demonstrates a worthwhile time-to-result gain.
- [x] Profile search overhead versus estimator-evaluation time across small,
  medium, and larger trial counts.
- [x] Establish reproducible wall-clock and quality baselines for the current
  surrogate and a random-search baseline.
- [x] Separate candidate-CV time from local surrogate optimization with a
  repeatable component profile.
- [ ] Optimize the measured bottleneck; retain a simple baseline or configurable
  alternative where it improves time-to-result.
- [x] Document trade-offs rather than promising a fixed speedup across datasets.

The focused local Galaxy3 sweep (10-tree RF, seeds 0–2) measured median
surrogate overhead of 0.7%, 7.1%, and 14.7% at budgets 8, 30, and 60,
respectively; the no-surrogate arm stayed below 0.7%. Median best CV scores
were close to random search and do not establish a general quality advantage.
The broader historical B=30/B=60 suites show median SRS overhead of 9.2%/27.8%
across six datasets and seven seeds. These residual timing measurements do not
isolate surrogate-fit cost from acquisition cost. The component profile in
`benchmarks/dtrack/profile_search_overhead.py` measured candidate-CV time near
2.36 seconds on a representative one-seed Galaxy3 run; the GP local-optimizer
and prediction wrapper consumed about 0.09 seconds cumulatively. The legacy
Hilbert arm incurred greater fit and optimization overhead, but that is not
the current default. No search behavior or default is changed; an optimization
remains open until a supported workload establishes a measurable benefit.

**Done when:** benchmark results show the effect on both runtime and search
quality, and the chosen behavior is covered by tests.

#### P1.3 Preserve DataFrame names and normalize API conventions

- **Status:** Partial — AML already passes DataFrames through, named
  categorical columns work, and EOA now accepts corrected aliases; broader
  public naming inconsistencies and deprecation policy remain.
- [x] Audit `AML` input handling for DataFrame-to-array conversion and
  positional-only categorical references.
- [x] Preserve feature names through fitting when supported by downstream
  estimators; verify named `cat_cols` for DataFrame inputs.
- [x] Introduce corrected `snake_case` aliases for EOA's misspelled
  `parents_porp`/`elits_porp` and singular `max_generation` arguments while
  retaining the legacy spellings.
- [ ] Add deprecation warnings and migration documentation for any eventual
  breaking change.

**Done when:** named inputs retain useful feature names and existing callers
continue to work through a documented migration period.

#### P1.4 Improve evaluation diagnostics and statistical comparison

- **Status:** Complete for single-target regression, multiclass ROC/calibration,
  paired repeated-CV comparison, and split-conformal regression. Cumulative
  gain/lift remain binary-only because they require a single positive-class
  ranking.
- [x] Add single-target regression diagnostics (predicted-versus-actual,
  residual summaries, and plots) on the tracker's existing split.
- [x] Extend ROC and calibration plots with well-defined multiclass
  visualizations; retain binary gain/lift for positive-class ranking.
- [x] Build paired comparisons on `nested_cv_evaluation` repeat-level scores.
- [x] Add uncertainty-aware paired comparison with bootstrap intervals and a
  paired effect size; document alignment and dependence assumptions.
- [x] Prototype split-conformal regression intervals with documented coverage
  assumptions and tests for calibration/evaluation separation.

**Done when:** diagnostics support the documented task types and comparisons
report uncertainty without treating a small score difference as definitive.

#### P1.5 Add practical model explanations

- **Status:** Complete for the scoped tracker APIs. SHAP is not added as a
  dependency or required path; reconsider it as an optional extra if users need
  estimator-specific local attributions beyond permutation importance and ICE.
- [x] Start with permutation importance and supported scikit-learn
  partial-dependence/ICE capabilities; make the chosen reference dataset and
  sampling behavior explicit.
- [x] Evaluate SHAP as an optional integration for compatible estimators rather
  than a required dependency. Scikit-learn provides the bounded global and
  profile explanations in scope; defer SHAP to avoid adding another integration
  until supported estimators and local-attribution needs are specified.
- [x] Handle wide inputs through explicit feature selection and a configurable
  feature cap; bound and report reference sampling and ICE curve counts.
- [x] Label explanations as model behavior, not causal findings; report
  permutation-repeat variability and document that stability across alternative
  reference samples is not guaranteed.

**Done when:** supported models have useful global and local diagnostics with
clear limitations and bounded runtime for the scoped global-importance and
feature-profile APIs.

#### P1.6 Improve sensitive-feature detection carefully

- **Status:** Complete — bounded local email, formatted US SSN, and formatted
  US phone pattern scanning returns aggregate counts only.
- [x] Assess whether value-pattern scanning is required beyond column-name
  checks, and define the supported patterns and false-positive behavior.
- [x] If added, scan locally, avoid persisting or displaying matched values,
  and return only the minimum useful finding.
- [x] Test unnamed columns, common formats, and false-positive cases.

**Done when:** detection is documented, privacy-preserving, and does not expose
cell contents in logs, reports, or API responses. The scan inspects at most
1,000 rows by default and allows callers to request up to 10,000 rows; the
report indicates when input rows were truncated.

#### P1.7 Automate retraining triggers safely

- **Status:** Complete for the webhook integration. External schedulers and
  ingestion watchers can submit bounded, deduplicated triggers; no in-process
  recurring scheduler or distributed queue is added.
- [x] Reuse the existing retraining runner and `scheduled`/`on_data` trigger
  checks; do not duplicate training or promotion logic.
- [x] Add an external-scheduler or webhook integration that evaluates a
  documented due/new-data condition and submits an idempotent job.
- [x] Require audit history, metric quality-gated staging promotion, and clear behavior for
  duplicate, stale, or missing trigger data.

**Done when:** scheduled and data-driven jobs can run without manual API
requests from a user (the external scheduler still calls the API), and cannot
bypass existing promotion checks. Idempotency is process-local; run one API
worker for this file-backed job manager.

#### P1.8 Evaluate optional post-hoc ensembles

- **Status:** Evaluated and deferred — the bounded probe showed no consistent
  benefit sufficient to justify a production API. Keep the single-best model
  as the default.
- [x] Compare the CV-best individual model with small probability-averaging
  subsets selected from top candidates.
- [x] Use training-only CV for candidate ranking, a separate calibration
  partition for ensemble selection, and an untouched test partition for the
  comparison.
- [x] Preserve the single-best model as the default; do not add bundle or
  inference complexity before evidence supports shipping an ensemble.

**Revisit when:** representative scientific workloads and repeated outer
evaluation demonstrate a reproducible gain and bundle provenance requirements
are specified. The current bounded result and reproduction command are in
[`benchmarks/DESIGN.md`](./benchmarks/DESIGN.md).

### P2 — Reproducible research workflows

#### P2.1 Add a useful scientific EDA entry point

- **Status:** Complete — added a bounded mixed-type profile helper and a
  registered-partition API endpoint with target inference and partition
  fingerprints.
- [x] Provide a bounded dataset profile covering distributions, missingness,
  target balance/distribution, associations, and potential redundancy.
- [x] Include row/feature limits and clear handling of mixed types and small
  datasets.
- [x] Present findings as diagnostic signals, not proof of leakage, causation,
  or scientific mechanisms.
- [x] Reuse existing registered-dataset and API flows; no separate UI was
  needed for this API-scoped change.

**Done when:** a user can inspect data quality and basic target/feature
relationships before launching model search, with tests for mixed data.
**Verified:** ``dataset_profile`` reports numeric quantiles and non-finite
counts, categorical frequencies, missingness, target and pairwise associations;
sampling is bounded and seeded when requested. The registered API supports a
selected partition, target inference, and partition fingerprint. It rejects
implicit feature truncation, and its report explicitly disclaims causal,
leakage, and mechanism claims. Focused validation: 97 tests passed across
``tests/test_optimized_paths.py`` and ``tests/test_phase5_endpoints.py``;
changed Python modules compiled and ``git diff --check`` passed.

#### P2.2 Improve scientific provenance and result export

- **Status:** Complete for API-created baseline, AutoML, and pipeline-optimization
  bundles; legacy/manual bundles remain explicitly incomplete.
- [x] Build on existing dataset fingerprints, partitions, run metadata, and
  model lineage.
- [x] Identify any missing reproducibility fields (transformations, target
  construction, excluded observations, seeds, environment, and evaluation
  protocol) and persist only fields needed to reconstruct or audit a run.
- [x] Export a machine-readable run configuration and a concise Markdown or
  HTML report with methodology, metrics, uncertainty, limitations, and lineage.

**Done when:** a saved report/config can identify the data and settings used
and reproduce the analysis to the documented level of determinism.
**Verified:** new API-trained bundles snapshot the exact training-partition
fingerprint, selected features, target, row count, transformations, estimator
or search configuration, seeds, scoring/CV protocol, and runtime versions.
Baseline, AutoML, and fixed-pipeline training are covered; model
``dataset_fingerprint`` now points to the training partition rather than the
last partition registered. Repeated nested-CV summaries (protocol, partition
fingerprints, repeat-level scores, and confidence interval) are saved to the
selected bundle, including registry-alias bundles, with the most recent 20
evaluations retained. The API exports a versioned JSON configuration and a
downloadable Markdown report; lineage now identifies the bundle's exact
training data. Missing target-construction and upstream-exclusion provenance is
reported as unknown. Manual/legacy bundles remain readable but may not have
settings that were never recorded. Focused validation: 52 tests passed across
the evaluation, bundle, and experiment endpoint suites; changed Python modules
compiled and ``git diff --check`` passed.

#### P2.3 Support Jupyter without building a new notebook environment

- **Status:** Complete — added two source-only notebooks and user-facing
  setup/run documentation; Jupyter remains optional.
- [x] Ensure major API workflows are usable from Python clients.
- [x] Add a small set of notebook examples/templates for data inspection,
  search, evaluation, and interpretation.
- [x] Prefer portable, executable notebooks/config exports over screenshots or
  a proprietary notebook runtime.
- [x] Keep widgets and a notebook service out of the package runtime; neither
  improves these bounded starter workflows enough to justify an extra
  dependency or service.

**Done when:** documented notebook workflows run against the public Python/API
surface and produce reproducible outputs.
**Verified:** ``scientific_workflow.ipynb`` executed end-to-end using seeded
synthetic data, bounded classification search, repeated nested-CV, and
validation-partition permutation importance. ``python_api_workflow.ipynb``
executed end-to-end against a temporary local API server, including multipart
partition registration, profile, search job polling, nested-CV polling, and
JSON/Markdown report retrieval. Both notebooks parse as nbformat 4 JSON and all
code cells compile. Sphinx documentation build and ``git diff --check`` passed.

#### P2.4 Prototype experiment design and active learning

- [ ] Start with a scoped design-of-experiments capability (for example Latin
  hypercube sampling) and validate its constraints and output format.
- [ ] Then evaluate uncertainty-aware or sensitivity-guided next-point
  selection using existing surrogates.
- [ ] Keep recommendations separate from executing real-world experiments;
  require users to validate physical constraints and safety.
- [ ] Benchmark information/optimization value against simple baselines.

**Done when:** a bounded prototype produces valid candidate designs and
demonstrates value on reproducible synthetic and representative tasks.

#### P2.5 Organize investigations around research projects

- [ ] First stabilize dataset, analysis, run, and report APIs.
- [ ] Define a minimal project model for questions, datasets, analyses, notes,
  and linked results before proposing a full workspace redesign.
- [ ] Keep production deployment and monitoring available as a separate
  operational workflow.

**Done when:** a project can link and reproduce existing analysis artifacts
without duplicating experiment-tracking or registry concepts.

### P3 — Defer until demand and foundations are established

- [ ] Evaluate time-series forecasting, survival analysis, and quantile
  regression individually, including their required splitters and metrics.
- [ ] Investigate domain-specific scientific formats and semantic/unit-aware
  data understanding only with clear user needs and a bounded first format.
- [ ] Treat document extraction, full LLM research assistance, symbolic
  regression, physics-informed/shape-constrained models, and multi-fidelity
  optimization as separate research spikes with explicit success criteria.
- [ ] Add a model class or integration only after proving API compatibility,
  evaluation correctness, resource requirements, and maintenance cost.

**Defer for now:** a broad LLM layer, mandatory TabPFN/SHAP/Optuna dependencies,
image-processing support, many new problem classes at once, automatic
retraining without operational safeguards, and a proprietary notebook system.

## Progress tracking

Update task checkboxes when evidence changes. When implementation starts, add
the date, task IDs, files/PRs changed, validation performed, and any decision
or blocker to the log below. Do not mark an item complete until its stated
acceptance criteria are met.

### Progress log

- **Planning:** roadmap created from the two assessments in `ailayer.md`.
  Existing MLOps and UI roadmap overlap was noted.
- **2026-11-04 — P0.1/P0.2 audit:** confirmed eager full-frame fitting in
  `DataPreprocess.encode()` and `mltrack.RegisterData(..., encode=True)`, plus
  the former AML categorical encoding outside candidate CV. Existing
  nested/repeated evaluation is already tracked as complete in
  `docs/mlops-roadmap.md`, so it is reused rather than reimplemented.
- **2026-11-04 — P0.2 implementation:** moved AML category encoding into each
  candidate pipeline for both `fit()` and `eoa_fit()`, with tests for fold-only
  category vocabularies, unknown categories at predict time, and feature-size
  bounds. Added/exported `FoldLocalDataPreprocessor` for numeric, categorical,
  datetime, and text DataFrames, optional scaling, and strict schema checks.
  Tests confirm held-out values do not enter learned imputations or
  vocabularies and that the transformer works in sklearn CV pipelines.
  `mltrack` warns when its legacy eager `encode=True` option is enabled.
  `DataPreprocess` now recognizes all-numeric object columns to preserve their
  values instead of assigning category codes. Documented the safe workflow in
  `docs/dataproc.rst`. All 69 tests in `test_optimized_paths.py` pass.
- **2026-11-04 — Started P0.3:** reproduced the Hilbert failure with 10 rows,
  3 features, and degree 3: its 20-element basis produces non-finite
  predictions and a runtime warning.
- **2026-11-04 — P0.3 Hilbert guard:** `FunctionSpace.FormBasis()` now checks
  Gram-matrix finiteness, rank, and positive definiteness before normalization.
  Rank-deficient or numerically ill-conditioned bases raise an actionable
  `ValueError`; the small-sample failure and a valid full-rank linear fit both
  have regression tests.
- **2026-11-04 — P0.3 reproducible benchmark:** added
  `benchmarks/dtrack/noise_gp_probe.py` to compare stock RBF GP,
  RBF+WhiteKernel, and random forest on clean and noisy trigonometric
  objectives in 1-3 dimensions (40 training points, 400 evaluation points,
  10 fixed seeds). The noise-aware GP substantially improved the noisy
  one-dimensional case (`RMSE 0.4643` vs `0.7905` for stock GP), but results
  varied by dimension; in dimensions 2-3, the random forest had lower RMSE than
  either GP on these small-sample cases. Two full runs of the 1D-3D,
  10-seed matrix were identical. Several GP fits emitted optimizer/kernel-bound
  convergence warnings. Earlier exploratory figures used an uncommitted setup
  and are not directly comparable; this repeatable benchmark is the current
  evidence and does not justify changing defaults without representative
  workload validation.
- **2026-11-05 — P0.3 real-workload probe:** added
  `benchmarks/dtrack/real_noise_gp_probe.py` to compare Hilbert, stock GP,
  RBF+WhiteKernel GP, random-forest surrogate, no-surrogate SRS, and random
  search on Galaxy3 plus bundled scikit-learn `wine` and `breast_cancer`
  classification tasks. At budget 30, all 42 Galaxy3 runs (seven seeds) and
  all 36 additional runs (three seeds per dataset) completed the exact
  evaluation budget. Noise-aware GP tied stock GP on median CV accuracy for
  all three tasks and did not improve median held-out accuracy. On Galaxy3, the
  paired CV result was 1/5/1 win/tie/loss with mean delta +0.00045; on wine it
  was 0/3/0 and on breast cancer 0/2/1. This bounded fixed-fold study does not
  establish a general ranking; defaults remain unchanged, and repeated/noisy
  evaluation settings remain untested. A follow-up audit found that the
  “Hilbert” arm actually used the default GP; only the stock-GP/noise-aware-GP
  comparisons from this initial run remain valid. The Hilbert comparisons were
  rerun with explicit runtime selection and are documented below.
- **2026-11-05 — P0.3 surrogate diagnostics:** exposed the resolved regressor,
  source (default or caller-supplied), effective engagement threshold, fit
  attempts/failures, prediction failures, and unsuccessful local optimizer
  counts through `surrogate_info_` and the search summary. Fit and prediction
  exceptions remain explicit and are re-raised; non-finite or non-scalar
  predictions now fail clearly instead of being passed to SciPy. Documented
  that there is no automatic surrogate fallback and described unsuccessful
  local-optimization behavior in `docs/surrogate.rst`. Focused tests cover
  default/custom selection, fit failures, and non-finite predictions. The
  diagnostics, benchmark, and optimized-path regression suite passes
  (82 tests), and the touched Python files compile cleanly.
- **2026-11-05 — P0.1 benchmark audit follow-up:** a recheck found
  `run_sksurrogate` relied on the library default after that default had become
  GaussianProcessRegressor, so the real-workload probe's “Hilbert” arm was
  mislabeled. The benchmark now injects degree-3 Hilbert explicitly and records
  the selected class from runtime diagnostics; a direct integration test and
  the focused benchmark tests pass. The corrected 78-run workload comparison
  completed; results and the decision to retain stock GP are in
  `benchmarks/DESIGN.md`.
- **2026-11-05 — P1.2 component profile:** added
  `benchmarks/dtrack/profile_search_overhead.py` and profiled equal-budget
  Galaxy3 searches. Candidate CV dominated the stock-GP run, while the
  user-selected Hilbert arm spent more time fitting its repeated basis and
  performing local optimization. No core optimization was applied because the
  measured work is small for the default GP and the Hilbert cost applies only
  when that optional legacy surface is selected.
- **2026-11-05 — P1.4 regression diagnostics:** added
  `mltrack.regression_diagnostics()` with MAE, RMSE, R2, and residual summaries,
  plus a persisted four-panel predicted/actual, residual, histogram, and Q-Q
  plot. Both reuse the tracker's cached 75/25 split; classifier, multi-output,
  malformed-shape, and non-finite data paths fail explicitly. Added API
  documentation that the diagnostics are descriptive rather than inference.
  Summary calculations are checked against predictions on the same cached
  split.
- **2026-11-05 — P1.4 multiclass ROC:** extended `plot_roc_curve` with
  per-class one-vs-rest and micro-average curves, requiring all classes to be
  represented in the held-out split. Binary curves explicitly require both
  classes; binary-only gain/lift behavior remains unchanged. Added a 3-class
  plot-persistence test.
- **2026-11-05 — P1.4 multiclass calibration:** extended
  `plot_calibration_curve` with per-class one-vs-rest reliability curves and
  probability histograms. It requires finite class probabilities and each
  class to appear in the held-out split; existing binary behavior remains.
  Added a 3-class plot-persistence test.
- **2026-11-05 — P1.4 paired comparison:** added
  `mltrack.compare_repeated_cv_results()` for paired repeat-level score
  differences, a deterministic percentile-bootstrap interval, and paired
  Cohen's d-z. It validates aligned finite scores and documents that callers
  must use the same data/splits and that repeated CV is not independent-sample
  inference.
- **2026-11-05 — P1.4 split conformal:** added public
  `split_conformal_regression_interval()` with a separate calibration set,
  absolute residual scores, and the finite-sample corrected quantile. The
  estimator is cloned and fit only on training data; a small-calibration case
  correctly produces unbounded intervals when the requested rank exceeds the
  calibration sample count. Documentation states the exchangeability and
  model/calibration independence assumptions.
  The final optimized-path suite passes (80 tests), including binary and
  multiclass plots, paired-score resampling, and conformal train/calibration
  separation; Python compilation and `git diff --check` pass.
- **2026-11-05 — P0.1 recommendation audit:** compared both supplied
  assessments with current implementation evidence and the existing
  MLOps roadmap. Corrected stale claims about DataFrame handling and
  monitoring persistence, marked overlapping capabilities rather than
  scheduling duplicates, and mapped remaining work to P1/P2/P3.
- **2026-11-04 — P0.4 promotion gate:** server-side promotion now loads the
  exact registered bundle, computes `check_bundle_quality`, and requires the
  passing report before approvals or alias changes. The shared deployment gate
  rejects missing/failed reports; automatic retraining promotion now checks
  quality too. The UI explains the automatic check, and API tests cover passing
  reports and failed-report rejection without promotion.
  Validation passed: 18 phase-3 endpoint tests, 69 optimized-path tests,
  Python compilation, `git diff --check`, and the web UI production build.
- **2026-11-04 — P0.5 monitoring audit:** verified inference observations
  already append latency, row count, success state, and optional error details
  to per-task/per-model JSONL logs; the summary endpoint rebuilds error rate,
  mean latency, and row throughput from those records. Added a regression test
  that writes observations and queries the summary through a fresh monitor
  instance. The suggested SQLite migration is deferred because the existing
  durable store already meets the restart/query requirement. Also found that
  schema-validation errors could contain unknown category values; monitoring
  now persists generic validation summaries while the API response retains
  its detailed error. Regression tests verify that raw values do not appear in
  monitor records. Retention remains undefined and is tracked above. All 7
  phase-7 endpoint tests pass. Monitoring JSONL logs remain unbounded (no
  automatic expiration); this current policy is documented above, but a
  configurable retention policy still needs an operational decision.
- **2026-11-04 — P1.1 AutoML presets:** confirmed the existing `default_config`
  is a broad, optional-dependency-heavy classification search rather than a
  small, task-specific quick start. Added explicit `AML(preset="classification")`
  and `AML(preset="regression")` searches using sklearn random forests and
  extremely randomized trees, one-estimator pipelines, task-appropriate
  default scoring, and a 20-evaluation per-candidate budget. Custom `config`
  and legacy defaults remain supported. Documented that budgets apply per
  candidate and added small-data fit/predict, parameter override, unknown
  preset, and compatibility tests. Validation passed: all 72
  `test_optimized_paths.py` tests, 18 phase-3 endpoint tests, 7 phase-7
  endpoint tests, Python compilation, and `git diff --check`.
- **2026-11-04 — P1.3 API audit:** verified `AML.fit` passes DataFrames through
  to sklearn pipelines rather than stripping names up front; a fitted
  `LogisticRegression` exposes the original `feature_names_in_`. Named
  categorical columns work with the fold-local encoder, so `cat_cols` now
  documents names as well as indices and has regression coverage. The source
  claim of unconditional `.values` conversion is therefore not a current gap.
  EOA misspellings and other legacy names remain unchanged pending an alias and
  deprecation design. The focused feature-name test and all 72
  `test_optimized_paths.py` tests pass.
- **2026-11-04 — P1.2 search-overhead baseline:** added a search-only total
  wall timer to the random-search arm and a regression test that checks it
  includes candidate evaluation time. Measured fixed-seed Galaxy3 runs at
  budgets 8/30/60 and analyzed the existing six-dataset B=30/B=60 results.
  Search overhead rose with budget for SRS but remained small in the
  no-surrogate arm; quality differences against random were small in the
  focused sweep. `benchmarks/DESIGN.md` records the timing definition,
  measurements, reproduction command, and limitations. No optimization or
  default change was made because the residual timer cannot identify whether
  surrogate fitting or acquisition is the bottleneck. Validation: the new
  benchmark timing test passes. The standard OpenML run could not be used in
  this environment because optional `openml` is not installed; the repository's
  local Galaxy3 benchmark required no additional dependency.
- **2026-11-04 — P1.3 EOA naming aliases:** added preferred `parents_prop`,
  `elites_prop`, and `max_generations` keywords to EOA while preserving the
  misspelled/singular legacy spellings and rejecting conflicting values.
  Updated AML's default parent-count handling so a supplied parent proportion
  is not masked by the default `num_parents=30`. Updated `docs/eoa.rst` with
  the preferred names and compatibility note. Tests cover corrected aliases,
  conflicts, legacy calls, and AML forwarding. Validation passed: 97 tests
  across the alias, benchmark-timing, optimized-path, and phase-4 endpoint
  suites; Python compilation and `git diff --check` also passed.
- **2026-11-05 — P1.5 model explanations:** added deterministic, bounded
  permutation importance with explicit feature selection for wide data,
  repeat-level score drops, reference/sampling metadata, and saved ranking
  plots. Added scikit-learn partial-dependence/ICE plots with explicit one- or
  two-feature selectors, categorical-column handling, row/grid/curve limits,
  and persisted figures. The default reference is the cached holdout; callers
  can choose a registered partition. Documentation distinguishes model
  behavior from causality and explains the sampling and stability limits.
  SHAP was assessed and deferred without adding a dependency. Focused
  validation: all 82 tests in `test_optimized_paths.py` pass; changed Python
  files compile, Pylance reports no syntax errors, and `git diff --check`
  passes. The test run reports existing sklearn convergence warnings and a
  Matplotlib/Python pickle deprecation warning when persisting a figure.
- **2026-11-05 — Started P1.6 sensitive-feature detection:** confirmed the
  existing scanner only inspected names and that the stored-partition API
  intentionally passed empty frames. The scoped implementation adds local,
  bounded pattern counts for email and formatted US SSN/phone values without
  returning cell contents, and makes stored-partition scans use the selected
  data locally. Column-name matching now uses whole tokens so `Unnamed: 0`
  does not spuriously match `name`. Validation passed: 102 tests across
  `test_optimized_paths.py` and `test_phase2_endpoints.py`; changed Python
  files compile, Pylance syntax checks and `git diff --check` pass. Existing
  sklearn convergence and Matplotlib/Python pickle deprecation warnings remain.
- **2026-11-05 — P1.7 external retraining triggers:** added
  `POST /api/retraining/{task_name}/trigger` for scheduler and ingestion
  webhooks. Events must be timezone-aware and no more than ten minutes old;
  data events must match the current registered training partition and
  fingerprint. Duplicate scheduled event IDs and data partition/fingerprint
  pairs return their existing job, while changed requests using the same key
  receive HTTP 409. Automated promotion is limited to staging and requires
  at least one finite metric threshold; failed thresholds block registration
  and promotion. Trigger evidence, thresholds, job result, and registry
  history are persisted. The existing JSON job manager guarantees duplicate
  suppression within one API process only; documentation calls out its
  single-worker deployment limit. Validation passed: 95 tests across
  `test_phase6_endpoints.py` and `test_optimized_paths.py`, plus compilation,
  Pylance syntax checks, and `git diff --check`.
- **2026-11-05 — P1.8 post-hoc ensemble evaluation:** added the reproducible
  `benchmarks/dtrack/posthoc_ensemble_probe.py`. On breast cancer, iris, and
  wine (10 deterministic stratified 60/20/20 splits each), candidates were
  ranked by training-only 3-fold CV; equal-weight subsets of the top three
  were selected on calibration and compared on untouched test data with the
  CV-best individual. The selected ensemble beat/tied/lost 5/21/4 times,
  mean paired accuracy delta +0.00112, median 0; a multi-model subset was
  selected in only 2/30 runs. Since the gain is small and inconsistent,
  no production ensemble API or bundle changes were made. Full protocol,
  results, and deferral rationale are documented in
  `benchmarks/DESIGN.md`.
