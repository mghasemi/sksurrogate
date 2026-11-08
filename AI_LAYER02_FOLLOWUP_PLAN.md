# AI Layer 02 Review Follow-up Plan

## Purpose and status

Track decisions and progress for incorporating the recommendations in
[`ai_layer02.md`](./ai_layer02.md) into SKSurrogate. This is a living intake
and follow-up document; update the status and progress log as work is verified
or completed.

**Status:** Phases 0–3 and 6 are complete. Phases 4 and 5 are in progress;
Phase 7 is not started. Broader inferential methods, provenance, and research
workflow work remain. This plan is based on a review of the documentation,
not a fresh audit of every implementation claim.

The existing [`AI_LAYER_ROADMAP.md`](./AI_LAYER_ROADMAP.md) remains the
canonical implementation plan and status record for overlapping work. Its
bounded EDA, evaluation, provenance, notebook, and API-first project work is
already complete in part or in full. This document tracks the additional
review-specific work and the decisions needed before expanding those
foundations.

Legend: `[ ]` open, `[~]` in progress, `[x]` done, `[>]` deferred or rejected
with rationale.

## Working principles

- Verify each review claim against current source code, tests, and generated
  docs before changing behavior. Treat the PDF page references as pointers,
  not proof that the source documentation is still inaccurate.
- Extend working foundations instead of duplicating them. Update the linked
  roadmap when a follow-up changes an existing milestone's scope or status.
- Prioritize scientific validity, reproducibility, and a coherent Research
  workspace over adding algorithms or deployment features for their own sake.
- Make methods, assumptions, data partitions, and uncertainty explicit.
  Never treat CV folds as independent observations or present association,
  attribution, or sensitivity as causation.
- Keep optional integrations optional. Do not introduce an LLM, document
  extraction, SHAP, or another heavyweight dependency into the core path
  without a bounded use case and a review of reproducibility and data handling.
- For any feature that proposes experiments, keep suggestions separate from
  real-world execution and require users to validate physical constraints and
  safety.

## Plan

### Phase 0 — Verify and correct user-facing documentation

**Priority:** Immediate; bounded documentation pass.

- [x] Inspect the current documentation sources and generated PDF to verify
  the reported contradictions and port/name examples.
- [x] Clarify the promotion quality-gate text: distinguish the default bundle
  loadability check from standalone custom metric thresholds. The normal
  promotion endpoint has no verified way to configure metric thresholds.
- [x] Align the Research Projects description with the current Projects UI.
  Retain accurate limitations (for example, notebook execution, deletion,
  reports, and authorization) rather than removing caveats that still apply.
- [x] Verify the actual configured API port, then make the Chapter 8 and
  Chapter 14 examples consistent with the deployment and other chapters.
- [x] Audit first-party snippets, feature tables, and migration examples for
  legacy `CamelCase` tracking calls. Use canonical `snake_case` names where
  available and remove or correct names that are not public APIs.
- [x] Rebuild the Sphinx HTML and PDF and inspect the affected sections after
  source edits. The generated manual has 241 pages; the root review PDF is
  untracked and was not overwritten.

**Done when:** the source docs and generated PDF agree with the implemented
promotion behavior, Projects UI, configured port, and canonical API names.

### Phase 1 — Audit existing foundations and check the quick-start preset

**Priority:** Before expanding features; avoids duplicated or misleading work.

- [x] Map each recommendation to existing code, tests, and the relevant
  section in [`AI_LAYER_ROADMAP.md`](./AI_LAYER_ROADMAP.md). Record whether it
  is existing, partial, missing, or intentionally deferred.
- [x] Test classification and regression presets on mixed DataFrames without
  `cat_cols`; verify the explicit categorical path and existing numeric-array
  coverage separately.
- [x] The mixed-DataFrame preset repro produced infinite CV losses and an
  unfitted estimator. Presets now use `FoldLocalDataPreprocessor` for
  DataFrames when `cat_cols` is omitted, with fold-local transforms. If all
  candidate pipelines fail, `AML.fit()` now raises a diagnostic `ValueError`
  instead of returning an unfitted pipeline. The zero-configuration claim is
  limited to DataFrames; array-like inputs must be numeric unless the explicit
  `cat_cols` path is used.
- [x] Reproduce the `HilbertRegressor` failure inside `SurrogateSearch`. A
  degree-5 basis with four support points raises the documented rank-deficiency
  `ValueError`; `surrogate_info_` records the failed fit and no fallback masks
  the error.
- [x] Decision complete: do not add an automatic fallback yet. A switching
  policy must specify retry accounting, evaluation budget, and surfaced
  diagnostics first; current explicit errors and fit diagnostics preserve
  failure visibility.

**Done when:** preset guarantees and surrogate failure behavior are evidenced
by focused tests, and each review item has a recorded disposition.

Audit summary:

| Review area | Existing foundation | Follow-up disposition |
| --- | --- | --- |
| Documentation discrepancies | Source docs and UI/API implementations | Corrected and rebuilt in Phase 0. |
| Presets and input preparation | Bounded RF/ExtraTrees presets; fold-local transformer | Preset DataFrames now preprocess fold-locally; explicit `cat_cols` and numeric-array paths remain. |
| EDA and statistical inference | `dataset_profile`; paired repeated-CV comparison | Extend, do not duplicate, in Phase 2. |
| Prediction uncertainty | Split-conformal helpers plus API/UI interval evaluation | Phase 3 adds Wilson coverage intervals, width and Winkler score, subgroup coverage, and split provenance; conditional coverage beyond user-specified groups remains open. |
| Explanations | Permutation importance, PDP/ICE, sensitivity analysis | Clarify interpretations and assess ALE/conditional/stability methods in Phase 5. |
| Research workspace and provenance | API-first project metadata, evidence records, and artifact metadata snapshots | Expand the UI workflow and define recovery/versioning behavior in Phase 4. |
| Reports and notebooks | JSON/Markdown run reports and executable notebooks | Audit for expanded scientific output needs in Phase 6. |
| Experiment design | Seeded Latin hypercube and bounded GP-LCB proposal MVP | Evaluate advanced acquisitions only with reproducible evidence in Phase 7. |
| LLM, document extraction, synthetic DGPs | No complete implementation identified in this review | Defer as separate strategic bets in Phase 7. |
| Surrogate fallback | Fit failures propagate and are recorded | Defer automatic switching until retry/budget semantics are specified. |

### Phase 2 — Strengthen scientific EDA and statistical inference

**Priority:** Highest product priority; build on, do not replace, the existing
bounded profile and repeated-CV comparison APIs.

- [x] Extend the current profile with robust
  descriptive statistics, distribution and missingness diagnostics,
  exact duplicate/constant/all-missing feature flags, duplicate rows, and IQR
  outlier signals, plus bounded descriptive numeric distributions by
  categorical group, Spearman/Kendall rank coefficients, and VIF/condition
  diagnostics for numeric features, plus bounded missingness-pattern counts.
  General nonlinear dependence measures, missingness maps/association tests,
  and temporal leakage checks remain open or deferred pending concrete method
  contracts.
- [x] Document the methods, assumptions, sample/feature limits, and
  reproducibility parameters for the extended profile. Outlier and association
  results remain diagnostic, not conclusions; group comparisons are
  descriptive only and perform no hypothesis tests.
- [x] Specify the inference/model-comparison contract and evaluate candidate
  methods. The existing paired repeat-level bootstrap and Cohen's ``d_z`` are
  descriptive summaries only: repeated-CV observations overlap, so the
  interval is not a nominal-coverage inferential interval and excluding zero
  is not a significance claim. BCa does not repair dependence; paired
  permutation needs a defensible exchangeability/randomization design;
  Cliff's delta answers a different ordinal-effect question; multiplicity
  correction needs a pre-specified comparison family. Defer these methods
  until an appropriate independent-unit evaluation design and calibrated
  inferential contract exist.
- [x] Make comparisons auditable: expose whether partitions, scoring,
  preprocessing, tuning boundaries, and test-set use are aligned; reject or
  clearly flag incomparable results. The nested-CV evaluator records partition
  fingerprints, CV specs, scoring identity, preprocessing path, repeat seed
  policy, group identity, fixed-estimator scope, and validation use. Paired
  comparisons reject known protocol mismatches and label legacy results,
  missing fingerprints, or opaque scoring as unverified.
- [x] Add targeted simulation tests for supported statistical behavior and
  edge cases before offering further claims. Seeded simulation checks paired
  percentile-bootstrap coverage for independent paired score draws (96/100
  coverage for a true 0.08 shift) and verifies constant-difference behavior;
  tests also cover invalid/unpaired inputs and protocol mismatch. Repeated-CV
  dependence is not validated as inference: no significance claim is made
  and dependence-sensitive inference is deferred.

**Done when:** EDA and comparisons return reproducible computational results
with explicit assumptions and prevent common invalid interpretations.

### Phase 3 — Make uncertainty reporting more informative

**Priority:** High; coordinate with the existing evaluation work.

- [x] Review current conformal interval helpers and evaluation surfaces. The
  Python helpers return intervals and assumptions; the API/UI already report
  observed coverage and calibration size but lacked uncertainty, interval
  width/score, and explicit split provenance.
- [x] Add interval diagnostics for observed versus nominal coverage, uncertainty
  around empirical coverage, interval width, and an appropriate interval score.
- [x] Present sample size and uncertainty alongside coverage; do not describe
  a small-sample estimate as proof of calibration.
- [x] Investigate subgroup coverage as a diagnostic, with
  clear caveats about sample size and the limits of marginal conformal
  guarantees. User-specified holdout groups are bounded; more advanced
  feature-region analysis remains out of scope.
- [x] Ensure reports and UI display the calibration/evaluation split and the
  assumptions needed to interpret results.

**Done when:** users can assess both coverage and interval usefulness without
mistaking empirical coverage on a small sample for a guarantee.

### Phase 4 — Connect research projects to evidence and reproducible artifacts

**Priority:** High; extend the existing API-first project metadata MVP.

- [~] Define a minimal research record model for hypotheses, analyses,
  findings, and research artifacts. Link claims to supporting or contradicting
  evidence and keep authoritative datasets, jobs, and bundles in their
  existing stores. The API now has append-only hypothesis, analysis, finding,
  and conclusion records with evidence references to project-linked artifacts;
  hypotheses have a bounded status lifecycle with append-only transition
  history, while their statement and evidence remain immutable. Hypotheses can
  also capture a linked project question, and records can accumulate linked
  evidence after creation through an append-only endpoint. Broader
  research-artifact ingestion remains open.
- [~] Design a separate Research workspace in the UI for the progression
  `Question → Hypothesis → Dataset → Analysis → Experiment → Result →
  Conclusion`; keep the current operational ML lifecycle available. The
  Projects summary now tracks these stages from existing project questions,
  hypothesis/conclusion records, dataset references, completed analysis/model
  jobs, and linked results. The project UI presents the checklist and directs
  actionable stages to the existing feature-analysis, experiment, dataset,
  and bundle pages; broader workflow orchestration remains open.
- [~] Strengthen references with immutable/versioned identities and snapshots
  of dataset fingerprints and relevant run configuration. Specify behavior
  for missing, changed, or deleted upstream artifacts. New links now capture
  dataset partition, bundle run-configuration, or job configuration/result
  fingerprints; edits preserve snapshots and the summary marks changed,
  missing, or legacy-unverified references. Snapshots do not recover deleted
  artifacts or prevent upstream changes.
- [>] Evaluate SQLite or another transactional store when concurrency and
  query requirements justify migrating the current file-backed project
  metadata. Current deployments are explicitly single-process for project
  writes and use bounded file lookups, so retain atomic JSON writes; revisit
  when multi-worker writes or query needs become requirements.
- [~] Add end-to-end tests showing that project evidence links resolve to the
  exact analysis inputs and outputs, and that unsupported capabilities are
  described accurately. Current tests validate links to exact registered
  partitions and bundle versions and verify changed/deleted snapshot behavior.
  The project progress summary now distinguishes completed modeling jobs from
  completed analysis jobs, with endpoint coverage. Nested-CV jobs now retain
  their submitted request; an integration test links one as project evidence
  and follows the exact request, partition fingerprint, result, and snapshot.
  Sensitivity jobs now retain their method request plus partition fingerprint,
  row count, and input columns; an API test links the result as project
  evidence. The project checklist now carries project context into data
  registration, Feature Analysis, and Experiments. A successfully registered
  partition is automatically linked; completed sensitivity, experiment, and
  pipeline-optimization jobs are linked through an idempotent additive-link
  endpoint, avoiding replacement of project metadata or loss of concurrently
  added artifact references; unfinished and failed jobs are not linked.
  Experiment and pipeline-optimization results also link the produced bundle
  when the result provides a model version. Launching analysis or experiments
  retains the submitted task identity, so changing the active task while a
  job runs does not misdirect its project link. Project summaries report
  completed analysis separately from model-training progress. Linked
  jobs now have a direct
  Inspect job link into the Jobs page's tracker and result view; research-
  record evidence repeats that link when its job still exists, so claims can
  be followed directly to their supporting output. A broader research
  workspace remains open.
  Saved evidence now also displays the linked artifact snapshot's current,
  changed, missing, or unverified state beside the claim. The result stage
  requires a non-null completed-job result or a linked model bundle; a
  completed job without a result no longer marks the stage complete.

**Done when:** a scientist can organize an investigation and inspect its
evidence and provenance without duplicating experiment or registry data.

### Phase 5 — Clarify and expand explanations

**Priority:** High, after the comparison and provenance contracts are clear.

- [x] Make the distinctions between predictive association, model
  attribution, input sensitivity, and causal effects prominent in API output,
  UI labels, and documentation. Explain and sensitivity endpoints return
  interpretation categories and non-causal caveats; Explain and Feature
  Analysis display the distinction in the UI.
- [~] Evaluate ALE and conditional approaches where correlated features or
  implausible PDP combinations can mislead. Scope interactions, subgroup
  explanations, error analysis, and resampling stability to supported models
  and bounded inputs. Documented ALE as a candidate that reduces reliance on
  off-support averaging but deferred implementation until categorical/empty-
  bin and classifier-output behavior are specified. Added an opt-in bounded
  permutation-importance sensitivity summary over 2–5 sampled reference sets;
  it reports descriptive cross-sample SD and top-rank frequency, not confidence
  intervals. PDP/ICE, conditional explanations, and ALE stability remain open.
- [ ] Revisit SHAP as an optional integration only after supported estimators,
  local-explanation needs, runtime limits, and dependency policy are specified.
- [x] Enable comparison of analytical perspectives (for example, correlation,
  permutation importance, SHAP when available, and Sobol sensitivity) while
  explaining that they answer different questions and need not agree. Feature
  Analysis and Explain now cross-link, and their copy distinguishes observed
  association, generated-input model sensitivity, and permutation score
  attribution without combining unlike values into one ranking. SHAP remains
  a separately deferred integration.
- [x] Add tests and examples that cover correlated inputs, out-of-support
  feature combinations, and explanation stability limits. A deterministic
  perfectly correlated example verifies that marginal PDP can leave observed
  support; API assertions verify repeatability for a fixed seed and that an
  alternative seed can change the explanation on the same data. An opt-in,
  bounded diagnostic now summarizes per-feature permutation-importance mean,
  sample SD, and top-rank frequency across distinct reference subsets. It is
  descriptive sensitivity, not a confidence interval. The endpoint reports
  reference row counts/sampling and settings; the Explain page surfaces these
  details and the optional summary.

**Done when:** supported explanations expose their reference data, limits,
and interpretation category and do not imply causal conclusions.

### Phase 6 — Improve scientific outputs and reusable workflows

**Priority:** Medium; reuse the existing report exports and notebooks.

- [x] Audit the current JSON/Markdown report and notebook templates against
  the expanded EDA, inference, uncertainty, explanation, and provenance needs.
  The API notebook demonstrates profiling, evaluation, and report exports; the
  scientific notebook covers seeded profiling, nested CV, and permutation
  importance. The run report covers methodology, metrics, uncertainty,
  limitations, and fingerprints. Both notebooks executed end-to-end in the
  current environment. The generic run report does not embed EDA or explanation
  outputs; those are now exported alongside run results by the scientific
  notebook rather than duplicating them in every run report.
- [x] Add publication-oriented tables, figures, and notebook templates only
  for workflows not already covered; keep examples executable and seeded. The
  scientific notebook exports profile/association tables, performance and
  permutation-importance CSVs, methodology/limitations metadata, and a 300-dpi
  figure separating repeated-CV from held-out validation results. Seeded inputs
  do not guarantee identical search/model results across repeated executions.
- [x] Ensure reports include methodology, assumptions, data/run identity,
  uncertainty, limitations, and links to source artifacts. The Markdown report
  now links to dataset metadata, run configuration, and bundle download, and
  already includes protocol/configuration, fingerprints, uncertainty, and
  known gaps. It now also presents recorded target handling, observation
  exclusion metadata, metric scope, and evaluation protocol details under an
  assumptions/scope section. Unknown upstream exclusions remain explicitly
  marked as not recorded.

**Done when:** a user can take a reproducible analysis from the API or Python
workflow into a clear, self-contained scientific report.

### Phase 7 — Defer strategic extensions until foundations are ready

**Priority:** Later; separate validation gates for each research bet.

- [ ] **LLM-assisted research:** first stabilize deterministic project,
  profile, run-config, explanation, and report APIs. Then evaluate a bounded
  project-synthesis workflow (for example, combining a profile, run
  configuration, explanation summaries, and drift report) that cites
  structured evidence, preserves uncertainty, and distinguishes generated
  interpretation from computed results.
- [ ] **Row-level explanations:** define a faithful local-explanation method
  and its API contract (for example, a row-level inference explain endpoint)
  before adding an LLM summary; do not label an ICE difference as a feature
  attribution.
- [ ] **Document-to-tabular extraction:** prototype separately from
  fold-local preprocessing. Require schema-driven extraction, source-page or
  passage provenance, confidence, human validation, unit normalization,
  duplicate/contradiction handling, and export to a typed dataset.
- [ ] **Controlled synthetic data:** add a small, seeded set of known
  data-generating processes (such as nonlinear interactions, correlated
  predictors, heteroscedastic noise, and distribution shift) to validate
  methods and benchmarks against known truth.
- [ ] **Experiment design:** assess expected/probability improvement,
  confidence-bound acquisition, noise-aware models, kernel selection,
  replication, constraints, batch, multi-objective, and multi-fidelity design
  against the bounded current implementation. Keep the existing
  proposal-only contract; defer broader methods until reproducible benchmarks
  demonstrate value.
- [ ] Keep symbolic regression, physics-informed modeling, survival analysis,
  and broad domain-specific algorithm collections out of this cycle unless a
  concrete demand and validation plan emerge.

**Done when:** each extension has a bounded user need, a reproducible
evaluation, clear limitations, and a decision to implement or defer.

## Progress log

- 2026-11-05: Created this review follow-up plan. No implementation or
  documentation fixes have been marked complete; source verification is the
  first step.
- 2026-11-05: Completed Phase 0 source-doc corrections for promotion checks,
  Projects UI availability, API ports, and first-party tracking names.
  Sphinx HTML and PDF builds succeeded; the PDF build produced 241 pages and
  the updated passages were verified in its text output. Existing Sphinx
  warnings remain.
- 2026-11-05: Began Phase 1. Added fold-local preprocessing for preset
  DataFrame inputs without `cat_cols`, explicit failure when all candidate
  pipelines have non-finite CV scores, and regression tests for classification,
  regression, explicit categorical encoding, and all-candidate failure.
  Validation: 129 tests passed across `test_optimized_paths.py`,
  `test_api_experiments.py`, and `test_api_simplification.py`.
- 2026-11-05: Reproduced rank-deficient `HilbertRegressor` failure during
  `SurrogateSearch`; the `ValueError` propagates and fit diagnostics record
  the failure. Automatic fallback is deferred.
- 2026-11-05: Completed the recommendation-to-roadmap audit. The review's
  broad EDA, inference, project, explanation, and provenance proposals extend
  existing MVPs rather than requiring replacement subsystems.
- 2026-11-05: Started Phase 2. Extended ``dataset_profile`` with robust numeric
  summaries, IQR outlier signals, constants, all-missing and exact duplicate
  feature flags, and duplicate-row counts explicitly scoped to sampled rows
  and selected features. Numeric moments are scaled before computation so
  extreme finite magnitudes do not overflow intermediate sums.
- 2026-11-05: Audited ``nested_cv_evaluation`` and found that categorical
  dummy columns were inferred before CV folds. Mixed-type DataFrames now use
  the existing ``FoldLocalDataPreprocessor`` within each fit; the result
  exposes this path, and the docs clarify that the API evaluates the supplied
  estimator rather than tuning/selecting among candidates.
- 2026-11-05: Added an evaluation protocol to nested-CV results and task
  metadata. ``compare_repeated_cv_results`` now rejects known mismatches and
  reports verified versus unverified alignment; unverifiable legacy results
  remain usable with an explicit unverified status. Focused validation passed:
  111 tests across optimized-path, time-series, and quantile evaluation suites;
  Sphinx HTML build and Python syntax checks succeeded. Existing estimator
  convergence and Python 3.14 deprecation warnings remain.
- 2026-11-05: Added optional ``group_by`` summaries to ``dataset_profile`` and
  the registered-partition profile endpoint. The result gives bounded numeric
  distributions and counts per group, combines low-frequency categories, and
  explicitly excludes missing group labels. No hypothesis tests or
  multiplicity-adjusted inference is claimed. Profile and endpoint tests passed
  (10 tests); broader Phase 2 inference and dependence work remains open.
- 2026-11-05: Added supplementary Spearman and Kendall coefficients for
  numeric target/feature and feature/feature pairs, retaining the existing
  Pearson association output unchanged. Pairwise complete finite sample
  counts are included, and documentation limits interpretation to monotonic
  rank association rather than general nonlinear dependence.
- 2026-11-05: Added complete-case VIF and standardized condition-number
  diagnostics for selected numeric features. Rank-deficient designs return a
  reason instead of infinity; columns are normalized before matrix operations
  to avoid overflow on extreme finite values. Focused profile tests pass
  without numerical warnings.
- 2026-11-05: Added bounded exact missingness-pattern counts over selected
  features. The report retains the most frequent patterns, aggregates the
  remainder, and states that counts apply only to the sampled rows and do not
  test missingness mechanisms.
- 2026-11-05: Phase 2 additions passed 126 tests across optimized-path,
  dataset-profile endpoint, time-series, and quantile suites. The Sphinx HTML
  build succeeded with 15 warnings (mostly existing references and excluded
  documents); changed Python files compiled, Pylance reported no diagnostics
  in the profile module or endpoint router, and ``git diff --check`` passed.
- 2026-11-05: Added bounded exact missingness-pattern counts over selected
  features. The report retains the most frequent patterns, aggregates the
  remainder, and states that counts apply only to the sampled rows and do not
  test missingness mechanisms. Final Phase 2-focused validation passed 127
  tests; the Sphinx HTML build succeeded with 15 warnings.
- 2026-11-05: Started Phase 3. Audited the existing conformal Python helpers
  and API/UI: empirical coverage and calibration size were reported, but there
  was no interval width/score or uncertainty interval, and fit/calibration/
  holdout provenance was incomplete. Added a public diagnostic helper and
  wired Wilson coverage intervals, width, Winkler score, and split counts into
  the endpoint and Explain UI.
- 2026-11-05: Extended interval diagnostics with Wilson coverage intervals,
  mean/median width, Winkler score, and optional bounded subgroup coverage.
  The API and Explain UI expose fit/calibration/holdout counts and split
  provenance; docs distinguish empirical-coverage uncertainty from conformal
  guarantees. The web UI production build passed, Sphinx HTML succeeded with
  three existing warnings, and Python compilation plus ``git diff --check``
  passed.
- 2026-11-05: Ran the combined Phase 2/3 EDA, API, time-series, and quantile
  regression suites: 141 tests and 21 subtests passed. A seeded simulation
  estimated 96% coverage across 100 independent paired-score samples for the
  percentile bootstrap; the simulation supports only that IID synthetic case,
  not repeated-CV inference. It also exposed a roundoff artifact that made
  Cohen's ``d_z`` enormous for a constant decimal shift; effect size is now
  reported as undefined when score variation is at floating-point precision.
  Python compilation and whitespace checks passed.
- 2026-11-05: Started Phase 4 with append-only project research records for
  hypotheses, analyses, findings, and conclusions. Findings/conclusions
  require linked evidence labeled as supporting or contradicting; evidence
  references must point to a project artifact, and a referenced artifact cannot
  be removed while a record depends on it. The Projects UI can add and display
  these records. New artifact links include preserved snapshots of dataset
  fingerprints, bundle run configuration/lineage, or job configuration/result
  fingerprints. The project summary reports current/changed/missing/unverified
  snapshot state. Focused API tests, the web UI build, and the Sphinx HTML build
  passed; Sphinx warnings are in pre-existing documents and API docstrings.
  Deleted-artifact recovery, record editing/deletion, and the full research
  progression remain open.
- 2026-11-05: Extended the Projects UI to attach multiple evidence artifacts
  per record, each independently labeled as supporting or contradicting.
  Added an HTTP-level roundtrip test that posts a project, a two-source finding,
  and then checks current snapshot status. The API accepts up to 20 evidence
  references per record; the UI exposes that bounded workflow.
- 2026-11-05: Evaluated whether to migrate project metadata to SQLite.
  Deferred migration: project writes are documented as single-process and
  current reads are bounded by a small metadata collection; no multi-worker
  or querying requirement presently justifies a storage change.
- 2026-11-05: Started Phase 5 by labeling API outputs and UI for descriptive
  Pearson association, permutation model attribution, partial-dependence model
  response profiles, and SALib input sensitivity. Added explicit non-causal
  caveats and described the API's independent rectangular sensitivity ranges,
  including the risk of unrealistic combinations for correlated/categorical
  inputs. The explanation/sensitivity and API regression suites passed (26
  tests); the UI production build and Sphinx HTML succeeded with three existing
  toctree warnings. Python compilation and ``git diff --check`` passed.
- 2026-11-05: Evaluated ALE as a bounded next method for correlated inputs.
  Documented its advantage over marginal PDP and the remaining support,
  categorical-bin, and classifier-output constraints; deferred implementation
  until those method contracts can be tested. Added exact seeded repeatability
  checks for the current explanation response and explicitly separated this
  from resampling stability. A deterministic perfect-correlation test now
  demonstrates marginal PDP predictions outside the observed joint support;
  the fitted model returns zero on every observed row but exceeds 1.0 on the
  marginal PDP grid. The API also returns reference partition/row counts,
  whether sampling occurred, and the settings/seed used, with tests for both
  sampled and all-row paths. A separate seed produces changed explanation
  output in the deterministic API fixture. The Explain UI displays
  source/used row counts, whether sampling occurred, the sampling seed when
  applicable, and permutation-repeat count. Optional permutation-importance
  sample sensitivity is now available for 2–5 independently drawn subsets,
  bounded by the row cap, and is explicitly not a confidence interval.
- 2026-11-05: Started Phase 6 audit. Confirmed the two existing notebooks cover
  the primary Python and API workflows, and the report already includes method,
  metrics, uncertainty, limitations, and lineage. Added direct API links to
  registered dataset metadata, machine-readable run configuration, and bundle
  download; focused endpoint coverage verifies the emitted links.
- 2026-11-05: Executed every code cell in both notebooks. The seeded offline
  workflow completed profiling, bounded search, repeated nested CV, and
  permutation importance. The API workflow completed data registration,
  profiling, search/evaluation jobs, report export, and research-project
  linkage against an isolated temporary API store. The audit found the current
  report/templates did not produce EDA summaries, explanation summaries, or
  publication figures at that point; the scientific notebook additions below
  addressed those outputs without duplicating them in every run report.
- 2026-11-05: Added publication outputs to the existing scientific notebook:
  profile columns and target associations, a performance summary, permutation
  importance, JSON metadata with sample/protocol/limitation details, and a
  300-dpi two-panel figure. Executed every notebook code cell and verified all
  six generated files in temporary storage. Repeated executions produced
  different score outputs despite fixed seeds, so the notebook now explicitly
  distinguishes seeded inputs from guaranteed result determinism.
- 2026-11-05: Added opt-in API/UI reference-sample sensitivity for permutation
  importance. The request is bounded to five samples and requires a source
  partition larger than the per-sample row cap. It reports descriptive mean,
  sample SD, and top-rank frequency per feature; tests verify seed repeatability
  and explicit rejection when resampling cannot vary. This does not address
  stability of PDP or establish generalization across datasets. Each sampled
  subset is distinct; same-seed repeats return identical summaries.
- 2026-11-05: Linked Feature Analysis and Explain as complementary perspectives.
  Added UI and documentation guidance distinguishing Pearson association,
  generated-input surrogate sensitivity, and permutation score attribution,
  and explicitly warned that their scales/ranks are not directly comparable.
- 2026-11-05: Nested-CV job records now retain the submitted request, and the
  result protocol includes train/validation partition fingerprints. An
  integration test links the completed job as project evidence and verifies
  its request/result plus the matching training-partition snapshot.
- 2026-11-05: Sensitivity jobs now retain their submitted method request and
  report their source partition fingerprint, row count, and feature columns.
  An API integration test links the completed job to a project analysis record
  and verifies the request, result, and artifact snapshot remain addressable.
- 2026-11-05: Added a distinct project analysis-progress stage for completed
  sensitivity and nested-CV jobs, leaving the modeling stage limited to
  experiment, optimization, and retraining jobs. The Projects UI links the
  analysis step to Feature Analysis; endpoint coverage keeps the two stages
  separate.
- 2026-11-05: Added an Inspect job link to linked job artifacts. The Jobs page
  reads a ``job_id`` query parameter so the existing tracker opens directly
  with that job selected; project evidence can now be followed to its recorded
  request and full result in the UI.
- 2026-11-05: Added the same Inspect job navigation to job references shown
  under saved research records, gated on the linked job still existing.
  Updated the Projects UI documentation; the API remains authoritative for
  missing/changed artifact state.
- 2026-11-05: Research records now show each evidence artifact's current
  snapshot state beside the claim, including changed, missing, and legacy
  unverified states, so evidence validity is visible in context.
- 2026-11-05: Added hypothesis statuses (proposed, under investigation,
  supported, contradicted, inconclusive) and append-only status history.
  Status changes preserve the original hypothesis text and evidence; the API,
  Projects UI, tests, and research-project documentation expose this lifecycle.
  Legacy hypotheses receive a read-time proposed status anchored to their
  original creation time and can then enter the same transition history.
- 2026-11-05: Connected hypotheses to project questions with an optional
  question selector. The API only accepts question text already present on
  the project and stores the captured text on the hypothesis record.
- 2026-11-05: Project summaries now track the requested research progression
  from question through hypothesis, dataset, analysis, model run, result, and
  conclusion. Tests verify completion derives from project metadata and linked
  artifact/job state without conflating analysis with model training.
- 2026-11-05: Added append-only evidence attachment for existing project
  records. This lets hypotheses begin without evidence and accumulate linked
  support or contradiction as analyses finish; the API rejects unlinked and
  duplicate references and enforces the 20-reference limit.
- 2026-11-05: Added nested-CV and pipeline-optimization to the Jobs page kind
  filter and updated job-page/tracker descriptions, so the complete set of
  project-linked analysis and modeling job kinds can be browsed directly.
- 2026-11-05: Tightened the research progression's result stage to require a
  non-null result from a completed linked job or an existing linked bundle.
  Regression coverage confirms a completed job record without a result does
  not satisfy the stage.
- 2026-11-05: Closed the remaining Phase 2 comparison items. Documented that
  repeat-level bootstrap intervals and paired ``d_z`` are descriptive, not
  nominal-coverage inference under repeated-CV dependence; BCa, paired
  permutation, Cliff's delta, and multiplicity correction are deferred with
  method-specific rationale. Tightened the seeded IID bootstrap simulation
  acceptance threshold to 90%; the fixed simulation yielded 96/100 coverage
  for a true 0.08 shift. Five focused paired-comparison tests passed. All
  Phase 0–2 checklist items are now marked done, including the explicit Phase 1
  decision to defer automatic surrogate fallback.
- 2026-11-06: The project checklist now opens Feature Analysis with project
  context, and successful sensitivity jobs are automatically linked back to
  the project. Failed or unfinished jobs are not linked; other jobs remain
  manually linkable. The web UI production build passed.
- 2026-11-06: Added an idempotent ``POST /api/projects/{project_id}/artifacts``
  endpoint for adding a single reference while preserving all project fields
  and existing snapshots. Project-launched sensitivity jobs now use this
  operation instead of replacing the project's artifact list.
- 2026-11-05: Extended project context to Experiments. Completed experiment
  and pipeline-optimization jobs launched there are automatically attached to
  the project; failed and unfinished jobs are not. The project evidence API's
  atomic append operation avoids overwriting concurrent artifact links.
- 2026-11-05: Project-launched analysis and experiment jobs now retain the
  task name used at submission for completion-time artifact linking. Switching
  the active task while a background job runs no longer changes the task
  identity used for the project reference. The web UI type-check/build passed.
- 2026-11-05: Completed experiment and pipeline-optimization jobs now also
  link their produced model bundle when the result contains ``model_version``.
  Link failures are surfaced in the Research Project card rather than silently
  ignored; bundle snapshots preserve the existing run configuration and
  lineage evidence.
- 2026-11-05: The project checklist now carries context into data registration.
  A successful partition upload links its returned task/partition identity
  through the additive endpoint; upload or linking failures remain visible in
  the Datasets page. Added regression coverage that the appended partition
  snapshot captures its registered fingerprint and reports current status.
  Validation passed: 20 project endpoint tests, the web UI production build,
  and ``git diff --check``.
- 2026-11-07: Completed the explainability visibility pass per
  ``HANDOFF-explainability-visibility.md`` (T1–T4; docs/assets only, no package
  source changes). T1: ``readme.rst`` gained an "Understand what you tuned"
  section between the hybrid loop and Quick start covering exactly the two
  shipped capabilities — model explanations (permutation importance + partial
  dependence via the control-plane explain endpoint and web UI) and sensitivity
  analysis (Sobol/Morris/δ, built slug ``sensitivity.html``) — with
  descriptive-not-causal wording per the §3 guardrails. T2: ``LAUNCH_PLAN.md``
  now leads differentiator #1 in §3.2 point 3 with robustness/stability citing
  ``docs/images/stability_contour.png`` as artifact (statistical evaluation and
  research workspace kept verbatim at positions 2–3; surrogate hybrid search
  retained as item 4), and the §2.5 media kit gained the stability figure plus
  explain-page screenshots extracted from existing manual/walkthrough shots with
  source pointers. T3: new asset script ``docs/images/make_stability_contour.py``
  (public API only — ``SurrogateRandomCV`` evaluation history over the suite's
  shared 2-D slice, DESIGN.md §4; stock un-tuned RBF GP; matplotlib) ships
  credit-g budget 48 seed 7 as ``docs/images/stability_contour.png`` (1200×750);
  an optdigits budget 36 variant is retained in scratch. Deviation: published
  runs under ``benchmarks/results/`` store per-evaluation scores without
  hyperparameter values, so the figure regenerates seeded points at
  suite-equivalent budgets instead of re-plotting archived runs. T4: media-kit
  inventory updated accordingly.
- 2026-11-07: Verification for the visibility pass: docutils parse of
  ``readme.rst`` clean; every new RTD link resolves to an existing ``docs/``
  file on this branch (controlplane, mltrace, sensitivity); the T3 script ran
  to completion in the project venv and again inside the Docker py3.12 dev
  image (handoff §5.2; requires an ephemeral `openml` install — see open
  questions), producing a byte-identical credit-g PNG in both environments
  (sha256 match). Peak-vs-
  plateau contrast quantified by a seeded re-run of the identical fits: credit-g
  shows a sharp peak (~1% of grid within 95% of surface range; ~0.13 acc points
  leaf-axis spread at fixed depth) while optdigits shows a broad plateau (~54%
  of grid within 75%). Test subset as sanity check (no source changes):
  ``test_benchmark_methods.py`` + ``test_surrogate_diagnostics.py`` = 8 passed.
  All changes left in the working tree for maintainer review; nothing committed.
  Open items / flags: (i) published runs under ``benchmarks/results/`` store no
  per-evaluation hyperparameters, so the figure regenerates seeded points at
  suite-equivalent budgets instead of re-plotting archived runs; (ii)
  ``benchmarks/datasets.py::_download_openml`` imports ``openml`` before its
  cache check, so any OpenML dataset load fails without the ``[benchmark]``
  extra even when a warm cache exists — the Docker verification therefore
  installed it ephemerally inside the container; (iii) §3.2 point 3 now lists
  four differentiators: robustness/stability was added as #1 per the handoff,
  and surrogate hybrid search (previously first) is retained at position 4 —
  maintainer may prefer it re-promoted or dropped.
