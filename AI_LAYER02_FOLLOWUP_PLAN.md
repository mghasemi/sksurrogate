# AI Layer 02 Review Follow-up Plan

## Purpose and status

Track decisions and progress for incorporating the recommendations in
[`ai_layer02.md`](./ai_layer02.md) into SKSurrogate. This is a living intake
and follow-up document; update the status and progress log as work is verified
or completed.

**Status:** Phases 0, 1, 3, and 6 are complete. Phases 2, 4, and 5 are in
progress; Phase 7 is not started. The broader inference, provenance, research
workflow, and reporting work remains. This plan is based on a review of the
documentation, not a fresh audit of every implementation claim.

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
- [>] Do not add an automatic fallback yet. Define a policy for switching
  regressors and accounting for retries and budgets before revisiting it; the
  current explicit error and fit diagnostics preserve failure visibility.

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
- [~] Specify a separate inference/model-comparison surface for uncertainty,
  paired comparisons, practical effect sizes, multiplicity, and stability.
  Evaluate candidates such as bootstrap/BCa intervals, paired permutation or
  other evaluation-design-appropriate comparisons, standardized effects or
  Cliff's delta, and Holm or Benjamini–Hochberg correction. Audit the existing
  paired repeated-CV bootstrap and avoid treating folds as independent
  samples.
- [x] Make comparisons auditable: expose whether partitions, scoring,
  preprocessing, tuning boundaries, and test-set use are aligned; reject or
  clearly flag incomparable results. The nested-CV evaluator records partition
  fingerprints, CV specs, scoring identity, preprocessing path, repeat seed
  policy, group identity, fixed-estimator scope, and validation use. Paired
  comparisons reject known protocol mismatches and label legacy results,
  missing fingerprints, or opaque scoring as unverified.
- [~] Add targeted simulation tests for statistical behavior and edge cases
  before offering additional tests or inferential claims. A seeded simulation
  now checks the paired percentile bootstrap under independent paired score
  draws and a constant-difference edge case; repeated-CV dependence and
  inferential validity remain unverified, so no significance claim is made.

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
  evidence. The project checklist now carries project context into Feature
  Analysis and automatically links a completed sensitivity job to that
  project through an idempotent additive-link endpoint, avoiding replacement
  of project metadata or loss of concurrently added artifact references;
  unfinished and failed jobs are not linked. Other job kinds remain manually
  linkable after completion. Project summaries report completed analysis
  separately from model-training progress. Linked jobs now have a direct
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
- 2026-11-06: The project checklist now opens Feature Analysis with project
  context, and successful sensitivity jobs are automatically linked back to
  the project. Failed or unfinished jobs are not linked; other jobs remain
  manually linkable. The web UI production build passed.
- 2026-11-06: Added an idempotent ``POST /api/projects/{project_id}/artifacts``
  endpoint for adding a single reference while preserving all project fields
  and existing snapshots. Project-launched sensitivity jobs now use this
  operation instead of replacing the project's artifact list.
