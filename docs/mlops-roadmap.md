# SKSurrogate MLOps Roadmap

Status: in progress

This document tracks the work needed to evolve SKSurrogate from an offline AutoML and experiment-tracking library into a reproducible, deployable, and monitorable MLOps system.

## Working Rules

- Implement one phase at a time.
- Keep each phase independently testable.
- Preserve existing public APIs unless a migration path is documented.
- Add focused tests before broad refactors.
- Do not mark a phase complete until its acceptance criteria and validation commands pass.
- Update this file at the start and end of every implementation session.

## Current Baseline

Completed before this roadmap:

- [X] Python virtual environment and dependency compatibility audit
- [X] NumPy, pandas, category-encoders, and scikit-learn compatibility fixes
- [X] AML estimator delegation: `predict`, `predict_proba`, `score`, `get_params`, `set_params`
- [X] Reproducible random-state propagation through AML, EOA, and surrogate sampling
- [X] Time and evaluation budgets: `time_limit`, `max_evals`
- [X] Out-of-fold stacking features to reduce target leakage
- [X] Group-aware CV propagation
- [X] Structured trial history, `cv_results_`, durations, failures, and Pareto frontier
- [X] mltrace environment and custom task metadata
- [X] Focused regression tests and runnable examples

## Phase 1: Data Identity and Schema Contracts

Status: complete

Goal: make every training dataset identifiable and validate train/serving compatibility.

Tasks:

- [x] Add a deterministic dataset fingerprint based on schema and content.
- [x] Record source URI, ingestion timestamp, row count, feature count, target name, and schema.
- [x] Define a serializable feature schema containing names, dtypes, nullability, and categorical values.
- [x] Add `validate_data()` before training and before prediction.
- [x] Track dataset provenance and partition identity for each registered dataset.
- [x] Record explicit train/validation/test partition history with source and fingerprint metadata.
- [x] Detect missing, extra, reordered, and incompatible columns.
- [x] Add configurable behavior for unknown categorical values and missing columns.
- [x] Record train/validation/test partition identities.

Acceptance criteria:

- Registering identical data produces the same fingerprint.
- A changed column, dtype, or value changes the fingerprint.
- Prediction rejects incompatible schemas with actionable errors.
- Schema and fingerprint are available through mltrace metadata.

Validation:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python examples/example03_mltrace.py
```

Dependencies: none.

## Phase 2: Explicit Evaluation Protocols

Status: complete

Goal: prevent optimistic evaluation and support reliable model comparison.

Tasks:

- [x] Add explicit train/validation/test dataset handling.
- [x] Add nested cross-validation for model selection and final performance estimation.
- [x] Support `KFold`, `StratifiedKFold`, and `GroupKFold` split configuration with safe fallback for small or underpopulated folds.
- [x] Preserve groups through every evaluation path when groups are available.
- [x] Add repeated CV and confidence intervals where practical.
- [x] Store fold-level predictions and metrics.
- [x] Add a final untouched validation evaluation method to `mltrack`.
- [x] Document which metrics are fold-averaged and which are pooled.

Metric semantics:

- Fold-averaged metrics are the values stored in `outer_scores`, `inner_scores`, and each entry under `fold_metrics`; each value reflects a single held-out fold or inner validation split before aggregation.
- Pooled metrics are the final validation score and the repeat-level confidence interval; these are computed from the full untouched validation partition or the full set of repeated outer-fold means, respectively.
- `mean_outer_score` is the arithmetic mean of the outer-fold scores and represents the cross-validation estimate for selection and comparison, while `final_validation_score` is the final held-out estimate on the reserved validation split.

Acceptance criteria:

- Search decisions never use the final test set.
- Grouped and time-ordered data do not leak between folds.
- Metrics include fold-level values and aggregate statistics.
- A model comparison can be reproduced from stored split metadata.

Dependencies: Phase 1.

## Phase 3: Trial Scheduling and Resource Governance

Status: complete

Goal: make AutoML runs predictable and safe to execute.

Tasks:

- [x] Add per-trial wall-clock timeouts.
- [x] Add per-trial memory and CPU controls where the platform permits.
- [x] Add fit-count, generation-count, and global budget enforcement.
- [x] Add early stopping when no improvement is observed.
- [x] Add trial pruning based on intermediate or partial CV results.
- [x] Record cancellation reason, resource usage, and termination status.
- [x] Ensure interrupted runs leave recoverable checkpoints.
- [x] Add a run summary with budget consumption.

Acceptance criteria:

- A configured global timeout stops all new trials.
- A timed-out trial is marked separately from a failed trial.
- Interrupted searches resume without corrupting prior results.
- Resource and termination information appears in `cv_results_` and mltrace.

Dependencies: Phase 2.

## Phase 4: Search Space and Failure Management

Status: in progress

Goal: prevent invalid configurations before expensive fitting and make failures actionable.

Tasks:

- [ ] Add conditional and hierarchical parameters.
- [ ] Add forbidden parameter combinations.
- [ ] Add log-scaled numeric distributions.
- [ ] Validate estimator parameters before starting a search.
- [ ] Reject invalid pipeline structures before evaluation.
- [ ] Record exception type, message, traceback, parameters, and fold.
- [ ] Add failure-rate summaries by estimator and pipeline.
- [ ] Add safeguards for zero-feature transformations.
- [ ] Add explicit handling for parameter-free estimators.

Acceptance criteria:

- Invalid configurations are rejected or skipped before fitting when detectable.
- Every failed trial has a readable failure record.
- Zero-feature pipelines never reach estimators that require features.
- Search results distinguish invalid, failed, timed-out, and completed trials.

Dependencies: Phase 3.

## Phase 5: Model Bundles and Registry Lifecycle

Status: not started

Goal: make trained models portable, versioned, and promotable.

Tasks:

- [ ] Define a model bundle format containing model, preprocessing, schema, metadata, metrics, and dependencies.
- [ ] Add dataset fingerprint and code/config version to each bundle.
- [ ] Add model version identifiers.
- [ ] Add registry states: candidate, validated, staging, production, archived.
- [ ] Add promotion and rollback records.
- [ ] Add compatibility checks when loading bundles.
- [ ] Add model aliases such as `latest`, `staging`, and `production`.
- [ ] Replace temporary files with safe atomic writes.
- [ ] Add optional MLflow-compatible export.

Acceptance criteria:

- A bundle can be loaded in a fresh process and produce identical predictions.
- Loading fails clearly when schema or dependency requirements are incompatible.
- Promotion and rollback history is queryable.
- Multiple model versions can coexist for one task.

Dependencies: Phases 1 and 2.

## Phase 6: Batch and Online Inference

Status: not started

Goal: make trained models usable outside the training process.

Tasks:

- [ ] Add a stable batch prediction API.
- [ ] Add prediction input/output schema validation.
- [ ] Add optional CLI batch prediction command.
- [ ] Add a lightweight HTTP serving adapter.
- [ ] Add health, readiness, and model-version endpoints.
- [ ] Add request ID and prediction audit fields.
- [ ] Add configurable handling for malformed requests.
- [ ] Add inference latency and throughput measurements.

Acceptance criteria:

- A stored model bundle can score a supported tabular input file.
- Invalid input produces structured validation errors.
- Online responses include model version and request ID.
- Batch and online predictions use the same preprocessing path.

Dependencies: Phase 5.

## Phase 7: Data Quality, Drift, and Performance Monitoring

Status: not started

Goal: detect when production data or model behavior changes.

Tasks:

- [ ] Add input data-quality checks.
- [ ] Add feature distribution drift metrics.
- [ ] Add categorical-value drift detection.
- [ ] Add missingness and range drift detection.
- [ ] Add prediction distribution drift detection.
- [ ] Add delayed-label performance monitoring.
- [ ] Add latency, error-rate, and throughput tracking.
- [ ] Add configurable thresholds and alert records.
- [ ] Link monitoring reports to model and dataset versions.

Acceptance criteria:

- A synthetic distribution shift produces a detectable drift report.
- Missingness and schema changes are reported separately from statistical drift.
- Performance monitoring can be updated when labels arrive later.
- Monitoring records identify the deployed model version.

Dependencies: Phases 1, 5, and 6.

## Phase 8: Governance, Fairness, and Security

Status: not started

Goal: provide auditability and responsible-use controls.

Tasks:

- [ ] Add immutable audit events for training, evaluation, promotion, and rollback.
- [ ] Add user/run ownership metadata.
- [ ] Add sensitive-feature declarations.
- [ ] Add group fairness metrics and reports.
- [ ] Add subgroup performance reports.
- [ ] Add PII and sensitive-column warnings.
- [ ] Add secret-management guidance and prevent secrets in metadata.
- [ ] Add retention and deletion controls for stored datasets and predictions.

Acceptance criteria:

- Every production promotion has an audit record.
- Fairness reports can compare configured groups.
- Sensitive metadata is not written to logs unintentionally.
- Stored artifacts can be deleted by task, model, and dataset identity.

Dependencies: Phases 1, 5, and 7.

## Phase 9: Distributed Execution and CI/CD Integration

Status: not started

Goal: scale searches and automate quality gates.

Tasks:

- [ ] Define an execution backend interface.
- [ ] Add a local process backend with durable trial state.
- [ ] Add optional Dask or Ray integration.
- [ ] Add queue-based worker coordination.
- [ ] Add concurrent-safe result storage.
- [ ] Add CI checks for schema compatibility, metrics, and model size.
- [ ] Add scheduled and data-triggered retraining entry points.
- [ ] Add deployment approval gates and rollback automation.

Acceptance criteria:

- Workers can resume incomplete trials safely.
- Results are not duplicated or lost under concurrent execution.
- CI can reject a model below a quality threshold.
- A retraining run can produce and promote a new version without manual file handling.

Dependencies: Phases 3, 5, 7, and 8.

## Phase 10: Documentation and Release Hardening

Status: not started

Goal: make the operational features discoverable and maintainable.

Tasks:

- [ ] Add end-to-end MLOps documentation and diagrams.
- [ ] Add runnable examples for each lifecycle stage.
- [ ] Add migration notes for existing SQLite databases and checkpoints.
- [ ] Add API reference coverage for public classes and methods.
- [ ] Add performance benchmarks and regression thresholds.
- [ ] Add supported Python/dependency version policy.
- [ ] Add release checklist and compatibility matrix.
- [ ] Add changelog entries for each phase.

Acceptance criteria:

- A new user can run an offline-to-deployment tutorial.
- Every public MLOps feature has a runnable example or focused test.
- Upgrade and rollback procedures are documented.
- CI validates tests, docs, examples, and packaging.

Dependencies: all prior phases.

## Suggested Implementation Order

1. Phase 1: Data identity and schema contracts
2. Phase 2: Explicit evaluation protocols
3. Phase 3: Trial scheduling and resource governance
4. Phase 4: Search space and failure management
5. Phase 5: Model bundles and registry lifecycle
6. Phase 6: Batch and online inference
7. Phase 7: Monitoring and drift
8. Phase 8: Governance and security
9. Phase 9: Distributed execution and CI/CD
10. Phase 10: Documentation and release hardening

## Session Log

### 2026-09-19

- Created this roadmap as an untracked repository document.
- Recorded the existing reliability, tracking, search, and testing baseline.
- Completed Phase 1: dataset fingerprinting, schema validation, configurable categorical/missing-column policies, provenance metadata, and explicit split partition history.
- Implemented the first Phase 2 milestone: explicit train/validation/test dataset retrieval by partition via `get_data(partition=...)` and `dataset_splits()`.
- Completed the nested CV evaluation milestone: `mltrack.nested_cv_evaluation()` now uses the train split for inner model selection and outer-fold evaluation, and the validation split for the final held-out score.
- Added group-aware handling for `GroupKFold` and safe fallbacks for insufficient groups so grouped evaluation remains valid when a fold becomes too small.
- Current state is Phase 1 complete and Phase 2 materially complete for KFold/StratifiedKFold/GroupKFold evaluation paths, with repeated CV, metric reporting, and time-series split support still pending.
