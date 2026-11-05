# SKSurrogate API Simplification Plan

## Purpose and status

Review the public API for duplication and redundancy so users write the least
code for the same goals. **No features are removed.** The API stays aligned with
scikit-learn (interchangeable where possible). Old names remain as aliases that
emit `DeprecationWarning`. This is a living plan: update task status and the
progress log whenever work starts, changes direction, or completes. Sits beside
[`AI_LAYER_ROADMAP.md`](./AI_LAYER_ROADMAP.md).

**Status:** Review complete; no implementation started.

Legend: `[ ]` open, `[~]` in progress, `[x]` done.

## Findings (from the initial review)

1. **Search/tuning**
   - `SurrogateRandomCV` deviates from `RandomizedSearchCV`: `params` (vs
     `param_distributions`), `max_iter` (vs `n_iter`), capitalised `Continue`.
   - `max_evals`, `min_evals`, `max_iter` overlap; ~35 flat constructor args.
   - Budget args (`time_limit`, `max_evals`, `cpu_limit`, `memory_limit`,
     `random_state`, `n_jobs`, `cv`, `scoring`) repeat in `AML` and
     `SurrogateRandomCV`.
   - Three surrogate loops overlap: `SurrogateSearch`,
     `SurrogateRandomCV.fit`, `AML.add_surrogate`/`optimize_pipeline`.
   - `Real`/`Integer`/`Categorical`/`HDReal` and `BoxSample`/`SphereSample`/
     `CompactSample` could accept plain tuples/lists and string selectors.
2. **AML/ensembling**
   - `AML` is not a `BaseEstimator` but hand-writes `get_params`/`set_params`.
   - `fit` and `eoa_fit` are two training paths.
   - `stack_*` args mirror `StackingEstimator`; `StackingEstimator` hand-writes
     `fit_transform`/`set_params`.
   - `EOA` operator classes are all exposed at top level.
3. **Preprocessing/feature selection**
   - `DataPreprocess` (stateful, no fit/transform), `FoldLocalDataPreprocessor`,
     `DateTime2Num` overlap.
   - `SensAprx`, `CorrelationThreshold` redefine `fit_transform`; `SensAprx`
     has 11 constructor args.
4. **Tracking/bundles/monitoring**
   - `mltrack` (~2.9k lines) mixes storage, CV, plots, diagnostics; mixed
     naming (`getBest`, `allTasks`, ...); `np2df` is a class acting as a function.
   - CV helpers (`cv_to_spec`, `build_cv`, ...) exported at top level.
   - `fairness_report`, `subgroup_performance_report`, `loss_report` share
     grouping logic; `InferenceMonitor` wraps overlapping functions.
   - `save_bundle`/`load_bundle` duplicate `ModelBundle`; inference/API repeat
     bundle loading; `api/training.py` may repeat `AML`/`SurrogateRandomCV`.
5. **General:** inconsistent module naming (`DataProcess.py`,
   `NpyProximation.py`); ~60 top-level exports with no core/extras split.

## Plan

### Phase 1: Baseline
- [ ] Create venv with sklearn; run `tests/` and `examples/`; record results.
- [ ] Snapshot public signatures (script + stored JSON) as a regression guard.

### Phase 2: Inventory
- [ ] Table of every public name/argument, duplicates, and callers across
      `api/`, `benchmarks/`, `webui/`, `examples/`, `docs/`.

### Phase 3: sklearn alignment (low risk)
- [ ] `SurrogateRandomCV`: `param_distributions`, `n_iter`, `continue_`; keep
      old names as deprecated aliases.
- [ ] Make `AML` a `BaseEstimator`; remove manual param code.
- [ ] Remove redundant `fit_transform`/`set_params` where mixins suffice.
- [ ] Make `DataPreprocess` a fit/transform estimator (keep current behaviour).

### Phase 4: Unify options (medium risk)
- [ ] Shared budget/stopping options across `AML` and `SurrogateRandomCV`.
- [ ] Merge `eoa_fit` into `fit(strategy=...)`.
- [ ] Collapse `stack_*` into `stacking=`.
- [ ] String shortcuts for samplers and EOA operators; tuples/lists as spaces.

### Phase 5: Consolidate optimisation loops
- [ ] Shared surrogate-loop core used by `SurrogateSearch`,
      `SurrogateRandomCV`, `AML`.

### Phase 6: Lifecycle layer
- [ ] Split `mltrack` into modules behind the same facade; snake_case aliases.
- [ ] One grouped-metrics helper behind monitoring reports.
- [ ] `save_bundle`/`load_bundle` delegate to `ModelBundle`; share loading in
      inference/API.

### Phase 7: Namespace and docs
- [ ] Core vs submodule exports; normalise module names, re-export old paths.
- [ ] Update readme, Sphinx docs, examples, `CHANGELOG.rst`.

### Phase 8: Verification
- [ ] Full tests, signature snapshot (no name removed), examples, deprecation
      warnings emitted with unchanged behaviour.

## Progress log

- 2026-11-05: Initial static API review completed; plan created. Next: start
  Phases 1-3.
