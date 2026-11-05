# SKSurrogate API Simplification Plan

## Purpose and status

Review the public API for duplication and redundancy so users write the least
code for the same goals. **No features are removed.** The API stays aligned with
scikit-learn (interchangeable where possible). Old names remain as aliases that
emit `DeprecationWarning`. This is a living plan: update task status and the
progress log whenever work starts, changes direction, or completes. Sits beside
[`AI_LAYER_ROADMAP.md`](./AI_LAYER_ROADMAP.md).

**Status:** All phases complete or consciously deferred (see Phase 6/7 notes); Phase 6 partly done, `mltrack` split deferred; Phase 7 docs done, namespace regrouping deferred; Phase 8 verification ongoing.

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
- [x] Use `.venv` (sklearn 1.9.1); baseline `tests/`: 230 passed. (`examples/` not yet run.)
- [x] Signature snapshot: `tests/api_snapshot.py --write` / verify (`tests/api_snapshot.json`); fails if any name, constructor arg or public member disappears.

### Phase 2: Inventory
- [ ] Table of every public name/argument, duplicates, and callers across
      `api/`, `benchmarks/`, `webui/`, `examples/`, `docs/`.

Decision: `DataPreprocess` stays a one-shot whole-frame encoder (fitting and
imputing on the full frame leaks across CV folds). `FoldLocalDataPreprocessor`
is the sklearn-style fit/transform equivalent; the docstring now says so.
Making `DataPreprocess` itself a transformer would duplicate it.

### Phase 3: sklearn alignment (low risk)
- [x] `SurrogateRandomCV`: `param_distributions`, `n_iter` added (`continue_`
      already existed); old names kept as plain aliases with a conflict check.
      No `DeprecationWarning` is emitted, because `clone` passes both spellings.
- [x] `SurrogateRandomCV` gains `best_params_`; `clone` works for it and `AML`.
- [x] `AML` is a `BaseEstimator`. Its custom `get_params`/`set_params` are kept
      on purpose: they encode preset-dependent defaults.
- [x] Removed redundant `fit_transform` in `SensAprx`/`CorrelationThreshold`.
      `StackingEstimator.fit_transform` kept (leakage-safe out-of-fold); its
      `set_params` override removed.
- [x] Bug fix: default `AML()` config referenced `sksurrogate.SensAprx`.
- [x] Decided against converting `DataPreprocess` (see decision above).

### Phase 4: Unify options (medium risk)
- [x] Budget options (`time_limit`, `max_evals`, `cpu_limit`, `memory_limit`, `random_state`, `n_jobs`) already share names/semantics in `AML` and `SurrogateRandomCV`; no new object needed. `max_iter`/`n_iter` vs `max_evals` kept distinct (iterations vs evaluations).
- [x] `AML.fit(X, y, strategy="surrogate"|"eoa", **eoa_kwargs)`; `eoa_fit` kept.
- [x] `AML(stacking={...})` overrides the `stack_*` arguments, which are kept.
- [x] Tuples/lists as spaces for `SurrogateRandomCV` (`as_space`) and `sampling="box"|"sphere"|"compact"` (`resolve_sampling`) for `SurrogateSearch`, `SurrogateRandomCV`, `AML.add_surrogate`. EOA operators already default sensibly; no shortcut needed.
- [x] Shorthand spaces in `AML` configs work (they go through `SurrogateRandomCV`); `SurrogateSearch` bounds are already plain tuples.

### Phase 5: Consolidate optimisation loops
- [x] Verified there is no duplicated loop: `AML` -> `SurrogateRandomCV` ->
      `SurrogateSearch` are already layered delegations (the `RandomizedSearchCV`
      calls in `AML` only tune the surrogate regressors). No code change.

### Phase 6: Lifecycle layer
- [x] snake_case aliases (`get_best`, `all_tasks`, ...) already exist.
- [ ] Deferred: splitting `mltrack` (~2.9k lines) into modules; cosmetic, high regression risk.
- [x] Shared `_grouped_inputs` validation for the grouped reports (metric logic untouched).
- [x] `ModelBundle.save` / `ModelBundle.load` added (delegate to the functions);
      `inference._as_bundle` replaces two copies of the load-and-check code.
- [x] `api/` already uses the library loaders; the three identical `ModelBundle(...)` constructions in `api/training.py` now share `_new_bundle`.

### Phase 7: Namespace and docs
- [ ] Deferred: core vs extras export split (top-level exports kept so nothing breaks). Module renaming rejected: `dataprocess.py` would collide with `DataProcess.py` on case-insensitive filesystems.
- [x] Docs (`surrogate.rst`, `opd.rst`) and `CHANGELOG.rst` updated. readme quick start added; examples unchanged (they already use the unaffected APIs).

### Phase 8: Verification
- [x] (2026-11-05: 241 tests, snapshot OK, examples 04-06 run) Full tests, signature snapshot (no name removed), examples, deprecation
      warnings emitted with unchanged behaviour.

## Open observations

- [x] Fixed: `SurrogateRandomCV.best_score_` now follows the scikit-learn sign
  (higher is better; `-inf` if no feasible trial). `best_estimator_score`
  remains the minimized loss that `AML` ranks by. `cv_results_["mean_test_score"]`
  was inconsistent with it; now fixed too (see below).
- [x] Fixed: `subgroup_performance_report(metric="precision")` divided by
  actual positives (recall); now TP / predicted positives.
- [x] `cv_results_["mean_test_score"]` is now the score (higher is better, NaN
  for failed trials); raw loss kept as `cv_results_["loss"]`; added
  `rank_test_score` and `best_index_`.
- [x] Fixed (pre-existing at HEAD): `AML.optimize_pipeline` returned only the
  last surrogate stage, so when a later stage had no budget left the pipeline
  scored `inf` and was unfitted. It now keeps the best stage.
- [x] Fixed: `StackingEstimator.set_params` silently ignored `cv`, `n_jobs` and
  unknown keys; removed so the scikit-learn implementation applies.
- [x] Fixed (pre-existing): `AML` always passed `continue_=True`, so a second
  `fit` in the same directory resumed the previous run's checkpoint (named after
  the pipeline) and could return an `inf`, unfitted result on different data.
  Added `AML(resume=False)`; resuming is now opt-in.

## Progress log

- 2026-11-05: Initial static API review completed; plan created. Next: start
  Phases 1-3.
- 2026-11-05: Phase 1 done (230 baseline tests, snapshot guard). Phase 3 mostly
  done and part of Phase 4 (shorthand spaces); 237 tests pass, snapshot OK. Added
  `tests/test_api_simplification.py`, changelog entry.
- 2026-11-05: Phase 4 done except shorthand in AML configs: `fit(strategy=)`,
  `stacking=`, sampler strings. 240 tests pass, snapshot OK. Next: Phase 5
  (shared surrogate-loop core) after inspecting the three loops.
- 2026-11-05: Phase 5 verified (no change needed). Phase 6/7 partial: bundle
  save/load methods, shared inference loader, grouped-input helper, docs. 241
  tests pass. Deferred: mltrack split, export regrouping.
- 2026-11-05: Fixed the two pre-existing bugs above (both real); added tests.
  243 tests pass.
- 2026-11-05: `cv_results_` made sklearn-style; fixed AML best-stage loss and
  StackingEstimator.set_params; shorthand verified in AML configs. 246 tests
  pass.
- 2026-11-05: Fixed stale-checkpoint resume in AML (`resume=False` default);
  reproduced first (second fit scored `inf`), regression test added. 247 tests
  pass. Remaining: `DataPreprocess` fit/transform, readme/examples, `api/` review.
- 2026-11-05: `DataPreprocess` decision documented; `api/training.py` bundle
  construction deduplicated; readme quick start added. 247 tests pass, snapshot
  OK. Test-suite time rose from ~40s to ~80s because AML fits now start fresh
  instead of silently resuming leftover checkpoints.
