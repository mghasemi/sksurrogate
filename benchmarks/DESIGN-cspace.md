# SKSurrogate Pipeline-Space Benchmark — Contract (C1)

Status: **v0.3 (pre-run, 2026-10-26)**. This document is the contract for the pipeline-space
track (C1 spec, C2 runner, C3 pilot, C4 full runs, C5 results page). It is separate from
`benchmarks/DESIGN.md`, which stays frozen at v1.9 and keeps governing the *fixed-pipeline*
suite; nothing here invalidates those results.

Changes after the pilot launch require a version bump here **and** a re-run of every arm
into a fresh canonical directory (never a patched table). The C-space, the protocols, and
the hypotheses below are frozen before any benchmark number exists.

---

## 0. Why this track exists

`DESIGN.md` §4 searches **one fixed pipeline** (StandardScaler → RandomForest(50)) over three
hyperparameters (224 configs). That answers "does surrogate guidance help tune an estimator?"
It does not answer the AutoML question users actually ask: **"does it help choose the
pipeline?"** — preprocessing transformers *and* estimator families.

Moving the same search machinery to a heterogeneous space changes the problem in three ways
that are properties of the current implementation, verified by reading `structsearch.py`:

1. **Categorical parameters are encoded ordinally.**
   `SurrogateRandomCV.fit` assigns each categorical the bounds `(0, len(items) - 1)` and
   decodes with `int(round(x))` (`structsearch.py:1108-1111`, `:1153-1157`). Estimator
   identity therefore becomes a **coordinate on a line**: `logreg → rf → et → hgb → svc → knn`
   is treated as a metric. A response surface fitted on that axis has no meaning across
   families, and the trust-region local step moves between model families as if that were a
   small perturbation.
2. **Conditional parameters still occupy a live coordinate.**
   `_apply_conditional_parameters` (`:995-1000`) only removes an inactive knob from the dict
   passed to `set_params`; `x0`/`bounds` are built for every parameter in `params_list`
   (`:1094-1131`) and the sampler keeps exploring the coordinate. For a candidate whose score
   cannot depend on that dimension, the surrogate sees pure label noise in that direction.
   SRS also pays evaluations to propose combinations that then get rejected.
3. **The default engagement floor scales combinatorially.**
   `SurrogateSearch.__init__` sets `MinEvals >= C(p + 3, p)` when it installs its own default
   `GaussianProcessRegressor` (`:278-287`). In the 3-parameter suite that is 20; with this
   track's 11 flat coordinates it is `C(14, 11) = 364`, so **the surrogate would never fire**
   at any realistic budget — silently degrading "SKSurrogate" to random search in the results
   table. Every SRS arm in this track therefore passes an explicit `regressor=` (which skips
   the clamp) and an explicit `min_evals=20`.

A fourth, mechanical obstacle: `'passthrough'`/`None` steps are rejected by
`_zero_feature_reason` (`:1002-1026`), which probes each intermediate step with
`.fit_transform`/`.fit` and records a `str` sentinel as a preprocessing failure. The C-space
therefore expresses "no transform" as `FunctionTransformer()` (identity).

**Principle.** Per `DESIGN.md` v1.4, this benchmark does not patch library code. The three
defects above are *measured*, recorded, and reported upstream (C5); the H2 arm works around
defect 1 through the public `regressor=` kwarg.

---

## 1. Environment (recorded 2026-10-26)

| Component | Version | Note |
|---|---|---|
| Python | 3.13.13 | `sksurrogate/.venv` |
| scikit-learn | 1.9.1 | |
| SKSurrogate | 0.2.1 | editable install of this repo |
| NumPy / pandas / SciPy | 2.5.3 / 2.3.3 / 1.18.1 | |
| Optuna | 4.9.0 | provisioned for this track (`pip install "optuna>=4,<5"`) |
| scikit-posthocs | 0.17.1 | Nemenyi (descriptive only, see §6) |
| FLAML | 2.7.0 | Protocol C |
| AutoGluon-Tabular | 1.6.3 | Protocol C |
| LightGBM / XGBoost | 4.7.0 / 2.1.4 | reached only inside FLAML/AutoGluon |
| pytest | >=8 | provisioned for this track |

SMAC3 is **out of scope for this round**. The space spec is left compatible (one flat dict in,
one config out) so it can be added later without re-running other arms.

---

## 2. The C-space (frozen)

Defined once in `benchmarks/cspace.py`; every arm consumes that module.

Every pipeline is `impute -> prep -> reduce -> clf`. The leading
`SimpleImputer(strategy="median", add_indicator=True)` is **fixed, not a search axis**.

| Axis | Options |
|---|---|
| `prep` | `identity` (`FunctionTransformer`), `standard` (`StandardScaler`), `robust` (`RobustScaler`), `quantile` (`QuantileTransformer(output_distribution="normal")`) — 4 |
| `reduce` | `identity`, `pca4`, `pca8`, `pca16`, `kbest16` (`SelectKBest(f_classif, k=16)`) — 5 |
| `clf` | `logreg` (C ∈ 5), `rf` (max_depth ∈ {4,8,16,None} × min_samples_leaf ∈ {1,5,20}), `et` (same), `hgb` (learning_rate ∈ 3 × max_leaf_nodes ∈ 3), `svc` (C ∈ 5 × gamma ∈ {scale,0.01,0.1}), `knn` (n_neighbors ∈ 4 × weights ∈ 2) — 61 |

**Cardinality:** $4 \times 5 \times 61 = \mathbf{1220}$ distinct pipelines. Asserted by
`tests/test_cspace_space.py::test_cardinality_is_frozen`.

Notes.
- **Missing values are imputed uniformly.** arrhythmia carries 408 missing cells in 5 columns
  (one column missing in 376 of 452 rows). Without a fixed imputer, LogReg/SVC/KNN — and every
  pipeline whose reducer needs complete data (`PCA`, `SelectKBest`) — fail there, so the arms'
  *failure handling* rather than their search quality would decide the comparison. One fixed step
  keeps the 1,220-pipeline count and the equality of arms; `add_indicator=True` retains the
  missingness pattern, and fitting inside the pipeline keeps it per-fold (no leakage).
- `max_depth = 0` encodes `None` (the flat space needs a numeric coordinate).
- **Shared knobs declare union value sets.** `C` is shared by `logreg` and `svc`, so both
  branches list the same five values `{0.01, 0.1, 1.0, 10.0, 100.0}`. The flat space has one
  `C` coordinate; if a branch excluded a value, a legitimate sampler draw would become an
  invalid candidate. Validity depends on *which* knob is active, never on which value it
  holds — anything else would be an artifact of the encoding rather than the effect under test.
- PCA stops at 16 components: the smallest dataset has 20 features, so a larger count would be a
  guaranteed fit failure rather than a search decision.
- `kbest16` requires `y`; `Pipeline` forwards `y` to intermediate `fit_transform`, so this is
  valid — but it means the reducer's selection is refit per CV fold. That is intended: per-fold
  selection is the leakage-free behaviour.
- The flat space has **11 coordinates**: `prep`, `reduce`, `clf` + 8 branch knobs (`C`,
  `max_depth`, `min_samples_leaf`, `learning_rate`, `max_leaf_nodes`, `gamma`, `n_neighbors`,
  `weights`). Its raw product is strictly larger than 1220 and mostly invalid; the `invalid_rate`
  metric monitors whether an arm pays for that product rather than for the space.

### 2.1 Branch gating

`cspace.srs_conditional()` maps each branch knob to a predicate on `clf`; `cspace.optuna_suggest`
branches natively. The two must generate the same logical space:

| Check | Where |
|---|---|
| All 1140 enumerated configs are valid and carry only their branch knobs | `test_every_config_is_valid_and_carries_only_its_branch_knobs` |
| 1,000 uniform draws are valid and branch-consistent | `test_srs_space_and_conditionals_generate_only_valid_configs` |
| SRS masked parameters resolve to a valid config | `test_srs_masked_params_build_a_valid_config` |

---

## 3. Protocols

Common to all protocols:
- one stratified 80/20 train/test split per `(dataset, seed)`, identical for every arm; the test
  set is touched once, after the search, to score the arm's chosen configuration;
- one **precomputed** `StratifiedKFold(5, shuffle=True, random_state=seed)` split list per cell,
  passed as an object so fold membership is identical across arms;
- `n_jobs=1` everywhere and `OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 1`
  (see `benchmarks/pipelines.py::pin_threads`), so wall-clock is not distorted by BLAS
  oversubscription (SVC benefits far more than RF);
- per-candidate records `(k, cv_score, wall_s, status)`, with invalid/pruned candidates keeping
  their position and `cv_score = None` (`DESIGN.md` §5 precedent);
- datasets: the six pinned in `benchmarks/datasets.json` (ionosphere, credit-g, arrhythmia,
  spambase, optdigits, galaxy3).

### Protocol A — equal wall-clock (**primary**)

$T = 120$ s per (method, dataset, seed).

Metrics: final held-out accuracy (primary), best CV, `n_evals` completed, normalized anytime AUC
of best-so-far CV over cumulative wall time, `invalid_rate`.

Rationale: in a C-space one evaluation ranges from ~5 ms (KNN on ionosphere) to ~2 s (SVC on
spambase, 5 folds). Equal-evaluations therefore rewards whichever families an arm happens to
draw and turns the comparison into a lottery over model cost. Wall-clock is the currency a
practitioner pays. The equal-evaluation protocol is kept as the secondary axis precisely because
it isolates search quality from candidate cost.

### Protocol B — equal evaluations (**secondary**)

$B = 40$ candidate evaluations.

Metrics mirror `DESIGN.md` §6: median evaluations-to-gap ($g = 0.01$ absolute, `g = 0.005`
reported alongside), final CV, seed-level win/loss counts.

Rationale for 40: the surrogate engages at $k \ge 20$ in this configuration, so the run spends
about 20 guided evaluations — the same guidance window in which the fixed-pipeline suite found
the GP advantage at $B = 60$ (`DESIGN.md` §12.4). A smaller budget would compare a surrogate
that barely fires; a much larger one would drift toward the flat-landscape regime documented in
`DESIGN.md` §12.3.

### Protocol C — system level (separate table)

$T = 120$ s; FLAML (`time_budget=120`, `eval_method="cv"`, `metric="accuracy"`), AutoGluon
(`presets="medium_quality"`, `time_limit=120`, `num_cpus=1`), and `srs-cspace`.

These systems choose their own pipelines and their own number of internal fits, so only final
held-out accuracy is comparable; `n_evals = None` is recorded honestly. **Protocol C results are
never pooled with A/B** and no cross-protocol significance test is run.

---

## 4. Arms

| Arm | Role | Protocols | Configuration |
|---|---|---|---|
| `random-cspace` | search control | A, B | `cspace.sample_uniform`, exactly $B$ draws |
| `optuna-tpe-cspace` | HPO baseline | A, B | `TPESampler(seed=s)`, `cspace.optuna_suggest` |
| `optuna-rand-cspace` | sampler control | A, B | `RandomSampler(seed=s)` — separates "TPE is smart" from "Optuna is a loop" |
| `srs-cspace` | **primary method** | A, B | `regressor=GaussianProcessRegressor()`, `min_evals=20`, `conditional=cspace.srs_conditional()` |
| `srs-onehot-cspace` | H2 encoding arm | A, B | same, `regressor=MixedCategoricalSurface(...)` |
| `srs-nosurr-cspace` | ablation | A, B | same, `min_evals=B` (surface never engages) |
| `flaml` | system | C | FLAML `AutoML` |
| `autogluon` | system | C | `TabularPredictor` |

SRS arms all pass `max_iter = 4 * B` / `4 * budget-as-ceiling` and `max_evals = B`, because
`max_iter` counts loop iterations (inclusive off-by-one, and `x0` is evaluated in `__init__`),
while failed local optimizations of the surface can consume iterations without evaluating
anything (`DESIGN.md` v1.1/v1.7 errata). `warm_start=False`, `Continue=False`, and the
`<task_name>.pkl` checkpoint is removed after each run — otherwise SRS restores state across
independent runs. The global `random` state is pinned per seed (same salt for all SRS arms) so
`x0` is bit-reproducible; SRS draws it from the module-level RNG, ignoring `random_state`.

---

## 5. Metrics

| Metric | Definition | Axis |
|---|---|---|
| `final_test_accuracy` | accuracy of the run's best-CV config, refit on the full train split, scored once on the held-out test set | quality |
| `final_cv_best` | best mean 5-fold CV score seen | quality |
| `evals_to_gap` | smallest $k$ with running-best within $g = 0.01$ of the run's own final best | speed |
| `auc_norm` | trapezoid area of best-so-far CV over cumulative wall time, divided by `(T × cell best CV)` | speed/quality under A |
| `invalid_rate` | invalid + forbidden + failed candidates / `n_evals` | budget efficiency |
| `engaged_at` | first $k$ at which the surface was fitted (`None` if never) | integrity |
| `family_mix` | histogram of `clf` among evaluated candidates | confound check |
| `total_wall_s` | measured wall clock, with machine description in the manifest | cost |

`family_mix` exists so a wall-clock win can be interrogated: if an arm wins by repeatedly
sampling KNN, the table must say so.

---

## 6. Statistics (fixed before the run)

- **Unit:** the cell $(dataset, seed)$; 6 datasets × 7 seeds = 42 paired observations per method
  pair.
- **Primary per-dataset test:** two-sided exact sign test on paired differences with ties
  excluded, reporting the tie count and the decisive $n$ (`DESIGN.md` §12.3 precedent);
  Wilcoxon signed-rank as a sensitivity check.
- **Effect size:** mean paired difference with a bootstrap 95% CI (10,000 resamples, fixed
  seed) and Cliff's delta.
- **Multiplicity:** Holm–Bonferroni within the single family "each arm vs `srs-cspace`", per
  dataset.
- **Cross-dataset:** sign test over the 6 per-dataset mean differences. **Limitation stated in
  advance:** 6 datasets cannot support a Friedman/Nemenyi critical-difference claim with 7 arms;
  any Nemenyi output is descriptive and labelled as such.
- **Never:** pool Protocol C with A/B; select the reported winner on test accuracy; report a
  p-value without the decisive $n$ and tie count.

---

## 7. Hypotheses (pre-registered; falsifiers fixed)

| ID | Question | Hypothesis | Falsifier |
|---|---|---|---|
| H1 | Same C-space and budget: does surrogate guidance beat TPE? | With ordinal categorical encoding, `srs-cspace` is **not** better than `optuna-tpe-cspace`: mean best-CV difference ≤ 0 in ≥ 4 of 6 datasets. | `srs-cspace` strictly better in ≥ 4/6 datasets. |
| H2 | Is the ordinal encoding the cause? | `srs-onehot-cspace` improves mean best-CV by ≥ 0.3 pt over `srs-cspace` in ≥ 4/6 datasets. | Improvement < 0.1 pt in ≥ 4/6 datasets — the encoding is not the bottleneck. |
| H3 | System level: what does a single-pipeline search lose to portfolio AutoML? | At equal $T$, FLAML and AutoGluon each beat `srs-cspace` on mean held-out accuracy in ≥ 4/6 datasets; the **gap in points** is the reported quantity. | `srs-cspace` ≥ both systems in ≥ 4/6 datasets. |
| H4 | (descriptive) Does the ranking survive cost heterogeneity? | No directional claim; report any ordering change between Protocol A and B and the `family_mix` behind it. | — |

Outcomes are reported as *confirmed / refuted / inconclusive* against the falsifier column. A
refuted hypothesis is a result, not a failure.

---

## 8. Integrity gates

| Gate | Check | Failure action |
|---|---|---|
| G0 provisioning | `optuna`, `flaml`, `autogluon`, `scikit-posthocs`, `pytest` import; versions in the manifest | abort before any run |
| G1 space parity | cardinality = 1220; every enumerated config valid; 1,000 uniform draws valid; masked SRS params resolve to valid configs | abort |
| G2 count parity | Protocol B: `n_evals == B` for every arm; Optuna trial count == B; SRS `engaged_at` not `None` | record `status="error"`, never silently proceed |
| G3 pre-launch fit gate | every config fits and predicts on every dataset (`python -m benchmarks.gate_cspace_fit`) | space is broken — fix the spec, bump this document |
| G4 engagement | every SRS cell records `extra["surrogate"]` and `extra["engaged_at"]` | re-run the cell |
| G5 provenance | `run_manifest.json`: git SHA + dirty flag + package versions + CPU model | dirt is recorded, not hidden |
| G6 determinism | one fixed cell re-run must be bit-identical in `(k, cv_score)` | abort; do not publish |
| G7 thread pinning | `OMP_NUM_THREADS=1` etc. asserted in the manifest | re-run |

### Invalidation policy

A result directory is canonical only if G0–G7 pass. A defective launch moves to
`benchmarks/results/_superseded/<reason>/` and the reason is recorded here as an errata entry.
Nothing is deleted.

---

## 9. Compute budget

| Stage | Matrix | Estimate |
|---|---|---|
| Smoke (CI) | ionosphere, 1 seed, $B=8$, 4 arms, RF(10) | < 3 min |
| Pilot (C3) | ionosphere + galaxy3 + credit-g, 2 seeds, $B=20$ / $T=30$ s, 6 arms | ~40 min |
| Protocol B (C4) | 6 datasets × 7 seeds × 6 search arms × 40 evals × 5 folds ≈ 50,000 CV fits | 10–16 h |
| Protocol A (C4) | 6 × 7 × 6 arms × 120 s ≈ 8.5 h wall | ~8.5 h |
| Protocol C (C4) | 6 × 7 × 3 systems × 120 s ≈ 2.5 h (+ AutoGluon overhead) | 3–5 h |
| Optional oracle | 1220 configs × 5 folds on 3 datasets × 1 seed, for a true regret normalizer | 4–6 h |

Full matrix ≈ 2–3 overnight runs. The pilot re-estimates these numbers before the full launch;
a >2× deviation triggers a re-scope (`T = 90` s, or Protocol B on 4 datasets) and a version bump.

---

## 10. Results placement

- Raw per-run JSON + `results.jsonl` + manifest: `benchmarks/results/<UTC stamp>/`, archived to
  Zenodo with the other benchmark data (DOI 10.5281/zenodo.23629024 family).
- Aggregates, figures, and markdown tables: version-controlled, published at
  `docs/benchmarks-cspace.md` and linked from `docs/benchmarks.md`.
- Upstream defect reports (the three implementation facts in §0) are filed with minimal
  reproducers and referenced from `CHANGELOG.rst` under `Unreleased`.

---

## 11. Revision log

**v0.1 → v0.2 (2026-10-26, pre-run — no benchmark number exists yet).** Three corrections made
while implementing `benchmarks/cspace.py` and `benchmarks/pipelines.py`, all before any run:

1. `C` was declared with *different* value sets in the `logreg` (4 values) and `svc` (4 values,
   different endpoints) branches while the flat space exposes one `C` coordinate. A legitimate
   sampler draw would then have been an invalid candidate — an artifact of the encoding, not the
   defect under test. Both branches now declare the same five values; cardinality 1140 → **1220**.
2. The earlier H3 ("SRS wastes budget on invalid candidates") is **not falsifiable as stated**:
   `_apply_conditional_parameters` runs before `set_params`, so SRS does produce valid
   candidates whenever the value sets are unions. `invalid_rate` is retained as a *monitored
   metric* (expectation ≈ 0 for every arm; anything else must be explained before publishing),
   and the hypotheses are renumbered H1–H4.
3. `MixedCategoricalSurface` was first written to round coordinates before the circular
   embedding, which made its surface piecewise constant like one-hot and would have confounded
   H2 with a mechanical optimiser failure. The embedding is now continuous in the raw
   coordinate, and categorical columns are *replaced* by their embedding rather than appended
   next to the raw ordinal column (which had leaked the ordinal signal straight back in).

**v0.2 → v0.3 (2026-10-26, pre-run).** The G3 gate found that arrhythmia contains 408 missing
cells in 5 columns, so LogReg, SVC, KNN and every `PCA`/`SelectKBest` pipeline failed there while
RF/ET/HGB survived on the identity branches. That is a confound of the first order: arms would
differ in *robustness to NaN*, not in search quality. A single fixed leading
`SimpleImputer(strategy="median", add_indicator=True)` was added to every pipeline. It is not a
search axis, the cardinality stays 1,220, and every arm sees it identically.

The pre-v0.3 pilot directory `benchmarks/results/cspace-B-20261026T222230Z` (36 cells, B=20) ran
before this change and **is not citable**; it is retained only as timing evidence. It also
confirmed the engagement arithmetic: with `min_evals = 20` and `max_evals = 20` the surface never
fires, and at `B = 40` it fires at exactly `k = 20` (20 guided evaluations).

## 12. Errata

*(empty — no run has been launched)*
