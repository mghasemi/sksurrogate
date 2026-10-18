# SKSurrogate Benchmark Suite — Design (B1)

Status: **v1.4** (2026-10-17; pre-full-run audit errata). Datasets and baselines are pinned below; this doc is the
contract for B2 (runner), B3 (ablation), B4 (publication). Changes after B2 starts require a
version bump here.

v1.1 errata (found while building the runner, verified against `structsearch.py`):
- Budget enforcement for `SurrogateRandomCV` is via `max_evals=B`, not `max_iter=B` (§2) — SRS's
  iteration counter has an inclusive off-by-one and its initial point is evaluated in `__init__`.

v1.2 errata (B3 implementation, verified against live runs 2026-10-17):
- The §9 ablation arms are runner methods: Arm B = `sksurrogate` (default engagement),
  Arm A = `srs-no-surrogate` (`min_evals=B`, surrogate never engages; same SRS machinery).
- Risk #2 parity check is enforced in the runner: an Optuna run whose trial count differs from
  the budget returns a recorded `error` instead of silently proceeding.
- The plot.py convergence fraction now matches §6 (95%); v1.1 code had 0.98 — superseded.

v1.3 errata (pre-full-run audit, verified by re-reading `structsearch.py` and the live smoke records):
- **Runner bug (§9 ablation contract was silently broken):** `_run_srs` hardcoded Arm B's default
  engagement formula into the `SurrogateRandomCV` call, discarding the per-arm `min_evals`
  override — so pre-fix runs gave *both* arms Arm B's requested threshold; at suite budget
  B = 30 that would have let even Arm A engage (at the clamped threshold of 20, see v1.4),
  violating §9's "never engages" contract. Fixed in `methods.py`; all SRS-arm results from
  result directories dated before this erratum are void and were re-run. (The bit-identical
  Arm A/B smoke curves at budget 8 have a *different* cause — the MinEvals clamping, v1.4.)
- **SRS initial point reproducibility:** `SurrogateRandomCV.fit` draws the starting point
  `x0` from CPython's *module-level* `random` state (ignoring the estimator's
  `random_state`). The runner now pins that global RNG deterministically per seed — same salt
  for both arms, so they share one identical starting point per seed — and restores the
  caller's state afterwards. Every SRS run is bit-reproducible across processes; noted as an
  upstream quirk (no library behavior changed).
- `skopt` failures at fit time (archived dependency vs modern sklearn) are now recorded as
  `skipped_archived`, not hard errors, so a broken baseline cannot fail the suite's exit code.

v1.4 errata (pre-full-run audit 2026-10-17; verified empirically by constructing live
`SurrogateRandomCV` objects exactly as `_run_srs` does):
- **MinEvals constructor clamping (§9 engagement is coarser than the requested `min_evals`).**
  `SurrogateSearch.__init__` (structsearch.py) raises the effective threshold to
  `C(n_params + deg, n_params)` whenever the default cubic Hilbert regressor is in use: with our
  3-parameter space that is **20**, so any requested `min_evals < 20` is silently raised to 20.
  Consequences at suite budget B = 30: Arm B engages only once 20 points have been evaluated —
  i.e., records k ≥ 20 get surrogate-guided candidates (10 of the 30 evaluations); for any
  budget < 21 neither arm ever engages and both SRS-arm curves are bit-identical *by construction*
  (this, not the v1.3 runner bug, is why the budget-8 smoke arms matched exactly). Arm A's
  `min_evals = B` is never clamped down (it is ≥ 20 for every suite budget), so its "never
  engages" contract holds at all budgets in this suite. No library code changed; engagement
  behavior is reported as-is, and any table comparing the two SRS arms must state that they
  coincide by design when B < 21.

## 1. Goal

The README claims SKSurrogate performance "validated against benchmarks" with no findable
evidence in-repo. This suite produces that evidence:

- Head-to-head comparison of SKSurrogate's surrogate-guided search against standard baselines,
  under **equal budgets measured two ways**: number of model evaluations (primary) and wall-clock
  time (secondary).
- A decisive ablation isolating the value of the *surrogate* component itself (B3): same outer
  loop with vs. without surrogate guidance.

Success criterion for positioning: at equal evaluation budget, SurrogateRandomCV reaches target
performance in fewer evaluations than RandomizedSearchCV on a majority of datasets, and within a
documented factor of Optuna-TPE. If the numbers do not show an advantage, Phase C messaging is
adjusted accordingly (B5 gate) — no spin.

## 2. Methods under test

All methods optimize the **same estimator over the same parameter space** with the **same budget**.

| Method | Entry point | Notes |
|---|---|---|
| `SKSurrogate` | `SKSurrogate.structsearch.SurrogateRandomCV` (sklearn-native) | Default surrogate configuration. Primary method. |
| `random` | Uniform random sampling over the shared discrete space — algorithmically identical to `sklearn.model_selection.RandomizedSearchCV(param_grid=SPACE, n_iter=budget)` for categorical list spaces; implemented directly in `methods.py` so per-evaluation wall time is logged with SKSurrogate's semantics. |
| `optuna-tpe` | `optuna.create_study(direction="maximize", sampler=TPESampler(seed=s))` | One objective call = one model evaluation. Counted exactly. |
| `skopt-bayes` (conditional) | `skopt.BayesSearchCV` via `RandomForestClassifier` wrapper | **Expected to fail import** on modern scikit-learn: the repo is archived (GitHub `archived=true`, last push 2024-02, verified 2026-10-17). Protocol: attempt import; on failure record `"skopt": {"status": "unavailable", "reason": "<ImportError>"}` in the results JSON and note it in docs as positioning evidence. If it imports cleanly, include normally. |

Budget **B = 30 model evaluations per (method, dataset, seed)** — a hard cap on candidate-model
evaluations, verified post-run as `n_evals == budget` for every arm:

- `SurrogateRandomCV`: `max_evals=B`. Not `max_iter=B`: SRS loops while
  `iteration <= max_iter` (inclusive off-by-one) and evaluates its initial point `x0` inside
  `__init__`, so a naive `max_iter=B` spends B+2 evaluations. `max_evals` is checked before each
  iteration and caps *total* objective calls exactly. Arm B requests
  `min_evals = min(max(5, B // 5), B)`; note the constructor clamping documented in the v1.4
  errata — with the default cubic regressor the effective engagement threshold is
  `max(requested, C(n_params + 3, n_params))` = **20** for this 3-parameter space (an earlier draft
  figure of `C(n+3, n) = 56` does not hold here), so at B = 30 the surrogate engages from record
  k ≥ 20. Arm A passes `min_evals = B`, which is never clamped down, so it never engages.
- `random`: exactly B uniform draws (with replacement) from the shared lists.
- Optuna: `study.optimize(..., n_trials=B)` — exactly B trials.
- skopt: `n_iter=B`, if it imports at all (it does not — see above).

## 3. Datasets (fixed)

Five OpenML datasets + the repo-bundled Galaxy3. All sizes verified live against the OpenML XML
API (`api/v1/xml/data/list`) on **2026-10-17**; `did` values are pinned and recorded in
`benchmarks/datasets.json` at B2 (download is by `did`, never by name, to avoid version drift).

| # | Dataset | OpenML did | n rows | d feats | Classes | Role |
|---|---|---:|---:|---:|---|---|
| 1 | ionosphere | 59 | 351 | 34 + 1 nominal | 2 | Tiny; fast pipeline validation (CC18 member) |
| 2 | credit-g | 31 | 1000 | 20 + 1 | 2 | Small binary (CC18 member) |
| 3 | arrhythmia | 5 | 452 | 279 + 1 | **13, imbalanced** | Hard: high-dim, multiclass, imbalanced (CC18 canonical hard task) |
| 4 | spambase | 44 | 4601 | 57 + 1 | 2 | Medium binary (CC18 member) |
| 5 | optdigits | 28 | 5620 | 64 + 1 | 10 | Medium multiclass (OpenML classic; added for 10-class coverage) |
| 6 | Galaxy3 | bundled: `data/Galaxy3-[GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz].tabular` | 1600 | 20 + target | 2 | Synthetic epistasis; already in repo (no download) |

Notes:
- `pageblocks`, `magic04`, and `german-numeric` were candidates but are **not present** in the
  current OpenML active dataset list (scanned all 6,528 actives on 2026-10-17) — dropped.
- Coverage: binary × multiclass, tiny → medium size, low-dim → high-dim, balanced → imbalanced.
- Largest set is n=5,620, so the full suite stays laptop-friendly (see §8).

## 4. Estimator and parameter space

One estimator family for all datasets/methods (comparability over per-tuning wins):

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

estimator = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=50, random_state=s))
```

Space (identical for every method; distributions expressed once in `benchmarks/methods.py`):

| Parameter | Domain |
|---|---|
| `rf__max_depth` | int 3–16 (inclusive) or None → encoded as 0 and mapped to None |
| `rf__min_samples_leaf` | log-uniform float, 1e-2 – 0.25 |
| `rf__criterion` | {gini, entropy} |

B2 implements this space as the **discrete grid** in `benchmarks/methods.py::SPACE`: max_depth on
the 14 integer values 3..16 (None excluded so every method shares one bit-identical candidate
set), min_samples_leaf on an 8-point log-spaced grid in [0.01, 0.25], criterion as-is —
14 × 8 × 2 = 224 configurations.

Preprocessing (scaler fit) is **inside** the pipeline, so its cost is included in every timing —
per GPT's design note that timings must reflect real usage.

## 5. Protocol

- **Inner CV**: StratifiedKFold(5) inside each search (`cv=5`).
- **Outer split**: one shared 80/20 stratified train/test split per (dataset, seed), identical for
  all methods; the final "honest" score is measured once on the held-out test set using each
  method's best configuration. Search itself never touches it.
- **Seeds**: S = **7** seeds per (method, dataset) — satisfies ≥5 and avoids even-split KFold
  symmetry artifacts. Seed controls: outer split, estimator `random_state`, sampler/surrogate RNG.
- **Per-evaluation logging**: every method emits a record after each model evaluation
  `(k, cv_score, wall_s)` → best-so-far curves are computed from these, not reconstructed
  afterwards.

## 6. Metrics

Primary (evaluation-budget axis):
1. **Best-so-far curve** per run: eval count k → best CV score seen by k. Plotted for all methods/datasets.
2. **Evaluations-to-reach-target**: smallest k such that best-so-far ≥ 95% of that run's own final
   best (self-contained target, robust to dataset difficulty). Reported median across seeds.

Secondary (wall-clock axis):
3. Wall time to the same target; total wall time per budget.
4. **Overhead breakdown** for SKSurrogate: surrogate model fit/predict time vs. candidate-model
   evaluation time (instrumented in B2 via a thin timing hook around the surrogate calls).

Reporting: accuracy as the common score across datasets, plus balanced_accuracy alongside for
arrhythmia (imbalanced 13-class) so class imbalance is visible but not allowed to drive claims.

## 7. Package layout (grows from `examples/benchmark_phase10.py`)

```
benchmarks/
├── __init__.py
├── datasets.json      # pinned dids + local Galaxy3 path (B2)
├── datasets.py        # download-by-did into ~/.cache/sksurrogate-benchmarks; load → DataFrame, y
├── methods.py         # parameter space definition; run_method(method, dataset, budget, seed) -> records
├── run.py             # CLI: python -m benchmarks.run --dataset all|ionosphere,... --budget 30 --seeds 7 [--smoke]
└── plot.py            # best-so-far PNGs, aggregate CSV, markdown table snippet for docs/benchmarks.md
```

- `examples/benchmark_phase10.py` (MLOps I/O timings) stays as-is: it is a performance-regression
  check, not part of this HPO suite.
- Outputs per run land in `benchmarks/results/<UTC-date>/`: one JSON per (method, dataset, seed),
  an aggregate CSV, and PNGs. The **aggregates + figures are version-controlled** under
  `docs/benchmarks/`; raw per-run JSON goes to Zenodo (B4) rather than the repo.

## 8. Runtime expectations and CI policy

- Full suite estimate: ~30 evals × 3 active methods × 7 seeds × 6 datasets ≈ 3,800 candidate
  fits; largest dataset n=5,620 with RF(n_estimators=50) → **order 1–3 h on a laptop** (single
  command: `python -m benchmarks.run --dataset all`). Documented as an overnight/laptop run.
- CI (`--smoke` job, added in B2): ionosphere only, budget 8, seeds 2, RF n_estimators=10 →
  < ~3 min on ubuntu-latest (2 vCPU). Keeps "reproducible" honest without bloating PR feedback.
- Full suite: manual dispatch + weekly schedule (public repo ⇒ free GitHub runners); results are
  re-run and diffed against the committed aggregates for drift detection.

## 9. Decisive ablation (B3, spec)

Same `SurrogateRandomCV` outer loop twice, identical space/estimator/budget/seeds:

- **Arm A (no surrogate)**: `min_evals = budget` → the response-surface approximation never
  engages; search degenerates to pure random sampling through the SRS machinery. This keeps the
  comparison *inside one code path* (only knob changed), isolating exactly the surrogate's
  contribution.
- **Arm B (surrogate on)**: default engagement request (`min_evals = max(5, budget // 5)`), subject to
  the constructor clamping of the v1.4 errata — effective threshold 20 in this space, so at B = 30
  the surrogate guides records k ≥ 20 only (and for B < 21 Arms A and B are identical by
  construction).

Reported per arm: evaluations-to-reach-target (§6) and wall time; plus Arm-B overhead breakdown
(§6.4). A plain `RandomizedSearchCV` run from §2 serves as a cross-check that Arm A behaves like
independent random search (sanity, not the primary contrast).

## 10. Risks / open items (tracked in B2)

1. **skopt import failure** — expected; recorded as data (§2), never silently skipped from docs.
2. **Evaluation counting parity** — Optuna trial count must equal candidate fits exactly
   (objective wraps a single fit+CV score); assert `len(study.trials) == budget` at the end.
3. **Surrogate overhead instrumentation** — thin timing hook around surrogate fit/predict calls;
   if it proves invasive, fallback: measure via `verbose=1` log deltas (documented downgrade).
4. **Machine variance for wall-clock claims** — all wall-clock numbers reported with the machine
   description (CPU model, vCPU count) in the results JSON and in docs; evaluation-count metrics
   are the primary claim because they are hardware-independent.

## 11. Acceptance mapping (B1)

- Baselines fixed & documented: §2 (incl. skopt archival status with verification date).
- Datasets fixed & documented: §3 (dids pinned, sizes verified live, drop reasons recorded).
- Budgets in evaluations **and** wall-clock: §5–§6. Seeds ≥5: S=7 (§5).
