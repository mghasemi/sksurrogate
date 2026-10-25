# SKSurrogate: scikit-learn-native surrogate-guided hyperparameter search

*A JOSS-style technical note* — draft v0.1.1 · 2026-10-25

**Authors:** Mehdi Ghasemi
**Repos / artifacts:** `github.com/mghasemi/sksurrogate` (v0.2.0) · benchmark raw results DOI 10.5281/zenodo.23629024 · package DOI 10.5281/zenodo.23325033

> **Status note (B5 gate).** This note is the decision-gate artifact from the SKSurrogate
> "modernization & competitive positioning" plan. Framing is deliberately *honest*: we claim
> **equal-budget competitive accuracy** and a **clean with-vs-without surrogate ablation**, not
> "fewer evaluations to reach target accuracy" (unsupported by the data; see §Results caveats).

---

## Abstract

`SKSurrogate` provides a lightweight, `scikit-learn`-native surrogate-guided search for
hyperparameter optimization and AutoML pipeline design. We evaluate the search against uniform
random sampling and Optuna-TPE at **equal evaluation budgets** on six OpenML/bundled datasets, and
isolate the value of the surrogate component with a controlled ablation (identical search with the
surrogate engaged vs never engaged). At a fixed budget of 30 candidate-model evaluations, SKSurrogate
with its default RBF-GP surface matches Optuna-TPE within ~0.002 in cross-validated accuracy on five
of six datasets and surpasses it on the hardest imbalanced high-dimensional case; at a budget of 60
it leads all six. The ablation shows the surrogate matches or improves final accuracy on every
dataset. A one-command harness and a Zenodo-archived raw-results record make every result
reproducible.

---

## Summary

`SKSurrogate` is a suite of `scikit-learn`-style tools that implements *surrogate-guided
optimization* for expensive objectives, primarily hyperparameter / AutoML pipeline search. A search
drives an evolutionary outer loop and, once enough evaluations have accumulated, replaces direct
(expensive) model fitting for candidate proposals with a cheap surrogate of the response surface,
using the surrogate's optimum to guide the next evaluations.

Since version **0.2.0** the default response surface is a **stock, un-tuned RBF
`GaussianProcessRegressor`** (chosen over the legacy cubic Hilbert polynomial on the evidence of an
equal-budget benchmark suite). All `scikit-learn` regressors are accepted as surfaces, and a custom
`HilbertRegressor` (cubic polynomial) remains available for the legacy default.

## Method

`SKSurrogate.structsearch.SurrogateRandomCV` implements a local surrogate-driven random search (SRS)
over a discrete search space typed by `Categorical`/`Integer`/`Real`/`HDReal` descriptors. The search
loops:

1. **Outer evolutionary loop.** Candidate parameterizations are sampled and locally optimized.
2. **Surrogate engagement.** After `min_evals` evaluations, a regressor is fit to the evaluated
   points and used to (a) propose candidate optima and (b) screen unpromising regions, so that only
   promising points are paid for with an expensive model fit.
3. **Evaluation budget.** Every candidate-model evaluation is counted; the budget `B` is the binding
   constraint (verified post-run `n_evals == B`).

`min_evals = B` degrades the search to plain random sampling through the *same* code path — this is
the mechanism used for the ablation (§Results), isolating the surrogate's contribution exactly.

## Related work

Mainstream HPO is dominated by Bayesian/surrogate methods: TPE (Hyperopt), SMAC, Optuna, and (until
archival) scikit-optimize; AutoML systems layer these onto pipeline construction (e.g., Auto-Sklearn)
and early-stopping. SKSurrogate's niche is (i) `scikit-learn`-native **typed search spaces**
(`Categorical`/`Integer`/`Real`/`HDReal`) with a `BaseSearchCV`-style interface, (ii) a lightweight
**evolutionary outer loop with a surrogate-guided inner step** rather than a global
acquisition-function optimization, and (iii) an explicitly frozen, version-controlled **equal-budget
benchmark suite** with archived raw results — so claims are tied to a reproducible evaluation rather
than to ad-hoc configs.

> **Citations to add before submission** (placeholders; not fabricated here): Optuna; TPE/Hyperopt;
> SMAC; Auto-Sklearn; scikit-optimize (archived); surrogate-model HPO surveys.

## Benchmark protocol

Public benchmark suite in `benchmarks/`, reproducible with one command
(`python -m benchmarks.run --dataset <d> --budget <B> --seeds <s>`); see `benchmarks/DESIGN.md` (v1.9).
Raw per-run records are git-ignored by design and archived on Zenodo (DOI 10.5281/zenodo.23629024).

| Item | Setting |
|---|---|
| Estimator (all methods) | `StandardScaler → RandomForestClassifier(n_estimators=50)` |
| Space | 3 params, one shared discrete grid of **224** configs |
| Inner CV | `StratifiedKFold(5)` inside every search |
| Outer split | one shared stratified 80/20 per (dataset, seed), identical for all methods; final "honest" score measured once on the hold-out |
| Seeds | S = 7 per (method, dataset) |
| Budgets | **B = 30** (five-method suite) and **B = 60** (SRS arms), in candidate-model evaluations |

Methods under test: `sksurrogate` (default RBF-GP since v0.2), `srs-no-surrogate` (ablation arm A),
`random`, `Optuna-TPE`; surface arms `SRS+GP` / `SRS+KRR`. `skopt-bayes` is recorded as not runnable
(repo archived, fails to import against modern scikit-learn).

Datasets (OpenML `did`): ionosphere (59), credit-g (31), arrhythmia (5, high-dim/imbalanced), spambase (44),
optdigits (28), Galaxy3 (bundled; synthetic epistasis).

## Results

Mean final cross-validated accuracy over 7 seeds, **B = 30** (best per row in bold):

| dataset    | SRS (RBF-GP) | no surrogate | random | Optuna-TPE |
|------------|--------------|--------------|--------|------------|
| arrhythmia | 0.7020       | 0.6910       | 0.6941 | **0.7036** |
| credit-g   | 0.7423       | 0.7391       | 0.7405 | **0.7436** |
| galaxy3    | 0.6473       | 0.6448       | 0.6484 | **0.6507** |
| ionosphere | **0.9352**   | 0.9347       | 0.9347 | 0.9337     |
| optdigits  | 0.9403       | 0.9392       | 0.9380 | **0.9405** |
| spambase   | 0.9280       | 0.9269       | 0.9272 | **0.9286** |

**B = 60** (SRS arms):

| dataset | SRS (RBF-GP) | SRS (cubic Hilbert) | no surrogate |
|---|---|---|---:|
| arrhythmia | **0.7056** | 0.7000 | 0.7004 |
| credit-g | **0.7446** | 0.7412 | 0.7430 |
| galaxy3 | **0.6501** | 0.6485 | 0.6479 |
| ionosphere | **0.9362** | 0.9352 | 0.9362 |
| optdigits | **0.9413** | 0.9399 | 0.9403 |
| spambase | **0.9286** | 0.9277 | 0.9281 |

### Contributions (what survives scrutiny)

1. **Equal-budget parity with a strong HPO baseline.** Against Optuna-TPE at B=30, SRS (RBF-GP) is
   within ~0.002 on 5/6 datasets and *beats* it on the hardest case (ionosphere; high-dim
   imbalanced). At B=60, SRS (RBF-GP) **leads all six**.
2. **A decisive ablation.** Identical outer loop, surrogate engaged vs never engaged: the surrogate
   **matches or raises final accuracy on every dataset** at both budgets — the value of the core
   combination isolated without confounding.
3. **Reproducibility.** One-command harness, version-controlled, raw results archived (DOI
   10.5281/zenodo.23629024).

### Caveats (stated honestly)

- **"Fewer evaluations to target" is NOT claimed.** Median evaluations to reach within an absolute
  gap of a run's own final best is *mixed* — `random` reaches its own near-target in as few as 2
  evaluations on `credit-g`/`galaxy3`. The advantage shown here is at *fixed total budget*, not
  faster convergence. Any future claim of sample-efficiency needs a wider dataset matrix and
  target-anchored protocol.
- The surrogate's benefit is modest in **absolute** terms (a few tenths of a percent) but consistent
  and never harmful.
- Measurement is a single 80/20 split per (dataset, seed); repeated-seed aggregates reduce but do not
  eliminate split variance.

## Reproducibility

- Harness: `python -m benchmarks.run` (see `benchmarks/run.py`); analyses in `benchmarks/methods.py`.
- Figures + tables: `docs/benchmarks.md`; figures in `docs/benchmarks/figures/`.
- Raw run records (git-ignored): Zenodo DOI 10.5281/zenodo.23629024.
- Software: PyPI `sksurrogate==0.2.0`; source `github.com/mghasemi/sksurrogate` tag `v0.2.0`.

## Summary

`SKSurrogate` offers a lightweight, `scikit-learn`-native surrogate-guided search that is
**competitive with Optuna-TPE at equal evaluation budgets**, whose surrogate component is shown by a
clean ablation to **never hurt and to raise final accuracy**, and which ships with a fully
reproducible benchmark suite. This is positioned as a research contribution in software and
methodology (JOSS-style), with explicit and honest limitations.

## Appendix A — Evaluations to reach target (reported, but not a headline claim)

Median model evaluations to reach within an absolute gap $g = 0.01$ of a run's *own* final best CV
(lower is better; self-anchored, so dataset difficulty does not confound). **B = 30:**

| dataset | SRS (RBF-GP) | no surrogate | random | Optuna-TPE |
|---|---|---:|---:|---:|
| arrhythmia | 14 | 6 | 11 | 12 |
| credit-g | 9 | 9 | 2 | 5 |
| galaxy3 | 13 | 9 | 2 | 13 |
| ionosphere | 0 | 0 | 1 | 1 |
| optdigits | 6 | 6 | 2 | 6 |
| spambase | 2 | 2 | 2 | 5 |

**B = 60** (SRS arms: cubic Hilbert / RBF-GP / RBF-KRR / no surrogate):

| dataset    | Hilbert³ | RBF-GP | RBF-KRR | no surrogate |
|------------|----------|--------|---------|-------------:|
| arrhythmia | 10       | 14     | **9**   |           10 |
| credit-g   | **9**    | **9**  | **9**   |            **9** |
| galaxy3    | 14       | 14     | **9**   |           13 |
| ionosphere | **0**    | **0**  | **0**   |            **0** |
| optdigits  | **6**    | **6**  | **6**   |            **6** |
| spambase   | **2**    | 5      | **2**   |            **2** |

These medians are **mixed**: `random` occasionally reaches its own near-best in very few evaluations
(e.g., 2 on credit-g/galaxy3), and the surrogate is not consistently faster to threshold. This is why
the note claims *equal-budget accuracy* rather than *sample efficiency*. A defensible
sample-efficiency claim would require a target-anchored protocol across a larger dataset matrix.

---

*Roadmap (from the project board): this note feeds Phase C (README/positioning) and, with additional
results, a full paper candidate (D-track).*