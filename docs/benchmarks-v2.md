# Benchmarks v2.0 — pre-registered protocol

**Status: protocol frozen 2026-11-24; pinned pilot complete; full matrix not approved (no-go).**
No v2.0 benchmark results exist yet. The previous suite is published and frozen as **v1.9** in
[benchmarks.md](benchmarks.md). The two are not comparable and are never pooled: the v0.6 search
contract redefined the evaluation currency (`n_iter`/`max_iter` now counts candidate evaluations
exactly), so v1.9 runs spent a different number of objective evaluations for the same nominal
budget.

The contract lives in `benchmarks/DESIGN.md` §16. This page is the user-facing summary.

## Why a new protocol

The 2026-11 review found the v1.9 evidence narrow: one estimator family (`StandardScaler → RF(50)`),
one 3-parameter space (224 categorical configurations), accuracy only, no wall-clock-normalized
table, and no synthetic objective with a known optimum. v2.0 widens the matrix and the statistics
before any competitive claim is edited.

## Pre-registered hypotheses

| # | Claim | Refuted if |
|---|---|---|
| **H1** | Under an evaluation budget, SRS-GP's mean final CV is at least TPE's on ≥ 50% of datasets, with the aggregate paired bootstrap 95% CI excluding 0. | The aggregate CI straddles 0, or GP wins < 50% of datasets. |
| **H2** | Under a wall-clock budget, SRS-GP is not worse than TPE by more than 10% of mean final CV. | TPE beats GP by > 10% on ≥ half of the datasets. |
| **H3** | The surrogate contributes: SRS-GP beats SRS-no-surrogate on paired seeds. | Paired sign test p > 0.05, or the direction reverses at B=60. |
| **H4** | On synthetic functions with known optima, SRS-GP's median simple regret at B=50 is no worse than TPE's. | Regret is worse than TPE on ≥ 50% of the six functions. |
| **H5** | At d ≥ 20 the cost is dominated by surrogate fitting, not candidate evaluation. | Surrogate-fit time < 25% of total wall time at d=20. |

Only H4 is an absolute "better optimizer" claim, because only the synthetic functions have a known
optimum. Everything else is a paired comparison under an equal budget.

## Matrix

| Dimension | v1.9 | v2.0 |
|---|---|---|
| Estimators | `RF(50)` | `SVC(rbf)`, `HistGradientBoostingClassifier`, `LogisticRegression`/`Ridge`, plus `RF(50)` anchor |
| Spaces | 3 categorical, 224 configs | mixed categorical/continuous; conditional; d = 10 and d = 30 continuous |
| Objectives | accuracy | regression (RMSE / R²), imbalanced (balanced accuracy), 2%-noise objective |
| Baselines | random, Optuna-TPE, skopt (archived) | + `RandomizedSearchCV`, SMAC3 if it installs, FLAML |
| Functions | — | Branin, Hartmann-3, Hartmann-6, Ackley, Rosenbrock, Levy + noisy variants |
| Cost regime | — | one expensive arm (≥ 1 s/eval) where surrogates should pay |

## Two currencies, never pooled

1. **Evaluation budget (primary).** Every arm spends exactly B candidate-model evaluations; the
   final held-out refit is excluded and reported separately.
2. **Wall-clock budget (secondary, machine-scoped).** Each run records total search wall time,
   summed candidate time, and surrogate overhead. Wall-clock tables are always published with the
   `run_manifest.json` machine description.

Paired statistics: percentile bootstrap CI of the mean paired delta (10,000 resamples, seed 0),
Wilcoxon signed-rank with Holm correction, Cliff's δ with magnitude bands, per-dataset rows plus an
aggregate. The self-target "evaluations to 1% of own final" metric is retained only as a secondary,
explicitly self-referential diagnostic.

**Power caveat:** with 7 seeds × 6 datasets, a per-dataset paired test has little power. Only the
aggregate is interpretable; per-dataset rows are descriptive.

## Pilot status

The pinned pilot is complete (`benchmarks/DESIGN-v2-cost.md`): 6 arms × 3 seeds × {B=30, B=60} ×
{`ionosphere`, `credit-g`}, 72/72 cells `ok` with `n_evals == budget`. Serial wall time was
128 s at B=30 and 295 s at B=60 on this machine. Median search overhead was 2.2–2.3% for the GP
surface at B=30 (4.7–5.2% at B=60) and 18.5–19% (44.5–45%) for the explicit Hilbert surface; the
no-surrogate arm stayed at 0.1%. All four real datasets are now measured for the component cost
model, which extrapolates **38.3 serial CPU-hours** for the full matrix.

**Owner decision (2026-11-24): pilot only — no-go for the full v2.0 matrix in this cycle.** The
pilot is descriptive, not evidence for H1–H5; the matrix is not approved and no README competitive
claim was edited. Revisit after the pilot is reviewed and v0.6.1 ships.

## Reproducing

```bash
# protocol smoke (fast, no OpenML dependency)
python -m benchmarks.synth
python -m pytest tests/test_benchmark_stats.py tests/test_benchmark_synth.py -q

# pilot (after the cost table is signed off for a larger matrix)
python -m benchmarks.run \
  --dataset ionosphere,credit-g --methods sksurrogate-gp,sksurrogate,sksurrogate-krr,srs-no-surrogate,random,optuna-tpe \
  --budget 30 --seeds 3
```

Raw runs land in `benchmarks/results/<UTC>/`, are archived to Zenodo, and are analysed with
`python -m benchmarks.analyze_surrogates <dir> --currency evals|wall`.
