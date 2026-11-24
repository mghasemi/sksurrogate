# Benchmark v2.0 cost model

Status: **measured component costs + pinned timing pilot; full matrix not approved** (not a benchmark result).

Machine: `{"cpus": 32, "platform": "Linux-7.2.15-76070215-generic-x86_64-with-glibc2.39", "python": "3.13.13", "stamp": "2026-11-24T05:09:36.486945+00:00"}`

## Candidate evaluation (5-fold CV, median of 3)

| dataset | family | seconds / evaluation |
|---|---|---:|
| credit-g | hgb | 0.6204 |
| credit-g | rf | 0.1735 |
| credit-g | svc | 0.0767 |
| ionosphere | logreg | 0.0162 |
| ionosphere | rf | 0.1448 |
| ionosphere | svc | 0.0149 |
| optdigits | hgb | 4.9130 |
| optdigits | svc | 0.6911 |
| spambase | hgb | 0.7475 |
| spambase | logreg | 0.0403 |

## Surrogate overhead (GP fit + one bounded LCB step, median of 3)

| parameter dim | evaluated points | fit (s) | acquisition (s) |
|---:|---:|---:|---:|
| 3 | 30 | 0.0033 | 0.0034 |
| 6 | 30 | 0.0024 | 0.0052 |
| 10 | 30 | 0.0022 | 0.0128 |
| 30 | 30 | 0.0036 | 0.0059 |

## Per-cell cost (surrogate arm)

| dataset | family | budget | per-eval (s) | one surrogate run (s) |
|---|---|---:|---:|---:|
| ionosphere | rf | 30 | 0.1448 | 4.41 |
| ionosphere | rf | 60 | 0.1448 | 8.96 |
| ionosphere | rf | 120 | 0.1448 | 18.05 |
| ionosphere | svc | 30 | 0.0149 | 0.51 |
| ionosphere | svc | 60 | 0.0149 | 1.16 |
| ionosphere | svc | 120 | 0.0149 | 2.46 |
| ionosphere | logreg | 30 | 0.0162 | 0.55 |
| ionosphere | logreg | 60 | 0.0162 | 1.24 |
| ionosphere | logreg | 120 | 0.0162 | 2.62 |
| credit-g | rf | 30 | 0.1735 | 5.27 |
| credit-g | rf | 60 | 0.1735 | 10.68 |
| credit-g | rf | 120 | 0.1735 | 21.49 |
| credit-g | svc | 30 | 0.0767 | 2.37 |
| credit-g | svc | 60 | 0.0767 | 4.87 |
| credit-g | svc | 120 | 0.0767 | 9.87 |
| credit-g | hgb | 30 | 0.6204 | 18.68 |
| credit-g | hgb | 60 | 0.6204 | 37.49 |
| credit-g | hgb | 120 | 0.6204 | 75.12 |
| spambase | hgb | 30 | 0.7475 | 22.49 |
| spambase | hgb | 60 | 0.7475 | 45.11 |
| spambase | hgb | 120 | 0.7475 | 90.36 |
| spambase | logreg | 30 | 0.0403 | 1.27 |
| spambase | logreg | 60 | 0.0403 | 2.68 |
| spambase | logreg | 120 | 0.0403 | 5.50 |
| optdigits | svc | 30 | 0.6911 | 20.80 |
| optdigits | svc | 60 | 0.6911 | 41.73 |
| optdigits | svc | 120 | 0.6911 | 83.60 |
| optdigits | hgb | 30 | 4.9130 | 147.46 |
| optdigits | hgb | 60 | 4.9130 | 295.04 |
| optdigits | hgb | 120 | 4.9130 | 590.22 |

## Extrapolated full matrix

| block | CPU-hours |
|---|---:|
| real-data matrix | 36.27 |
| (of which high-dimensional cells) | 8.86 |
| synthetic functions | 0.65 |
| expensive-objective arm | 1.40 |
| **total (serial)** | **38.32** |

Assumptions: {"arms": 9, "budgets": [30, 60, 120], "engagement_floor": 20, "expensive_objective_s_per_eval": 1.0, "seeds": 7, "surrogate_arms": 4, "synthetic_functions": 6}

Wall-clock scales with the number of parallel workers; this table is serial CPU time.

## Gate

**No full matrix runs until the owner records a go/no-go decision here.**

Decision: **full v2.0 matrix deferred; a reduced matrix is approved** (owner, 2026-11-24).
The pinned pilot is complete (below). The 38.32 serial CPU-hour matrix is **not** approved for this
cycle: with 7 seeds per cell no per-dataset paired test has useful power, so the spend would buy a
claim the study cannot then make. Approved instead:

- the **synthetic suite at full seeds** (`benchmarks/synth.py`: Branin, Hartmann-3, Hartmann-6,
  Ackley, Rosenbrock, Levy with deterministic noisy variants and regret) — known optima, so
  "better optimizer" is a well-posed claim measured by regret rather than CV on a real dataset;
- the **H5 overhead cells** at d = 20 and d = 30 on the two pilot datasets — the only hypothesis the
  pilot left unsupported, and search overhead is measurable without the rest of the model matrix;
- `optdigits` x HistGB (4.91 s/eval, the cell that dominates the cost model) stays deferred until the
  synthetic evidence motivates it.

Revisit the full matrix after the reduced run reports. `v0.8.0` ships with no competitive claim
beyond v1.9. This file is the record; no claim in `docs/benchmarks-v2.md` depends on an unrun matrix.

## Notes

**Pinned datasets.** OpenML authentication was restored: openml 0.15 stores a 10-character mask
when the key comes from `OPENML_API_KEY`, so `benchmarks.datasets` re-assigns the real environment
value before a download (the key is never written to disk). `credit-g`, `spambase` and `optdigits`
are now cached and measured.

**Superseded estimate.** An earlier two-dataset run reported 14.6 serial CPU-hours as a lower
bound; with all four datasets measured the estimate is **38.32 serial CPU-hours**. The increase is
dominated by `optdigits`/HistGradientBoosting (4.91 s per candidate evaluation) and the
high-dimensional block.

## Measured timing pilot (pinned datasets)

Run 2026-11-24 (`benchmarks/results/20261124T050358Z`, `...T050608Z`): 6 arms × 3 seeds ×
{B=30, B=60} × {ionosphere, credit-g}. Every one of the 72 cells is `ok` with
**`n_evals == budget`** (0 mismatches), so the v0.6 exact-budget contract holds at both budgets.

| budget | runs | serial wall (s) | candidate time (s) | search overhead (s) |
|---:|---:|---:|---:|---:|
| 30 | 36 | 128.0 | 122.3 | 5.7 |
| 60 | 36 | 294.9 | 250.0 | 44.9 |

Median search overhead as a fraction of total wall time:

| arm | B=30 | B=60 |
|---|---:|---:|
| `sksurrogate-gp` (library default) | 2.2–2.3% | 4.7–5.2% |
| `sksurrogate-krr` | 1.8–1.9% | 3.9–4.0% |
| `sksurrogate` (Hilbert, explicit) | 18.5–19.0% | 44.5–45.0% |
| `srs-no-surrogate` | 0.1% | 0.1% |
| `random` / `optuna-tpe` | ~0% | ~0% |

Pilot scores are descriptive only (2 datasets × 3 seeds; no claim is drawn): on `credit-g` the
median best CV was 0.7400 for both SRS-GP and Hilbert, 0.7362/0.7387 for random and
0.7263/0.7350 for TPE at B=30/B=60; on `ionosphere` every arm tied at 0.9286.

Reading for H5: at the d=3 anchor the GP surface costs a few percent of wall time and Hilbert is
the expensive surface; the component sweep shows the d=10 acquisition step at ~13 ms against a
0.15–4.9 s candidate evaluation. H5 (surrogate cost dominates at d ≥ 20) is therefore **not
supported by this evidence** and needs the d=20+ cells before any statement.


