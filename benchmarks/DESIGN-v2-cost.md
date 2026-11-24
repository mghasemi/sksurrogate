# Benchmark v2.0 cost model

Status: **measured component costs + explicit extrapolation** (not a benchmark result).

Machine: `{"cpus": 32, "platform": "Linux-7.2.15-76070215-generic-x86_64-with-glibc2.39", "python": "3.13.13", "stamp": "2026-11-24T04:46:41.108999+00:00"}`

## Candidate evaluation (5-fold CV, median of 3)

| dataset | family | seconds / evaluation |
|---|---|---:|
| galaxy3 | hgb | 0.6107 |
| galaxy3 | rf | 0.2086 |
| ionosphere | logreg | 0.0182 |
| ionosphere | rf | 0.1452 |
| ionosphere | svc | 0.0153 |

## Surrogate overhead (GP fit + one bounded LCB step, median of 3)

| parameter dim | evaluated points | fit (s) | acquisition (s) |
|---:|---:|---:|---:|
| 3 | 30 | 0.0032 | 0.0026 |
| 6 | 30 | 0.0025 | 0.0052 |
| 10 | 30 | 0.0019 | 0.0127 |
| 30 | 30 | 0.0034 | 0.0057 |

## Per-cell cost (surrogate arm)

| dataset | family | budget | per-eval (s) | one surrogate run (s) |
|---|---|---:|---:|---:|
| ionosphere | rf | 30 | 0.1452 | 4.41 |
| ionosphere | rf | 60 | 0.1452 | 8.94 |
| ionosphere | rf | 120 | 0.1452 | 18.01 |
| ionosphere | svc | 30 | 0.0153 | 0.52 |
| ionosphere | svc | 60 | 0.0153 | 1.15 |
| ionosphere | svc | 120 | 0.0153 | 2.41 |
| ionosphere | logreg | 30 | 0.0182 | 0.60 |
| ionosphere | logreg | 60 | 0.0182 | 1.32 |
| ionosphere | logreg | 120 | 0.0182 | 2.76 |
| galaxy3 | rf | 30 | 0.2086 | 6.31 |
| galaxy3 | rf | 60 | 0.2086 | 12.75 |
| galaxy3 | rf | 120 | 0.2086 | 25.61 |
| galaxy3 | hgb | 30 | 0.6107 | 18.38 |
| galaxy3 | hgb | 60 | 0.6107 | 36.87 |
| galaxy3 | hgb | 120 | 0.6107 | 73.86 |

## Extrapolated full matrix

| block | CPU-hours |
|---|---:|
| real-data matrix | 12.57 |
| (of which high-dimensional cells) | 8.87 |
| synthetic functions | 0.65 |
| expensive-objective arm | 1.40 |
| **total (serial)** | **14.63** |

Assumptions: {"arms": 9, "budgets": [30, 60, 120], "engagement_floor": 20, "expensive_objective_s_per_eval": 1.0, "seeds": 7, "surrogate_arms": 4, "synthetic_functions": 6}

Wall-clock scales with the number of parallel workers; this table is serial CPU time.

## Gate

**No full matrix runs until the owner records a go/no-go decision here.**

Decision: _(pending)_

## Notes

**Pilot substitution.** The protocol pins *ionosphere* + *credit-g*. This environment has no OpenML API credential, so *credit-g* (did 31) cannot be downloaded; the bundled *galaxy3* set was substituted for the timing probe only. This is a cost instrument, not protocol evidence. **Coverage gap.** spambase and optdigits were not measured for the same reason, so the extrapolated total is a lower bound. Re-run with the pinned datasets once credentials are available.

## Measured timing pilot (proxy datasets)

Run 2026-11-24, 6 arms × 3 seeds × {B=30, B=60} × {ionosphere, galaxy3}; every one of the
72 cells completed with **`n_evals == budget`** (0 mismatches), i.e. the exact-budget contract
holds at the v2.0 budgets.

| budget | runs | serial wall (s) | candidate time (s) | search overhead (s) |
|---:|---:|---:|---:|---:|
| 30 | 36 | 128.4 | 122.7 | 5.6 |
| 60 | 36 | 284.6 | 245.5 | 39.0 |

Median search overhead as a fraction of total wall time:

| arm | B=30 | B=60 |
|---|---:|---:|
| `sksurrogate-gp` (library default) | 2.1–2.3% | 5.0–6.3% |
| `sksurrogate-krr` | 2.0% | 3.6–4.4% |
| `sksurrogate` (Hilbert, explicit) | 18.4–18.8% | 44.5–44.9% |
| `srs-no-surrogate` | 0.1% | 0.1% |
| `random` / `optuna-tpe` | ~0% | ~0% |

Reading: at the v1.9 anchor space (d=3) the GP surface costs a few percent of wall time; the
cubic Hilbert basis is the expensive one. H5 (surrogate cost dominates at d ≥ 20) is **not
testable** from this pilot: the parameter dimension stays 3, and the component sweep above shows
the d=10 acquisition step at ~13 ms versus a 0.15–0.6 s candidate evaluation.

**Gate status: pending.** Two things are needed before the full matrix: (1) an OpenML API
credential to run the protocol-pinned pilot and to measure the larger datasets; (2) the owner's
go/no-go decision recorded below.

