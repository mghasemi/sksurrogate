# SKSurrogate Benchmarks

Head-to-head comparison of SKSurrogate's surrogate-guided search (`SurrogateRandomCV`, SRS) against
standard AutoML/HPO baselines — uniform random sampling and Optuna-TPE — under **equal evaluation
budgets**, plus a decisive ablation isolating the value of the surrogate component itself. This page is
the published evidence behind the README's "validated against benchmarks" claim; the full protocol
contract, errata history, and acceptance mapping live in [`benchmarks/DESIGN.md`](../benchmarks/DESIGN.md) (v1.9).

Raw per-run records are git-ignored by design and archived on Zenodo (DOI: _pending — see Reproducibility_);
everything below is reproducible from the committed harness + those archives.

## Protocol

| Item | Setting |
|---|---|
| Estimator (all methods) | `StandardScaler → RandomForestClassifier(n_estimators=50)` |
| Parameter space | 3 parameters, one shared discrete grid of **224 configurations**: `max_depth ∈ {3…16}` (14), `min_samples_leaf` log-spaced 8 points in [1e-2, 0.25], `criterion ∈ {gini, entropy}` |
| Inner CV | StratifiedKFold(5) inside every search |
| Outer split | One shared stratified 80/20 per (dataset, seed), identical for all methods; held out from search entirely — final "honest" score measured once on it |
| Seeds | S = 7 per (method, dataset); control the outer split, estimator RNG, and sampler/surrogate RNG |
| Budgets | **B = 30** (full five-method suite) and **B = 60** (SRS arms), measured as candidate-model evaluations; verified post-run `n_evals == budget` for every cell. Wall-clock is secondary — evaluation counts are hardware-independent |

Datasets (pinned by OpenML `did`; sizes verified live against the OpenML API on 2026-10-17):

| Dataset | did / source | n rows | Features | Classes | Role |
|---|---:|---:|---:|---:|---|
| ionosphere | 59 | 351 | 34 | 2 | Tiny; pipeline validation |
| credit-g | 31 | 1,000 | 7 numeric + 13 nominal → d′=61 after one-hot | 2 | Small binary |
| arrhythmia | 5 | 452 | 280 | **13, imbalanced** | Hard: high-dim, multiclass, imbalanced |
| spambase | 44 | 4,601 | 58 | 2 | Medium binary |
| optdigits | 28 | 5,620 | 64 | 10 | Medium multiclass |
| Galaxy3 | bundled in repo | 1,600 | 20 | 2 | Synthetic epistasis (no download) |

## Methods under test

All methods optimize the **same estimator over the same space with the same budget**:

| Method | What it is |
|---|---|
| `sksurrogate` | SRS with its default response surface: cubic Hilbert polynomial (`HilbertRegressor(deg=3)`). Primary method. |
| `srs-no-surrogate` (ablation Arm A) | Identical SRS outer loop, surrogate never engages (`min_evals = B`) — search degenerates to random sampling through one code path. Isolates the surrogate's contribution exactly. |
| `random` | Uniform sampling over the shared discrete space (≡ `RandomizedSearchCV` on categorical list spaces). |
| `optuna-tpe` | `OptunaStudy(TPESampler(seed=s))`, one trial = one model evaluation, counted exactly. |
| `skopt-bayes` | **Not runnable**: the scikit-optimize repo is archived (last push 2024-02, verified 2026-10-17) and fails to import against modern scikit-learn — recorded as data, per contract. |

Two further SRS arms test response-surface quality at identical engagement (`min_evals = C(3+3,3) = 20`,
so the surface model is the only variable): `sksurrogate-gp` (stock sklearn `GaussianProcessRegressor`, RBF)
and `sksurrogate-krr` (stock `KernelRidge`, RBF, α=1.0). Hyperparameters deliberately un-tuned — a test of
surface quality, not a tuned-kernel contest.

## Results at B = 30 (full five-method suite + surface arms)

Canonical run `benchmarks/results/20261018T003121Z` (210 records: 6 datasets × 5 methods × 7 seeds),
merged with the v1.8 surface arms (`20261024T224454Z`). Mean final CV accuracy over 7 seeds — best per row in bold:

| dataset | SRS + Hilbert³ (default) | SRS + GP | SRS + KRR | SRS, no surrogate (Arm A) | random | Optuna-TPE |
|---|---:|---:|---:|---:|---:|---:|
| arrhythmia | 0.6985 | 0.7020 | 0.6910 | 0.6910 | 0.6941 | **0.7036** |
| credit-g | 0.7400 | 0.7423 | 0.7384 | 0.7391 | 0.7405 | **0.7436** |
| galaxy3 | 0.6455 | 0.6473 | 0.6443 | 0.6448 | 0.6484 | **0.6507** |
| ionosphere | 0.9342 | **0.9352** | 0.9342 | 0.9347 | 0.9347 | 0.9337 |
| optdigits | 0.9399 | 0.9403 | 0.9392 | 0.9392 | 0.9380 | **0.9405** |
| spambase | 0.9277 | 0.9280 | 0.9269 | 0.9269 | 0.9272 | **0.9286** |

Primary metric — median model evaluations to reach within an absolute gap $g = 0.01$ of the run's own final
best CV (lower is better; self-target, so dataset difficulty does not confound):

| dataset | SRS + Hilbert³ | SRS + GP | SRS + KRR | no surrogate (A) | random | Optuna-TPE |
|---|---:|---:|---:|---:|---:|---:|
| arrhythmia | 10.0 | 14.0 | **6.0** | **6.0** | 11.0 | 12.0 |
| credit-g | 9.0 | 9.0 | 9.0 | 9.0 | **2.0** | 5.0 |
| galaxy3 | 9.0 | 13.0 | 9.0 | 9.0 | **2.0** | 13.0 |
| ionosphere | **0.0** | **0.0** | **0.0** | **0.0** | 1.0 | 1.0 |
| optdigits | 6.0 | 6.0 | 6.0 | 6.0 | **2.0** | 6.0 |
| spambase | **2.0** | **2.0** | **2.0** | **2.0** | **2.0** | 5.0 |

Best-so-far curves, seed-averaged (canonical four arms):

| ionosphere | credit-g | arrhythmia |
|:---:|:---:|:---:|
| ![](benchmarks/figures/b30_ionosphere_best_sofar.png) | ![](benchmarks/figures/b30_credit-g_best_sofar.png) | ![](benchmarks/figures/b30_arrhythmia_best_sofar.png) |

| spambase | optdigits | galaxy3 |
|:---:|:---:|:---:|
| ![](benchmarks/figures/b30_spambase_best_sofar.png) | ![](benchmarks/figures/b30_optdigits_best_sofar.png) | ![](benchmarks/figures/b30_galaxy3_best_sofar.png) |

Response-surface comparison (cubic Hilbert vs RBF-GP vs RBF-KRR, plus random; seed-averaged):

| ionosphere | credit-g | arrhythmia |
|:---:|:---:|:---:|
| ![](benchmarks/figures/b30_ionosphere_surrogate_comparison.png) | ![](benchmarks/figures/b30_credit-g_surrogate_comparison.png) | ![](benchmarks/figures/b30_arrhythmia_surrogate_comparison.png) |

| spambase | optdigits | galaxy3 |
|:---:|:---:|:---:|
| ![](benchmarks/figures/b30_spambase_surrogate_comparison.png) | ![](benchmarks/figures/b30_optdigits_surrogate_comparison.png) | ![](benchmarks/figures/b30_galaxy3_surrogate_comparison.png) |

## B = 60 extension (SRS arms: does guidance compound?)

At B = 60 each SRS run receives ~40 surrogate-guided evaluations instead of 10, so a better surface has a real
window to work. Canonical SRS-arm directory `benchmarks/results/20261018T051214Z` (fixed runner) merged with the
surface arms (`20261024T224524Z`; 84/84 cells, all terminated by the evaluation budget). Mean final CV accuracy:

| dataset | SRS + Hilbert³ (default) | SRS + GP | SRS + KRR | SRS, no surrogate (Arm A) |
|---|---:|---:|---:|---:|
| arrhythmia | 0.7000 | **0.7056** | 0.6933 | 0.7004 |
| credit-g | 0.7412 | **0.7446** | 0.7384 | 0.7430 |
| galaxy3 | 0.6485 | **0.6501** | 0.6443 | 0.6479 |
| ionosphere | 0.9352 | **0.9362** | 0.9352 | **0.9362** |
| optdigits | 0.9399 | **0.9413** | 0.9392 | 0.9403 |
| spambase | 0.9277 | **0.9286** | 0.9269 | 0.9281 |

Primary metric (median evals to $g = 0.01$ of own final best):

| dataset | SRS + Hilbert³ | SRS + GP | SRS + KRR | no surrogate (A) |
|---|---:|---:|---:|---:|
| arrhythmia | 10.0 | 14.0 | **9.0** | 10.0 |
| credit-g | **9.0** | **9.0** | **9.0** | **9.0** |
| galaxy3 | 14.0 | 14.0 | **9.0** | 13.0 |
| ionosphere | **0.0** | **0.0** | **0.0** | **0.0** |
| optdigits | **6.0** | **6.0** | **6.0** | **6.0** |
| spambase | **2.0** | 5.0 | **2.0** | **2.0** |

SRS ablation pair (surrogate on vs off), seed-averaged:

| ionosphere | credit-g | arrhythmia |
|:---:|:---:|:---:|
| ![](benchmarks/figures/b60srs_ionosphere_best_sofar.png) | ![](benchmarks/figures/b60srs_credit-g_best_sofar.png) | ![](benchmarks/figures/b60srs_arrhythmia_best_sofar.png) |

| spambase | optdigits | galaxy3 |
|:---:|:---:|:---:|
| ![](benchmarks/figures/b60srs_spambase_best_sofar.png) | ![](benchmarks/figures/b60srs_optdigits_best_sofar.png) | ![](benchmarks/figures/b60srs_galaxy3_best_sofar.png) |

Response-surface comparison at B = 60 (three surfaces):

| ionosphere | credit-g | arrhythmia |
|:---:|:---:|:---:|
| ![](benchmarks/figures/b60_ionosphere_surrogate_comparison.png) | ![](benchmarks/figures/b60_credit-g_surrogate_comparison.png) | ![](benchmarks/figures/b60_arrhythmia_surrogate_comparison.png) |

| spambase | optdigits | galaxy3 |
|:---:|:---:|:---:|
| ![](benchmarks/figures/b60_spambase_surrogate_comparison.png) | ![](benchmarks/figures/b60_optdigits_surrogate_comparison.png) | ![](benchmarks/figures/b60_galaxy3_surrogate_comparison.png) |

## Findings

1. **RBF-GP is a consistent, significant upgrade over the default cubic Hilbert surface** — top mean CV on all six
   datasets at both budgets; seed-level vs the Hilbert arm 20W/17T/5L at B=30 and 25W/13T/4L at B=60 (two-sided exact
   binomial $p \approx 4\times10^{-3}$ and $p \approx 1\times10^{-4}$ respectively). The advantage *grows* with the
   guidance window — evidence of a real effect, not seed luck. Recommendation: RBF-GP as SKSurrogate's recommended
   default surface (v0.2 candidate; available today via the public `regressor=` kwarg).
2. **Stock KernelRidge is worse than doing nothing once the window opens** — it never wins a seed against Hilbert
   (B=60: 0W/23T/19L, all decisive seeds lost) and sits below the no-surrogate arm on mean CV in all six datasets at
   B=60. An un-tuned RBF-KRR fits noise as readily as signal at these budgets; do not ship it.
3. **The default cubic Hilbert polynomial is the weak link**: at B=60 the no-surrogate arm matches or beats it on mean
   CV in five of six datasets (margins ≤ 0.2 pt, within seed noise but uniformly directional). A surrogate helps only
   when its surface quality exceeds a random draw; with the default surface that is not always true, while RBF-GP
   matches/beats no-surrogate on all six datasets at B=60.
4. **Positioning vs AutoML baselines (honest version)**: at B=30 Optuna-TPE remains top in 5/6 datasets and every
   method stays within ~0.5 pt — the near-optimum RF(50) landscape is flat enough that "how fast" does not separate
   methods, and convergence speed shows no consistent ordering at either budget (medians 2–14; uniform random is
   fastest-to-target on four of six datasets). SRS's case rests on *final quality under an equal evaluation budget*,
   where the GP surface wins — not on reaching a saturated target faster. No spin: these are small-budget,
   three-parameter results, and TPE remains competitive.

## Reproducibility

```bash
# full suite (laptop-scale, ~1–3 h)
python -m benchmarks.run --dataset all --budget 30            # five methods (skopt recorded as skipped_archived)
python -m benchmarks.run --methods sksurrogate,srs-no-surrogate --budget 60   # B=60 SRS arms
python -m benchmarks.run --methods sksurrogate-gp,sksurrogate-krr --budget {30,60}

# aggregates + tables (this page's numbers)
python -m benchmarks.analyze_surrogates \
    benchmarks/results/20261018T003121Z benchmarks/results/20261024T224454Z --budget 30
python -m benchmarks.analyze_surrogates \
    benchmarks/results/20261018T051214Z benchmarks/results/20261024T224524Z --budget 60
```

| Run directory | Contents | Provenance |
|---|---|---|
| `benchmarks/results/20261018T003121Z` | B=30 canonical suite (210 records) | DESIGN v1.6; git `0d81fd5`, clean tree at launch |
| `benchmarks/results/20261018T051214Z` | B=60 canonical SRS arms (84 cells) | DESIGN v1.7 fixed runner (no-op-iteration defect fix) |
| `benchmarks/results/20261024T224454Z` | B=30 surface arms GP/KRR (84 cells) | DESIGN v1.8; manifests record base `eec6de9`, dirty tree at launch — reproducible from harness at commit `0e9850b` |
| `benchmarks/results/20261024T224524Z` | B=60 surface arms GP/KRR (84 cells) | as above; all cells `n_evals == 60`, terminated by `max_evals` |

Raw per-run JSON + run manifests are archived on Zenodo: _DOI pending_ (see `benchmarks/DESIGN.md` §12 for the
full errata/provenance history of each directory, including two invalidated early launches kept under
`benchmarks/results/_superseded/`).
