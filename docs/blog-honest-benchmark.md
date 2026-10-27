# An Honest Benchmark of My Own AutoML Package

*Where SKSurrogate beats Optuna-TPE, where it loses — and why the null results are the point.*

*Mehdi Ghasemi — October 2026*

> **TL;DR**
> - At equal evaluation budgets on six public datasets, my surrogate-guided search (**SRS**) in [SKSurrogate](https://pypi.org/project/sksurrogate/) does **not** beat Optuna-TPE on accuracy — TPE is top on 5 of 6 datasets and everything sits within ~0.5 pt.
> - What *is* real: the **response surface matters more than the search mechanism**. An un-tuned RBF Gaussian process beats my default polynomial regressor with seed-level significance ($p \approx 10^{-4}$ at B=60), while an un-tuned kernel ridge regressor is **worse than doing nothing**.
> - The most useful result was a defect in *my own* pipeline: ordinal encoding of categorical features was silently masking the surrogate's advantage. Fixing it flipped SRS above TPE on 3 of 6 datasets — small, not significant at n=7 seeds, but uniformly directional.

## Why I wrote this post

Most AutoML packages ship "state-of-the-art" claims that are impossible to reproduce: no protocol contract, no archived raw runs, baselines chosen to lose. SKSurrogate was built the other way around — an equal-budget benchmark suite with a frozen protocol, pre-registered hypotheses with explicit falsifiers, integrity gates run before any canonical launch, and every raw record archived on Zenodo so the advertised "validated against benchmarks" claim is verifiable without rerunning anything.

That means I also have to publish what the data says when it's inconvenient: **at small budgets, my package does not beat TPE.** This post walks through three findings from ~460 archived runs — one null result, one mechanism explanation, and one defect we caught in ourselves.

## The setup, in 60 seconds

All methods optimize the *same* estimator (`StandardScaler → RandomForestClassifier(50)`) over the *same* shared discrete grid (224 configurations across 3 parameters), with stratified 5-fold CV inside every search and one shared held-out 80/20 outer split per dataset-seed. Budgets are measured in **candidate-model evaluations** — a hardware-independent currency. Seven seeds per cell; 6 datasets from OpenML (pinned by `did`) plus one bundled synthetic set, B = 30 and B = 60 evaluations:

| Arm | What it is |
|---|---|
| SRS + RBF-GP | The package's search with its v0.2 default surface (stock un-tuned `GaussianProcessRegressor`) |
| SRS, no surrogate | Identical outer loop, surrogate never engages — the ablation that isolates the surrogate's contribution |
| random / Optuna-TPE | Uniform sampling and TPE as the standard baselines (`scikit-optimize` is archived upstream; recorded as "not runnable" rather than silently dropped) |

## Finding 1: parity is the honest headline

Mean final CV accuracy, B = 30 (bold = best per row):

| dataset | SRS + RBF-GP | no surrogate | random | Optuna-TPE |
|---|---:|---:|---:|---:|
| arrhythmia | 0.7020 | 0.6910 | 0.6941 | **0.7036** |
| credit-g | 0.7423 | 0.7391 | 0.7405 | **0.7436** |
| galaxy3 | 0.6473 | 0.6448 | 0.6484 | **0.6507** |
| ionosphere | **0.9352** | 0.9347 | 0.9347 | 0.9337 |
| optdigits | 0.9403 | 0.9392 | 0.9380 | **0.9405** |
| spambase | 0.9280 | 0.9269 | 0.9272 | **0.9286** |

TPE is top on five of six datasets, and the whole table spans roughly half a point — a RandomForest(50) over three parameters is a flat landscape where no search method separates. I deliberately do *not* claim "fewer evaluations to reach target accuracy" either: on four of six datasets, uniform random reaches its own near-best in as few as 2 evaluations. If you're tuning one model with three knobs on a laptop budget, the honest advice is that your choice among these tools will not move your accuracy much.

## Finding 2: surface quality beats search mechanism

Where the data *does* separate things is inside SRS itself — which response surface you fit to the evaluations:

- **RBF-GP vs my legacy cubic Hilbert polynomial:** top mean CV on all six datasets at both budgets; seed-level wins of 20W/17T/5L (B=30) and 25W/13T/4L (B=60), exact binomial $p \approx 4\times10^{-3}$ and $p \approx 10^{-4}$. The advantage *grows* as the guidance window opens — a real effect, not seed luck. This is why v0.2 ships RBF-GP as the default.
- **The ablation says the surrogate earns its keep with a good surface:** at B=60, SRS + GP matches or raises final accuracy on all six datasets vs the no-surrogate arm (e.g., credit-g 0.7446 vs 0.7430; arrhythmia 0.7056 vs 0.7004).
- **Un-tuned KernelRidge is worse than doing nothing:** at B=60 it sits below the no-surrogate arm on mean CV in *all six* datasets and never wins a seed against the Hilbert arm (0W/23T/19L). At these budgets an un-tuned RBF kernel ridge fits noise as readily as signal. If you build this kind of thing: don't ship it.

A controlled synthetic study (the "D-track", 2,160 additional runs) explains *why* the surfaces separate the way they do — a regime map over objective family × dimension × sample size × noise:

| objective family | Hilbert poly | RBF-GP | Random forest |
|---|---:|---:|---:|
| polynomial (in-span) | **14**/18 | 2 | 2 |
| Gaussian bump | 2 | **10** | 6 |
| rational | 3 | 5 | **10** |
| trigonometric | 0 | 1 | **17** |

The mechanisms, not just the winners: the polynomial regressor is *exact* on in-span objectives even at $n=10$ (its error elsewhere is basis-truncation, flat in sample size); the stock RBF-GP has no noise term and interpolates noise — RMSE jumps from $\approx 0$ clean to 0.6–2.9 with moderate noise; RF is the robust generalist that never gets exactness. Real HPO landscapes (CV accuracy over a discrete grid) are not polynomial, which is exactly why the GP surface wins in Finding 2 and why my original Hilbert default was the weak link.

## Finding 3: the defect we found in ourselves

The pipeline-space track ("C-space") searches what an AutoML system actually chooses — **1,220 pipelines** (4 preprocessors × 5 reducers × 61 classifier configurations) — under three pre-registered protocols (equal wall-clock 120 s; equal 40 evaluations; wall budget as a cap), with hypotheses H1–H4 and their falsifiers fixed before the first canonical run. Six hundred thirty cells, zero invalid candidates. What came back:

1. **No arm separates on held-out accuracy** — every paired comparison is null after Holm correction ($p_{\text{Holm}} = 1.000$ throughout). Flat again.
2. **But SRS is evaluation-efficient:** it reaches within 1% of its own final best in a median of **6 evaluations**, against 11–17 for every other arm, at identical final accuracy.
3. **That advantage doesn't transfer to wall-clock.** The GP surface fit grows as $O(n^3)$ — per-fit cost rises from 3 ms to 174 ms over a run — so 49% of SRS's 120 s budget is search overhead (22% fitting, 27% acquisition), buying ~half as many candidates as random search. And TPE spends 28–65% of its budget *re-proposing configurations it already evaluated*.
4. **The ordinal-encoding defect was load-bearing.** With the default ordinal encoding, SRS loses to or ties TPE (H1: confirmed). Replacing it with a mixed categorical surface lifts best-CV by ≥0.3 pt in 4 of 6 datasets (+1.63 on galaxy3, +0.51 ionosphere, +0.43 credit-g, +0.37 optdigits) and moves SRS **above TPE in three of them**. Not significant after multiplicity correction — only 3 of the 4 bootstrap CIs exclude zero. But it is uniformly directional, and it changed what we ship next.
5. **Portfolio AutoML still wins on accuracy:** FLAML beats every search arm on 5 of 6 datasets at the same wall cap (AutoGluon: 3/6 — which *refuted* our pre-registered joint hypothesis H3; that's pre-registration doing its job, telling us exactly what to believe and how strongly).

## What this means if you're choosing an HPO tool

- **Small flat spaces, laptop budgets:** TPE or plain random search is fine. The differences are below the noise of your problem.
- **Surrogate-guided search earns its keep when evaluations are expensive** (each one costs minutes) and you care about final quality at a fixed budget — that's where SRS + a good surface shows up, with ~half the evaluations to get near-best on pipeline spaces.
- **If your space has categorical features, check how they're encoded.** Ordinal encoding of unordered categories silently degrades surrogate guidance — the single most actionable lesson from this study.
- **Match the surface to the objective's shape** when you can: polynomial-like/noisy → polynomial basis; clean and smooth → GP (with a noise term if evaluations are noisy); unknown shape → tree-based generalist or an ensemble with fallback.

## How to reproduce all of this

```bash
pip install sksurrogate          # v0.2.0, PyPI
python -m benchmarks.run --dataset all --budget 30      # full five-method suite (~1–3 h laptop-scale)
python -m benchmarks.dtrack.d2_compare                  # the regime-map study
```

- Protocol contract + errata history: `benchmarks/DESIGN.md` (v1.9) in [github.com/mghasemi/sksurrogate](https://github.com/mghasemi/sksurrogate)
- Raw B=30/B=60 records (462 runs): [doi.org/10.5281/zenodo.23629024](https://doi.org/10.5281/zenodo.23629024)
- D-track raw data + report: [doi.org/10.5281/zenodo.23652613](https://doi.org/10.5281/zenodo.23652613)
- Software archive (v0.2.0): [doi.org/10.5281/zenodo.23631823](https://doi.org/10.5281/zenodo.23631823)

## Caveats I'm not hiding

Seven seeds per cell; budgets of 30–60 evaluations; the fixed-pipeline suite's space is only three parameters wide; D-track objectives are smooth and low-dimensional ($d \le 3$); the wall-clock arm uses a different currency (seconds, hardware-dependent) and is explicitly *not* comparable to the evaluation-count tables. These are small-budget results, and TPE remains competitive — that's stated in the docs too, no spin required. The suite being public means anyone can extend it with more seeds or datasets; I'd rather be wrong in a verifiable way than impressive in an unverifiable one.
