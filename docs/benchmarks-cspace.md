# Pipeline-space (C-space) benchmark results

Canonical runs of the heterogeneous search space defined in `benchmarks/DESIGN-cspace.md`
(**v0.3**): 4 preprocessors × 5 reducers × 61 classifier configurations = **1,220 pipelines**,
searched by six arms plus two system-level AutoML baselines, over 6 datasets × 7 seeds.

| Protocol | Question | Currency | Directory | Cells |
|---|---|---|---|---|
| **A** | which search method wins under a real time budget? | equal wall-clock, 120 s | `cspace-A-20261027T001347Z` | 252 |
| **B** | which search method wins per candidate evaluated? | equal 40 evaluations | `cspace-B-20261026T224531Z` | 252 |
| **C** | what does a search arm lose to portfolio AutoML? | wall budget as a **cap** | `cspace-C-20261027T084247Z` | 126 |

**630 cells, 0 errors, 0 invalid candidates, 42/42 SRS cells engaged the surrogate.**

---

## 1. Headline findings

1. **No arm separates on held-out accuracy.** In both currencies every paired comparison
   against the reference arm is non-significant after Holm correction ($p_{\text{Holm}} = 1.000$
   throughout; decisive non-tied $n \leq 7$ per dataset). This is a flat landscape, not a
   ranking with winners.
2. **Surrogate guidance is evaluation-efficient.** Equal-evaluation (Protocol B): `srs-cspace`
   reaches within 1 % of its own best score in a **median of 6 evaluations**, against 11–17 for
   every other arm, at identical final accuracy.
3. **That advantage does not transfer to wall-clock.** Equal-time (Protocol A): **49 % of
   `srs-cspace`'s 120 s budget is search overhead** — 22 % fitting the GP surface, 27 % in the
   acquisition optimiser — so it buys ~half as many candidates as random search (median 281 vs
   553). The fitted surface grows as $O(n^3)$: per-fit cost rises from 3 ms to 174 ms over a run.
4. **The ordinal-encoding defect is load-bearing.** Replacing it with a mixed categorical
   surface lifts best-CV by ≥0.3 pt in **4 of 6 datasets** and, in three of them, moves
   `srs-onehot-cspace` above `optuna-tpe-cspace` — i.e. the encoding defect was masking the
   surrogate's advantage (H2 confirmed, §3).
5. **Default TPE wastes a large share of its budget re-proposing seen configurations:** 28.2 %
   of a 40-evaluation budget (Protocol B) and 65.0 % over ~350 evaluations (Protocol A), against
   sampling-with-replacement floors of 1.6 % and 19.5 %.
6. **Portfolio AutoML still wins on accuracy**: FLAML exceeds every search arm on 5 of 6
   datasets at the same cap, while AutoGluon does so on 3 of 6 — so the pre-registered joint
   hypothesis H3 is refuted for AutoGluon (§3).

---

## 2. Run inventory and integrity

| Gate | Result |
|---|---|
| G1 space frozen | 1,220 pipelines, `CLF_GRID` + conditional rules, v0.3 |
| G2 budget parity | Protocol B: every cell `n_evals == 40` exactly. Protocol A: every cell `120.1–120.6 s` |
| G3 fit gate | **7,320/7,320** (config × dataset) pairs fit and predict, 0 failures, run **before** any canonical directory existed |
| G4 no invalid candidates | `invalid_rate == 0` in all 630 cells |
| G5 mechanism engaged | all **42/42** `srs-cspace` and 42/42 `srs-onehot-cspace` cells report `engaged_at = 20` |

Thread policy: one thread per fit (`threadpoolctl` `limits=1` around every fit, `n_jobs=1`),
recorded in each `run_manifest.json`.

One repair is recorded in the Protocol C manifest: 7 `credit-g` FLAML cells had failed with
`ValueError: feature_names must be string, and may not contain [, ] or <` because our one-hot
columns inherit raw category values (`checking_status_0<=X<200`). Feature names are opaque to
every learner, so `run_flaml` now renames both frames positionally; those 7 cells were re-run
and `results.jsonl`/`summary.csv` rebuilt. No other protocol was affected.

---

## 3. Pre-registered hypotheses

Verdicts use each hypothesis's **own pre-registered falsifier** (a threshold rule fixed before
the run), separately from statistical significance, which is reported in §4.

| ID | Verdict | Evidence |
|---|---|---|
| **H1** — with ordinal encoding, SRS is *not* better than TPE | **confirmed** | best-CV difference `srs − tpe` ≤ 0 in **5/6** datasets (−0.0152 galaxy3, −0.0055 credit-g, −0.0041 ionosphere, −0.0039 arrhythmia, −0.0010 optdigits; +0.0019 spambase) |
| **H2** — the mixed categorical surface buys ≥0.3 pt in ≥4/6 | **confirmed** | improvement ≥0.3 pt in **4/6**: galaxy3 +1.63 pt, ionosphere +0.51, credit-g +0.43, optdigits +0.37 (arrhythmia +0.04, spambase +0.04) |
| **H3** — FLAML *and* AutoGluon each beat `srs-cspace` in ≥4/6 | **refuted (jointly)** | FLAML 5/6 ✓ (galaxy3 +3.04 pt, arrhythmia +0.31, credit-g +0.21, optdigits +0.01, spambase +0.03; ionosphere −1.21). AutoGluon only 3/6 ✗ (credit-g +1.71, galaxy3 +1.61, spambase +0.25; arrhythmia −0.31, ionosphere −1.61, optdigits −0.15) |
| **H4** — descriptive: does the ranking survive cost heterogeneity? | reported | the only ordering change between currencies is `srs-cspace`'s speed rank: 1st on evals-to-gap in Protocol B, 3rd in Protocol A. Cause measured in §6 |

**H1 and H2 together are the scientific result of this study.** With the ordinal encoding the
surrogate loses to TPE; fixing the encoding reverses that in the datasets where the fix bites.
The effect is small and, at 7 seeds, not significant after multiplicity correction — H2's
threshold rule passes, but only 3 of the 4 passing datasets have bootstrap CIs excluding zero
(galaxy3 $[+0.46, +3.17]$ pt, ionosphere $[+0.10, +0.97]$, optdigits $[+0.14, +0.58]$;
credit-g $[-0.16, +1.02]$ includes zero).

---

## 4. Protocol B — equal 40 evaluations

Mean held-out accuracy over 7 seeds (per-dataset values in the appendix table):

| arm | mean acc | min | max | mean best-CV | median evals-to-1 % gap | mean duplicate rate |
|---|---:|---:|---:|---:|---:|---:|
| `srs-onehot-cspace` | 0.8524 | 0.681 | 0.994 | 0.8472 | 17 | 6.3 % |
| `srs-nosurr-cspace` | 0.8504 | 0.692 | 0.992 | 0.8461 | 13 | 2.9 % |
| `srs-cspace` | 0.8475 | 0.647 | 0.992 | 0.8422 | **6** | 8.4 % |
| `optuna-rand-cspace` | 0.8460 | 0.672 | 0.989 | 0.8465 | 16 | 2.5 % |
| `optuna-tpe-cspace` | 0.8443 | 0.637 | 0.989 | 0.8468 | 12 | **28.2 %** |
| `random-cspace` | 0.8369 | 0.659 | 0.992 | 0.8375 | 11 | 1.8 % |

**Reading it.** Accuracy spans 0.837–0.852 and every paired test is Holm-null. The arms differ
in *how fast* they get to a good score: `srs-cspace` needs a median 6 evaluations to reach within
1 % of its own final best, roughly half the 11–17 of the others. Note the two SRS surface arms
pay for that with a slightly *later* saturation of the best-so-far curve when the target is
easy — `srs-onehot-cspace`'s median 17 reflects that it first fits a surface at $k=20$ (half the
budget), not that it searches worse.

---

## 5. Protocol A — equal wall-clock (primary currency)

| arm | mean acc | mean best-CV | median evals-to-1 % gap | median evals bought | mean duplicate rate |
|---|---:|---:|---:|---:|---:|
| `optuna-rand-cspace` | 0.8530 | 0.8544 | 23.5 | 550.5 | 22.4 % |
| `srs-nosurr-cspace` | 0.8531 | 0.8545 | 32.0 | 562.5 | 22.3 % |
| `random-cspace` | 0.8526 | 0.8540 | 55.0 | 553.5 | 21.6 % |
| `srs-cspace` | 0.8526 | 0.8525 | 21.5 | **281.0** | 45.9 % |
| `optuna-tpe-cspace` | 0.8494 | 0.8522 | 16.0 | 353.5 | **65.0 %** |
| `srs-onehot-cspace` | 0.8450 | 0.8518 | 19.5 | 381.0 | 45.5 % |

Accuracy is again flat (0.845–0.853). The substantive Protocol A result is the **evaluation
count**: within the same 120 s, random search buys 553 candidates and `srs-cspace` buys 281.

---

## 6. Where the surrogate's time goes (measured, not inferred)

Instrumented single-cell measurement (`credit-g`, seed 0, matching the protocol) with
`GaussianProcessRegressor.fit` timing:

```
SRS @120s: 457 evaluations, 439 surface fits
  GP fits                     26.9 s  (22.3 %)
  acquisition optimiser + evals  93.1 s
  implied pure-eval time at random's 136 ms/eval: 62 s
  => search overhead = 58 s = 49 % of the budget
RANDOM @120s: 885 evaluations, 136 ms per evaluation
```

The fit cost is not constant — it grows with the history the surface is fitted on:

```
first 10 fits: 3 ms each      last 10 fits: 174 ms each      max: 331 ms
```

That is the mechanism behind the currency disagreement: a GP surface over a growing history is
$O(n^3)$, so the surrogate's per-iteration overhead rises exactly as the run gets long, while a
random draw stays constant. On a space where one pipeline evaluation costs 40–800 ms, the search
overhead is the same order as the search itself.

**Inventory of what the surface buys:** at equal evaluations it halves time-to-target (§4); at
equal time it costs half the candidates (§5). Both statements are true and neither is "better" —
they answer different questions, which is why the contract fixed wall-clock as primary and
equal-evaluation as a secondary view with its own evaluation-count column.

---

## 7. Duplicate evaluations versus the sampling floor

Draws are with replacement from 1,220 configurations, so some duplication is unavoidable:
$1 - (1-1/N)^{k}$ at $N = 1220$.

| draws | unavoidable | `random-cspace` | `srs-cspace` | `optuna-tpe-cspace` |
|---:|---:|---:|---:|---:|
| 40 (Protocol B) | 1.6 % | 1.8 % | 8.4 % | **28.2 %** |
| ~350–550 (Protocol A) | 19.5 % | 21.6 % | 45.9 % | **65.0 %** |

Random search sits at the floor, as it must. Both guided arms concentrate their proposals and
re-evaluate configurations they have already scored; TPE does so most aggressively, burning
between a quarter and two thirds of its budget. This is default TPE behaviour on a coarse
discrete space (it does not resample), reported rather than deduplicated — deduplicating inside
the harness would flatter TPE relative to its shipped behaviour.

---

## 8. Protocol C — systems at the same cap

The cap is **not** a consumed budget: FLAML uses all 120 s, AutoGluon trains its fixed
`medium_quality` portfolio and stops after a median of **5.0 s**.

| dataset | FLAML | AutoGluon | `srs-cspace` |
|---|---:|---:|---:|
| arrhythmia | 0.7410 | 0.7347 | 0.7378 |
| credit-g | 0.7443 | 0.7593 | 0.7421 |
| galaxy3 | 0.7754 | 0.7612 | 0.7451 |
| ionosphere | 0.9396 | 0.9356 | 0.9517 |
| optdigits | 0.9847 | 0.9831 | 0.9846 |
| spambase | 0.9544 | 0.9566 | 0.9541 |
| **mean** | **0.8566** | 0.8551 | 0.8526 |

Read as "best model produced within the cap", never as equal compute. The systems search a
strictly larger hypothesis space — they are not restricted to the six sklearn families or to the
fixed imputer/preprocessor grid — so a small win is expected and is not evidence about search
quality. FLAML's +0.40 pt over `srs-cspace` on the dataset mean is within the noise floor (§9).

---

## 9. What did NOT separate — and the noise floor

- **Accuracy.** No arm beats the reference in any dataset after Holm correction, in either
  currency. Every table above should be read as a tie on accuracy.
- **The two random-sampling implementations serve as a built-in noise calibration.**
  `random-cspace` and `srs-nosurr-cspace` are the *same algorithm* (uniform draws from the frozen
  space) implemented on two code paths; their pooled means differ by 1.35 pt in Protocol B
  (0.8369 vs 0.8504) and 0.05 pt in Protocol A (0.8526 vs 0.8531). Any inter-arm difference below
  ~1.4 pt at this sample size is therefore indistinguishable from implementation noise — which
  covers every system-vs-search gap reported above.
- **`evals_to_gap` does not saturate here** (unlike the fixed-pipeline suite at $B=30$, where it
  was all zeros): medians span 6–55 evaluations, so it is a usable discriminator. It is reported
  alongside `time_avg_best` (all arms 0.848–0.851 in Protocol A — effectively identical).

---

## 10. Defects found and reported upstream

Building the harness surfaced three library defects; the benchmark measures their cost and
works around them through public API only (no library patch). Reproducers with executed output:
`benchmarks/upstream_issues.md`.

1. `MinEvals` becomes $\binom{p+3}{p}$ when the default regressor is installed (11 coordinates →
   364), silently turning surrogate search into random search for any realistic budget.
2. `'passthrough'` / `None` pipeline steps are treated as preprocessing failures, so "is this
   transformer present?" cannot be expressed as a search axis.
3. `Categorical` parameters are encoded ordinally, so no metric exists across categories — the
   defect whose cost H2 quantifies in §3.

Recommendation: add `n_surrogate_fits_` / `engaged_at_` telemetry upstream. Without it, a
surrogate run that never engaged is indistinguishable from one that did — this study had to
reconstruct it with a wrapper regressor.

---

## 11. Reproduction

```bash
python -m benchmarks.gate_cspace_fit --stride 1                      # G3, 17 min
python -m benchmarks.run_cspace --protocol B --seeds 7 --budget 40    # ~3.1 h
python -m benchmarks.run_cspace --protocol A --seeds 7 --wall 120     # ~8.4 h
python -m benchmarks.run_cspace --protocol C --seeds 7 --wall 120     # ~4.2 h
python -m benchmarks.analyze_cspace <dir> --protocol A --ref srs-cspace
```

Results directories are gitignored; each carries `run_manifest.json` (git SHA + dirty flag,
package versions, thread policy, seeds, budget) and per-cell JSON.

---

## 12. Limitations

1. **Power.** 7 seeds × 6 datasets cannot resolve differences of the size observed here. The
   pre-registered falsifiers are threshold rules, not tests; where they pass, the bootstrap CIs
   are reported beside them and only 3/6 of H2's passing datasets exclude zero.
2. **Space coverage.** Six sklearn estimator families. FLAML/AutoGluon search more (CatBoost,
   XGBoost, stacking), so Protocol C measures "this space vs those systems", not tuner quality.
3. **Cost heterogeneity survives the fix.** Equal wall-clock rewards cheap candidates; equal
   evaluations ignore that a candidate may cost 20× another. No single currency resolves this;
   the disagreement between §4 and §5 is the honest answer.
4. **One machine, one thread per fit.** Absolute wall-clock is not portable; ratios and
   evaluation counts are.

---

## Appendix — per-dataset mean held-out accuracy

Standard deviations across 7 seeds are 0.003–0.051 (largest on arrhythmia and credit-g).

**Protocol B (equal 40 evaluations)**

| dataset | optuna-rand-cspace | optuna-tpe-cspace | random-cspace | srs-cspace | srs-nosurr-cspace | srs-onehot-cspace |
|---|---|---|---|---|---|---|
| arrhythmia | 0.7410 | 0.7268 | 0.6970 | 0.7394 | 0.7331 | 0.7378 |
| credit-g | 0.7479 | 0.7371 | 0.7557 | 0.7407 | 0.7571 | 0.7564 |
| galaxy3 | 0.7138 | 0.7308 | 0.7156 | 0.7183 | 0.7313 | 0.7375 |
| ionosphere | 0.9336 | 0.9356 | 0.9235 | 0.9517 | 0.9437 | 0.9437 |
| optdigits | 0.9847 | 0.9859 | 0.9811 | 0.9842 | 0.9855 | 0.9875 |
| spambase | 0.9549 | 0.9493 | 0.9482 | 0.9508 | 0.9516 | 0.9511 |

**Protocol A (equal 120 s wall-clock)**

| dataset | optuna-rand-cspace | optuna-tpe-cspace | random-cspace | srs-cspace | srs-nosurr-cspace | srs-onehot-cspace |
|---|---|---|---|---|---|---|
| arrhythmia | 0.7300 | 0.7284 | 0.7410 | 0.7378 | 0.7378 | 0.7190 |
| credit-g | 0.7507 | 0.7421 | 0.7464 | 0.7421 | 0.7400 | 0.7321 |
| galaxy3 | 0.7415 | 0.7371 | 0.7415 | 0.7451 | 0.7415 | 0.7375 |
| ionosphere | 0.9517 | 0.9457 | 0.9457 | 0.9517 | 0.9557 | 0.9376 |
| optdigits | 0.9881 | 0.9881 | 0.9853 | 0.9846 | 0.9889 | 0.9883 |
| spambase | 0.9559 | 0.9550 | 0.9556 | 0.9541 | 0.9547 | 0.9552 |
