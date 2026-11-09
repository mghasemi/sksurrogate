# Stage D4 — Acquisition-extension readiness assessment (R7)

**Internal working document.**

**Status:** **assessment complete 2026-11-08**; D4 remains **gated**. This is an
analysis-only deliverable: no production acquisition code, endpoint, or UI change
ships from it. Mehdi waived the "at least one real expensive-experiment user"
entry gate for the assessment only — the D4 exit criterion still requires a real
user and a representative equal-budget benchmark.

**Item:** `AI_LAYER02_REMAINING_PLAN.md` Stage D4 / R7 — expected/probability
improvement, confidence-bound and noise-aware models, kernel selection,
replication, constraints, batch/multi-objective and multi-fidelity design,
assessed against the current bounded GP-LCB implementation.

## What was assessed

Harness: `benchmarks/design_readiness.py` (report schema
`sksurrogate.design-readiness.v1`), run over the three synthetic 2-D objectives
already used by `benchmarks/doe_active_learning.py` (`trig`, `mixed`,
`rational` on $[-1,1]^2$).

Protocol: 5 seeds, initial Latin-hypercube design of 6, total evaluation budget
20 (14 sequential proposals per run), 256-candidate pool per step, Gaussian
observation noise $\sigma \in \{0, 0.05\}$, paired step seeds across policies.
The shipped policy is exercised through `SKSurrogate.design.suggest_next_point`
(LCB, $\beta=2$); EI and PI rank the *same* seeded candidate pool with the same
Matern-5/2, fixed-unit-length-scale surrogate. The noise-aware variant uses the
same kernel **plus** a `WhiteKernel` whose noise level is optimized while the
length scale stays fixed.

150 runs, 3,000 per-evaluation records; wall time ≈ 3.7 s.

## Results (descriptive, synthetic)

Mean final regret (best true value evaluated minus dense-grid optimum; lower is
better), averaged over the three objectives and five seeds:

| Policy | $\sigma=0$ | $\sigma=0.05$ |
| --- | ---: | ---: |
| uniform random | 0.1106 | 0.1106 |
| **LCB (shipped)** | **0.00443** | **0.01537** |
| EI | 0.00394 | **0.00843** |
| PI | **0.00355** | 0.02050 |
| LCB + WhiteKernel (noise-aware) | 0.05366 | 0.05673 |

Mean regret AUC (area under the best-so-far regret curve, same runs):

| Policy | $\sigma=0$ | $\sigma=0.05$ |
| --- | ---: | ---: |
| uniform random | 0.1926 | 0.1926 |
| **LCB (shipped)** | **0.1186** | 0.1232 |
| EI | 0.1222 | **0.1223** |
| PI | 0.1209 | 0.1326 |
| LCB + WhiteKernel | 0.1400 | 0.1424 |

Paired outcomes vs the shipped LCB, per objective and noise setting (wins /
losses / ties over 5 seeds):

| Policy | trig, 0 | mixed, 0 | rational, 0 | trig, 0.05 | mixed, 0.05 | rational, 0.05 |
| --- | --- | --- | --- | --- | --- | --- |
| EI | 0/0/5 | 0/0/5 | 1/0/4 | 2/1/2 | 2/2/1 | 2/1/2 |
| PI | 1/0/4 | 0/0/5 | 1/0/4 | 2/0/3 | 1/4/0 | 1/4/0 |
| LCB + WhiteKernel | 0/5/0 | 0/0/5 | 0/2/3 | 0/4/1 | 4/0/1 | 2/3/0 |

## Honest read

1. **The surrogate already does the heavy lifting.** Every GP policy beats
   uniform random by roughly an order of magnitude in mean final regret
   (0.004–0.02 vs 0.11). The acquisition rule is a second-order choice on this
   evidence.
2. **In the noise-free setting, acquisition changes buy nothing measurable.**
   EI and PI are statistically indistinguishable from LCB: EI ties on 14 of 15
   paired comparisons (one win), PI on 13 of 15 (two wins), and mean/AUC
   differences are within seed-level noise.
3. **Under synthetic observation noise, EI has a weak, suggestive edge.** Mean
   final regret 0.0084 vs LCB 0.0154 (≈45% lower) with 6 wins / 4 losses / 5 ties
   across the paired runs; the AUC difference is negligible (0.1223 vs 0.1232).
   With five seeds this is not evidence of a robust improvement — it is a
   candidate for a larger, user-driven benchmark, not a ship decision.
4. **PI is not a candidate under noise.** Mean regret is 33% worse than LCB and
   it loses 8 of 12 non-tie pairs; the PI objective saturates, so it exploits
   less than EI.
5. **The noise-aware model as specified is a regression, not an upgrade.** With
   the length scale fixed at 1, the `WhiteKernel` absorbs genuine signal as
   noise: it loses 5/5 on the oscillatory `trig` objective even with clean data
   and is worse overall in both settings. It helps only `mixed` + noise
   (4/0/1). A usable noise-aware model needs joint length-scale/noise
   optimization (and a hyperparameter budget) before it can even be compared
   fairly.
6. **Not assessed, by design:** kernel selection, replication/blocking,
   constraints, batch design, multi-objective and multi-fidelity. Each needs a
   real cost/feasibility model and a user; assessing them on unit-cost synthetic
   objectives would be a proxy for a proxy.

## Recommendation

- **Do not ship EI/PI/noise-aware acquisitions now.** They do not meet the D4
  exit criterion, and the only non-neutral signal — EI under synthetic noise —
  is too weak at this sample size to justify API surface and maintenance.
- **Keep D4 gated** on a real expensive-experiment user and a representative
  protocol (cost model, noise level, batch/constraint requirements). When one
  appears, the first bounded prototype to evaluate is **EI for noisy scalar
  objectives**, benchmarked with this harness extended to the user's regime.
- **Treat the noise-aware result as a design warning:** kernel noise adaptation
  is not a drop-in feature under a fixed-degree-of-freedom surrogate; a future
  attempt must co-adapt length scales and be evaluated on the existing clean
  tasks before touching production.

## Reproduce

```console
python -m benchmarks.design_readiness --out DIR            # default protocol
python -m benchmarks.design_readiness --seeds 2 --budget 8 --initial 4 \
    --candidates 64 --families trig mixed --noise 0 0.05    # smoke
```

The report contains the full per-evaluation records (`x`, `y_true`,
`y_observed`, acquisition metadata), run summaries, and paired-vs-LCB counts.
Tests: `tests/test_benchmark_design_readiness.py` (9 tests) pin the equal-budget
record shape, strategy validation, seeded determinism, observation-noise
recording, and CLI report contract. Contract: `benchmarks/DESIGN.md` §15.

## Limitations

Synthetic objectives and synthetic Gaussian noise only; fixed surrogate family
and fixed $\beta=2$; five seeds; no confidence intervals or significance tests;
no real experimental cost, feasibility, or safety model. The fixed length scale
was previously selected on these same objective families, so these are
development numbers, not an independent estimate of generalization.
