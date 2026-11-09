# Stage D3 — Row-level explanation contract (R9a)

**Internal working document.**

**Status:** contract **approved 2026-11-08** (Mehdi); bounded implementation and
tests landed the same day in `api/explanations.py`,
`api/routers/explain.py`, the Explain UI, and `docs/mltrace.rst`. This document
is the method/API contract the implementation follows; any change to the
semantics below requires a new decision here first.

**Item:** `AI_LAYER02_REMAINING_PLAN.md` Stage D3 / R9a — "faithful
local-explanation method and its API contract first; do not label an ICE
difference as a feature attribution; optional LLM summary only after the
deterministic layer exists."

## Decision

Ship exactly one deterministic local-explanation layer: **ordered-occlusion
chain decomposition**. It is exact, estimator-agnostic, reference-realistic
(every evaluated point differs from the observed row only in the features
already restored), and it discloses its one arbitrary choice — feature order —
as a measured quantity. No sampling-approximation method (LIME/SHAP-style) is
mixed into this layer, and no language-model summary is generated.

## Candidate methods, assessed and rejected for this layer

| Candidate | Why it is not this layer's method |
| --- | --- |
| Single-feature occlusion deltas ("drop feature j, take the difference") | Not additive when features interact: the leftover residual gets silently attributed to "everything else", which reads like an attribution but is a modelling artifact |
| ICE slope or ICE range at the row | A response profile, not a contribution; the plan explicitly forbids labelling an ICE difference as a feature attribution |
| LIME-style local surrogate | Sampling-based, unstable across seeds, and offers no exactness property for the fitted model |
| SHAP | Deferred by the Stage C1 decision (`AI_LAYER02_C1_SHAP_DECISION.md`); also needs the row-level contract this document creates |
| Counterfactual optimisation | A different question (what change would flip the prediction), and unbounded without constraints |

## Method

For a row $x$, baseline vector $b$, and a feature order
$\pi=(\pi_1,\dots,\pi_n)$:

$$
f(x)-f(b)=\sum_{j=1}^{n}\Bigl[f\bigl(x_{\pi_{\le j}}\cup b_{\pi_{>j}}\bigr)-f\bigl(x_{\pi_{<j}}\cup b_{\pi_{>j}}\bigr)\Bigr].
$$

Each bracket is the model response for the "hybrid" row in which the first $j$
features have been restored from $x$ and the rest remain at $b$; the sum
telescopes exactly for **any** estimator, with no additivity assumption. The
implementation evaluates the chain with direct model calls, reports each
feature's step contribution, and enforces/reports

$$
\bigl(\text{baseline prediction}\bigr)+\sum_j c_j \;=\; f(\text{explained row})
$$

within a recorded tolerance (`reconstruction.max_abs_residual`; observed ≈
$10^{-16}$ on the test fixtures).

**Properties, stated plainly**

- *Exactness* — the reconstruction is exact by construction; a nonzero residual
  is floating-point noise, not "unexplained variance".
- *Order dependence* — with interactions or correlated features, the same row
  and baseline give different per-feature credits under different orders. With
  $K$ distinct seeded orders the endpoint reports per-feature mean, sample SD,
  min, and max. The SD is disclosed order-dependence, **not** a confidence
  interval, and the order-averaged value is only the symmetric (Shapley) value
  when *all* $n!$ orders are used — we never claim that unless it is literally
  true.
- *Baseline dependence* — a different baseline row or reference partition
  changes every contribution while leaving the reconstruction exact. The
  baseline descriptor is part of the response.
- *Subset scope* — when only some features are explained, the reconstruction
  describes the **hybrid row** (listed features at observed values, the rest at
  baseline) and the response marks `explanation_scope = "subset_hybrid"`. It is
  never presented as a decomposition of the full prediction.
- *No causality* — contributions are model-response differences for this fitted
  bundle. Correlated features can share or mask credit, and the decomposition
  does not restrict the row to a valid joint distribution beyond the baseline
  swap.

## Baseline contract

| Mode | Values | Recorded in the response |
| --- | --- | --- |
| `reference_mean` (default) | finite mean of each numeric feature and the most frequent level of each non-numeric feature over the reference partition (bounded random subsample, `reference_max_rows` 20–1000, seeded) | partition, source/used rows, sampling mode, per-feature `value_sources` (`finite_mean` / `most_frequent`) |
| `row` | an explicitly chosen stored row of a partition (`baseline_row_index`) | partition, row index, `value_sources = "row"`, `used_rows = 1`, `sampling = "explicit_row"` |

The baseline partition must contain every explained feature; missing columns are
a 400 naming them. A partition whose numeric column has no finite values cannot
produce a mean baseline (400).

## Scalar response contract

Identical to the accumulated-local-effects contract: regression uses `predict`;
binary classification uses the positive-class probability
(`predict_proba[:, 1]`) with a `decision_function` fallback, and the payload
labels the scalar used. Multiclass bundles are rejected (422) because one
contribution vector would have to choose a class arbitrarily. Contributions on
different response scales are not comparable.

## API contract

`POST /api/evaluation/{task}/{version}/explain/row`

| Field | Bound | Meaning |
| --- | --- | --- |
| `partition` | stored partition | rows the explained row is drawn from |
| `row_index` | `0 ≤ i < row_count` | row to explain (0-based) |
| `features` | ≤ 24 names, default all | explained features; a subset switches to hybrid scope |
| `baseline` | `reference_mean` \| `row` | baseline mode |
| `baseline_partition` | stored partition, defaults to `partition` | reference partition for the baseline |
| `baseline_row_index` | required for `baseline="row"` | baseline row |
| `orders` | 1–10 | distinct seeded feature orders |
| `reference_max_rows` | 20–1000 | cap for the reference-mean subsample |
| `random_state` | int | controls subsampling and the order draw |

Errors (400): row out of range, unknown feature, more than 24 features, more
orders than distinct permutations exist (`at most n! distinct orders`), missing
or out-of-range baseline row, baseline partition missing features. 422:
multiclass classifier, non-sklearn estimator.

Response: `row_index`, `row_count`, `response` (scalar label), `reference_sample`,
`baseline` (mode/partition/row/prediction/values/value_sources), `prediction`
(full row), `explained_prediction` (hybrid row), `explanation_scope`,
`explained_features`, `settings`, `orders`, `contributions`
(feature/mean/sample_std/min/max sorted by |mean|), `reconstruction`
(baseline_prediction, explained_prediction, sum_contributions, target_change,
max_abs_residual, tolerance, exact_by_construction), and `interpretation`
(`category: "model_attribution"`, `method: "ordered_occlusion_chain"`,
`causal_effect: false`, limitations). Same seed ⇒ identical payload.

## UI contract

Explain page → **Row-level explanation** card: 0-based row index, 1/5/10 orders,
a contribution bar chart, a table of mean/SD/min/max, baseline and row
predictions, the reconstruction residual with its tolerance, and the limitation
list. It must state order-dependence explicitly and must not label the chart
"feature importance" or imply causality.

## Test plan (implemented)

1. Kernel exactness and order symmetry on an interaction model
   $f=(x_0-x_1)^2$: both orders reconstruct $f(x)-f(b)$ exactly and split the
   interaction differently; distinct-permutation bounds raise.
2. End-to-end recovery on a plain linear bundle: contribution
   $c_j=\beta_j(x_j-\bar x_j)$ to 1e-8, exact reconstruction, ordering by
   absolute contribution, same-seed determinism.
3. Non-linear preset bundle: K = 5 orders produce non-zero order spread,
   `min ≤ mean ≤ max`, exact reconstruction.
4. Subset scope: `features=["a"]` reports `subset_hybrid` and reconstructs the
   hybrid prediction, not the full one.
5. Classifier scale: probability-labelled response with row-baseline mode,
   `explicit_row` sampling metadata.
6. Error surface: row bounds, unknown feature, order bound, missing/out-of-range
   baseline row.

## Explicitly deferred

Inline row payloads (serving-time explanations) — the stored partition is the
provenance anchor; ICE local profiles — separate response-profile object, not an
attribution; exact Shapley values for all $n!$ orders; LLM summaries — D2 stays
gated on post-launch API stability and a bounded use case.
