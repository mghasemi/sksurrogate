# Stage C1 — SHAP decision gate (R4)

**Internal working document.**

**Status:** decision **accepted** (Mehdi, 2026-11-08): *formally defer SHAP for the
v0.3 cycle; keep it as a candidate optional extra with a named reconsideration
gate.* No code, no dependency, no schema change landed with this document.

**Item:** `AI_LAYER02_REMAINING_PLAN.md` Stage C1 / R4 — one-page spec covering
supported estimators, the local-explanation use case, runtime limits,
dependency policy, and test plan; outcome recorded here and in
[`AI_LAYER_ROADMAP.md`](./AI_LAYER_ROADMAP.md) (P1.5).

## Decision

Do **not** add `shap` to the core path, to a default extra, or to a shipped
endpoint in this cycle. Record it as a *candidate optional extra*
(`sksurrogate[explain]`) that may be re-opened only when the gate below is met.
This is a deferral with rationale, not a permanent rejection.

## What SHAP would add that we do not already ship

| Capability | Shipped today | SHAP-only marginal gain |
| --- | --- | --- |
| Global attribution | Permutation importance, reference-sample SD + top-rank frequency (`api/routers/explain.py`) | Dedicated additive attributions for tree ensembles; not a new decision |
| Global response profiles | PDP (marginal) and ALE (local bins), both with non-causal labels (`docs/mltrace.rst`) | Nothing new for the global question |
| Local / row-level attribution | **Absent by design** — no row-level explanation contract (`AI_LAYER02_REMAINING_PLAN.md` D3, R9a) | The only genuinely new capability: per-row additive contributions |

Because the control plane has no row-level explanation contract, adopting SHAP
now would either (a) expose local attributions without the faithful
local-explanation contract D3 requires, or (b) add a dependency with no
consumer. Neither is acceptable under the "optional integrations stay optional"
rule.

## Estimator coverage (verified against current source)

- Preset experiments (`SKSurrogate/aml.py::_preset_config`) use only
  `RandomForest{Classifier,Regressor}` and `ExtraTrees{Classifier,Regressor}` —
  all covered by `shap.TreeExplainer`.
- Legacy `config` experiments accept arbitrary scikit-learn pipelines
  (`api/training.py`), including linear/logistic models
  (`LinearExplainer`) and unbounded others (`KernelExplainer`), so a SHAP
  integration would need a three-way explainer-selection policy plus an
  explicit fallback contract — surface we would have to define before shipping.

## Runtime budget (measured, not estimated)

Environment: project `.venv` (scikit-learn 1.9.1, NumPy 2.5.3) plus `shap`
0.54.1 installed **only** into a scratch virtualenv, never into the project —
`/home/mehdi/.hermes/profiles/coder/cache/scratch/shapenv`. Synthetic
reference of 500 rows × 6 features, `RandomForestRegressor(n_estimators=100)`:

| Explainer | Configuration | Wall time |
| --- | --- | --- |
| `TreeExplainer` | 200 rows, 6 features | 0.38 s |
| `LinearExplainer` | 200 rows, 6 features (logistic) | < 0.01 s |
| `KernelExplainer` | k-means background 50 rows, setup | 0.03 s |
| `KernelExplainer` | 100 rows explained | 2.3 s |
| `KernelExplainer` | 500 rows explained | 11.6 s |

Additivity residual was ≤ 3.3e-14 (`TreeExplainer`) and 1.1e-16
(`KernelExplainer`). The kernel path scales linearly in explained rows and
roughly quadratically in background size, so a synchronous endpoint capped at
the existing 1,000-row reference ceiling would spend ~20–25 s per request on a
small gradient-boosted or ensembled bundle — an order of magnitude above the
existing explanation endpoint. That is a contract decision (background size,
row cap, job vs. synchronous) we would have to expose and pin, not a free lunch.

## Dependency policy

- `shap` 0.54.1 install requirements (from PyPI metadata, checked
  2026-11-08): `numpy>=2`, `scipy`, `scikit-learn`, `pandas`, `tqdm>=4.27.0`,
  `packaging>20.9`, `slicer==0.0.8`, `cloudpickle`. All but `slicer` are
  already required by SKSurrogate, so the *dependency weight* is small —
  earlier notes assuming `numba`/`llvmlite` no longer hold for this release.
  The objection is therefore reproducibility and API surface, not download
  size.
- If adopted: a dedicated extra (`sksurrogate[explain]`), imported lazily on
  first use with an actionable `ImportError` naming the extra; never imported
  by `SKSurrogate/__init__.py`; benchmark/Docker images unchanged.
- Determinism: `KernelExplainer` approximates with a background sample and its
  internal sampling; we would have to expose reference provenance and a seed
  exactly as the current explanation contract does. `TreeExplainer` is
  deterministic and additive.

## Test plan (only if the gate below opens)

1. Seeded determinism: same bundle/seed → identical SHAP matrix (tree) and a
   recorded, reproducible background sample (kernel).
2. Additivity: `sum(shap_values) + expected_value ≈ predict` within 1e-6 for
   each shipped explainer path, on a synthetic fixture with a known linear
   response.
3. Dependency-absent path: endpoint raises the documented 422/install hint
   when the extra is missing (mirrors the existing `[benchmark]` pattern).
4. Bounded runtime: assert the row cap and background-size cap reject
   over-budget requests *before* computing.
5. Non-causal labelling: response carries `category: "model_attribution"`,
   `causal_effect: false`, and limitations alongside the values, matching the
   existing explanation payload contract.

## Reconsideration gate

Re-open only when **all** hold:

1. D3 (row-level explanation contract, R9a) is approved — the faithful local
   explanation semantics exist and SHAP would implement them rather than
   invent them.
2. At least one real adopter names estimator-specific local attributions that
   permutation importance, PDP, and ALE cannot serve.
3. A bounded request shape (row cap, background provenance, seed, tree/kernel
   selection) is specified against the measured runtime above.

Until then, `AI_LAYER_ROADMAP.md` P1.5 keeps "SHAP deferred" as the recorded
outcome, and R4 in `AI_LAYER02_REMAINING_PLAN.md` stays closed for this cycle
with this document as its rationale.
