# SKSurrogate AI-Layer Follow-up — Remaining Work Plan

**Internal working document.**

## Purpose and status

This document re-baselines the unfinished work from
[`AI_LAYER02_FOLLOWUP_PLAN.md`](./AI_LAYER02_FOLLOWUP_PLAN.md) after a fresh
verification pass against the current source on branch `ailayer` (2026-11-08).
Where the follow-up plan tracked review recommendations as they were being
implemented, this document tracks only what is *actually still missing*, why
it has not been implemented yet, and in which order it should be picked up.

**Status (reconciled 2026-11-08):** Phases 0–3 and 6 of the follow-up plan are
complete. Phase 4 is complete except R1 (B1: the tier-1 `reference` kind landed
2026-11-08 with the post-launch gate waived; `file_artifact` is still gated on
adopter feedback) and R2 (B3: orchestration design gated on demonstrated
usage). Phase 5 is complete: C1
formally deferred SHAP with a measured decision record, and C2/C3 shipped ALE
plus PDP/ALE reference-subset stability summaries. Phase 7 is partly landed: D1
(controlled DGPs with exact-truth validation), D2 (opt-in cited project-note
synthesis, post-launch gate waived by Mehdi), and D3 (ordered-occlusion
row-level explanations) are done; D4 remains gated on a real
expensive-experiment user (its analysis-only readiness assessment is complete),
and D5 remains gated on a concrete document set. The v0.3.0
public launch target was **amended 2026-11-09 from Nov 17/18 to December** (working
target Tue/Wed ≈ Dec 15/16, gated on the maintainer's real-dataset testing — see
LAUNCH_PLAN decision 5) and is tracked in
[`LAUNCH_PLAN.md`](./LAUNCH_PLAN.md) — nothing in Stages B–D below blocks the
cut (see LAUNCH_PLAN §6).

Legend: `[ ]` open, `[~]` gated/in progress with a named blocker, `[>]`
deferred or rejected with rationale.

## Verified current state (what exists today)

| Area | Built and evidenced by | Where the remainder lives in this plan |
| --- | --- | --- |
| Phase 4 research records | Append-only hypothesis/analysis/finding/conclusion records; bounded hypothesis status lifecycle with append-only history; question links; evidence attachment (20-reference cap); `tests/test_projects_endpoints.py` (872 lines, 20 test functions) | Stage B |
| Phase 4 evidence snapshots | Fingerprint-based snapshots (`sha256` of the serialized artifact reference, schema `sksurrogate.project-artifact-snapshot.v1`, `api/routers/projects.py`); statuses `current` / `changed` / `missing` / `unverified`; idempotent `POST /api/projects/{id}/artifacts`; automatic links for registered partitions and completed sensitivity/experiment/pipeline jobs (failed/unfinished not linked); Inspect-job navigation from projects and research records | Stage B |
| Phase 5 explanations | Interpretation categories + non-causal caveats in API/UI/docs; Feature Analysis ↔ Explain cross-linking with "not directly comparable" warnings; opt-in permutation-importance sample sensitivity over 2–5 distinct reference subsets (descriptive, not CIs); seeded repeatability tests; deterministic perfect-correlation fixture showing marginal PDP leaves observed support; Stage C ships ALE (`POST .../ale`) and reference-subset PDP/ALE stability blocks; **Stage D3 adds exact ordered-occlusion row explanations (`POST .../explain/row`) with disclosed order-dependence** | Stage C/D |
| ALE status | Implemented 2026-11-08 in `api/explanations.py` + `POST /api/evaluation/{task}/{version}/ale`, with the typed contract in `docs/mltrace.rst` (numeric-only bins, duplicate-edge collapse, sparse-bin merging, labelled scalar response, subset stability) | Stage C (C2) — closed |
| Stage D synthesis | Opt-in LLM-assisted cited-note drafts (`api/synthesis.py`, `api/synthesis_facts.py`, `POST /api/synthesis/projects/{id}`), aggregate-only facts `S1–S5`, enforced `[S#]` citations, generated-note provenance in `api/routers/projects.py`, Projects UI card; contract in `AI_LAYER02_D2_LLM_SYNTHESIS_SCOPE.md`; `tests/test_api_synthesis.py` (7 tests, stub provider) | Stage D — closed |
| Phase 7 research bets | D1 harness (`benchmarks/dgps.py`), D2 synthesis, and D3 row-level explanations are shipped; D4 acquisition extensions and D5 document extraction have no implementation | Stage D (D4/D5) |

## What actually remains

Numbered items R1–R9 are the complete remainder of the follow-up plan. Each is
mapped to a work stage below and tracked in Vikunja project 40 ("SKSurrogate
public launch (v0.3)") under **Phase 4** tasks P4.1–P4.9.

| # | Item | Follow-up origin | Blocker kind | Unblocked when |
| --- | --- | --- | --- | --- |
| R1 | Broader research-artifact ingestion beyond dataset partitions, bundles, and jobs | Phase 4 `[~]` | Scope (no bounded user need defined) | **Tier-1 `reference` implemented 2026-11-08** (API/UI/tests/docs; gate waived); `file_artifact` remains gated on post-launch adopter feedback naming a file kind |
| R2 | Research-workspace orchestration beyond the current stage checklist | Phase 4 `[~]` | Scope + design | Guided workflow demonstrated by real usage (B3) |
| R3 | Artifact recovery/versioning semantics for evidence snapshots | Phase 4 `[x]` | Storage decision | **Closed 2026-11-08** — detect-and-report only accepted; per-status contract in `docs/research-projects.rst`, pinning tests landed (B2 done); upgrade path recorded, re-open on real adopter demand |
| R4 | SHAP revisit as optional integration | Phase 5 `[x]` | **Closed 2026-11-08** — formally deferred for this cycle (no dependency); decision + measured runtime + gate in `AI_LAYER02_C1_SHAP_DECISION.md` | D3's row-level contract now exists (2026-11-08); re-open only when an adopter names the need and a bounded request shape (row cap, background provenance, seed, tree/kernel selection) is specified |
| R5 | ALE contract + bounded implementation; PDP/ICE stability diagnostics | Phase 5 `[x]` | **Closed 2026-11-08** — ALE shipped in the API/UI layer with the contract in `docs/mltrace.rst`; PDP stability summaries shipped; ICE stability scoped out explicitly (ICE is not part of the API explanation contract) | Seeded tests pass (`tests/test_api_explanations.py`); ALE validated against exact DGP truth in `benchmarks/dgps.py` |
| R6 | Controlled synthetic DGPs with known ground truth | Phase 7 `[x]` | **Closed 2026-11-08 (D1)** — ALE and average PDP/subset-stability validated against exact known responses at floating-point precision; the harness stays the validation slot for future methods | Future Stage B/C methods are validated there first as they land |
| R7 | Experiment-design acquisition extensions (EI/PI, noise-aware models, batch/multi-objective, …) | Phase 7 `[~]` | **Analysis-only readiness assessment complete 2026-11-08** (`benchmarks/design_readiness.py`, `AI_LAYER02_D4_READINESS_ASSESSMENT.md`); only the real-user gate remains for any implementation | At least one real expensive-experiment user; reproducible equal-budget benchmark in that regime with an honest read (D4 exit) |
| R8 | LLM-assisted project synthesis | Phase 7 `[x]` | **Closed 2026-11-08 (D2)** — bounded opt-in cited-note synthesis implemented in the API/UI layer; contract + data-handling review in `AI_LAYER02_D2_LLM_SYNTHESIS_SCOPE.md`; post-launch gate explicitly waived by Mehdi | Broad LLM assistance remains a separate spike with its own entry gate |
| R9 | Row-level explanations; document-to-tabular extraction | Phase 7 `[~]` | **Row-level (R9a/D3) closed 2026-11-08** with an exact ordered-occlusion contract + API/UI; extraction (R9b/D5) stays gated on a concrete document set | Extraction: schema-driven provenance + human-validation design once demand exists |

## Blockers: why these items were not implemented yet

1. ~~**Decision blockers, by design.**~~ **Resolved 2026-11-08.** SHAP (R4) was
   decided — formal deferral for this cycle, no dependency, with measured
   runtime and a named reconsideration gate (`AI_LAYER02_C1_SHAP_DECISION.md`).
   The LLM layer (R8) was decided and implemented as a bounded opt-in cited-note
   synthesis layer, with the post-launch stability gate explicitly waived by
   Mehdi and a data-handling review recorded in
   `AI_LAYER02_D2_LLM_SYNTHESIS_SCOPE.md`. The working principle held: optional
   integrations stayed optional and the core path stayed dependency-light.
   LAUNCH_PLAN §6 still states that announcement scope must not wait on
   Phase 4/5 completion.

2. ~~**Method-contract gaps (R5).**~~ **Resolved 2026-11-08 (C2).** The ALE
   contract now exists in `docs/mltrace.rst` (numeric-only bins with a recorded
   reason for skipped nominal categoricals, duplicate-edge collapse,
   deterministic sparse-bin merging, labelled scalar response, subset
   stability) and the implementation is validated against exact known response
   curves in the DGP harness. ALE remains documented as a *supplement* to PDP
   that does not solve sparse support or conditional explanation automatically.

3. ~~**Storage semantics undecided (R3).**~~ **Resolved 2026-11-08 (B2).**
   "Detect and report only" was accepted: snapshots remain fingerprint-based
   (`sha256` of the serialized artifact reference), recovery is explicitly not
   offered, and `docs/research-projects.rst` carries the per-status behavior
   contract with pinning tests. The recorded upgrade path (per-kind `content`
   under snapshot schema v2, size cap + accounting) stays open pending a real
   adopter request.

4. **Scope undefined for R1/R2.** "Research artifact" today means exactly
   dataset partitions, model bundles, and jobs — everything else is out of the
   linkable set by construction. For R1 the ambiguity is now bounded: the
   scoping document accepted a typed superset `{reference, file_artifact}`, but
   implementation still waits on post-launch adopter feedback naming a kind.
   For R2 (B3), what users need beyond the current checklist (guided stage
   transitions? reruns from records?) is unknown until real usage exists;
   building a guided workspace now risks solving the wrong problem.

5. ~~**Two concrete code fixes gate all Phase 7 benchmarking (R7).**~~
   **Resolved 2026-11-08 (A1+A2).** `benchmarks/datasets.py` now checks the warm
   CSV cache before importing `openml` (regression-tested in slim environments),
   and benchmark run records are schema v2 with the full per-evaluation
   candidate configuration, while readers stay tolerant of legacy v1 archives.
   With the code prerequisite met, R7/D4 has exactly one remaining gate: at
   least one real expensive-experiment user.

6. ~~**Synthetic-data gap (R6)**~~ **Resolved 2026-11-08 (D1).**
   `benchmarks/dgps.py` provides four seeded DGPs with exact known truth, a
   method panel, a schema-v1 report, and exact-response validation for the
   landed ALE and PDP/stability kernels (errors ≈1e-15). It remains the
   validation slot for future methods as they land.

7. **Remaining gates (the complete list).** (a) B1: post-launch adopter
   feedback that names an artifact kind beyond partitions/bundles/jobs; (b) B3:
   demonstrated usage of the project checklist that justifies guided
   transitions; (c) D4/R7: one real expensive-experiment user; (d) D5/R9b: a
   concrete document set plus demand. No code is blocked on a decision any more
   — only on evidence that does not exist yet.

## Work plan

### Stage A — Pre-launch quick fixes that unblock everything else (~½ day total; optional before the cut)

- [x] **A1** Fix `openml` import order in `benchmarks/datasets.py`: check the
  warm cache first, lazy-import only on a cache miss (keep the actionable
  `ImportError`). Add a regression test that a cached CSV loads in an
  environment without `openml` installed. *(Unblocks R7; safe to land before
  or after v0.3.0 — not launch-blocking. Done 2026-11-08: cache checked first,
  lazy import on miss only; regression-tested in `tests/test_benchmark_datasets.py`, verified end-to-end with the import blocked.)*
- [x] **A2** Record per-evaluation hyperparameters (and the full candidate
  config) in benchmark runner output; version the JSON schema and keep archive
  readers tolerant of the legacy field-less records. *(Prerequisite for any
  acquisition-function comparison against existing or future archives.)*

### Stage B — Close out Phase 4 research workspace (post-launch, after first-adopter feedback)

- [~] **B1** Artifact-ingestion scoping — **scoping/decision accepted and
  tier-1 implemented 2026-11-08.**
  [`AI_LAYER02_B1_ARTIFACT_INGESTION_SCOPE.md`](./AI_LAYER02_B1_ARTIFACT_INGESTION_SCOPE.md)
  adopts the bounded typed superset `{dataset_partition, model_bundle, job} ∪
  {reference, file_artifact}`. **Tier-1 `reference` shipped:** `{kind, url,
  title?, doi?, label?}` with absolute-http(s) validation (≤2,048 chars, never
  fetched), URL dedupe, v2 field-only snapshot (`content_embedded: false`),
  always-`unverified` summary status, evidence support, removal protection,
  backward-compatible old-kind shapes, Projects UI form and external-link
  rendering, and tests (roundtrip/validation, evidence/removal, legacy shapes,
  HTTP surface). Remaining gate: `file_artifact` waits on post-launch adopter
  feedback naming a specific in-root file kind (its B2 dependency is settled:
  detect-only `{size_bytes, sha256}`, no embedded bytes). *(R1)*
- [x] **B2** Artifact-recovery decision + bounded implementation — drafted 2026-11-08 in
  [`AI_LAYER02_B2_ARTIFACT_RECOVERY.md`](./AI_LAYER02_B2_ARTIFACT_RECOVERY.md) and **decision accepted by Mehdi the same day**: commit to "detect and report only" (no content embedding; v1 behavior unchanged). §3 landed 2026-11-08: per-status behavior contract in `docs/research-projects.rst` + pinning tests for `unverified` legacy records and no-rebaseline-on-edit (`tests/test_projects_endpoints.py`). Verified: full suite 302 passed / 32 subtests in a clean environment; Sphinx HTML build clean. Legacy v1 records remain readable by construction; the recorded upgrade path (per-kind `content` under snapshot schema v2 with cap + accounting) stays open pending real adopter demand. Tracked-file changes left uncommitted for review. *(R3)*
- [ ] **B3** Orchestration design from launch feedback: guided stage
  transitions (Question → … → Conclusion) vs the current checklist; bounded UI
  work only against demonstrated needs, reusing existing pages rather than a
  parallel workspace. *(R2)*
- [x] **B4** SQLite/transactional-store migration — rejected with rationale in
  Phase 4 (`[>]`): single-process project writes, bounded file lookups; revisit
  only when multi-worker writes or query needs become actual requirements.

### Stage C — Complete explanations (post-launch)

- [x] **C1** SHAP decision gate — **done 2026-11-08**, decision accepted:
  formal deferral for the v0.3 cycle, no dependency. One-page record
  [`AI_LAYER02_C1_SHAP_DECISION.md`](./AI_LAYER02_C1_SHAP_DECISION.md) covers
  preset/legacy estimator coverage (presets are RF/ExtraTrees only), the
  local-attribution use case (which no shipped endpoint serves — it is D3/R9a),
  measured runtime (scratch-venv `shap` 0.54.1: TreeExplainer 0.38 s / 200 rows;
  KernelExplainer 2.3 s / 100 rows, 11.6 s / 500 rows with a 50-row background),
  dependency policy (0.54.1 needs `slicer`/`cloudpickle`/`tqdm`, no `numba`), and
  a five-part test plan if it is ever adopted. Reconsideration gate: D3's
  row-level contract + a named adopter need + a bounded request shape. Roadmap
  P1.5 updated. *(R4 closed)*
- [x] **C2** ALE contract + bounded implementation — **done 2026-11-08.** New
  pure-NumPy module `api/explanations.py` (no `SKSurrogate` core change, no new
  dependency) and endpoint `POST /api/evaluation/{task}/{version}/ale`:
  numeric-only equal-frequency quantile bins with duplicate-edge collapse,
  deterministic sparse-bin merging (`min_bin_fraction` 0.01), per-curve
  excluded-row counts, feature selection by explicit list or top-k permutation
  importance, labelled scalar response (`predict` / positive-class probability /
  decision value; multiclass rejected 422), and descriptive 2–5-subset bin
  stability. Seeded tests in `tests/test_api_explanations.py` include the Phase 5
  perfect-correlation fixture (ALE stays on support where marginal PDP exceeds
  1.0 off-support). Contract replaces the "not exposed" note in
  `docs/mltrace.rst`; Explain UI card added; ALE validated against exact known
  DGP response curves (`ale_true_response_max_abs_error` ≈ 1e-15). *(R5 closed)*
- [x] **C3** PDP reference-subset stability — **done 2026-11-08**, scoped to PDP.
  Explain responses carry an additive `partial_dependence_stability` block
  (per-grid mean / sample SD / min / max plus per-subset means, same canonical
  grid, direct model calls) when 3–5 reference samples are requested; UI renders
  dashed mean ± SD lines. Documented as descriptive, not a confidence band.
  ICE stability is **not** implemented and the reason is recorded in
  `docs/mltrace.rst`: individual ICE curves are a plotting feature of
  `plot_partial_dependence`, not part of the API explanation contract, so there
  is no bounded sample to summarize. *(R5)*

### Stage D — Gated research bets (each with explicit entry and exit criteria)

- [x] **D1** Controlled synthetic DGPs — **done 2026-11-08.** Foundation:
  `benchmarks/dgps.py`, four seeded DGPs with exact known coefficients/noise
  (`interactions`, `correlated`, `heteroscedastic`, `shift`), method panel
  (ridge/random-forest/kernel-ridge, fresh per seed), and `validate()` writing a
  schema-v1 report; contract in `benchmarks/DESIGN.md` §14; 17 tests in
  `tests/test_benchmark_dgps.py`. Exit criterion now met for the landed Stage C
  methods: `_ale_truth_checks` matches the ALE kernel to the exact noise-free
  response (~1e-15), and the new `_pdp_truth_checks` matches average partial
  dependence and its subset summary — including the known `x2**2` term — with
  `pdp_true_response_max_abs_error` ≈ 1e-15 and additive-subset curves differing
  only by their reference-mean offset (shape deviation ≈ 1e-15). The harness
  remains the validation slot for future Stage B/C methods as they land. *(R6)*
- [x] **D2** LLM-assisted project synthesis — **done 2026-11-08**, with the
  post-launch stability gate explicitly waived by Mehdi. Contract and
  data-handling review in
  [`AI_LAYER02_D2_LLM_SYNTHESIS_SCOPE.md`](./AI_LAYER02_D2_LLM_SYNTHESIS_SCOPE.md):
  a bounded, opt-in draft of one cited project note from computed artifacts.
  Provider is a user-supplied OpenAI-compatible chat-completions endpoint over
  stdlib HTTP (`SKSURROGATE_LLM_BASE_URL` / `_MODEL` / optional `_API_KEY`,
  `_TIMEOUT`, `_MAX_TOKENS`), disabled by default with a 503 and no outbound
  request; no new dependency and no core change. Facts are aggregates only
  (`S1` profile, optional `S2` run, `S3` bundle, optional `S4` explanations,
  optional `S5` stored alerts) — never raw rows, paths, model objects, or
  credentials. Drafts must cite supplied ids; unknown citations, uncited,
  empty, or over-length drafts fail the job by name. `POST
  /api/synthesis/projects/{id}` runs as a `project-synthesis` job and does not
  save anything; `POST /api/projects/{id}/notes` persists generated notes only
  with model id, matching citations, and a bounded computed-facts payload
  (append-only, backward compatible with human notes). Projects UI gained a
  **Draft synthesis** card (provider status, task/bundle/job selectors from
  linked artifacts, explanation toggle, generated-interpretation rendering
  with citation chips and a computed-facts table, explicit "Add as note").
  Tests: `tests/test_api_synthesis.py` (7 tests, local stub provider, no
  network) covering disabled config, cited happy path with aggregate-only
  prompt assertions, citation rejection, provider error mapping, and note
  provenance. *(R8)*
- [x] **D3** Row-level explanations — **done 2026-11-08**, contract approved and
  implemented. Contract in
  [`AI_LAYER02_D3_ROW_LEVEL_CONTRACT.md`](./AI_LAYER02_D3_ROW_LEVEL_CONTRACT.md):
  ordered-occlusion chain decomposition only; single-feature occlusion deltas,
  ICE-as-attribution, LIME-style surrogates, SHAP (deferred), and counterfactual
  search are recorded as rejected for this layer. `POST
  /api/evaluation/{task}/{version}/explain/row` credits each feature with the
  model-response change when restored from a reference-mean or explicit-row
  baseline in a stated order; the chain telescopes exactly (residual and
  tolerance reported), `orders` 1–10 draws distinct seeded permutations with
  per-feature mean/sample SD/min/max as disclosed order-dependence, feature
  subsets report `subset_hybrid` scope, and binary classifiers use the labelled
  scalar response (multiclass 422). Explain UI card added. Seeded tests in
  `tests/test_api_explanations.py`: interaction exactness/order symmetry,
  linear coefficient recovery to 1e-8, non-linear order spread, subset scope,
  classifier scale, and the full error surface. No LLM summary was added —
  D2 remains gated. *(R9a closed)*
- [~] **D4** Experiment-design acquisition extensions — **readiness assessment
  complete 2026-11-08 (analysis only); implementation still gated.** With the
  real-user gate waived for the assessment, `benchmarks/design_readiness.py`
  compares the shipped GP-LCB policy with EI, PI, and a fixed-length-scale
  noise-aware GP at equal budgets on the three synthetic DOE objectives (5
  seeds, 6 + 14 evaluations, 256 candidates, noise 0/0.05; 150 runs, 3,000
  per-evaluation records). Findings in
  [`AI_LAYER02_D4_READINESS_ASSESSMENT.md`](./AI_LAYER02_D4_READINESS_ASSESSMENT.md):
  all GP policies beat random by ~an order of magnitude; EI/PI are neutral
  without noise; EI has a weak, non-significant edge under synthetic noise
  (6 wins / 4 losses / 5 ties; mean final regret 0.0084 vs 0.0154); PI is worse
  under noise; the noise-aware model regressed (0.054/0.057 vs LCB
  0.0044/0.0154), so noise adaptation needs joint length-scale fitting.
  **Recommendation:** ship nothing; keep the real-user gate. *Entry for any
  implementation:* at least one real expensive-experiment user and a
  representative cost/noise/batch protocol. *Exit:* reproducible equal-budget
  benchmark in that regime, published with an honest read; proposal-only
  contract preserved. Batch/multi-objective/multi-fidelity were not assessed.
  *(R7)*
- [ ] **D5** Document-to-tabular extraction: prototype separately from fold-local
  preprocessing; require schema-driven extraction, source-page/passage
  provenance, confidence scores, human validation, unit normalization,
  duplicate/contradiction handling, and export to a typed dataset. *Entry:*
  concrete demand with a sample document set. *(R9b)*

**Explicitly out of scope for this cycle** (carried over unchanged): symbolic
regression, physics-informed modeling, survival analysis, broad domain-specific
algorithm collections — unless a concrete demand and validation plan emerge.

## Cross-cutting rules (inherited from the follow-up plan)

- Verify each claim against current source before changing behavior; treat doc
  references as pointers, not proof.
- Extend working foundations instead of duplicating them; keep this file,
  `AI_LAYER_ROADMAP.md`, and `LAUNCH_PLAN.md` consistent when scope changes.
- Keep statistical methods descriptive where dependence is present: repeated-CV
  comparisons make no significance claims (Phase 2 decision stands).
- Descriptive summaries are not confidence intervals; association, attribution,
  and sensitivity are never presented as causation.
- Optional integrations stay optional; the core path stays dependency-light.

## Coordination with the launch

- Stages B–D do **not** block v0.3.0 (LAUNCH_PLAN §6: "do not let announcement
  scope wait on Phase 4/5 completion"). Stage A is small enough to land before
  the cut if convenient, but it is not a release gate.
- Vikunja project **40 — "SKSurrogate public launch (v0.3)"** carries this plan:
  launch phases P0–P3 due dates moved to December on 2026-11-09 (working launch day Tue Dec 15;
  see LAUNCH_PLAN decision 5); the post-launch remainder is
  staged under **Phase 4** tasks **P4.1–P4.9**, mirroring R1–R9 and Stages A–D.
  As of the 2026-11-08 reconciliation: P4.1/P4.2 (A1/A2), P4.4 (B2), P4.6 (C1),
  P4.7 (C2/C3), and P4.8 (D1) are done; P4.9 records D2 and D3 as complete with
  D4/D5 still gated. P4.3 (B1) and P4.5 (B3) remain open with their named
  evidence gates.

## Progress log

- 2026-11-08: Audited every open item in `AI_LAYER02_FOLLOWUP_PLAN.md` against
  current `ailayer` source (branch is 17 commits ahead of master; full suite
  passed with 290 tests + 32 subtests in a clean environment,
  `env -u PYTHONPATH ./.venv/bin/python -m pytest tests`). Confirmed Phases 0–3
  and 6 complete. Re-baselined the remainder into R1–R9 / Stages A–D with the
  blocker analysis above. Verified in source: (i) `openml` import precedes the
  warm-cache check in `benchmarks/datasets.py`; (ii) archived benchmark runs
  store no per-evaluation hyperparameters; (iii) evidence snapshots are
  fingerprint-only (`sha256`, schema v1) in `api/routers/projects.py`, so
  recovery is not currently possible.
- 2026-11-08: **Stage A complete.** *A1:* `_download_openml` now checks the warm CSV cache before importing `openml` — cached datasets load in slim environments without the `[benchmark]` extra; regression-tested (`tests/test_benchmark_datasets.py`) and verified end-to-end with the import blocked. *A2:* benchmark run records are schema v2 — every per-evaluation entry carries the full candidate configuration (all `SPACE` keys), each run JSON is stamped `schema_version`, and `run_manifest.json` records `record_schema_version`; `plot.py`/`analyze_surrogates.py` readers treat both as optional. Verified live: a smoke run wrote a v2 archive with full per-eval configs across all available arms × 2 seeds, and the real legacy v1 archive (`results/20261018T051214Z`, 84 ok-runs) re-read cleanly through `_load_records`, best-so-far curves, target tables, and `analyze_surrogates.py`. Reader tolerance is pinned by new tests in `tests/test_benchmark_archive_readers.py`; design contract updated to v1.14 (`benchmarks/DESIGN.md`). R7's "reproducible evaluation" exit criterion now has its code prerequisite; the remaining gate for D4 is a real expensive-experiment user.
- 2026-11-08: **Stage B started — B1 scoping drafted.** `AI_LAYER02_B1_ARTIFACT_INGESTION_SCOPE.md` (≤2 pages) scopes artifact ingestion from in-repo evidence only (pre-launch, so no first-adopter feedback yet): candidate-kind table with verified locations, decision for a bounded typed superset `{reference, file_artifact}` over open URIs or a permanently closed set, endpoint-free API design extending the existing project endpoints with snapshot schema v2 and per-kind validation/dedup, and a five-part test plan mirroring `tests/test_projects_endpoints.py`. No code started; B1 marked `[~]` gated on post-launch feedback acceptance.
- 2026-11-08: **Stage B continued — B2 decision drafted.** `AI_LAYER02_B2_ARTIFACT_RECOVERY.md` answers R3's storage question from in-repo evidence only: v1 snapshots already embed all small structured per-kind details (partition schema metadata, bundle reproducibility record, job configuration/result fingerprints), so content embedding would add only partition row data, job-result payloads, and model weights — at the cost of whole-file project reads on every list/summary (`api/routers/projects.py` :326–338), a second write path breaking single-file atomicity (:169–175), new restore APIs into upstream stores, and a provenance hazard (one-shot captures re-materialized as "current" would make the drift check self-fulfilling). Recommends committing to detect-and-report only: the bounded work is a per-status behavior contract in `docs/research-projects.rst` (replacing its one-line no-recovery sentence) plus tests pinning `unverified` legacy records and snapshot preservation; no schema change (v2 arrives via B1's new kinds without content fields). Recorded upgrade path: a per-kind optional `content` field under the existing v2 schema with size cap + stored-bytes accounting, gated on at least one real adopter request. No code started; B2 marked `[~]` gated on decision acceptance (post-launch per stage design; docs/tests-only work may land pre-cut if preferred).
- 2026-11-08: **B2 decision accepted.** Mehdi accepted the detect-and-report-only recommendation in `AI_LAYER02_B2_ARTIFACT_RECOVERY.md` §4. B2 stays `[~]`: remaining work is its §3 bounded docs + tests (behavior-neutral; may land pre-cut or post-launch per stage design). Deliverables left untracked alongside the other plan docs, by user choice.
- 2026-11-08: **Stage B continued — B2 §3 landed; R3 closed.** Per the accepted detect-and-report-only decision: (i) `docs/research-projects.rst` now carries a per-status behavior contract replacing the one-line no-recovery sentence — which upstream events produce `current` / `changed` / `missing` / `unverified` for each of the partition, bundle, and job kinds; snapshots are captured once at link time and never re-baselined (no supported way to reset drift on an existing reference); evidence-referenced artifacts cannot be removed via `PUT`; durability is a property of the upstream stores (`mltrace/<task>.db`, `bundles/` + `registry/`, `jobs/*.json` under the configured root) with backup guidance. (ii) Two pinning tests added to `tests/test_projects_endpoints.py`: legacy records without a stored snapshot report `unverified` (partition + bundle, following the existing on-disk-legacy pattern), and project edits after an upstream change preserve the stored snapshots byte-for-byte while the summary still reports `changed`/`current`. Verified in clean env: full suite 302 passed / 32 subtests (`env -u PYTHONPATH ./.venv/bin/python -m pytest tests`; +2 vs pre-change collection), Sphinx HTML build exit 0 with the new section rendered. B2 marked `[x]`; R3 closed with rationale recorded in `AI_LAYER02_B2_ARTIFACT_RECOVERY.md` §4 — upgrade path (per-kind optional `content` under snapshot schema v2, size cap + accounting) stays open pending at least one real adopter request naming the kind. Tracked-file changes (`docs/research-projects.rst`, `tests/test_projects_endpoints.py`) left uncommitted for user review; plan docs remain untracked by choice.
- 2026-11-08: **B2 tracked changes committed; Stage D started — D1 foundation built.** The B2 §3 deliverables were committed as `e486b75` ("docs(projects): pin evidence-snapshot behavior contract (Stage B2, R3)"); plan docs remain untracked by choice. D1: new `benchmarks/dgps.py` — four seeded parametric DGPs with exact known ground truth (`interactions`: x0·x1 term; `correlated`: block correlations 0.9 on two feature pairs; `heteroscedastic`: σ(x) ∝ (1+2|x2|); `shift`: quadratic x2² term plus mean shift +2 at test), a method panel (ridge / random-forest / kernel-ridge, fresh per seed), and `validate()` reporting per-method rmse/r² on the test split vs an iid reference split with degradation ratios, plus per-DGP ground-truth probes; writes versioned `dgp_validation.json` (`sksurrogate.dgp-validation.v1`). Contract documented in `benchmarks/DESIGN.md` §14 (v1.15), including the deliberate relationship to `SKSurrogate/synthdat.py` (dataset-like resampling vs known-process generation — both coexist, not merged). Verified: 14 new tests green (`tests/test_benchmark_dgps.py`) and a full default CLI run with sane descriptive numbers (linear models visibly miss the interaction; shift degrades every method ~3.2–3.5×, RF least). D1 stays `[~]`: its exit criterion (Stages B/C methods validated against known values) is open until those methods exist — they will be validated here first when C2/C3 land.

- 2026-11-08: **Stage C complete (C1, C2, C3), C-method D1 hook landed.** Direct user
  direction: implement the explanation features in the API/UI layer, not in the
  `SKSurrogate` core unless necessary. *C1:* `AI_LAYER02_C1_SHAP_DECISION.md` records the
  accepted decision "formally defer SHAP this cycle, no dependency", with verified preset
  estimator coverage (RF/ExtraTrees only), the missing-consumer argument (row-level
  explanations are D3/R9a), measured runtime in a throwaway scratch venv (shap 0.54.1:
  TreeExplainer 0.38 s / 200 rows; KernelExplainer 2.3 s / 100 rows, 11.6 s / 500 rows
  with a 50-row k-means background; additivity residual ≤ 3.3e-14), dependency metadata
  (shap 0.54.1 needs numpy/scipy/sklearn/pandas/tqdm/packaging/slicer/cloudpickle — no
  numba), a five-part test plan, and the reconsideration gate. Roadmap P1.5 updated.
  *C2:* new pure-NumPy `api/explanations.py` (no `SKSurrogate` change, no new dependency)
  and `POST /api/evaluation/{task}/{version}/ale`; contract replaces the "not exposed"
  paragraph in `docs/mltrace.rst` — numeric-only equal-frequency bins with duplicate-edge
  collapse, deterministic sparse-bin merging at `min_bin_fraction` 0.01, per-curve
  excluded-row counts, labelled scalar response (predict / positive-class probability /
  decision value; multiclass 422), explicit-or-top-k feature selection, non-causal
  interpretation labels. *C3:* `/explain` now returns an additive
  `partial_dependence_stability` block (canonical grid, per-grid mean / sample SD / min /
  max / per-subset means) for 3–5-subset requests; documented as descriptive, not a CI;
  ICE stability explicitly out of the API contract with the reason recorded. UI: typed
  client (`computeAle`, `PdpStability`, `AleCurve`) and an Explain page ALE card (10/20/30
  bins, per-bin counts, skipped-feature reasons, limitations) plus dashed mean ± SD lines
  on PDP and ALE charts. Docs: `docs/mltrace.rst`, `docs/ui-manual.rst`,
  `docs/walkthrough.rst` (also corrected its interval example payload from `coverage` to
  `alpha`), `CHANGELOG.rst`. D1: `benchmarks/dgps.py::_ale_truth_checks` compares the ALE
  kernel with the exact noise-free response of the `correlated` and `shift` DGPs
  (including the known quadratic) and matches to ~1e-15; DESIGN.md bumped to v1.16.
  **Verified:** full suite 327 passed + 32 subtests in a clean env (`env -u PYTHONPATH
  ./.venv/bin/python -m pytest tests`; HEAD collection 316 → +11 = 9 new API tests + 2
  new DGP tests, delta accounted for exactly), `npm run build` (tsc -b + vite) exit 0,
  Sphinx HTML build exit 0 with zero warnings and the new sections rendered. Tracked
  changes committed in path-scoped commits; plan docs remain untracked by choice.
- 2026-11-08: **Stage D started — D1 closed, D3 contract + implementation landed.**
  *D1:* `benchmarks/dgps.py` gained `_pdp_truth_checks` (shared `_truth_response`
  helper): average PDP on the exact noise-free response of the `correlated` and
  `shift` DGPs equals `h_j(g) + C_j` (C_j read from the same function at
  `x_j = 0`) with max abs error ~4e-16 / 9e-16, and reference-subset curves may
  differ only by their offset (shape deviation ~1e-15); the descriptive subset
  SD (~0.10 / 0.14) is reported rather than asserted to zero, because subset
  reference means legitimately shift an additive curve. DESIGN.md v1.17, 17
  tests in `tests/test_benchmark_dgps.py`. D1 flipped to `[x]`: its exit
  criterion is met for the landed Stage C methods (ALE + PDP/stability).
  *D3:* contract `AI_LAYER02_D3_ROW_LEVEL_CONTRACT.md` (approved) and bounded
  implementation — pure-NumPy ordered-occlusion chain decomposition in
  `api/explanations.py` (`chain_contributions`, `row_contribution_summary`,
  `distinct_permutations`) plus `POST /api/evaluation/{task}/{version}/explain/row`.
  Contributions telescope exactly (max residual ~1e-16 against a reported
  tolerance), reference-mean or explicit-row baselines are recorded with
  per-feature value sources, `orders` 1–10 are distinct seeded permutations with
  mean/sample-SD/min/max as disclosed order-dependence, feature subsets report
  `subset_hybrid`, multiclass is rejected 422, and same-seed requests are
  bit-identical. UI: Explain page **Row-level explanation** card (row index,
  1/5/10 orders, contribution chart, mean/SD/min/max table, reconstruction
  residual, limitations). Docs: `docs/mltrace.rst`, `docs/ui-manual.rst`,
  `docs/walkthrough.rst`, `CHANGELOG.rst`, roadmap P1.5 table + log. Tests:
  `tests/test_api_explanations.py` (+6: kernel exactness/order symmetry, linear
  recovery to 1e-8, non-linear order spread, subset scope, classifier scale,
  error surface). *D2/D4/D5 stay gated*: no adopter feedback, no real
  expensive-experiment user, no sample document set — no scope invented.
  **Verified:** full suite 334 passed + 32 subtests in a clean env (baseline
  commit `9bc3ae9`: 327 → +7 = 6 new API tests + 1 new DGP test, delta exact),
  `npm run build` (tsc -b + vite) exit 0, Sphinx HTML build exit 0 with zero
  warnings. Tracked changes committed path-scoped; plan docs and the D3 contract
  remain untracked by choice.
- 2026-11-08: **Stage D continued — D2 contract + bounded opt-in implementation
  landed (post-launch gate waived).** `AI_LAYER02_D2_LLM_SYNTHESIS_SCOPE.md`
  records the bounded use case, OpenAI-compatible provider contract, aggregate
  fact list (S1 profile, S2 run, S3 bundle, S4 explanations, S5 alerts),
  citation enforcement, persistence/provenance rules, data-handling review
  (prompt not stored; job result and saved notes follow normal retention), API
  and UI contracts, and the 7-test plan. Implementation: `api/synthesis.py`
  (stdlib `urllib` provider client with timeout/HTTP/format error mapping and
  `[S#]` validation), `api/synthesis_facts.py` (bounded fact gathering from
  stored artifacts, `_sanitize` for NaN/NumPy), `api/routers/synthesis.py`
  (`GET /api/synthesis/status`, `POST /api/synthesis/projects/{id}` job),
  generated-note provenance in `api/routers/projects.py` (rejects generated
  notes without model/citations or citation/fact mismatch, 30,000-char facts
  cap), and the Projects UI **Draft synthesis** card with generated badges and
  citation/computed-fact rendering; Jobs kind filter extended. No new
  dependency; `SKSurrogate/` untouched. **Verified:** `tests/test_api_synthesis.py`
  7 passed against a local stub HTTP provider (no network), live end-to-end run
  against a stub provider through the real server + built SPA (job completed,
  draft cited `S1/S3/S4`, note persisted with provenance; `refetchIntervalInBackground`
  added after a headless-tab polling stall exposed it), `npm run build` exit 0,
  Sphinx HTML build exit 0. D4/D5 remain gated on a real expensive-experiment
  user and a sample document set; no scope invented.

- 2026-11-08: **Plan reconciliation after Stages C/D (documentation only, no
  code).** The document's own status text had drifted behind the work: the
  header still described Phase 5 as having two gated method items and Phase 7 as
  "not started"; blocker 1 said SHAP and the LLM layer had no decided spec;
  blockers 3/5/6 still described R3 storage, the A1/A2 code fixes, and the R6
  synthetic-data gap as open; the "verified current state" table said Phase 7
  had no implementation; and R7/R1's unblocked-when columns still presented
  pending prerequisites. Reconciled all of these against the landed artifacts
  and tests: Phase 5 is complete (C1 deferred by decision, C2/C3 shipped),
  Phase 7 has D1/D2/D3 shipped, blockers 1/2/3/5/6 are marked resolved with
  dated evidence, a new blocker 7 lists the complete remaining evidence gates
  (B1 adopter feedback, B3 demonstrated usage, D4 a real expensive-experiment
  user, D5 a sample document set), B1 records the accepted typed-superset
  decision, D4 states that Stage A is met and only the user gate remains, and
  the coordination section records the current P4.x task states. Verified by
  grep that no stale "not started"/"unfilled"/"undecided"/"currently impossible"
  claims remain for closed items. No source or tracked file changed; plan docs
  remain untracked by choice.

- 2026-11-08: **B1 tier-1 (`reference`) implemented** with the post-launch gate waived by
  Mehdi for this bounded kind. `ProjectArtifactReference` now accepts
  `{kind: "reference", url, title?, doi?, label?}`: absolute `http(s)` URL only
  (≤2,048 chars), never fetched; references reject `task_name`/`artifact_id` and
  non-reference kinds reject `url`/`doi` (422). Identity/dedupe is by URL, the
  snapshot is schema v2 with stored fields only (`content_embedded: false`), and
  the summary reports `exists: true` with `snapshot_status: "unverified"` rather
  than pretending reachability. References do not advance workflow stages but
  can satisfy finding/conclusion evidence and cannot be removed via `PUT` while
  referenced; old-kind artifact shapes and snapshots are unchanged (v1,
  `exclude_none` dump). UI: **Add external reference** form in Linked artifacts,
  external-link rendering with DOI and unverified badge, and evidence picker
  labels. Docs: `research-projects.rst`, `ui-manual.rst`, `CHANGELOG.rst`, B1
  scope §5. **Verified:** `tests/test_projects_endpoints.py` 27 passed (+5 new:
  reference roundtrip/snapshot/summary, validation, evidence/removal protection,
  legacy shapes, HTTP surface), live run through the built SPA on a real server
  (reference added, rendered as `reference … DOI … snapshot unverified`, appeared
  in the evidence picker), `npm run build` exit 0. `file_artifact` remains gated.

- 2026-11-08: **D4 readiness assessment completed (analysis only; no production
  code).** Built `benchmarks/design_readiness.py` (schema
  `sksurrogate.design-readiness.v1`, CLI, 9 tests) comparing the shipped
  `suggest_next_point` LCB policy against EI, PI, and a Matern+WhiteKernel
  noise-aware variant at equal budgets, paired seeds, and identical candidate
  pools on the three synthetic DOE objectives. Full default run: 5 seeds, 3
  families, noise 0/0.05, budget 20 (6+14), 256 candidates, 150 runs, 3,000
  per-evaluation records, 3.7 s. **Findings:** random 0.1106 mean final regret;
  LCB 0.0044 clean / 0.0154 noisy; EI 0.0039 / 0.0084; PI 0.0036 / 0.0205;
  noise-aware 0.0537 / 0.0567. Paired vs LCB: EI 1W/0L/14T clean and 6W/4L/5T
  noisy; PI 2W/0L/13T and 4W/8L/3T; noise-aware 0W/7L/8T and 6W/7L/2T. Read:
  acquisitions are second-order on this evidence; EI is the only plausible
  direction under noise and remains unproven at five seeds; the
  fixed-length-scale noise-aware GP is a regression (it absorbs signal as
  noise). Decision record `AI_LAYER02_D4_READINESS_ASSESSMENT.md`, contract
  `benchmarks/DESIGN.md` §15 (v1.18), pointer in `docs/experiment-design.rst`,
  CHANGELOG, roadmap P2.4 status. **D4 stays `[~]`/gated**: no acquisition
  code, endpoint, or UI change shipped; the exit criterion still needs a real
  expensive-experiment user and a representative protocol. Batch,
  multi-objective, multi-fidelity, constraints, and replication were not
  assessed (they need a cost/feasibility model).
