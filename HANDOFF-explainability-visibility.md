# Handoff — SKSurrogate v0.3: explainability visibility pass (docs/assets only)

**Internal working document.** Prepared by the advisor profile, 2026-11-07.
**Approved by maintainer 2026-11-07.** Target executor: **Coder profile**.

> **Scope in one line:** documentation and asset changes that make the already-shipped
> explainability/sensitivity features visible in launch materials, plus one new figure.
> **No package source code, no new dependencies, no license changes, no commits/pushes.**

---

## 1. Context (verified 2026-11-07 — re-verify before editing)

- Repo: `/home/mehdi/Code/Python/sksurrogate`. Working branch **`ailayer`** (32 commits
  ahead of v0.2.0). ReadTheDocs builds from `master`, so nothing here is live until the
  planned `ailayer` → `master` merge (LAUNCH_PLAN decision 4, resolved: stability pass
  first, then merge, then cut v0.3.0; launch target ≈ Nov 17/18).
- Launch plan of record: **`LAUNCH_PLAN.md`** (v1, 2026-11-07). Read at minimum:
  §1 (Option A positioning — "end-to-end scientific MLOps environment"),
  §2.4–§2.5 (demo assets, media kit), §3.2 (blog outline), §7 ("must not claim" guardrails),
  §8 (effort blocks). Vikunja project **#40** tracks the launch tasks.
- **Why this task exists:** the maintainer approved extracting the *robustness/stability*
  narrative as differentiator #1 in launch copy and closing the visibility gap for
  explainability features that already ship on `ailayer`. Docs/assets only — zero code risk,
  launch date intact.

### Verified feature inventory (already implemented on `ailayer` — document it, do NOT build it)

| Feature | Location | Notes |
| --- | --- | --- |
| Model explanations API: permutation importance + partial dependence plots for a stored bundle | `api/routers/explain.py` — `POST /api/evaluation/{task_name}/{model_version}/explain` | UI "explain page" exists (see `docs/walkthrough.html`, `ui-manual`) |
| Explanation engine (permutation-based feature scores, interaction pairs) | `SKSurrogate/mltrace.py` (~L2201–2500) | sklearn-style; documented in the scientific-workflow guide (`docs/mltrace.rst`) |
| Sensitivity analysis: Sobol / Morris / δ-moment via SALib | `SKSurrogate/sensapprx.py`; also a search option in `aml.py` | RTD page **`docs/sensitivity.rst` already exists on this branch** |

- Licensing decision (LAUNCH_PLAN §10, resolved 2026-11-07): **plain MIT everywhere.**
  No AGPL layer, no commercial tier, no CLA machinery. Do not introduce any of these in copy.
- Known state: the `ailayer` stability pass has surfaced several failing tests (see
  LAUNCH_PLAN progress log). **Do not attempt to fix unrelated test failures** — note them
  in your report if you hit them; this task touches no source code, so they are orthogonal.

## 2. Tasks

### T1 — `readme.rst`: make explainability/sensitivity part of the feature story

Current state (verified): there is **no feature table** yet (that arrives with the §2.2
README rewrite), and the README currently never mentions explanations or sensitivity
analysis. The "Why not Optuna / vanilla BO?" bullet list ends ~L26; the drift section
starts ~L122.

Changes:
- **A.** Add a short section (suggested title: *"Understand what you tuned"*) between the
  hybrid-loop section and Quick start, covering exactly two capabilities that ship today:
  - Model explanations — permutation importance + partial dependence plots for stored
    model bundles (API endpoint above; UI explain page). Link to the control-plane /
    scientific-workflow guides.
  - Sensitivity analysis — Sobol/Morris/δ-moment indices identifying which hyperparameters
    dominate score variance and where interaction effects live. Link to
    `https://sksurrogate.readthedocs.io/en/latest/sensitivity.html` (verify the built URL
    from `docs/sensitivity.rst`; adjust if the slug differs).
- **B.** Keep Option A positioning throughout — no "AI"-style claims, no new hero copy.

Acceptance: RST syntax valid; every link resolves to a file that exists under `docs/` on
this branch; wording complies with §3 guardrails below.

### T2 — `LAUNCH_PLAN.md`: record the decision and reorder differentiators

- **§3.2 point 3** (three differentiators): reorder so **robustness/stability is #1**,
  citing as artifact link the new figure (`docs/images/stability_contour.png`); shift
  statistical evaluation → #2, research workspace → #3 (keep their text unchanged).
- **§2.4 (demo assets):** add one checklist item: *"Stability contour figure — brittle peak
  vs stable plateau; script + PNG per T3."*
- **§2.5 (media kit):** note that the explain-page screenshot is **extracted from existing
  manual/walkthrough screenshots** (`docs/walkthrough.html`, `ui-manual`), not newly captured.
- **Progress log:** append a 2026-11-07 entry: maintainer approved extracting the stability
  narrative as differentiator #1 and a docs-only visibility pass for shipped explainability;
  LLM-based explanation remains Phase 7, post-launch (next candidate item).

### T3 — Stability contour figure (the only "code" in this task — an asset script, outside package source)

Goal: one concrete visual of the *brittle peak vs stable plateau* contrast on a real
benchmark dataset. This is the launch's workhorse artifact for differentiator #1.

Constraints:
- Script location: follow the existing figure convention — check `docs/benchmarks/figures/`
  layout first; if scripts live there, add it there; otherwise use `docs/images/make_stability_contour.py`.
- Use **only the package's public API** (fit surrogate surface over a 2-D slice of two
  hyperparameters on one benchmark dataset) + matplotlib (already a dependency). No new deps.
- Output: `docs/images/stability_contour.png` (~1200 px wide, blog-quality), plus the
  script so it is reproducible.
- Dataset: pick from existing `benchmarks/results/` runs where the contrast is genuinely
  visible (e.g., optdigits or credit-g). If no dataset shows a clean peak-vs-plateau
  contrast in a 2-D slice, use the clearest available and say so honestly in the caption —
  do not cherry-pick an unrepresentative slice to manufacture drama.
- Caption must state exactly what is shown: *surrogate posterior mean of cross-validated
  score over a two-dimensional hyperparameter slice*. It must **not** claim robustness to
  data drift or any out-of-sample guarantee (see §3).

### T4 — Media kit list (§2.5)

Add the two new assets to the media-kit asset inventory: stability contour figure and the
explain-page screenshot (with source pointer per T2/T3).

## 3. Wording guardrails (README, blog copy, figure caption)

**Allowed claims:** local curvature / flat valley vs sharp peak *in hyperparameter space*;
"which hyperparameters dominate score variance"; interaction effects between hyperparameters
(sensitivity indices); permutation importance + partial dependence plots; "the surrogate
surface makes the neighbourhood of the optimum inspectable."

**Not allowed without further evidence:** any claim about robustness to data drift/shift;
"proves/guarantees model stability"; compliance, audit-report, or Model-Card language.
Sensitivity indices describe association structure of the fitted surrogate — not causation
(working principle in `AI_LAYER02_FOLLOWUP_PLAN.md`). No LLM-feature claims at all: Phase 7
is post-launch and LAUNCH_PLAN §7 forbids implying otherwise.

## 4. Out of scope — do NOT touch

- **LLM features**, including the `POST /api/projects/{id}/synthesize` endpoint (Phase 7,
  item 1). Explicitly post-launch; it is recorded in T2 only as the *next candidate*, not to be built.
- Package source under `SKSurrogate/` and `api/` — this pass is docs/assets only.
- New dependencies of any kind (no SHAP, no new plotting libraries beyond matplotlib).
- License files (`LICENSE` / `License.txt`) — cleanup belongs to the separate Phase-0
  hygiene task; do not mix it into this change set.
- Stray root artifacts (`*.pkl`), `HANDOFF.md` trimming — separate hygiene pass (§2.3).
- **No git commits, pushes, tags, or PyPI uploads.** Leave changes in the working tree and
  report them for maintainer review (the repo's standing workflow: explicit approval before commit).

## 5. Verification before reporting

1. RST/Sphinx check on `readme.rst` (local build per `.readthedocs.yml`: Python 3.11,
   myst-parser from `docs/requirements.txt`) — at minimum a docutils parse with no errors.
2. T3 script runs to completion in the project dev environment (Docker py3.12 per
   `docker-compose.yml` / `Dockerfile`, mirroring CI) and produces the PNG; visually verify
   that a peak-vs-plateau contrast is actually visible before reporting success.
3. Test suite: no source changes are expected, so pytest should be unaffected — run it (or
   an affordable subset) as a sanity check only and report the result.
4. Link audit: every new link in `readme.rst` resolves to a file that exists under `docs/`
   on this branch; RTD URL slugs double-checked against actual filenames.

## 6. Report format (for maintainer review)

1. File list with unified diffs (or per-file diff summary).
2. Figure: path + caption text exactly as written + one-line note on dataset/slice choice.
3. Deviations from this spec, each with a reason.
4. Open questions (e.g., if the peak-vs-plateau contrast was weaker than expected and you
   need the maintainer to pick between datasets).

## Appendix — key paths

| What | Where |
| --- | --- |
| Launch plan of record | `LAUNCH_PLAN.md` (§1, §2.4–§2.5, §3.2, §7) |
| Phase 7 definition (post-launch LLM scope) | `AI_LAYER02_FOLLOWUP_PLAN.md`, "Phase 7" section + working principles |
| README to edit | `readme.rst` |
| Sensitivity docs page (already on branch) | `docs/sensitivity.rst` |
| Explanations API / engine | `api/routers/explain.py`; `SKSurrogate/mltrace.py` (~L2201–2500) |
| Sensitivity estimator | `SKSurrogate/sensapprx.py` |
| Benchmark result dirs (for T3 dataset choice) | `benchmarks/results/` |
| Existing figure convention | `docs/benchmarks/figures/`; README logo uses `./docs/images/funnel.png` |
