# dev-notes — contributor working notes

This directory holds design notes, implementation plans, handoffs and decision
records. **Nothing here is user documentation.** If a reader of the docs needs
it, it belongs in `docs/`, not here.

## Rules

1. **One home per artefact.** User guides, tutorials, API reference and release
   notes live in `docs/`. Plans, scopes, roadmaps, handoffs and "what I did and
   why" records live here.
2. **No new root-level notes.** `*PLAN*.md`, `HANDOFF-*.md`, `AI_LAYER*.md` at
   the repository root fail `tests/test_repo_layout.py`; put them here instead.
3. **Filenames stay stable** even after a milestone lands — code docstrings
   reference these files by path (`dev-notes/ui-plan.md` §4 and so on), so
   renaming costs a repo-wide edit. Add a status line inside the document
   (`Status: landed in 0.4.0`) rather than moving it.
4. **State the status at the top** of every plan: `proposed`, `in progress`,
   `landed <version>`, `abandoned <reason>`. A stale plan without a status is
   worse than no plan.
5. **Not packaged.** `dev-notes/` is excluded from the wheel and the sdist
   (`setup.py` packages only `SKSurrogate`; `MANIFEST.in` lists `docs/`,
   `tests/` and `examples/`), and it is not built by Sphinx.
6. **Local-only drafts** go in `dev-notes/private/`, which is gitignored. Use it
   for notes that are not ready to share; promote them here when they are.

## Inventory

| Document | What it is |
| --- | --- |
| `ui-plan.md` | Web UI & control-plane API architecture and phased delivery |
| `ui-gap-implementation-plan.md` | Phase-by-phase gap closure between library/API and UI (Phases 1–4) |
| `visual-search-space-editor-plan.md` | Visual search-space editor design (Experiments → Advanced) |
| `mlops-roadmap.md` | MLOps roadmap and phase history |
| `phase9.rst` | Execution, CI/CD and job-lifecycle operations plan |
| `phase10.rst` | Documentation and release-hardening plan |
| `UI_DOCKER_UPGRADE_PLAN.md` | Docker/UI upgrade plan and its checklists |
| `AI_LAYER02_REMAINING_PLAN.md` | AI-layer follow-up plan (index of stages A–D) |
| `AI_LAYER02_D2_LLM_SYNTHESIS_SCOPE.md` | Stage D2 scope: opt-in LLM-assisted project synthesis |
| `AI_LAYER02_D3_ROW_LEVEL_CONTRACT.md` | Stage D3 row-level explanation contract |
| `AI_LAYER02_D4_READINESS_ASSESSMENT.md` | Stage D4 readiness assessment (acquisition extensions) |
| `HANDOFF-explainability-visibility.md` | Handoff: explainability surfacing in the UI |

## Referenced but intentionally local-only

`AI_LAYER02_REMAINING_PLAN.md` cites decision records that are **not tracked**
(`.gitignore` names them explicitly, because they quote internal deliberation):
`AI_LAYER02_ROADMAP.md`, `AI_LAYER02_B1_ARTIFACT_INGESTION_SCOPE.md`,
`AI_LAYER02_B2_ARTIFACT_RECOVERY.md`, `AI_LAYER02_C1_SHAP_DECISION.md`, and the
MCP execution plan. A fresh clone therefore cannot open those links; that is a
deliberate choice, not a broken repository. Promote one of them here (and drop
its `.gitignore` line) if it should be public.
