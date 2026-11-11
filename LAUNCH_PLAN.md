# SKSurrogate Public Launch Plan

**Internal working document.**

## Purpose and status

Plan for announcing SKSurrogate publicly as an end-to-end MLOps environment
with AI-assisted scientific workflows. This is a living plan: update task
status and the progress log as work lands.

**Status:** Draft v1, 2026-11-07. No launch tasks started yet. Pending three
decisions from the maintainer (Section 9).

## Verified starting point (audited 2026-11-07; LLM row re-verified 2026-11-10)

| Asset | State |
| --- | --- |
| GitHub repo `mghasemi/sksurrogate` | Public, CI badge live (`ci.yml`), 1 star / 2 forks |
| PyPI | `sksurrogate` published: 0.1.0, 0.1.1, **0.2.0** (current) |
| Git tags | v0.1.1 and v0.2.0 pushed to origin; **no GitHub Releases exist** |
| Branch topology (verified 2026-11-07) | `master` = `origin/master` at `8f7f9ec` (Oct 27): since v0.2.0 it carries only docs/packaging work (JOSS bundle, technical note, D-track study + DOI, README rewrite, benchmark table, honest-benchmark blog post). The **dev branch `ailayer`** = `origin/ailayer` at `0f8c96f` (Nov 6) is **32 commits ahead of v0.2.0** and holds the Phase 4/5 research-workspace features (project evidence links, hypothesis lifecycle with status history, snapshot state display, conformal diagnostics, extended dataset profile, fold-local preprocessing UI) plus the 241-page manual PDF. Working tree clean; checked out on `ailayer`. **RTD builds from `master`**, so ailayer's docs are not yet live at "latest". |
| Documentation | Live on ReadTheDocs incl. honest-benchmark blog post; 241-page manual PDF in repo root (untracked) |
| Benchmarks | Equal-budget benchmark suite published with raw-results DOI `10.5281/zenodo.23629024` |
| Packaging extras | Dockerfile, docker-compose.yml (`--profile prod` single port), CITATION.cff |
| README (`readme.rst`) | Positions the package as a "drop-in replacement for BayesSearchCV" and explicitly keeps UI/API "out of the headline" — **contradicts the new end-to-end positioning** |
| LLM features in code | **Shipped, opt-in.** Bounded cited-note synthesis in `api/synthesis.py` + `api/routers/synthesis.py` (re-verified 2026-11-10): disabled unless `SKSURROGATE_LLM_BASE_URL` and `SKSURROGATE_LLM_MODEL` are set; receives computed aggregates only, never raw dataset rows; stateless (SKSurrogate stores no prompt and no draft); every claim must cite `[S1]…[Sn]` or the draft is rejected. Row-level explanations and ALE also shipped. Document-to-table extraction and acquisition extensions remain unimplemented. Authority is `AI_LAYER02_REMAINING_PLAN.md`; the `AI_LAYER02_FOLLOWUP_PLAN.md` line "Phase 7 is not started" is stale. |
| Repo hygiene issues | Stray artifacts at root: `optim_task.pkl`, `sklearn.linear_model.LogisticRegression.pkl`; internal review docs (`ailayer.md`, `ai_layer02.md`, the two plans) tracked in a public repo; `HANDOFF.md` public |

## 1. Positioning — the single most important decision

**The problem with "equipped with AI" as written.** Not that no LLM surface
exists — one ships — but that "equipped with AI" implies an autonomous
assistant, while what ships is a *bounded, opt-in citation generator*.
Overstating it invites exactly the credibility damage the honest-benchmark post
worked to prevent. The product's *actual* intelligence:

- surrogate-based optimization (GP response surface, trust-region contraction) — that is ML driving search;
- sensitivity analysis (Sobol/Morris/δ-index), conformal prediction intervals, experiment design with GP-LCB active learning;
- opt-in LLM-assisted project synthesis: an OpenAI-compatible endpoint the user supplies, fed only computed aggregates, stateless on the SKSurrogate side, with every claim required to cite `[S1]…[Sn]` or the draft is rejected;
- a deterministic-first foundation (run-config + report.md endpoints, project evidence records) that keeps every plot, table and heuristic working with the LLM disabled.

**Recommended positioning (Option A — verifiable):**

> **SKSurrogate: the end-to-end scientific MLOps environment.**
> One open-source tool from dataset registration to deployed, monitored models —
> with surrogate-based optimization, statistical evaluation (nested CV + conformal
> intervals), sensitivity analysis, experiment design for expensive experiments,
> and a research workspace that tracks evidence. Built on scikit-learn; drop-in
> `BaseSearchCV` tuner included.

This is 100% backed by shipped code and the manual. "AI" appears as a *feature*
("surrogate-driven optimization", "active learning for experiment design",
"opt-in cited synthesis"), never as a banner. The most credible line in the
announcement is the transparency one: **"bring your own model endpoint; only
computed aggregates leave the machine; every generated claim must cite a source
or it is rejected; and with no endpoint configured the tool is fully
air-gapped."** Note the precise scope: SKSurrogate itself stores no prompt and
no draft, but retention at the *model provider* is whatever the user's chosen
provider does — which is exactly why the endpoint is user-supplied.

**Option B — RESOLVED 2026-11-08; no longer a decision.** The bounded feature it
proposed — dataset profile + run-config + explanation summaries into cited
project notes against a user-supplied model endpoint — shipped in Stage D2 as
`POST /api/synthesis/projects/{project_id}` (plus `GET /api/synthesis/status`),
implemented in `api/synthesis.py` + `api/routers/synthesis.py` and covered by
`tests/test_api_synthesis.py`. It stays off by default and stays a *feature*, not
a headline: the launch claim is opt-in, provider-agnostic, aggregate-only
synthesis, never "AI-equipped".

**Target audience (primary):** scientists and technical teams whose workflow
includes expensive or iterative experiments — pharma/bio, materials, energy,
industrial R&D, quant research — who currently stitch together Jupyter + Optuna
+ scikit-learn + a deployment script + drift checks by hand. **Secondary:**
practicing ML engineers who want an honest HPO/AutoML comparison and a
drop-in `BaseSearchCV` replacement. The scientific angle is the differentiator:
FLAML/Optuna/AutoGluon all optimize models; none of them tracks research
questions, evidence, experiment design for physical experiments, or dataset
identity/drift the way SKSurrogate does.

**One-liner candidates (pick one, use it everywhere):**
1. "SKSurrogate takes you from raw dataset to deployed, drift-monitored model in one scikit-learn-native environment — and keeps a research record of every step."
2. "End-to-end MLOps for scientists: surrogate optimization, statistical evaluation, experiment design, and evidence-tracked research projects in one open-source tool."

## 2. Phase 0 — Pre-launch hardening (do all before announcing)

### 2.1 Cut the release so install == announcement
- [x] Decide version number: **v0.3.0** (research workspace +
      evidence tracking are new documented capabilities, not patch fixes). Decided 2026-11-08.
- [x] Finalize `CHANGELOG.rst` for v0.3.0 — `Unreleased` converted to
      `[0.3.0] - 2026-11-08` covering the **48** commits ahead of v0.2.0 (the
      earlier "32" was stale; recomputed 2026-11-08): project evidence links,
      hypothesis lifecycle with status history, snapshot state display,
      conformal diagnostics, extended dataset profile, fold-local preset
      preprocessing, ALE and row-level explanations, opt-in LLM synthesis,
      DGP/readiness benchmark harnesses, and core logging/telemetry.
- [~] Bump version in `SKSurrogate/__init__.py` (**done 2026-11-08**, `0.3.0`; `docs/conf.py`
      aligned), rebuild (**wheel built and clean-install verified**). **PyPI upload is an
      owner action** (credentials); nothing has been published from this session.
- [~] Tag v0.3.0 (**annotated tag created locally 2026-11-08**); push and the
      **first-ever GitHub Release** with full notes (Section 4) remain **owner actions** —
      they close the current gap where tags exist but no releases do.
- [~] Verify from a clean environment: **wheel install verified 2026-11-08** in a fresh
      venv (`0.3.0` from site-packages, README-style quick start, callbacks + audit export,
      LICENSE shipped in `dist-info/licenses/`). Remaining for the owner after the PyPI push:
      `pip install sksurrogate==0.3.0` from the index, plus `docker compose --profile prod up -d`
      and one dataset → experiment round trip. **The announcement must never outpace what installs.**

### 2.2 Rewrite the README for the new story
- [ ] Hero: end-to-environment claim + lifecycle diagram (data → search →
      evaluate → explain → promote → deploy → monitor, with research workspace
      spanning it). Replace "drop-in replacement" headline; keep drop-in tuner as
      a secondary hook (it is genuinely good for the HPO audience).
- [ ] Feature table mapping each lifecycle stage to its manual chapter.
- [ ] Demo: 10–20 s GIF of dataset → experiment → explain in the UI.
- [ ] "Why not Optuna / FLAML / AutoGluon?" section (constraints, trust-region
      contraction, scikit-learn-native, data identity/drift, research workspace).
- [ ] Honest benchmark block with link to `docs/benchmarks.md`, the RTD blog
      post, and the Zenodo DOI — lead with "we published our benchmark and it is
      not all in our favor; here are the numbers."
- [ ] Badges: CI (exists), PyPI version, ReadTheDocs, MIT license, Zenodo DOI.

### 2.3 Repo hygiene pass (before traffic arrives)
- [ ] Remove stray root artifacts: `optim_task.pkl`,
      `sklearn.linear_model.LogisticRegression.pkl` (and gitignore patterns so
      they stay out).
- [ ] Decide the fate of internal docs, all currently public at repo root:
      - keep in place (transparency; shows rigorous process) — acceptable; or
      - move to `docs/planning/` with a note that they are working documents.
    Either way, add one line in each file's header: "Internal working document."
- [ ] Review `HANDOFF.md` for anything internal-only worth trimming before it
      gets discovered by launch traffic.
- [ ] Ensure `.gitignore` covers the 7 MB documentation PDF (already untracked;
      confirm it stays that way) — or, if desired, link to it from README as a
      "full manual (PDF)" download via GitHub raw URL of an untracked-but-released
      artifact. Simpler: keep it local and rely on RTD + the Zenodo DOI.

### 2.4 Demo assets (the launch's workhorses)
- [ ] **90–180 s screen recording** of the web UI end-to-end: upload dataset →
      quick-start experiment → evaluation with conformal intervals → explain page
      → promote bundle → deploy/monitor view. Upload to YouTube (unlisted is
      fine for launch; public later) — GitHub does not host video well.
- [ ] **10–20 s GIF** of the same flow trimmed, embedded in README and blog post.
- [ ] **Colab notebook** (or at least a pinned executable notebook link from RTD
      Chapter 14) so readers can run the scientific workflow without cloning.
- [ ] **Stability contour figure**: surrogate response surface over a 2-D
      hyperparameter slice showing a brittle peak vs stable plateau — script + PNG
      per T3 of HANDOFF-explainability-visibility.md (`docs/images/make_stability_contour.py`).
- [ ] Optional: hosted demo instance (`docker compose` on a VPS or free-tier VM,
      read-only API key). Big trust booster; only if willing to maintain it for
      ~2 weeks post-launch. If not hosting, the one-line `docker compose up` in
      README is the fallback — that is acceptable.

### 2.5 Media kit (one file: `docs/media-kit.md`, kept private or untracked)
- [ ] Elevator pitch (30 s), short blurb for newsletters (1–2 sentences),
      five feature bullets, license/pricing line ("MIT licensed, free, no cloud
      account required"), maintainer bio (one paragraph: PhD mathematician, 15+
      years research → deployed systems).
- [ ] Screenshot set (6–8): dashboard, experiment detail, explain, projects/evidence,
      experiment design, monitoring. All already exist in the manual — reuse; the
      explain-page shots are extracted from the existing manual/walkthrough screenshots
      (`docs/ui-manual.rst` / `docs/walkthrough.rst`, RTD pages `ui-manual.html` and
      `walkthrough.html`), not newly captured.
- [ ] Stability contour figure: `docs/images/stability_contour.png` (2-D hyperparameter
      slice, brittle peak vs stable plateau); regenerate with
      `docs/images/make_stability_contour.py`.
- [ ] Explain-page screenshots: cropped from the existing manual/walkthrough screenshots
      above — no new captures; keep source pointers next to each shot.
- [ ] **Objection/FAQ sheet for comment sections** (the single most-read doc on
      launch day). Prepare answers to: "How is this different from Optuna?" /
      "FLAML beat you in your own benchmarks?" / "What does 'AI' mean here?" /
      "Why scikit-learn-native instead of a new framework?" / "Is this
      maintained? What happens if the maintainer leaves?" / "Enterprise use:
      licensing, support?" — with links to sources for every claim.

## 3. Phase 1 — Launch content (write during Phase 0)

### 3.1 GitHub Release notes (v0.3.0)
Structure: one-paragraph story → highlights grouped by lifecycle stage →
honest benchmark paragraph → quick start (Python + Docker, each ≤6 lines) →
links (docs, blog post, DOI, roadmap incl. AI layer) → contributors/thanks.

### 3.2 Announcement blog post (~1200–1800 words)
Publish on RTD alongside `blog-honest-benchmark.md` and link from the Google
Site SKSurrogate page (the homepage section already exists). Title candidates:

- "Introducing SKSurrogate v0.3: one open-source environment from dataset to deployed, monitored model"
- "I built an end-to-end scientific MLOps tool — here's what it does, what it can't do yet, and the numbers to prove both"

Outline (mirrors the honesty that already won credibility):
1. **The problem:** a scientific ML workflow today is 5–6 stitched tools; none
   of them keeps research questions, evidence, or dataset identity together with
   the model lifecycle.
2. **What SKSurrogate is:** one diagram, the full lifecycle + research workspace.
3. **Four differentiators** (each with a concrete artifact link):
   - robustness/stability — brittle peak vs stable plateau in hyperparameter space,
     visible at a glance; the surrogate response surface makes the neighbourhood of the
     optimum inspectable. Artifact: `docs/images/stability_contour.png` (regenerate via
     `docs/images/make_stability_contour.py`; caption and allowed claims live in its docstring);
   - statistical evaluation — repeated nested CV, paired comparison protocol
     checks, split-conformal intervals with coverage diagnostics;
   - research workspace — projects, hypotheses with status lifecycle, evidence
     records with artifact snapshots, experiment design (LHS + GP-LCB) for
     expensive physical experiments;
   - surrogate hybrid search — equal-budget benchmark vs Optuna/FLAML/random,
     published with raw data DOI and an honest read of where it loses.
4. **What it is not:** not an autonomous AI assistant. What ships is opt-in,
   provider-agnostic synthesis — your endpoint, computed aggregates only,
   stateless on our side, citations enforced, nothing persisted without an
   explicit action. With no endpoint configured the tool still works,
   air-gapped.
5. **Quick start** (Python snippet + docker compose line + demo video link).
6. **Roadmap & call to action:** issue tracker, docs, "tell us what your
   workflow looks like."

### 3.3 Supporting posts (schedule for weeks 2–4 post-launch, not day one)
- [ ] Deep dive: experiment design / active learning for wet-lab and physical
      scientists (the most original feature — least competition).
- [ ] Deep dive: conformal prediction intervals + what "90% coverage on a small
      holdout" does and doesn't mean (continues the honesty brand; strong HN/Reddit material).
- [ ] Deep dive: evidence-tracked research projects ("your models, with receipts").

## 4. Phase 2 — Launch sequence and channels

**Timing:** Tuesday or Wednesday morning, ~08:00–10:00 Pacific (HN peak). Do
not spread launch-day effort thinner than this ordering allows; the first 48 h
of comment presence matter more than channel count.

| # | Channel | When | What to do | Notes for a first-timer |
| --- | --- | --- | --- | --- |
| 1 | GitHub Release v0.3.0 | T-0 (morning) | Publish release + tag; verify PyPI upload landed (`pip index versions sksurrogate`) and RTD rebuilt on master | The anchor everything links to |
| 2 | Blog post live (RTD + Google Site link) | T-0 (same morning, after #1) | Post per §3.2; update homepage SKSurrogate page with "New: v0.3 announcement" | Homepage already lists the library — one-line edit |
| 3 | **Show HN** | T+0, ~9–11 am PT | Title pattern: `Show HN: SkSurrogate – End-to-end scientific MLOps environment (scikit-learn-native)`; first comment = maintainer intro + links to benchmark + video + "happy to answer technical questions" | Highest-leverage channel for a dev tool. Expect 2–8 h of live Q&A. Read the FAQ sheet before posting. Never argue; link evidence. One submission only — resubmissions are punished hard |
| 4 | Lobsters | T+1 day | Same post, different tone (more technical, less promo) | Smaller, very high signal audience |
| 5 | Reddit r/MachineLearning + r/Python | T+2–3 days | Not "I made X" — write the experiment-design or conformal deep dive (§3.3) as a genuine contribution and mention SKSurrogate in passing; self-promo rules vary, read each sub's sidebar first | Value-first framing is what gets upvoted here |
| 6 | Newsletter submissions (batch in one sitting) | T+0 → T+2 days | Python Weekly (submit.pyweekly.com), ML News (mlnews.io submission form), The New Stack / InfoQ community post if applicable. Use the media-kit blurb; attach video + benchmark DOI | Low effort, compounds for weeks |
| 7 | Awesome-list PRs | T+1 week | `awesome-automl`, `awesome-mlops`, `awesome-bayesian-optimization` (one-liner entries per their format) | Long-tail discoverability; do after the dust settles so maintainers see traction first |
| 8 | LinkedIn (personal profile) | T+0 or T+1 | Professional-network post: what you built, why it matters to scientific teams, links. Tag nothing; keep it factual with the demo GIF | Doubles as consulting positioning for Upwork/Toptal clients who find you here |
| 9 | X/Twitter | Only if active | Thread = launch tweet + video + benchmark table screenshot; otherwise skip entirely | Inconsistent presence looks worse than none |

**Explicitly not doing on day one:** paid ads, conference talks (submit later —
PyCon sprints/lightning talks are a natural follow-up once traction exists),
press outreach (no angle yet; revisit after 1 month of downloads).

## 5. Phase 3 — Post-launch operations (weeks 2–4)

- [ ] **Response SLA:** every issue/comment answered within 24 h with sources.
      Log recurring questions and fold them into docs/FAQ the same week.
- [ ] **Triage ritual (15 min/day, first two weeks):** new issues → label
      `bug` / `question` / `enhancement`; questions that reveal doc gaps become
      doc PRs — visible responsiveness is a launch asset in itself.
- [ ] **Metrics to watch (weekly snapshot into the progress log below):** PyPI
      downloads (pepy.tech), stars/forks, open/closed issues, HN/Lobsters points
      and comment sentiment, RTD pageviews if available, demo uptime (if hosted).
- [ ] **Success criteria for launch v1** (define before T-0 so the result is
      interpretable): e.g. ≥50 new PyPI downloads/day for 7 days, ≥25 stars,
      ≥3 substantive technical discussions (HN/Reddit/issues) where at least one
      led to a doc fix or feature request we act on. Missing star targets with
      strong issue quality is still a successful launch — the audience that
      matters here files issues and cites the DOI, not stars.
- [ ] **Follow-up cadence:** publish one §3.3 deep-dive every 1–2 weeks for a
      month; each doubles as fresh newsletter/Reddit material and feeds the
      Toptal engineering blog pipeline (the first pitch is already in motion).
- [ ] **Convert early adopters:** any team using it on real work → ask for a
      one-paragraph quote or a case-study issue thread; these become the next
      announcement's social proof.
- [ ] **Long tail (month 2+):** evaluate JOSS submission for SKSurrogate itself
      (the Irene JOSS track is already in motion — same author, same rigor bar);
      consider arXiv preprint of `paper/technical-note.md` if it reaches draft
      quality; Zenodo software DOI via CITATION.cff if not yet minted.

## 6. Risk register (first-launch failure modes)

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| "AI-equipped" claim challenged because the shipped synthesis is bounded, off by default and provider-dependent | Medium | Never say "AI-equipped". Say opt-in cited synthesis with the transparency framing (BYO endpoint, aggregate-only, air-gapped mode) and link `AI_LAYER02_D2_LLM_SYNTHESIS_SCOPE.md` |
| Quick start broken for a reader (env, port, deps) | Medium | §2.1 clean-verify gate: pip install + docker compose both verified on the exact version being announced |
| HN thread goes quiet or turns skeptical about benchmarks | Medium | FAQ sheet pre-written; honesty framing already established by the published benchmark post; respond with links, not opinions |
| 32 unpushed commits drift further before release | High if launch slips >1 week | Cut v0.3.0 in Phase 0 immediately; do not let announcement scope wait on Phase 4/5 completion (the documented features are already in master) |
| Maintainer bandwidth: comment duty + issue triage collides with other work | Medium | Block 2–3 h/day for the first 48 h explicitly; pre-answer top-10 FAQ questions so replies are fast and sourced |
| Demo instance (if hosted) goes down during HN peak | Low-Medium | Prefer self-hosted `docker compose` instructions over a live demo unless uptime is guaranteed; if hosting, monitor actively launch day |

## 7. What the announcement must not claim

- No LLM features exist yet — do not imply otherwise ("AI research assistant
  coming" with a roadmap link is fine; "asks questions in natural language" is not).
- The benchmark does **not** beat every baseline at equal budget (FLAML/TLA-style
  wins on several datasets at wall-clock parity per the blog post; Optuna-TPE
  edges SKSurrogate on 4 of 6 in the README table) — lead with what the numbers
  actually show: within ~0.002 everywhere, leads at 60-eval budget, ablation
  shows the surrogate earns its cost. The honesty is the brand; protect it.
- Do not claim multi-worker/production-scale MLOps beyond what the docs say
  (single-process project writes, bounded scopes — all documented accurately).

## 8. Effort estimate and sequencing

| Block | Work | Est. effort |
| --- | --- | --- |
| A | Version bump, changelog, PyPI upload, tag + first GH Release | half day |
| B | README rewrite + media kit + FAQ sheet | 1–2 days (writing-heavy) |
| C | Demo video + GIF + Colab pinning | half day (recording + editing) |
| D | Repo hygiene pass | 1 h |
| E | Blog post write + homepage one-liner | half day |
| F | Launch-day execution (§4 sequence) + comment duty | launch morning + 2–3 h/day × 2 days |

**Critical path:** A → B/E in parallel → C → D → T-0. Realistic minimum to a
well-executed first announcement: **~1 focused week**, launch on the following
Tuesday/Wednesday.

## 9. Decisions needed from the maintainer (blocking)

1. ~~**Positioning:**~~ **RESOLVED 2026-11-07: Option A.** "End-to-end scientific
   MLOps environment"; AI framed as shipped features + explicit LLM roadmap; no
   build before launch.
2. ~~**Version & timing:**~~ **RESOLVED 2026-11-07:** cut v0.3.0 now, target a
   Tue/Wed launch ~1 week out (≈ Nov 17/18). **Scope correction pending:** the
   Phase 4/5 features live on dev branch `ailayer`, not `master` — see new
   decision 4 below.
   **AMENDED 2026-11-09:** launch moved to December (working target Tue/Wed ≈ Dec
   15/16) — the maintainer's real-dataset testing takes several weeks and gates the
   announcement; see decision 5.
3. ~~**Channels to staff:**~~ **DEFERRED 2026-11-07 by maintainer** ("postpone
   and discuss this phase in detail"). Channel table in §4 remains the working
   default until discussed.
4. ~~**Release scope (branch topology):**~~ **RESOLVED 2026-11-07: Option (c).**
   Hold the cut ~2–3 days; stability pass on `ailayer` first — (i) full test run,
   (ii) docs cross-link audit, (iii) changelog draft for v0.3.0 — then merge
   `ailayer` → `master`, verify CI green, and cut v0.3.0 with the workspace +
   evidence tracking included. **Launch target AMENDED 2026-11-09** to December
   (working target Tue/Wed ≈ Dec 15/16; see decision 5).
5. **New — launch gate and date amendment (added 2026-11-09):** the maintainer runs
   SKSurrogate on real datasets over the coming weeks and provides per-feature
   opinions; the announcement must not outpace that feedback. Working launch target:
   **Tue/Wed Dec 15/16** (was Nov 17/18). Final date to be confirmed as test results
   land (~Dec 5); if testing slips, the date moves within December, with content
   freeze ~one week before T-0. Tracked in Vikunja project #40 as a Phase 1 gate task.

## 10. Tracking and open discussions

**Vikunja:** project **40 — "SKSurrogate public launch (v0.3)"** (created 2026-11-07,
Gantt/Kanban/Table views active). Structure: Phase 0 #866 (subtasks #873–#875),
Phase 1 #867 (#876–#879), Phase 2 #868 (#880–#884), Phase 3 #869 (#885–#887);
open discussions: licensing #870, website #871, channels #872. Due dates run
Nov 9 (stability) to Nov 30 (metrics).

### Open discussion: licensing (task #870, decide by ~Nov 12)

Current state: MIT declared in `setup.py` (`license="MIT"` + classifier) and
CITATION.cff, reflected on PyPI — but **no LICENSE file exists**, so GitHub shows
"No license" for the repo. Options:

- **(a) Plain MIT everywhere** (recommended default): add a standard `LICENSE`
  with the correct copyright-holder name before the v0.3.0 cut; one-line "MIT
  licensed, free" in README/media kit. Zero friction for scientific adoption.
- **(b) MIT core + optional commercial support tier:** same code stays fully
  open; paid SLA/support contracts sit on top (monetizes enterprise buyers via
  consulting rather than a license wall). Requires a short SUPPORT note or
  landing-page section and changes the "free" line in all launch copy.

**Resolved 2026-11-07:** Plain MIT everywhere; add the standard `LICENSE` file
(Copyright (c) 2026 Mehdi Ghasemi) before the v0.3.0 cut — staged as **P1.5**.
**Committed 2026-11-08** in `fb5b682` (`LICENSE` is now tracked; the wheel ships it
in `dist-info/licenses/LICENSE`). No paid support tier for launch.

### Open discussion: dedicated website (task #871, decide by ~Nov 14)

Current state: docs on ReadTheDocs (built from `master`) + a SKSurrogate section
inside the Google Site personal homepage. Options:

- **(a) Keep as-is** — zero cost; landing experience is a section of a personal site.
- **(b) GitHub Pages project site for `mghasemi/sksurrogate`** (recommended
  minimum): free static landing page with one-liner, demo GIF, links to
  RTD/PyPI/GitHub; still a github.io URL.
- **(c) Custom domain on a free tier** (Cloudflare Pages / Netlify / GitHub
  Pages), e.g. `sksurrogate.dev` or similar — brand polish + SEO-friendly URL;
  domain availability to be checked at decision time.

**Resolved 2026-11-07:** Option (b) — GitHub Pages project site for
`mghasemi/sksurrogate`, staged as **P2.6**, due Nov 14. Docs stay on RTD; the
Google Site section keeps linking out to it. Announcement links will point at
the new landing page once live, with RTD as the docs link throughout.

## Progress log

- 2026-11-07: Maintainer decisions recorded — Option A positioning; cut v0.3.0
  now, launch ≈ Nov 17/18 (Tue/Wed); channel selection deferred for a dedicated
  discussion. Branch topology verified: the 32 commits ahead of v0.2.0 live on
  dev branch `ailayer` (pushed to origin), not master — new blocking decision 4
  added (release scope: merge ailayer first vs docs-only release). Awaiting that
  answer, then Block A (version bump + changelog) starts.
- 2026-11-07: Draft v1 written after verifying repo state (tags, PyPI 0.2.0, no
  GitHub Releases, master ahead by 32 commits, no LLM features in code, README
  positioned as drop-in tuner). Awaiting the three decisions in Section 9.
- 2026-11-07: Stability pass started (P0): full test run in Docker py3.12
  (pytest + `examples/example02.py` smoke, mirroring CI) running in the
  background; log at advisor scratch dir `sk-stability-run.log`. Early output
  already shows several failing tests on `ailayer` — triage when it lands.
- 2026-11-07: Vikunja project **#40 "SKSurrogate public launch (v0.3)"** created;
  all phases + open discussions staged as tasks #8–#24 (see section 10).
- 2026-11-07: Decisions recorded — website = GitHub Pages project site for
  `mghasemi/sksurrogate` (P2.6, due Nov 14); licensing = plain MIT, LICENSE file
  drafted in working tree (Copyright (c) 2026 Mehdi Ghasemi), commit staged as
  P1.5. Discussion tasks #870 and #871 closed; channel selection (#872) remains
  open for a dedicated discussion.
- 2026-11-07: Launch copy updated per HANDOFF-explainability-visibility.md — explainability added to README; differentiators reordered with robustness/stability as #1 (stability contour figure); media kit assets extended.
- 2026-11-08: **Logging/telemetry merged; v0.3.0 cut started.** Branch `logging-telemetry`
  (L0–LQ, commits `964e5b6` + `437676c`) merged into `ailayer` as `406bbc0` (merge dry-run was
  conflict-free; suite still 362 passed / 32 subtests). Release prep committed as `fb5b682`:
  `CHANGELOG` `Unreleased` → `[0.3.0] - 2026-11-08` (48 commits since v0.2.0 — the plan's "32" was
  stale), version `0.3.0` in `SKSurrogate/__init__.py` and `docs/conf.py`, tracked MIT `LICENSE`
  (P1.5), and a new `docs/telemetry.rst` chapter wired into the toctree and cross-linked from the
  hyperparameter-optimization chapter. Sphinx HTML build exit 0 with zero warnings; the rendered
  page carries the licence/logger/audit contracts and the seed-trial `phase=None` caveat.
  Wheel `sksurrogate-0.3.0-py3-none-any.whl` built (from a clean copy: the repo has a root-owned
  stale `build/` from a docker run that needs `sudo` to remove) and installed into a fresh
  `--system-site-packages` venv, which reports 0.3.0 from site-packages, runs the quick start plus
  callbacks/audit export, and ships `dist-info/licenses/LICENSE`. Annotated tag `v0.3.0` created
  locally. **Owner actions remaining:** push `ailayer` (and merge to `master`), PyPI upload,
  GitHub Release, then the index-install + docker round trip (P1.3/P1.4). Nothing pushed.

- 2026-11-08: **Pushed and retagged.** Owner instructed "Push"; `ailayer` went to origin
  (`0f8c96f..4662725`, then `..c19a544`) and `v0.3.0` was published. Inspection showed the tag
  pointed at the release-prep commit `fb5b682`, i.e. *without* the Logs feature; the owner chose to
  move it rather than ship 0.3.1, so `CHANGELOG.rst` folded `Unreleased` into `[0.3.0] - 2026-11-08`
  (`c19a544`), the tag was recreated on that commit and force-pushed (tag object 415ac01 → 8cd84c4,
  peel `c19a544` = branch head). Verified with `git ls-remote` and `git ls-tree v0.3.0`, which lists
  `api/logs.py`, `api/routers/logs.py`, `webui/src/pages/Logs.tsx` and `docs/telemetry.rst`.
  No GitHub Release exists yet and no CI run fired for the branch (workflow targets master/PRs).
  Remaining: PyPI upload and the first GitHub Release (owner credentials), then the index-install
  and docker round trip (P1.4).

- 2026-11-09: **Licence cleanup + final retag.** The tag shipped two licence files (`LICENSE`
  © 2026 and the older tracked `License.txt` © 2019). Owner asked to keep one; `License.txt` was
  deleted, the single `LICENSE` now reads `Copyright (c) 2019-2026 Mehdi Ghasemi` (a year range is
  standard MIT phrasing), `MANIFEST.in` includes `LICENSE`, the readme badge points at it and
  `setup.py` declares `license_files=["LICENSE"]` (commit `604ac75`). Tag `v0.3.0` moved to that
  commit (`eb314c9` -> `604ac75`) and force-pushed; `origin/ailayer` is the same commit. Artifacts
  rebuilt from the tag into `/home/mehdi/dist/sksurrogate-0.3.0/`: wheel
  `6904f350…0702cb` (123,763 B) and sdist `a0603406…73036` (7,705,916 B). Verified: one licence entry
  in the wheel (`dist-info/licenses/LICENSE`, 2019-2026), only `LICENSE` in the sdist, both install
  into fresh venvs and report 0.3.0. Ready for twine upload + GitHub Release.

- 2026-11-09: **Launch pushed to December.** The maintainer will test SKSurrogate on real
  datasets over the coming weeks and give per-feature opinions; that work passes the Nov 17/18
  window, so the launch target was amended to **Tue/Wed ≈ Dec 15/16 (working target** — final date
  confirmed as test results land ~Dec 5; see new decision 5). Vikunja project #40 updated: open
  Phase 1–3 due dates shifted ~4 weeks (merge/release cut Dec 10–11, content freeze Dec 12–14,
  launch day Tue Dec 15, comment duty through Dec 17, weekly metrics through mid-Jan 2027), and a
  gate task "Real-dataset test by maintainer — per-feature feedback" (due Dec 5) was added under
  Phase 1. `AI_LAYER02_REMAINING_PLAN.md` launch references updated to match. Dated entries above
  are historical records and were not rewritten.

- 2026-11-09: **Product website deployed (Phases 1–5 of the website build).** Single static site
  built from the Atelier design (`/home/mehdi/Code/Atelier/designs/sksurrogate`): graphite /
  ultramarine / ember, S1–S13 per `content-plan.md`. Copy is verbatim where mandated — headline +
  positioning (LAUNCH_PLAN §1), benchmark table + caption reproduced from the blog "Finding 1"
  with the verified claim **"TPE is top on five of six datasets"** (owner-confirmed: TPE stays 5 of
  6, evidence supports it), D-track regime map from Finding 2, stability-figure caption verbatim
  from `make_stability_contour.py`, quick-start snippet verbatim from the readme, DOIs + repro
  commands verbatim. Version strip is **v0.3.0** (owner decision; in progress). All figures/numbers
  carry inline source comments; no links to ailayer-only docs until the v0.3.0 cut lands them on
  master/RTD — D-track uses the durable Zenodo DOI instead. Deployed on an orphan `gh-pages`
  branch (site files at root only, commit `a92c58a`) → live at the default URL
  https://mghasemi.github.io/sksurrogate/ . Pre-release QA all green: every local asset and remote
  link 200; canonical + og:url point at **sksurrogate.com** (apex) per content-plan S1 — that
  domain is still pending the Cloudflare purchase, so CNAME + apex DNS stay in the launch-day
  switchover and the default URL must not be shared until then. Custom 404 page verified live.
  Remaining at launch: PyPI upload of v0.3.0 (artifacts ready per entry above), first GitHub
  Release, domain purchase + DNS flip; if v0.3.0 slips past launch day the strip is a one-line bump.


- 2026-11-10: **A2 applied — the "no LLM" claims were stale and are corrected.** Stage D2 shipped
  bounded opt-in synthesis, so the verified-state row, the positioning analysis, the Option B block,
  the blog outline item 4 and the risk-register row were rewritten around what actually ships:
  provider-agnostic, aggregate-only, stateless on the SKSurrogate side, citation-enforced, off by
  default, air-gapped with no endpoint. Deliberately did **not** claim provider-side zero retention —
  the endpoint is user-supplied, so retention is the user's provider's policy, and that distinction is
  now stated explicitly. `AI_LAYER02_FOLLOWUP_PLAN.md` still carries "Phase 7 is not started" at
  line 11 and remains the one stale source; `AI_LAYER02_REMAINING_PLAN.md` is authoritative.
