# B2 — Artifact-Recovery Decision (Stage B, R3)

**Internal working document.**

**Status:** Drafted 2026-11-08 on branch `ailayer`; **decision accepted by Mehdi
2026-11-08** — commit to "detect and report only" (§4). §3 landed the same day:
per-status behavior contract in `docs/research-projects.rst` plus pinning tests
for `unverified` legacy records and no-rebaseline-on-edit
(`tests/test_projects_endpoints.py`); full suite 302 passed / 32 subtests in a
clean environment, Sphinx HTML build clean. Tracked-file changes left
uncommitted pending review; this item closes R3 with the §4 upgrade path intact.
This is the Stage B/B2 deliverable of
[`AI_LAYER02_REMAINING_PLAN.md`](./AI_LAYER02_REMAINING_PLAN.md): decide between content-level
snapshots under a size cap and "detect and report only", with legacy v1 records readable either way.

## 1. Question and evidence base

The snapshot mechanism as it exists today (all locations verified in source):

- One snapshot per linked artifact is captured at first link (`add_project_artifact`,
  `api/routers/projects.py` :455; project creation via `_serialize_artifacts`), then preserved
  verbatim by every later `PUT` (:297) — it is **never re-baselined**.
- A snapshot is schema `sksurrogate.project-artifact-snapshot.v1`: `{schema, captured_at,
  identity, fingerprint, details}` (:278–284). The fingerprint is a sha256 over canonical JSON of
  identity + details; per-kind details are *metadata*: partition → the mltrace split record's
  `fingerprint/ingestion_timestamp/rows/columns/target/feature_count` (:226–239); bundle → the run
  configuration's identity, lineage, run, and dependencies (:240–251); job → stable fingerprints of
  its recorded configuration and result (:252–268).
- Drift checking happens at summary time: for each still-existing artifact the snapshot is recomputed
  and compared → `current` / `changed`; validation failure → `missing`; no stored snapshot (legacy)
  → `unverified` (:386–393).
- Storage: one JSON file per project under `{root}/projects/`, written atomically via tmp-file +
  rename under a single process lock (:166–175); layout in `api/config.py`. Upstream content lives
  elsewhere: partition rows in the task's SQLite db (`mltrace/{task}.db`), bundle files (`.bundle`)
  under `bundles/` / `registry/`, job records at `jobs/{job_id}.json`.
- Existing documentation already carries a one-line no-recovery contract — "a deleted artifact cannot
  be recovered from project metadata" (`docs/research-projects.rst`:69–70); existing tests pin the
  `changed` / `current` / `missing` transitions and snapshot preservation on re-PUT
  (`tests/test_projects_endpoints.py`:399–456).

The structural fact that drives the decision: **v1 snapshots already embed everything small and
structured** — partition schema metadata, the full bundle reproducibility record, job config/result
fingerprints. A content-embedding scheme would add only three things not present today: partition row
data, job-result payloads, and model weights.

## 2. Option A — Content snapshots below a size cap (rejected for now)

What it requires:

1. Per-kind content extraction plus size caps, and storage accounting (bytes stored per snapshot and
   per project), surfaced somewhere in the API or docs.
2. A storage-location decision that collides with the current store's invariants: inline content
   bloats every `list_projects` / `GET` / summary call, because those parse *all* project files whole
   (:326–338); sidecar files under `projects/` break the single-file atomic-rename write invariant
   (:169–175) and add a second failure domain to a store that B4 deliberately kept minimal (the
   SQLite migration was rejected with rationale for exactly this kind of growth).
3. A restore path: re-materializing captured content into upstream stores is a new *write* API against
   the mltrace db / bundle directories / job records — each with its own invariants (e.g., partition
   re-registration through the tracker) — plus provenance flagging so restored copies are never
   presented as current state.
4. A semantics hazard: captures are one-shot and never refreshed (:297). Restoring a *changed*
   artifact would silently substitute stale content; the next summary would then report `current`
   (the re-materialized copy matches its own fingerprint), turning an honest drift check into a
   self-fulfilling record — directly against what the shipped evidence-snapshot behavior promises.

What it buys: recovery of *deleted* artifacts below the cap — partition rows and job records (usually
small); model weights are never covered by any useful cap (MB–GB). No adopter has requested this, and
Stage B is post-launch by design.

## 3. Option B — "Detect and report only" (recommended)

Commits to the current honest behavior *as the* behavior:

- `current` / `changed` / `missing` / `unverified` remain the complete status vocabulary; snapshots
  stay fingerprint + details, captured once at link time and preserved on edits.
- Durability is a property of the upstream stores (mltrace db, bundles, job records), not of
  `projects/` — documented as an operational fact with backup guidance; recovery is out of scope by
  design, for the reasons in §2.

Bounded work (this item's "implementation"):

1. `docs/research-projects.rst`: replace the one-line no-recovery sentence with a per-status behavior
   contract — which upstream events produce each status for each kind; that snapshots are never
   re-baselined and there is no supported way to reset drift on an artifact referenced by evidence
   (evidence-referenced artifacts cannot be removed via `PUT`, :470–479); that `unverified` marks a
   legacy record without a stored snapshot; durability/backup guidance.
2. Tests in `tests/test_projects_endpoints.py`: pin the two gaps — a legacy record without a
   `snapshot` field reports `unverified`; `PUT` preserves prior snapshots (extending the existing :399
   test). Full suite in a clean environment (`env -u PYTHONPATH ./.venv/bin/python -m pytest tests`).
3. No schema change: v1 stays as-is; when B1's new kinds land they stamp snapshot schema **v2 without
   content fields** (per B1 §3), so the "v2" name is spent on additive kind support, not recovery —
   and readers stay tolerant of v1 by construction.

## 4. Decision

**Commit to Option B — detect and report only.** Rationale: (i) no demonstrated demand exists pre-
launch; (ii) v1 already embeds all small structured details, so embedding would add only row data /
job payloads / weights at real storage and API cost; (iii) one-shot captures make restore a provenance
hazard for the drift check we sell; (iv) the current store's invariants — whole-file reads on every
listing, single-file atomicity, single process (B4) — are designed against exactly this kind of growth.

Upgrade path (recorded so R3 can re-open cheaply): if post-launch usage shows concrete recovery demand
for a specific kind (most plausibly job results), implement a per-kind optional `content` field under
the existing snapshot schema v2 with an explicit size cap and stored-bytes accounting, plus a flagged
re-materialization endpoint that keeps the original `captured_at` and labels restored content as such.
Readers stay tolerant by construction, so this needs no migration. Entry gate for that flip: at least
one real adopter request naming the kind; otherwise this decision stands.

## 5. Dependencies and sequencing

- **B1 interaction:** under this decision `file_artifact` (B1 tier 2) ships with content *detection*
  only — its v2 snapshot carries `{size_bytes, sha256}`, no embedded bytes (consistent with B1 §3/§5).
- **Entry gate:** acceptance of the decision in §4. Stage B's post-launch design governs implementation
  timing; because the work is docs + tests only and behavior-neutral, it may land before the cut if
  preferred.
- **Exit:** docs section landed and tests pass in a clean environment → R3 closes with rationale
  recorded here and in `AI_LAYER02_REMAINING_PLAN.md` (Stage B / R3). Legacy v1 records remain readable
  throughout, by construction.
