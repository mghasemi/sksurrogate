# B1 — Artifact-Ingestion Scoping (Stage B, R1)

**Internal working document.**

**Status:** tier-1 ``reference`` **implemented 2026-11-08** (API, UI, tests, docs) with the
post-launch gate waived by Mehdi for this bounded kind; ``file_artifact`` remains gated on
post-launch adopter feedback per §5. Drafted 2026-11-08 on branch `ailayer`; typed-superset
decision accepted by Mehdi the same day. This is the Stage B/B1 deliverable of [`AI_LAYER02_REMAINING_PLAN.md`](./AI_LAYER02_REMAINING_PLAN.md):
which artifact kinds beyond dataset partitions, model bundles, and jobs do users need to
link; a typed-superset-vs-closed-set decision; and the API design + test plan written
*before any code*.

## 1. Question and evidence base

The project API links exactly three kinds today — `dataset_partition`, `model_bundle`,
`job` (`ProjectArtifactReference`, `api/routers/projects.py`) — each validated at link
time against the per-task mltrace store, bundle/registry directories, or job records;
snapshots are fingerprint-only (schema `…artifact-snapshot.v1`), so they detect drift but
never recover content.

Stage B is post-launch by design: **no first-adopter feedback exists yet** at drafting
time. The scoping below therefore draws on in-repo evidence only — where research
artifacts actually live, what the workflows already produce, and what earlier phases
recorded as deferred:

| Candidate kind | Where it lives today (verified) | Signal |
| --- | --- | --- |
| External reference (URL / DOI / citation) | No home in the system at all | Findings/conclusions must cite literature; notes are text-only and cannot carry evidence relations, so there is no legal way to attach a citation as evidence. Strongest scientific need; zero system risk. |
| In-root file artifact (exported run configs/reports, monitoring logs/alerts) | `exports/{task}/{version}-<ts>/` (`api/routers/bundles.py`), plus `monitoring/`, `predictions/`, `checkpoints/` under the API root (`api/config.py`) | P2.2 already produces a machine-readable run configuration and Markdown report per bundle; D2 names drift reports as synthesis input. No way to cite "this file" as evidence today. |
| Figures (contours, curves) | Script/notebook/benchmark byproducts only (`docs/images/make_stability_contour.py`, `benchmarks/plot.py`); **no API job produces images** | Real but indirect: the producing job is linkable; figures have no stored location under root. Served by a file kind only once files get a home there. |
| Benchmark archive (`benchmarks/results/<ts>/`) | Repo-relative, outside the API root by construction (raw data → Zenodo) | Stage D's evidence will be exactly these v2-schema archives; direct linking would couple projects to repo layout — defer until D4 makes archives first-class. Cite via `reference` once published with a DOI/URL. |
| Notebooks | Source-only under `docs/`; runtime deliberately out of scope (roadmap P2.3) | Stays rejected per the standing decision: no notebook service in the package runtime. |
| Open / untyped URIs | — | **Rejected.** Breaks existence validation, makes snapshot status meaningless; conflicts with the honest-provenance rule. |

## 2. Decision — bounded typed superset, closed beyond it

The linkable set becomes `{dataset_partition, model_bundle, job} ∪ {reference, file_artifact}`:

- `reference` (tier 1) — external citation, no system access; zero-risk, adopted first.
- `file_artifact` (tier 2) — regular file under the API root at link time, size-bounded;
  its snapshot is the first *content-level* fingerprint in the project store, which makes
  B2's detect-only-vs-embed decision the natural home for this kind (§5).

Everything else stays out until post-launch usage shows a concrete need. Adding kinds later
is additive (new enum value + validator branch), so deferring has no migration cost — that is
why "typed superset" rather than "open URI" is the right boundary: the set stays closed by
construction and extends one validated kind at a time.

## 3. API design (design only — no code before launch)

No new endpoints; existing create / `PUT` / `POST …/artifacts` / records / evidence / summary
endpoints extend per kind:

| | `reference` | `file_artifact` |
| --- | --- | --- |
| Shape | `{kind, url, title?, doi?, label?}` — no task scope (external) | `{kind, path, label?}` — `path` relative to the configured API root (`exports/…`, `monitoring/…`) |
| Validation at link time | scheme http(s), non-empty netloc, `url` ≤ 2048 chars; the API **never fetches** the URL | resolves within root (traversal / absolute / symlink escape rejected), regular file exists, size ≤ 10 MiB |
| Snapshot (v2) | fingerprint of stored fields; summary status always `unverified` — reachability is not checked and that is documented as such | `{size_bytes, sha256}` of content; re-read at summary time → `current` / `changed` / `missing` |
| Dedup key | `(kind, url)` | `(kind, path)` |

Snapshots for new entries stamp `sksurrogate.project-artifact-snapshot.v2`; the project file
schema stays `research-project.v2` (fields are additive). Readers stay tolerant: v1 snapshots
and projects without the new kinds load unchanged, and `_serialize_artifacts` keeps prior
snapshots on re-PUT exactly as it does today.

**Semantics that do not change:** the stage checklist remains driven only by
partitions/bundles/jobs — references and files never advance a stage (documented explicitly).
Evidence rules are identical: linked to the project first, ≤ 20 per record, findings/conclusions
require ≥ 1 evidence (a `reference` may satisfy that), artifacts used as evidence cannot be
removed via `PUT`. Caps unchanged (100 artifacts/project, 1000 records).

## 4. Test plan (write before code; mirrors `tests/test_projects_endpoints.py`)

1. **Reference:** accepts http(s); rejects other schemes, blank URLs, missing netloc, over-length;
   idempotent re-append for the same URL; a reference used as evidence in a finding/conclusion
   round-trips; summary reports `unverified`; tests make no network call (no fetch path exists to exercise).
2. **File artifact:** links an existing small file under the test root; rejects `..` traversal,
   absolute paths, symlink escape, missing files, over-cap sizes; content sha256 stable across re-reads
   and re-PUT; mutate bytes → summary `changed`; delete → `missing`.
3. **Legacy tolerance:** a fixture project JSON whose field shape is copied from a real v1 store (no new
   kinds, v1 snapshots) loads through list/get/summary unchanged; PUT preserves prior v1 snapshots for old artifacts.
4. **Caps and guards:** the 100-artifact cap counts all five kinds; evidence-referenced artifacts of any
   kind cannot be removed via `PUT`; the per-record 20-evidence cap is unchanged.
5. **Gates:** full suite in a clean environment (`env -u PYTHONPATH ./.venv/bin/python -m pytest tests`);
   Sphinx build after adding the `docs/research-projects.rst` section; `git diff --check`.

## 5. Dependencies and sequencing

- Stage B is post-launch by design, but Mehdi waived that gate for the tier-1 `reference`
  kind on 2026-11-08; the rest of this section still governs `file_artifact`.
- **Landed 2026-11-08 (tier 1):** `reference` is implemented as specified in §3 — model/validator
  in `api/routers/projects.py`, v2 field-only snapshot, always-`unverified` summary status,
  URL dedupe, evidence support, UI form and external-link rendering, docs in
  `docs/research-projects.rst`/`docs/ui-manual.rst`, and tests in
  `tests/test_projects_endpoints.py` (roundtrip/validation, evidence + removal protection,
  backward-compatible shapes, HTTP surface).
- `file_artifact` waits on **B2**: under "embed below cap", snapshot v2 carries an optional content field
  for small files (recovery); under "detect only" it ships with sha256 detection and the no-recovery behavior
  is documented in `research-projects.rst`. Either way, readers stay tolerant of v1.
- **Exit:** scoping accepted *and* at least one new kind confirmed by real usage → bounded implementation per
  §3–4; otherwise the set stays closed as-is and this document records that rationale for R1's closure.
