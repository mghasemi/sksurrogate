# Stage D2 — LLM-assisted project synthesis: scope and contract

**Internal working document.**

**Status:** contract **approved 2026-11-08** by Mehdi, with an explicit waiver of
the plan's *post-launch API stability* entry gate for this cycle. The bounded
opt-in implementation landed the same day in `api/synthesis.py`,
`api/synthesis_facts.py`, `api/routers/synthesis.py`, the project-note schema,
and the Projects UI. D4/D5 remain gated and untouched.

**Item:** `AI_LAYER02_REMAINING_PLAN.md` Stage D2 / R8 — "bounded workflow
combining dataset profile + run configuration + explanation summaries (+ drift
report) into cited project notes via a user-supplied model endpoint; generated
interpretation clearly separated from computed results; core stays
dependency-free (opt-in)."

## Bounded use case

Draft **one** short project note that summarises *already computed* artifacts of
one project: a dataset profile, an optional run/job record, a model bundle, and
optional explanation summaries and monitoring alerts. The draft must cite the
computed sources inline; it is displayed next to those computed facts and is
saved only when the user explicitly adds it as a note.

Explicitly **not** in scope: chat/multi-turn assistance, automatic note
creation, free-form questions, data inspection beyond the served aggregates,
model selection or evaluation, and any autonomous action.

## Provider contract

- **OpenAI-compatible chat completions** (`POST {base_url}/chat/completions`)
  over the Python standard library only (`urllib.request`). No SDK, no new
  dependency, no import at `SKSurrogate` core level.
- **Disabled by default.** Configuration is read from the environment at call
  time; a missing base URL or model is a `503` with an actionable message and
  *no outbound request*.

| Variable | Meaning |
| --- | --- |
| `SKSURROGATE_LLM_BASE_URL` | Provider base URL, e.g. `https://api.openai.com/v1` |
| `SKSURROGATE_LLM_MODEL` | Model identifier sent in the request |
| `SKSURROGATE_LLM_API_KEY` | Optional bearer token; never echoed in responses or logs |
| `SKSURROGATE_LLM_TIMEOUT` | Optional request timeout, seconds (default 60) |
| `SKSURROGATE_LLM_MAX_TOKENS` | Optional output budget (default 700) |

Requests use `temperature: 0`, a bounded response read (1 MiB cap), and no
automatic retries. HTTP errors, timeouts, unreachable hosts, and non-conforming
responses become explicit failures naming the cause.

## Computed sources (what may leave the process)

Facts are assembled from stored artifacts and reduced to aggregates:

| Id | Kind | Contents |
| --- | --- | --- |
| `S1` | dataset profile | row counts, sampling metadata, feature count, per-column `kind`/missing rate/uniqueness/constant flags, target descriptor, top 5 target associations, data-quality flags |
| `S2` | run (optional) | job kind/status/timestamps, whitelisted run configuration keys, scalar result fields and metric scalars |
| `S3` | model bundle | metrics, created-at, feature count and names, preprocessing flag, dataset-fingerprint prefix |
| `S4` | explanations (optional) | importance means/stds (top 5), ALE ranges per feature (top 3), skipped features, availability notes |
| `S5` | drift alerts (optional) | recent persisted monitoring alerts for the version (max 10) |

**Never sent:** dataset rows, row-level values, partition file paths, model
objects, credentials, environment variables, or arbitrary files. The module
that talks to the provider receives only the rendered JSON facts.

## Prompt and citation contract

- The system message forbids inventing numbers, features, datasets, or runs and
  requires computed facts and interpretation to stay distinct and non-causal.
- Every factual sentence must carry a `[S1]`…`[S12]` citation. The server
  rejects a draft that is empty, longer than 6,000 characters, cites no source,
  or cites an id that was not supplied. A rejected draft fails the job with the
  reason; it is never silently accepted.
- The draft is capped at 220 words by instruction and 12 sources by contract.

## API and persistence contract

- `GET /api/synthesis/status` → `{enabled, provider{configured, model, host,
  timeout_seconds, max_tokens, protocol}, instructions}`. Credentials never
  appear.
- `POST /api/synthesis/projects/{project_id}` → queued `project-synthesis` job.
  The request records `task_name`, optional `model_version`/`job_id`, partition,
  source toggles, and an optional ≤500-character instruction.
- The job result contains `draft {text, citations, word_count}`, `sources`
  (id/kind/label), `computed_facts` (the exact payload sent), `provider`, and an
  `interpretation` block with `category: "generated_interpretation"` and
  `causal_effect: false`.
- **Persistence is explicit.** The draft is not added to the project by the
  synthesis job. `POST /api/projects/{id}/notes` accepts `source: "generated"`
  with `model`, `citations`, `sources`, `computed_facts`, and
  `synthesis_job_id`; it rejects generated notes without citations, without a
  model id, with citations that do not match a supplied fact, or with a facts
  payload above 30,000 characters. Human notes keep their shape and gain
  `source: "human"`. Notes remain append-only.

## Data-handling review (recorded)

- **What leaves the process:** the aggregate facts only, rendered into a
  prompt. Nothing else from the store is read for the request.
- **What is stored:** the prompt is **not** persisted, but the job record
  persists the result (draft, sources, facts) exactly like every other job
  result, under the configured jobs directory and retention rules. Generated
  notes persist the same payload by explicit user action; computed facts are
  capped at 30,000 characters.
- **Provider trust:** the endpoint is user-supplied, so the operator is
  responsible for the provider's retention and logging behaviour. The feature is
  off unless the operator configures it.
- **No training feedback loop:** drafts are never used to fit or tune models,
  and the feature does not write to datasets or bundles.

## UI contract

Projects → **Draft synthesis** card: provider status, task/bundle/job selectors
populated from the project's linked artifacts, explanation/drift toggles, an
optional instruction, and a **Generate draft** action. The result is rendered as
*Generated interpretation* with citation chips, a **Computed facts sent** table,
the raw computed-fact payload in a collapsible block, and the limitation list;
**Add as note** persists it. Generated notes in the Notes list carry a
`generated` badge, their citations, model id, and collapsible computed facts.

## Test plan (implemented)

`tests/test_api_synthesis.py` (7 tests, local stub HTTP provider, no network):

1. Status reports unconfigured without leaking secrets.
2. Submission is refused with `503` when unconfigured.
3. Happy path: cited draft, source ids, computed facts, provider metadata,
   authenticated request shape, aggregate-only prompt (no row payloads, no CSV
   path), bounded prompt size, provider called exactly once.
4. Unknown citation fails the job (`unknown source`).
5. Uncited draft fails the job (`cites no sources`).
6. Provider HTTP 500 surfaces in the job error.
7. Generated notes require provenance (model, matching citations) and persist
   it; human notes stay backward compatible.

## Explicitly deferred

Streaming responses, multi-turn interaction, automatic note creation, provider
retries/cost accounting, non-OpenAI protocols, tool calling, sending raw rows or
documents, and LLM summaries of row-level explanations (which depend on D3
outputs but are not part of this layer).
