# Contributing to SKSurrogate

SKSurrogate is a surrogate-based optimization library with a scientific control
plane on top. Contributions are judged on **evidence**, not on feature count.

## Scope rules (from the 2026-11 reliability review)

1. **No new optimizer without benchmark evidence.** A new acquisition strategy,
   surrogate family or search heuristic lands with a pre-registered protocol
   (`benchmarks/DESIGN.md`), a paired comparison against the incumbent, and
   intervals — not with a single run that happens to win. The existing
   statistics live in `SKSurrogate/compare.py`; use them.
2. **No general-purpose MLOps expansion.** The platform exists to run scientific
   experiments (datasets → optimize → compare → explain → validate → deploy),
   not to become a model registry for production traffic. If a request can be
   answered by a library call plus an artifact, prefer that.
3. **No LLM-first differentiation.** Generated text is interpretation, never a
   result. Drafts must cite computed facts, label every sentence
   `[observed]` / `[inferred]` / `[hypothesized]`, and pass the causal-language
   gate in `api/synthesis.py`. A feature that only works with a model in the
   loop is not a core feature.

## What a change must ship

- A failing test first, then the fix (see `tests/test_search_invariants.py` for
  the style used on the optimizer).
- Documentation next to the code: a `docs/*.rst` section for user-visible
  behavior, a `docs/changelog/` entry, and the UI manual when the web UI
  changes.
- Numbers, not adjectives: any performance claim quotes the run, the protocol
  and the comparison it came from, and states what it does not measure.
- `env -u PYTHONPATH ./.venv/bin/python -m pytest tests -q --no-header` green
  (the leaked `PYTHONPATH` in some shells shadows the project venv).

## Review checklist

- Does the change preserve the import direction (`SKSurrogate/` never imports
  `api`, `webui` or `sksurrogate_mcp`)? `tests/test_layer_boundaries.py` enforces
  it.
- Is randomness seeded from an explicit generator? The search contract is in
  `docs/search-contract.rst`.
- Does a new artifact carry a manifest and a checksum? `SKSurrogate/manifest.py`
  and `save_bundle` are the pattern.
- Does an explanation state its quantity, provenance and reconstruction status?
  `api/explanations.py` documents the contract.

Internal design notes and plans are not part of this repository's public
history; they live in `dev-notes/` or outside the tree.
