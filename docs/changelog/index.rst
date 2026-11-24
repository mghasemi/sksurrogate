Changelog
=========

Release notes, newest first. Each release is its own page, so no single
file grows without bound and a release stays readable on its own. Entries
describe behaviour a user can observe; internal churn lives in
``dev-notes/`` instead.

A row marked **not published** describes work that was drafted as its own
version but never shipped under that number. Those pages are kept because
docstrings, the test suite and commit messages cite them; the content appears
in the release named in the highlights.

.. list-table::
   :header-rows: 1
   :widths: 12 14 74

   * - Release
     - Date
     - Highlights

   * - ``0.8.1``
     - 2026-11-24
     - Documentation and packaging-claim release: ``[api]``/``[ui]`` described as dependencies for a source checkout, a guard test against the claim returning, and the C-space study added to the benchmark toctree. No behaviour change

   * - ``0.8.0``
     - 2026-11-24
     - Cumulative: auditable search engine and provenance, paired model comparison, evidence-grounded explanations and drafts, platform hardening. Folds in the extras restructuring (0.4.0), the MCP server (0.5.0) and the reliability contract (0.6.0)

   * - ``0.6.0``
     - not published
     - Search-engine reliability: exact evaluation budget, reproducible ``random_state``, auditable counters and the published search contract — shipped in 0.8.0

   * - ``0.5.0``
     - not published
     - MCP server for agents: curated tools over a running control plane, plus an ``api_request`` escape hatch — shipped in 0.8.0

   * - ``0.4.0``
     - not published
     - Optional extras: base install down to seven packages (39 s/633 MB → 7 s/250 MB) — shipped in 0.8.0

   * - ``0.3.1``
     - 2026-11-23
     - Partition defaults, temporal experiments and the Advanced scorer — 169 entries, five pages. Container image only; never uploaded to PyPI

   * - ``0.3.0``
     - 2026-11-08
     - Web UI log browser and severity-aware logging

   * - ``0.2.0``
     - 2026-10-24
     - Response-surface default change

   * - ``0.1.1``
     - 2026-10-11
     - Republished 0.1.0 with canonical MIT metadata; code identical

.. toctree::
   :maxdepth: 1
   :caption: Releases

   0.8.1
   0.8.0
   0.6.0
   0.5.0
   0.4.0
   0.3.0
   0.2.0
   0.1.1

.. toctree::
   :maxdepth: 1
   :caption: 0.3.1 (in five parts)

   0.3.1-scoring-search-and-partitions
   0.3.1-time-series-and-forecasting
   0.3.1-benchmarks-ensembles-and-calibration
   0.3.1-web-ui-data-page-and-jobs
   0.3.1-experiments-auth-and-fixes

0.3.1 in full is 115 KB; the five parts above are contiguous slices of
it, in document order.
