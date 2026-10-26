Changelog
=========

Unreleased
----------

JOSS submission and documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Opened the JOSS submission (``paper/paper.md``, ``paper/paper.bib``) as PR
  #8633 on ``openjournals/joss-papers``; added ``CITATION.cff`` pointing at the
  software record DOI 10.5281/zenodo.23631823 and the technical note
  ``docs/technical-note-rbf-gp-default.md``.
* D-track Hilbert-space surrogate study: new benchmark modules
  (``benchmarks/dtrack/{d1_approx,d2_compare,analyze}.py``), generated report
  ``docs/benchmarks/dtrack.md``, and standalone write-up
  ``docs/dtrack-study.md``; raw results (D1: 3816 cells, D2: 2160 cells) are
  archived at DOI 10.5281/zenodo.23652613.
* Phase C README rewrite with PyPI metadata polish; ORCID and affiliation
  filled in across docs; benchmark archive pointer updated to the v0.2.0
  software record (DOI 10.5281/zenodo.23631823).
* WebUI dependency bump: vite 8.3.4.

API
~~~

* ``RegisterData`` accepts an optional ``preprocess_kwargs`` mapping, forwarded
  to ``DataPreprocess`` (e.g. ``bin_threshold``, ``cat_threshold``,
  ``force_impute``) when encoding is enabled; defaults are unchanged.

[0.2.0] - 2026-10-24
--------------------

Response-surface default change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``SurrogateSearch`` now defaults its response surface to a stock, un-tuned RBF
  Gaussian process (`sklearn.gaussian_process.GaussianProcessRegressor`) instead of the
  legacy cubic Hilbert polynomial (`HilbertRegressor(deg=3)`). Evidence: equal-budget
  benchmark suite -- the RBF-GP surface attains top mean CV on all six datasets at both
  B=30 and B=60, with seed-level win rates of 20W/17T/5L (B=30) and 25W/13T/4L (B=60)
  against Hilbert (two-sided exact binomial p approx. 4e-3 / 1e-4); the un-tuned RBF-KRR
  alternative never wins a seed at either budget. See ``benchmarks/DESIGN.md`` sec. 12;
  raw per-run records are archived at DOI 10.5281/zenodo.23629024 (software:
  DOI 10.5281/zenodo.23631823). The surrogate-firing floor (`MinEvals`) is unchanged, so
  the firing behavior matches that benchmark configuration exactly.
* Restore the v0.1 behaviour with `regressor=HilbertRegressor(deg=...)`; no new dependency
  (scikit-learn was already required).

Dependencies and packaging
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Lower-bound dependency pins in ``setup.py``/``requirements.txt`` exclude the
  vulnerable ranges flagged by Dependabot (numpy <1.22, scipy <1.10,
  scikit-learn <1.5, joblib <1.2, tqdm <4.66.3, fastapi <=0.109.0, lightgbm
  <4.6, uvicorn <0.11.7, python-multipart <0.0.31); bounds sit far below the
  versions CI currently installs and no upper pins were added.
* WebUI lockfile: source-map-js pinned to a non-vulnerable version
  (CVE-2026-93749).

[0.1.1] - 2026-10-11
--------------------

Republished from 0.1.0 with canonical MIT license metadata; code is identical
to 0.1.0.

Control-plane API and Web UI (branch ``ui``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added a FastAPI control plane under ``api/`` covering datasets, bundles,
  quality gates, registry/deployment, inference, monitoring, sensitivity,
  experiments, background jobs with WebSocket progress streaming, retraining,
  and a lineage traceability endpoint.
* Added a React/Vite web UI under ``webui/`` with pages for every pipeline
  stage, live job tracking, a responsive shell, and route-level code splitting.
* Added opt-in shared-key API authentication via the ``SKSURROGATE_API_KEY``
  environment variable (accepted through the ``X-API-Key`` header, Bearer
  tokens, or an ``api_key`` query parameter on WebSocket routes); unset keeps
  all endpoints open for local use.
* Made CORS origins configurable via ``SKSURROGATE_API_CORS_ORIGINS``, defaulting
  to local-only origins instead of a wildcard; the web UI stores and sends an
  API key from its Settings page.
* Added learning-curve evaluation: ``GET /api/evaluation/{task}/metrics`` lists
  the metrics relevant to the task's problem family, and
  ``GET /api/evaluation/{task}/learning-curves`` computes cross-validated
  training/held-out curves per bundle for one metric via scikit-learn. The
  Evaluation page's "Metric comparison" card now has one tab per metric on top
  of the side-by-side overview chart.
* Experiment searches gained surrogate-assisted mode
  (``surrogate_mode``/``surrogate_itrs``), conditional parameters
  (per-parameter ``depends_on``) and top-level ``forbidden`` rules, a per-run
  Pareto frontier (score vs. candidate duration) and ``top_pipelines`` in the
  job result, plus a ``POST /api/experiments/{task}/optimize-pipeline`` endpoint
  that tunes one explicit component sequence into a bundle. The Experiments page
  gained a search-strategy radio, live JSON validation hints, a "Top pipelines"
  table with per-row optimization, and a Pareto scatter chart.
* Added the ``stacking`` directive to experiment search spaces
  (``res``/``probs``/``decision``/``cv``/``n_jobs``), which configures the
  out-of-fold ``StackingEstimator`` wrappers ``AML`` applies to intermediate
  estimators; ``AML`` gained the matching ``stack_cv``/``stack_n_jobs``
  constructor parameters.

Bug fixes
~~~~~~~~~

* ``SurrogateRandomCV`` no longer refits a rejected combination. The optimizer's
  solution can land inside a forbidden region (or in one whose folds all
  failed), and refitting those settings either raised — e.g. ``lbfgs`` with
  ``penalty='l1'`` — or silently returned a banned model. The best *completed*
  trial is used instead, and the refit is skipped entirely when no trial
  completed.
* Fixed the experiments stage failing with ``ValueError: too many values to
  unpack (expected 2)``: the stored cross-validation spec (a JSON dictionary)
  was handed straight to ``AML``, and scikit-learn's ``check_cv`` treated that
  dictionary as an iterable of ``(train, test)`` folds and unpacked its keys.
  The API now rebuilds the splitter with ``SKSurrogate.mltrace.build_cv`` while
  keeping the JSON spec for the bundle audit event, and ``AML`` normalizes any
  dictionary ``cv`` the same way so no caller can repeat the mistake.
* ``EOA`` now rejects ``num_parents`` larger than the population size with a
  message naming both values, instead of failing later inside
  ``random.sample`` with ``Sample larger than population or is negative``.
* ``SKSurrogate.aml.default_config`` is defined at module level again, so
  ``AML()`` works without an explicit ``config``.
* Inference on a stored partition no longer fails with ``400: extra columns:
  target``. A registered partition is training data, so it carries the target
  label while the bundle schema holds only features; ``predict_batch`` validates
  strictly and rejected the label outright. The control-plane API now projects a
  stored partition onto the bundle schema (which also normalizes column order)
  and reports the dropped columns as ``ignored_columns``. Inline ``rows`` are
  still validated exactly as sent, since they represent the serving payload.

Phase 10: Documentation and release hardening
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added an offline-to-deployment lifecycle example covering bundles, registry
  promotion, batch inference, and drift reporting.
* Added a full synthetic classification workflow covering nested evaluation,
  quality gates, approval-gated promotion, fairness, delayed labels, and
  runtime monitoring.
* Added a full synthetic regression workflow covering nested evaluation, R2
  quality gates, approval-gated promotion, numeric drift, delayed-label
  MAE/MSE, and runtime monitoring.
* Added SQLite database and EOA checkpoint migration and rollback guidance.
* Documented public API coverage, supported runtime policy, benchmark thresholds,
  release checks, and the compatibility matrix.
* Added opt-in benchmarks for bundle loading, batch inference, and drift reports.

Phase 9: Distributed execution and CI/CD integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added durable local and optional Dask execution backends.
* Added bundle quality gates, triggered retraining, approval-gated deployment,
  and rollback workflows.
* Added Phase 9 operational documentation.

Phase 8: Governance, fairness, and security
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added audit events, ownership metadata, sensitive-feature reports, fairness
  and subgroup reports, and retention-aware artifact deletion.

Phase 7: Data quality, drift, and performance monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added schema, quality, distribution, range, prediction, delayed-label, and
  runtime monitoring reports with configurable alerts.

Phase 6: Batch and online inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added schema-validated batch prediction, the batch CLI, and a dependency-free
  WSGI serving adapter with health and readiness endpoints.

Phase 5: Model bundles and registry lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added portable model bundles, atomic persistence, compatibility checks,
  registry aliases, promotion and rollback history, and MLflow-compatible export.

Earlier phases
~~~~~~~~~~~~~~

* Phases 1 through 4 added dataset identity and schema contracts, explicit
  evaluation protocols, resource governance, search-space validation, and
  structured failure handling.
