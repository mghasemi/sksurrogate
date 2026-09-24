Changelog
=========

Unreleased
----------

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
