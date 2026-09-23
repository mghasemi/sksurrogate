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
