=================
Research workflow
=================

The control plane and the library implement one workflow, not a menu of
features. Each stage produces an artifact and names the surface that consumes
it. If a step has no artifact on disk (or in the registry), the next stage will
not see it.

.. list-table:: Dataset → Deploy
   :header-rows: 1
   :widths: 13 30 27 30

   * - Stage
     - What you decide
     - Artifact it produces
     - Surface that consumes it
   * - **Dataset**
     - target column, partition names, which columns are ignored
     - a registered partition (``datasets/<task>/<partition>.csv``) plus a
       dataset fingerprint and the task's ``mltrace`` record
     - every later stage; ``GET /api/datasets/{task}`` reports the fingerprint
       and schema
   * - **Diagnose**
     - whether the data supports the model you intend to fit
     - a bounded profile, missingness overview, duplicate/leakage checks and an
       EDA report
     - ``POST /api/datasets/{task}/profile``, ``POST /api/datasets/{task}/inspect``,
       ``GET /api/datasets/{task}/{partition}/eda-report``; the Datasets page
       renders the same blocks
   * - **Design**
     - next points for an expensive external experiment (not model search)
     - a Latin-hypercube plan and one suggested next point with the observed
       result attached
     - ``POST /api/design/latin-hypercube``, ``POST /api/design/suggest-next``;
       Experiment Design page
   * - **Optimize**
     - search space, scorer, budget, surrogate family
     - a fitted bundle with ``metrics``, ``cv_results_`` and a schema-v2
       manifest (plus the search's ``evaluation_history`` digest)
     - ``POST /api/experiments/{task}/run``, ``POST /api/bundles/{task}/train-baseline``,
       ``POST /api/ensembles/{task}/fit``; jobs carry the result record
   * - **Compare**
     - which candidate is actually better, and at what cost
     - a paired comparison record: per-fold scores, oriented deltas with
       bootstrap intervals, Holm-adjusted Wilcoxon p-values, effect sizes, a
       Pareto front and calibration
     - ``POST /api/comparison/{task}`` (see :doc:`model-comparison`);
       ``POST /api/evaluation/{task}/nested-cv`` for the unbiased outer score
   * - **Explain**
     - what the fitted model depends on, and where that reading is fragile
     - permutation importance (raw and grouped), PDP/ALE curves, row
       attributions with a checked reconstruction, prediction intervals, and a
       provenance block on every response
     - ``POST /api/evaluation/{task}/{version}/explain``, ``.../ale``,
       ``.../explain/row``, ``.../intervals``; Explain page
   * - **Validate**
     - whether the model is fit to ship
     - a nested-CV outer score, quality-gate report, leakage warnings and
       (for forecasts) a rolling-origin backtest
     - ``POST /api/quality-gates/{task}/{version}/check``,
       ``POST /api/evaluation/{task}/backtest``;
       ``bundle.metadata["reproducibility"]`` and the manifest carry the record
   * - **Deploy**
     - which exact version is served, and to whom
     - a serving pin (``serving/<task>.json`` with an append-only history) and
       batch/scored prediction artifacts
     - ``POST /api/serving/{task}``, ``POST /api/serving/{task}/predict``,
       ``POST /api/inference/{task}/predict-batch``; the
       ``sksurrogate-batch-predict`` CLI and ``ModelBundle.predict_batch``

Where the artifacts live
========================

* **Bundles** — ``<API home>/bundles/<task>/<version>.bundle`` plus the
  ``.bundle.manifest.json`` sidecar. ``load_bundle(..., verify=True)`` refuses a
  bundle whose bytes do not match the sidecar digest.
* **Manifests** — schema v2 (``SKSurrogate.manifest``): dataset fingerprint and
  partition, preprocessing, estimator/optimizer configuration, search space,
  seeds *and their ``SeedSequence`` stream provenance*, evaluation-history
  digest, dependency versions, hardware, git revision and dirty flag, artifact
  fingerprints and known gaps.
* **Jobs** — ``<API home>/jobs/<job_id>.json``; the request, status, result and
  error. Per-job log lines live in ``<API home>/logs/<job_id>.jsonl``.
* **Explanations** — returned synchronously, not persisted, unless you generate
  an explainability report (``POST /api/reports/{task}/explain``), which writes
  the figures and machine-readable ``results.json`` as a job artifact.
* **Comparisons** — returned synchronously; the numbers are reproducible from
  the two bundles plus the request (its settings block is in the payload).

Library-first equivalence
=========================

Every core stage is available without the API:

.. code-block:: python

   from SKSurrogate import AML, ModelBundle, load_bundle, save_bundle

   search = AML(preset="regression", random_state=0).fit(X_train, y_train)
   bundle = ModelBundle(
       search.best_estimator_,
       task_name=task,
       schema={column: "float64" for column in X_train.columns},
   )
   save_bundle(bundle, "artifacts/model.bundle")

   from SKSurrogate.compare import compare_scores

   compare_scores({"a": fold_scores_a, "b": fold_scores_b}, control="a")

The control plane stores and orchestrates those calls; it does not own the
mathematics. See :doc:`architecture` for the dependency direction that keeps
that true.
