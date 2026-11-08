=====================================
Step-by-Step Web UI User Manual
=====================================

For a complete worked example on a real dataset, with screenshots and the
matching REST and Python calls for every step, see :doc:`walkthrough`.

This manual walks through a typical SKSurrogate workflow in the web UI:
register data, inspect features, search and evaluate models, package and
approve a model, run inference, and monitor its behavior. Each screen is tied
to the corresponding SKSurrogate library feature so you can move between the
UI and Python/API workflows.

The screenshots use a disposable ``manual-demo`` task and the sample CSV files
linked below. They illustrate the interface; your task names, model versions,
metrics, and available actions will differ.

.. contents:: In this manual
   :local:
   :depth: 2

Before you start
================

The control-plane API and web UI are optional applications run from a source
checkout. They are separate from using ``SKSurrogate`` as a Python library.
Follow :doc:`controlplane` for dependency installation, service startup, API
keys, CORS, and storage configuration. In brief:

1. From the repository root, install the API dependencies and start the API:

   .. code-block:: console

      python -m pip install -r requirements.txt
      uvicorn api.main:app --reload --host 127.0.0.1 --port 8013

2. In another terminal, start the web UI:

   .. code-block:: console

      cd webui
      npm ci
      npm run dev

3. Open the local URL printed by Vite. The UI's development server proxies
   ``/api`` requests and job WebSockets to port ``8013`` by default.

Alternatively, run everything with Docker (see :doc:`controlplane`):

.. code-block:: console

   docker compose --profile prod up --build -d   # UI and API on http://localhost:8013
   docker compose --profile dev up --build       # hot reload; UI on http://localhost:5174

The production image serves the built UI and the API from one port, runs as a
non-root user, reports health at ``/api/health``, and keeps data in the named
volume ``sksurrogate-data`` (``/data``). Set ``SKSURROGATE_API_KEY`` for any
non-local deployment. Optional Python packages can be added at build time with
``--build-arg EXTRAS="pkg1 pkg2"``.

For a local walkthrough, download the
:download:`training CSV <ui-manual-example.csv>` and the separate
:download:`validation CSV <ui-manual-validation.csv>`. They are small,
synthetic examples for interface practice, not meaningful model benchmarks.

.. warning::

   The API is open when ``SKSURROGATE_API_KEY`` is unset. Do not expose a
   keyless service to an untrusted network. If you enable a key, enter it on
   **Settings / Audit**. The UI stores the API base URL and key in this
   browser's local storage; use a trusted browser profile.

The workflow at a glance
========================

The UI is organized around a task, which groups data, runs, bundles, registry
history, inference records, and monitoring. The usual flow is:

``Task → Datasets → Feature Analysis → Experiments → Evaluation → Explain →
Model Bundles → Quality Gates → Registry & Deployment → Inference → Monitoring
→ Retraining``

The sidebar groups pages by stage (Data, Model, Release, Operate). **Projects**
groups the datasets, bundles and jobs of one investigation and shows which
stages are done; **Experiment Design** plans physical or simulated experiments
before any data exists. A "Getting started" card on the Dashboard walks a first
run.

You can also train a baseline directly under **Model Bundles**, skip optional
feature analysis, or use the API and Python package independently of the UI.
Long-running sensitivity, experiment, nested-CV, and retraining operations
appear in **Jobs**.

1. Open the Dashboard and select a task
---------------------------------------

The **Dashboard** summarizes API health, background jobs, monitoring alerts,
and links to the main pipeline stages. Use the **Task** picker in the top bar
to select an existing task or choose **+ Add new task…** and enter a short,
stable name such as ``customer-churn``. The new name is held in the browser
until you create task data or run a task operation.

.. figure:: images/ui-dashboard.png
   :alt: SKSurrogate Dashboard with task selector and pipeline stage links
   :width: 100%

   Dashboard overview. The task picker is shared across task-specific pages.

The task name is the common identifier used by ``mltrack``, model bundles,
registry aliases, and monitoring records. Choose the right task before
uploading data or launching work; use a different task when the data or model
lifecycle should be kept separate.

2. Register and validate datasets
---------------------------------

Open **Datasets**. For each CSV partition:

1. Set **Partition** to ``train``, ``validation``, or ``test`` (or another
   descriptive partition name).
2. Drop the CSV into the upload area or click it to browse.
3. Click **Inspect CSV**. This previews the row count, columns, inferred data
   types, possible target columns, and a bounded PII-like value-pattern and
   sensitive-column scan. The scan reports pattern counts only; it does not
   expose matched cell contents. Inspection does not commit the file.
4. Choose the **Target column**. Review the detected type of every column and
   change any incorrect type before committing. If appropriate, enable
   **Binarize / ordinal-encode target labels** to encode target categories.
5. Click **Commit dataset**. The API registers the partition and schema,
   calculates a dataset fingerprint, and stores the CSV under the task.

Use the same target and compatible feature columns across partitions. For the
tutorial, register the training file as ``train`` and the second file as
``validation``. In a real workflow, keep validation data separate from
training data; do not use a copy of the training rows as a holdout.

If the task already has a registered schema, **Validate upload** checks a new
CSV against it before you commit. The Python tracking layer is
``SKSurrogate.mltrace.mltrack``: ``register_data`` records dataset metadata,
``validate_data`` checks training data, and ``validate_prediction_data``
checks inference inputs. The checks catch column, order, dtype, and category
differences; use the explicit compatibility options in the Python API only
when a relaxed contract is intended.

.. figure:: images/ui-datasets.png
   :alt: Dataset page with CSV upload and saved cross-validation settings
   :width: 100%

   Dataset registration and the task-level cross-validation settings.

After registration, review **Registered metadata** for the target, feature
count, fingerprint, and partitions. Use **Target statistics** to inspect the
target distribution. The optional **Synthetic data** card can generate a
downloadable sample from a registered partition; choose the row count, source
partition, marginal or joint distribution, and a generation type for each
column. Synthetic output is a generated file, not a replacement for the
registered source dataset.

Choose **Cross-validation partitioning** deliberately. The selected splitter
and its parameters are stored for the task and reused by downstream
cross-validation. For classification, use a stratified splitter when class
balance matters; for grouped or time-ordered observations, select a splitter
that preserves that structure rather than randomly mixing related or future
observations across folds.

3. Inspect and reduce features (optional)
-----------------------------------------

Open **Feature Analysis** after registering a training partition.

* **Input sensitivity** measures how the fitted surrogate responds over
  generated independent feature ranges using Sobol, Morris, or delta-moment
  methods. Choose the method, number of top features, and training partition,
  then click **Run sensitivity analysis**. This runs as a background job.
  Correlated and categorical inputs may produce unrealistic perturbations;
  these indices are not observed feature associations or causal effects.
* **Correlation pruning** applies a threshold to the absolute pairwise
  correlation between features. It reports which features would be kept or
  dropped; it does not silently rewrite the registered CSV.
* **Heatmaps** show sensitivity ranking, Pearson correlation, or feature
  weights. Weight values can come from stored ``mltrack.feature_weights`` data
  or be computed from target correlation and feature variance when appropriate.
* **Consensus top features** summarizes how often each feature appears in the
  top-ranked sets stored for the task.

These views complement **Explain**, which reports permutation importance and
partial-dependence profiles for a selected fitted bundle. Pearson feature
correlation describes observed pairwise association; target correlation is a
marginal data association; input-sensitivity indices describe a fitted
surrogate over generated ranges; permutation importance measures score change
under feature shuffling. The methods have different targets, perturbations,
reference data, and scales, so they need not agree and their raw values should
not be treated as one combined ranking. Use the links between Feature Analysis
and Explain to review their interpretations together.

These correspond to ``SKSurrogate.sensapprx.SensAprx``,
``CorrelationThreshold``, and the feature-analysis methods on
``SKSurrogate.mltrace.mltrack`` (including ``heatmap`` and ``top_features``).
Use these analyses to guide a smaller experiment search space or domain
review. Correlations describe observed feature association; sensitivity
indices describe surrogate responses to generated inputs. Neither is causal
evidence.

.. figure:: images/ui-features.png
   :alt: Feature Analysis page showing sensitivity and correlation controls
   :width: 100%

   Feature ranking and correlation tools for the selected task.

4. Search model pipelines
-------------------------

Open **Experiments**. The **Quick start** tab needs no search space: choose
*Classification* or *Regression*, a budget (Fast, Balanced, Thorough), and a
scoring metric (a sensible default is preselected), then click **Find best
model**. It runs a bounded random-forest / extra-trees search through
``AML(preset=...)`` with the surrogate strategy. Use the **Advanced** tab for
full control, described next.

In **Advanced**, the page starts with a JSON search-space example. Each
top-level key identifies a fully-qualified scikit-learn estimator class;
parameter definitions specify real, integer, or categorical ranges. Validate
the JSON and search-space definitions before starting. The **Add stacking
component** and **Add forbidden rule** buttons help construct supported
directives without hand-writing them.

Then configure the run:

1. Set the trial length, maximum generation, number of parents, and training
   partition. Keep an experiment small for an initial smoke test.
2. Choose a scoring metric that matches the task and the business objective.
   The selected scorer is the objective the search optimizes.
3. Choose **Evolutionary (EOA)** or **Surrogate-assisted** search. The latter
   adds surrogate iterations and may take longer per generation.
4. **Search engine** selects ``AML.fit(strategy="eoa")`` (default) or
   ``strategy="surrogate"`` (surrogate search only; generation and parent
   settings are ignored).
5. Keep the **Local** execution backend unless a Dask-enabled API environment
   is available.
6. Click **Start experiment**, then follow progress on the page or in **Jobs**.

The API builds an ``SKSurrogate.aml.AML`` search and uses
``SKSurrogate.eoa.EOA`` for evolutionary optimization; the surrogate path can
also use ``SurrogateRandomCV``. The optional ``stacking`` directive configures
``StackingEstimator`` wrappers that generate out-of-fold intermediate
features. Conditional parameters and forbidden combinations constrain
incompatible choices.

When the job completes, inspect the score history, top pipelines, and
score-versus-duration Pareto frontier. Use **Optimize this structure** to tune
one of the returned component sequences. A completed search creates a bundle
that appears in **Model Bundles**. Use **Jobs** to inspect job status and
details; failed experiment jobs may offer a resume action when a checkpoint
allows it.

.. figure:: images/ui-experiments.png
   :alt: Experiments page with JSON search space and run controls
   :width: 100%

   Search-space editor and experiment settings. The run controls continue
   below the visible portion of the screen.

5. Evaluate candidate models
----------------------------

Open **Evaluation** to compare the metrics recorded on each bundle. The
overview compares the bundles side by side; metric tabs compute learning
curves across training sizes using the task's saved splitter. Classification
bundles also provide ROC, calibration, cumulative-gain, and lift diagnostics.

For a more rigorous estimate, choose a bundle in **Nested CV**, set inner and
outer fold counts and repeats, and click **Run nested CV**. The inner loop
selects/tunes the estimator; the outer loop estimates generalization. This is
a background job and can be expensive. Keep an untouched validation or test
partition for a final check after tuning.

Evaluation uses scikit-learn scorers and cross-validation through the
control-plane evaluation routes and the tracking/evaluation helpers. Learning
curves and diagnostic curves are distinct from nested CV: inspect the
displayed partition, splitter, and curve description before comparing results.
If a metric is unavailable for a bundle, the page reports the per-bundle
failure rather than treating it as a score.

.. figure:: images/ui-evaluation.png
   :alt: Evaluation page for comparing bundles and running nested cross-validation
   :width: 100%

   Evaluation controls and metric comparison. Charts appear after at least one
   bundle has been created.

Explain a model
~~~~~~~~~~~~~~~

Open **Explain**, choose a bundle, and click **Explain model** for global model
attribution by permutation importance (the score drop when a feature is
shuffled) and fitted-model response profiles from partial-dependence curves.
The former measures model-score sensitivity, not feature-target association;
the latter can evaluate implausible combinations when features are dependent.
Neither estimates causal effects. The explanation response reports its
reference partition, source and used row counts, whether the rows were
subsampled, and the row cap, permutation-repeat count, and seed. The seed
controls both row subsampling (when needed) and permutation-repeat generation;
this metadata aids repeatability. Optionally select 3 or 5 reference samples
and a row cap to inspect descriptive variation in permutation importance
across subsamples; the UI reports per-feature sample standard deviation and
top-rank frequency. This is not a confidence interval or a guarantee of
robustness, and it requires more source rows than the selected row cap. With
the default of one sample, no cross-sample stability summary is estimated.
The Explain page shows the reference row counts, sampling status, seed when
sampling occurs, and permutation-repeat count. With 3 or 5 reference samples
selected, the partial-dependence charts also gain dashed mean ± sample-standard-
deviation lines computed over the same grid for each distinct subset; like the
importance summary, this describes reference-sample sensitivity and is not a
confidence band.

The **Accumulated local effects (ALE)** card supplements partial dependence
when features are dependent: choose 10, 20 or 30 quantile bins and click
**Compute ALE**. ALE accumulates model-response differences between adjacent
observed bins, so each curve reports its bin boundaries, the reference rows per
bin, and any excluded non-finite rows instead of averaging over feature
combinations that never occur. The response label names the scalar being
explained — ``predict`` for regression, the positive-class probability for
binary classifiers — and nominal categorical features are listed as skipped
because they have no canonical ordering. ALE still describes the fitted model,
not a causal effect, and correlated features can make one feature's local
differences reflect another's distribution.

For regression bundles, **Compute intervals**
produces split-conformal prediction intervals at 80/90/95% target coverage and
reports observed coverage with a Wilson interval, mean and median width, and
mean interval score on held-out rows. It identifies the fit, calibration, and
holdout rows/partitions. The conformal guarantee is marginal and assumes
exchangeable calibration and future data; the Wilson interval separately
assumes independent held-out coverage indicators. Neither should be read as
evidence of conditional coverage or protection against distribution shift.
Optionally enter a feature column to inspect bounded subgroup coverage; these
small-group estimates are diagnostic and do not imply conditional coverage.

6. Create and inspect model bundles
------------------------------------

You can create a bundle either by completing an experiment or by fitting one
baseline in **Model Bundles**. For a baseline, select an estimator, training
partition, and optional validation partition, then click **Train baseline**.
Available baselines include linear/logistic regression and random-forest
regressors/classifiers. A validation partition adds a ``score`` metric; leave
it empty only when you intentionally do not have a holdout.

Select a model version and click **Inspect** to review its metrics, feature
schema, dataset fingerprint, dependencies, audit events, and lineage. A
``ModelBundle`` packages the fitted estimator and its metadata; the bundle is
saved as a versioned ``.bundle`` file. You can also:

* click **Jump to best** to select the bundle with the best rankable metric;
* use **Preserve snapshot** for a legacy tracking snapshot, or recover an
  existing snapshot into a new bundle version;
* download the bundle or export it to an MLflow-compatible model directory.

Bundle persistence and loading correspond to ``ModelBundle``, ``save_bundle``,
``load_bundle``, and ``export_mlflow`` in ``SKSurrogate.modelbundle``.
Serialized models should only be loaded from trusted sources.

.. figure:: images/ui-bundles.png
   :alt: Model Bundles page showing a trained baseline and its version
   :width: 100%

   A bundle's generated model version and validation score.

7. Run quality gates before release
------------------------------------

In **Quality Gates**, select the bundle version. Optionally set a maximum
bundle file size and define numeric metric minimums as a JSON object, for
example:

.. code-block:: json

   {"score": 0.75}

Click **Run quality gate** and review each check and any failures. The
underlying ``SKSurrogate.ci.check_bundle_quality`` helper checks that the
bundle can be loaded and evaluates the configured metric thresholds and
optional file-size limit. For an application-specific schema contract, also
validate the relevant dataset or inference input; a metric threshold alone
does not establish that a model is suitable for deployment.

.. figure:: images/ui-quality-gates.png
   :alt: Quality Gates page with bundle metric threshold controls
   :width: 100%

   Select the metrics recorded in a bundle to add a minimum threshold.

.. important::

   The **Promote** form independently reruns the default bundle check, which
   verifies that the registered bundle can be loaded. It does not carry
   forward the custom metric thresholds or bundle-size limit from this page.
   The promotion endpoint recomputes its default report; a report submitted
   with the request is not used to enforce those custom thresholds. Enforce
   any required metric or size policy separately in your release process
   before promotion.

8. Register, approve, and promote a model
-----------------------------------------

Open **Registry & Deployment**:

1. In **Register a bundle**, select the version from Model Bundles and click
   **Register**. The registry records it as a ``candidate`` and makes it
   available through the ``latest`` alias.
2. In **Promote**, select that version, choose a lifecycle state such as
   ``validated`` or ``production``, enter approver names, and set the number
   of required unique names.
3. Promote through your organization's review stages. The page displays the
   aliases, lifecycle history, approval events, audit log, and lineage.
4. If a production alias needs to point to another version, use **Rollback**
   to select the target model version and lifecycle alias.
5. Use **Use in Inference** to open the selected registry alias in the
   Inference page.

This workflow uses ``ModelRegistry`` and ``DeploymentApprovalGate`` from
``SKSurrogate.modelbundle`` and ``SKSurrogate.deployment``. Lifecycle states
include candidate, validated, staging, production, and archived.

.. figure:: images/ui-registry.png
   :alt: Registry lifecycle with validated and production aliases
   :width: 100%

   Model version lifecycle, current aliases, lineage, and registry actions.

.. warning::

   Approver names in this UI are free-text entries, not verified user
   identities. The current v1 control plane does not provide user accounts or
   role-based authorization. Enabling the optional shared API key protects
   access to the API but does not turn those names into authenticated
   approvals.

9. Run online or batch inference
--------------------------------

On **Inference**, explicitly resolve a model by version or registry alias.
For a quick request, choose a stored partition or enter JSON rows and click
**Run prediction**. Turn on **Validate before run** to check the input
against the task's registered data schema before prediction. Review the
returned predictions, model version, request ID, latency, row count, ignored
columns, and lineage.

For stored partitions, the API projects the data onto the bundle's feature
schema, which removes the target label and normalizes feature order. Inline
rows are treated as the serving payload and must match the bundle schema.
Schema failures are shown as errors rather than silently coercing unknown
features.

Use **Batch run** to score the selected input in bulk. The output is saved as
a CSV artifact under the task's predictions directory and can be linked to
monitoring and registry lineage. These actions wrap
``SKSurrogate.inference.predict_batch`` and its schema validation.

.. figure:: images/ui-inference.png
   :alt: Inference page with model and stored-partition selectors
   :width: 100%

   Choose an exact model version or registry alias before scoring.

10. Monitor serving quality and data drift
------------------------------------------

Open **Monitoring** after scoring data:

* **Serving summary** reports request count, rows, mean latency, throughput,
  and error rate for the selected model version.
* **Drift check** compares a reference partition with a current partition.
  Review schema changes, missingness, numeric population stability index
  (PSI), categorical distribution distance, range drift, and threshold
  alerts.
* **Prediction distribution drift** compares two prediction series from
  partitions, persisted batch artifacts, or inline values.
* **Subgroup performance** compares a selected metric across groups.
* **PII & sensitive columns** scans names and up to 1,000 rows per column for
  common email, formatted US SSN, and formatted US phone patterns. It reports
  counts only and never includes matched cell contents; pattern matches may
  be false positives and are not identity validation.
* **Fairness** compares group-level selection rates and true-positive rates.
  Supply correctly aligned ``y_true``, ``y_pred``, and group arrays.
* **Delayed-label performance** compares saved predictions with labels that
  arrive later; choose the task type and model version.

Inference requests are logged automatically. The reports correspond to
``drift_report``, ``prediction_distribution_report``,
``subgroup_performance_report``, ``sensitive_feature_report``,
``fairness_report``, ``delayed_label_performance``, and ``InferenceMonitor``
in ``SKSurrogate.monitoring``. These are decision-support signals: review
alerts and subgroup definitions in context rather than treating thresholds
as universal fairness or drift guarantees.

.. figure:: images/ui-monitoring.png
   :alt: Monitoring page with serving summary and drift comparison controls
   :width: 100%

   Monitoring offers separate views for serving health, feature drift,
   prediction drift, fairness, and delayed outcomes.

11. Configure a retraining run
------------------------------

Open **Retraining** when you need a new model version:

1. Choose **Baseline** for one estimator or **Experiment (AML)** to search a
   JSON-defined parameter space.
2. Select the training partition, scorer when searching, and an available
   execution backend.
3. Set a trigger: ``always`` runs now; ``scheduled`` and ``on_data`` evaluate
   the corresponding **due** or **has new data** boolean.
4. Optionally choose a registry promotion state. The completed retraining job
   creates a new model version and can register it in that state.
5. Click **Start retraining** and follow the job in the page or in **Jobs**.

The Python workflow maps to ``SKSurrogate.retraining.RetrainingJob`` and the
execution backends. The trigger choices are evaluated by a submitted run;
they do not install a recurring scheduler or continuously watch for new data.
For automated delivery, an external scheduler or ingestion watcher can call
``POST /api/retraining/{task_name}/trigger`` with a fresh event ID and
timestamp. Data-trigger events must reference the registered training
partition and its current fingerprint; repeated delivery of the same
partition/fingerprint is deduplicated. Automated promotion requires metric
thresholds and is limited to **staging**; use the approval workflow for
production. See :doc:`phase9` for the event contract and operational limits.

.. figure:: images/ui-retraining.png
   :alt: Retraining page with trainer and trigger controls
   :width: 100%

   Choose a one-off retraining run and its optional trigger/promotion settings.

12. Follow jobs and audit activity
----------------------------------

Use **Jobs** to filter background work by kind and status, inspect progress,
and review results or errors. Sensitivity analysis, experiments, nested CV,
and retraining can run in the background; their job records remain visible
after navigation. A failed experiment can expose a resume action when its
checkpoint is usable.

Use **Settings / Audit** to configure the API base URL and shared API key for
this browser, and to review task registry audit events. Connection settings
are browser-local; registry audit history is task data stored by the API.

.. figure:: images/ui-jobs.png
   :alt: Jobs page with kind and status filters
   :width: 100%

   Filter and inspect asynchronous work from across tasks.

.. figure:: images/ui-settings.png
   :alt: Settings page showing API connection and audit log
   :width: 100%

   Configure this browser's API connection and inspect registry audit events.

Feature-to-library reference
============================

The table summarizes where the UI workflow meets the package API. For
additional Python examples, see :doc:`code`, :doc:`mltrace`,
:doc:`modelbundles`, :doc:`inference`, and :doc:`monitoring`.

+----------------------+------------------------------------------------------+
| UI area              | Main SKSurrogate feature                             |
+======================+======================================================+
| Datasets             | ``mltrack.register_data``, schema validation, CV     |
+----------------------+------------------------------------------------------+
| Feature Analysis     | ``SensAprx``, ``CorrelationThreshold``, ``mltrack``  |
+----------------------+------------------------------------------------------+
| Experiments          | ``AML``, ``EOA``, ``SurrogateRandomCV``, stacking    |
+----------------------+------------------------------------------------------+
| Evaluation           | Nested CV, learning curves, sklearn scorers          |
+----------------------+------------------------------------------------------+
| Model Bundles        | ``ModelBundle``, ``save_bundle``, ``load_bundle``    |
+----------------------+------------------------------------------------------+
| Quality Gates        | ``check_bundle_quality``, ``assert_bundle_quality``  |
+----------------------+------------------------------------------------------+
| Registry & Deployment| ``ModelRegistry``, ``DeploymentApprovalGate``        |
+----------------------+------------------------------------------------------+
| Inference            | ``predict_batch``, ``SchemaValidationError``         |
+----------------------+------------------------------------------------------+
| Monitoring           | Drift, fairness, subgroup, delayed-label reports     |
+----------------------+------------------------------------------------------+
| Retraining           | ``RetrainingJob``, execution backends                |
+----------------------+------------------------------------------------------+

API users can browse the live request and response schemas at
``http://127.0.0.1:8013/docs`` while the service is running. To use SKSurrogate
without the web application, import the same library features directly from
Python as documented in :doc:`code`.
