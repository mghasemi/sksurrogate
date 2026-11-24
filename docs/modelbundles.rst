.. _modelbundles:

====================================
Model Bundles and Registry
====================================

``SKSurrogate`` provides portable model bundles for moving a fitted estimator from
training to validation or deployment. A bundle stores the fitted model together with
its preprocessing reference, feature schema, metrics, dataset fingerprint, configuration
version, runtime dependency versions, task name, and model version. Since format **v2**,
bundles trained on time-series data additionally carry an optional ``temporal`` block —
the serving state of their forecasting design — plus optionally a calibrated interval
block (see :ref:`format-versions`).

Several bundles can also be combined into one model with scikit-learn's own
meta-estimators — see :doc:`ensembles` for voting, averaging, stacking, bagging and
boosting over a selection of trained models.

Creating and loading a bundle
=============================

Create a bundle from any fitted scikit-learn-compatible estimator and persist it with
an atomic write:

.. code-block:: python

    from pathlib import Path
    from sklearn.linear_model import LogisticRegression
    from SKSurrogate import ModelBundle, load_bundle, save_bundle

    model = LogisticRegression(max_iter=200).fit(X_train, y_train)
    bundle = ModelBundle(
        model,
        task_name="customer-churn",
        schema=feature_schema,
        preprocessing={"description": "standardized numeric features"},
        metrics={"validation_accuracy": 0.91},
        dataset_fingerprint=dataset_fingerprint,
        config_version="config-2026-09-22",
    )
    save_bundle(bundle, Path("artifacts/customer-churn.bundle"))

    loaded = load_bundle(
        "artifacts/customer-churn.bundle",
        expected_schema=feature_schema,
    )
    predictions = loaded.predict(X_new)

``ModelBundle.predict`` delegates directly to the fitted estimator and skips
schema validation; use ``predict_batch`` when the input must be checked against
the recorded schema (see :doc:`inference`).

``load_bundle`` checks the bundle format, and can reject an incompatible feature schema
or runtime dependency set. Pass ``strict_dependencies=False`` when the environment is
managed separately and dependency comparison is not required.

The artifact is serialized with ``joblib`` and should only be loaded from trusted
sources. The model version is generated automatically unless ``model_version`` is
provided explicitly.

.. _format-versions:

Bundle format versions and temporal bundles
===========================================

The on-disk record carries a ``format_version`` field. Version **1** is the original
tabular layout; version **2** — what :func:`~SKSurrogate.save_bundle` writes today (
``BUNDLE_FORMAT_VERSION = 2``) — adds two optional fields to it, and both formats load
through the same :func:`~SKSurrogate.load_bundle` (``SUPPORTED_BUNDLE_FORMAT_VERSIONS = (1, 2)``).
A v1 record simply arrives without temporal content, so existing tabular bundles keep
working unchanged.

A **temporal** bundle is what a forecasting workflow produces — it adds:

* ``temporal``: the serialized serving state of its design (:class:`~SKSurrogate.TemporalServeState`)
  — series spec, target column, feature configuration, and the direct/recursive strategy; and
* ``interval_calibration`` (optional): the split-conformal per-horizon offsets fitted by a
  calibrated backtest (method, alpha, target step, residual count) plus any recorded
  calibration notes.

Serving such a bundle means forecasting from history rather than scoring feature rows:

.. code-block:: python

    result = loaded.predict_forecast(history_frame)   # one series in time order
    print(result.times)        # origin+1 .. origin+h (design horizon for direct bundles)
    print(result.means, result.lower, result.upper)   # bounds None unless calibrated

``predict_forecast`` validates the history exactly as described in :doc:`inference`
(one series, time order, enough tail rows for every feature the design reads) and then
delegates to :func:`~SKSurrogate.serve_forecast`. For a temporal bundle,
:func:`~SKSurrogate.predict_batch` routes into this same path — its input is the
history frame and the result has one row per forecast step. The two shapes of bundle
answer different questions, and each refuses the other's job with an actionable error
rather than faking it: a tabular bundle asked to *forecast* raises ``ValueError`` at
every serving path (400 over HTTP).

Registry lifecycle
==================

``ModelRegistry`` stores multiple versions for a task in a filesystem directory. New
versions start in the ``candidate`` state and are available through the ``latest``
alias:

.. code-block:: python

    from SKSurrogate import ModelRegistry

    registry = ModelRegistry("artifacts/registry")
    version = registry.register(bundle)
    registry.promote("customer-churn", version, "validated")
    registry.promote("customer-churn", version, "production")

    production = registry.load("customer-churn", "production")
    latest = registry.load("customer-churn", "latest")
    history = registry.history("customer-churn")

The supported lifecycle states are ``candidate``, ``validated``, ``staging``,
``production``, and ``archived``. Promotion updates the corresponding state alias and
records a timestamped history entry. Roll back an alias to a known version with:

.. code-block:: python

    registry.rollback("customer-churn", "production", model_version="previous-version")

Named model aliases
-------------------

Lifecycle aliases are rewritten by promotion and rollback, so they cannot identify a
*specific* experiment for long. A **named alias** is a user-chosen, stable handle for an
immutable model version — useful when referring to a run in a paper, a notebook or a
conversation:

.. code-block:: python

    registry.set_named_alias("customer-churn", "baseline", version, actor="mehdi")
    registry.resolve_version("customer-churn", "baseline")   # -> version
    registry.load("customer-churn", alias="baseline")        # same bundle
    registry.load_version("customer-churn", "baseline")      # same bundle
    registry.named_aliases("customer-churn")                 # {name: {model_version, ...}}
    registry.delete_named_alias("customer-churn", "baseline")

Alias names are lowercase slugs (letters, digits, ``.``, ``_``, ``-``; max 64
characters) and may not collide with the lifecycle names
(``candidate``/``validated``/``staging``/``production``/``archived``/``latest``).
Every set/delete is appended to the registry history and audit log with the acting
identity, and re-pointing an existing alias records the previous version. Exact
version ids always take precedence over aliases during resolution, and deleting an
alias never deletes the version it pointed at.

Artifact retention and deletion
================================

Registry storage also supports versioned dataset and prediction artifacts so a task can be
retained or cleaned up by task, model version, and dataset identity. Use
``register_dataset`` and ``register_prediction`` to keep raw data and prediction outputs
linked to a specific dataset fingerprint:

.. code-block:: python

    from pathlib import Path
    from SKSurrogate import ModelRegistry

    registry = ModelRegistry("artifacts/registry")
    dataset_path = Path("artifacts/train.csv")
    prediction_path = Path("artifacts/preds.csv")

    registry.register_dataset(
        "customer-churn",
        "dataset-v42",
        dataset_path,
        metadata={"rows": 25000, "target": "label"},
    )
    registry.register_prediction(
        "customer-churn",
        "model-v7",
        "dataset-v42",
        prediction_path,
        prediction_id="nightly-run-01",
        metadata={"source": "batch-scoring"},
    )

Later, delete artifacts by identity when retention should be enforced or a dataset must be
purged after a retention period:

.. code-block:: python

    deleted = registry.delete_artifacts(
        task_name="customer-churn",
        model_version="model-v7",
        dataset_fingerprint="dataset-v42",
    )

The return value contains the number of deleted dataset and prediction artifacts. The
registry also records an ``delete_artifacts`` audit event so each cleanup stays
traceable to the task, model version, and dataset fingerprint being removed.

Registry and bundle writes use temporary files followed by atomic replacement, so a
partially written artifact is not presented as a completed bundle or registry index.

MLflow-compatible export
========================

MLflow is optional. ``export_mlflow`` creates a conventional MLflow model directory
without requiring MLflow during training:

.. code-block:: python

    from SKSurrogate import export_mlflow

    export_mlflow(bundle, "artifacts/customer-churn-mlflow")

The directory contains ``MLmodel``, the sklearn ``model.pkl``, and ``bundle.json`` with
the full bundle metadata. Environments that install MLflow can consume the standard
``MLmodel`` and sklearn flavor artifacts.

Run configuration and reports
=============================

Bundles created by the API's baseline, AutoML experiment, and pipeline-optimization
workflows snapshot the settings needed to describe their training run. The snapshot
includes the training-partition fingerprint, row count, features and target, search or
estimator configuration, transformations, scoring/CV protocol, random seed, and runtime
versions available at training time. It records unknown target construction and
upstream exclusions as gaps rather than assuming they did not occur. The bundle's
``dataset_fingerprint`` identifies the training partition used for that model.

Export the machine-readable run configuration or a concise Markdown report:

.. code-block:: text

    GET /api/bundles/{task_name}/{model_version}/run-config
    GET /api/bundles/{task_name}/{model_version}/report.md

The JSON response is versioned and includes model/data lineage, stored metrics,
dependencies, and any repeated-CV evaluations saved for that bundle. The Markdown
report summarizes methodology, recorded target handling and observation
exclusions, metric scope, evaluation protocol, metrics, uncertainty,
limitations, and lineage. Unknown upstream exclusions remain marked as
unrecorded. Its source-artifact section links back to registered dataset
metadata, the run configuration, and the downloadable bundle.
Run nested cross-validation through ``POST /api/evaluation/{task_name}/nested-cv``
to attach its repeat-level confidence interval and protocol to the bundle. The
interval is a normal approximation over repeat-level mean outer scores; a single
repeat yields a degenerate, non-informative interval. Baseline or search metrics
without a saved repeated-CV evaluation are point estimates with no reported
uncertainty.

Older or manually-created bundles may lack a provenance snapshot. Their export still
works, but explicitly reports the missing fields; it cannot reconstruct settings that
were not persisted when the model was trained.

Experiment manifests and artifact checksums
===========================================

Every bundle written by ``save_bundle`` is now accompanied by a schema-v2
**experiment manifest** sidecar, ``<name>.bundle.manifest.json``. The manifest is
the single object that answers *what exactly produced this artifact*:

* dataset partition, its fingerprint and row count;
* preprocessing/transformation steps;
* estimator and optimizer configuration, plus the search-space definition;
* random seeds **and their provenance** — the ``init_candidate`` and ``sampling``
  child streams derived from one ``SeedSequence``;
* a digest of the evaluation history;
* interpreter and pinned dependency versions, optional backends when installed;
* hardware (platform, CPU count, BLAS/LAPACK build) and the git revision with a
  dirty flag;
* scoring/validation protocol, artifact fingerprints and the known gaps;
* the leakage-warnings block described below.

The bundle record also carries the manifest additively under ``manifest``; the
legacy ``metadata["reproducibility"]`` record remains readable and is upgraded
in place when read through :class:`SKSurrogate.manifest.ExperimentManifest`.
Version-1 records (the flat ``schema_version: 1`` dict) still load: readers must
tolerate missing v2 fields rather than refuse an archived artifact.

The sidecar records the SHA-256 of the exact bundle bytes. Verify it before
loading an artifact received from elsewhere:

.. code-block:: python

    from SKSurrogate import load_bundle, verify_bundle

    verdict = verify_bundle("models/churn.bundle")
    assert verdict["verified"], verdict
    bundle = load_bundle("models/churn.bundle", verify=True)

``verify_bundle`` raises :class:`SKSurrogate.BundleIntegrityError` when one byte
differs from the recorded checksum. A bundle written before manifests existed has
no sidecar: it is reported as ``verified: False`` (not an integrity failure), and
``load_bundle(..., verify=True)`` or ``SKSURROGATE_STRICT_BUNDLES=1`` makes the
absence fail closed.

Leakage checks
==============

The experiment result payload (and the manifest stored with its bundle) includes
an advisory ``leakage_warnings`` block:

* **duplicate rows across train/validation** — identical rounded feature vectors
  appearing on both sides, counted with their fraction of the validation split;
* **single-column target-leakage candidates** — a feature whose AUC against the
  majority class exceeds 0.98 (either direction) or, for a regression target,
  whose absolute Pearson correlation exceeds 0.995;
* **temporal-split suspicion** — monotone datetime columns, which make a random
  split score the model outside its training period.

These are heuristics, not proof. The block says so next to the finding, and it
never removes a row or a column: the split and the columns remain the analyst's
decision. The Experiments page renders the same block as a **Leakage checks**
card, and the run report carries the exploratory-versus-unbiased distinction so a
tuned score is not presented as an out-of-sample estimate.

Compatibility metadata
======================

Bundles can carry the dataset identity and schema recorded by ``mltrack``:

* ``dataset_fingerprint`` identifies the exact training data content and schema.
* ``schema`` describes expected columns, dtypes, nullability, and categorical values.
* ``config_version`` identifies the training configuration used to produce the model.
* ``dependencies`` records the versions of core runtime packages.
* ``metrics`` stores evaluation results associated with the artifact.

These fields are metadata contracts; input validation remains the responsibility of the
inference workflow that consumes the bundle.
