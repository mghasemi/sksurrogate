====================================
Model Bundles and Registry
====================================

``SKSurrogate`` provides portable model bundles for moving a fitted estimator from
training to validation or deployment. A bundle stores the fitted model together with
its preprocessing reference, feature schema, metrics, dataset fingerprint, configuration
version, runtime dependency versions, task name, and model version.

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

``load_bundle`` checks the bundle format, and can reject an incompatible feature schema
or runtime dependency set. Pass ``strict_dependencies=False`` when the environment is
managed separately and dependency comparison is not required.

The artifact is serialized with ``joblib`` and should only be loaded from trusted
sources. The model version is generated automatically unless ``model_version`` is
provided explicitly.

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
