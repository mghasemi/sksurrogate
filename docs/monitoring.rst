====================
Drift Monitoring
====================

``drift_report`` compares a reference training or validation DataFrame with current
tabular data. It keeps schema and data-quality changes separate from statistical
distribution drift and returns a JSON-safe report suitable for logging or alerting.

.. code-block:: python

    from SKSurrogate import drift_report

    report = drift_report(
        reference_train,
        current_batch,
        model_version="customer-churn-v3",
        reference_dataset_fingerprint="train-fingerprint",
        current_dataset_fingerprint="batch-fingerprint",
        psi_threshold=0.2,
        categorical_threshold=0.2,
        missingness_threshold=0.1,
        range_threshold=0.1,
    )

Report sections
===============

* ``schema_changes`` lists missing columns, extra columns, reordered columns, and
  dtype changes.
* ``data_quality`` records row counts and per-column reference/current missingness
  rates and deltas.
* ``drift.numeric`` reports population stability index (PSI) values and reference
  quantile bins for numeric features.
* ``drift.categorical`` reports total variation distance and both categorical
  distributions.
* ``drift.range`` reports the reference min/max, current min/max, and the current
  out-of-reference-range rate for numeric features.
* ``alerts`` contains threshold crossings with their type, column, and measured value.
* ``model_version`` and dataset fingerprints identify the deployment and datasets
  associated with the report.

Schema changes and missingness are reported even when a column cannot be compared for
distribution drift. Numeric PSI uses reference-derived quantile bins; categorical drift
includes values observed only in the current data. Thresholds are configurable per
report, and the default values are available under ``report["thresholds"]``.

Range drift is reported for numeric features as the current out-of-reference-range
rate. Use ``prediction_distribution_report`` for prediction drift, and
``delayed_label_performance`` when labels arrive after inference:

.. code-block:: python

    from SKSurrogate import delayed_label_performance, prediction_distribution_report

    prediction_report = prediction_distribution_report(
        reference_predictions, current_predictions, model_version="model-v3"
    )
    performance = delayed_label_performance(
        predictions, delayed_labels, model_version="model-v3"
    )

For regression tasks, pass ``task_type="regression"`` to receive delayed-label MAE
and MSE instead of classification accuracy. Prediction drift supports numeric and
categorical prediction values and uses the same thresholded alert format as feature
drift.

Governance and fairness checks
==============================

The monitoring module also includes governance checks for model fairness and sensitive
feature use before training or production serving. Use ``sensitive_feature_report`` to
highlight PII-like and user-declared sensitive columns before they reach a model:

.. code-block:: python

    import pandas as pd
    from SKSurrogate import sensitive_feature_report

    frame = pd.DataFrame({
        "email": ["a@example.com", "b@example.com"],
        "ssn": ["123", "456"],
        "amount": [25.0, 42.0],
    })
    report = sensitive_feature_report(frame, sensitive_features=["ssn"])

The report returns ``pii_columns`` and ``sensitive_columns`` together with a list of
warning strings, which can be attached to a training run or CI gate before model
registration. Sensitive keys such as ``api_key``, ``password``, and ``token`` are also
redacted automatically from bundle audit metadata and registry event payloads.

Use ``fairness_report`` to compare selection rates and true-positive rates across
configured groups. This is useful for detecting demographic parity or equal-opportunity
imbalance before a model is promoted to production:

.. code-block:: python

    import numpy as np
    from SKSurrogate import fairness_report

    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    fairness = fairness_report(y_true, y_pred, groups)

The returned dictionary contains a per-group summary under ``fairness["groups"]`` and
aggregate gaps under ``fairness["fairness"]``. The demographic parity gap measures the
spread in positive prediction rate across groups; the equal-opportunity gap measures the
spread in true-positive rate across groups.

Use ``subgroup_performance_report`` to compare any metric across groups and quantify the
largest observed gap:

.. code-block:: python

    from SKSurrogate import subgroup_performance_report

    subgroup_metrics = subgroup_performance_report(
        y_true,
        y_pred,
        groups,
        metric="accuracy",
    )

The report returns a metric value for each group together with ``fairness_gap``. The same
helper supports ``accuracy``, ``precision``, ``recall``, ``f1``, and ``loss`` metrics.

``InferenceMonitor`` accumulates request latency, error rate, row throughput, and
model version for a serving process:

.. code-block:: python

  from SKSurrogate import InferenceMonitor

  monitor = InferenceMonitor(model_version="model-v3")
  monitor.record(latency_ms=42.5, rows=500, success=True)
  monitor.record(latency_ms=80.0, rows=500, success=False)
  summary = monitor.summary()

The summary contains request count, row count, error rate, mean latency, and aggregate
row throughput. Persistent monitoring storage and configurable alert sinks remain
planned Phase 7 work.