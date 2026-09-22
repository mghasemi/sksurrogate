"""Demonstrate a complete synthetic classification MLOps workflow."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from SKSurrogate import (
    DeploymentApprovalGate,
    InferenceMonitor,
    ModelBundle,
    ModelRegistry,
    assert_bundle_quality,
    delayed_label_performance,
    drift_report,
    fairness_report,
    load_bundle,
    mltrack,
    predict_batch,
    save_bundle,
)


def main():
    # Create a reproducible labeled dataset and reserve an untouched validation set.
    features, labels = make_classification(
        n_samples=240,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        weights=[0.6, 0.4],
        random_state=7,
    )
    columns = ["feature_%d" % index for index in range(features.shape[1])]
    frame = pd.DataFrame(features, columns=columns)
    train, validation, train_labels, validation_labels = train_test_split(
        frame, labels, test_size=0.25, random_state=7, stratify=labels
    )
    train_with_target = train.assign(target=train_labels)
    validation_with_target = validation.assign(target=validation_labels)

    with tempfile.TemporaryDirectory(prefix="sksurrogate-classification-") as workdir:
        workdir = Path(workdir)
        # Register explicit partitions so model selection cannot use validation labels.
        tracker = mltrack("synthetic-classification", db_name=str(workdir / "mltrace.db"))
        tracker.RegisterData(train_with_target, "target", partition="train")
        tracker.RegisterData(validation_with_target, "target", partition="validation")
        evaluation = tracker.nested_cv_evaluation(
            LogisticRegression(max_iter=500, random_state=7),
            inner_cv=3,
            outer_cv=3,
            train_partition="train",
            validation_partition="validation",
        )

        # Fit the candidate on training data and capture its serving contract.
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=500, random_state=7)
        )
        model.fit(train, train_labels)
        validation_score = model.score(validation, validation_labels)
        schema = {column: {"dtype": str(train[column].dtype)} for column in columns}
        bundle = ModelBundle(
            model,
            task_name="synthetic-classification",
            schema=schema,
            metrics={
                "accuracy": float(validation_score),
                "mean_outer_score": float(evaluation["mean_outer_score"]),
            },
            owner="tutorial-user",
            run_id="synthetic-classification-run",
            dependencies={},
        )
        bundle.record_audit_event(
            "training",
            validation_accuracy=float(validation_score),
            outer_score=float(evaluation["mean_outer_score"]),
        )
        bundle_path = save_bundle(bundle, workdir / "model.bundle")
        loaded = load_bundle(bundle_path)

        # Require schema compatibility and a minimum metric before deployment.
        quality = assert_bundle_quality(
            bundle_path,
            expected_schema=schema,
            metric_thresholds={"accuracy": 0.70},
        )
        registry = ModelRegistry(workdir / "registry")
        model_version = registry.register(loaded)
        gate = DeploymentApprovalGate(registry)
        gate.promote(
            loaded.task_name,
            model_version,
            "production",
            approvers=["reviewer"],
            quality_report=quality,
        )

        # Score shifted serving data, then inspect input drift and prediction quality.
        serving = validation.copy()
        serving["feature_0"] += 2.5
        predictions = predict_batch(
            loaded,
            serving,
            output_path=workdir / "predictions.csv",
            request_id="synthetic-classification-request",
        )
        predicted_labels = predictions["prediction"].to_numpy()
        drift = drift_report(train, serving, model_version=model_version)
        delayed = delayed_label_performance(
            predicted_labels, validation_labels, model_version=model_version
        )
        # Use a simple feature-derived grouping to demonstrate fairness reporting.
        groups = np.where(validation["feature_0"].to_numpy() >= 0.0, "high", "low")
        fairness = fairness_report(validation_labels, predicted_labels, groups)
        monitor = InferenceMonitor(model_version)
        monitor.record(
            predictions.attrs["inference_metrics"]["latency_ms"],
            predictions.attrs["inference_metrics"]["rows"],
        )

        print("outer CV score: %.3f" % evaluation["mean_outer_score"])
        print("validation accuracy: %.3f" % validation_score)
        print("model version:", model_version)
        print("production predictions:", len(predictions))
        print("drift alerts:", len(drift["alerts"]))
        print("delayed accuracy: %.3f" % delayed["metrics"]["accuracy"])
        print("fairness gap: %.3f" % fairness["fairness"]["demographic_parity_gap"])
        print("runtime requests:", monitor.summary()["requests"])
        print("audit events:", len(registry.audit_log(loaded.task_name)))
        tracker.close()


if __name__ == "__main__":
    main()