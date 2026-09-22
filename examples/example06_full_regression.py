"""Demonstrate a complete synthetic regression MLOps workflow."""

import tempfile
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
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
    load_bundle,
    mltrack,
    predict_batch,
    save_bundle,
)


def main():
    # Create a reproducible continuous-target dataset and reserve validation rows.
    generated = make_regression(
        n_samples=240,
        n_features=6,
        n_informative=4,
        noise=8.0,
        random_state=7,
        coef=False,
    )
    features, labels = generated[0], generated[1]
    columns = ["feature_%d" % index for index in range(features.shape[1])]
    frame = pd.DataFrame(features, columns=columns)
    train, validation, train_labels, validation_labels = train_test_split(
        frame, labels, test_size=0.25, random_state=7
    )
    train_with_target = train.assign(target=train_labels)
    validation_with_target = validation.assign(target=validation_labels)

    with tempfile.TemporaryDirectory(prefix="sksurrogate-regression-") as workdir:
        workdir = Path(workdir)
        # Register explicit partitions so evaluation remains reproducible and isolated.
        tracker = mltrack("synthetic-regression", db_name=str(workdir / "mltrace.db"))
        tracker.RegisterData(train_with_target, "target", partition="train")
        tracker.RegisterData(validation_with_target, "target", partition="validation")
        evaluation = tracker.nested_cv_evaluation(
            LinearRegression(),
            inner_cv=3,
            outer_cv=3,
            train_partition="train",
            validation_partition="validation",
        )

        # Fit the candidate and capture its feature schema and regression metrics.
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(train, train_labels)
        validation_score = model.score(validation, validation_labels)
        validation_predictions = model.predict(validation)
        schema = {column: {"dtype": str(train[column].dtype)} for column in columns}
        bundle = ModelBundle(
            model,
            task_name="synthetic-regression",
            schema=schema,
            metrics={
                "r2": float(validation_score),
                "mean_outer_score": float(evaluation["mean_outer_score"]),
            },
            owner="tutorial-user",
            run_id="synthetic-regression-run",
            dependencies={},
        )
        bundle.record_audit_event(
            "training",
            validation_r2=float(validation_score),
            outer_score=float(evaluation["mean_outer_score"]),
        )
        bundle_path = save_bundle(bundle, workdir / "model.bundle")
        loaded = load_bundle(bundle_path)

        # Require schema compatibility and a minimum R2 score before deployment.
        quality = assert_bundle_quality(
            bundle_path,
            expected_schema=schema,
            metric_thresholds={"r2": 0.80},
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

        # Shift one serving feature to demonstrate numeric drift and delayed labels.
        serving = validation.copy()
        serving["feature_0"] += 18.0
        predictions = predict_batch(
            loaded,
            serving,
            output_path=workdir / "predictions.csv",
            request_id="synthetic-regression-request",
        )
        predicted_values = predictions["prediction"].to_numpy()
        drift = drift_report(train, serving, model_version=model_version)
        delayed = delayed_label_performance(
            predicted_values,
            validation_labels,
            model_version=model_version,
            task_type="regression",
        )
        monitor = InferenceMonitor(model_version)
        monitor.record(
            predictions.attrs["inference_metrics"]["latency_ms"],
            predictions.attrs["inference_metrics"]["rows"],
        )

        print("outer CV R2: %.3f" % evaluation["mean_outer_score"])
        print("validation R2: %.3f" % validation_score)
        print("validation MAE: %.3f" % abs(validation_labels - validation_predictions).mean())
        print("model version:", model_version)
        print("production predictions:", len(predictions))
        print("drift alerts:", len(drift["alerts"]))
        print("delayed MAE: %.3f" % delayed["metrics"]["mae"])
        print("delayed MSE: %.3f" % delayed["metrics"]["mse"])
        print("runtime requests:", monitor.summary()["requests"])
        print("audit events:", len(registry.audit_log(loaded.task_name)))
        tracker.close()


if __name__ == "__main__":
    main()