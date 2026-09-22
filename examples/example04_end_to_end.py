"""Run a small offline-to-deployment SKSurrogate workflow."""

import tempfile
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from SKSurrogate import ModelBundle, ModelRegistry, drift_report, load_bundle, predict_batch, save_bundle


def main():
    features, target = make_classification(
        n_samples=160,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=7,
    )
    frame = pd.DataFrame(features, columns=["feature_%d" % index for index in range(4)])
    train, serving = train_test_split(frame, test_size=0.25, random_state=7)
    train_target, serving_target = train_test_split(target, test_size=0.25, random_state=7)

    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=7, max_iter=500))
    model.fit(train, train_target)
    accuracy = model.score(serving, serving_target)
    schema = {
        column: {"dtype": str(train[column].dtype)}
        for column in train.columns
    }
    bundle = ModelBundle(
        model,
        task_name="phase10-classification",
        schema=schema,
        metrics={"accuracy": accuracy},
        owner="example-user",
        run_id="phase10-demo-run",
        dependencies={},
    )
    bundle.record_audit_event("training", status="validated")

    with tempfile.TemporaryDirectory(prefix="sksurrogate-phase10-") as workdir:
        workdir = Path(workdir)
        bundle_path = save_bundle(bundle, workdir / "model.bundle")
        loaded = load_bundle(bundle_path)
        registry = ModelRegistry(workdir / "registry")
        version = registry.register(loaded)
        registry.promote(loaded.task_name, version, "production")

        predictions = predict_batch(
            loaded,
            serving,
            output_path=workdir / "predictions.csv",
            request_id="phase10-demo-request",
        )
        drift = drift_report(train, serving, model_version=version)

        print("model version:", version)
        print("validation accuracy: %.3f" % accuracy)
        print("predictions:", len(predictions))
        print("prediction file:", workdir / "predictions.csv")
        print("drift alerts:", len(drift["alerts"]))


if __name__ == "__main__":
    main()