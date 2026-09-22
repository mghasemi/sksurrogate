"""Measure and check the main portable MLOps paths."""

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from SKSurrogate import ModelBundle, drift_report, load_bundle, predict_batch, save_bundle


THRESHOLDS_SECONDS = {
    "bundle_round_trip": 2.0,
    "batch_prediction": 2.0,
    "drift_report": 2.0,
}


def _measure(operation, repetitions):
    durations = []
    for _ in range(repetitions):
        started = time.perf_counter()
        operation()
        durations.append(time.perf_counter() - started)
    return statistics.median(durations)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--check", action="store_true", help="fail when a threshold is exceeded")
    arguments = parser.parse_args(argv)
    if arguments.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    features, target = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=4,
        random_state=7,
    )
    frame = pd.DataFrame(features, columns=["feature_%d" % index for index in range(6)])
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=7, max_iter=500))
    model.fit(frame, target)
    bundle = ModelBundle(
        model,
        task_name="phase10-benchmark",
        schema={column: {"dtype": str(frame[column].dtype)} for column in frame.columns},
        dependencies={},
    )

    with tempfile.TemporaryDirectory(prefix="sksurrogate-benchmark-") as workdir:
        bundle_path = Path(workdir) / "model.bundle"
        save_bundle(bundle, bundle_path)

        timings = {
            "bundle_round_trip": _measure(lambda: load_bundle(bundle_path), arguments.repetitions),
            "batch_prediction": _measure(lambda: predict_batch(bundle, frame), arguments.repetitions),
            "drift_report": _measure(lambda: drift_report(frame, frame, model_version=bundle.model_version), arguments.repetitions),
        }

    failed = []
    for name, duration in timings.items():
        threshold = THRESHOLDS_SECONDS[name]
        print("%s: %.6fs (threshold %.1fs)" % (name, duration, threshold))
        if duration > threshold:
            failed.append(name)
    if arguments.check and failed:
        raise SystemExit("performance thresholds exceeded: %s" % ", ".join(failed))


if __name__ == "__main__":
    main()