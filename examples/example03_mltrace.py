"""Demonstrate the SKSurrogate mltrace workflow."""

import tempfile

import matplotlib

matplotlib.use("Agg")

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit

from SKSurrogate import mltrack, np2df


X, y = make_classification(
    n_samples=160,
    n_features=6,
    n_informative=4,
    n_redundant=1,
    random_state=7,
)
columns = [f"feature_{index}" for index in range(X.shape[1])] + ["target"]
frame = np2df(np.column_stack((X, y)), columns)()

with tempfile.TemporaryDirectory(prefix="sksurrogate-mltrace-") as workdir:
    tracker = mltrack(
        "classification-demo",
        db_name=f"{workdir}/mltrace.db",
        cv=ShuffleSplit(n_splits=3, test_size=0.25, random_state=7),
    )
    tracker.register_data(frame, "target")
    tracker.update_task(
        {"description": "A small reproducible mltrace classification workflow."}
    )

    weights = tracker.feature_weights(weights=("pearson", "variance"))
    model = tracker.log_model(
        RandomForestClassifier(n_estimators=20, random_state=7),
        "Random forest demo",
    )
    metrics = tracker.log_metrics(model)

    tracker.plot_learning_curve(
        model,
        "Random forest learning curve",
        measure="accuracy",
        train_sizes=[0.5, 1.0],
    )
    tracker.plot_calibration_curve(model, "Random forest calibration", fig_index=2)
    tracker.plot_roc_curve(model, "Random forest ROC")
    tracker.plot_cumulative_gain(model, title="Random forest cumulative gain")
    tracker.plot_lift_curve(model, title="Random forest lift")

    tracker.preserve_model(model)
    recovered = tracker.recover_model(model.mltrack_id)

    print("data shape:", tracker.get_dataframe().shape)
    print("metrics:", metrics)
    print("weighted features:", list(weights["feature"]))
    print("top features:", list(tracker.top_features(3).keys()))
    print("stored plots:", len(tracker.all_plots(model.mltrack_id)))
    print("recovered model:", type(recovered).__name__)
