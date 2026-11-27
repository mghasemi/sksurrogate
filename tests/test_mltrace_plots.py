"""Learning-curve plotting: the measure must match the task.

The regression case is a regression test: ``plot_learning_curve`` used to fall back to
``accuracy`` for any unrecognised ``measure``, so a regressor produced an all-NaN chart
inside scikit-learn's warning instead of a curve.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold

from SKSurrogate import mltrack


def _tracker(tmp_path, frame, target, cv):
    tracker = mltrack("plot-learning-curve", db_name=str(tmp_path / "mltrace.db"))
    tracker.RegisterData(frame, target)
    tracker.SetCV(cv)
    return tracker


def _curve_scores(plt_object):
    """Return the (train, validation) y-values of the two plotted curves."""
    axes = plt_object.gcf().axes[0]
    train_line, validation_line = axes.get_lines()[:2]
    return (
        np.asarray(train_line.get_ydata(), dtype=float),
        np.asarray(validation_line.get_ydata(), dtype=float),
    )


def test_regression_learning_curve_uses_r2(tmp_path):
    rng = np.random.default_rng(7)
    frame = {
        "x1": rng.normal(size=160),
        "x2": rng.normal(size=160),
        "y": rng.normal(size=160),
    }
    import pandas as pd

    data = pd.DataFrame(frame)
    data["y"] = data["x1"] * 2.0 + rng.normal(scale=0.2, size=160)

    tracker = _tracker(tmp_path, data, "y", KFold(n_splits=3, shuffle=True, random_state=0))
    model = tracker.LogModel(RandomForestRegressor(n_estimators=12, random_state=0))
    try:
        plot = tracker.plot_learning_curve(
            model,
            "regression learning curve",
            measure="r2",
            train_sizes=np.linspace(0.4, 1.0, 3),
        )
        train_scores, validation_scores = _curve_scores(plot)
    finally:
        tracker.close()

    assert np.isfinite(train_scores).all(), train_scores
    assert np.isfinite(validation_scores).all(), validation_scores
    # A learnable target: held-out R2 has to beat the constant-mean baseline (0.0).
    assert validation_scores.min() > 0.5


def test_regression_learning_curve_rejects_classification_measure(tmp_path):
    import pandas as pd

    rng = np.random.default_rng(3)
    data = pd.DataFrame({"x": rng.normal(size=80), "y": rng.normal(size=80)})
    tracker = _tracker(tmp_path, data, "y", KFold(n_splits=3, shuffle=True, random_state=0))
    model = tracker.LogModel(RandomForestRegressor(n_estimators=8, random_state=0))
    try:
        with pytest.raises(ValueError, match="not available for a regression target"):
            tracker.plot_learning_curve(model, "bad measure", measure="accuracy")
    finally:
        tracker.close()


def test_classification_learning_curve_keeps_accuracy_default(tmp_path):
    import pandas as pd

    rng = np.random.default_rng(11)
    data = pd.DataFrame(
        {
            "x1": rng.normal(size=200),
            "x2": rng.normal(size=200),
            "label": (rng.normal(size=200) > 0).astype(int),
        }
    )
    tracker = _tracker(
        tmp_path, data, "label", StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    )
    model = tracker.LogModel(RandomForestClassifier(n_estimators=10, random_state=0))
    try:
        plot = tracker.plot_learning_curve(model, "classification learning curve")
        train_scores, validation_scores = _curve_scores(plot)
    finally:
        tracker.close()

    assert np.isfinite(train_scores).all()
    assert np.isfinite(validation_scores).all()
    assert 0.0 <= validation_scores.min() <= 1.0
