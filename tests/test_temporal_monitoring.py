"""Phase P8.1 tests: time-window drift and horizon-aware delayed labels.

Covers the two monitoring extensions in ``TIME_SERIES_PLAN.md`` phase 8:

* ``drift_report`` window mode — two-frame trailing windows and the
  single-frame "last n versus the previous n" form, with the report recording
  which mode ran and refusing impossible windows instead of silently scoring a
  shorter slice;
* ``delayed_label_performance`` — per-horizon scoring and a ``maturity`` gate so
  a horizon-12 label is not scored before it exists, with pendings counted
  rather than dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from SKSurrogate import delayed_label_performance, drift_report


def _timeline(n: int = 200, shift_from: int | None = None, shift: float = 5.0, seed: int = 0):
    """Ordered unit-spaced series; rows from ``shift_from`` on are displaced."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=n)
    if shift_from is not None:
        values[shift_from:] += shift
    return pd.DataFrame(
        {"time": np.arange(n, dtype=int), "value": values}
    )


class TestDriftWindowMode:
    def test_single_frame_previous_vs_last_window(self):
        frame = _timeline(shift_from=150)
        report = drift_report(
            None, frame, reference_window=50, current_window=50, time_column="time"
        )
        assert report["windows"]["mode"] == "single_frame_previous_tail"
        assert report["windows"]["reference"] == 50
        assert report["windows"]["current"] == 50
        # Post-trim row counts: the comparison saw exactly two 50-row windows.
        assert report["data_quality"]["reference_rows"] == 50
        assert report["data_quality"]["current_rows"] == 50
        # The displacement is inside the current window, so numeric drift fires.
        psi = report["drift"]["numeric"]["value"]["value"]
        assert psi > 0.2
        assert any(alert["type"] == "numeric_drift" for alert in report["alerts"])

    def test_two_frame_tail_windows_respect_time_order(self):
        frame = _timeline(300, seed=1)
        # Same values, shuffled: after ordering by time the tails are identical,
        # so any drift alert would mean the slice took arbitrary rows.
        shuffled = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
        report = drift_report(
            frame,
            shuffled,
            reference_window=40,
            current_window=40,
            time_column="time",
        )
        assert report["windows"]["mode"] == "two_frames_tail"
        assert report["data_quality"]["reference_rows"] == 40
        assert report["data_quality"]["current_rows"] == 40
        assert not [a for a in report["alerts"] if a["type"] == "numeric_drift"]

    def test_windows_ignore_older_observations(self):
        """A big shift far in the past must not leak into a recent-window report."""
        t = np.arange(300)
        # Repetition period divides the window offset (5 | 50), so the two
        # trailing windows hold *identical* values — any reported value drift
        # could only come from the ancient regime change.
        frame = pd.DataFrame({"time": t.astype(float), "value": ((t % 5) + 1).astype(float)})
        frame.loc[:99, "value"] += 10.0  # ancient regime change
        report = drift_report(
            None, frame, reference_window=50, current_window=50, time_column="time"
        )
        assert not [
            a for a in report["alerts"] if a["type"] == "numeric_drift" and a["column"] == "value"
        ]
        assert report["drift"]["numeric"]["value"]["value"] == pytest.approx(0.0)
        # The timeline marker itself is not a comparison column (it must differ).
        assert "time" not in report["drift"]["numeric"]

    def test_whole_frame_mode_reports_no_windows(self):
        report = drift_report(_timeline(), _timeline(seed=5))
        assert report["windows"] == {"mode": None, "reference": None, "current": None}

    def test_window_errors_are_actionable(self):
        frame = _timeline(30)
        with pytest.raises(ValueError, match="frame has 30"):
            drift_report(None, frame, reference_window=20, current_window=20)
        with pytest.raises(ValueError, match="at least 1"):
            drift_report(frame, frame, current_window=0)
        with pytest.raises(ValueError, match="needs reference_window"):
            drift_report(None, frame, current_window=10)


class TestDelayedLabelMaturity:
    def test_per_horizon_scoring_separates_steps(self):
        labels = np.array([1, 0, 1, 1, 0, 0])
        predictions = np.array([1, 0, 0, 1, 1, 1])
        horizons = np.array([1, 1, 1, 2, 2, 2])
        report = delayed_label_performance(
            predictions, labels, task_type="classification", horizons=horizons
        )
        # Horizon 1: 2 of 3 right; horizon 2: 1 of 3 right.
        assert report["by_horizon"]["1"]["accuracy"] == pytest.approx(2 / 3)
        assert report["by_horizon"]["2"]["accuracy"] == pytest.approx(1 / 3)
        assert report["horizons"] == [1, 2]
        # Pooled metric is over all six rows (3 of 6), not one step's score.
        assert report["metrics"]["accuracy"] == pytest.approx(0.5)

    def test_maturity_scores_only_elapsed_labels(self):
        labels = np.array([1, 1, 1, 0, 0, 0])
        predictions = np.zeros(6, dtype=int)  # always predicts 0
        horizons = np.array([1, 1, 2, 2, 3, 3])
        report = delayed_label_performance(
            predictions,
            labels,
            task_type="classification",
            horizons=horizons,
            maturity=2,
        )
        # Only the four horizon<=2 rows are scored: one of four right.
        assert report["rows"] == 4
        assert report["metrics"]["accuracy"] == pytest.approx(0.25)
        assert report["pending_rows"] == 2
        assert report["maturity"] == 2
        assert set(report["by_horizon"]) == {"1", "2"}
        # Horizon 1: labels 1,1 → both wrong. Horizon 2: labels 1,0 → one right.
        assert report["by_horizon"]["1"]["accuracy"] == 0.0
        assert report["by_horizon"]["2"]["accuracy"] == pytest.approx(0.5)

    def test_maturity_without_horizons_is_refused(self):
        with pytest.raises(ValueError, match="maturity needs horizons"):
            delayed_label_performance([1, 0], [1, 0], maturity=1)

    def test_nothing_matured_yet_is_refused_not_empty(self):
        with pytest.raises(ValueError, match="no labels have matured"):
            delayed_label_performance([1, 0], [1, 0], horizons=[5, 5], maturity=2)

    def test_plain_call_is_unchanged(self):
        report = delayed_label_performance([1, 0, 1], [1, 0, 0], task_type="classification")
        assert report["rows"] == 3
        assert report["metrics"]["accuracy"] == pytest.approx(2 / 3)
        assert "by_horizon" not in report
        assert "maturity" not in report
        assert "pending_rows" not in report
