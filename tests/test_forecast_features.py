"""Tests for leakage-safe forecast feature construction — plan phase P2.

The contract of :class:`SKSurrogate.temporal.ForecastFeatureBuilder` is narrow and
checkable: every target-derived feature reads at most up to ``t - horizon``,
windows never centre, series never borrow each other's history, and a design
matrix pairs each feature row with the label at exactly ``t + horizon`` (direct)
or ``t + 1`` (recursive).
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from SKSurrogate import TemporalSpec
from SKSurrogate.temporal import (
    ForecastFeatureBuilder,
    LeakageError,
    assert_no_leakage,
    make_supervised,
)


def _series(periods=20, start="2026-01-01", freq="D", offset=0.0):
    """Single-series frame with a strictly increasing, easily checked target."""
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=periods, freq=freq),
            "y": np.arange(periods, dtype=float) + offset,
        }
    )


def _panel(periods=20):
    """Two-series panel; series B is series A shifted by 100."""
    return pd.concat(
        [
            _series(periods).assign(store="A"),
            _series(periods, offset=100.0).assign(store="B"),
        ],
        ignore_index=True,
    )


class TestFeatureValues(unittest.TestCase):
    """Golden values: the shifts are exact, not approximate."""

    def test_lag_reads_exactly_k_steps_back(self):
        frame = _series()
        builder = ForecastFeatureBuilder(
            spec=TemporalSpec(time_column="time", horizon=2), target="y", lags=[2, 3]
        )
        builder.fit(frame)
        features = builder.transform(frame)
        # transform() keeps the frame's index labels; row 4 reads y[2] / y[1].
        self.assertEqual(features.loc[4, "y__lag_2"], 2.0)
        self.assertEqual(features.loc[4, "y__lag_3"], 1.0)
        self.assertEqual(features.index.min(), 3)  # first row with lag_3 present
        self.assertEqual(len(features), len(frame) - 3)

    def test_rolling_window_ends_at_the_horizon(self):
        frame = _series()
        builder = ForecastFeatureBuilder(
            spec=TemporalSpec(time_column="time", horizon=2),
            target="y",
            lags=[2],
            rolling=[{"window": 3, "stats": ["mean", "max"]}],
        )
        builder.fit(frame)
        features = builder.transform(frame)
        # At origin row 4 the window covers rows 0..2 (t-h = 2): values 0,1,2.
        self.assertEqual(features.loc[4, "y__roll_mean_3"], 1.0)
        self.assertEqual(features.loc[4, "y__roll_max_3"], 2.0)
        # No partial windows: three shifted values are needed before the first row.
        self.assertFalse(features["y__roll_mean_3"].isna().any())
        self.assertEqual(features.index.min(), 4)

    def test_diff_and_pct_change_use_the_shifted_history(self):
        frame = _series()
        builder = ForecastFeatureBuilder(
            spec=TemporalSpec(time_column="time", horizon=2),
            target="y",
            lags=[2],
            diffs=[1],
            pct_changes=[1],
        )
        builder.fit(frame)
        features = builder.transform(frame)
        self.assertEqual(features.loc[4, "y__diff_1"], 1.0)
        self.assertAlmostEqual(features.loc[4, "y__pct_change_1"], 1.0 / 1.0)
        self.assertAlmostEqual(features.loc[6, "y__pct_change_1"], 1.0 / 3.0)

    def test_pct_change_with_zero_denominator_is_nan_not_inf(self):
        frame = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=8, freq="D"),
                "y": [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        builder = ForecastFeatureBuilder(
            spec=TemporalSpec(time_column="time", horizon=1),
            target="y",
            pct_changes=[1],
            dropna=False,
        )
        builder.fit(frame)
        features = builder.transform(frame)
        values = features["y__pct_change_1"]
        self.assertFalse(np.isinf(values.to_numpy()).any())
        self.assertTrue(values.isna().any())

    def test_calendar_features_are_deterministic(self):
        frame = _series(periods=10, start="2026-01-01")
        builder = ForecastFeatureBuilder(
            spec=TemporalSpec(time_column="time", horizon=1),
            target="y",
            lags=[1],
            calendar=["dow", "doy", "month", "is_weekend"],
        )
        builder.fit(frame)
        features = builder.transform(frame)
        # lag_1 drops row 0, so the first kept origin is 2026-01-02 (a Friday,
        # dow == 4). 2026-01-03 is a Saturday.
        self.assertEqual(features.index.min(), 1)
        self.assertEqual(features.loc[1, "cal__dow"], 4)
        self.assertEqual(features.loc[2, "cal__dow"], 5)
        self.assertEqual(features.loc[3, "cal__is_weekend"], 1)
        self.assertEqual(features.loc[1, "cal__month"], 1)
        self.assertEqual(features.loc[1, "cal__doy"], 2)

    def test_fourier_terms_are_bounded_and_repeat_with_the_period(self):
        frame = _series(periods=29)
        builder = ForecastFeatureBuilder(
            spec=TemporalSpec(time_column="time", horizon=1),
            target="y",
            lags=[1],
            fourier={"periods": [7], "terms": 2},
        )
        builder.fit(frame)
        features = builder.transform(frame)
        sin_1 = features["fourier__sin_7_1"].to_numpy()
        self.assertTrue(np.all(np.abs(sin_1) <= 1.0 + 1e-12))
        # A period of 7 steps repeats seven rows later.
        self.assertAlmostEqual(float(sin_1[0]), float(sin_1[7]), places=9)

    def test_exogenous_columns_pass_through_unchanged(self):
        frame = _series(periods=10)
        frame["price"] = np.linspace(1.0, 2.0, 10)
        builder = ForecastFeatureBuilder(
            spec=TemporalSpec(time_column="time", horizon=1),
            target="y",
            lags=[1],
            exogenous=["price"],
        )
        builder.fit(frame)
        features = builder.transform(frame)
        kept = features.index
        np.testing.assert_allclose(
            features["price"].to_numpy(), frame.loc[kept, "price"].to_numpy()
        )


class TestSeriesIsolation(unittest.TestCase):
    """One series' history must never enter another series' features."""

    def test_panel_lags_are_computed_per_series(self):
        frame = _panel()
        builder = ForecastFeatureBuilder(
            spec=TemporalSpec(time_column="time", series_id="store", horizon=2),
            target="y",
            lags=[2],
            rolling=[{"window": 3, "stats": ["mean"]}],
        )
        builder.fit(frame)
        features = builder.transform(frame)
        # Rows 4 / 24 are the fifth observation of store A / store B.
        self.assertEqual(features.loc[4, "y__lag_2"], 2.0)
        self.assertEqual(features.loc[24, "y__lag_2"], 102.0)
        # A window on B never averages A's zeros: 100, 101, 102.
        self.assertEqual(features.loc[24, "y__roll_mean_3"], 101.0)

    def test_direct_design_keeps_series_and_target_times_aligned(self):
        frame = _panel()
        design = make_supervised(
            frame,
            TemporalSpec(time_column="time", series_id="store", horizon=2),
            "y",
            lags=[2],
        )
        first = design.meta.iloc[0]
        self.assertEqual(first["series"], "A")
        self.assertEqual(first["origin_time"], pd.Timestamp("2026-01-03"))
        self.assertEqual(first["target_time"], pd.Timestamp("2026-01-05"))
        self.assertEqual(design.y.iloc[0], 4.0)
        self.assertEqual(int(first["target_step"]), 2)


class TestDesignConstruction(unittest.TestCase):
    """``make_supervised`` pairs features with the right label and metadata."""

    def test_direct_design_predicts_horizon_steps_ahead(self):
        frame = _series()
        design = make_supervised(
            frame, TemporalSpec(time_column="time", horizon=2), "y", lags=[2]
        )
        self.assertEqual(design.strategy, "direct")
        self.assertEqual(design.horizon, 2)
        self.assertEqual(design.target_step, 2)
        self.assertEqual(len(design.X), len(design.y))
        self.assertEqual(len(design.X), len(design.meta))
        self.assertEqual(design.dropped_rows, 4)  # two lag NaNs + two target NaNs
        self.assertEqual(design.y.iloc[0], 4.0)
        self.assertEqual(design.meta["origin_time"].iloc[0], pd.Timestamp("2026-01-03"))
        self.assertEqual(design.meta["target_time"].iloc[0], pd.Timestamp("2026-01-05"))

    def test_recursive_design_predicts_the_next_step(self):
        frame = _series()
        design = make_supervised(
            frame,
            TemporalSpec(time_column="time", horizon=2),
            "y",
            strategy="recursive",
            lags=[2],
        )
        self.assertEqual(design.strategy, "recursive")
        self.assertEqual(design.horizon, 2)      # declared forecast horizon
        self.assertEqual(design.target_step, 1)  # label is one step ahead
        self.assertEqual(design.y.iloc[0], 3.0)
        self.assertEqual(design.meta["target_time"].iloc[0], pd.Timestamp("2026-01-04"))

    def test_describe_is_json_safe_and_records_provenance(self):
        import json

        design = make_supervised(
            _panel(),
            TemporalSpec(time_column="time", series_id="store", horizon=3),
            "y",
            lags=[3],
            rolling=[2],
        )
        payload = json.loads(json.dumps(design.describe()))
        self.assertEqual(payload["horizon"], 3)
        self.assertEqual(payload["temporal_spec"]["series_id"], "store")
        shifts = {
            feature["name"]: feature["shift"] for feature in payload["features"]
        }
        self.assertEqual(shifts["y__lag_3"], 3)
        self.assertEqual(shifts["y__roll_mean_2"], 3)  # never below the horizon

    def test_every_target_derived_feature_shift_is_at_least_the_horizon(self):
        design = make_supervised(
            _series(),
            TemporalSpec(time_column="time", horizon=4),
            "y",
            lags=[4, 7],
            rolling=[{"window": 2, "stats": ["mean", "std"]}],
            diffs=[1],
            pct_changes=[2],
        )
        for feature in design.feature_specs:
            if feature.column == "y":
                self.assertGreaterEqual(feature.shift, 4, feature.name)

    def test_dropna_false_keeps_incomplete_rows(self):
        frame = _series()
        design = make_supervised(
            frame,
            TemporalSpec(time_column="time", horizon=2),
            "y",
            lags=[2],
            dropna=False,
        )
        self.assertEqual(len(design.X), len(frame))
        self.assertEqual(design.dropped_rows, 0)


class TestGuardsAndErrors(unittest.TestCase):
    """Horizon rules, ordering, and the empirical leakage guard."""

    def test_lag_shorter_than_horizon_is_rejected(self):
        frame = _series()
        with self.assertRaises(ValueError) as ctx:
            ForecastFeatureBuilder(
                spec=TemporalSpec(time_column="time", horizon=3), target="y", lags=[1]
            ).fit(frame)
        self.assertIn("shorter than the forecast horizon", str(ctx.exception))

    def test_shuffled_frame_is_rejected_at_fit(self):
        frame = _series().sample(frac=1.0, random_state=5).reset_index(drop=True)
        with self.assertRaises(ValueError):
            ForecastFeatureBuilder(
                spec=TemporalSpec(time_column="time", horizon=1), target="y", lags=[1]
            ).fit(frame)

    def test_transform_requires_fit(self):
        builder = ForecastFeatureBuilder(
            spec=TemporalSpec(time_column="time", horizon=1), target="y"
        )
        with self.assertRaises(NotFittedError):
            builder.transform(_series())

    def test_missing_columns_and_bad_options_are_rejected(self):
        frame = _series()
        with self.assertRaises(ValueError):
            ForecastFeatureBuilder(
                spec=TemporalSpec(time_column="time", horizon=1), target="absent"
            ).fit(frame)
        with self.assertRaises(ValueError):
            ForecastFeatureBuilder(
                spec=TemporalSpec(time_column="time", horizon=1),
                target="y",
                exogenous=["absent"],
            ).fit(frame)
        with self.assertRaises(ValueError):
            ForecastFeatureBuilder(
                spec=TemporalSpec(time_column="time", horizon=1),
                target="y",
                rolling=[{"window": 3, "stats": ["nope"]}],
            ).fit(frame)
        with self.assertRaises(ValueError):
            ForecastFeatureBuilder(
                spec=TemporalSpec(time_column="time", horizon=1),
                target="y",
                calendar=["fortnight"],
            ).fit(frame)
        numeric_time = pd.DataFrame({"t": np.arange(10), "y": np.arange(10.0)})
        with self.assertRaises(ValueError) as ctx:
            ForecastFeatureBuilder(
                spec=TemporalSpec(time_column="t", horizon=1),
                target="y",
                calendar=["dow"],
            ).fit(numeric_time)
        self.assertIn("datetime", str(ctx.exception))

    def test_builder_is_clonable(self):
        builder = ForecastFeatureBuilder(
            spec=TemporalSpec(time_column="time", horizon=2),
            target="y",
            lags=[2],
            rolling=[{"window": 3, "stats": ["mean"]}],
        )
        copy = clone(builder)
        self.assertEqual(copy.get_params()["target"], "y")
        self.assertEqual(copy.get_params()["lags"], [2])

    def test_leakage_guard_accepts_the_builder(self):
        frame = _panel()
        spec = TemporalSpec(time_column="time", series_id="store", horizon=2)
        builder = ForecastFeatureBuilder(
            spec=spec, target="y", lags=[2], rolling=[3], diffs=[1], calendar=["dow"]
        )
        builder.fit(frame)
        report = assert_no_leakage(builder, frame, spec)
        self.assertTrue(report["ok"])
        self.assertGreaterEqual(report["cut_points_checked"], 1)

    def test_leakage_guard_rejects_a_future_reading_feature(self):
        frame = _series()
        spec = TemporalSpec(time_column="time", horizon=2)

        class ReadsTheLabel:
            def transform(self, data):
                return pd.DataFrame(
                    {"label": data["y"].astype(float).shift(-2)}, index=data.index
                ).dropna()

        with self.assertRaises(LeakageError):
            assert_no_leakage(ReadsTheLabel(), frame, spec)

    def test_leakage_guard_rejects_a_centered_window(self):
        frame = _series()
        spec = TemporalSpec(time_column="time", horizon=2)

        class CenteredWindow:
            def transform(self, data):
                values = data["y"].astype(float)
                return pd.DataFrame(
                    {"centered": values.rolling(3, center=True, min_periods=1).mean()},
                    index=data.index,
                )

        with self.assertRaises(LeakageError) as ctx:
            assert_no_leakage(CenteredWindow(), frame, spec)
        self.assertIn("change when observations after", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
