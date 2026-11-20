"""Tests for horizon-aware prediction intervals — plan phase P5.

The calibration is split-conformal on backtest residuals, so the tests pin three
things: the order statistics are the ones claimed (hand-computed), the coverage is
approximately nominal on fresh data (Monte-Carlo, deterministic seed), and a bundle
without a calibration refuses to emit an interval instead of inventing a width of zero.
"""

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from SKSurrogate import (
    IntervalCalibration,
    ModelBundle,
    apply_calibration,
    calibrate_from_residuals,
    conformal_quantile_level,
    interval_diagnostics,
    load_bundle,
    mltrack,
    save_bundle,
)

from api.config import settings
from api.deps import open_tracker
from api.jobs import job_manager
from api.routers import evaluation as evaluation_router
from api.routers.datasets import TemporalSpecRequest, set_temporal


class TestConformalLevel(unittest.TestCase):
    def test_finite_sample_level_formula(self):
        # ceil((n+1)(1-alpha))/n, capped at 1
        self.assertEqual(conformal_quantile_level(9, 0.1), 1.0)  # 9/9
        self.assertAlmostEqual(conformal_quantile_level(99, 0.1), 90.0 / 99.0)
        self.assertAlmostEqual(conformal_quantile_level(19, 0.05), 19.0 / 19.0)
        self.assertAlmostEqual(conformal_quantile_level(999, 0.05), 950.0 / 999.0)

    def test_invalid_inputs(self):
        for alpha in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                conformal_quantile_level(10, alpha)
        with self.assertRaises(ValueError):
            conformal_quantile_level(0, 0.1)


class TestCalibration(unittest.TestCase):
    def test_absolute_offsets_are_the_conformal_quantile(self):
        residuals = [-2.0, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 2.0, 0.0, 0.0]
        calibration = calibrate_from_residuals(residuals, alpha=0.2, method="absolute")
        # level = ceil(11 * 0.8) / 10 = 0.9 -> 90th percentile of |r| is 2
        self.assertAlmostEqual(float(calibration.finite_sample_level), 0.9)
        self.assertEqual(calibration.lower_offset, -2.0)
        self.assertEqual(calibration.upper_offset, 2.0)
        self.assertEqual(calibration.n_residuals, 10)
        self.assertEqual(calibration.empirical_coverage, 1.0)

    def test_asymmetric_offsets_are_the_order_statistics(self):
        residuals = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        calibration = calibrate_from_residuals(residuals, alpha=0.2, method="asymmetric")
        # low index = floor(11 * 0.1) = 1 -> -2
        # high index = ceil(11 * 0.9) - 1 = 9 -> 6
        self.assertEqual(calibration.lower_offset, -2.0)
        self.assertEqual(calibration.upper_offset, 6.0)
        self.assertEqual(calibration.empirical_coverage, 0.9)

    def test_zero_residuals_give_a_zero_width_interval(self):
        calibration = calibrate_from_residuals([0.0] * 12, alpha=0.1)
        self.assertEqual(calibration.lower_offset, 0.0)
        self.assertEqual(calibration.upper_offset, 0.0)

    def test_min_residuals_and_validation(self):
        with self.assertRaises(ValueError):
            calibrate_from_residuals([0.1, 0.2, 0.3], alpha=0.1, min_residuals=10)
        with self.assertRaises(ValueError):
            calibrate_from_residuals(np.arange(10.0), alpha=1.0)
        with self.assertRaises(ValueError):
            calibrate_from_residuals(np.arange(10.0), alpha=0.1, method="bootstrap")

    def test_round_trip_and_bounds(self):
        calibration = calibrate_from_residuals(np.linspace(-2, 2, 21), alpha=0.2)
        restored = IntervalCalibration.from_dict(calibration.to_dict())
        self.assertEqual(restored.to_dict(), calibration.to_dict())
        lower, upper = calibration.bounds([10.0])
        self.assertAlmostEqual(float(lower[0]), 10.0 + calibration.lower_offset)
        self.assertAlmostEqual(float(upper[0]), 10.0 + calibration.upper_offset)
        lower2, upper2 = apply_calibration([10.0], calibration.to_dict())
        self.assertAlmostEqual(float(lower2[0]), float(lower[0]))
        self.assertAlmostEqual(float(upper2[0]), float(upper[0]))


class TestMonteCarloCoverage(unittest.TestCase):
    """On fresh, exchangeable residuals the interval covers about 1 - alpha."""

    def _coverage(self, method):
        rng = np.random.default_rng(12345)
        calibration_set = rng.normal(size=1000)
        evaluation_set = rng.normal(size=5000)
        calibration = calibrate_from_residuals(
            calibration_set, alpha=0.1, method=method
        )
        lower, upper = calibration.bounds(np.zeros_like(evaluation_set))
        covered = (evaluation_set >= lower) & (evaluation_set <= upper)
        return float(np.mean(covered))

    def test_absolute_method_covers_nominal(self):
        coverage = self._coverage("absolute")
        # Binomial standard error at n=5000, p=0.9 is ~0.004; allow a wide margin.
        self.assertAlmostEqual(coverage, 0.90, delta=0.02)

    def test_asymmetric_method_covers_nominal(self):
        coverage = self._coverage("asymmetric")
        self.assertAlmostEqual(coverage, 0.90, delta=0.02)


class TestIntervalDiagnostics(unittest.TestCase):
    def test_horizon_grouping_splits_coverage_by_step(self):
        actual = [1.0, 5.0, 1.2, 4.0, 0.9, 6.0]
        lower = [0.0, 4.0, 0.0, 3.0, 0.0, 5.0]
        upper = [2.0, 6.0, 2.0, 5.0, 2.0, 6.0]
        report = interval_diagnostics(
            actual, lower, upper, alpha=0.2, horizon=[1, 2, 1, 2, 1, 2]
        )
        self.assertTrue(report["horizon_grouped"])
        groups = {entry["group"]: entry for entry in report["subgroup_coverage"]["groups"]}
        self.assertEqual(set(groups), {"1", "2"})
        # Every observation sits inside its interval (the closed upper bound counts),
        # so both horizon groups are fully covered.
        self.assertEqual(groups["1"]["empirical_coverage"], 1.0)
        self.assertEqual(groups["2"]["empirical_coverage"], 1.0)
        self.assertEqual(groups["1"]["row_count"], 3)

    def test_without_groups_there_is_no_subgroup_block(self):
        report = interval_diagnostics([1.0], [0.0], [2.0], alpha=0.2)
        self.assertFalse(report["horizon_grouped"])
        self.assertIsNone(report["subgroup_coverage"])


class TestBacktestIntervals(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self._tmp.name) / "intervals.db")
        rng = np.random.default_rng(7)
        self.frame = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=160, freq="D"),
                "y": np.cumsum(rng.normal(size=160)),
            }
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _tracker(self, horizon=2):
        tracker = mltrack("intervals-task", db_name=self.db)
        tracker.RegisterData(
            self.frame, "y", partition="train",
            temporal={"time_column": "time", "horizon": horizon},
        )
        return tracker

    def test_backtest_reports_a_calibration_and_per_horizon_diagnostics(self):
        tracker = self._tracker(horizon=2)
        try:
            result = tracker.backtest(
                LinearRegression(), n_splits=20, horizon=2, step=2, alpha=0.1
            )
        finally:
            tracker.close()
        intervals = result["intervals"]
        self.assertTrue(intervals["available"])
        calibration = intervals["calibration"]
        self.assertEqual(calibration["target_step"], 2)
        self.assertEqual(calibration["alpha"], 0.1)
        self.assertLessEqual(calibration["lower_offset"], 0.0)
        self.assertGreaterEqual(calibration["upper_offset"], 0.0)
        self.assertLessEqual(calibration["lower_offset"], calibration["upper_offset"])
        diagnostics = intervals["diagnostics"]
        self.assertTrue(diagnostics["horizon_grouped"])
        groups = diagnostics["subgroup_coverage"]["groups"]
        self.assertEqual([entry["group"] for entry in groups], ["2"])
        self.assertIn("mean_interval_width", diagnostics)

    def test_width_grows_with_the_horizon_on_a_random_walk(self):
        widths = {}
        tracker = self._tracker(horizon=1)
        try:
            for horizon in (1, 2, 3, 4):
                result = tracker.backtest(
                    LinearRegression(),
                    n_splits=15,
                    horizon=horizon,
                    step=horizon,
                    alpha=0.2,
                    min_residuals=10,
                    feature_kwargs={"lags": [horizon]},
                    baselines=("naive",),
                )
                intervals = result["intervals"]
                self.assertTrue(intervals["available"], intervals)
                calibration = intervals["calibration"]
                widths[horizon] = calibration["upper_offset"] - calibration["lower_offset"]
        finally:
            tracker.close()
        ordered = [widths[horizon] for horizon in (1, 2, 3, 4)]
        for previous, current in zip(ordered, ordered[1:]):
            self.assertGreaterEqual(current, previous * 0.98, ordered)

    def test_too_few_residuals_is_reported_not_faked(self):
        tracker = self._tracker(horizon=1)
        try:
            result = tracker.backtest(
                LinearRegression(), n_splits=2, horizon=1, min_residuals=10
            )
        finally:
            tracker.close()
        intervals = result["intervals"]
        self.assertFalse(intervals["available"])
        self.assertIn("residual", intervals["reason"])
        self.assertNotIn("calibration", intervals)

    def test_invalid_alpha_and_method_are_refused(self):
        tracker = self._tracker(horizon=1)
        try:
            with self.assertRaises(ValueError):
                tracker.backtest(LinearRegression(), n_splits=5, alpha=1.0)
            with self.assertRaises(ValueError):
                tracker.backtest(
                    LinearRegression(), n_splits=5, interval_method="bootstrap"
                )
        finally:
            tracker.close()


class TestBundleCalibration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "v1.bundle"

    def tearDown(self):
        self._tmp.cleanup()

    def test_calibration_round_trips_through_save_and_load(self):
        calibration = calibrate_from_residuals(np.linspace(-2, 2, 21), alpha=0.2).to_dict()
        bundle = ModelBundle(
            LinearRegression().fit([[0.0], [1.0]], [0.0, 1.0]),
            task_name="t",
            model_version="v1",
            interval_calibration=calibration,
        )
        save_bundle(bundle, self.path)
        loaded = load_bundle(self.path, strict_dependencies=False)
        self.assertEqual(loaded.interval_calibration, calibration)
        lower, upper = loaded.interval_bounds([5.0])
        self.assertAlmostEqual(float(lower[0]), 5.0 + calibration["lower_offset"])
        self.assertAlmostEqual(float(upper[0]), 5.0 + calibration["upper_offset"])

    def test_uncalibrated_bundle_refuses_intervals(self):
        bundle = ModelBundle(LinearRegression(), model_version="v2")
        with self.assertRaises(ValueError) as ctx:
            bundle.interval_bounds([1.0])
        self.assertIn("no interval calibration", str(ctx.exception))

    def test_legacy_record_without_the_field_still_loads(self):
        record = ModelBundle(LinearRegression(), model_version="v3").to_record()
        record.pop("interval_calibration")
        restored = ModelBundle.from_record(record)
        self.assertIsNone(restored.interval_calibration)


class TestBacktestRouteIntervals(unittest.TestCase):
    def setUp(self):
        self._original_root = settings.root
        self._tmp = tempfile.TemporaryDirectory()
        settings.root = Path(self._tmp.name)
        settings.ensure()
        rng = np.random.default_rng(11)
        self.frame = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=120, freq="D"),
                "y": np.cumsum(rng.normal(size=120)),
            }
        )
        with open_tracker("interval-route") as tracker:
            tracker.RegisterData(
                self.frame, "y", partition="train",
                temporal={"time_column": "time", "horizon": 2},
            )
        set_temporal(
            "interval-route",
            TemporalSpecRequest(time_column="time", horizon=2, freq="D"),
        )
        from SKSurrogate import make_supervised, TemporalSpec

        design = make_supervised(
            self.frame, TemporalSpec(time_column="time", horizon=2), "y", lags=[2]
        )
        model = LinearRegression().fit(design.X, design.y)
        save_bundle(
            ModelBundle(model, task_name="interval-route", model_version="v1", schema={}, dependencies={}),
            settings.task_bundles_dir("interval-route") / "v1.bundle",
        )

    def tearDown(self):
        settings.root = self._original_root
        self._tmp.cleanup()

    def test_route_stores_the_calibration_on_the_bundle(self):
        response = evaluation_router.run_backtest(
            "interval-route",
            evaluation_router.BacktestRequest(
                model_version="v1",
                horizon=2,
                n_splits=12,
                step=2,
                features={"lags": [2]},
                alpha=0.1,
            ),
        )
        deadline = time.time() + 120
        record = None
        while time.time() < deadline:
            record = job_manager.get(response["job_id"])
            if record is not None and record["status"] in {"completed", "failed"}:
                break
            time.sleep(0.2)
        self.assertIsNotNone(record)
        self.assertEqual(record["status"], "completed", record.get("error"))
        result = record["result"]
        self.assertTrue(result["intervals"]["available"])

        reloaded = load_bundle(
            settings.task_bundles_dir("interval-route") / "v1.bundle",
            strict_dependencies=False,
        )
        self.assertIsNotNone(reloaded.interval_calibration)
        self.assertEqual(reloaded.interval_calibration["alpha"], 0.1)
        lower, upper = reloaded.interval_bounds([0.0])
        self.assertLess(float(lower[0]), 0.0)
        self.assertGreater(float(upper[0]), 0.0)
        entries = [entry for entry in reloaded.metadata["evaluations"] if entry.get("intervals")]
        self.assertTrue(entries[-1]["intervals"]["available"])


if __name__ == "__main__":
    unittest.main()
