"""Tests for forecast metrics and naive baselines — plan phase P4.

Every metric is checked against a hand-computed value: forecast metrics are easy to
implement plausibly and wrongly, so the fixtures are closed-form, not "looks about
right". The integration tests then pin the two claims that matter for a leaderboard:
baselines are scored on the same origins as the model, and MASE == 1 is parity with
seasonal naive rather than a win.
"""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from SKSurrogate import (
    BASELINE_METHODS,
    baseline_prediction,
    bias,
    forecast_ranking,
    interval_coverage,
    interval_width,
    mae,
    mase,
    mase_denominator,
    mltrack,
    pinball_loss,
    rmse,
    rmse_scaled,
    smape,
    summarize,
    winkler_score,
)


class TestPointMetrics(unittest.TestCase):
    def setUp(self):
        self.actual = [1.0, 2.0, 3.0, 4.0]
        self.predicted = [1.0, 2.0, 4.0, 3.0]

    def test_mae_rmse_bias(self):
        self.assertAlmostEqual(mae(self.actual, self.predicted), 0.5)
        self.assertAlmostEqual(rmse(self.actual, self.predicted), math.sqrt(0.5))
        self.assertAlmostEqual(bias(self.actual, self.predicted), 0.0)
        self.assertAlmostEqual(bias([1.0], [3.0]), 2.0)  # over-forecast

    def test_smape_matches_the_hand_computed_percentage(self):
        # Terms: 0, 0, 2*1/(3+4), 2*1/(4+3) -> mean 2/7/2 = 1/7 -> 14.2857 %
        expected = 100.0 * (0.0 + 0.0 + 2.0 / 7.0 + 2.0 / 7.0) / 4.0
        self.assertAlmostEqual(smape(self.actual, self.predicted), expected)
        self.assertAlmostEqual(smape(self.actual, self.predicted, as_percent=False), expected / 100.0)

    def test_smape_is_zero_safe(self):
        self.assertEqual(smape([0.0, 0.0], [0.0, 0.0]), 0.0)
        self.assertAlmostEqual(smape([0.0, 1.0], [0.0, 1.0]), 0.0)
        value = smape([0.0, 2.0], [1.0, 2.0])
        self.assertTrue(np.isfinite(value))

    def test_mase_scales_by_the_in_sample_difference(self):
        insample = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(mase_denominator(insample, seasonality=1), 1.0)
        self.assertEqual(mase([5.0], [7.0], insample, seasonality=1), 2.0)
        # A 2-step denominator on a straight line of slope 1 is 2.
        self.assertEqual(mase_denominator(insample, seasonality=2), 2.0)
        self.assertEqual(mase([5.0], [7.0], insample, seasonality=2), 1.0)

    def test_mase_is_one_for_a_naive_forecast_on_a_random_walk(self):
        insample = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        prediction = baseline_prediction(insample, "naive", horizon=1)
        self.assertEqual(prediction, 1.0)
        self.assertEqual(mase([0.0], [prediction], insample, seasonality=1), 1.0)

    def test_mase_is_undefined_for_a_constant_series(self):
        self.assertTrue(np.isnan(mase_denominator([5.0, 5.0, 5.0], seasonality=1)))
        self.assertTrue(math.isnan(mase([5.0], [5.1], [5.0, 5.0, 5.0], seasonality=1)))

    def test_mase_rejects_too_short_history(self):
        with self.assertRaises(ValueError):
            mase_denominator([1.0, 2.0], seasonality=2)
        with self.assertRaises(ValueError):
            mase_denominator([1.0, 2.0, 3.0], seasonality=0)

    def test_rmse_scaled_divides_by_the_same_scale(self):
        insample = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(
            rmse_scaled([5.0], [7.0], insample, seasonality=1), 2.0
        )
        self.assertAlmostEqual(
            rmse_scaled([5.0], [7.0], insample, seasonality=2), 1.0
        )

    def test_mismatched_shapes_are_refused(self):
        with self.assertRaises(ValueError):
            mae([1.0, 2.0], [1.0])
        with self.assertRaises(ValueError):
            mae([], [])


class TestProbabilisticMetrics(unittest.TestCase):
    def test_pinball_median_equals_half_the_mae(self):
        actual = [1.0, 2.0, 3.0]
        predicted = [1.0, 2.0, 4.0]
        self.assertAlmostEqual(
            pinball_loss(actual, predicted, 0.5), 0.5 * mae(actual, predicted)
        )

    def test_pinball_matches_its_definition(self):
        # y=1, f=2 at q=0.9: y < f so the loss is (1 - q) * (f - y) = 0.1
        self.assertAlmostEqual(pinball_loss([1.0], [2.0], 0.9), 0.1)
        # y=3, f=1 at q=0.9: y > f so the loss is q * (y - f) = 1.8
        self.assertAlmostEqual(pinball_loss([3.0], [1.0], 0.9), 1.8)

    def test_pinball_rejects_invalid_quantiles(self):
        for quantile in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                pinball_loss([1.0], [1.0], quantile)

    def test_winkler_score_penalises_misses_by_two_over_alpha(self):
        # Inside: score is just the width.
        self.assertAlmostEqual(winkler_score([0.0], [-1.0], [1.0], alpha=0.2), 2.0)
        # One unit above the upper bound: width + (2 / 0.2) * 1 = 12.
        self.assertAlmostEqual(winkler_score([2.0], [-1.0], [1.0], alpha=0.2), 12.0)
        # One unit below: same penalty for symmetry.
        self.assertAlmostEqual(winkler_score([-2.0], [-1.0], [1.0], alpha=0.2), 12.0)

    def test_coverage_and_width(self):
        actual = [0.0, 2.0, 1.0]
        lower = [-1.0, -1.0, 0.0]
        upper = [1.0, 1.0, 0.5]
        self.assertAlmostEqual(interval_coverage(actual, lower, upper), 1.0 / 3.0)
        self.assertAlmostEqual(interval_width(lower, upper), (2.0 + 2.0 + 0.5) / 3.0)

    def test_winkler_rejects_invalid_alpha(self):
        for alpha in (0.0, 1.0, -0.2):
            with self.assertRaises(ValueError):
                winkler_score([0.0], [-1.0], [1.0], alpha=alpha)


class TestSummarize(unittest.TestCase):
    def test_pooled_block_and_optional_blocks(self):
        block = summarize(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 4.0, 3.0],
            y_insample_scale=[1.0, 1.0, 1.0, 1.0],
            seasonality=1,
            quantile=0.5,
            lower=[0.5, 1.5, 2.5, 3.5],
            upper=[1.5, 2.5, 4.5, 4.5],
            alpha=0.2,
        )
        metrics = block["metrics"]
        self.assertAlmostEqual(metrics["mae"], 0.5)
        self.assertAlmostEqual(metrics["mase"], 0.5)
        self.assertAlmostEqual(metrics["pinball_loss"], 0.25)
        self.assertAlmostEqual(metrics["coverage"], 1.0)
        self.assertIn("winkler_score", metrics)
        self.assertEqual(block["primary"], metrics)

    def test_per_series_blocks_and_macro_aggregation(self):
        block = summarize(
            [10.0, 11.0, 1.0, 3.0],
            [10.0, 12.0, 1.0, 1.0],
            series=["A", "A", "B", "B"],
            horizon=[1, 1, 1, 1],
            aggregation="macro",
        )
        self.assertEqual(block["per_series"]["A"]["mae"], 0.5)
        self.assertEqual(block["per_series"]["B"]["mae"], 1.0)
        self.assertAlmostEqual(block["primary"]["mae"], 0.75)  # unweighted mean
        self.assertEqual(block["per_horizon"]["1"]["rows"], 4)
        # Pooled would weight A and B equally here, which is the documented default.
        self.assertAlmostEqual(block["metrics"]["mae"], 0.75)

    def test_mase_is_omitted_without_a_scale(self):
        block = summarize([1.0, 2.0], [1.0, 3.0])
        self.assertNotIn("mase", block["metrics"])

    def test_scale_length_must_match(self):
        with self.assertRaises(ValueError):
            summarize([1.0, 2.0], [1.0, 2.0], y_insample_scale=[1.0])


class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.history = np.arange(1.0, 15.0)  # last observation is 14

    def test_naive_returns_the_last_observation(self):
        for horizon in (1, 3, 12):
            self.assertEqual(baseline_prediction(self.history, "naive", horizon), 14.0)

    def test_seasonal_naive_repeats_the_phase_one_season_back(self):
        # m = 7: h=1 -> 8 (one season before the first forecast step),
        # h=7 -> 14 (the last observation), h=8 -> 8 again.
        self.assertEqual(baseline_prediction(self.history, "seasonal_naive", 1, 7), 8.0)
        self.assertEqual(baseline_prediction(self.history, "seasonal_naive", 7, 7), 14.0)
        self.assertEqual(baseline_prediction(self.history, "seasonal_naive", 8, 7), 8.0)
        self.assertEqual(baseline_prediction(self.history, "seasonal_naive", 2, 1), 14.0)

    def test_drift_extrapolates_the_average_change(self):
        self.assertEqual(baseline_prediction(self.history, "drift", 1), 15.0)
        self.assertEqual(baseline_prediction(self.history, "drift", 2), 16.0)

    def test_mean_is_the_historical_average(self):
        self.assertEqual(baseline_prediction(self.history, "mean", 5), float(np.mean(self.history)))

    def test_unknown_method_and_empty_history_are_refused(self):
        with self.assertRaises(ValueError):
            baseline_prediction(self.history, "ets", 1)
        with self.assertRaises(ValueError):
            baseline_prediction([], "naive", 1)
        with self.assertRaises(ValueError):
            baseline_prediction([1.0, 2.0], "seasonal_naive", 1, 7)


class TestRanking(unittest.TestCase):
    def test_a_model_worse_than_the_benchmark_is_flagged(self):
        ranking = forecast_ranking(
            [
                {"label": "model", "metrics": {"mase": 1.4}},
                {"label": "baseline:seasonal_naive", "metrics": {"mase": 1.0}},
                {"label": "baseline:naive", "metrics": {"mase": 1.8}},
            ]
        )
        self.assertEqual(ranking["best"]["label"], "baseline:seasonal_naive")
        self.assertFalse(ranking["winner_beats_baseline"])
        model = next(item for item in ranking["ranked"] if item["label"] == "model")
        self.assertTrue(model["worse_than_baseline"])
        self.assertFalse(model["beats_baseline"])

    def test_exact_parity_is_not_a_win_and_not_a_loss(self):
        ranking = forecast_ranking([{"label": "model", "metrics": {"mase": 1.0}}])
        entry = ranking["ranked"][0]
        self.assertFalse(entry["beats_baseline"])
        self.assertFalse(entry["worse_than_baseline"])
        self.assertFalse(ranking["winner_beats_baseline"])

    def test_missing_metrics_are_unranked_not_last(self):
        ranking = forecast_ranking(
            [
                {"label": "model", "metrics": {"mase": 0.8}},
                {"label": "broken", "metrics": {"mase": float("nan")}},
                {"label": "absent", "metrics": {}},
            ]
        )
        self.assertEqual(ranking["best"]["label"], "model")
        self.assertTrue(ranking["winner_beats_baseline"])
        self.assertEqual({item["label"] for item in ranking["unranked"]}, {"broken", "absent"})


class TestBacktestIntegration(unittest.TestCase):
    """Baselines and MASE are computed on the same origins as the model."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self._tmp.name) / "metrics.db")
        rng = np.random.default_rng(3)
        periods = 120
        values = 0.4 * np.arange(periods) + rng.normal(scale=0.3, size=periods)
        self.frame = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=periods, freq="D"),
                "y": values,
            }
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _tracker(self, temporal=None):
        tracker = mltrack("metrics-task", db_name=self.db)
        tracker.RegisterData(
            self.frame, "y", partition="train",
            temporal=temporal or {"time_column": "time", "horizon": 5},
        )
        return tracker

    def test_backtest_reports_every_baseline_with_mase_and_smape(self):
        tracker = self._tracker()
        try:
            result = tracker.backtest(
                LinearRegression(), n_splits=4, horizon=5, step=5, seasonality=5
            )
        finally:
            tracker.close()
        self.assertEqual(set(result["baselines"]), set(BASELINE_METHODS))
        for method in BASELINE_METHODS:
            block = result["baselines"][method]
            self.assertEqual(block["scored_origins"], 20)  # 4 origins x 5 steps
            self.assertIn("mase", block)
            self.assertIn("smape", block)
        self.assertIn("mase", result["pooled"])
        self.assertIn("smape", result["pooled"])
        self.assertEqual(result["config"]["seasonality"], 5)
        # Each origin carries the in-sample scale and its baseline predictions.
        record = result["origins"][0]
        self.assertIn("mase_scale", record)
        self.assertEqual(set(record["baselines"]), set(BASELINE_METHODS))

    def test_a_deliberately_bad_model_loses_to_the_baseline(self):
        class LastValue(LinearRegression):
            """Predicts a constant: the mean label of the training window."""

            def predict(self, X):
                return np.full(X.shape[0], self._constant)

            def fit(self, X, y):
                self._constant = float(np.mean(y))
                return self

        tracker = self._tracker(temporal={"time_column": "time", "horizon": 5})
        try:
            result = tracker.backtest(
                LastValue(), n_splits=4, horizon=5, step=5, seasonality=1,
                baselines=("naive",),
            )
        finally:
            tracker.close()
        self.assertGreater(result["pooled"]["mase"], 1.0)
        model_entry = next(
            item for item in result["ranking"]["ranked"] if item["label"] == "model"
        )
        self.assertTrue(model_entry["worse_than_baseline"])
        self.assertFalse(result["ranking"]["winner_beats_baseline"])

    def test_baseline_selection_is_configurable_and_validated(self):
        tracker = self._tracker()
        try:
            result = tracker.backtest(
                LinearRegression(), n_splits=3, baselines=("naive", "drift")
            )
            self.assertEqual(set(result["baselines"]), {"naive", "drift"})
            with self.assertRaises(ValueError):
                tracker.backtest(LinearRegression(), n_splits=3, baselines=("ets",))
            with self.assertRaises(ValueError):
                tracker.backtest(LinearRegression(), n_splits=3, seasonality=0)
        finally:
            tracker.close()

    def test_result_is_json_safe_with_baselines(self):
        import json

        tracker = self._tracker()
        try:
            result = tracker.backtest(LinearRegression(), n_splits=3, baselines=("naive",))
        finally:
            tracker.close()
        payload = json.loads(json.dumps(result))
        self.assertIn("baselines", payload)
        self.assertIn("ranking", payload)


if __name__ == "__main__":
    unittest.main()
