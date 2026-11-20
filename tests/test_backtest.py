"""Tests for rolling-origin backtesting — plan phase P3.

Covers the fold geometry (``rolling_origin_indices`` / ``RollingOriginSplit``), the
tracker-level ``mltrack.backtest`` (refit policies, per-origin records, panel
handling, cost ceiling) and the ``POST /api/evaluation/{task}/backtest`` job.

The property under test is the one that makes a forecast score meaningful: every
training window lies strictly before its test window, and each recorded prediction
is paired with the actual value ``horizon`` steps after that origin.
"""

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from SKSurrogate import (
    RollingOriginSplit,
    TemporalSpec,
    build_cv,
    cv_param_defs,
    cv_to_spec,
    make_supervised,
    mltrack,
    rolling_origin_indices,
)

from api.config import settings
from api.deps import open_tracker
from api.jobs import job_manager
from api.routers import evaluation as evaluation_router
from api.routers.datasets import TemporalSpecRequest, set_temporal
from SKSurrogate import ModelBundle, save_bundle

TASK = "backtest-task"
JOB_TIMEOUT_SECONDS = 120.0


def _frame(periods=80, slope=0.5, noise=0.0, seed=0):
    """Trended series; the label at t+h is a linear function of lag_h."""
    rng = np.random.default_rng(seed)
    values = slope * np.arange(periods, dtype=float)
    if noise:
        values = values + rng.normal(scale=noise, size=periods)
    return pd.DataFrame(
        {"time": pd.date_range("2026-01-01", periods=periods, freq="D"), "y": values}
    )


def _panel(periods=80):
    return pd.concat(
        [
            _frame(periods).assign(store="A"),
            _frame(periods).assign(store="B", y=lambda frame: frame["y"] + 100.0),

        ],
        ignore_index=True,
    )


class _CountingRegressor(LinearRegression):
    """LinearRegression that records every fit on the class (clone resets instances)."""

    fit_calls = 0

    def fit(self, X, y, **kwargs):
        type(self).fit_calls += 1
        return super().fit(X, y, **kwargs)


class TestFoldGeometry(unittest.TestCase):
    """Origins move forward; the test window is always the future of training."""

    def test_train_strictly_precedes_test_and_test_has_horizon_rows(self):
        folds = rolling_origin_indices(60, n_splits=4, horizon=3, step=3)
        self.assertEqual(len(folds), 4)
        previous_end = None
        for train_idx, test_idx in folds:
            self.assertEqual(len(test_idx), 3)
            self.assertTrue(np.all(np.diff(test_idx) == 1))
            self.assertEqual(train_idx[-1] + 1, test_idx[0])
            self.assertLess(train_idx[-1], test_idx[0])
            if previous_end is not None:
                self.assertEqual(train_idx[-1] - previous_end, 3)
            previous_end = train_idx[-1]
        self.assertEqual(folds[-1][1][-1], 59)  # the last fold reaches the end
        self.assertEqual(len(folds[0][0]), 48)

    def test_expanding_grows_and_sliding_keeps_a_fixed_window(self):
        expanding = rolling_origin_indices(60, n_splits=4, horizon=3, step=3)
        sliding = rolling_origin_indices(
            60, n_splits=4, horizon=3, step=3, min_train=10, expanding=False
        )
        self.assertEqual([len(train) for train, _ in expanding], [48, 51, 54, 57])
        self.assertEqual([len(train) for train, _ in sliding], [10, 10, 10, 10])
        self.assertEqual([int(train[0]) for train, _ in sliding], [38, 41, 44, 47])

    def test_insufficient_history_raises_with_the_geometry(self):
        with self.assertRaises(ValueError) as ctx:
            rolling_origin_indices(10, n_splits=5, horizon=3, step=3)
        self.assertIn("not enough observations", str(ctx.exception))
        with self.assertRaises(ValueError):
            rolling_origin_indices(60, n_splits=2, horizon=3, min_train=100, expanding=False)

    def test_splitter_protocol_and_cv_spec_round_trip(self):
        splitter = RollingOriginSplit(n_splits=3, horizon=4, step=2, min_train=20, expanding=False)
        X = np.arange(80).reshape(-1, 1)
        self.assertEqual(splitter.get_n_splits(X), 3)
        folds = list(splitter.split(X))
        self.assertEqual(len(folds), 3)
        spec = cv_to_spec(splitter)
        self.assertEqual(spec["type"], "RollingOriginSplit")
        restored = build_cv(spec)
        self.assertEqual(cv_to_spec(restored), spec)
        self.assertEqual([len(train) for train, _ in restored.split(X)], [20, 20, 20])
        names = [definition["name"] for definition in cv_param_defs("RollingOriginSplit")]
        for expected in ("n_splits", "horizon", "step", "min_train", "expanding"):
            self.assertIn(expected, names)


class TestBacktest(unittest.TestCase):
    """``mltrack.backtest`` records honest, per-origin numbers."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self._tmp.name) / "backtest.db")
        self.frame = _frame()
        _CountingRegressor.fit_calls = 0

    def tearDown(self):
        self._tmp.cleanup()

    def _tracker(self, frame=None, temporal=None):
        tracker = mltrack(TASK, db_name=self.db)
        tracker.RegisterData(
            frame if frame is not None else self.frame,
            "y",
            partition="train",
            temporal=temporal or {"time_column": "time", "horizon": 3},
        )
        return tracker

    def test_records_cover_every_origin_and_pair_truth_with_t_plus_h(self):
        tracker = self._tracker()
        try:
            result = tracker.backtest(
                LinearRegression(), n_splits=4, horizon=3, step=3
            )
        finally:
            tracker.close()

        self.assertEqual(result["config"]["n_splits"], 4)
        self.assertEqual(result["config"]["horizon"], 3)
        self.assertEqual(len(result["origins"]), 12)  # 4 origins x 3 steps
        self.assertEqual(len(result["folds"]), 4)

        frame = self.frame.set_index("time")["y"]
        for record in result["origins"]:
            origin = pd.Timestamp(record["origin_time"])
            target = pd.Timestamp(record["target_time"])
            self.assertEqual((target - origin).days, 3)
            self.assertEqual(record["target_step"], 3)
            self.assertAlmostEqual(record["y_true"], float(frame.loc[target]), places=9)

        # A linear trend is exactly a lag model: near-zero error, far better than
        # the naive "last observed value" forecast for a 3-step horizon.
        self.assertLess(result["pooled"]["mae"], 0.05)
        naive = np.mean([abs(frame.loc[pd.Timestamp(r["target_time"])] - frame.loc[pd.Timestamp(r["origin_time"])]) for r in result["origins"]])
        self.assertLess(result["pooled"]["mae"], naive)

    def test_refit_policies_control_the_number_of_fits(self):
        tracker = self._tracker()
        try:
            _CountingRegressor.fit_calls = 0
            tracker.backtest(_CountingRegressor(), n_splits=4, refit="every_fold")
            self.assertEqual(_CountingRegressor.fit_calls, 4)

            _CountingRegressor.fit_calls = 0
            first = tracker.backtest(_CountingRegressor(), n_splits=4, refit="first")
            self.assertEqual(_CountingRegressor.fit_calls, 1)
            self.assertEqual(first["fits"], 1)
            self.assertEqual([fold["refitted"] for fold in first["folds"]], [True, False, False, False])

            _CountingRegressor.fit_calls = 0
            every = tracker.backtest(_CountingRegressor(), n_splits=4, refit="every:2")
            self.assertEqual(_CountingRegressor.fit_calls, 2)
            self.assertEqual([fold["refitted"] for fold in every["folds"]], [True, False, True, False])
        finally:
            tracker.close()

    def test_cost_limit_and_unsupported_strategy_are_refused(self):
        tracker = self._tracker()
        try:
            with self.assertRaises(ValueError) as ctx:
                tracker.backtest(LinearRegression(), n_splits=20, cost_limit=5)
            self.assertIn("cost_limit", str(ctx.exception))
            with self.assertRaises(ValueError) as ctx:
                tracker.backtest(LinearRegression(), strategy="recursive")
            self.assertIn("direct", str(ctx.exception))
        finally:
            tracker.close()

    def test_task_without_temporal_spec_is_refused(self):
        tracker = mltrack(TASK, db_name=self.db)
        try:
            tracker.RegisterData(self.frame, "y", partition="train")
            with self.assertRaises(ValueError) as ctx:
                tracker.backtest(LinearRegression())
            self.assertIn("temporal spec", str(ctx.exception))
        finally:
            tracker.close()

    def test_panel_backtest_reports_per_series_and_uses_each_series_history(self):
        tracker = self._tracker(
            frame=_panel(),
            temporal={"time_column": "time", "series_id": "store", "horizon": 2},
        )
        try:
            result = tracker.backtest(
                LinearRegression(), n_splits=3, horizon=2, step=2
            )
        finally:
            tracker.close()
        self.assertEqual(set(result["per_series"]), {"A", "B"})
        for series in ("A", "B"):
            self.assertEqual(result["per_series"][series]["rows"], 6)
            self.assertLess(result["per_series"][series]["mae"], 0.05)
        series_in_records = {record["series"] for record in result["origins"]}
        self.assertEqual(series_in_records, {"A", "B"})
        self.assertEqual(result["skipped_series"], {})

    def test_result_is_json_safe_and_persisted(self):
        tracker = self._tracker()
        try:
            result = tracker.backtest(LinearRegression(), n_splits=3)
            payload = json.loads(json.dumps(result))
            self.assertIn("pooled", payload)
            self.assertIn(result["artifact_key"], tracker.GetMetadata())
            stored = tracker.GetMetadata()[result["artifact_key"]]
            self.assertEqual(stored["folds"], result["folds"])
        finally:
            tracker.close()

    def test_feature_configuration_changes_the_design(self):
        tracker = self._tracker()
        try:
            default = tracker.backtest(LinearRegression(), n_splits=3)
            richer = tracker.backtest(
                LinearRegression(),
                n_splits=3,
                feature_kwargs={"lags": [3, 7], "rolling": [{"window": 5, "stats": ["mean"]}]},
            )
        finally:
            tracker.close()
        default_features = [feature["name"] for feature in default["design"]["features"]]
        richer_features = [feature["name"] for feature in richer["design"]["features"]]
        self.assertIn("y__lag_3", default_features)
        self.assertIn("y__lag_7", richer_features)
        self.assertIn("y__roll_mean_5", richer_features)
        self.assertNotEqual(default["artifact_key"], richer["artifact_key"])


class TestBacktestEndpoint(unittest.TestCase):
    """The route queues a job whose result carries the backtest summary."""

    def setUp(self):
        self._original_root = settings.root
        self._tmp = tempfile.TemporaryDirectory()
        settings.root = Path(self._tmp.name)
        settings.ensure()
        self.frame = _frame()
        with open_tracker(TASK) as tracker:
            tracker.RegisterData(
                self.frame, "y", partition="train",
                temporal={"time_column": "time", "horizon": 3},
            )
        set_temporal(
            TASK,
            TemporalSpecRequest(time_column="time", horizon=3, freq="D"),
        )
        from sklearn.linear_model import LinearRegression as _LR

        design = make_supervised(
            self.frame, TemporalSpec(time_column="time", horizon=3), "y", lags=[3]
        )
        model = _LR().fit(design.X, design.y)
        bundle = ModelBundle(
            model, task_name=TASK, model_version="v1", schema={}, dependencies={}
        )
        save_bundle(bundle, settings.task_bundles_dir(TASK) / "v1.bundle")

    def tearDown(self):
        settings.root = self._original_root
        self._tmp.cleanup()

    def _wait(self, job_id, timeout=JOB_TIMEOUT_SECONDS):
        deadline = time.time() + timeout
        while time.time() < deadline:
            record = job_manager.get(job_id)
            if record is not None and record["status"] in {"completed", "failed"}:
                return record
            time.sleep(0.2)
        raise AssertionError("backtest job did not finish within %.0fs" % timeout)

    def test_route_queues_a_job_and_records_the_backtest(self):
        response = evaluation_router.run_backtest(
            TASK,
            evaluation_router.BacktestRequest(
                model_version="v1",
                horizon=3,
                n_splits=4,
                step=3,
                features={"lags": [3]},
            ),
        )
        self.assertEqual(response["status"], "queued")
        record = self._wait(response["job_id"])
        self.assertEqual(record["status"], "completed", record.get("error"))
        result = record["result"]
        self.assertEqual(result["folds"][0]["test_rows"], 3)
        self.assertEqual(len(result["origins"]), 12)
        self.assertLess(result["pooled"]["mae"], 0.05)
        self.assertEqual(result["model_version"], "v1")

        from SKSurrogate import load_bundle

        reloaded = load_bundle(
            settings.task_bundles_dir(TASK) / "v1.bundle", strict_dependencies=False
        )
        types = [entry["type"] for entry in reloaded.metadata.get("evaluations", [])]
        self.assertIn("rolling_origin_backtest", types)

    def test_route_requires_exactly_one_bundle_reference(self):
        with self.assertRaises(HTTPException) as ctx:
            evaluation_router.run_backtest(
                TASK, evaluation_router.BacktestRequest()
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_route_reports_a_task_without_temporal_spec(self):
        with open_tracker("no-temporal") as tracker:
            tracker.RegisterData(self.frame, "y", partition="train")
        bundle = ModelBundle(
            LinearRegression().fit(self.frame[["time"]].astype("int64"), self.frame["y"]),
            task_name="no-temporal",
            model_version="v1",
            schema={},
            dependencies={},
        )
        save_bundle(bundle, settings.task_bundles_dir("no-temporal") / "v1.bundle")
        response = evaluation_router.run_backtest(
            "no-temporal", evaluation_router.BacktestRequest(model_version="v1")
        )
        record = self._wait(response["job_id"])
        self.assertEqual(record["status"], "failed")
        self.assertIn("temporal spec", record["error"])


if __name__ == "__main__":
    unittest.main()
