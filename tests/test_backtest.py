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
    ForecastFeatureBuilder,
    ModelBundle,
    RollingOriginSplit,
    TemporalServeState,
    TemporalSpec,
    build_cv,
    cv_param_defs,
    cv_to_spec,
    make_supervised,
    mltrack,
    rolling_origin_indices,
    save_bundle,
    serve_forecast,
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

    def test_cost_limit_and_unknown_strategy_are_refused(self):
        tracker = self._tracker()
        try:
            with self.assertRaises(ValueError) as ctx:
                tracker.backtest(LinearRegression(), n_splits=20, cost_limit=5)
            self.assertIn("cost_limit", str(ctx.exception))
            with self.assertRaises(ValueError) as ctx:
                tracker.backtest(LinearRegression(), strategy="two-step")
            self.assertIn("recursive", str(ctx.exception))
        finally:
            tracker.close()

    def test_nested_cv_warns_when_timeseriessplit_gap_is_shorter_than_the_horizon(self):
        import warnings

        from sklearn.model_selection import TimeSeriesSplit

        tracker = self._tracker(temporal={"time_column": "time", "horizon": 7})
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                tracker.nested_cv_evaluation(
                    LinearRegression(),
                    inner_cv=TimeSeriesSplit(n_splits=3, gap=1),
                    outer_cv=TimeSeriesSplit(n_splits=4, gap=7),
                    train_partition="train",
                )
        finally:
            tracker.close()
        messages = [str(item.message) for item in caught]
        self.assertTrue(
            any("forecast horizon is 7" in message for message in messages), messages
        )
        # The outer splitter already satisfies the horizon rule, so only inner_cv warns.
        self.assertEqual(sum("inner_cv" in message for message in messages), 1)
        self.assertEqual(sum("outer_cv" in message for message in messages), 0)

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


class TestRecursiveBacktest(unittest.TestCase):
    """``strategy="recursive"`` replays the horizon through the serving walk.

    Plan P3 deferred recursive evaluation until P6 supplied the serving state;
    both now share :func:`SKSurrogate.temporal.recursive_steps`, so an offline
    fold and a served forecast for the same origin cannot disagree.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self._tmp.name) / "recursive.db")
        # Exact linear series: a one-step model on lag h reproduces it bit-exact,
        # so any deviation is a bookkeeping error, not model noise.
        t = np.arange(1, 81, dtype=float)
        self.frame = pd.DataFrame({"time": t, "y": 2.0 * t})
        self.spec = TemporalSpec(time_column="time", horizon=3)

    def tearDown(self):
        self._tmp.cleanup()

    def _tracker(self):
        tracker = mltrack(TASK, db_name=self.db)
        tracker.RegisterData(
            self.frame,
            "y",
            partition="train",
            temporal={"time_column": "time", "horizon": 3},
        )
        return tracker

    def test_one_origin_per_fold_with_every_step_recorded(self):
        tracker = self._tracker()
        try:
            result = tracker.backtest(
                LinearRegression(),
                horizon=3,
                n_splits=4,
                strategy="recursive",
                feature_kwargs={"lags": [3]},
            )
        finally:
            tracker.close()

        self.assertEqual(result["config"]["strategy"], "recursive")
        self.assertEqual(sorted(result["per_horizon"]), ["1", "2", "3"])
        self.assertEqual(len(result["origins"]), 12)  # 4 origins x 3 steps
        self.assertEqual(len({record["origin_time"] for record in result["origins"]}), 4)
        self.assertEqual(
            [record["target_step"] for record in result["origins"][:3]], [1, 2, 3]
        )
        # Truth and prediction are paired with the right future timestamps.
        for record in result["origins"]:
            self.assertAlmostEqual(record["y_true"], record["y_pred"], places=9)
            self.assertEqual(float(record["target_time"]) - float(record["origin_time"]),
                             float(record["target_step"]))
        self.assertLess(result["pooled"]["mae"], 1e-9)

    def test_timings_and_refit_policy_are_recorded(self):
        tracker = self._tracker()
        try:
            result = tracker.backtest(
                LinearRegression(),
                horizon=3,
                n_splits=3,
                refit="first",
                strategy="recursive",
                feature_kwargs={"lags": [3]},
            )
        finally:
            tracker.close()

        self.assertEqual(result["fits"], 1)
        folds = result["folds"]
        self.assertGreater(folds[0]["fit_time"], 0.0)
        self.assertEqual(folds[1]["fit_time"], 0.0)
        for fold in folds:
            self.assertIn("predict_time", fold)
            self.assertGreaterEqual(fold["predict_time"], 0.0)
        for record in result["origins"]:
            self.assertIn("fit_time", record)
            self.assertIn("predict_time", record)

    def test_served_forecast_matches_the_fold_prediction_for_the_same_origin(self):
        tracker = self._tracker()
        try:
            result = tracker.backtest(
                LinearRegression(),
                horizon=3,
                n_splits=1,
                strategy="recursive",
                feature_kwargs={"lags": [3]},
            )
            frame = tracker.get_dataframe("train")
        finally:
            tracker.close()

        records = [record for record in result["origins"] if record["fold"] == 0]
        self.assertEqual(len(records), 3)
        origin_time = float(records[0]["origin_time"])

        # Same training set the fold used: every origin strictly before the cutoff.
        design = make_supervised(
            frame, self.spec, "y", strategy="recursive", lags=[3]
        )
        train_mask = design.meta["origin_time"].to_numpy(dtype=float) < origin_time
        model = LinearRegression().fit(design.X[train_mask], design.y[train_mask])
        builder = ForecastFeatureBuilder(spec=self.spec, target="y", lags=[3])
        builder.fit(frame)
        from sklearn.pipeline import Pipeline

        bundle = ModelBundle(
            Pipeline([("features", builder), ("model", model)]),
            task_name=TASK,
            temporal=TemporalServeState.from_parts(
                self.spec, "y", builder, strategy="recursive"
            ).to_dict(),
        )
        history = frame[frame["time"] <= origin_time]
        served = serve_forecast(bundle, history, horizon=3)
        self.assertTrue(
            np.allclose(
                np.asarray(served.means, dtype=float),
                [record["y_pred"] for record in records],
                rtol=0.0,
                atol=1e-9,
            )
        )


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

    def test_route_unwraps_the_temporal_feature_builder(self):
        """A P6 temporal bundle carries the builder inside the pipeline.

        The backtest builds its own design from ``features``, so the route must
        refit the bare estimator; cloning the whole pipeline fails on the
        already-materialized design with "missing column(s): time, y".
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline

        from SKSurrogate import ForecastFeatureBuilder, TemporalServeState

        spec = TemporalSpec(time_column="time", horizon=3)
        builder = ForecastFeatureBuilder(spec=spec, target="y", lags=[3])
        builder.fit(self.frame)
        design = builder.build(self.frame)
        model = LinearRegression().fit(design.X, design.y)
        temporal_bundle = ModelBundle(
            Pipeline([("features", builder), ("model", model)]),
            task_name=TASK,
            model_version="v2",
            schema={},
            dependencies={},
            temporal=TemporalServeState.from_parts(
                spec, "y", builder, strategy="direct"
            ).to_dict(),
        )
        save_bundle(temporal_bundle, settings.task_bundles_dir(TASK) / "v2.bundle")

        response = evaluation_router.run_backtest(
            TASK,
            evaluation_router.BacktestRequest(
                model_version="v2",
                horizon=3,
                n_splits=3,
                features={"lags": [3]},
            ),
        )
        record = self._wait(response["job_id"])
        self.assertEqual(record["status"], "completed", record.get("error"))
        self.assertLess(record["result"]["pooled"]["mae"], 0.05)

    def test_route_queues_a_job_and_records_the_backtest(self):
        response = evaluation_router.run_backtest(
            TASK,
            evaluation_router.BacktestRequest(
                model_version="v1",
                horizon=3,
                n_splits=4,
                step=3,
                features={"lags": [3]},
                seasonality=3,
                baselines=["naive", "seasonal_naive"],
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
        # P4: baselines scored on the same origins, and a MASE ranking guard.
        self.assertIn("naive", result["baselines"])
        self.assertEqual(set(result["baselines"]), {"naive", "seasonal_naive"})
        self.assertIn("ranking", result)
        self.assertEqual(result["config"]["seasonality"], 3)

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
