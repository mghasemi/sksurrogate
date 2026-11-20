"""Tests for the temporal (time-series) data contract — plan phase P1.

Covers ``SKSurrogate.temporal`` (spec validation, frequency inference, frame
validation), its ``mltrack`` integration (registration, metadata persistence,
feature exclusion, splitter enforcement) and the control-plane surfaces
(``POST /{task}/register`` with a ``temporal`` field, ``GET/PUT/DELETE
/{task}/temporal``).

The contract's whole point is that nothing is inferred silently: declaring a
``TemporalSpec`` either validates the frame or fails with a report naming the
offending rows.
"""

import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile
from sklearn.model_selection import TimeSeriesSplit

from SKSurrogate import (
    TEMPORAL_CV_SPLITTERS,
    TemporalSpec,
    TemporalValidationError,
    cv_is_temporal_safe,
    infer_freq,
    mltrack,
    temporal_columns,
    validate_temporal_frame,
)

from api.config import settings
from api.deps import open_tracker
from api.routers import datasets as datasets_router

TASK = "temporal-contract"


def _daily_frame(periods=40, series=("A", "B")):
    """A regular daily panel with a deterministic target."""
    time = pd.date_range("2026-01-01", periods=periods, freq="D")
    frames = []
    for offset, name in enumerate(series):
        frames.append(
            pd.DataFrame(
                {
                    "time": time,
                    "series": name,
                    "lag_1": np.roll(np.arange(periods, dtype=float), 1) + offset,
                    "target": np.arange(periods, dtype=float) + 10.0 * offset,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _upload(frame):
    return UploadFile(file=io.BytesIO(frame.to_csv(index=False).encode()), filename="data.csv")


class TestTemporalSpec(unittest.TestCase):
    """Spec validation and serialization."""

    def test_round_trip_preserves_every_field(self):
        spec = TemporalSpec(
            time_column="ts",
            series_id="store",
            freq="h",
            horizon=12,
            order="decreasing",
            gap_tolerance=2,
        )
        restored = TemporalSpec.from_dict(spec.to_dict())
        self.assertEqual(restored, spec)
        self.assertEqual(restored.temporal_columns, ["ts", "store"])

    def test_blank_optional_fields_become_none(self):
        spec = TemporalSpec.from_dict(
            {"time_column": "ts", "series_id": "", "freq": "", "horizon": 4}
        )
        self.assertIsNone(spec.series_id)
        self.assertIsNone(spec.freq)
        self.assertEqual(spec.horizon, 4)

    def test_rejects_invalid_values(self):
        cases = [
            ({"time_column": ""}, "time_column"),
            ({"time_column": "t", "horizon": 0}, "horizon"),
            ({"time_column": "t", "horizon": True}, "horizon"),
            ({"time_column": "t", "order": "random"}, "order"),
            ({"time_column": "t", "gap_tolerance": -1}, "gap_tolerance"),
            ({"time_column": "t", "freq": "not-a-freq"}, "freq"),
            ({"time_column": "t", "series_id": "t"}, "different"),
            ({"time_column": "t", "typo_column": "x"}, "unknown"),
            ({}, "time_column"),
        ]
        for payload, needle in cases:
            with self.assertRaises(ValueError, msg=payload) as ctx:
                TemporalSpec.from_dict(payload)
            self.assertIn(needle, str(ctx.exception).lower(), msg=payload)

    def test_temporal_columns_helper_accepts_none(self):
        self.assertEqual(temporal_columns(None), [])
        self.assertEqual(temporal_columns({"time_column": "t"}), ["t"])


class TestTemporalValidation(unittest.TestCase):
    """Frame validation: the failure modes must name the offending rows."""

    def test_regular_daily_frame_reports_inferred_frequency(self):
        report = validate_temporal_frame(
            _daily_frame(), TemporalSpec(time_column="time", series_id="series", freq="D", horizon=2)
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["time_kind"], "datetime")
        self.assertEqual(report["series_count"], 2)
        self.assertEqual(report["inferred_freq"], "D")
        self.assertEqual(report["missing_periods"], 0)
        self.assertEqual(report["series_lengths"], {"A": 40, "B": 40})
        self.assertEqual(report["time_range"]["start"], "2026-01-01 00:00:00")
        self.assertEqual(report["errors"], [])

    def test_unsorted_rows_raise_and_name_the_first_inversion(self):
        frame = _daily_frame(series=("A",))
        shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
        with self.assertRaises(TemporalValidationError) as ctx:
            validate_temporal_frame(shuffled, TemporalSpec(time_column="time", freq="D"))
        message = str(ctx.exception)
        self.assertIn("is not increasing", message)
        self.assertIn("2026-", message)
        self.assertIn("sort the partition", message)
        self.assertTrue(ctx.exception.report["errors"])

    def test_decreasing_order_is_accepted_when_declared(self):
        frame = _daily_frame(series=("A",)).iloc[::-1].reset_index(drop=True)
        report = validate_temporal_frame(
            frame, TemporalSpec(time_column="time", freq="D", order="decreasing")
        )
        self.assertTrue(report["ok"])
        with self.assertRaises(TemporalValidationError):
            validate_temporal_frame(frame, TemporalSpec(time_column="time", freq="D"))

    def test_duplicate_timestamps_are_rejected(self):
        frame = _daily_frame(series=("A",))
        frame = pd.concat([frame, frame.iloc[[5]]], ignore_index=True)
        with self.assertRaises(TemporalValidationError) as ctx:
            validate_temporal_frame(frame, TemporalSpec(time_column="time", freq="D"))
        self.assertTrue(
            any("duplicate" in message for message in ctx.exception.report["errors"]),
            ctx.exception.report["errors"],
        )

    def test_unparseable_time_is_rejected(self):
        frame = _daily_frame(series=("A",)).astype({"time": object})
        frame.loc[3, "time"] = "not-a-timestamp"
        with self.assertRaises(TemporalValidationError) as ctx:
            validate_temporal_frame(frame, TemporalSpec(time_column="time", freq="D"))
        self.assertIn("unparseable", str(ctx.exception))
        self.assertIn("3", str(ctx.exception.report["errors"]) + str(ctx.exception))

    def test_missing_time_is_rejected(self):
        frame = _daily_frame(series=("A",))
        frame.loc[2, "time"] = None
        with self.assertRaises(TemporalValidationError):
            validate_temporal_frame(frame, TemporalSpec(time_column="time", freq="D"))

    def test_missing_series_id_is_rejected(self):
        frame = _daily_frame()
        frame.loc[1, "series"] = None
        with self.assertRaises(TemporalValidationError) as ctx:
            validate_temporal_frame(
                frame, TemporalSpec(time_column="time", series_id="series", freq="D")
            )
        self.assertIn("missing value", str(ctx.exception))

    def test_declared_frequency_gap_is_rejected(self):
        frame = _daily_frame(series=("A",)).drop(index=10).reset_index(drop=True)
        with self.assertRaises(TemporalValidationError) as ctx:
            validate_temporal_frame(frame, TemporalSpec(time_column="time", freq="D"))
        self.assertIn("missing period", str(ctx.exception))
        report = ctx.exception.report
        self.assertGreaterEqual(report["missing_periods"], 1)

    def test_gap_tolerance_permits_declared_gaps(self):
        frame = _daily_frame(series=("A",)).drop(index=10).reset_index(drop=True)
        report = validate_temporal_frame(
            frame, TemporalSpec(time_column="time", freq="D", gap_tolerance=1)
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["missing_periods"], 1)

    def test_undeclared_gaps_are_a_warning_not_an_error(self):
        frame = _daily_frame(series=("A",)).drop(index=10).reset_index(drop=True)
        report = validate_temporal_frame(frame, TemporalSpec(time_column="time", horizon=2))
        self.assertTrue(report["ok"])
        self.assertTrue(report["warnings"])

    def test_numeric_time_axis_is_not_read_as_timestamps(self):
        frame = pd.DataFrame({"step": np.arange(0, 60, 2), "target": np.arange(30.0)})
        report = validate_temporal_frame(frame, TemporalSpec(time_column="step", horizon=2))
        self.assertEqual(report["time_kind"], "numeric")
        self.assertEqual(report["inferred_freq"], "step=2")
        self.assertEqual(report["missing_periods"], 0)

    def test_missing_column_is_reported(self):
        with self.assertRaises(TemporalValidationError) as ctx:
            validate_temporal_frame(
                _daily_frame(), TemporalSpec(time_column="absent", freq="D")
            )
        self.assertIn("not found", str(ctx.exception))

    def test_empty_frame_is_rejected(self):
        with self.assertRaises(TemporalValidationError):
            validate_temporal_frame(
                pd.DataFrame({"time": [], "target": []}), TemporalSpec(time_column="time")
            )

    def test_infer_freq_handles_regular_irregular_and_short_input(self):
        self.assertEqual(infer_freq(pd.date_range("2026-01-01", periods=20, freq="h")), "h")
        self.assertEqual(infer_freq(pd.date_range("2026-01-01", periods=12, freq="MS")), "MS")
        self.assertIsNone(infer_freq([pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03")]))
        self.assertIsNone(infer_freq(["a", "b", "c"]))

    def test_splitter_affinity(self):
        self.assertTrue(cv_is_temporal_safe(TimeSeriesSplit(n_splits=3)))
        self.assertTrue(cv_is_temporal_safe({"type": "TimeSeriesSplit"}))
        self.assertFalse(cv_is_temporal_safe({"type": "KFold", "n_splits": 3}))
        self.assertFalse(cv_is_temporal_safe(5))
        self.assertFalse(cv_is_temporal_safe(None))
        self.assertIn("TimeSeriesSplit", TEMPORAL_CV_SPLITTERS)


class TestTrackerIntegration(unittest.TestCase):
    """``mltrack`` registration, persistence and splitter enforcement."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self._tmp.name) / "temporal.db")
        self.frame = _daily_frame()

    def tearDown(self):
        self._tmp.cleanup()

    def _tracker(self):
        return mltrack(TASK, db_name=self.db)

    def test_registration_stores_spec_and_excludes_temporal_columns(self):
        tracker = self._tracker()
        spec = TemporalSpec(time_column="time", series_id="series", freq="D", horizon=3)
        try:
            tracker.RegisterData(self.frame, "target", partition="train", temporal=spec)
            self.assertEqual(tracker.GetTemporalSpec(), spec.to_dict())
            report = tracker.dataset_splits()["train"]["temporal"]
            self.assertTrue(report["ok"])
            self.assertEqual(report["series_count"], 2)
            # time + series leave the feature matrix; lag_1 stays.
            self.assertEqual(tracker.X.shape[1], 1)
            stored = tracker.get_dataframe(partition="train")
            self.assertIn("time", stored.columns)
            self.assertIn("series", stored.columns)
        finally:
            tracker.close()

    def test_spec_survives_reopening_the_tracker(self):
        tracker = self._tracker()
        try:
            tracker.RegisterData(
                self.frame, "target", partition="train",
                temporal=TemporalSpec(time_column="time", series_id="series", freq="D"),
            )
        finally:
            tracker.close()
        reopened = self._tracker()
        try:
            stored = reopened.GetTemporalSpec()
            self.assertEqual(stored["time_column"], "time")
            self.assertEqual(stored["series_id"], "series")
            X, _ = reopened.get_data(partition="train")
            self.assertEqual(X.shape[1], 1)
        finally:
            reopened.close()

    def test_invalid_frame_is_not_registered(self):
        tracker = self._tracker()
        shuffled = self.frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
        try:
            with self.assertRaises(TemporalValidationError):
                tracker.RegisterData(
                    shuffled, "target", partition="train",
                    temporal=TemporalSpec(time_column="time", freq="D"),
                )
            self.assertIsNone(tracker.GetTemporalSpec())
        finally:
            tracker.close()

    def test_set_cv_rejects_shuffled_splitters_for_temporal_tasks(self):
        tracker = self._tracker()
        try:
            tracker.RegisterData(
                self.frame, "target", partition="train",
                temporal=TemporalSpec(time_column="time", series_id="series", freq="D", horizon=2),
            )
            with self.assertRaises(ValueError) as ctx:
                tracker.SetCV({"type": "KFold", "n_splits": 3})
            self.assertIn("TimeSeriesSplit", str(ctx.exception))
            with self.assertRaises(ValueError):
                tracker.SetCV(3)
            stored = tracker.SetCV({"type": "TimeSeriesSplit", "n_splits": 3})
            self.assertEqual(stored["type"], "TimeSeriesSplit")
        finally:
            tracker.close()

    def test_tabular_tasks_keep_shuffled_cv(self):
        tracker = self._tracker()
        try:
            tracker.RegisterData(self.frame.drop(columns=["series"]), "target", partition="train")
            self.assertEqual(tracker.SetCV({"type": "KFold", "n_splits": 3})["type"], "KFold")
        finally:
            tracker.close()

    def test_set_temporal_spec_validates_known_columns_and_partitions(self):
        tracker = self._tracker()
        try:
            tracker.RegisterData(self.frame, "target", partition="train")
            with self.assertRaises(ValueError):
                tracker.SetTemporalSpec(TemporalSpec(time_column="absent"))
            spec, reports = tracker.SetTemporalSpec(
                TemporalSpec(time_column="time", series_id="series", freq="D")
            )
            self.assertEqual(spec["time_column"], "time")
            self.assertIn("train", reports)
            self.assertTrue(reports["train"]["ok"])
            tracker.ClearTemporalSpec()
            self.assertIsNone(tracker.GetTemporalSpec())
        finally:
            tracker.close()

    def test_target_cannot_be_a_temporal_column(self):
        tracker = self._tracker()
        try:
            with self.assertRaises(ValueError):
                tracker.RegisterData(
                    self.frame, "target", partition="train",
                    temporal=TemporalSpec(time_column="target"),
                )
        finally:
            tracker.close()


class TestTemporalEndpoints(unittest.TestCase):
    """Control-plane surfaces for the temporal declaration."""

    def setUp(self):
        self._original_root = settings.root
        self._tmp = tempfile.TemporaryDirectory()
        settings.root = Path(self._tmp.name)
        settings.ensure()
        self.frame = _daily_frame()

    def tearDown(self):
        settings.root = self._original_root
        self._tmp.cleanup()

    def _register(self, frame=None, temporal=None):
        return asyncio.run(
            datasets_router.register_dataset(
                TASK,
                target="target",
                partition="train",
                file=_upload(frame if frame is not None else self.frame),
                temporal=json.dumps(temporal) if temporal is not None else None,
            )
        )

    def test_register_with_temporal_declaration(self):
        response = self._register(
            temporal={"time_column": "time", "series_id": "series", "freq": "D", "horizon": 2}
        )
        self.assertTrue(response["temporal"]["validation"]["ok"])
        self.assertEqual(response["temporal"]["spec"]["time_column"], "time")
        self.assertIn("time", response["ignored_columns"])
        self.assertIn("series", response["ignored_columns"])
        with open_tracker(TASK) as tracker:
            self.assertIsNotNone(tracker.GetTemporalSpec())

    def test_register_rejects_unsorted_frame(self):
        shuffled = self.frame.sample(frac=1.0, random_state=11).reset_index(drop=True)
        with self.assertRaises(HTTPException) as ctx:
            self._register(
                frame=shuffled,
                temporal={"time_column": "time", "freq": "D"},
            )
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(ctx.exception.headers.get("X-Reason"), "temporal-validation")
        self.assertIn("report", ctx.exception.detail)

    def test_get_put_delete_temporal_roundtrip(self):
        self._register()
        empty = datasets_router.get_temporal(TASK)
        self.assertIsNone(empty["spec"])
        self.assertIsNone(empty["cv_temporal_safe"])

        body = datasets_router.TemporalSpecRequest(
            time_column="time", series_id="series", freq="D", horizon=2
        )
        payload = datasets_router.set_temporal(TASK, body)
        self.assertEqual(payload["spec"]["horizon"], 2)
        self.assertIn("train", payload["validation"])
        self.assertFalse(payload["cv_temporal_safe"])  # default splitter is shuffled
        self.assertEqual(payload["temporal_columns"], ["time", "series"])
        self.assertIn("time", payload["ignored_columns"])

        cleared = datasets_router.clear_temporal(TASK)
        self.assertIsNone(cleared["spec"])
        self.assertEqual(cleared["temporal_columns"], [])
        self.assertNotIn("time", cleared["ignored_columns"])

    def test_cv_endpoint_refuses_shuffled_splitter_after_declaration(self):
        self._register(
            temporal={"time_column": "time", "series_id": "series", "freq": "D"}
        )
        with self.assertRaises(HTTPException) as ctx:
            datasets_router.set_cv(
                TASK, datasets_router.SetCVRequest(spec={"type": "KFold", "n_splits": 3})
            )
        self.assertEqual(ctx.exception.status_code, 400)
        ok = datasets_router.set_cv(
            TASK, datasets_router.SetCVRequest(spec={"type": "TimeSeriesSplit", "n_splits": 3})
        )
        self.assertEqual(ok["cv"]["type"], "TimeSeriesSplit")

    def test_put_rejects_unknown_temporal_column(self):
        self._register()
        with self.assertRaises(HTTPException) as ctx:
            datasets_router.set_temporal(
                TASK, datasets_router.TemporalSpecRequest(time_column="nope")
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_endpoints_require_a_registered_dataset(self):
        with self.assertRaises(HTTPException) as ctx:
            datasets_router.get_temporal("no-such-task")
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
