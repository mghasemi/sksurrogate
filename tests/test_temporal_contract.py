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
    ForecastFeatureBuilder,
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


    def test_month_only_labels_are_refused_with_an_explanation(self):
        # "1-Jan" carries no year. pandas 3/dateutil fills in year 1 (which cannot
        # be represented in int64 nanoseconds and used to escape as an
        # OverflowError → HTTP 500); pandas 2 coerces it to NaT. Either way the
        # answer must be a validation error that names the column.
        frame = pd.DataFrame(
            {"time": ["1-Jan", "1-Feb", "1-Mar"], "target": [1.0, 2.0, 3.0]}
        )
        with self.assertRaises(TemporalValidationError) as ctx:
            validate_temporal_frame(frame, TemporalSpec(time_column="time"))
        message = str(ctx.exception)
        self.assertIn("'time'", message)
        self.assertTrue(
            "nanosecond" in message or "unparseable" in message, message
        )

    def test_years_outside_the_nanosecond_range_are_refused(self):
        # datetime64[s] keeps its resolution through to_datetime, so the values
        # really are year 1000 and the int64-nanosecond cast overflows — on every
        # pandas version (unlike the string form, which pandas 2 turns into NaT).
        times = np.array(
            ["1000-01-01", "1000-02-01", "1000-03-01"], dtype="datetime64[s]"
        )
        frame = pd.DataFrame({"time": times, "target": [1.0, 2.0, 3.0]})
        with self.assertRaises(TemporalValidationError) as ctx:
            validate_temporal_frame(frame, TemporalSpec(time_column="time"))
        message = str(ctx.exception)
        self.assertIn("nanosecond", message)
        self.assertIn("1000-01-01", message)

    def test_timezone_aware_times_are_normalised_to_utc(self):
        frame = pd.DataFrame(
            {
                "time": [
                    "2026-01-01T00:00:00+02:00",
                    "2026-01-02T00:00:00+02:00",
                    "2026-01-03T00:00:00+02:00",
                ],
                "target": [1.0, 2.0, 3.0],
            }
        )
        report = validate_temporal_frame(frame, TemporalSpec(time_column="time", freq="D"))
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["time_kind"], "datetime")
        self.assertEqual(report["inferred_freq"], "D")


    @staticmethod
    def _year_month_frame():
        """The shampoo-style shape: "1-Jan" .. "3-Dec" (year index, month)."""
        rows = []
        for year in range(1, 4):
            for month in ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"):
                rows.append("%d-%s" % (year, month))
        return pd.DataFrame({"Month": rows, "target": np.arange(len(rows), dtype=float)})

    def test_period_labels_are_refused_without_an_anchor(self):
        # No year in the labels: pandas reads "1-Jan" as year 1, which is outside
        # the nanosecond range. The error must say so instead of overflowing.
        with self.assertRaises(TemporalValidationError) as ctx:
            validate_temporal_frame(
                self._year_month_frame(), TemporalSpec(time_column="Month", freq="MS")
            )
        # pandas 3 parses "1-Jan" to year 1 (outside the nanosecond range);
        # pandas 2 coerces it to NaT. Either way it must be a validation error
        # that names the column, never an OverflowError.
        message = str(ctx.exception)
        self.assertIn("'Month'", message)
        self.assertTrue(
            "nanosecond" in message or "unparseable" in message, message
        )
        # The advice has to name the mechanism that actually works on this data.
        self.assertIn("period_start", message)

    def test_period_start_builds_the_axis_from_row_order(self):
        frame = self._year_month_frame()
        report = validate_temporal_frame(
            frame,
            TemporalSpec(
                time_column="Month", freq="MS", horizon=3, period_start="2001-01-01"
            ),
        )
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["time_source"], "period_start")
        self.assertEqual(report["inferred_freq"], "MS")
        self.assertEqual(report["time_range"]["start"], "2001-01-01 00:00:00")
        self.assertEqual(report["time_range"]["end"], "2003-12-01 00:00:00")

    def test_period_start_requires_a_frequency(self):
        with self.assertRaises(ValueError) as ctx:
            TemporalSpec(time_column="Month", period_start="2001-01-01")
        self.assertIn("requires freq", str(ctx.exception))

    def test_period_start_rejects_a_bad_or_aware_timestamp(self):
        with self.assertRaises(ValueError):
            TemporalSpec(time_column="Month", freq="MS", period_start="not-a-date")
        with self.assertRaises(ValueError) as ctx:
            TemporalSpec(
                time_column="Month", freq="MS", period_start="2001-01-01T00:00:00+02:00"
            )
        self.assertIn("timezone-naive", str(ctx.exception))

    def test_period_start_applies_per_series(self):
        frame = pd.DataFrame(
            {
                "time": ["p1", "p2", "p3", "p1", "p2", "p3"],
                "series": ["A", "A", "A", "B", "B", "B"],
                "target": np.arange(6, dtype=float),
            }
        )
        report = validate_temporal_frame(
            frame,
            TemporalSpec(
                time_column="time", series_id="series", freq="D",
                period_start="2026-01-01",
            ),
        )
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["series_count"], 2)
        # Each series starts at the anchor, not at a global row index.
        self.assertEqual([entry["rows"] for entry in report["series"]], [3, 3])

    def test_period_start_feeds_calendar_features(self):
        frame = self._year_month_frame()
        spec = TemporalSpec(
            time_column="Month", freq="MS", horizon=1, period_start="2001-01-01"
        )
        builder = ForecastFeatureBuilder(
            spec=spec, target="target", lags=[1], calendar=["month"]
        ).fit(frame)
        design = builder.build(frame)
        self.assertIn("cal__month", design.X.columns)
        months = design.X["cal__month"].dropna().astype(int).tolist()
        self.assertEqual(months[:3], [2, 3, 4])  # first row has no lag-1 feature

    def test_time_format_reads_ambiguous_text(self):
        frame = pd.DataFrame(
            {
                "when": ["03/01/2026", "04/01/2026", "05/01/2026"],
                "target": [1.0, 2.0, 3.0],
            }
        )
        # day-first: with "%d/%m/%Y" these are 3, 4 and 5 January 2026. Pandas
        # would guess month-first for "03/01/2026" and read March.
        report = validate_temporal_frame(
            frame,
            TemporalSpec(time_column="when", freq="D", time_format="%d/%m/%Y"),
        )
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["time_range"]["start"], "2026-01-03 00:00:00")

    def test_bad_time_format_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            TemporalSpec(time_column="when", time_format="%Q-%b")
        self.assertIn("strptime", str(ctx.exception))


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

    def test_cv_endpoint_offers_temporal_only_splitters(self):
        """RollingOriginSplit is temporal-only, so GET /cv must expose it.

        P7 needs the UI to offer order-respecting splitters for a temporal task;
        the standard list omits RollingOriginSplit, and without its param_defs
        the panel could show the name but never edit n_splits/step/min_train.
        """
        self._register(
            temporal={"time_column": "time", "series_id": "series", "freq": "D"}
        )
        payload = datasets_router.get_cv(TASK)
        self.assertIn("RollingOriginSplit", payload["temporal_options"])
        self.assertIn("TimeSeriesSplit", payload["temporal_options"])
        self.assertIn("RollingOriginSplit", payload["param_defs"])
        names = {defn["name"] for defn in payload["param_defs"]["RollingOriginSplit"]}
        self.assertIn("horizon", names)
        # The declaration itself also advertises the same legal set.
        temporal = datasets_router.get_temporal(TASK)
        self.assertEqual(sorted(temporal["temporal_splitters"]), sorted(payload["temporal_options"]))

    def test_put_refuses_month_only_labels_with_422_not_500(self):
        # The reporter's dataset: Month = "1-Jan", "1-Feb", ... — no year.
        rows = 12
        frame = pd.DataFrame(
            {
                "Month": ["%d-Jan" % (i + 1) for i in range(rows)],
                "target": np.arange(rows, dtype=float),
            }
        )
        self._register(frame=frame)
        body = datasets_router.TemporalSpecRequest(time_column="Month", freq="MS")
        with self.assertRaises(HTTPException) as ctx:
            datasets_router.set_temporal(TASK, body)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(ctx.exception.headers.get("X-Reason"), "temporal-validation")
        detail = ctx.exception.detail["message"]
        self.assertIn("Month", detail)
        self.assertTrue("nanosecond" in detail or "unparseable" in detail, detail)

    def test_put_accepts_period_start_for_period_labels(self):
        # The reported dataset: Month = "1-Jan" ... "3-Dec" (year index, month).
        rows = [
            "%d-%s" % (year, month)
            for year in range(1, 4)
            for month in ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
        ]
        frame = pd.DataFrame({"Month": rows, "target": np.arange(len(rows), dtype=float)})
        self._register(frame=frame)
        body = datasets_router.TemporalSpecRequest(
            time_column="Month", freq="MS", horizon=3, period_start="2001-01-01"
        )
        payload = datasets_router.set_temporal(TASK, body)
        report = payload["validation"]["train"]
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["time_source"], "period_start")
        self.assertEqual(payload["spec"]["period_start"], "2001-01-01")

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
