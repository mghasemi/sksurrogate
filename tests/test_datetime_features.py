"""Datetime features: the declared type is real, and the number is stable.

A datetime dtype does not survive either storage path (``to_csv`` writes ISO text,
SQLite has no date type), so a column the dataset declares as ``datetime64`` came
back as ``object`` and was **one-hot encoded** — one column per distinct timestamp.
These tests pin the fix (metadata + coercion on load) and the numerical contract
that goes with it:

* ``DatetimeCentering`` emits **seconds relative to a fitted anchor**, not raw epoch
  seconds. Raw seconds made ``[1, x]`` ill-conditioned (``cond ~ 1e11``) and left L2
  penalties applied to a ``1e-6``-scale slope effectively inert.
* the anchor subtraction happens in the column's own resolution, so a year-1 or
  year-1000 timestamp yields a large finite number instead of an ``OverflowError``.
* the legacy ``DateTime2Num`` / ``DataPreprocess.encode()`` path runs again (it
  raised ``UFuncTypeError`` on numpy >= 2 for the numeric input it is fed).
"""

import importlib.util
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from SKSurrogate import mltrack  # noqa: E402
from SKSurrogate.DataProcess import (  # noqa: E402
    DataPreprocess,
    DateTime2Num,
    DatetimeCentering,
    FoldLocalDataPreprocessor,
)

from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402
from api.training import load_partition  # noqa: E402

TASK = "datetime-features"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    return TestClient(app, raise_server_exceptions=False)


def _frame(rows=6):
    days = pd.date_range("2026-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "when": days.strftime("%Y-%m-%d"),
            "x": np.arange(rows, dtype=float),
            "y": 2.0 * np.arange(rows, dtype=float),
        }
    )


def _register(client, frame=None, overrides=None, task=TASK):
    buffer = io.StringIO()
    (frame if frame is not None else _frame()).to_csv(buffer, index=False)
    data = {"target": "y", "partition": "train"}
    if overrides is not None:
        data["type_overrides"] = json.dumps(overrides)
    return client.post(
        f"/api/datasets/{task}/register",
        data=data,
        files={"file": ("train.csv", buffer.getvalue().encode(), "text/csv")},
    )


class TestDatetimeCentering:
    def test_seconds_are_relative_to_the_fitted_anchor(self):
        days = pd.date_range("2026-01-01", periods=4, freq="D")
        frame = pd.DataFrame({"d": days})
        transformer = DatetimeCentering().fit(frame)
        assert transformer.anchors_["d"] == days[0]
        assert transformer.transform(frame)["d"].tolist() == [0.0, 86400.0, 172800.0, 259200.0]

    def test_unparseable_values_become_nan_not_inf(self):
        frame = pd.DataFrame({"d": ["2026-01-01", "not-a-date", "2026-01-03"]})
        out = DatetimeCentering().fit(frame).transform(frame)["d"]
        assert np.isnan(out.iloc[1])
        assert np.isfinite(out.iloc[0]) and np.isfinite(out.iloc[2])

    def test_dates_outside_the_nanosecond_range_do_not_overflow(self):
        # datetime64[s] keeps its resolution; a year-1000 value cannot be cast to
        # int64 nanoseconds, but subtracting an anchor in seconds can.
        values = np.array(["1000-01-01", "1000-01-02", "1000-01-03"], dtype="datetime64[s]")
        frame = pd.DataFrame({"d": values})
        out = DatetimeCentering().fit(frame).transform(frame)["d"].tolist()
        assert out == [0.0, 86400.0, 172800.0]

    def test_timezone_aware_input_is_normalised_to_utc(self):
        frame = pd.DataFrame({"d": ["2026-01-01T00:00:00+02:00", "2026-01-02T00:00:00+02:00"]})
        assert DatetimeCentering().fit(frame).transform(frame)["d"].tolist() == [0.0, 86400.0]

    def test_constant_column_centres_to_zero(self):
        frame = pd.DataFrame({"d": ["2026-01-01"] * 3})
        assert DatetimeCentering().fit(frame).transform(frame)["d"].tolist() == [0.0, 0.0, 0.0]


class TestFoldLocalPreprocessor:
    def test_datetime_column_uses_centred_seconds_not_one_hot(self):
        frame = pd.DataFrame(
            {
                "d": pd.date_range("2026-01-01", periods=4, freq="D"),
                "x": np.arange(4.0),
            }
        )
        pre = FoldLocalDataPreprocessor().fit(frame)
        assert list(pre.get_feature_names_out()) == ["numeric__x", "datetime__d"]
        values = np.asarray(pre.transform(frame))
        assert values[:, 1].tolist() == [0.0, 86400.0, 172800.0, 259200.0]

    def test_scale_numeric_standardises_the_centred_seconds(self):
        frame = pd.DataFrame(
            {
                "d": pd.date_range("2026-01-01", periods=5, freq="D"),
                "x": np.arange(5.0),
            }
        )
        pre = FoldLocalDataPreprocessor(scale_numeric=True).fit(frame)
        values = np.asarray(pre.transform(frame))[:, 1]
        assert abs(values.mean()) < 1e-12
        assert abs(values.std() - 1.0) < 1e-12

    def test_object_date_column_is_still_one_hot(self):
        # Documents the difference the coercion makes: without a declared dtype a
        # date string is a category, and one column per distinct value appears.
        frame = pd.DataFrame(
            {"d": ["2026-01-01", "2026-01-02", "2026-01-03"], "x": np.arange(3.0)}
        )
        pre = FoldLocalDataPreprocessor().fit(frame)
        names = list(pre.get_feature_names_out())
        assert sum(name.startswith("categorical__d_") for name in names) == 3


class TestLegacyConversion:
    def test_datetime2num_accepts_numeric_seconds(self):
        values = pd.DataFrame({"d": [0.0, 86400.0, 172800.0]})
        assert DateTime2Num(cols=["d"]).fit_transform(values)["d"].tolist() == [0.0, 1.0, 2.0]

    def test_datetime2num_accepts_datetimes(self):
        values = pd.DataFrame({"d": pd.date_range("2026-01-01", periods=3, freq="D")})
        assert DateTime2Num(cols=["d"]).fit_transform(values)["d"].tolist() == [0.0, 1.0, 2.0]

    def test_datetime2num_constant_column_maps_to_lower(self):
        values = pd.DataFrame({"d": pd.to_datetime(["2026-01-01"] * 3)})
        assert DateTime2Num(cols=["d"]).fit_transform(values)["d"].tolist() == [0.0, 0.0, 0.0]

    def test_encode_runs_on_a_datetime_frame(self, tmp_path):
        # Regression: DataPreprocess.encode() raised UFuncTypeError on numpy >= 2
        # (float seconds divided by a timedelta64) for every datetime column.
        frame = pd.DataFrame(
            {
                "d": pd.date_range("2026-01-01", periods=5, freq="D"),
                "x": np.arange(5.0),
                "y": np.arange(5.0),
            },
            index=range(5),
        )
        preprocess = DataPreprocess(frame.copy())
        preprocess.deduce_types()
        preprocess.encode()
        assert preprocess.transformed_df["d"].tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]

    def test_tracker_encode_true_registers_a_datetime_frame(self, tmp_path):
        frame = pd.DataFrame(
            {
                "d": pd.date_range("2026-01-01", periods=6, freq="D"),
                "x": np.arange(6.0),
                "y": np.arange(6.0),
            },
            index=range(6),
        )
        tracker = mltrack("encode-dates", db_name=str(tmp_path / "track.db"), encode=True)
        with pytest.warns(UserWarning):
            tracker.RegisterData(frame, "y", partition="train")
        stored = tracker.get_dataframe(partition="train")
        assert stored["d"].dtype == np.dtype("float64")


class TestRegistrationAndLoading:
    def test_datetime_override_is_coerced_on_load(self, client):
        response = _register(client, overrides={"when": "datetime64"})
        assert response.status_code == 200, response.text
        deduced = response.json().get("dataset_deduced_types") or {}
        assert deduced.get("when") == "datetime64"

        frame = load_partition(TASK, "train")
        assert pd.api.types.is_datetime64_any_dtype(frame["when"])

        # And the preprocessor moves it onto the datetime branch, not one-hot.
        pre = FoldLocalDataPreprocessor().fit(frame[["when", "x"]])
        assert list(pre.get_feature_names_out()) == ["numeric__x", "datetime__when"]

    def test_datetime_override_must_parse(self, client):
        frame = _frame().assign(when=["nope"] * 6)
        response = _register(client, frame=frame, overrides={"when": "datetime64"})
        assert response.status_code == 400, response.text
        detail = response.json()["detail"]
        assert "declared datetime64" in detail
        assert "nope" in detail

    def test_undeclared_date_string_is_not_coerced(self, client):
        response = _register(client)
        assert response.status_code == 200, response.text
        frame = load_partition(TASK, "train")
        # Not coerced to datetime: the column keeps its string type. The exact storage
        # class is a pandas-version detail — pandas 2 infers ``object`` for text, pandas 3
        # infers its ``str`` dtype — so assert the type family and the values, not
        # ``dtype == object`` (which is what the "current deps" CI runner runs on pandas 3).
        assert not pd.api.types.is_datetime64_any_dtype(frame["when"])
        assert frame["when"].dtype == object or pd.api.types.is_string_dtype(frame["when"])
        assert frame["when"].tolist() == _frame()["when"].tolist()
