"""Phase P6 acceptance tests: multi-step temporal forecast inference.

Covers plan phase P6 (P6.1-P6.4) of ``TIME_SERIES_PLAN.md``:

* the serving state round-trips inside a v2 bundle record, and the v1 reader
  still loads tabular bundles without a ``temporal`` block;
* batch inference treats history as history - stable validation, one row per
  forecast step, monitoring metrics, atomic CSV output;
* recursive serving matches a hand-written step-by-step Python reference (with
  state feedback beyond the design horizon) and direct serving matches an
  origin-position reference, both within 1e-9;
* histories shorter than the required tail are refused with an actionable
  message instead of silently producing partial forecasts;
* ``POST /{task}/forecast`` on the inference and serving routers round-trips
  through the FastAPI test client, and tabular bundles get a 400 rather than a
  faked forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from collections.abc import Mapping
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from SKSurrogate import (
    ForecastFeatureBuilder,
    IntervalCalibration,
    ModelBundle,
    ModelRegistry,
    TemporalServeState,
    TemporalSpec,
    calibrate_from_residuals,
    load_bundle,
    predict_batch_forecast,
    save_bundle,
    serve_forecast,
)

H = 4   # design horizon of the default bundles
N = 60  # history length per series


# ---------------------------------------------------------------------------
# Data and bundle construction (deterministic: pure functions, no RNG)
# ---------------------------------------------------------------------------

def _series_a(t):
    t = np.asarray(t, dtype=float)
    return np.sin(t / 3.0) + 0.02 * t


def _series_b(t):
    t = np.asarray(t, dtype=float)
    return 1.5 + np.cos(t / 2.5) - 0.01 * t


def _history(n: int = N) -> pd.DataFrame:
    """A single series on a unit-spaced numeric axis (time = 1..n)."""
    t = np.arange(1, n + 1, dtype=float)
    return pd.DataFrame({"time": t, "y": _series_a(t)})


def _panel(n: int = N) -> pd.DataFrame:
    """Two stores, each with its own contiguous unit-spaced axis."""
    a = pd.DataFrame({"store": "a", "time": np.arange(1.0, n + 1), "y": None})
    b = pd.DataFrame({"store": "b", "time": np.arange(1.0, n + 1), "y": None})
    a["y"] = _series_a(a["time"])
    b["y"] = _series_b(b["time"])
    return pd.concat([a, b], ignore_index=True)


def _train(spec: TemporalSpec, frame: pd.DataFrame, strategy: str, lags=(H,), target="y"):
    """Fit the feature builder and a linear estimator; return (pipeline, design)."""
    builder = ForecastFeatureBuilder(spec=spec, target=target, lags=list(lags))
    builder.fit(frame)
    design = builder.build(frame, target_horizon=1 if strategy == "recursive" else None)
    model = LinearRegression().fit(design.X, design.y)
    return Pipeline([("features", builder), ("model", model)]), design


def _bundle(
    task_name: str = "tsfc",
    panel: bool = False,
    strategy: str = "direct",
    horizon: int = H,
    lags=(H,),
    calibration: "dict | None" = None,  # Mapping from IntervalCalibration.to_dict()
) -> ModelBundle:
    frame = _panel() if panel else _history()
    spec = TemporalSpec(
        time_column="time", series_id="store" if panel else None, horizon=horizon
    )
    pipeline, design = _train(spec, frame, strategy=strategy, lags=lags)
    state = TemporalServeState.from_parts(
        spec, "y", pipeline.named_steps["features"], strategy=strategy
    )
    # Score the estimator directly: transform() needs the raw history (it builds
    # lag features from the target column), so pipeline.score(design.X, ...) would
    # reject the design matrix.
    fitted = pipeline.named_steps["model"]
    residuals = design.y.to_numpy(dtype=float) - fitted.predict(design.X)
    ss_tot = float(np.sum((design.y - design.y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residuals**2)) / ss_tot if ss_tot > 0 else 0.0
    return ModelBundle(
        pipeline,
        task_name=task_name,
        metrics={"r2": r2},
        temporal=state.to_dict(),
        dependencies={},
        interval_calibration=calibration,
    )


def _residuals_for(frame: pd.DataFrame, spec: TemporalSpec, strategy: str, lags=(H,)) -> np.ndarray:
    _, design = _train(spec, frame, strategy=strategy, lags=lags)
    model = LinearRegression().fit(design.X, design.y)
    return design.y.to_numpy(dtype=float) - model.predict(design.X)


def _calibrated_bundle(**kwargs) -> ModelBundle:
    panel = kwargs.get("panel", False)
    frame = _panel() if panel else _history()
    spec = TemporalSpec(
        time_column="time", series_id="store" if panel else None, horizon=H
    )
    residuals = _residuals_for(frame, spec, strategy=kwargs.get("strategy", "direct"), lags=kwargs.pop("lags", (H,)))
    calibration = calibrate_from_residuals(residuals, alpha=0.1, method="absolute")
    # ModelBundle stores the calibration as its JSON mapping (see to_record);
    # tests that need .bounds() reconstruct it via IntervalCalibration.from_dict.
    return _bundle(calibration=calibration.to_dict(), **kwargs)


def _cal_of(bundle: ModelBundle) -> IntervalCalibration:
    """Rebuild the calibration object from the mapping a bundle stores."""
    value = bundle.interval_calibration
    assert isinstance(value, Mapping), "this test only uses calibrated bundles"
    return IntervalCalibration.from_dict(value)


# ---------------------------------------------------------------------------
# Step-by-step references (the oracles the plan's acceptance compares against)
# ---------------------------------------------------------------------------

def _direct_reference(bundle: ModelBundle, history: pd.DataFrame, horizon: int):
    """Predict each of the last ``horizon`` origins with the stored pipeline."""
    transformer = bundle.model.named_steps["features"]
    estimator = bundle.model.named_steps["model"]
    features = transformer.transform(history)  # fitted dropna=True -> rows H..n-1
    n = len(history)
    return np.array(
        [float(estimator.predict(features.loc[[p]])[0]) for p in range(n - horizon, n)]
    )


def _recursive_reference(bundle: ModelBundle, history: pd.DataFrame, steps: int):
    """Hand-written one-step recursion with state feedback (the plan's oracle).

    Each step appends the previous prediction to the working frame and re-runs
    the *fitted* transformer over it - exactly what a user scripting forecasts in
    Python would do by hand.
    """
    transformer = bundle.model.named_steps["features"]
    estimator = bundle.model.named_steps["model"]
    working = history.reset_index(drop=True).copy()
    last_time = float(working["time"].iloc[-1])
    observed_step = spec_step(working)
    step = observed_step if observed_step is not None else 1.0
    preds: list[float] = []
    for k in range(steps):
        # Features are read at the *current* last origin, whose label is the next
        # observation: that is the step-k forecast (t + k + 1). Appending the
        # synthetic row first would predict t + k + 2 and shift the whole path.
        x_last = transformer.transform(working).iloc[[-1]]
        pred = float(estimator.predict(x_last)[0])
        row = {column: np.nan for column in working.columns}
        row["time"] = last_time + (k + 1) * step
        if "store" in working.columns:
            row["store"] = working["store"].iloc[-1]
        row["y"] = pred
        working = pd.concat([working, pd.DataFrame([row])], ignore_index=True)
        preds.append(pred)
    return np.array(preds)


def _tail(frame: pd.DataFrame, m: int) -> pd.DataFrame:
    """``m`` further real points of series A beyond ``frame``'s last timestamp."""
    last = float(frame["time"].iloc[-1])
    t = np.arange(last + 1.0, last + m + 1.0)
    out = pd.DataFrame({"time": t})
    if "store" in frame.columns:
        out["store"] = frame["store"].iloc[-1]
    out["y"] = _series_a(t)
    return out


def spec_step(frame: pd.DataFrame):
    values = pd.to_numeric(frame["time"], errors="coerce").dropna().to_numpy()
    if len(values) < 2:
        return None
    diffs = np.diff(np.sort(values))
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if diffs.size else None


# ---------------------------------------------------------------------------
# P6.2 / P6.4 - the temporal block lives in a v2 bundle record, v1 still loads
# ---------------------------------------------------------------------------

class TestTemporalBundleFormat:
    def test_v2_record_roundtrip(self, tmp_path):
        from SKSurrogate.modelbundle import BUNDLE_FORMAT_VERSION

        assert BUNDLE_FORMAT_VERSION == 2
        original = _bundle()
        path = tmp_path / "temporal.bundle"
        save_bundle(original, path)
        loaded = load_bundle(path)

        assert loaded.temporal == original.temporal
        reloaded_state = TemporalServeState.from_dict(loaded.temporal)
        history = _history()
        expected = np.asarray(serve_forecast(original, history).means, dtype=float)
        got = np.asarray(loaded.predict_forecast(history).means, dtype=float)
        assert np.allclose(got, expected, rtol=0.0, atol=1e-9)

    def test_v1_bundle_still_loads_and_predicts_tabularly(self):
        rng = np.random.default_rng(7)
        n = 40
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = 1.5 * x1 - 0.7 * x2 + 0.3
        tabular = ModelBundle(
            LinearRegression().fit(np.column_stack([x1, x2]), y),
            task_name="tsfc_v1",
            schema={"features": ["x1", "x2"], "target": "y"},
            metrics={"r2": 0.9},
        )
        record = tabular.to_record()
        assert "temporal" not in record or record["temporal"] is None

        v1 = dict(record)
        v1["format_version"] = 1
        v1.pop("temporal", None)

        loaded = ModelBundle.from_record(v1)
        assert loaded.temporal is None
        frame = pd.DataFrame({"x1": [0.5, -1.0], "x2": [0.25, 0.75]})
        predictions = np.asarray(loaded.predict(frame), dtype=float)
        expected = tabular.predict(frame)
        assert predictions.shape == (2,)
        assert np.allclose(predictions, np.asarray(expected, dtype=float), rtol=0.0, atol=1e-9)

        with pytest.raises(ValueError, match="no temporal design"):
            loaded.predict_forecast(pd.DataFrame({"time": [1.0], "y": [0.1]}))


# ---------------------------------------------------------------------------
# P6.3 - batch forecast contract (programmatic side)
# ---------------------------------------------------------------------------

class TestBatchForecastContract:
    def test_frame_shape_metrics_and_request_id(self):
        bundle = _bundle()
        result = predict_batch_forecast(bundle, _history(), request_id="req-1")

        assert list(result.columns) == [
            "time",
            "series_id",
            "prediction",
            "lower",
            "upper",
            "model_version",
            "request_id",
        ]
        assert len(result) == H
        assert (result["request_id"] == "req-1").all()
        metrics = result.attrs["inference_metrics"]
        assert set(metrics) >= {"latency_ms", "rows"}
        assert metrics["latency_ms"] >= 0.0
        assert metrics["rows"] == H

    def test_direct_matches_origin_reference_within_1e9(self):
        bundle = _bundle()
        history = _history()
        result = serve_forecast(bundle, history)
        expected_times = [float(history["time"].iloc[-1] + k) for k in range(1, H + 1)]

        assert list(result.times) == pytest.approx(expected_times, abs=1e-9)
        got = np.asarray(result.means, dtype=float)
        ref = _direct_reference(bundle, history, H)
        assert got.shape == (H,)
        assert np.allclose(got, ref, rtol=0.0, atol=1e-9)

    def test_recursive_matches_step_by_step_with_feedback_within_1e9(self):
        # Design horizon 4 with lag 4: requesting 8 steps forces the serving path
        # to read its own predictions (state feedback) from step 5 onward.
        bundle = _bundle(strategy="recursive")
        history = _history()
        result = serve_forecast(bundle, history, horizon=8)

        got = np.asarray(result.means, dtype=float)
        ref = _recursive_reference(bundle, history, 8)
        assert got.shape == (8,)
        assert np.allclose(got, ref, rtol=0.0, atol=1e-9)

        # Feedback check: from step H onward the lag window contains earlier
        # predictions. Re-running each of those steps on a frame extended with
        # *true* future values must give different features - and hence
        # different forecasts - or state feedback would have never happened.
        n = len(history)
        transformer = bundle.model.named_steps["features"]
        estimator = bundle.model.named_steps["model"]
        feats_true = transformer.transform(
            pd.concat([history, _tail(history, 8)], ignore_index=True)
        )
        for k in range(H, 8):
            no_feedback = float(estimator.predict(feats_true.loc[[n + k]])[0])
            assert not np.allclose(ref[k], no_feedback), (
                f"step {k}: forecast equals the true-value reading - state "
                "feedback never happened"
            )

    def test_direct_refuses_horizon_outside_design(self):
        bundle = _bundle()  # design horizon H
        with pytest.raises(ValueError, match="design horizon"):
            serve_forecast(bundle, _history(), horizon=H + 1)

    def test_short_history_raises_with_required_tail(self):
        bundle = _bundle()  # direct, lag H -> required tail max(H, H+H) == 8
        state = TemporalServeState.from_dict(bundle.temporal)
        assert state.required_length() == 2 * H

        with pytest.raises(ValueError, match="needs at least %d" % (2 * H)):
            serve_forecast(bundle, _history(n=2 * H - 1))

        # Exactly the required tail serves without complaint.
        result = serve_forecast(bundle, _history(n=2 * H))
        assert len(np.asarray(result.means)) == H

    def test_empty_history_rejected(self):
        bundle = _bundle()
        empty = pd.DataFrame(columns=["time", "y"])
        with pytest.raises(ValueError, match="history is empty"):
            serve_forecast(bundle, empty)

    def test_tabular_bundle_refused_not_faked(self):
        rng = np.random.default_rng(3)
        n = 40
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        tabular = ModelBundle(
            LinearRegression().fit(np.column_stack([x1, x2]), 1.5 * x1 - 0.7 * x2),
            task_name="tsfc_tab",
            schema={"x1": {"dtype": "float64"}, "x2": {"dtype": "float64"}},
            metrics={"r2": 0.9},
        )
        frame = pd.DataFrame({"time": np.arange(1.0, n + 1), "y": _series_a(np.arange(1, n + 1))})

        with pytest.raises(ValueError, match="no temporal design"):
            tabular.predict_forecast(frame)
        # Batch entry point: the frame passes the classic schema check only when it
        # matches; either way a forecast must be refused, never faked.
        matching = pd.DataFrame({"x1": [0.1], "x2": [-0.5]})
        with pytest.raises(ValueError, match="no temporal design"):
            predict_batch_forecast(tabular, matching)

    def test_predict_batch_dispatches_temporal_bundles(self):
        from SKSurrogate import predict_batch

        bundle = _bundle()
        history = _history()
        via_classic = predict_batch(bundle, history)
        via_forecast = predict_batch_forecast(bundle, history)

        assert len(via_classic) == H
        assert np.allclose(
            via_classic["prediction"].to_numpy(dtype=float),
            via_forecast["prediction"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-9,
        )

    def test_atomic_csv_output(self, tmp_path):
        bundle = _bundle()
        out = tmp_path / "forecast.csv"
        result = predict_batch_forecast(bundle, _history(), output_path=out)

        assert out.exists()
        on_disk = pd.read_csv(out)
        assert list(on_disk.columns) == [c for c in result.columns]
        np.testing.assert_allclose(
            on_disk["prediction"].to_numpy(dtype=float),
            result["prediction"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-9,
        )


# ---------------------------------------------------------------------------
# P5 -> P6 wiring - intervals appear only when the bundle is calibrated
# ---------------------------------------------------------------------------

class TestForecastIntervals:
    def test_intervals_absent_unless_calibrated(self):
        result = serve_forecast(_bundle(), _history())
        assert result.lower is None and result.upper is None
        frame = result.to_frame()
        assert frame["lower"].isna().all() and frame["upper"].isna().all()

    def test_calibrated_bounds_equal_means_plus_offsets(self):
        bundle = _calibrated_bundle()
        history = _history()
        result = serve_forecast(bundle, history)
        cal = _cal_of(bundle)

        means = np.asarray(result.means, dtype=float)
        lower = np.asarray(result.lower, dtype=float)
        upper = np.asarray(result.upper, dtype=float)
        assert lower.shape == means.shape and upper.shape == means.shape
        expected_lower, expected_upper = cal.bounds(means)
        assert np.allclose(lower, expected_lower, rtol=0.0, atol=1e-9)
        assert np.allclose(upper, expected_upper, rtol=0.0, atol=1e-9)
        assert (lower < means).all() and (means < upper).all()

    def test_calibrated_batch_frame_carries_bounds(self):
        bundle = _calibrated_bundle()
        result = predict_batch_forecast(bundle, _history())
        cal = IntervalCalibration.from_dict(bundle.interval_calibration)
        prediction = result["prediction"].to_numpy(dtype=float)
        expected_lower, expected_upper = cal.bounds(prediction)
        assert np.allclose(result["lower"].to_numpy(dtype=float), expected_lower, atol=1e-9)
        assert np.allclose(result["upper"].to_numpy(dtype=float), expected_upper, atol=1e-9)


# ---------------------------------------------------------------------------
# Acceptance (plan P6): a two-series panel bundle serves a 12-step forecast with
# intervals, and predict_forecast matches the step-by-step reference within 1e-9.
# ---------------------------------------------------------------------------

class TestPanelAcceptance:
    def test_two_series_panel_serves_twelve_steps_with_intervals(self):
        bundle = _calibrated_bundle(panel=True, strategy="recursive")
        frame = _panel()
        history_a = frame[frame["store"] == "a"].reset_index(drop=True)
        history_b = frame[frame["store"] == "b"].reset_index(drop=True)

        for label, history in (("a", history_a), ("b", history_b)):
            result = serve_forecast(bundle, history, horizon=12)
            assert result.horizon == 12
            got = np.asarray(result.means, dtype=float)
            ref = _recursive_reference(bundle, history, 12)
            assert np.allclose(got, ref, rtol=0.0, atol=1e-9), label

            # P6.2 entry point agrees with the serving state path.
            via_bundle = np.asarray(
                bundle.predict_forecast(history, horizon=12).to_frame()["mean"], dtype=float
            )
            assert np.allclose(via_bundle, got, rtol=0.0, atol=1e-9), label

            # Intervals: present, ordered around the mean, from the calibration.
            lower = np.asarray(result.lower, dtype=float)
            upper = np.asarray(result.upper, dtype=float)
            cal = IntervalCalibration.from_dict(bundle.interval_calibration)
            expected_lower, expected_upper = cal.bounds(got)
            assert np.allclose(lower, expected_lower, atol=1e-9), label
            assert np.allclose(upper, expected_upper, atol=1e-9), label
            assert (lower < got).all() and (got < upper).all(), label

        # The two stores must not be interchangeable: different histories give
        # different forecasts.
        fa = serve_forecast(bundle, history_a, horizon=12).means
        fb = serve_forecast(bundle, history_b, horizon=12).means
        assert not np.allclose(np.asarray(fa), np.asarray(fb))


# ---------------------------------------------------------------------------
# P6.3 - HTTP round-trip through the FastAPI test client
# ---------------------------------------------------------------------------

TASK = "tsfc"
TAB_TASK = "tsfc_tab"


@pytest.fixture()
def api_root(tmp_path, monkeypatch):
    from api import auth
    from api.config import settings

    monkeypatch.setattr(settings, "root", tmp_path)
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "require_auth", False)
    settings.ensure()
    auth.THROTTLE._failures.clear()
    return tmp_path


@pytest.fixture()
def client():
    from api.main import app

    return TestClient(app)


def _register(api_root, bundle):
    registry = ModelRegistry(api_root / "registry")
    version = registry.register(bundle)
    registry.promote(bundle.task_name, version, "production")
    return registry, version


class TestForecastEndpoints:
    def test_inference_forecast_roundtrip(self, api_root, client):
        bundle = _bundle(task_name=TASK)
        _register(api_root, bundle)
        history = _history()

        response = client.post(
            f"/api/inference/{TASK}/forecast",
            json={
                "alias": "production",
                "horizon": H,
                "history": history.to_dict(orient="records"),
                "request_id": "http-req-1",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["task_name"] == TASK
        assert body["strategy"] == "direct"
        assert body["horizon"] == H
        assert body["request_id"] == "http-req-1"
        last_time = float(history["time"].iloc[-1])
        assert [float(t) for t in body["times"]] == [last_time + k for k in range(1, H + 1)]
        ref = _direct_reference(bundle, history, H)
        assert np.allclose(body["predictions"], ref, rtol=0.0, atol=1e-9)
        # Not calibrated -> bounds are null and no note is attached.
        assert body["lower_bounds"] is None and body["upper_bounds"] is None
        assert body["interval_note"] is None
        assert set(body["metrics"]) >= {"latency_ms", "rows"}

    def test_inference_forecast_roundtrip_calibrated(self, api_root, client):
        bundle = _calibrated_bundle(task_name=TASK)
        _register(api_root, bundle)
        history = _history()
        cal = IntervalCalibration.from_dict(bundle.interval_calibration)

        response = client.post(
            f"/api/inference/{TASK}/forecast",
            json={"alias": "production", "horizon": H, "history": history.to_dict(orient="records")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        means = np.asarray(body["predictions"], dtype=float)
        expected_lower, expected_upper = cal.bounds(means)
        assert np.allclose(body["lower_bounds"], expected_lower, atol=1e-9)
        assert np.allclose(body["upper_bounds"], expected_upper, atol=1e-9)

    def test_serving_forecast_roundtrip(self, api_root, client):
        bundle = _bundle(task_name=TASK)
        registry, version = _register(api_root, bundle)
        history = _history()

        activation = client.post(f"/api/serving/{TASK}", json={"model_version": version})
        assert activation.status_code == 200, activation.text

        response = client.post(
            f"/api/serving/{TASK}/forecast",
            json={"horizon": H, "history": history.to_dict(orient="records")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["resolve_mode"] == "version"
        ref = _direct_reference(bundle, history, H)
        assert np.allclose(body["predictions"], ref, rtol=0.0, atol=1e-9)

    def test_short_history_gets_400(self, api_root, client):
        bundle = _bundle(task_name=TASK)
        _register(api_root, bundle)

        response = client.post(
            f"/api/inference/{TASK}/forecast",
            json={"alias": "production", "history": _history(n=2 * H - 1).to_dict(orient="records")},
        )
        assert response.status_code == 400
        assert "needs at least" in response.json()["detail"]

    def test_tabular_bundle_gets_400_not_a_faked_forecast(self, api_root, client):
        rng = np.random.default_rng(11)
        n = 40
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        tabular = ModelBundle(
            LinearRegression().fit(np.column_stack([x1, x2]), 1.5 * x1 - 0.7 * x2),
            task_name=TAB_TASK,
            schema={"x1": {"dtype": "float64"}, "x2": {"dtype": "float64"}},
            metrics={"r2": 0.9},
        )
        _register(api_root, tabular)

        rows = [{"x1": float(v), "x2": 0.0} for v in np.arange(1.0, n + 1)]
        response = client.post(
            f"/api/inference/{TAB_TASK}/forecast",
            json={"alias": "production", "history": rows},
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "no temporal design" in str(detail)

    def test_forecast_requires_history(self, api_root, client):
        bundle = _bundle(task_name=TASK)
        _register(api_root, bundle)

        response = client.post(f"/api/inference/{TASK}/forecast", json={"alias": "production"})
        assert response.status_code == 422  # pydantic: history is required


# ---------------------------------------------------------------------------
# Period labels (spec.period_start): the anchor IS the axis
# ---------------------------------------------------------------------------

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _period_labels(n: int = 36) -> list[str]:
    """``1-Jan``..``3-Dec``: *year index, month*, the shampoo-style label."""
    return ["%d-%s" % (index // 12 + 1, _MONTHS[index % 12]) for index in range(n)]


def _period_bundle(n: int = 36, horizon: int = 2) -> tuple[ModelBundle, pd.DataFrame]:
    labels = _period_labels(n)
    history = pd.DataFrame(
        {"Month": labels, "y": 100.0 + 2.0 * np.arange(n, dtype=float)}
    )
    spec = TemporalSpec(
        time_column="Month", horizon=horizon, freq="MS", period_start="2001-01-01"
    )
    pipeline, _design = _train(
        spec, history, strategy="recursive", lags=(horizon,), target="y"
    )
    state = TemporalServeState.from_parts(
        spec, "y", pipeline.named_steps["features"], strategy="recursive"
    )
    bundle = ModelBundle(
        pipeline,
        task_name="period-labels",
        metrics={},
        temporal=state.to_dict(),
        dependencies={},
    )
    return bundle, history


def test_period_labelled_history_serves_from_the_declared_anchor():
    """A bundle declared with ``period_start`` must accept the labels it trained on.

    ``1-Jan`` is *year 1, January*: parsing the column reads years 1..3 (outside
    pandas' nanosecond range) and, worse, month/day confusion can make the axis
    run backwards (``3-Oct`` -> 2026-10-03 then ``2-Jan`` -> 2026-01-02), so
    ``validate_history`` answered ``history time values must be strictly
    increasing`` for the bundle's own dataset. The axis must be built from the
    anchor and row order — the rule ``_time_axis`` already applies at
    registration — and the forecast must continue that grid.
    """
    bundle, history = _period_bundle()
    state = TemporalServeState.from_dict(bundle.temporal)

    frame, label = state.validate_history(history)
    assert label is None
    assert str(frame["Month"].iloc[0]) == "2001-01-01 00:00:00"
    assert str(frame["Month"].iloc[-1]) == "2003-12-01 00:00:00"

    result = serve_forecast(bundle, history, horizon=3)
    assert [str(pd.Timestamp(value)) for value in result.times] == [
        "2004-01-01 00:00:00",
        "2004-02-01 00:00:00",
        "2004-03-01 00:00:00",
    ]
    assert np.all(np.isfinite(np.asarray(result.means)))


def test_panel_period_labels_serve_one_series_at_a_time():
    """The anchor is per series: a panel's rows are ranked within their series."""
    labels = _period_labels(24)
    frame = pd.concat(
        [
            pd.DataFrame({"store": store, "Month": labels, "y": 10.0 + np.arange(24.0)})
            for store in ("a", "b")
        ],
        ignore_index=True,
    )
    spec = TemporalSpec(
        time_column="Month", series_id="store", horizon=2, freq="MS",
        period_start="2001-01-01",
    )
    builder = ForecastFeatureBuilder(spec=spec, target="y", lags=(2,))
    builder.fit(frame)
    state = TemporalServeState.from_parts(spec, "y", builder, strategy="recursive")
    one_series, label = state.validate_history(frame[frame["store"] == "b"])
    assert label == "b"
    assert str(one_series["Month"].iloc[0]) == "2001-01-01 00:00:00"
    with pytest.raises(ValueError, match="exactly one 'store' value"):
        state.validate_history(frame)
