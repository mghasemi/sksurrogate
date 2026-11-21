"""Batch-prediction CSV download endpoint (``GET /api/inference/{task}/predictions/...``).

The Inference page's batch-run success note links to this URL so an operator can
retrieve a persisted artifact without shell access to the server. Contract under
test:

* the exact file ``POST .../predict-batch`` wrote is served back byte-identical,
  with a CSV media type and a ``Content-Disposition`` carrying its filename;
* an unknown artifact answers **404** (a data question, not an error path);
* caller-supplied segments that would escape the storage root are refused at the
  containment guard (**400**), so nothing outside the predictions directory is
  ever read — mirroring every other artifact endpoint.

Run with ``pytest tests/test_prediction_download.py -v``.
"""

import io
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from api import acl, auth
from api.config import settings
from api.main import app
from SKSurrogate import ModelBundle, ModelRegistry

TASK = "prediction-download"
FEATURES = {"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0, 4.0]}


def _bundle(shift):
    frame = pd.DataFrame(FEATURES)
    target = [value + shift for value in frame["a"]]
    return ModelBundle(
        LinearRegression().fit(frame, target),
        task_name=TASK,
        schema={"a": {"dtype": "float64"}, "b": {"dtype": "float64"}},
        metrics={"r2": 1.0},
    )


def _open_root(tmp_path, monkeypatch):
    """Point the API at a throwaway root and clear identity state."""
    monkeypatch.setattr(settings, "root", tmp_path)
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "require_auth", False)
    settings.ensure()
    auth.reset_store()
    acl.reset_store()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A fresh root with one registered model and a live client."""
    _open_root(tmp_path, monkeypatch)
    registry = ModelRegistry(settings.registry_dir)
    version = registry.register(_bundle(0.0))
    client = TestClient(app)
    yield SimpleNamespace(client=client, version=version)
    auth.THROTTLE._failures.clear()


def _batch(env, request_id="req-1"):
    response = env.client.post(
        f"/api/inference/{TASK}/predict-batch",
        json={"model_version": env.version, "rows": [{"a": 1.0, "b": 2.0}], "request_id": request_id},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_batch_artifact_is_served_back(env):
    body = _batch(env)
    filename = f"{body['request_id']}.csv"
    on_disk = settings.task_predictions_dir(TASK) / env.version / filename
    assert on_disk.is_file(), "predict-batch must persist the CSV before answering"

    response = env.client.get(f"/api/inference/{TASK}/predictions/{env.version}/{filename}")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    assert f'filename="{filename}"' in response.headers.get("content-disposition", "")

    # Byte-identical to what the batch run wrote — no re-encoding in transit.
    assert response.content == on_disk.read_bytes()

    served = pd.read_csv(io.BytesIO(response.content))
    assert list(served.columns) == ["prediction", "model_version", "request_id"]
    assert len(served) == 1
    # target was ``a``; the row scored is a=1.0, b=2.0 → prediction ≈ 1.0.
    assert float(served["prediction"].iloc[0]) == pytest.approx(1.0)


def test_missing_artifact_is_a_404(env):
    _batch(env)  # create one artifact so the version directory exists
    response = env.client.get(f"/api/inference/{TASK}/predictions/{env.version}/nope.csv")
    assert response.status_code == 404, response.text


def test_non_csv_filename_is_rejected(env):
    _batch(env)
    response = env.client.get(f"/api/inference/{TASK}/predictions/{env.version}/notes.txt")
    assert response.status_code == 400, response.text


def test_traversal_does_not_leave_the_predictions_tree(env):
    # A file planted at the storage root must stay unreachable: the encoded
    # ``..`` segment is refused by the containment guard before any path join.
    stray = settings.root / "evil.csv"
    stray.write_text("top,secret\n1,x\n", encoding="utf-8")

    response = env.client.get(f"/api/inference/{TASK}/predictions/%2E%2E/evil.csv")
    assert response.status_code == 400, response.text


def test_artifacts_are_namespaced_per_task(env):
    # The endpoint serves exactly ``<root>/predictions/<task>/<version>/<file>``:
    # an artifact stored under another task is reachable only through that task's
    # URL, never through this one. (In open mode every task name is addressable —
    # the ACL layer gates it when auth is enforced.)
    body = _batch(env)
    filename = f"{body['request_id']}.csv"

    other_dir = settings.task_predictions_dir("other-task") / env.version
    other_dir.mkdir(parents=True, exist_ok=True)
    (other_dir / filename).write_bytes(
        b"prediction,model_version,request_id\n9.9,other,other-req\n"
    )

    own = env.client.get(f"/api/inference/{TASK}/predictions/{env.version}/{filename}")
    assert own.status_code == 200, own.text
    other = env.client.get(f"/api/inference/other-task/predictions/{env.version}/{filename}")
    assert other.status_code == 200, other.text
    # Each URL returns its own task's file — the contents must not be crossed.
    assert own.content != other.content


# ---------------------------------------------------------------------------
# Forecast payload → the classic time,prediction,lower,upper CSV
# ---------------------------------------------------------------------------

def _temporal_bundle():
    """A minimal direct-horizon temporal bundle: ``y = 2t`` with lag 2."""
    import numpy as np
    from sklearn.linear_model import LinearRegression as _LR
    from sklearn.pipeline import Pipeline

    from SKSurrogate import ForecastFeatureBuilder, TemporalServeState, TemporalSpec

    spec = TemporalSpec(time_column="time", horizon=2)
    frame = pd.DataFrame({"time": np.arange(1.0, 41.0)})
    frame["y"] = 2.0 * frame["time"]
    builder = ForecastFeatureBuilder(spec=spec, target="y", lags=[2])
    builder.fit(frame)
    design = builder.build(frame)
    pipeline = Pipeline(
        [("features", builder), ("model", _LR().fit(design.X, design.y))]
    )
    bundle = ModelBundle(
        pipeline,
        task_name=TASK,
        schema={},
        metrics={},
        temporal=TemporalServeState.from_parts(
            spec, "y", builder, strategy="direct"
        ).to_dict(),
    )
    return bundle, frame


def test_forecast_payload_round_trips_to_the_download_csv(tmp_path, monkeypatch):
    """The Inference page's forecast download is built from this payload.

    ``docs/forecasting.rst`` promises ``time,prediction,lower,upper``; the client
    writes exactly those columns from ``times`` / ``predictions`` /
    ``lower_bounds`` / ``upper_bounds``. This test pins that the server supplies
    every column (with nulls when the bundle is uncalibrated), so the download
    cannot silently lose a field.
    """
    _open_root(tmp_path, monkeypatch)
    bundle, frame = _temporal_bundle()
    version = ModelRegistry(settings.registry_dir).register(bundle)
    client = TestClient(app)
    history = [
        {"time": float(row.time), "y": float(row.y)}
        for row in frame.itertuples(index=False)
    ]

    response = client.post(
        f"/api/inference/{TASK}/forecast",
        json={"model_version": version, "history": history, "horizon": 2},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    predictions = body["predictions"]
    lower = body["lower_bounds"] or [None] * len(predictions)
    upper = body["upper_bounds"] or [None] * len(predictions)
    table = pd.DataFrame(
        {
            "time": body["times"],
            "prediction": predictions,
            "lower": lower,
            "upper": upper,
        }
    )
    lines = table.to_csv(index=False).splitlines()
    assert lines[0] == "time,prediction,lower,upper"
    assert len(table) == 2
    # y = 2t: the next two values after t=40 are 82 and 84.
    assert float(table["prediction"].iloc[0]) == pytest.approx(82.0)
    assert float(table["prediction"].iloc[1]) == pytest.approx(84.0)
    # Uncalibrated bundle: the columns exist and are empty, never fabricated.
    assert table["lower"].isna().all() and table["upper"].isna().all()
