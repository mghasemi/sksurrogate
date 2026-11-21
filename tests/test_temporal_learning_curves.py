"""Learning curves on a temporal task are fitted on the generated design.

Before this path existed, ``compute_learning_curve`` sliced the raw partition by
the bundle's recorded schema. A temporal bundle's schema names the raw history
frame (time/series columns included), while ``load_partition`` drops exactly those
columns, so *every* bundle of a temporal task failed with

    Bundle '<id>' expects feature columns absent from partition 'main': Month

and the Evaluation page reported all of them as "Skipped N bundle(s)".

Contract under test:

* a temporal bundle's curve is computed on the generated one-step forecast design
  through its bare estimator (``feature_source: temporal_design``), so no bundle
  is skipped;
* the walk-forward folds are honoured as *training lengths*: the curve must reach
  the longest window the splitter offers, not stop at the first origin (sklearn's
  own ``learning_curve`` anchors sizes on the smallest fold);
* scores are finite and the curve grows with the training length.
"""

import importlib.util
import io
import json
import time

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402

TASK = "temporal-curves"
HORIZON = 2


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    return TestClient(app)


def _frame(rows=90):
    t = np.arange(1, rows + 1, dtype=float)
    y = 0.4 * t + 3.0 * np.sin(2.0 * np.pi * t / 4)
    return pd.DataFrame({"time": t, "y": y})


def _register(client):
    buffer = io.StringIO()
    _frame().to_csv(buffer, index=False)
    response = client.post(
        "/api/datasets/%s/register" % TASK,
        data={
            "target": "y",
            "partition": "train",
            "temporal": json.dumps({"time_column": "time", "horizon": HORIZON}),
        },
        files={"file": ("train.csv", buffer.getvalue().encode(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _baseline(client):
    response = client.post(
        "/api/bundles/%s/train-baseline" % TASK,
        json={"estimator": "linear_regression", "seasonality": 4},
    )
    assert response.status_code == 200, response.text
    return response.json()["model_version"]


def _wait(client, job_id, attempts=240):
    for _ in range(attempts):
        record = client.get("/api/jobs/" + job_id).json()
        if record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.25)
    raise AssertionError("job did not finish")


def test_learning_curves_run_on_the_design_and_reach_the_longest_window(client):
    _register(client)
    version = _baseline(client)

    response = client.get(
        "/api/evaluation/%s/learning-curves" % TASK,
        params={"scoring": "r2", "n_points": 5, "model_versions": version},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["errors"] == []

    curve = payload["series"][0]
    assert curve["feature_source"] == "temporal_design"
    sizes = curve["train_sizes"]
    assert sizes == sorted(set(sizes))
    # 90 raw rows minus the lag/horizon rows of the one-step design; the curve must
    # cover most of that, not stop at the splitter's first origin.
    assert sizes[-1] >= 60
    finite = [value for value in curve["validation_scores_mean"] if value is not None]
    assert len(finite) == len(sizes)
    assert all(np.isfinite(finite))
    assert sizes[0] < sizes[-1]


def test_the_report_learning_curves_no_longer_skip_temporal_bundles(client):
    """The explainability report shares compute_learning_curve: no per-curve errors."""
    from api.routers import reports

    _register(client)
    version = _baseline(client)
    diagnostics = reports._evaluation_diagnostics(TASK, version, [])
    learning = diagnostics["Learning curves"]
    assert learning["train_partition"] == "train"
    assert learning["curves"]
    # One curve per regression metric; every one must have run on the design.
    assert all(curve.get("error") is None for curve in learning["curves"])
    assert {curve["series"].get("feature_source") for curve in learning["curves"]} == {
        "temporal_design"
    }
    assert all(
        any(value is not None for value in curve["series"]["validation_scores_mean"])
        for curve in learning["curves"]
    )
