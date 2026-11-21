"""Experiments on a temporal task run on the generated forecast design.

A temporal partition has no feature column of its own: the time column is excluded
by the temporal contract and the target is the label. The tabular search therefore
saw an empty matrix and every candidate died with the opaque
``ValueError: at least one array or dtype is required``. The search now builds the
same one-step design sensitivity and the baselines use, and the bundle it produces
carries a ``ForecastFeatureBuilder`` so it can serve forecasts recursively.
"""

import time

import numpy as np
import pandas as pd
import pytest

if __import__("importlib").util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402

SERIES = 48


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    return TestClient(app, raise_server_exceptions=False)


def _series():
    index = pd.date_range("2001-01-01", periods=SERIES, freq="MS")
    values = 100 + 10 * np.sin(np.arange(SERIES) / 6 * 2 * np.pi) + np.arange(SERIES) * 1.5
    return pd.DataFrame({"Month": index.strftime("%Y-%m-%d"), "Sales": values})


def _register(client, task, frame=None, target="Sales", partition="main"):
    frame = _series() if frame is None else frame
    response = client.post(
        "/api/datasets/%s/register" % task,
        data={"target": target, "partition": partition},
        files={"file": ("series.csv", frame.to_csv(index=False).encode(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response


def _declare_temporal(client, task, horizon=4):
    response = client.put(
        "/api/datasets/%s/temporal" % task,
        json={"time_column": "Month", "freq": "MS", "horizon": horizon},
    )
    assert response.status_code == 200, response.text


def _wait(client, job_id, attempts=240):
    for _ in range(attempts):
        record = client.get("/api/jobs/" + job_id).json()
        if record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def _submit_experiment(client, task):
    body = {
        "preset": "regression",
        "strategy": "exhaustive",
        "length": 1,
        "random_state": 0,
        "scoring": "neg_root_mean_squared_error",
        "train_partition": "main",
    }
    response = client.post("/api/experiments/%s/run" % task, json=body)
    assert response.status_code == 200, response.text
    return _wait(client, response.json()["job_id"])


def test_the_search_runs_on_the_one_step_design(client):
    from SKSurrogate import load_bundle, serve_forecast

    _register(client, "tem-exp")
    _declare_temporal(client, "tem-exp")
    record = _submit_experiment(client, "tem-exp")
    assert record["status"] == "completed", record.get("error")

    result = record["result"]
    assert result["model_version"]
    bundle = load_bundle(settings.task_bundles_dir("tem-exp") / (result["model_version"] + ".bundle"))
    # The served bundle is a forecast pipeline: builder first, then the estimator.
    block = bundle.temporal or {}
    assert block["strategy"] == "recursive"
    assert block["target"] == "Sales"
    assert list(bundle.schema) == ["Month", "Sales"]
    # The serving contract other routes rely on: the builder is the "features"
    # step and the selected estimator is "model".
    assert [name for name, _ in bundle.model.steps] == ["features", "model"]

    history = _series().tail(12)
    forecast = serve_forecast(bundle, history)
    assert len(forecast.means) == 4
    assert np.isfinite(forecast.means).all()


def test_the_bundle_forecasts_through_the_inference_endpoint(client):
    from SKSurrogate import load_bundle

    _register(client, "tem-serve")
    _declare_temporal(client, "tem-serve")
    record = _submit_experiment(client, "tem-serve")
    assert record["status"] == "completed", record.get("error")
    version = record["result"]["model_version"]
    # Serving needs the bundle on disk in the task directory; the job stored it there.
    assert (settings.task_bundles_dir("tem-serve") / (version + ".bundle")).exists()

    response = client.post(
        "/api/inference/tem-serve/forecast",
        json={
            "model_version": version,
            "history": _series().tail(12).to_dict(orient="records"),
            "horizon": 4,
        },
    )
    assert response.status_code == 200, response.text
    assert len(response.json().get("points") or response.json().get("times") or []) == 4


def test_a_featureless_tabular_partition_answers_an_actionable_400(client):
    """The opaque sklearn error must not be the first thing a user sees."""
    _register(client, "flat-task")
    response = client.put("/api/datasets/flat-task/ignored-columns", json={"columns": ["Month"]})
    assert response.status_code == 200, response.text

    record = _submit_experiment(client, "flat-task")
    assert record["status"] == "failed"
    assert "has no feature columns" in record["error"]
    assert "Month" in record["error"]


def test_the_experiment_bundle_is_backtested_recursively(client):
    """The rolling-origin backtest must refit the bare estimator, not the pipeline.

    Fitting a temporal bundle's pipeline on the backtest's own design matrix fails
    with ``ForecastFeatureBuilder is missing required column(s)`` — the builder
    wants raw history. The experiment bundle's one-step model also has to be scored
    on its own labels, i.e. recursively.
    """
    _register(client, "tem-bt")
    _declare_temporal(client, "tem-bt")
    experiment = _submit_experiment(client, "tem-bt")
    assert experiment["status"] == "completed", experiment.get("error")

    response = client.post(
        "/api/evaluation/tem-bt/backtest",
        json={
            "model_version": experiment["result"]["model_version"],
            "horizon": 4,
            "n_splits": 3,
            "scoring": "neg_mean_absolute_error",
        },
    )
    assert response.status_code == 200, response.text
    record = _wait(client, response.json()["job_id"])
    assert record["status"] == "completed", record.get("error")
    result = record["result"]
    assert result["config"]["strategy"] == "recursive"
    assert result["pooled"]["mae"] is not None


def test_explanations_run_on_the_generated_design(client):
    """Explain/ALE/row act on the design, not on a 0-column raw partition.

    The reported failure: ``Found array with 0 feature(s) (shape=(36, 0))`` — a
    temporal partition has no raw feature column, so explanations must use the
    same generated matrix the search and the backtest see.
    """
    _register(client, "tem-explain")
    _declare_temporal(client, "tem-explain")
    experiment = _submit_experiment(client, "tem-explain")
    assert experiment["status"] == "completed", experiment.get("error")
    version = experiment["result"]["model_version"]

    explain = client.post(
        f"/api/evaluation/tem-explain/{version}/explain", json={"top_k": 1, "max_rows": 200}
    )
    assert explain.status_code == 200, explain.text
    payload = explain.json()
    assert payload["feature_source"] == "temporal_design"
    assert payload["importance"]
    assert "__lag_" in payload["importance"][0]["feature"]
    assert payload["errors"] == []

    ale = client.post(f"/api/evaluation/tem-explain/{version}/ale", json={"top_k": 1})
    assert ale.status_code == 200, ale.text
    assert ale.json()["feature_source"] == "temporal_design"

    row = client.post(
        f"/api/evaluation/tem-explain/{version}/explain/row",
        json={"row_index": 0, "orders": 5},
    )
    assert row.status_code == 200, row.text
    row_payload = row.json()
    assert row_payload["feature_source"] == "temporal_design"
    # The one-feature design has exactly one distinct order; asking for five is
    # clamped and reported, not refused.
    assert row_payload["orders"]["count"] == 1
    assert row_payload["orders"]["requested"] == 5
    assert row_payload["orders"]["available"] == 1


def test_nested_cv_runs_on_the_design_with_order_respecting_folds(client):
    """Nested CV of a temporal bundle scores the one-step design, not raw columns.

    The reported failure: the bundle's pipeline (feature builder first) received a
    design matrix and died with "ForecastFeatureBuilder works on pandas
    DataFrames". The folds must also respect observation order, and a stacking
    wrapper's inner CV has to survive the small outer/inner blocks.
    """
    _register(client, "tem-ncv")
    _declare_temporal(client, "tem-ncv")
    experiment = _submit_experiment(client, "tem-ncv")
    assert experiment["status"] == "completed", experiment.get("error")

    response = client.post(
        "/api/evaluation/tem-ncv/nested-cv",
        json={
            "model_version": experiment["result"]["model_version"],
            "inner_cv": 2,
            "outer_cv": 2,
            "scoring": "neg_mean_absolute_error",
        },
    )
    assert response.status_code == 200, response.text
    record = _wait(client, response.json()["job_id"])
    assert record["status"] == "completed", record.get("error")
    result = record["result"]
    assert len(result["outer_scores"]) == 2
    assert result["mean_outer_score"] is not None
    assert result["evaluation_protocol"]["train_partition"] == "main"


def test_nested_cv_refuses_a_shuffled_splitter_on_a_temporal_task(client):
    _register(client, "tem-ncv-bad")
    _declare_temporal(client, "tem-ncv-bad")
    experiment = _submit_experiment(client, "tem-ncv-bad")
    assert experiment["status"] == "completed", experiment.get("error")
    response = client.post(
        "/api/evaluation/tem-ncv-bad/nested-cv",
        json={
            "model_version": experiment["result"]["model_version"],
            "outer_cv": {"type": "KFold", "n_splits": 2},
        },
    )
    assert response.status_code == 400
    assert "order-respecting" in response.json()["detail"]


def test_pipeline_optimization_points_a_temporal_task_at_the_design_search(client):
    _register(client, "tem-pipe")
    _declare_temporal(client, "tem-pipe")
    response = client.post(
        "/api/experiments/tem-pipe/optimize-pipeline",
        json={"seq": ["sklearn.tree.DecisionTreeRegressor"], "scoring": "neg_root_mean_squared_error"},
    )
    assert response.status_code == 200, response.text
    record = _wait(client, response.json()["job_id"])
    assert record["status"] == "failed"
    assert "is temporal" in record["error"]
    assert "Experiments page" in record["error"]
