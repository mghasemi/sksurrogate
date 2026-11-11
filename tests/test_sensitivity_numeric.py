"""Sensitivity analysis on mixed-type partitions.

Two regressions are pinned here:

* **Non-numeric columns** — a registered churn CSV carries ``Surname`` /
  ``Geography`` / ``Gender`` as object columns. The sensitivity and
  correlation-pruning paths fed the raw matrix to a numeric procedure, which
  raised ``ValueError: could not convert string to float: 'Hargrave'`` (sklearn
  SVR) and the equivalent numpy error from ``corrcoef``. Non-numeric columns are
  now excluded and reported, and an all-string partition answers 400 instead of
  failing inside the fit.
* **SALib version drift** — SALib 1.6 removed the single ``delta`` key and
  returns ``delta_balanced`` / ``delta_raw`` / ``delta_step``. The container is
  built from unpinned requirements, so it had 1.6 while the dev venv still had
  1.5: the same code path returned weights locally and raised ``KeyError:
  'delta'`` in Docker.
"""

import importlib.util
import time

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from SKSurrogate import SensAprx  # noqa: E402
from api import acl, auth  # noqa: E402
from api.config import settings  # noqa: E402
from api.features import NumericFeatureError, numeric_train_xy  # noqa: E402
from api.main import app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "require_auth", False)
    settings.ensure()
    auth.reset_store()
    acl.reset_store()
    return TestClient(app)


def _mixed_frame(rows=120, bad_rows=0):
    rng = np.random.RandomState(0)
    frame = pd.DataFrame(
        {
            "RowNumber": np.arange(rows),
            "Surname": ["Hargrave"] * rows,
            "Geography": rng.choice(["France", "Germany", "Spain"], size=rows),
            "Gender": rng.choice(["Male", "Female"], size=rows),
            "CreditScore": rng.randint(350, 850, size=rows),
            "Age": rng.randint(18, 80, size=rows),
            "Balance": rng.uniform(0, 150000, size=rows),
            "HasCrCard": rng.randint(0, 2, size=rows),
        }
    )
    frame["Exited"] = (frame["Age"] > 40).astype(int)
    for index in range(bad_rows):
        frame.loc[index, "Balance"] = np.nan
    return frame


def _register(client, task, frame, target="Exited"):
    response = client.post(
        "/api/datasets/%s/register" % task,
        data={"target": target, "partition": "train"},
        files={"file": ("mixed.csv", frame.to_csv(index=False), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response


# --------------------------------------------------------------------------- #
# The projection itself
# --------------------------------------------------------------------------- #


def test_projection_excludes_object_columns_and_keeps_booleans():
    projection = numeric_train_xy(_mixed_frame(), "Exited")
    assert projection.columns == ["RowNumber", "CreditScore", "Age", "Balance", "HasCrCard"]
    assert projection.excluded_names == ["Surname", "Geography", "Gender"]
    assert projection.excluded[0]["dtype"] == "object"
    assert projection.X.dtype == np.float64
    assert projection.X.shape == (120, 5)


def test_rows_with_non_finite_values_are_dropped_and_counted():
    projection = numeric_train_xy(_mixed_frame(bad_rows=3), "Exited")
    assert projection.dropped_rows == 3
    assert projection.X.shape[0] == 117
    assert np.isfinite(projection.X).all()


def test_all_string_features_raise_a_numeric_feature_error():
    frame = pd.DataFrame({"a": ["x"] * 10, "b": ["y"] * 10, "y": range(10)})
    with pytest.raises(NumericFeatureError) as excinfo:
        numeric_train_xy(frame, "y")
    assert "need numeric feature columns" in str(excinfo.value)
    assert "a, b" in str(excinfo.value)


def test_non_numeric_target_is_refused():
    frame = pd.DataFrame({"a": range(10), "y": ["no"] * 5 + ["yes"] * 5})
    with pytest.raises(NumericFeatureError) as excinfo:
        numeric_train_xy(frame, "y")
    assert "numeric (encoded) target" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# HTTP: correlation pruning and matrices on a mixed frame
# --------------------------------------------------------------------------- #


def test_correlation_threshold_works_and_reports_exclusions(client):
    _register(client, "mixed", _mixed_frame())
    response = client.post("/api/sensitivity/mixed/correlation-threshold", params={"threshold": 0.7})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["excluded_feature_names"] == ["Surname", "Geography", "Gender"]
    assert "Surname" not in payload["feature_columns"]
    assert set(payload["kept_feature_names"]) | set(payload["dropped_feature_names"]) == set(
        payload["feature_columns"]
    )
    assert payload["rows_dropped"] == 0


def test_correlation_matrix_and_weights_heatmap_use_numeric_columns(client):
    _register(client, "mixed", _mixed_frame())
    matrix = client.get("/api/sensitivity/mixed/correlation-matrix")
    assert matrix.status_code == 200, matrix.text
    assert "Surname" not in matrix.json()["labels"]

    weights = client.get("/api/sensitivity/mixed/weights-heatmap")
    assert weights.status_code == 200, weights.text
    payload = weights.json()
    assert "Surname" not in payload["labels"]
    assert [entry["column"] for entry in payload["excluded_columns"]] == [
        "Surname",
        "Geography",
        "Gender",
    ]


def test_all_string_partition_answers_400_not_a_crashing_job(client):
    frame = pd.DataFrame({"name": ["a", "b", "c", "d"] * 5, "city": ["x", "y"] * 10})
    frame["y"] = range(20)
    _register(client, "textual", frame, target="y")
    response = client.post("/api/sensitivity/textual/correlation-threshold")
    assert response.status_code == 400
    assert "numeric feature columns" in response.json()["detail"]


def test_sensitivity_job_completes_on_a_mixed_partition(client):
    """End-to-end: the job result carries weights for the numeric block only."""
    _register(client, "mixedjob", _mixed_frame())
    submitted = client.post(
        "/api/sensitivity/mixedjob/run",
        json={"method": "morris", "n_features_to_select": 3, "train_partition": "train"},
    )
    assert submitted.status_code == 200, submitted.text
    job_id = submitted.json()["job_id"]
    record = None
    deadline = time.time() + 300
    while time.time() < deadline:
        record = client.get("/api/jobs/%s" % job_id).json()
        if record["status"] in ("completed", "failed"):
            break
        time.sleep(1)
    assert record is not None and record["status"] == "completed", (record or {}).get("error")
    result = record["result"]
    assert result["feature_columns"] == ["RowNumber", "CreditScore", "Age", "Balance", "HasCrCard"]
    assert [entry["column"] for entry in result["evaluation_protocol"]["excluded_columns"]] == [
        "Surname",
        "Geography",
        "Gender",
    ]
    assert len(result["weights"]) == len(result["feature_columns"])
    assert len(result["top_feature_names"]) == 3


# --------------------------------------------------------------------------- #
# SALib compatibility
# --------------------------------------------------------------------------- #


def test_delta_estimator_key_is_resolved_across_salib_versions():
    analyzer = SensAprx(method="delta-mmnt")
    assert analyzer._delta_index({"delta": [1.0]}) == [1.0]
    assert analyzer.delta_estimator_ == "delta"

    analyzer = SensAprx(method="delta-mmnt")
    assert analyzer._delta_index({"delta_balanced": [2.0], "delta_raw": [3.0]}) == [2.0]
    assert analyzer.delta_estimator_ == "delta_balanced"

    with pytest.raises(RuntimeError) as excinfo:
        SensAprx(method="delta-mmnt")._delta_index({"S1": [0.0]})
    assert "available keys" in str(excinfo.value)


def test_delta_analysis_runs_on_the_installed_salib():
    """The real path: whatever SALib is installed must yield one weight per feature."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 4))
    y = 2 * X[:, 0] + X[:, 1] + 0.1 * rng.normal(size=200)
    analyzer = SensAprx(n_features_to_select=2, method="delta-mmnt").fit(X, y)
    assert analyzer.weights_ is not None and len(analyzer.weights_) == 4
    assert analyzer.delta_estimator_ in {"delta", "delta_balanced", "delta_raw", "delta_step"}
    assert {int(index) for index in analyzer.top_features_} == {0, 1}
