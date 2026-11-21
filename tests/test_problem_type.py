"""Declared problem type: one decision, stored, read by every screen.

Before this module existed the control plane never stored "classification or
regression". The target column was registered, and the family was re-derived
later — from the Quick-start preset (which defaulted to classification), from
whichever estimator was configured, or from the fitted bundle. Screens could
therefore disagree, and a regression task's first experiment could silently run a
classification search.

Contract under test:

* ``infer_problem_type`` is conservative and explainable (bool/object → class,
  integer 0/1 → class, everything else numeric → regression);
* registration stores ``problem_type`` + source + reason, and an explicit
  form/endpoint value overrides the inference with source ``"user"``;
* a temporal task is pinned to regression: inference is re-pinned, and an
  explicit classification declaration is refused (400);
* baselines and monitoring follow the declaration instead of a hard-coded default;
* Quick start without a preset resolves to the declared type.
"""

import importlib.util
import io
import json

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402
from api.problem_type import infer_problem_type  # noqa: E402

REGRESSION_TASK = "pt-regression"
CLASSIFICATION_TASK = "pt-classification"
TEMPORAL_TASK = "pt-temporal"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    return TestClient(app)


def _csv(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _register(client, task, frame, target="y", **form):
    data = {"target": target, "partition": "train"}
    data.update(form)
    response = client.post(
        f"/api/datasets/{task}/register",
        data=data,
        files={"file": ("train.csv", _csv(frame), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _regression_frame(rows=80):
    return pd.DataFrame({"x": np.arange(rows, dtype=float), "y": 2.0 * np.arange(rows)})


def _classification_frame(rows=80):
    labels = np.where(np.arange(rows) % 2 == 0, "alpha", "beta")
    return pd.DataFrame({"x": np.arange(rows, dtype=float), "y": labels})


class TestInference:
    def test_boolean_target_is_classification(self):
        frame = pd.DataFrame({"y": [True, False, True, False]})
        assert infer_problem_type(frame, "y")["problem_type"] == "classification"

    def test_object_target_is_classification(self):
        frame = pd.DataFrame({"y": ["a", "b", "a", "c"]})
        assert infer_problem_type(frame, "y")["problem_type"] == "classification"

    def test_binary_integer_target_is_classification(self):
        frame = pd.DataFrame({"y": [0, 1, 0, 1, 1]})
        inferred = infer_problem_type(frame, "y")
        assert inferred["problem_type"] == "classification"
        assert "2 distinct values" in inferred["reason"]

    def test_float_target_is_regression(self):
        frame = pd.DataFrame({"y": [0.5, 1.25, 3.0, 4.75]})
        inferred = infer_problem_type(frame, "y")
        assert inferred["problem_type"] == "regression"
        assert inferred["cardinality"] == 4

    def test_integer_counts_are_regression(self):
        # > 2 distinct integer values: counts, not classes.
        frame = pd.DataFrame({"y": [0, 3, 7, 9, 12]})
        assert infer_problem_type(frame, "y")["problem_type"] == "regression"

    def test_missing_target_column_raises(self):
        with pytest.raises(ValueError):
            infer_problem_type(pd.DataFrame({"x": [1.0]}), "y")


class TestDeclaration:
    def test_registration_stores_the_inference(self, client):
        payload = _register(client, REGRESSION_TASK, _regression_frame())
        assert payload["problem_type"] == "regression"
        assert payload["problem_type_source"] == "inferred"
        assert payload["problem_type_inferred"] == "regression"
        assert "numeric target" in payload["problem_type_reason"]

        metadata = client.get(f"/api/datasets/{REGRESSION_TASK}").json()
        assert metadata["problem_type"] == "regression"
        assert metadata["problem_type_source"] == "inferred"
        assert metadata["temporal"] == {"declared": False, "spec": None}

    def test_registration_accepts_an_override(self, client):
        payload = _register(
            client, "pt-override", _regression_frame(), problem_type="classification"
        )
        assert payload["problem_type"] == "classification"
        assert payload["problem_type_source"] == "user"
        # The inference is still recorded, so the UI can show the disagreement.
        assert payload["problem_type_inferred"] == "regression"

    def test_binarized_label_stays_classification(self, client):
        # Binarizing stores a float 0/1 code; the dtype rule alone would call that
        # regression, so the declaration must keep the classification meaning.
        payload = _register(
            client, "pt-binarized", _classification_frame(), binarize_label="true"
        )
        assert payload["problem_type"] == "classification"
        assert payload["problem_type_source"] == "inferred"
        assert "binarized" in payload["problem_type_reason"]

    def test_put_and_delete_problem_type(self, client):
        _register(client, REGRESSION_TASK, _regression_frame())

        response = client.put(
            f"/api/datasets/{REGRESSION_TASK}/problem-type",
            json={"problem_type": "classification"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["problem_type"] == "classification"
        assert body["problem_type_source"] == "user"
        assert body["problem_type_inferred"] == "regression"

        reset = client.delete(f"/api/datasets/{REGRESSION_TASK}/problem-type")
        assert reset.status_code == 200, reset.text
        assert reset.json()["problem_type"] == "regression"
        assert reset.json()["problem_type_source"] == "inferred"

    def test_put_rejects_an_unknown_type(self, client):
        _register(client, REGRESSION_TASK, _regression_frame())
        response = client.put(
            f"/api/datasets/{REGRESSION_TASK}/problem-type",
            json={"problem_type": "clustering"},
        )
        assert response.status_code == 422

    def test_retarget_reinfers_the_problem_type(self, client):
        _register(client, REGRESSION_TASK, _regression_frame())
        client.put(
            f"/api/datasets/{REGRESSION_TASK}/problem-type",
            json={"problem_type": "classification"},
        )
        # Re-pointing the target invalidates the override: the new target decides.
        response = client.put(
            f"/api/datasets/{REGRESSION_TASK}/target", json={"target": "x"}
        )
        assert response.status_code == 200, response.text
        assert response.json()["problem_type"] == "regression"
        assert response.json()["problem_type_source"] == "inferred"


class TestTemporalPinning:
    def _temporal_frame(self):
        time = pd.date_range("2026-01-01", periods=60, freq="D").strftime("%Y-%m-%d")
        return pd.DataFrame({"date": time, "y": np.arange(60.0)})

    def test_temporal_inference_is_pinned_to_regression(self, client):
        payload = _register(
            client,
            TEMPORAL_TASK,
            self._temporal_frame(),
            temporal=json.dumps({"time_column": "date", "horizon": 3}),
        )
        assert payload["problem_type"] == "regression"
        assert payload["temporal"]["spec"]["horizon"] == 3

    def test_temporal_binary_label_stays_regression(self, client):
        frame = self._temporal_frame()
        frame["y"] = (np.arange(60) % 2).astype(int)  # inference alone says class
        payload = _register(
            client,
            "pt-temporal-binary",
            frame,
            temporal=json.dumps({"time_column": "date", "horizon": 2}),
        )
        assert payload["problem_type"] == "regression"
        assert payload["problem_type_reason"].startswith("temporal task is regression-shaped")

    def test_explicit_classification_with_temporal_is_refused(self, client):
        response = client.post(
            f"/api/datasets/pt-temporal-refused/register",
            data={
                "target": "y",
                "partition": "train",
                "temporal": json.dumps({"time_column": "date", "horizon": 2}),
                "problem_type": "classification",
            },
            files={"file": ("train.csv", _csv(self._temporal_frame()), "text/csv")},
        )
        assert response.status_code == 400
        assert "regression-shaped" in response.json()["detail"]

    def test_put_classification_on_a_temporal_task_is_refused(self, client):
        _register(
            client,
            TEMPORAL_TASK,
            self._temporal_frame(),
            temporal=json.dumps({"time_column": "date", "horizon": 2}),
        )
        response = client.put(
            f"/api/datasets/{TEMPORAL_TASK}/problem-type",
            json={"problem_type": "classification"},
        )
        assert response.status_code == 400
        assert "regression-shaped" in response.json()["detail"]


class TestDownstreamDefaults:
    def test_baseline_estimator_follows_the_declaration(self, client):
        _register(client, CLASSIFICATION_TASK, _classification_frame())
        response = client.post(f"/api/bundles/{CLASSIFICATION_TASK}/train-baseline", json={})
        assert response.status_code == 200, response.text
        assert response.json()["estimator"] == "logistic_regression"

    def test_baseline_default_is_linear_regression_for_regression(self, client):
        _register(client, REGRESSION_TASK, _regression_frame())
        response = client.post(f"/api/bundles/{REGRESSION_TASK}/train-baseline", json={})
        assert response.status_code == 200, response.text
        assert response.json()["estimator"] == "linear_regression"

    def test_mismatched_baseline_estimator_is_refused(self, client):
        _register(client, CLASSIFICATION_TASK, _classification_frame())
        response = client.post(
            f"/api/bundles/{CLASSIFICATION_TASK}/train-baseline",
            json={"estimator": "linear_regression"},
        )
        assert response.status_code == 400
        assert "declared problem type" in response.json()["detail"]

    def test_monitoring_follows_the_declaration(self, client):
        _register(client, REGRESSION_TASK, _regression_frame())
        # No task_type on purpose: a regression task must not be scored with accuracy.
        response = client.post(
            f"/api/monitoring/{REGRESSION_TASK}/delayed-label-performance",
            json={"predictions": [1.0, 2.0, 3.0], "labels": [1.1, 1.9, 3.2]},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert "mae" in body["metrics"]
        assert "accuracy" not in body["metrics"]

    def test_quick_start_defaults_to_the_declared_type(self, client):
        _register(client, REGRESSION_TASK, _regression_frame())
        from api.routers.experiments import RunExperimentRequest, _declared_preset, _effective_scoring

        assert _declared_preset(REGRESSION_TASK) == "regression"
        request = RunExperimentRequest(max_generations=1)
        assert _effective_scoring(request, _declared_preset(REGRESSION_TASK)) == (
            "neg_root_mean_squared_error"
        )

    def test_undeclared_task_keeps_the_request_shape_error(self, client):
        # No registration → no declaration → the historical 422 contract.
        response = client.post("/api/experiments/never-registered/run", json={"budget": 1})
        assert response.status_code == 422
