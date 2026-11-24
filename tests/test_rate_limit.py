"""Rate limiting: bucket arithmetic, route classification and the 429 surface.

The limiter is off by default (single-process local deployment), so the
integration test turns it on with a small job limit and drives the real
``train-baseline`` route. Buckets are per process and keyed by client address;
``LIMITER.reset()`` keeps these tests independent of each other.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api import ratelimit  # noqa: E402
from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("api-rate-limit")
    patcher = pytest.MonkeyPatch()
    patcher.setattr(settings, "root", root)
    settings.ensure()
    yield TestClient(app)
    patcher.undo()


@pytest.fixture(autouse=True)
def reset_limiter():
    ratelimit.LIMITER.reset()
    yield
    ratelimit.LIMITER.reset()


def _register(client, task="rl-demo", rows=40, seed=3):
    rng = np.random.RandomState(seed)
    a = rng.normal(size=rows)
    frame = pd.DataFrame({"a": a, "b": rng.normal(size=rows), "y": 2.0 * a})
    response = client.post(
        "/api/datasets/%s/register" % task,
        data={"target": "y", "partition": "train"},
        files={"file": ("t.csv", frame.to_csv(index=False), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return task


def test_token_bucket_refills_over_time():
    bucket = ratelimit.TokenBucket(per_minute=60, burst=2, now=0.0)
    assert bucket.take(0.0)[0] is True
    assert bucket.take(0.0)[0] is True
    allowed, retry = bucket.take(0.0)
    assert allowed is False
    assert retry == pytest.approx(1.0)
    allowed, retry = bucket.take(0.5)
    assert allowed is False
    assert retry == pytest.approx(0.5)
    assert bucket.take(1.0)[0] is True


def test_route_classification_limits_work_not_observation():
    assert ratelimit.classify("POST", "/api/experiments/demo/run") == "jobs"
    assert ratelimit.classify("POST", "/api/bundles/demo/train-baseline") == "jobs"
    assert ratelimit.classify("POST", "/api/evaluation/demo/nested-cv") == "jobs"
    assert ratelimit.classify("POST", "/api/inference/demo/predict") == "inference"
    assert ratelimit.classify("POST", "/api/serving/demo/predict") == "inference"
    # Reading or polling must never be throttled behind the work it observes.
    assert ratelimit.classify("GET", "/api/experiments/demo/run") is None
    assert ratelimit.classify("GET", "/api/inference/demo/predict") is None
    assert ratelimit.classify("POST", "/api/jobs/abc/stop") is None
    assert ratelimit.classify("POST", "/api/serving/demo") is None


def test_job_submissions_are_limited_with_retry_after(client, monkeypatch):
    task = _register(client)
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_jobs_per_minute", 2)
    body = {"estimator": "random_forest_regressor", "validation_partition": None}
    for _ in range(2):
        response = client.post("/api/bundles/%s/train-baseline" % task, json=body)
        assert response.status_code == 200, response.text
    refused = client.post("/api/bundles/%s/train-baseline" % task, json=body)
    assert refused.status_code == 429, refused.text
    assert refused.headers["retry-after"].isdigit()
    assert "Too many jobs requests" in refused.json()["detail"]
    # A read is not throttled by the exhausted bucket.
    assert client.get("/api/bundles/%s" % task).status_code == 200


def test_limiting_is_off_by_default(client, monkeypatch):
    task = _register(client, task="rl-off")
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    monkeypatch.setattr(settings, "rate_limit_jobs_per_minute", 1)
    body = {"estimator": "random_forest_regressor", "validation_partition": None}
    for _ in range(3):
        response = client.post("/api/bundles/%s/train-baseline" % task, json=body)
        assert response.status_code == 200, response.text
