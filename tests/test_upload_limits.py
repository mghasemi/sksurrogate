"""Upload size caps: refuse an oversized body before buffering or parsing it.

The cap is ``SKSURROGATE_MAX_UPLOAD_MB`` (``Settings.max_upload_mb``); the test
sets it to 1 MB so the fixture stays small. The failing-first behavior is 413
with the limit in the message, not a 500 from the parser or an OOM.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("api-upload-limits")
    patcher = pytest.MonkeyPatch()
    patcher.setattr(settings, "root", root)
    settings.ensure()
    yield TestClient(app)
    patcher.undo()


def _frame(rows=60, seed=2):
    rng = np.random.RandomState(seed)
    a = rng.normal(size=rows)
    return pd.DataFrame({"a": a, "b": rng.normal(size=rows), "y": 2.0 * a})


def test_an_oversized_upload_is_refused_with_413(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    payload = b"a,b,y\n" + b"1.0,2.0,3.0\n" * 120_000  # ~2.4 MB
    response = client.post(
        "/api/datasets/too-big/register",
        data={"target": "y", "partition": "train"},
        files={"file": ("big.csv", payload, "text/csv")},
    )
    assert response.status_code == 413, response.text
    assert "1 MB upload limit" in response.json()["detail"]
    assert "SKSURROGATE_MAX_UPLOAD_MB" in response.json()["detail"]


def test_the_inspect_endpoint_uses_the_same_cap(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    payload = b"a,b,y\n" + b"1.0,2.0,3.0\n" * 120_000
    response = client.post(
        "/api/datasets/too-big/inspect",
        files={"file": ("big.csv", payload, "text/csv")},
    )
    assert response.status_code == 413, response.text


def test_a_normal_upload_still_succeeds_under_the_cap(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    frame = _frame()
    response = client.post(
        "/api/datasets/upload-ok/register",
        data={"target": "y", "partition": "train"},
        files={"file": ("small.csv", frame.to_csv(index=False), "text/csv")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["rows"] == len(frame)
