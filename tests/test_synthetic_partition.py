"""Synthetic sampling resolves the task's own partition instead of assuming "train".

``POST /api/datasets/{task}/synthetic`` defaulted the partition to the literal
``"train"``. Any task whose data was registered under another name — e.g. a task
whose single partition is called like the task itself — answered
``404 No dataset partition 'train' stored`` even though its data was there, and the
web UI showed the real partition in the select while still sending ``"train"``.

The default now follows what is actually registered: ``train`` when it exists,
otherwise the first registered partition. An explicit wrong name still 404s, but the
message names the partitions that do exist.
"""

import importlib.util
import io

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402

TASK = "synthetic-partitions"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    return TestClient(app)


def _frame(rows=60):
    return pd.DataFrame(
        {
            "x": np.arange(rows, dtype=float),
            "group": np.where(np.arange(rows) % 2 == 0, "a", "b"),
            "y": 2.0 * np.arange(rows),
        }
    )


def _register(client, partition, task=TASK):
    buffer = io.StringIO()
    _frame().to_csv(buffer, index=False)
    response = client.post(
        f"/api/datasets/{task}/register",
        data={"target": "y", "partition": partition},
        files={"file": ("train.csv", buffer.getvalue().encode(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_omitted_partition_uses_the_registered_one(client):
    # The user's case: the only partition is not called "train".
    _register(client, "TSTrain")
    response = client.post(f"/api/datasets/{TASK}/synthetic", json={"num": 5})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["partition"] == "TSTrain"
    assert body["rows"] == 5


def test_train_is_still_preferred_when_present(client):
    _register(client, "train")
    response = client.post(f"/api/datasets/{TASK}/synthetic", json={"num": 5})
    assert response.status_code == 200, response.text
    assert response.json()["partition"] == "train"


def test_explicit_partition_is_honoured(client):
    _register(client, "TSTrain")
    response = client.post(
        f"/api/datasets/{TASK}/synthetic", json={"num": 5, "partition": "TSTrain"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["partition"] == "TSTrain"


def test_unknown_partition_names_the_available_ones(client):
    _register(client, "TSTrain")
    response = client.post(
        f"/api/datasets/{TASK}/synthetic", json={"num": 5, "partition": "holdout"}
    )
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "'holdout'" in detail
    assert "TSTrain" in detail  # the actionable part: what does exist


def test_unregistered_task_still_reports_no_dataset(client):
    response = client.post("/api/datasets/never-registered/synthetic", json={"num": 5})
    assert response.status_code == 404
    assert "No dataset registered" in response.json()["detail"]
