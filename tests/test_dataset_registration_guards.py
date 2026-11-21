"""Registration guards for degenerate uploads.

Two real 500s found while investigating a ``PUT /datasets/{task}/temporal`` failure:

* an **empty CSV** (header only) crashed type deduction with ``ZeroDivisionError``
  inside ``DataPreprocess.is_categorical``;
* a **target column with no observed values** was accepted, producing a task nothing
  can be modelled on.

Both are user input, not server faults, so they answer 400 with an explanation. The
underlying library division is guarded too, because an all-missing *feature* column is
legitimate: it must be classified (as unusable), not crash the upload.
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


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    return TestClient(app, raise_server_exceptions=False)


def _csv(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _register(client, task, frame, target="y"):
    return client.post(
        f"/api/datasets/{task}/register",
        data={"target": target, "partition": "train"},
        files={"file": ("train.csv", _csv(frame), "text/csv")},
    )


def _frame(rows=20):
    return pd.DataFrame(
        {"x": np.arange(rows, dtype=float), "y": 2.0 * np.arange(rows, dtype=float)}
    )


def test_empty_csv_is_a_400_not_a_crash(client):
    response = _register(client, "empty-upload", _frame().head(0))
    assert response.status_code == 400, response.text
    assert "no data rows" in response.json()["detail"]


def test_all_missing_target_is_refused(client):
    response = _register(client, "no-target", _frame().assign(y=[np.nan] * 20))
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "no observed values" in detail
    assert "'y'" in detail


def test_all_missing_feature_column_still_registers(client):
    # An entirely missing feature is a data-quality fact, not an error: it must be
    # typed (as unusable) instead of dividing by a population of zero.
    response = _register(client, "missing-feature", _frame().assign(extra=[np.nan] * 20))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rows"] == 20
    assert "extra" in body["columns"]
