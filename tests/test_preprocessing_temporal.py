"""Duplicate and encoding-preview endpoints on temporal tasks.

Both endpoints describe the *feature* view of a partition. A temporal task has no
raw feature columns (time/series are excluded by the contract, the target is the
label), so they used to answer 400 — "No feature columns available to define
duplicate rows" and "No active feature columns to encode" — for data that is
perfectly analysable.

Contract under test:

* duplicates are judged on the temporal row identity (``time`` or ``series +
  time``), reported via ``key_kind``, with a warning that registration already
  enforces that key; an extra raw feature column is still used as the key when one
  exists;
* the encoding preview shows the generated forecast design, all numeric
  (``kind: generated``, with builder provenance), and never claims a one-hot plan;
* a tabular task keeps its previous behaviour exactly.
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

TASK = "preprocessing-temporal"
TABULAR = "preprocessing-tabular"
HORIZON = 4


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    return TestClient(app)


def _csv(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _temporal_frame(rows=90, extra_feature=False):
    t = pd.date_range("2026-01-01", periods=rows, freq="D")
    frame = pd.DataFrame(
        {"date": t.strftime("%Y-%m-%d"), "y": 10.0 + 0.5 * np.arange(rows)}
    )
    if extra_feature:
        frame.insert(1, "temperature", np.linspace(-5.0, 25.0, rows))
    return frame


def _register(client, frame, task, temporal=None):
    data = {"target": "y", "partition": "train"}
    if temporal is not None:
        data["temporal"] = json.dumps(temporal)
    response = client.post(
        f"/api/datasets/{task}/register",
        data=data,
        files={"file": ("train.csv", _csv(frame), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_duplicates_use_the_temporal_row_identity(client):
    _register(
        client,
        _temporal_frame(),
        TASK,
        {"time_column": "date", "horizon": HORIZON, "freq": "D"},
    )
    response = client.get(f"/api/datasets/{TASK}/train/duplicates")
    assert response.status_code == 200, response.text
    report = response.json()

    assert report["key_kind"] == "temporal_identity"
    assert report["features"] == ["date"]
    assert report["temporal_columns"] == ["date"]
    assert report["duplicate_rows"] == 0
    assert report["rows_removed"] == 0
    warning = " ".join(report["warnings"]).lower()
    assert "temporal row identity" in warning
    assert "registration" in warning


def test_duplicates_use_an_extra_raw_feature_when_one_exists(client):
    _register(
        client,
        _temporal_frame(extra_feature=True),
        TASK,
        {"time_column": "date", "horizon": HORIZON, "freq": "D"},
    )
    response = client.get(f"/api/datasets/{TASK}/train/duplicates")
    assert response.status_code == 200, response.text
    report = response.json()
    # A raw feature column exists, so the feature-key semantics is unchanged.
    assert report["key_kind"] == "features"
    assert report["features"] == ["temperature"]


def test_duplicates_accept_the_temporal_columns_as_an_explicit_key(client):
    _register(
        client,
        _temporal_frame(),
        TASK,
        {"time_column": "date", "horizon": HORIZON, "freq": "D"},
    )
    # Explicitly asking for the time column must not be refused as "ignored".
    response = client.get(
        f"/api/datasets/{TASK}/train/duplicates", params={"columns": ["date"]}
    )
    assert response.status_code == 200, response.text
    assert response.json()["features"] == ["date"]


def test_deduplicate_preview_reports_the_temporal_key(client):
    _register(
        client,
        _temporal_frame(),
        TASK,
        {"time_column": "date", "horizon": HORIZON, "freq": "D"},
    )
    response = client.post(
        f"/api/datasets/{TASK}/deduplicate",
        json={"partition": "train", "preview_only": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["key_kind"] == "temporal_identity"
    assert body["features"] == ["date"]
    assert body["rows_removed"] == 0


def test_encoding_preview_shows_the_generated_design(client):
    _register(
        client,
        _temporal_frame(),
        TASK,
        {"time_column": "date", "horizon": HORIZON, "freq": "D"},
    )
    response = client.get(f"/api/datasets/{TASK}/train/encoding-preview")
    assert response.status_code == 200, response.text
    report = response.json()

    assert report["feature_source"] == "temporal_design"
    assert report["input_features"] == ["y__lag_%d" % HORIZON]
    assert report["output_features"] == report["input_features"]
    entry = report["per_column"][0]
    assert entry["kind"] == "generated"
    assert entry["op"] == "lag" and entry["shift"] == HORIZON
    assert entry["dtype"] == "float64"
    # Numeric by construction: no categorical vocabulary is claimed.
    assert entry["output_feature_count"] == 1
    assert "generated" in entry["encoding"].lower()
    assert len(report["preview"]) > 0
    assert any("generated forecast design" in note for note in report["notes"])


def test_tabular_endpoints_are_unchanged(client):
    frame = pd.DataFrame(
        {
            "x": np.arange(60.0),
            "cat": ["a" if i % 2 else "b" for i in range(60)],
            "y": np.arange(60.0) * 2.0,
        }
    )
    _register(client, frame, TABULAR)

    duplicates = client.get(f"/api/datasets/{TABULAR}/train/duplicates")
    assert duplicates.status_code == 200, duplicates.text
    body = duplicates.json()
    assert body["key_kind"] == "features"
    assert body["features"] == ["x", "cat"]

    preview = client.get(f"/api/datasets/{TABULAR}/train/encoding-preview")
    assert preview.status_code == 200, preview.text
    report = preview.json()
    assert "feature_source" not in report
    assert report["input_features"] == ["x", "cat"]
    kinds = {entry["column"]: entry["kind"] for entry in report["per_column"]}
    assert kinds == {"x": "numeric", "cat": "categorical"}
