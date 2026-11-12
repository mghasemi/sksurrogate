"""Data-preprocessing endpoints: overview, missingness and materialized imputation."""

import importlib.util
import io
import json

import pandas as pd
import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api import logs  # noqa: E402
from api.config import settings  # noqa: E402
from api.deps import open_tracker  # noqa: E402
from api.main import app  # noqa: E402

TASK = "prep-task"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    logs.store.clear()
    return TestClient(app)


def _frame():
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, None, 4.0, 5.0, None, 7.0, 8.0],
            "z": [1, 2, 3, 4, 5, 6, 7, 8],
            "cat": ["a", "b", None, "a", None, "b", "a", "b"],
            "y": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, None],
        }
    )


def _register(client, frame=None):
    buffer = io.StringIO()
    (frame if frame is not None else _frame()).to_csv(buffer, index=False)
    response = client.post(
        f"/api/datasets/{TASK}/register",
        data={"target": "y", "partition": "train"},
        files={"file": ("train.csv", buffer.getvalue().encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200, response.text


def _preprocessing(client, partition="train"):
    response = client.get(f"/api/datasets/{TASK}/{partition}/preprocessing")
    assert response.status_code == 200, response.text
    return response.json()


def _impute(client, **body):
    defaults = {"partition": "train", "method": "mean"}
    defaults.update(body)
    return client.post(f"/api/datasets/{TASK}/impute", json=defaults)


def test_preprocessing_overview_reports_types_missingness_and_preview(client):
    _register(client)
    payload = _preprocessing(client)
    assert payload["rows"] == 8
    assert payload["target"] == "y"
    assert payload["columns"] == ["x", "z", "cat", "y"]
    assert payload["deduced_types"]["x"] == "float64"

    columns = {entry["column"]: entry for entry in payload["missing"]["columns"]}
    assert columns["x"]["missing"] == 2
    assert columns["x"]["missing_pct"] == pytest.approx(0.25)
    assert columns["cat"]["missing"] == 2
    assert columns["z"]["missing"] == 0
    assert columns["y"]["is_target"] is True
    assert payload["missing"]["target_missing"] == 1
    assert payload["missing"]["rows_with_any_missing"] == 3
    assert payload["missing"]["complete_pct"] == pytest.approx(0.625)

    # Quality card data and the row-level missingness pattern come from the profile.
    assert payload["quality"]["constant_features"] == []
    assert payload["quality"]["duplicate_rows"]["count"] == 0
    assert payload["missingness_patterns"]["patterns"]
    assert len(payload["preview"]) == 8
    method_names = {method["name"] for method in payload["methods"]}
    assert {"drop_rows", "mean", "median", "most_frequent", "constant", "knn", "iterative"} <= method_names


def test_preview_only_imputation_writes_nothing(client):
    _register(client)
    before = (settings.task_dataset_dir(TASK) / "train.csv").read_text()
    response = _impute(client, method="median", columns=["x"], preview_only=True)
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["written"] is False
    assert report["columns"] == ["x"]
    assert report["missing_before"]["x"] == 2
    assert report["missing_after"]["x"] == 0
    assert report["cells_imputed"] >= 2
    assert not (settings.task_dataset_dir(TASK) / "train_imputed.csv").exists()
    assert (settings.task_dataset_dir(TASK) / "train.csv").read_text() == before


def test_saving_imputation_creates_new_partition_and_history(client):
    _register(client)
    source = (settings.task_dataset_dir(TASK) / "train.csv").read_text()
    response = _impute(client, method="median", columns=["x"], output_partition="train_imputed")
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["written"] is True
    assert report["output_partition"] == "train_imputed"
    assert report["dataset_fingerprint"]

    imputed_path = settings.task_dataset_dir(TASK) / "train_imputed.csv"
    assert imputed_path.exists()
    imputed = pd.read_csv(imputed_path)
    assert imputed["x"].isna().sum() == 0
    # The target is never imputed.
    assert imputed["y"].isna().sum() == 1
    # The source partition is untouched.
    assert (settings.task_dataset_dir(TASK) / "train.csv").read_text() == source

    metadata = client.get(f"/api/datasets/{TASK}").json()
    assert "train_imputed" in metadata["partitions"]
    with open_tracker(TASK) as tracker:
        metadata = tracker.GetMetadata()
    history = metadata["imputation_history"]
    assert history[-1]["output_partition"] == "train_imputed"
    assert history[-1]["method"] == "median"
    assert history[-1]["actor"]

    # The derived partition is visible; cat and target remain missing because the
    # imputation selected only x.
    derived = _preprocessing(client, "train_imputed")
    assert derived["missing"]["missing_cells"] == 3


def test_most_frequent_and_drop_rows(client):
    _register(client)
    cat = _impute(client, method="most_frequent", columns=["cat"], preview_only=True).json()
    assert cat["missing_after"]["cat"] == 0
    assert cat["categorical_shift"]["cat"]["after"]

    dropped = _impute(client, method="drop_rows", columns=["x", "cat"], preview_only=True).json()
    assert dropped["rows_after"] == 5
    assert dropped["rows_dropped"] == 3
    assert dropped["missing_after"]["x"] == 0 and dropped["missing_after"]["cat"] == 0


def test_numeric_only_methods_validate_selected_columns(client):
    _register(client)
    for method in ("mean", "median", "knn", "iterative"):
        response = _impute(client, method=method, columns=["cat"], preview_only=True)
        assert response.status_code == 400, response.text
        assert "requires numeric columns" in response.json()["detail"]

    knn = _impute(client, method="knn", columns=["x"], params={"n_neighbors": 2}, preview_only=True)
    assert knn.status_code == 200, knn.text
    assert knn.json()["missing_after"]["x"] == 0
    iterative = _impute(client, method="iterative", columns=["x"], params={"max_iter": 2}, preview_only=True)
    assert iterative.status_code == 200, iterative.text


def test_target_ignored_unknown_and_output_collisions_are_rejected(client):
    _register(client)
    target = _impute(client, method="mean", columns=["y"]).json()
    assert "target column" in target["detail"].lower()

    # Mark z ignored, then refuse to impute it explicitly.
    ignored = client.put(
        f"/api/datasets/{TASK}/ignored-columns", json={"columns": ["z"]}
    )
    assert ignored.status_code == 200
    ignored_attempt = _impute(client, method="mean", columns=["z"]).json()
    assert "ignored" in ignored_attempt["detail"].lower()

    unknown = _impute(client, method="mean", columns=["nope"]).json()
    assert "unknown column" in unknown["detail"].lower()

    same = _impute(client, method="mean", output_partition="train")
    assert same.status_code == 400
    assert "must differ" in same.json()["detail"]

    first = _impute(client, method="mean", columns=["x"], output_partition="copy")
    assert first.status_code == 200
    collision = _impute(client, method="mean", columns=["x"], output_partition="copy")
    assert collision.status_code == 409
    assert "already exists" in collision.json()["detail"]


def test_target_missing_rows_can_be_dropped(client):
    _register(client)
    response = _impute(
        client, method="most_frequent", drop_missing_target=True, preview_only=True
    )
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["target_missing_before"] == 1
    assert report["target_missing_after"] == 0
    assert report["rows_after"] == 7


def test_encoding_preview_reports_fold_local_matrix(client):
    _register(client)
    response = client.get(f"/api/datasets/{TASK}/train/encoding-preview")
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["input_feature_count"] == 3
    assert report["output_feature_count"] >= 3
    per_column = {entry["column"]: entry for entry in report["per_column"]}
    assert per_column["x"]["kind"] == "numeric"
    assert per_column["cat"]["kind"] == "categorical"
    assert per_column["cat"]["cardinality"] == 2
    assert any(name.endswith("cat_a") for name in per_column["cat"]["output_columns"])
    assert report["preview"]
    assert any("per training fold" in note for note in report["notes"])


def test_encoding_preview_with_leading_ignored_columns(client):
    _register(client)
    ignored = client.put(
        f"/api/datasets/{TASK}/ignored-columns", json={"columns": ["x"]}
    )
    assert ignored.status_code == 200
    response = client.get(f"/api/datasets/{TASK}/train/encoding-preview")
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["input_features"] == ["z", "cat"]
    per_column = {entry["column"]: entry for entry in report["per_column"]}
    assert per_column["z"]["kind"] == "numeric"
    assert per_column["cat"]["kind"] == "categorical"
    assert per_column["cat"]["output_feature_count"] >= 2


def _register_clipped(client):
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 100.0, -100.0, 5.0, 6.0],
            "z": [1, 2, 3, 4, 5, 6, 7, 8],
            "cat": list("aabbccdd"),
            "y": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    _register(client, frame)


def test_outlier_clipping_preview_and_save(client):
    _register_clipped(client)
    source = (settings.task_dataset_dir(TASK) / "train.csv").read_text()
    preview = client.post(
        f"/api/datasets/{TASK}/clip",
        json={"partition": "train", "method": "iqr", "preview_only": True},
    )
    assert preview.status_code == 200, preview.text
    report = preview.json()
    assert report["written"] is False
    assert report["cells_clipped"] >= 2
    stats = report["columns_report"]["x"]
    assert stats["clipped_low"] == 1 and stats["clipped_high"] == 1
    assert stats["after"]["max"] < 100 and stats["after"]["min"] > -100
    assert not (settings.task_dataset_dir(TASK) / "train_clipped.csv").exists()
    assert (settings.task_dataset_dir(TASK) / "train.csv").read_text() == source

    saved = client.post(
        f"/api/datasets/{TASK}/clip",
        json={
            "partition": "train",
            "method": "iqr",
            "columns": ["x"],
            "output_partition": "train_clipped",
        },
        )
    assert saved.status_code == 200, saved.text
    assert saved.json()["dataset_fingerprint"]
    clipped = pd.read_csv(settings.task_dataset_dir(TASK) / "train_clipped.csv")
    assert clipped["x"].max() < 100 and clipped["x"].min() > -100
    with open_tracker(TASK) as tracker:
        history = tracker.GetMetadata()["clipping_history"]
    assert history[-1]["output_partition"] == "train_clipped"
    assert history[-1]["cells_clipped"] >= 2

    # Categorical columns cannot be winsorized and the target is refused.
    for columns in (["cat"], ["y"]):
        rejected = client.post(
            f"/api/datasets/{TASK}/clip",
            json={"partition": "train", "method": "iqr", "columns": columns, "preview_only": True},
        )
        assert rejected.status_code == 400


def test_duplicate_analysis_warns_about_balance_and_leakage(client):
    frame = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 3, 3, 3],
            "b": ["u", "u", "u", "v", "v", "w", "w", "w"],
            "y": [0, 1, 0, 0, 0, 1, 0, 1],
        }
    )
    _register(client, frame)
    response = client.get(f"/api/datasets/{TASK}/train/duplicates")
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["duplicate_rows"] == 8
    assert report["duplicate_groups"] == 3
    assert report["rows_removed"] == 5
    assert report["conflicting_label_groups"] == 2
    assert report["conflicting_label_rows"] == 6
    joined = " ".join(report["warnings"]).lower()
    assert "balance" in joined
    assert "split hazard" in joined
    assert "contradictory" in joined

    preview = client.post(
        f"/api/datasets/{TASK}/deduplicate",
        json={"partition": "train", "preview_only": True},
    )
    assert preview.status_code == 200, preview.text
    assert preview.json()["rows_after"] == 3
    assert not (settings.task_dataset_dir(TASK) / "train_deduplicated.csv").exists()

    saved = client.post(
        f"/api/datasets/{TASK}/deduplicate",
        json={"partition": "train", "output_partition": "train_dedup"},
    )
    assert saved.status_code == 200, saved.text
    deduped = pd.read_csv(settings.task_dataset_dir(TASK) / "train_dedup.csv")
    assert len(deduped) == 3
    assert len(pd.read_csv(settings.task_dataset_dir(TASK) / "train.csv")) == 8
    with open_tracker(TASK) as tracker:
        history = tracker.GetMetadata()["deduplication_history"]
    assert history[-1]["output_partition"] == "train_dedup"
    assert history[-1]["rows_removed"] == 5

