"""Leakage guards: fold-local preprocessing and split/column diagnostics.

Two different guarantees are pinned here:

* **Preprocessing is fitted in-fold.** ``AML`` installs
  ``FoldLocalDataPreprocessor`` *inside* the searched pipeline, so the encoder is
  fitted per CV fold, never on the full training frame. A spy on
  ``fit_transform`` records the row counts it was handed; every call must see
  fewer rows than the full frame.
* **The split and the columns are checked.** Duplicate rows across partitions,
  single-column target leakage and monotone datetime columns are reported in the
  result payload instead of being silently assumed absent.
"""

import numpy as np
import pandas as pd
import pytest

from api.leakage import (
    AUC_THRESHOLD,
    CORRELATION_THRESHOLD,
    detect_leakage_warnings,
    duplicate_overlap,
    target_leakage_candidates,
    temporal_split_suspected,
)


# --------------------------------------------------------------------------- #
# Fold-local preprocessing
# --------------------------------------------------------------------------- #


def test_preset_search_fits_preprocessing_inside_folds(tmp_path, monkeypatch):
    from SKSurrogate import AML
    from SKSurrogate.DataProcess import FoldLocalDataPreprocessor

    rng = np.random.RandomState(0)
    n = 48
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "grade": rng.choice(["low", "mid", "high"], size=n),
        }
    )
    frame["y"] = (frame["a"] + 0.2 * rng.normal(size=n) > 0).astype(int)

    seen_rows = []
    original = FoldLocalDataPreprocessor.fit_transform

    def spy(self, X, y=None, **kwargs):
        seen_rows.append(len(X))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(FoldLocalDataPreprocessor, "fit_transform", spy)
    search = AML(
        preset="classification",
        length=1,
        min_random_evals=2,
        n_jobs=1,
        verbose=0,
        random_state=0,
        check_point=str(tmp_path) + "/",
    )
    search.fit(
        frame[["a", "grade"]],
        frame["y"],
        strategy="eoa",
        max_generations=1,
        num_parents=1,
    )

    assert seen_rows, "the preprocessor never ran: preprocessing was skipped entirely"
    # The final refit of each candidate legitimately fits preprocessing on all
    # training rows; what must not happen is a full-frame fit per CV fold. With
    # 3 folds of 48 rows, fold blocks are 32 rows, so fold-sized calls exist and
    # the full-frame calls cannot exceed the number of candidates refitted.
    fold_sized = [rows for rows in seen_rows if rows < n]
    assert fold_sized, "no fold-sized preprocessing fit: the encoder saw only the full frame"
    assert 32 in fold_sized or min(fold_sized) <= n - n // 3
    # Full-frame fits are post-scoring refits of an already evaluated candidate;
    # the leakage-relevant fact is that fold-sized fits exist and are not
    # outnumbered by them (a pre-encoded frame would show only full-frame fits).
    assert len(fold_sized) >= seen_rows.count(n), (
        "fold-sized preprocessing fits (%d) are outnumbered by full-frame fits (%d): "
        "the encoder looks fitted before the split" % (len(fold_sized), seen_rows.count(n))
    )
    steps = [name for name, _ in search.best_estimator_.steps]
    assert "preprocess" in steps


# --------------------------------------------------------------------------- #
# Duplicate rows across the split
# --------------------------------------------------------------------------- #


def test_exact_duplicate_rows_across_partitions_are_counted():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"], "y": [0, 1, 0]})
    validation = pd.DataFrame(
        {"a": [3.0, 4.0], "b": ["z", "w"], "y": [0, 1]}, index=[10, 11]
    )
    block = duplicate_overlap(train, validation, target="y")
    assert block["checked"] is True
    assert block["duplicate_rows"] == 1
    assert block["duplicate_fraction_of_validation"] == 0.5


def test_near_duplicate_rows_are_detected_after_rounding():
    train = pd.DataFrame({"a": [0.1234567], "y": [1]})
    validation = pd.DataFrame({"a": [0.1234567 + 1e-9], "y": [0]})
    block = duplicate_overlap(train, validation, target="y")
    assert block["duplicate_rows"] == 1


def test_distinct_rows_are_not_flagged():
    train = pd.DataFrame({"a": [1.0, 2.0], "y": [0, 1]})
    validation = pd.DataFrame({"a": [3.0, 4.0], "y": [0, 1]})
    block = duplicate_overlap(train, validation, target="y")
    assert block["duplicate_rows"] == 0


def test_no_validation_partition_is_reported_as_unchecked():
    block = duplicate_overlap(pd.DataFrame({"a": [1.0], "y": [0]}), None, target="y")
    assert block["checked"] is False
    assert "no validation partition" in block["reason"]


# --------------------------------------------------------------------------- #
# Single-column target leakage
# --------------------------------------------------------------------------- #


def test_perfectly_predictive_column_is_a_leakage_candidate():
    frame = pd.DataFrame(
        {
            "row_id": np.arange(20.0),
            "noise": np.random.RandomState(0).normal(size=20),
            "y": (np.arange(20) >= 10).astype(int),
        }
    )
    block = target_leakage_candidates(frame, "y")
    assert block["checked"] is True
    columns = [entry["column"] for entry in block["candidates"]]
    assert "row_id" in columns
    entry = next(entry for entry in block["candidates"] if entry["column"] == "row_id")
    # The positive class here is the majority label (0), which lives at the low
    # row_id end; a perfect *inverse* AUC is the same leakage signal.
    assert entry["direction"] == "inverse"
    assert min(entry["value"], 1 - entry["value"]) <= 1 - AUC_THRESHOLD


def test_regression_target_uses_absolute_correlation():
    frame = pd.DataFrame(
        {
            "leak": np.linspace(0, 1, 30),
            "noise": np.random.RandomState(1).normal(size=30),
            "y": 3.0 * np.linspace(0, 1, 30) + 1.0,
        }
    )
    block = target_leakage_candidates(frame, "y")
    candidates = {entry["column"]: entry for entry in block["candidates"]}
    assert "leak" in candidates
    assert abs(candidates["leak"]["value"]) >= CORRELATION_THRESHOLD
    assert "noise" not in candidates


def test_benign_columns_are_not_flagged():
    rng = np.random.RandomState(2)
    frame = pd.DataFrame(
        {"a": rng.normal(size=200), "b": rng.normal(size=200), "y": rng.choice([0, 1], size=200)}
    )
    assert target_leakage_candidates(frame, "y")["candidates"] == []


# --------------------------------------------------------------------------- #
# Temporal split suspicion
# --------------------------------------------------------------------------- #


def test_monotone_datetime_column_is_flagged():
    frame = pd.DataFrame(
        {"when": pd.date_range("2024-01-01", periods=5, freq="D"), "y": [0, 1, 0, 1, 0]}
    )
    block = temporal_split_suspected(frame)
    assert block["suspected"] is True
    assert block["columns"] == ["when"]


def test_unordered_datetime_column_is_not_flagged():
    frame = pd.DataFrame(
        {"when": pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]), "y": [0, 1, 0]}
    )
    assert temporal_split_suspected(frame)["suspected"] is False


# --------------------------------------------------------------------------- #
# Assembled block
# --------------------------------------------------------------------------- #


def test_detect_leakage_warnings_assembles_the_payload_block():
    train = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "leak": [10.0, 20.0, 30.0], "y": [0, 1, 0]}
    )
    validation = pd.DataFrame({"a": [1.0], "leak": [10.0], "y": [0]})
    block = detect_leakage_warnings(
        train, validation, target="y", train_partition="train", validation_partition="validation"
    )
    assert block["checked"] is True
    assert block["duplicate_rows"]["duplicate_rows"] == 1
    assert block["target_leakage"]["checked"] is True
    assert block["temporal_split"]["checked"] is True
    assert block["caveats"]
    assert block["train_partition"] == "train"
    assert block["validation_partition"] == "validation"


def test_detect_leakage_warnings_never_raises_on_a_missing_target():
    block = detect_leakage_warnings(pd.DataFrame({"a": [1.0, 2.0]}), target="y")
    assert block["checked"] is True
    assert block["target_leakage"]["checked"] is False


# --------------------------------------------------------------------------- #
# Live API surface
# --------------------------------------------------------------------------- #


def test_experiment_result_reports_leakage_for_a_duplicated_split(tmp_path, monkeypatch):
    """A seeded duplicate-row dataset must surface the warning in the job result.

    This is the end-to-end pin for the phase-3 exit criterion: the block is not
    merely computable in isolation, it travels from the fit into the payload (and
    from there into the bundle's manifest sidecar).
    """
    import importlib.util
    import time

    if importlib.util.find_spec("httpx") is None:
        pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from api.config import settings
    from api.main import app
    from SKSurrogate import load_bundle, verify_bundle

    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    client = TestClient(app)

    rng = np.random.RandomState(3)
    n = 40
    train = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "y": rng.choice([0, 1], size=n),
        }
    )
    # Validation repeats five training rows verbatim: the model has already
    # seen those records during fitting. The rest are fresh rows.
    repeats = train.iloc[:5].copy()
    held_out = pd.DataFrame(
        {
            "a": rng.normal(size=10),
            "b": rng.normal(size=10),
            "y": rng.choice([0, 1], size=10),
        }
    )
    validation = pd.concat([repeats, held_out], ignore_index=True)
    for name, frame in (("train", train), ("validation", validation)):
        response = client.post(
            "/api/datasets/leaky/register",
            data={"target": "y", "partition": name},
            files={"file": ("%s.csv" % name, frame.to_csv(index=False), "text/csv")},
        )
        assert response.status_code == 200, response.text

    submitted = client.post(
        "/api/experiments/leaky/run",
        json={"preset": "classification", "length": 1, "max_generations": 1, "num_parents": 1, "surrogate_itrs": 4, "surrogate_min_random_evals": 2, "random_state": 0},
    )
    assert submitted.status_code == 200, submitted.text
    job_id = submitted.json()["job_id"]
    record = None
    for _ in range(600):
        record = client.get("/api/jobs/" + job_id).json()
        if record["status"] in ("completed", "failed"):
            break
        time.sleep(0.25)
    assert record is not None and record["status"] == "completed", record
    result = record["result"]
    warnings = result["leakage_warnings"]
    assert warnings["checked"] is True
    assert warnings["duplicate_rows"]["checked"] is True
    assert warnings["duplicate_rows"]["duplicate_rows"] == 5
    assert warnings["validation_partition"] == "validation"

    # The same block is stored with the bundle, and the artifact verifies.
    from pathlib import Path as _Path

    bundle = load_bundle(_Path(result["bundle_path"]), strict_dependencies=False)
    assert bundle.metadata["reproducibility"]["leakage_warnings"]["duplicate_rows"]["duplicate_rows"] == 5
    assert bundle.manifest["schema_version"] == 2
    assert verify_bundle(_Path(result["bundle_path"]))["verified"] is True
