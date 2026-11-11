from unittest.mock import patch

import json

import numpy as np

from benchmarks.methods import (
    RECORD_SCHEMA_VERSION,
    SPACE,
    run_random,
    run_sksurrogate,
)


def test_random_search_records_total_search_wall_time():
    with (
        patch("benchmarks.methods._evaluate_candidate", return_value=0.8),
        patch("benchmarks.methods._final_test_scores", return_value=(0.8, 0.8)),
    ):
        record = run_random(
            np.empty((1, 1)),
            np.empty(1),
            np.empty((1, 1)),
            np.empty(1),
            budget=3,
            seed=1,
            n_estimators=1,
        )

    assert record.n_evals == 3
    assert record.total_wall_s is not None
    assert record.total_wall_s >= sum(item["wall_s"] for item in record.records)


def test_random_search_records_full_per_evaluation_params():
    """Schema v2 (Stage A): every evaluation records its full candidate config."""
    with (
        patch("benchmarks.methods._evaluate_candidate", side_effect=lambda *a, **k: 0.5),
        patch("benchmarks.methods._final_test_scores", return_value=(0.8, 0.8)),
    ):
        record = run_random(
            np.empty((1, 1)),
            np.empty(1),
            np.empty((1, 1)),
            np.empty(1),
            budget=4,
            seed=2,
            n_estimators=1,
        )

    # Every per-evaluation entry carries the full config (all SPACE keys), values in space.
    for rec in record.records:
        assert set(rec["params"]) == set(SPACE)
        for name, value in rec["params"].items():
            assert value in SPACE[name]

    # best_params is exactly the params of the argmax completed evaluation (ties -> first).
    best_rec = max(record.records, key=lambda r: r["cv_score"] if r["cv_score"] is not None else -np.inf)
    assert record.best_params == best_rec["params"]

    # The schema version is stamped on the record and survives the JSON round-trip.
    assert record.schema_version == RECORD_SCHEMA_VERSION == 2
    payload = json.loads(record.to_json())
    assert payload["schema_version"] == 2
    for rec in payload["records"]:
        assert set(rec["params"]) == set(SPACE)


def test_sksurrogate_benchmark_uses_and_records_hilbert_surface(tmp_path, monkeypatch):
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=0,
    )
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0
    )
    monkeypatch.chdir(tmp_path)

    record = run_sksurrogate(
        Xtr, ytr, Xte, yte, budget=21, seed=0, n_estimators=1
    )

    assert record.n_evals == 21
    assert record.extra["surrogate"].endswith(".HilbertRegressor")
    assert record.extra["surrogate_source"] == "user"

    # Schema v2 (Stage A): the structsearch arm must keep per-evaluation candidate configs.
    for rec in record.records:
        assert set(rec["params"]) == set(SPACE)
        for name, value in rec["params"].items():
            assert value in SPACE[name]
    completed = [r for r in record.records if r["cv_score"] is not None]
    best_rec = max(completed, key=lambda r: r["cv_score"])
    assert record.best_params == best_rec["params"]
