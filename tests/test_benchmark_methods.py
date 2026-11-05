from unittest.mock import patch

import numpy as np

from benchmarks.methods import run_random, run_sksurrogate


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
