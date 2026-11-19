"""Runner-level tests for the C-space arms (DESIGN-cspace.md §4, gates G2/G4).

These are small-budget runs on a real dataset, not mocks: the properties under
test (exact evaluation counting, surrogate engagement, no-surrogate ablation)
only exist in the real search loop.
"""

import pytest
from sklearn.datasets import load_breast_cancer

# The TPE arm needs the optional [benchmark] extra; skip when it is not installed
# (the benchmark-smoke CI job installs .[benchmark] and runs these).
pytest.importorskip("optuna")

from benchmarks.cspace_methods import (
    SRS_MIN_EVALS,
    cv_splits,
    run_optuna_tpe_cspace,
    run_random_cspace,
    run_srs_cspace,
    run_srs_nosurr_cspace,
    run_srs_onehot_cspace,
    train_test_split_frame,
)

X, y = load_breast_cancer(return_X_y=True)


def _cell(seed=0):
    Xtr, Xte, ytr, yte = train_test_split_frame(X, y, seed)
    return Xtr, ytr, Xte, yte


def test_budget_is_exact_and_the_surrogate_engages(tmp_path, monkeypatch):
    """G2: n_evals == budget, and the surface fires at the configured threshold."""
    monkeypatch.chdir(tmp_path)  # SRS checkpoints into the CWD
    Xtr, ytr, Xte, yte = _cell()
    record = run_srs_cspace(Xtr, ytr, Xte, yte, budget=24, seed=0)

    assert record.status == "ok", record.error
    assert record.n_evals == 24
    assert record.extra["engaged_at"] == SRS_MIN_EVALS
    assert record.extra["n_surface_fits"] > 0
    assert record.extra["surrogate_class"] == "GaussianProcessRegressor"
    assert record.extra["invalid_rate"] == 0.0
    assert record.final_test_accuracy is not None


def test_no_surrogate_ablation_never_engages(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Xtr, ytr, Xte, yte = _cell()
    record = run_srs_nosurr_cspace(Xtr, ytr, Xte, yte, budget=24, seed=0)

    assert record.n_evals == 24
    assert record.extra["engaged_at"] is None
    assert record.extra["n_surface_fits"] == 0


def test_onehot_arm_uses_the_mixed_categorical_surface(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Xtr, ytr, Xte, yte = _cell()
    record = run_srs_onehot_cspace(Xtr, ytr, Xte, yte, budget=21, seed=0)

    assert record.n_evals == 21
    assert record.extra["surrogate_class"] == "MixedCategoricalSurface"
    assert record.extra["engaged_at"] == SRS_MIN_EVALS


def test_optuna_tpe_is_counted_exactly_and_never_invalid():
    """G2: Optuna trial count == budget, and conditional branching yields 0 invalid."""
    Xtr, ytr, Xte, yte = _cell()
    record = run_optuna_tpe_cspace(Xtr, ytr, Xte, yte, budget=8, seed=0)

    assert record.status == "ok", record.error
    assert record.n_evals == 8
    assert record.extra["invalid_rate"] == 0.0
    assert record.extra["sampler"] == "tpe"


def test_random_arm_draws_only_valid_configs():
    Xtr, ytr, Xte, yte = _cell()
    record = run_random_cspace(Xtr, ytr, Xte, yte, budget=6, seed=0)

    assert record.n_evals == 6
    assert record.extra["invalid_rate"] == 0.0
    assert set(record.extra["family_mix"]) <= {
        "logreg", "rf", "et", "hgb", "svc", "knn"
    }


def test_wall_clock_protocol_stops_near_the_limit(tmp_path, monkeypatch):
    """Protocol A: the time cap binds; overrun is at most one candidate."""
    monkeypatch.chdir(tmp_path)
    Xtr, ytr, Xte, yte = _cell()
    record = run_srs_cspace(Xtr, ytr, Xte, yte, wall=3.0, seed=0)

    assert record.n_evals >= 1
    assert record.total_wall_s == pytest.approx(3.0, abs=2.5)
    assert record.extra["protocol"] == "A"
