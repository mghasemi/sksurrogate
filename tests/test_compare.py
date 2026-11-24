"""Unit tests for :mod:`SKSurrogate.compare`.

Hand-computed fixtures where possible: the bootstrap interval only has to be
deterministic and bracket the estimate; Holm, Cliff's delta, Brier, ECE and the
Pareto front all have exact closed forms that are asserted directly.
"""

import numpy as np
import pytest

from SKSurrogate.compare import (
    bootstrap_ci,
    brier_score,
    cliffs_delta,
    compare_scores,
    expected_calibration_error,
    holm_correction,
    paired_deltas,
    fitted_parameter_count,
    pareto_frontier,
    rank_stability,
    simpler_within_tolerance,
)


def test_bootstrap_ci_is_seeded_and_brackets_the_estimate():
    values = [0.1, 0.2, 0.25, 0.4, 0.5, 0.55, 0.7]
    first = bootstrap_ci(values, n_resamples=500, seed=7)
    second = bootstrap_ci(values, n_resamples=500, seed=7)
    assert first == second
    assert first["estimate"] == pytest.approx(float(np.mean(values)))
    assert first["low"] <= first["estimate"] <= first["high"]
    assert first["n"] == len(values)
    # A different seed is a different resample stream; the point estimate is not.
    other = bootstrap_ci(values, n_resamples=500, seed=8)
    assert other["estimate"] == first["estimate"]


def test_bootstrap_ci_refuses_an_empty_sample():
    with pytest.raises(ValueError):
        bootstrap_ci([float("nan")])


def test_paired_deltas_drops_only_incomplete_pairs_and_keeps_ties():
    deltas = paired_deltas([1.0, float("nan"), 3.0, 4.0], [0.0, 1.0, 3.0, float("nan")])
    np.testing.assert_allclose(deltas, [1.0, 0.0])
    with pytest.raises(ValueError):
        paired_deltas([1.0, 2.0], [1.0])


def test_holm_correction_matches_the_step_down_definition():
    result = holm_correction([0.01, 0.04, 0.03], alpha=0.05)
    # sorted p: .01 (x3 = .03), .03 (x2 = .06), .04 (x1 = .04) -> running max .03 .06 .06
    np.testing.assert_allclose(result["adjusted"], [0.03, 0.06, 0.06], atol=1e-12)
    assert result["reject"] == [True, False, False]


def test_cliffs_delta_two_sample_counts_superiority_pairs():
    result = cliffs_delta([1.0, 2.0], [0.0, 3.0])
    # greater pairs: (1,0),(2,0) = 2 ; less pairs: (1,3),(2,3) = 2
    assert result["delta"] == pytest.approx(0.0)
    assert result["magnitude"] == "negligible"
    strong = cliffs_delta([5.0, 6.0], [0.0, 1.0])
    assert strong["delta"] == pytest.approx(1.0)
    assert strong["magnitude"] == "large"


def test_compare_scores_detects_a_known_improvement():
    # Six pairs: a two-sided Wilcoxon needs n=6 for its smallest p (0.031) to
    # fall below 0.05, so a five-pair fixture could never be distinguishable.
    control = [0.70, 0.72, 0.71, 0.69, 0.73, 0.70]
    treatment = [0.75, 0.77, 0.76, 0.74, 0.78, 0.75]
    report = compare_scores({"baseline": control, "candidate": treatment})
    assert report["control"] == "baseline"
    assert report["metric_direction"] == "higher_is_better"
    assert report["models"]["candidate"]["mean"] > report["models"]["baseline"]["mean"]
    (row,) = report["comparisons"]
    assert row["treatment"] == "candidate"
    assert row["mean_delta"] == pytest.approx(0.05)
    assert row["wins"] == 6 and row["losses"] == 0
    assert row["ci_low"] > 0.0
    assert row["p_adjusted"] < 0.05
    assert row["distinguishable"] is True
    assert row["cliffs_magnitude"] == "large"


def test_compare_scores_identical_models_are_not_distinguishable():
    scores = [0.80, 0.81, 0.79, 0.82, 0.80]
    report = compare_scores({"a": scores, "b": list(scores)})
    (row,) = report["comparisons"]
    assert row["mean_delta"] == pytest.approx(0.0)
    assert row["ties"] == len(scores)
    assert row["ci_low"] == pytest.approx(0.0)
    assert row["ci_high"] == pytest.approx(0.0)
    assert row["wilcoxon_p"] == 1.0
    assert row["distinguishable"] is False


def test_compare_scores_orientates_lower_is_better_metrics():
    # Error metric: the candidate has a lower error, which is better.
    control = [0.40, 0.42, 0.41, 0.39, 0.43, 0.40]
    treatment = [0.30, 0.32, 0.31, 0.29, 0.33, 0.30]
    report = compare_scores(
        {"baseline": control, "candidate": treatment}, higher_is_better=False
    )
    assert report["metric_direction"] == "lower_is_better"
    (row,) = report["comparisons"]
    assert row["mean_delta"] == pytest.approx(0.10)  # oriented: control - treatment
    assert row["wins"] == 6
    assert row["distinguishable"] is True


def test_compare_scores_rejects_unaligned_or_tiny_inputs():
    with pytest.raises(ValueError):
        compare_scores({"only": [0.1, 0.2]})
    with pytest.raises(ValueError):
        compare_scores({"a": [0.1, 0.2, 0.3], "b": [0.1, 0.2]})
    with pytest.raises(ValueError):
        compare_scores({"a": [0.1, 0.2], "b": [0.2, 0.3]}, control="missing")


def test_pareto_frontier_drops_dominated_models():
    entries = {
        "big": {"score": 0.90, "complexity": 10, "cost": 1.0},
        "small": {"score": 0.80, "complexity": 5, "cost": 1.0},
        "worse": {"score": 0.80, "complexity": 20, "cost": 2.0},
    }
    result = pareto_frontier(entries)
    assert result["frontier"] == ["big", "small"]
    # Both frontier members dominate "worse"; any of them is a valid witness.
    assert result["dominated"]["worse"] in {"big", "small"}
    assert result["incomplete"] == []


def test_pareto_frontier_reports_incomplete_entries():
    entries = {
        "complete": {"score": 0.9, "complexity": 1, "cost": 1},
        "no_cost": {"score": 0.8, "complexity": 1, "cost": None},
    }
    result = pareto_frontier(entries)
    assert result["frontier"] == ["complete"]
    assert result["incomplete"] == ["no_cost"]


def test_rank_stability_counts_modal_rank_agreement():
    stable = rank_stability({"a": [0.9, 0.9, 0.9, 0.9], "b": [0.5, 0.5, 0.5, 0.5]})
    assert stable["models"]["a"]["mean_rank"] == pytest.approx(1.0)
    assert stable["models"]["a"]["rank_consistency"] == pytest.approx(1.0)
    assert stable["models"]["b"]["mean_rank"] == pytest.approx(2.0)
    assert stable["models"]["a"]["unstable"] is False

    flip = rank_stability({"a": [0.9, 0.1, 0.9, 0.1], "b": [0.1, 0.9, 0.1, 0.9]})
    assert flip["models"]["a"]["best_folds"] == 2
    assert flip["models"]["a"]["rank_consistency"] == pytest.approx(0.5)


def test_brier_score_exact_cases():
    assert brier_score([0, 1], np.array([0.0, 1.0])) == pytest.approx(0.0)
    # Multiclass-sum convention: a 0.5 forecast contributes 0.25 to each class.
    assert brier_score([0, 0, 0, 0], np.full(4, 0.5)) == pytest.approx(0.5)
    # Multiclass: a confident wrong prediction costs 2, a correct one costs 0.
    multiclass = brier_score(
        ["a", "b"], np.array([[1.0, 0.0], [0.9, 0.1]]), labels=["a", "b"]
    )
    assert multiclass == pytest.approx((0.0 + (0.81 + 0.81)) / 2.0)


def test_expected_calibration_error_exact_cases():
    # Ten confident predictions, seven correct -> ECE = |0.7 - 1.0| = 0.3.
    y_true = [1] * 7 + [0] * 3
    probabilities = np.tile([0.0, 1.0], (10, 1))
    result = expected_calibration_error(y_true, probabilities, n_bins=5)
    assert result["ece"] == pytest.approx(0.3)
    assert sum(item["count"] for item in result["bins"]) == 10
    # A perfectly calibrated split: 0.75 confidence with 75% accuracy.
    y_true = [1, 1, 1, 0]
    probabilities = np.tile([0.25, 0.75], (4, 1))
    assert expected_calibration_error(y_true, probabilities)["ece"] == pytest.approx(0.0)


def test_fitted_parameter_count_walks_nested_estimators():
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]).fit(X, y)
    # Scaler: mean_, var_, scale_ (2 each) = 6. Regressor: coef_ (2) + the SVD's
    # singular_ (2) = 4. Scalars (intercept_, n_samples_seen_) are not counted.
    assert fitted_parameter_count(model) == 10
    assert fitted_parameter_count(np.array([1.0, 2.0, 3.0])) == 3


def test_simpler_within_tolerance_prefers_the_smaller_eligible_model():
    entries = {
        "best": {"score": 0.90, "complexity": 100},
        "near": {"score": 0.895, "complexity": 10},
        "far": {"score": 0.70, "complexity": 2},
    }
    result = simpler_within_tolerance(entries, tolerance=0.01)
    assert result["best"] == "best"
    assert result["simpler"] == "near"
    lower = simpler_within_tolerance(
        entries, higher_is_better=False, tolerance=1.0
    )
    assert lower["best"] == "far"


def test_simpler_within_tolerance_returns_none_when_best_is_simplest():
    result = simpler_within_tolerance(
        {"best": {"score": 0.9, "complexity": 1}, "other": {"score": 0.8, "complexity": 5}},
        tolerance=0.01,
    )
    assert result["simpler"] is None


def test_compare_scores_accepts_a_control_that_is_not_first():
    # The Holm-adjusted index must follow the treatment order, not the model order.
    report = compare_scores(
        {
            "better": [0.9, 0.92, 0.91, 0.93, 0.90, 0.92],
            "control": [0.7, 0.72, 0.71, 0.73, 0.70, 0.72],
            "worse": [0.5, 0.52, 0.51, 0.53, 0.50, 0.52],
        },
        control="control",
    )
    rows = {row["treatment"]: row for row in report["comparisons"]}
    assert set(rows) == {"better", "worse"}
    assert rows["better"]["mean_delta"] > 0
    assert rows["worse"]["mean_delta"] < 0
    for row in rows.values():
        assert row["p_adjusted"] >= row["wilcoxon_p"]
    # Two equal raw p-values with m=2 adjust to the same 0.0625.
    assert rows["better"]["p_adjusted"] == pytest.approx(0.0625)
    assert rows["worse"]["p_adjusted"] == pytest.approx(0.0625)
