"""Hand-computable fixtures for the paired statistics toolkit.

Every expected value here is derived by hand (or is a published exact
Wilcoxon probability) so a regression in the implementation is visible without
trusting the implementation to generate its own baseline. The statistical
caveats live in ``benchmarks/stats.py``: paired-only inputs, percentile
bootstrap (not Bayesian), Holm FWER control, Cliff's delta as an ordinal
effect size.
"""

import numpy as np
import pytest

from benchmarks.stats import (
    bootstrap_ci,
    bootstrap_paired_ci,
    cliffs_delta,
    holm_correction,
    paired_deltas,
    paired_report_row,
    wilcoxon_holm,
    wilcoxon_pvalues,
)

# Exact two-sided Wilcoxon p for ten strictly positive differences (n=10, all
# ranks used): 2 / 2**10 = 0.001953125.
EXACT_P10 = 0.001953125


def test_paired_deltas_exact_and_nan_pair_dropped():
    deltas = paired_deltas([3, 4, 5], [1, 2, 3])
    np.testing.assert_array_equal(deltas, [2.0, 2.0, 2.0])

    deltas = paired_deltas([1.0, np.nan, 4.0, 5.0], [0.0, 2.0, np.nan, 1.0])
    np.testing.assert_array_equal(deltas, [1.0, 4.0])


def test_paired_deltas_rejects_length_mismatch():
    with pytest.raises(ValueError):
        paired_deltas([1.0, 2.0], [1.0])


def test_bootstrap_ci_constant_vector_collapses():
    result = bootstrap_ci([2.0, 2.0, 2.0, 2.0], n_resamples=500, seed=3)
    assert result["estimate"] == 2.0
    assert result["low"] == 2.0
    assert result["high"] == 2.0
    assert result["n"] == 4
    assert result["n_resamples"] == 500
    assert result["confidence"] == 0.95


def test_bootstrap_ci_known_sample_mean_and_band():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # mean 3.5
    result = bootstrap_ci(values, n_resamples=2000, seed=0)
    assert result["estimate"] == 3.5
    assert result["low"] <= result["estimate"] <= result["high"]
    assert set(result) == {"estimate", "low", "high", "n", "n_resamples", "confidence"}


def test_bootstrap_ci_is_deterministic_per_seed():
    # Spread-out continuous values, so the 2.5/97.5 percentile endpoints are
    # not pinned to the same discrete grid point for every RNG draw.
    values = [0.1, 0.7, 1.3, 2.9, 5.1, 9.7]
    first = bootstrap_ci(values, n_resamples=4000, seed=0)
    second = bootstrap_ci(values, n_resamples=4000, seed=0)
    assert (first["low"], first["high"]) == (second["low"], second["high"])

    other = bootstrap_ci(values, n_resamples=4000, seed=1)
    assert (other["low"], other["high"]) != (first["low"], first["high"])


def test_bootstrap_ci_rejects_empty_input():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_paired_bootstrap_ci_constant_shift():
    treatment = [3.0, 4.0, 5.0, 6.0]
    control = [1.0, 2.0, 3.0, 4.0]
    result = bootstrap_paired_ci(treatment, control, n_resamples=500, seed=0)
    assert result["mean_delta"] == 2.0
    assert result["low"] == 2.0
    assert result["high"] == 2.0
    assert result["wins"] == 4
    assert result["ties"] == 0
    assert result["losses"] == 0
    assert result["n_pairs"] == 4


def test_cliffs_delta_perfect_separation_and_identity():
    forward = cliffs_delta([3, 4, 5], [1, 2])
    assert forward["delta"] == 1.0
    assert forward["magnitude"] == "large"
    assert forward["n_treatment"] == 3
    assert forward["n_control"] == 2

    reversed_ = cliffs_delta([1, 2], [3, 4, 5])
    assert reversed_["delta"] == -1.0
    assert reversed_["magnitude"] == "large"

    identical = cliffs_delta([1, 2, 3], [1, 2, 3])
    assert identical["delta"] == 0.0
    assert identical["magnitude"] == "negligible"


def test_cliffs_delta_ignores_nan_and_rejects_empty():
    assert cliffs_delta([3, np.nan, 5], [1, 2])["delta"] == 1.0
    with pytest.raises(ValueError):
        cliffs_delta([np.nan], [1.0])


def test_wilcoxon_exact_p_for_ten_positive_differences():
    result = wilcoxon_pvalues(np.arange(1, 11, dtype=float), np.zeros(10))
    assert result["p_value"] == EXACT_P10
    assert result["n_pairs"] == 10
    assert result["zero_differences"] == 0
    assert result["statistic"] == 0.0


def test_wilcoxon_all_ties_returns_documented_sentinel():
    result = wilcoxon_pvalues([2.0, 3.0, 4.0], [2.0, 3.0, 4.0])
    assert result["p_value"] == 1.0
    assert result["statistic"] == 0.0
    assert result["zero_differences"] == 3
    assert result["n_pairs"] == 3


def test_holm_correction_matches_hand_computation():
    raw = [EXACT_P10, 0.04]
    result = holm_correction(raw)
    # m=2 sorted ascending: 2*0.001953125 = 0.00390625; then max(that, 1*0.04).
    assert result["adjusted"] == [0.00390625, 0.04]
    assert result["reject"] == [True, True]
    assert result["alpha"] == 0.05


def test_holm_correction_preserves_order_and_clips():
    result = holm_correction([0.04, 0.001])
    assert result["adjusted"] == [0.04, 0.002]
    assert result["reject"] == [True, True]

    clipped = holm_correction([0.5, 0.6, 0.9])
    assert clipped["adjusted"] == [1.0, 1.0, 1.0]
    assert clipped["reject"] == [False, False, False]


def test_wilcoxon_holm_adjusts_two_methods():
    control = np.zeros(10)
    report = wilcoxon_holm(
        {
            "shift1": np.arange(1, 11, dtype=float),
            # All-tie method contributes the p=1.0 sentinel; Holm must still
            # adjust the other method and leave this one unrejected.
            "no_effect": np.zeros(10),
        },
        control,
    )
    assert report["shift1"]["p_value"] == EXACT_P10
    assert report["shift1"]["p_adjusted"] == 0.00390625
    assert report["shift1"]["reject"] is True
    assert report["no_effect"]["p_value"] == 1.0
    assert report["no_effect"]["p_adjusted"] == 1.0
    assert report["no_effect"]["reject"] is False


def test_paired_report_row_positive_shift_is_distinguishable():
    row = paired_report_row(
        "shift", np.full(10, 2.0), np.zeros(10), seed=0
    )
    assert row["mean_delta"] == 2.0
    assert row["ci_low"] == 2.0 and row["ci_high"] == 2.0
    assert (row["wins"], row["ties"], row["losses"]) == (10, 0, 0)
    assert row["wilcoxon_p"] == EXACT_P10
    assert row["cliffs_delta"] == 1.0
    assert row["distinguishable"] is True


def test_paired_report_row_identical_arrays_are_not_distinguishable():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    row = paired_report_row("same", values, values.copy(), seed=0)
    assert row["mean_delta"] == 0.0
    assert (row["ci_low"], row["ci_high"]) == (0.0, 0.0)
    assert row["wilcoxon_p"] == 1.0
    assert row["cliffs_delta"] == 0.0
    assert row["distinguishable"] is False
