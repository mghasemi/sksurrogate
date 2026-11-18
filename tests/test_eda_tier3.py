"""Tier-3 EDA diagnostics: formal tests, effect sizes, partition drift,
summary stability and the Markdown report."""

import numpy as np
import pandas as pd
import pytest

from api.eda import (
    effect_sizes,
    formal_tests,
    markdown_report,
    partition_drift,
    stability,
)


def test_formal_tests_separate_normal_from_skewed():
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "normal": rng.normal(size=600),
            "skewed": rng.exponential(size=600),
            "constant": np.ones(600),
        }
    )
    report = formal_tests(frame)
    rows = {entry["column"]: entry for entry in report["columns"]}
    assert rows["normal"]["rejects_normality"] is False
    assert rows["skewed"]["rejects_normality"] is True
    assert rows["constant"]["reason"] == "fewer_than_3_distinct_values"
    assert report["method"] == "anderson_darling"
    assert "trivial departures" in report["caveat"]
    for entry in report["columns"]:
        if entry["statistic"] is not None:
            # Newer SciPy reports a p-value instead of critical values.
            assert entry["p_value"] is not None or entry["critical_value"] is not None


def test_effect_sizes_report_direction_and_bootstrap_interval():
    rng = np.random.default_rng(5)
    n = 400
    frame = pd.DataFrame(
        {
            "x": np.concatenate([rng.normal(size=n), rng.normal(loc=1.2, size=n)]),
            "group": ["a"] * n + ["b"] * n,
        }
    )
    report = effect_sizes(frame, target="group", bootstrap=200)
    comparison = report["comparisons"][0]
    assert comparison["group_a"] == "a" and comparison["group_b"] == "b"
    assert comparison["cohens_d"] < 0  # b is larger
    assert comparison["direction"] == "higher_in_b"
    low, high = comparison["cohens_d_ci"]
    assert low <= comparison["cohens_d"] <= high
    low, high = comparison["cliffs_delta_ci"]
    assert low <= comparison["cliffs_delta"] <= high
    assert comparison["cohens_d"] < -1.0
    assert report["bootstrap"] == 200


def test_effect_sizes_refuse_numeric_target_and_singleton_groups():
    numeric_target = pd.DataFrame({"x": np.arange(20.0), "y": np.arange(20.0) * 2})
    report = effect_sizes(numeric_target, target="y")
    assert report["reason"] == "grouping_column_is_numeric"
    assert report["comparisons"] == []

    sparse = pd.DataFrame({"x": np.arange(12.0), "g": ["a"] * 11 + ["b"]})
    report = effect_sizes(sparse, target="g")
    assert report["reason"] == "fewer_than_two_eligible_groups"


def test_cliffs_delta_is_extreme_under_complete_separation():
    frame = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0], "g": ["a"] * 3 + ["b"] * 3}
    )
    report = effect_sizes(frame, target="g", min_group_rows=3, bootstrap=60)
    comparison = report["comparisons"][0]
    assert comparison["cliffs_delta"] == -1.0
    assert comparison["direction"] == "higher_in_b"


def test_partition_drift_numeric_detects_shift_and_ignores_identical_samples():
    rng = np.random.default_rng(7)
    reference = pd.DataFrame({"x": rng.normal(size=600)})
    identical = pd.DataFrame({"x": rng.normal(size=600)})
    shifted = pd.DataFrame({"x": rng.normal(loc=1.2, size=600)})

    same = partition_drift(reference, identical)["numeric"][0]
    assert same["ks_p_value"] > 0.01
    assert abs(same["psi"]) < 0.1

    moved = partition_drift(reference, shifted)["numeric"][0]
    assert moved["ks_p_value"] < 1e-10
    assert moved["psi"] > 0.2
    assert moved["shift_in_pooled_sd"] > 1.0


def test_partition_drift_categorical_compares_marginals():
    # Regression pin: a crosstab of the two independent samples tests the wrong
    # thing and reported p ~ 0.87 for these marginals; the homogeneity test on
    # the 2 x k count table rejects strongly.
    reference = pd.DataFrame({"c": ["a"] * 400 + ["b"] * 400})
    identical = pd.DataFrame({"c": ["a"] * 400 + ["b"] * 400})
    shifted = pd.DataFrame({"c": ["a"] * 160 + ["b"] * 640})

    same = partition_drift(reference, identical)["categorical"][0]
    assert same["p_value"] > 0.5
    assert same["total_variation"] == 0.0

    moved = partition_drift(reference, shifted)["categorical"][0]
    assert moved["p_value"] < 1e-10
    assert moved["total_variation"] == pytest.approx(0.3, abs=0.02)
    assert moved["cramers_v"] > 0.1


def test_stability_is_scaled_by_the_iqr():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {"centred": rng.normal(size=800), "shifted": rng.normal(loc=50.0, size=800)}
    )
    report = stability(frame)
    # A zero-centred column must not explode the ratio: values are expressed as a
    # fraction of the column's IQR.
    assert report["max_relative_difference"] < 0.5
    assert report["effective_rows"] == 800
    for entry in report["columns"]:
        assert entry["relative_differences"]["mean"] < 0.5
        assert entry["scale"] > 0


def test_markdown_report_carries_every_section_and_scope():
    rng = np.random.default_rng(2)
    n = 120
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "cat": rng.choice(["u", "v"], size=n),
            "target": np.where(rng.normal(size=n) > 0, "yes", "no"),
        }
    )
    payload = {
        "task_name": "demo",
        "partition": "train",
        "target": "target",
        "rows": n,
        "columns": list(frame.columns),
        "ignored_columns": [],
        "feature_limit": 100,
        "row_sampling": {"method": "none"},
        "missing": {"missing_cells": 0, "total_cells": 3 * n, "missing_pct": 0.0,
                    "complete_rows": n, "complete_pct": 1.0, "target_missing": 0,
                    "columns": [{"column": "x", "dtype": "float64", "missing": 0,
                                 "missing_pct": 0.0, "unique": n}]},
        "column_profiles": [
            {"column": "x", "kind": "numeric",
             "distribution": {"min": -1.0, "q25": -0.5, "median": 0.0, "q75": 0.5, "max": 1.0}},
            {"column": "cat", "kind": "categorical",
             "distribution": {"top": [{"value": "u", "proportion": 0.5, "count": 60}]}},
        ],
        "target_associations": [{"feature": "x", "method": "correlation_ratio", "value": 0.1}],
        "target_composition": {
            "kind": "categorical",
            "distinct_classes": 2,
            "minority_share": 0.5,
            "imbalance_ratio": 1.0,
            "recommended_folds": 5,
            "stratification_feasible": True,
            "classes": [
                {"label": "yes", "count": 60, "share": 0.5},
                {"label": "no", "count": 60, "share": 0.5},
            ],
        },
        "multicollinearity": {"condition_number": 2.0, "vif": [{"feature": "x", "value": 1.1, "r_squared": 0.1}]},
        "quality_flags": {"flags": [], "duplicates": {"feature_duplicates": 0, "exact_rows": 0,
                                                      "conflicting_label_groups": 0,
                                                      "conflicting_label_rows": 0},
                          "scope": "sampled rows"},
        "missing_mechanism": {"columns": ["x"], "mcar": {"statistic": None, "reason": "no_missing_values"},
                              "shifts": [], "interpretation": "No missing values."},
        "type_audit": {"issues": [], "interpretation": "No mismatch."},
        "formal_tests": {"columns": [{"column": "x", "n": n, "statistic": 0.4, "p_value": 0.5,
                                      "critical_value": 0.75, "rejects_normality": False,
                                      "reason": None}],
                         "caveat": "Normal-like."},
        "effect_sizes": {"grouping": "target", "bootstrap": 200,
                         "groups": [{"label": "yes", "rows": 60}, {"label": "no", "rows": 60}],
                         "comparisons": [{"feature": "x", "group_a": "yes", "group_b": "no",
                                          "cohens_d": 0.2, "cohens_d_ci": [0.05, 0.35],
                                          "cliffs_delta": 0.1, "cliffs_delta_ci": [0.02, 0.2]}],
                         "interpretation": "Descriptive."},
        "stability": {"effective_rows": n, "rows": n, "max_relative_difference": 0.05,
                      "columns": [{"column": "x", "kind": "numeric", "max_relative_difference": 0.05}],
                      "interpretation": "Stable."},
    }
    markdown = markdown_report(payload, generated_at="2026-11-12T00:00:00Z")
    for heading in (
        "## Missingness",
        "## Target composition",
        "## Distributions",
        "## Associations",
        "## Collinearity",
        "## Data-quality flags",
        "## Missing-data mechanism",
        "## Formal tests",
        "## Effect sizes",
        "## Summary stability",
        "## Methods and limits",
    ):
        assert heading in markdown, heading
    assert "2026-11-12T00:00:00Z" in markdown
    assert "never modifies the partition" in markdown
    assert "demo" in markdown
