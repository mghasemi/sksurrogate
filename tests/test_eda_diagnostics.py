"""Tier-2 EDA diagnostics: quality flags, missingness mechanism, type audit.

The module under test is pure pandas/NumPy, so these tests build frames with
known problems and assert the honest signal (and the absence of false positives
on ordinary columns).
"""

import numpy as np
import pandas as pd
import pytest

from api.eda import little_mcar, missing_mechanism, quality_flags, semantic_audit


def _frame(n=400, seed=7):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 3))
    frame = pd.DataFrame(base, columns=["a", "b", "c"])
    frame["a_copy"] = frame["a"]
    frame["row_id"] = np.arange(n)
    frame["surname"] = ["name%d" % (index % 300) for index in range(n)]
    frame["constantish"] = np.where(rng.random(n) < 0.97, "same", "rare")
    frame["label"] = np.where(base[:, 0] > 0.4, "pos", "neg")
    frame["numeric_text"] = np.round(base[:, 1], 3).astype(str)
    frame["dateish"] = pd.date_range("2024-01-01", periods=n, freq="D").astype(str)
    frame.loc[rng.choice(n, 40, replace=False), "b"] = np.nan
    frame.loc[rng.choice(n, 30, replace=False), "c"] = np.nan
    return frame


def test_quality_flags_finds_identifiers_without_flagging_measurements():
    frame = _frame()
    report = quality_flags(frame, target="label")

    identifiers = {entry["column"] for entry in report["identifier_columns"]}
    # Name hints and row indices are identifiers; continuous measurements are not.
    assert {"row_id", "surname"} <= identifiers
    assert "a" not in identifiers and "b" not in identifiers
    assert "a_copy" not in identifiers

    # A strictly increasing integer row index is recognised even without an "id" name.
    numbered = frame.copy()
    numbered["sequence"] = np.arange(len(numbered))
    report = quality_flags(numbered.drop(columns=["row_id"]), target="label")
    reasons = {entry["column"]: entry["reason"] for entry in report["identifier_columns"]}
    assert "consecutive integer row index" in reasons["sequence"]


def test_quality_flags_separates_date_and_numeric_text_columns():
    report = quality_flags(_frame(), target="label")
    assert [entry["column"] for entry in report["date_like_columns"]] == ["dateish"]
    assert [entry["column"] for entry in report["numeric_text_columns"]] == ["numeric_text"]
    kinds = {(flag["kind"], flag["column"]) for flag in report["flags"]}
    assert ("date_like", "dateish") in kinds
    assert ("numeric_text", "numeric_text") in kinds
    assert ("identifier_like", "dateish") not in kinds


def test_quality_flags_detects_near_duplicates_quasi_constant_and_leakage():
    frame = _frame()
    report = quality_flags(
        frame,
        target="label",
        target_associations=[{"feature": "a", "method": "correlation_ratio", "value": 0.995}],
    )
    duplicates = {entry["feature"]: entry for entry in report["near_duplicate_columns"]}
    assert duplicates["a_copy"]["duplicate_of"] == "a"
    assert duplicates["a_copy"]["method"] in {"identical", "equal_after_rounding"}

    quasi = {entry["column"] for entry in report["quasi_constant"]}
    assert "constantish" in quasi

    leak_kinds = {(entry["kind"], entry["column"]) for entry in report["leakage"]}
    assert ("near_perfect_target_association", "a") in leak_kinds
    # The label is a threshold on `a`, so the class ranges are disjoint.
    assert any(kind == "disjoint_class_ranges" for kind, _ in leak_kinds)


def test_quality_flags_reports_conflicting_duplicate_labels():
    frame = pd.DataFrame(
        {
            "x": [1.0, 1.0, 2.0, 3.0],
            "y": [5.0, 5.0, 6.0, 7.0],
            "label": ["a", "b", "a", "b"],
        }
    )
    report = quality_flags(frame, target="label")
    assert report["duplicates"]["conflicting_label_groups"] == 1
    assert report["duplicates"]["conflicting_label_rows"] == 2
    assert any(flag["kind"] == "conflicting_labels" for flag in report["flags"])


def test_quality_flags_outlier_counts_use_iqr_and_mad():
    frame = pd.DataFrame(
        {"x": list(np.linspace(0.0, 1.0, 50)) + [1000.0], "label": ["a"] * 51}
    )
    report = quality_flags(frame, target="label")
    outliers = {entry["column"]: entry for entry in report["outliers"]}
    assert outliers["x"]["iqr_count"] >= 1
    assert outliers["x"]["mad_count"] >= 1
    assert outliers["x"]["mad_bound"] > 0


def test_quality_flags_is_empty_on_a_clean_frame():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=200),
            "y": rng.normal(size=200),
            "label": np.where(rng.normal(size=200) > 0, "a", "b"),
        }
    )
    report = quality_flags(frame, target="label")
    assert report["identifier_columns"] == []
    assert report["near_duplicate_columns"] == []
    assert report["duplicates"]["conflicting_label_groups"] == 0


def _mcar_like(seed=5, n=600):
    rng = np.random.default_rng(seed)
    full = rng.multivariate_normal(
        [0, 0, 0], [[1, 0.6, 0.4], [0.6, 1, 0.5], [0.4, 0.5, 1]], size=n
    )
    frame = pd.DataFrame(full, columns=["x", "y", "z"])
    frame.loc[rng.random(n) < 0.15, "x"] = np.nan
    frame.loc[rng.random(n) < 0.15, "y"] = np.nan
    return frame


def test_little_mcar_does_not_reject_under_mcar():
    frame = _mcar_like()
    matrix = frame.to_numpy(dtype=float)
    result = little_mcar(matrix, np.isfinite(matrix))
    assert result["reason"] is None
    assert result["df"] > 0
    assert result["p_value"] > 0.01


def test_little_mcar_rejects_when_missingness_depends_on_values():
    """Truncation of correlated variables shifts the pattern means.

    Little's test compares each missingness pattern's observed means with the ML
    estimates; dropping the upper tail of one correlated variable moves those
    means and the statistic grows. Note the limit: missingness that depends only
    on the missing variable's own value with independent columns is hard to
    detect (the test is a test of MCAR, not a general MNAR detector).
    """
    rng = np.random.default_rng(5)
    full = rng.multivariate_normal(
        [0, 0, 0], [[1, 0.6, 0.4], [0.6, 1, 0.5], [0.4, 0.5, 1]], size=600
    )
    full[full[:, 0] > 0.4, 0] = np.nan
    full[full[:, 1] < -0.4, 1] = np.nan
    result = little_mcar(full, np.isfinite(full))
    assert result["p_value"] < 1e-6
    assert result["statistic"] > result["df"]


def test_missing_mechanism_reports_scope_and_refusal_reasons():
    frame = _mcar_like()
    report = missing_mechanism(frame, target=None)
    assert report["columns"] == ["x", "y"]
    assert report["mcar"]["method"] == "little_mcar"
    assert report["scope"].startswith("sampled rows")

    single = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
    report = missing_mechanism(single, target=None)
    assert report["mcar"]["statistic"] is None
    assert report["mcar"]["reason"] == "fewer_than_two_columns_with_missing_values"

    complete = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    report = missing_mechanism(complete, target=None)
    assert report["mcar"]["reason"] == "no_missing_values"


def test_missing_mechanism_shift_points_at_the_missing_column():
    frame = _mcar_like()
    report = missing_mechanism(frame, target=None)
    # Every shift entry names the column whose missingness conditions the mean.
    assert report["shifts"]
    assert {entry["missing_column"] for entry in report["shifts"]} <= {"x", "y"}
    for entry in report["shifts"]:
        assert entry["rows_missing"] >= 5
        assert set(entry) == {
            "missing_column",
            "numeric_column",
            "mean_observed",
            "mean_missing",
            "standardized_difference",
            "rows_missing",
        }


def test_semantic_audit_flags_type_mismatches():
    frame = pd.DataFrame(
        {
            "stored_number": ["1.5", "2.5", "3.5"],
            "mixed": ["1", "2", "north"],
            "flag": ["yes", "no", "yes"],
            "when": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "messy": ["a", "a ", "A"],
            "ok": ["x", "y", "z"],
        }
    )
    report = semantic_audit(frame)
    issues = {(entry["column"], entry["issue"]) for entry in report["issues"]}
    assert ("stored_number", "numeric_text") in issues
    assert ("mixed", "mixed_type") in issues
    assert ("flag", "boolean_text") in issues
    assert ("when", "date_like_text") in issues
    assert ("messy", "whitespace") in issues
    assert ("messy", "case_inconsistent") in issues
    assert not any(column == "ok" for column, _ in issues)


def test_semantic_audit_skips_numeric_and_target_columns():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "label": ["a", "b", "a"]})
    report = semantic_audit(frame, target="label")
    assert report["issues"] == []
