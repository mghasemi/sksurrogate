"""Pre-flight checks for experiment submission (advisory, never blocking)."""

import numpy as np
import pandas as pd
import pytest

from api.preflight import submit_preflight


def _dirty_frame(n=300, seed=5):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "row_id": np.arange(n),
            "surname": [f"name{i % 200}" for i in range(n)],
            "amount": np.round(100 + 15 * base, 3),
            "skewed": rng.lognormal(mean=0.0, sigma=1.2, size=n),
            "sparse": np.where(rng.random(n) < 0.65, np.nan, base),
            "label": np.where(base > 1.4, "rare", "common"),
        }
    )
    frame.loc[rng.choice(n, 40, replace=False), "amount"] = np.nan
    frame["amount_copy"] = frame["amount"]
    return frame


def test_preflight_reports_identifiers_imbalance_missing_and_skew():
    report = submit_preflight(_dirty_frame(), target="label")
    codes = {entry["code"] for entry in report["warnings"]} | {
        entry["code"] for entry in report["hints"]
    }
    assert "identifier_columns" in codes
    assert "class_imbalance" in codes
    assert "missing_cells" in codes or "high_missing_columns" in codes
    assert "skewed_features" in codes
    assert "near_duplicate_features" in codes
    assert report["blocking"] is False
    assert report["has_warnings"] is True
    # Warnings sort before hints.
    severities = [entry["severity"] for entry in report["messages"]]
    assert severities == sorted(severities, key={"err": 0, "warn": 1, "info": 2}.get)


def test_preflight_is_quiet_on_a_clean_frame():
    rng = np.random.default_rng(7)
    n = 400
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "label": np.where(rng.normal(size=n) > 0, "yes", "no"),
        }
    )
    report = submit_preflight(frame, target="label")
    assert report["warnings"] == []
    assert report["hints"] == []
    assert report["blocking"] is False
    assert report["effective_rows"] == n


def test_preflight_respects_ignored_columns():
    frame = _dirty_frame()
    without = submit_preflight(frame, target="label")
    ignored = submit_preflight(frame, target="label", ignored=["row_id", "surname"])
    without_codes = {entry["code"] for entry in without["messages"]}
    ignored_codes = {entry["code"] for entry in ignored["messages"]}
    assert "identifier_columns" in without_codes
    assert "identifier_columns" not in ignored_codes
    assert ignored["active_features"] == without["active_features"] - 2


def test_preflight_flags_small_effective_sample_and_conflicting_labels():
    frame = pd.DataFrame(
        {
            "x": [1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            "label": ["a", "b", "a", "b", "a", "b"],
        }
    )
    report = submit_preflight(frame, target="label")
    codes = {entry["code"] for entry in report["messages"]}
    assert "conflicting_labels" in codes
    assert "small_effective_sample" in codes


def test_preflight_rejects_bad_input():
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError):
        submit_preflight(frame, target="missing")
    empty = pd.DataFrame({"x": [], "y": []})
    with pytest.raises(ValueError):
        submit_preflight(empty, target="y")
