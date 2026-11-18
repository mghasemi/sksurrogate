"""Non-blocking pre-flight checks for experiment submission.

The Data page explains a partition; this module answers the narrower question a
user asks when pressing *Find best model*: what is about to go wrong, without
blocking the run. Every message is advisory — the control plane never refuses a
submission because of a pre-flight finding, and nothing here modifies data.

Rules are derived from the same primitives the EDA page uses
(:func:`api.eda.quality_flags`, :func:`api.preprocessing.target_balance`) plus a
few cheap direct checks (missing-cell rate, per-column missing rate, skewness,
complete-case row count). Thresholds are returned with the messages so the UI can
explain why a hint fired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .eda import quality_flags
from .preprocessing import target_balance

#: A class ratio at or above this is reported as imbalance.
IMBALANCE_RATIO = 3.0
#: Missing-cell rate above this is worth a hint.
MISSING_CELL_RATE = 0.10
#: A single column missing more than this is worth a warning.
COLUMN_MISSING_RATE = 0.50
#: |skewness| above this suggests a transform or a tree-based model.
SKEWNESS_THRESHOLD = 2.0
#: Complete-case rows below this (or below 10x the feature count) are flagged.
MIN_EFFECTIVE_ROWS = 100
ROWS_PER_FEATURE = 10


def _is_numeric(series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series)) and not pd.api.types.is_bool_dtype(series)


def submit_preflight(frame, *, target, ignored=None, max_rows=5000, random_state=0):
    """Return advisory warnings and hints for a pending experiment submission.

    ``blocking`` is always ``False``: this is advice, not validation. Messages
    name the offending columns (capped) so the user can act on them.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("frame must contain at least one row")
    if target not in frame.columns:
        raise ValueError("target column %r is not in frame" % target)
    ignored = set(ignored or [])
    active = [column for column in frame.columns if column != target and column not in ignored]
    if not active:
        raise ValueError("no active feature columns to check")

    sample = frame
    sampled = False
    if max_rows and len(frame) > max_rows > 0:
        sample = frame.sample(max_rows, random_state=random_state)
        sampled = True

    flags = quality_flags(frame, target=target, ignored=ignored, max_rows=max_rows, random_state=random_state)
    composition = target_balance(frame, target=target)

    messages = []

    def add(code, severity, message, columns=None):
        messages.append(
            {"code": code, "severity": severity, "message": message, "columns": columns or []}
        )

    identifiers = [entry["column"] for entry in flags["identifier_columns"]]
    if identifiers:
        add(
            "identifier_columns",
            "warn",
            "%d identifier-like column(s) are part of the modeling features; they add "
            "noise and can inflate runtime. Consider ignoring them." % len(identifiers),
            identifiers[:5],
        )
    if flags["high_cardinality"]:
        names = [entry["column"] for entry in flags["high_cardinality"]]
        add(
            "high_cardinality",
            "warn",
            "%d text column(s) have at least half as many levels as rows; one-hot "
            "encoding will be wide." % len(names),
            names[:5],
        )
    near_duplicates = [entry["feature"] for entry in flags["near_duplicate_columns"]]
    if near_duplicates:
        add(
            "near_duplicate_features",
            "info",
            "%d near-duplicate feature column(s) detected; one of each pair is usually "
            "enough." % len(near_duplicates),
            near_duplicates[:5],
        )
    duplicates = flags["duplicates"]
    if duplicates["conflicting_label_groups"]:
        add(
            "conflicting_labels",
            "warn",
            "%d duplicate feature group(s) carry more than one label; that is a leakage "
            "signal and can flatter cross-validation." % duplicates["conflicting_label_groups"],
        )

    if composition.get("kind") == "categorical":
        ratio = composition.get("imbalance_ratio")
        minority = composition.get("minority_share") or 0.0
        if ratio is not None and ratio >= IMBALANCE_RATIO:
            add(
                "class_imbalance",
                "warn",
                "Class imbalance is %.1f:1 (minority %.1f%%); accuracy will favour the "
                "majority class. Consider a balanced metric or class weights."
                % (ratio, minority * 100),
                [(composition.get("classes") or [{}])[0].get("label")],
            )
        if composition.get("recommended_folds") and composition["recommended_folds"] < 5:
            add(
                "limited_stratification",
                "info",
                "The smallest class allows at most %d stratified folds."
                % composition["recommended_folds"],
            )

    feature_frame = sample[active]
    cells = int(feature_frame.size)
    missing_cells = int(feature_frame.isna().sum().sum())
    missing_rate = missing_cells / cells if cells else 0.0
    if missing_rate > MISSING_CELL_RATE:
        add(
            "missing_cells",
            "info",
            "%.1f%% of feature cells are missing; imputation is applied inside each "
            "training fold, so the reported metric reflects the filled data." % (missing_rate * 100),
        )
    heavy = [
        column
        for column in active
        if sample[column].isna().mean() > COLUMN_MISSING_RATE
    ]
    if heavy:
        add(
            "high_missing_columns",
            "warn",
            "%d column(s) are missing more than %.0f%% of rows; imputing them creates "
            "mostly synthetic values." % (len(heavy), COLUMN_MISSING_RATE * 100),
            heavy[:5],
        )

    numeric_active = [column for column in active if _is_numeric(sample[column])]
    skewed = []
    for column in numeric_active:
        values = pd.to_numeric(sample[column], errors="coerce").dropna()
        if len(values) < 8 or values.nunique() < 3:
            continue
        skewness = float(values.skew())
        if np.isfinite(skewness) and abs(skewness) > SKEWNESS_THRESHOLD:
            skewed.append((column, skewness))
    if skewed:
        add(
            "skewed_features",
            "info",
            "%d numeric column(s) are strongly skewed (|skew| > %.0f); a transform or a "
            "tree-based model may fit better than a linear one." % (len(skewed), SKEWNESS_THRESHOLD),
            [column for column, _ in skewed[:5]],
        )

    complete_rows = int(feature_frame.dropna().shape[0]) if len(numeric_active) else int(len(sample))
    effective_floor = max(MIN_EFFECTIVE_ROWS, len(active) * ROWS_PER_FEATURE)
    if complete_rows < effective_floor:
        add(
            "small_effective_sample",
            "warn",
            "Only %d complete-case row(s) across %d feature(s) (guideline: %d). "
            "Cross-validation will be noisy." % (complete_rows, len(active), effective_floor),
        )

    constants = [
        column for column in active if sample[column].nunique(dropna=True) <= 1
    ]
    if constants:
        add(
            "constant_features",
            "info",
            "%d constant feature(s) carry no information and are skipped by the search."
            % len(constants),
            constants[:5],
        )

    severity_order = {"err": 0, "warn": 1, "info": 2}
    messages.sort(key=lambda entry: severity_order.get(entry["severity"], 3))
    warnings = [entry for entry in messages if entry["severity"] in {"err", "warn"}]
    hints = [entry for entry in messages if entry["severity"] == "info"]
    return {
        "rows": int(len(sample)),
        "sampled": bool(sampled),
        "target": target,
        "active_features": len(active),
        "effective_rows": complete_rows,
        "complete_rows": complete_rows,
        "messages": messages,
        "warnings": warnings,
        "hints": hints,
        "has_warnings": bool(warnings),
        "blocking": False,
        "scope": "Sampled rows and active (non-ignored, non-target) features.",
        "thresholds": {
            "imbalance_ratio": IMBALANCE_RATIO,
            "missing_cell_rate": MISSING_CELL_RATE,
            "column_missing_rate": COLUMN_MISSING_RATE,
            "skewness": SKEWNESS_THRESHOLD,
            "min_effective_rows": MIN_EFFECTIVE_ROWS,
            "rows_per_feature": ROWS_PER_FEATURE,
        },
        "interpretation": (
            "Advisory only: the experiment still runs. Warnings point at data issues "
            "that change how a score should be read; hints describe mechanics."
        ),
    }


__all__ = ["submit_preflight"]
