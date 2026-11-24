"""Leakage diagnostics for a fitted experiment's split.

The pipeline itself is leakage-safe by construction: preprocessing lives inside
the searched ``Pipeline`` and is therefore fitted in every CV fold (pinned by
``tests/test_leakage_guards.py``). What the code cannot prevent is a *split* or a
*column* that leaks, so this module reports the three cheap, high-signal cases the
review asked for:

1. **Duplicate / near-duplicate rows across train and validation.** Two rows with
   the same rounded feature vector but different splits inflate a validation score
   without any learning. Hashing rounded numeric columns plus the remaining raw
   columns is an exact duplicate check for the rounded representation and a
   near-duplicate check for continuous noise below 1e-6.
2. **Target-leakage candidate columns.** A single feature that predicts the target
   almost perfectly inside the training partition (AUC > 0.98 either direction, or
   |Pearson r| > 0.995 for a regression target) is a candidate for being target-
   derived — an ID, a post-outcome field, or an accidental copy.
3. **Temporal-split suspicion.** A monotone datetime column plus a random split
   means later observations train a model scored on earlier ones (or vice versa);
   the report names the column instead of guessing the user's intent.

Every check is advisory. The block is returned with the experiment payload and
stored in the run manifest, so a reader sees the caveat next to the score rather
than having to know to ask for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: A single column must beat this AUC (in either direction) to be a candidate.
AUC_THRESHOLD = 0.98
#: Absolute Pearson correlation above this marks a regression-target candidate.
CORRELATION_THRESHOLD = 0.995
#: Scan at most this many feature columns: a wide frame must not turn a fit into
#: a statistics job, and the first columns are the ones a user put there first.
MAX_SCANNED_FEATURES = 256
#: Numeric rounding for the duplicate hash ("near-duplicate" at 1e-6).
ROUND_DECIMALS = 6

_CAVEATS = (
    "Leakage checks are advisory heuristics, not proof that a split or column is "
    "sound. A candidate column deserves inspection, not automatic removal.",
    "Duplicate rows across train and validation make the validation score "
    "optimistic: the identical record was already seen during fitting.",
    "A single feature with near-perfect predictive power inside the training "
    "partition is often target-derived (an identifier or a post-outcome field).",
    "A monotone datetime column with a random split means the model was scored on "
    "observations that precede its training rows; use a temporal splitter.",
)


def _feature_frame(frame, target):
    columns = [column for column in frame.columns if column != target]
    return frame[columns]


def _row_hashes(frame):
    """Stable hashes of each row, after rounding numeric columns."""
    numeric = frame.select_dtypes(include=[np.number]).round(ROUND_DECIMALS)
    rest = frame.drop(columns=list(numeric.columns), errors="ignore")
    combined = pd.concat([numeric, rest], axis=1)
    return pd.util.hash_pandas_object(combined, index=False, categorize=True)


def duplicate_overlap(train_frame, validation_frame, *, target=None):
    """Count rows whose rounded feature vector occurs on both sides of the split."""
    if validation_frame is None:
        return {
            "checked": False,
            "reason": "no validation partition is registered for this task",
            "train_rows": int(len(train_frame)),
            "validation_rows": None,
            "duplicate_rows": None,
        }
    train_features = _feature_frame(train_frame, target)
    validation_features = _feature_frame(validation_frame, target)
    train_hashes = set(_row_hashes(train_features).tolist())
    validation_hashes = _row_hashes(validation_features)
    duplicates = int(np.isin(validation_hashes.to_numpy(), list(train_hashes)).sum())
    validation_rows = int(len(validation_features))
    return {
        "checked": True,
        "train_rows": int(len(train_features)),
        "validation_rows": validation_rows,
        "duplicate_rows": duplicates,
        "duplicate_fraction_of_validation": (
            round(duplicates / validation_rows, 6) if validation_rows else 0.0
        ),
        "rounding_decimals": ROUND_DECIMALS,
    }


def _binary_target(target_values):
    """Return boolean positive-class mask when the target is a classification.

    ``None`` means "treat as a regression target". A numeric target with more
    than ten distinct values is a regression; a string/categorical target is a
    classification even with many levels (the AUC is then one-vs-rest, which is
    still the right leakage signal for the majority class).
    """
    series = pd.Series(target_values).dropna()
    if series.empty:
        return None
    if not pd.api.types.is_numeric_dtype(series.dtype):
        return series
    distinct = series.nunique()
    if distinct <= 10 and np.all(np.equal(np.mod(series.astype(float), 1), 0)):
        return series
    return None


def _auc(values, positive_mask):
    """Mann-Whitney AUC of ``values`` against a boolean positive mask."""
    frame = pd.DataFrame({"value": values, "positive": positive_mask}).dropna()
    if frame.empty:
        return None
    n_pos = int(frame["positive"].sum())
    n_neg = int(len(frame) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = frame["value"].rank().to_numpy()
    positive_ranks = ranks[frame["positive"].to_numpy()]
    return float(
        (positive_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def _encoded_column(series):
    """Numeric view of a feature column, or ``None`` when it cannot be ranked."""
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series.astype(float)
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return series.astype("int64").astype(float)
    codes, _ = pd.factorize(series, sort=False)
    codes = pd.Series(codes, index=series.index).astype(float)
    codes[codes < 0] = np.nan
    return codes


def target_leakage_candidates(frame, target):
    """Rank features by single-column predictive power; return the suspicious ones.

    Classification targets use AUC (a candidate is anything whose AUC beats
    :data:`AUC_THRESHOLD` in *either* direction). Regression targets use absolute
    Pearson correlation. The check runs on the training partition only: a column
    the model can learn from is what matters, not what a held-out split shows.
    """
    if target not in frame.columns:
        return {"checked": False, "reason": "target column is absent from the frame"}
    target_values = frame[target]
    features = _feature_frame(frame, target)
    # Limit the scan before any per-column work happens.
    features = features.iloc[:, :MAX_SCANNED_FEATURES]
    classification = _binary_target(target_values)
    candidates = []
    if classification is not None:
        # Majority class carries the signal for a multi-class target.
        mode = classification.mode()
        if mode.empty:
            return {"checked": False, "reason": "target has no observed values"}
        positive = (classification == mode.iloc[0]).to_numpy()
        for column in features.columns:
            values = _encoded_column(frame[column]).reindex(frame.index)
            auc = _auc(values, positive)
            if auc is None:
                continue
            strength = max(auc, 1.0 - auc)
            if strength >= AUC_THRESHOLD:
                candidates.append(
                    {
                        "column": column,
                        "statistic": "auc",
                        "value": round(float(auc), 6),
                        "direction": "positive" if auc >= 0.5 else "inverse",
                        "threshold": AUC_THRESHOLD,
                    }
                )
        method = "AUC against the majority class"
    else:
        numeric_target = pd.to_numeric(target_values, errors="coerce")
        for column in features.columns:
            if not pd.api.types.is_numeric_dtype(frame[column].dtype):
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            pair = pd.DataFrame({"x": values, "y": numeric_target}).dropna()
            if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
                continue
            correlation = float(pair["x"].corr(pair["y"]))
            if not np.isfinite(correlation):
                continue
            if abs(correlation) >= CORRELATION_THRESHOLD:
                candidates.append(
                    {
                        "column": column,
                        "statistic": "pearson",
                        "value": round(correlation, 6),
                        "direction": "positive" if correlation >= 0 else "inverse",
                        "threshold": CORRELATION_THRESHOLD,
                    }
                )
        method = "absolute Pearson correlation"
    candidates.sort(key=lambda entry: abs(entry["value"]), reverse=True)
    return {
        "checked": True,
        "method": method,
        "scanned_columns": int(features.shape[1]),
        "candidates": candidates,
    }


def temporal_split_suspected(frame):
    """Name monotone datetime columns: a hint that the split should be temporal."""
    columns = []
    for column in frame.columns:
        series = frame[column]
        if not pd.api.types.is_datetime64_any_dtype(series.dtype):
            continue
        observed = series.dropna()
        if len(observed) < 2:
            continue
        if observed.is_monotonic_increasing or observed.is_monotonic_decreasing:
            columns.append(column)
    return {
        "checked": True,
        "columns": columns,
        "suspected": bool(columns),
    }


def detect_leakage_warnings(train_frame, validation_frame=None, *, target=None, train_partition="train", validation_partition=None):
    """Assemble the leakage-warnings block stored with an experiment.

    Never raises: a diagnostic that can break a completed fit is worse than a
    missing diagnostic, so any internal error is reported in the block.
    """
    try:
        return {
            "checked": True,
            "train_partition": train_partition,
            "validation_partition": validation_partition,
            "duplicate_rows": duplicate_overlap(train_frame, validation_frame, target=target),
            "target_leakage": target_leakage_candidates(train_frame, target) if target else {"checked": False, "reason": "no target column"},
            "temporal_split": temporal_split_suspected(_feature_frame(train_frame, target) if target else train_frame),
            "caveats": list(_CAVEATS),
        }
    except Exception as exc:  # pragma: no cover - defensive by design
        return {
            "checked": False,
            "reason": "leakage diagnostics failed: %s" % exc,
            "caveats": list(_CAVEATS),
        }
