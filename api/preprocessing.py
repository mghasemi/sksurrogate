"""Missing-value analysis and materialized imputation for one partition.

The Data Preprocessing page reports what a registered partition looks like and
can write an *imputed copy* of it as a new partition. The source partition is
never modified: imputation changes the data fingerprint, and downstream runs
must be able to point at exactly which version of the data they used.

Imputation methods are the sklearn primitives the package already depends on:

* ``drop_rows`` — remove rows with a missing value in any selected column;
* ``mean`` / ``median`` — numeric columns only;
* ``most_frequent`` — any dtype;
* ``constant`` — any dtype, with a caller-supplied fill value;
* ``knn`` — numeric columns only (``KNNImputer``);
* ``iterative`` — numeric columns only (``IterativeImputer``, MICE-style).

The target column is never imputed. Ignored columns are excluded from the
default selection and refused when named explicitly: they are not part of any
modeling path, so silently imputing them would create a false sense of a clean
dataset.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

#: Ordered method catalogue exposed to the UI.
METHODS = [
    {
        "name": "drop_rows",
        "label": "Drop rows with missing values",
        "numeric_only": False,
        "params": [],
    },
    {
        "name": "mean",
        "label": "Mean (numeric columns)",
        "numeric_only": True,
        "params": [],
    },
    {
        "name": "median",
        "label": "Median (numeric columns)",
        "numeric_only": True,
        "params": [],
    },
    {
        "name": "most_frequent",
        "label": "Most frequent value",
        "numeric_only": False,
        "params": [],
    },
    {
        "name": "constant",
        "label": "Constant fill value",
        "numeric_only": False,
        "params": [
            {
                "name": "constant_value",
                "kind": "string",
                "required": True,
                "help": "Value used for every selected column.",
            }
        ],
    },
    {
        "name": "knn",
        "label": "K-nearest neighbours (numeric columns)",
        "numeric_only": True,
        "params": [
            {
                "name": "n_neighbors",
                "kind": "int",
                "default": 5,
                "min": 1,
                "max": 50,
            }
        ],
    },
    {
        "name": "iterative",
        "label": "Iterative / MICE (numeric columns)",
        "numeric_only": True,
        "params": [
            {
                "name": "max_iter",
                "kind": "int",
                "default": 10,
                "min": 1,
                "max": 100,
            }
        ],
    },
]

_METHOD_NAMES = {method["name"] for method in METHODS}


class ImputationError(ValueError):
    """Caller-facing validation error (mapped to HTTP 400)."""


def _finite(value):
    """JSON-safe float: NaN/inf become ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_numeric(series):
    return bool(
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_datetime64_any_dtype(series)
    )


def _numeric_summary(series):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return {
        "count": int(numeric.count()),
        "mean": _finite(numeric.mean()),
        "std": _finite(numeric.std()),
        "min": _finite(numeric.min()),
        "q25": _finite(numeric.quantile(0.25)),
        "q50": _finite(numeric.quantile(0.50)),
        "q75": _finite(numeric.quantile(0.75)),
        "max": _finite(numeric.max()),
    }


def _top_values(series, limit=5):
    counts = series.dropna().astype(str).value_counts().head(limit)
    total = max(int(series.notna().sum()), 1)
    return [
        {"value": value, "count": int(count), "share": float(count / total)}
        for value, count in counts.items()
    ]


def missing_overview(frame, *, target, ignored, deduced_types=None):
    """Per-column missingness plus row-level completeness for the whole frame."""
    deduced_types = deduced_types or {}
    ignored = set(ignored or [])
    rows = int(len(frame))
    columns = []
    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        columns.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "deduced_type": deduced_types.get(column),
                "ignored": column in ignored,
                "is_target": column == target,
                "missing": missing,
                "missing_pct": float(missing / rows) if rows else 0.0,
                "unique": unique,
                "constant": bool(unique == 1),
                "all_missing": bool(rows and missing == rows),
                "numeric_summary": _numeric_summary(series) if _is_numeric(series) else None,
                "top_values": None if _is_numeric(series) else _top_values(series),
            }
        )
    feature_columns = [column for column in frame.columns if column != target]
    cells = rows * len(frame.columns)
    missing_cells = int(frame.isna().sum().sum())
    rows_with_any = int(frame[feature_columns].isna().any(axis=1).sum()) if feature_columns else 0
    return {
        "rows": rows,
        "columns": columns,
        "total_cells": cells,
        "missing_cells": missing_cells,
        "missing_pct": float(missing_cells / cells) if cells else 0.0,
        "rows_with_any_missing": rows_with_any,
        "rows_with_any_missing_pct": float(rows_with_any / rows) if rows else 0.0,
        "complete_rows": rows - rows_with_any,
        "complete_pct": float((rows - rows_with_any) / rows) if rows else 0.0,
        "target_missing": int(frame[target].isna().sum()) if target in frame.columns else 0,
    }


def _validate_selection(frame, *, target, ignored, columns):
    """Resolve the selected imputation columns and validate them."""
    ignored = set(ignored or [])
    if columns is None:
        selected = [
            column
            for column in frame.columns
            if column != target and column not in ignored and frame[column].isna().any()
        ]
    else:
        if not isinstance(columns, (list, tuple)) or not all(
            isinstance(column, str) for column in columns
        ):
            raise ImputationError("columns must be a list of column names")
        selected = []
        for column in columns:
            if column not in frame.columns:
                raise ImputationError("Unknown column %r" % column)
            if column == target:
                raise ImputationError("The target column %r cannot be imputed" % column)
            if column in ignored:
                raise ImputationError(
                    "Column %r is ignored for this dataset; re-enable it before imputing" % column
                )
            if column not in selected:
                selected.append(column)
    if not selected:
        raise ImputationError(
            "No columns with missing values were selected"
        )
    return selected


def _require_numeric(frame, columns, method):
    non_numeric = [column for column in columns if not _is_numeric(frame[column])]
    if non_numeric:
        raise ImputationError(
            "Method %r requires numeric columns; non-numeric selected: %s. "
            "Use most_frequent/constant or exclude those columns."
            % (method, ", ".join(repr(column) for column in non_numeric))
        )


def _apply_imputation(frame, selected, method, params):
    """Return the imputed copy; raises :class:`ImputationError` on bad input."""
    if method not in _METHOD_NAMES:
        raise ImputationError(
            "method must be one of %s (got %r)"
            % (", ".join(method["name"] for method in METHODS), method)
        )
    params = params or {}
    if method == "drop_rows":
        return frame.dropna(subset=selected).copy()

    numeric_methods = {"mean", "median", "knn", "iterative"}
    if method in numeric_methods:
        _require_numeric(frame, selected, method)

    result = frame.copy()
    if method in {"mean", "median"}:
        imputer = SimpleImputer(strategy=method)
        result[selected] = imputer.fit_transform(result[selected])
        return result

    if method == "most_frequent":
        imputer = SimpleImputer(strategy="most_frequent")
        result[selected] = imputer.fit_transform(result[selected])
        return result

    if method == "constant":
        value = params.get("constant_value")
        if value is None or str(value).strip() == "":
            raise ImputationError("constant_value is required for the constant method")
        for column in selected:
            if _is_numeric(result[column]):
                try:
                    fill = float(value)
                except (TypeError, ValueError):
                    raise ImputationError(
                        "constant_value %r is not numeric; column %r is numeric"
                        % (value, column)
                    )
                result[column] = result[column].fillna(fill)
            else:
                result[column] = result[column].fillna(str(value))
        return result

    if method == "knn":
        neighbours = params.get("n_neighbors", 5)
        if isinstance(neighbours, bool) or not isinstance(neighbours, int) or not 1 <= neighbours <= 50:
            raise ImputationError("n_neighbors must be an integer between 1 and 50")
        imputer = KNNImputer(n_neighbors=neighbours)
        values = imputer.fit_transform(result[selected])
        result[selected] = values
        return result

    max_iter = params.get("max_iter", 10)
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or not 1 <= max_iter <= 100:
        raise ImputationError("max_iter must be an integer between 1 and 100")
    imputer = IterativeImputer(max_iter=max_iter, random_state=0)
    values = imputer.fit_transform(result[selected])
    result[selected] = values
    return result


def impute_partition(
    frame,
    *,
    target,
    ignored,
    method,
    columns=None,
    params=None,
    drop_missing_target=False,
):
    """Impute a copy of ``frame`` and return ``(frame, report)``.

    The comparison report is computed on the *selected* columns only: it states
    how many cells changed and how the marginal distribution moved, which is the
    honest measure of what a materialized imputation did to the data.
    """
    if target not in frame.columns:
        raise ImputationError("target column %r is not present in the partition" % target)
    selected = _validate_selection(
        frame, target=target, ignored=ignored, columns=columns
    )
    before = {column: int(frame[column].isna().sum()) for column in selected}
    rows_before = int(len(frame))

    result = _apply_imputation(frame, selected, method, params)
    rows_dropped = rows_before - int(len(result))
    if drop_missing_target and target in result.columns:
        before_target_drop = int(len(result))
        result = result.dropna(subset=[target]).copy()
        rows_dropped += before_target_drop - int(len(result))

    after = {column: int(result[column].isna().sum()) for column in selected}
    numeric_shift = {}
    categorical_shift = {}
    for column in selected:
        original = frame[column]
        updated = result[column]
        if _is_numeric(original):
            numeric_shift[column] = {
                "before": _numeric_summary(original),
                "after": _numeric_summary(updated),
            }
        else:
            categorical_shift[column] = {
                "before": _top_values(original),
                "after": _top_values(updated),
            }

    report = {
        "method": method,
        "columns": selected,
        "rows_before": rows_before,
        "rows_after": int(len(result)),
        "rows_dropped": int(rows_dropped),
        "missing_before": before,
        "missing_after": after,
        "cells_imputed": int(sum(before.values()) - sum(after.values())),
        "target_missing_before": int(frame[target].isna().sum()),
        "target_missing_after": int(result[target].isna().sum()),
        "numeric_shift": numeric_shift,
        "categorical_shift": categorical_shift,
    }
    return result, report


class ClipError(ValueError):
    """Caller-facing outlier-clipping validation error (mapped to HTTP 400)."""


#: Ordered clipping catalogue exposed to the UI.
CLIP_METHODS = [
    {
        "name": "iqr",
        "label": "IQR fences (Q1 − k·IQR, Q3 + k·IQR)",
        "params": [
            {"name": "factor", "kind": "float", "default": 1.5, "min": 0.1, "max": 10.0,
             "help": "Tukey multiplier k; 1.5 is the classic fence, 3.0 flags only far outliers."}
        ],
    },
    {
        "name": "quantile",
        "label": "Quantile winsorizing",
        "params": [
            {"name": "lower", "kind": "float", "default": 0.01, "min": 0.0, "max": 0.49,
             "help": "Lower tail quantile; 0.01 clips the bottom 1%."},
            {"name": "upper", "kind": "float", "default": 0.99, "min": 0.51, "max": 1.0,
             "help": "Upper tail quantile; 0.99 clips the top 1%."},
        ],
    },
    {
        "name": "zscore",
        "label": "Z-score clipping",
        "params": [
            {"name": "threshold", "kind": "float", "default": 3.0, "min": 0.5, "max": 10.0,
             "help": "Clip values whose absolute z-score exceeds this threshold."}
        ],
    },
]

_CLIP_METHOD_NAMES = {method["name"] for method in CLIP_METHODS}


def _as_float(params, name, default, minimum, maximum):
    value = (params or {}).get(name, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ClipError("%s must be a number" % name)
    if not (minimum <= value <= maximum):
        raise ClipError("%s must be between %s and %s" % (name, minimum, maximum))
    return value


def _clip_bounds(series, method, params):
    """Return finite ``(lower, upper)`` clipping bounds for one numeric column."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ClipError("column %r has no numeric values to clip" % series.name)
    if method == "iqr":
        factor = _as_float(params, "factor", 1.5, 0.1, 10.0)
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        return q1 - factor * iqr, q3 + factor * iqr
    if method == "quantile":
        lower = _as_float(params, "lower", 0.01, 0.0, 0.49)
        upper = _as_float(params, "upper", 0.99, 0.51, 1.0)
        if not lower < upper:
            raise ClipError("lower must be smaller than upper")
        return float(values.quantile(lower)), float(values.quantile(upper))
    if method == "zscore":
        threshold = _as_float(params, "threshold", 3.0, 0.5, 10.0)
        mean = float(values.mean())
        std = float(values.std())
        if not math.isfinite(std) or std <= 0:
            raise ClipError("column %r has zero variance; z-score clipping is undefined" % series.name)
        return mean - threshold * std, mean + threshold * std
    raise ClipError(
        "method must be one of %s (got %r)"
        % (", ".join(method["name"] for method in CLIP_METHODS), method)
    )


def _validate_clip_selection(frame, *, target, ignored, columns):
    ignored = set(ignored or [])
    available = [
        column
        for column in frame.columns
        if column != target and column not in ignored and _is_numeric(frame[column])
    ]
    if columns is None:
        selected = available
    else:
        if not isinstance(columns, (list, tuple)) or not all(isinstance(c, str) for c in columns):
            raise ClipError("columns must be a list of column names")
        selected = []
        for column in columns:
            if column not in frame.columns:
                raise ClipError("Unknown column %r" % column)
            if column == target:
                raise ClipError("The target column %r cannot be clipped" % column)
            if column in ignored:
                raise ClipError(
                    "Column %r is ignored for this dataset; re-enable it before clipping" % column
                )
            if column not in available:
                raise ClipError(
                    "Column %r is not numeric; outlier clipping applies to numeric features only"
                    % column
                )
            if column not in selected:
                selected.append(column)
    if not selected:
        raise ClipError("No numeric feature columns were selected")
    return selected


def clip_partition(frame, *, target, ignored, method, columns=None, params=None):
    """Winsorize numeric features and return ``(frame, report)``.

    Values below/above the column's bounds are replaced by the bounds. The report
    states the bounds, how many values each tail clipped, and the change in the
    column's marginal distribution.
    """
    if method not in _CLIP_METHOD_NAMES:
        raise ClipError(
            "method must be one of %s (got %r)"
            % (", ".join(entry["name"] for entry in CLIP_METHODS), method)
        )
    if target not in frame.columns:
        raise ClipError("target column %r is not present in the partition" % target)
    selected = _validate_clip_selection(frame, target=target, ignored=ignored, columns=columns)
    result = frame.copy()
    per_column = {}
    total_clipped = 0
    for column in selected:
        series = result[column]
        lower, upper = _clip_bounds(series, method, params)
        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise ClipError("column %r produced non-finite clipping bounds" % column)
        values = pd.to_numeric(series, errors="coerce")
        below = int((values < lower).sum())
        above = int((values > upper).sum())
        clipped = below + above
        total_clipped += clipped
        result[column] = values.clip(lower=lower, upper=upper)
        per_column[column] = {
            "lower": _finite(lower),
            "upper": _finite(upper),
            "clipped_low": below,
            "clipped_high": above,
            "clipped": clipped,
            "clipped_pct": float(clipped / max(int(values.notna().sum()), 1)),
            "before": _numeric_summary(frame[column]),
            "after": _numeric_summary(result[column]),
        }
    return result, {
        "method": method,
        "params": dict(params or {}),
        "columns": selected,
        "rows": int(len(frame)),
        "cells_clipped": int(total_clipped),
        "columns_report": per_column,
    }


def _target_distribution(series):
    """Compact JSON-safe target distribution for the duplicate-impact report."""
    counts = series.dropna()
    if _is_numeric(counts) and counts.nunique() > 10:
        return {"kind": "numeric", "summary": _numeric_summary(counts)}
    values = _top_values(counts, limit=12)
    return {"kind": "categorical", "values": values}


def _distribution_delta(before, after):
    """Largest absolute change in a categorical/numeric marginal."""
    if before.get("kind") == "categorical":
        before_map = {entry["value"]: entry["share"] for entry in before.get("values", [])}
        after_map = {entry["value"]: entry["share"] for entry in after.get("values", [])}
        deltas = []
        for value in set(before_map) | set(after_map):
            deltas.append((value, after_map.get(value, 0.0) - before_map.get(value, 0.0)))
        if not deltas:
            return {"value": None, "share_before": 0.0, "share_after": 0.0, "delta": 0.0}
        value, delta = max(deltas, key=lambda item: abs(item[1]))
        return {
            "value": value,
            "share_before": before_map.get(value, 0.0),
            "share_after": after_map.get(value, 0.0),
            "delta": delta,
        }
    before_mean = (before.get("summary") or {}).get("mean")
    after_mean = (after.get("summary") or {}).get("mean")
    if before_mean is None or after_mean is None:
        return None
    return {
        "value": "mean",
        "share_before": before_mean,
        "share_after": after_mean,
        "delta": after_mean - before_mean,
    }


def duplicate_analysis(
    frame,
    *,
    target,
    ignored,
    columns=None,
    keep="first",
    max_rows=20000,
    random_state=0,
):
    """Assess exact duplicate rows and the distributional cost of removing them.

    Duplicates are judged on the **feature** columns, not on the target: rows with
    identical features but different labels are the dangerous case (leakage across
    a random split or contradictory labels), and full-row duplicates are reported
    separately. The report is descriptive only; ``drop_duplicate_rows`` performs
    the removal.
    """
    ignored = set(ignored or [])
    if target not in frame.columns:
        raise ClipError("target column %r is not present in the partition" % target)
    available = [column for column in frame.columns if column != target and column not in ignored]
    if columns is None:
        features = available
    else:
        if not isinstance(columns, (list, tuple)) or not all(isinstance(c, str) for c in columns):
            raise ClipError("columns must be a list of column names")
        features = []
        for column in columns:
            if column not in frame.columns:
                raise ClipError("Unknown column %r" % column)
            if column == target:
                raise ClipError("The target column cannot be part of the duplicate key")
            if column in ignored:
                raise ClipError("Column %r is ignored for this dataset" % column)
            if column not in features:
                features.append(column)
    if not features:
        raise ClipError("No feature columns available to define duplicate rows")
    if keep not in {"first", "last"}:
        raise ClipError("keep must be 'first' or 'last'")

    sampled = False
    working = frame
    if max_rows and len(frame) > max_rows > 0:
        working = frame.sample(max_rows, random_state=random_state)
        sampled = True

    duplicate_mask = working.duplicated(subset=features, keep=False)
    duplicate_groups = int(working.loc[duplicate_mask, features].drop_duplicates().shape[0])
    duplicate_rows = int(duplicate_mask.sum())
    removable_mask = working.duplicated(subset=features, keep=keep)
    removable = int(removable_mask.sum())
    full_duplicate_rows = int(working.duplicated(keep=False).sum())

    conflicting_groups = 0
    conflicting_rows = 0
    if target in working.columns and duplicate_rows:
        grouped = working.loc[duplicate_mask].groupby(features, dropna=False)[target].nunique(dropna=True)
        conflicting_groups = int((grouped > 1).sum())
        if conflicting_groups:
            conflicting_keys = set(grouped[grouped > 1].index)
            keys = list(working.loc[duplicate_mask, features].itertuples(index=False, name=None))
            conflicting_rows = int(sum(1 for key in keys if key in conflicting_keys))

    deduplicated = working.drop_duplicates(subset=features, keep=keep)
    rows_before = int(len(working))
    rows_after = int(len(deduplicated))
    removed = rows_before - rows_after
    target_before = _target_distribution(working[target])
    target_after = _target_distribution(deduplicated[target])
    target_delta = _distribution_delta(target_before, target_after)

    numeric_shift = {}
    for column in features:
        if _is_numeric(working[column]):
            before = _numeric_summary(working[column])
            after = _numeric_summary(deduplicated[column])
            if before and after and before.get("mean") is not None and after.get("mean") is not None:
                change = after["mean"] - before["mean"]
                numeric_shift[column] = {
                    "before": before,
                    "after": after,
                    "mean_delta": change,
                    "mean_delta_pct": (
                        change / abs(before["mean"]) if before["mean"] not in (None, 0) else None
                    ),
                }

    warnings = []
    if removed:
        warnings.append(
            "Removing %d of %d rows (%.1f%%) changes the sample: every statistic, split "
            "and model fitted afterwards describes the deduplicated data, not the "
            "partition you registered."
            % (removed, rows_before, 100.0 * removed / max(rows_before, 1))
        )
    if target_delta and abs(target_delta.get("delta") or 0.0) >= 0.005:
        if target_before.get("kind") == "categorical":
            warnings.append(
                "The target balance shifts: class %r goes from %.1f%% to %.1f%% (%+.1f pp), "
                "so accuracy and threshold-based metrics will move with it."
                % (
                    target_delta.get("value"),
                    100.0 * (target_delta.get("share_before") or 0.0),
                    100.0 * (target_delta.get("share_after") or 0.0),
                    100.0 * (target_delta.get("delta") or 0.0),
                )
            )
        else:
            warnings.append(
                "The target mean shifts from %.6g to %.6g (%+.2f%%), which changes the "
                "baseline and the residual scale."
                % (
                    target_delta.get("share_before") or 0.0,
                    target_delta.get("share_after") or 0.0,
                    100.0 * ((target_delta.get("delta") or 0.0) / abs(target_delta.get("share_before") or 1.0)),
                )
            )
    significant_feature = [
        (column, stats)
        for column, stats in numeric_shift.items()
        if stats["mean_delta_pct"] is not None and abs(stats["mean_delta_pct"]) >= 0.05
    ]
    if significant_feature:
        column, stats = max(significant_feature, key=lambda item: abs(item[1]["mean_delta_pct"]))
        warnings.append(
            "Feature %r mean moves by %+.1f%% (%.6g → %.6g); the marginal distribution "
            "of the deduplicated data is not the registered one."
            % (column, 100.0 * stats["mean_delta_pct"], stats["before"]["mean"], stats["after"]["mean"])
        )
    if conflicting_groups:
        warnings.append(
            "%d duplicate feature group(s), covering %d rows, carry differing target values. "
            "These are contradictory labels or cross-split leakage: random cross-validation "
            "on the original data is optimistically biased, and deduplication keeps one "
            "arbitrary label per group."
            % (conflicting_groups, conflicting_rows)
        )
    if duplicate_rows:
        warnings.append(
            "Duplicates are a split hazard: if identical feature rows land in both training "
            "and validation folds, the validation score is not an out-of-sample estimate. "
            "Deduplicate before splitting or use a grouped splitter."
        )
    if sampled:
        warnings.append(
            "The duplicate profile was computed on a random sample of %d rows; counts are "
            "scaled to the sample, not the full partition." % rows_before
        )

    return {
        "features": features,
        "keep": keep,
        "target": target,
        "sampled": sampled,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": removed,
        "rows_removed_pct": float(removed / max(rows_before, 1)),
        "duplicate_rows": duplicate_rows,
        "duplicate_groups": duplicate_groups,
        "removable_rows": removable,
        "full_duplicate_rows": full_duplicate_rows,
        "conflicting_label_groups": conflicting_groups,
        "conflicting_label_rows": conflicting_rows,
        "target_before": target_before,
        "target_after": target_after,
        "target_shift": target_delta,
        "numeric_shift": numeric_shift,
        "warnings": warnings,
    }


def drop_duplicate_rows(frame, *, target, ignored, columns=None, keep="first", max_rows=20000):
    """Drop exact feature-duplicate rows and return ``(frame, report)``."""
    report = duplicate_analysis(
        frame,
        target=target,
        ignored=ignored,
        columns=columns,
        keep=keep,
        max_rows=max_rows,
    )
    if report["sampled"]:
        raise ClipError(
            "Refusing to remove duplicates from a sampled profile; raise max_rows or "
            "pass explicit feature columns"
        )
    result = frame.drop_duplicates(subset=report["features"], keep=keep).copy()
    report["rows_after"] = int(len(result))
    report["rows_removed"] = int(len(frame) - len(result))
    report["rows_removed_pct"] = float(report["rows_removed"] / max(len(frame), 1))
    return result, report


def encoding_preview(frame, *, target, ignored, sample_rows=1000, max_outputs=40, random_state=0):
    """Show how the modeling pipeline encodes the active feature columns.

    A ``FoldLocalDataPreprocessor`` is fitted on a bounded sample, so the reported
    feature names and the previewed matrix are exactly what a model sees — with
    the important caveat that in cross-validation this vocabulary is re-learned
    inside every training fold.
    """
    import numpy as np

    from SKSurrogate.DataProcess import FoldLocalDataPreprocessor, _is_declared_text_dtype

    ignored = set(ignored or [])
    features = [column for column in frame.columns if column != target and column not in ignored]
    if not features:
        raise ImputationError("No active feature columns to encode")
    sample = frame[features]
    sampled = False
    if sample_rows and len(sample) > sample_rows > 0:
        sample = sample.sample(sample_rows, random_state=random_state)
        sampled = True
    try:
        preprocessor = FoldLocalDataPreprocessor()
        matrix = preprocessor.fit_transform(sample)
        names = list(preprocessor.get_feature_names_out())
        groups = list(preprocessor.transformer_.transformers_)
    except Exception as exc:  # pragma: no cover - surfaced verbatim to the UI
        raise ImputationError("Could not fit the encoding preview: %s" % exc)

    column_names = list(sample.columns)
    per_column_outputs = {column: [] for column in features}
    for _name, transformer, columns in groups:
        positions = [columns] if isinstance(columns, (int, np.integer)) else list(columns)
        labels = [column_names[int(pos)] if isinstance(pos, (int, np.integer)) else str(pos) for pos in positions]
        try:
            subnames = [str(entry) for entry in transformer.get_feature_names_out()]
        except Exception:
            subnames = []
        if len(labels) == 1:
            per_column_outputs[labels[0]].extend(subnames)
        elif len(subnames) == len(labels):
            for label, subname in zip(labels, subnames):
                per_column_outputs[label].append(subname)
        else:
            for label in labels:
                token = label + "_"
                per_column_outputs[label].extend(
                    subname for subname in subnames if subname.split("__")[-1].startswith(token)
                )

    per_column = []
    for column in features:
        series = frame[column]
        dtype = series.dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            kind, encoding = "datetime", "Epoch seconds + median imputation"
        elif _is_declared_text_dtype(dtype):
            kind, encoding = "text", "Per-column TF-IDF features"
        elif pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
            kind, encoding = "categorical", "One-hot (missing → “__MISSING__”, unseen → all zeros)"
        else:
            kind, encoding = "numeric", "Passthrough + median imputation"
        outputs = list(per_column_outputs.get(column, []))
        per_column.append(
            {
                "column": column,
                "dtype": str(dtype),
                "kind": kind,
                "encoding": encoding,
                "cardinality": int(series.nunique(dropna=True)),
                "missing": int(series.isna().sum()),
                "output_feature_count": len(outputs),
                "output_columns": outputs[:20],
                "output_columns_truncated": max(len(outputs) - 20, 0),
                "high_cardinality": bool(
                    kind in {"categorical", "text"} and series.nunique(dropna=True) > 50
                ),
            }
        )

    dense = matrix[:8]
    if hasattr(dense, "toarray"):
        dense = dense.toarray()
    dense = np.asarray(dense)
    shown = names[:max_outputs]
    preview = [
        {name: _finite(row[index]) if _is_numeric(pd.Series(row)) else str(row[index])
         for index, name in enumerate(shown)}
        for row in dense
    ]
    return {
        "input_features": features,
        "input_feature_count": len(features),
        "output_features": names,
        "output_feature_count": len(names),
        "shown_output_features": shown,
        "preview": preview,
        "per_column": per_column,
        "sampled": sampled,
        "sample_rows": int(len(sample)),
        "notes": [
            "Encoding is learned per training fold in cross-validation, not once on the "
            "whole partition; categorical vocabularies and text vocabularies can differ "
            "between folds, and unseen categories at prediction time are ignored.",
            "Missing numeric values are median-imputed, missing categories become an "
            "explicit “__MISSING__” level, and datetimes become epoch seconds.",
            "Declared text columns (pandas StringDtype) receive TF-IDF features; plain "
            "object columns are treated as categorical.",
        ],
    }


def target_balance(frame, *, target, distribution=None, max_classes=50):
    """Exact target composition for the Data-page target card.

    Classification targets report **exact** class counts over the full partition
    (no sampling), the minority share and the largest fold count that keeps every
    class represented at least twice. Regression targets reuse the bounded
    ``dataset_profile`` summary (passed as ``distribution``) and are labelled as
    shape diagnostics rather than a distributional test.

    ``distribution`` is the target ``distribution`` block returned by
    ``SKSurrogate.dataset_profile``; when omitted the numeric branch reports only
    counts.
    """
    if target not in frame.columns:
        raise ValueError("target column %r is not in frame" % target)
    series = frame[target]
    rows = int(len(series))
    missing = int(series.isna().sum())
    if _is_numeric(series):
        distinct_values = int(series.nunique(dropna=True))
        return {
            "kind": "numeric",
            "rows": rows,
            "missing": missing,
            "distinct_values": distinct_values,
            # A 0/1 float column is numeric by dtype but classification-shaped in
            # practice; flag it so the card can say so instead of implying regression.
            "binary_like": distinct_values == 2,
            "distribution": distribution,
            "scope": "exact counts; shape statistics from the bounded profile sample",
            "interpretation": (
                "Skewness, kurtosis and IQR outlier counts describe shape; they are not "
                "normality tests and they do not justify a transformation on their own."
            ),
        }
    counts = series.dropna().astype(str).value_counts()
    class_count = int(counts.size)
    total = int(counts.sum())
    if not class_count:
        return {
            "kind": "categorical",
            "rows": rows,
            "missing": missing,
            "distinct_classes": 0,
            "classes": [],
            "classes_truncated": 0,
            "scope": "full partition rows",
            "interpretation": "The target has no observed values.",
        }
    minority = int(counts.min())
    majority = int(counts.max())
    ordered = counts.head(max_classes)
    return {
        "kind": "categorical",
        "rows": rows,
        "missing": missing,
        "distinct_classes": class_count,
        "classes": [
            {
                "label": label,
                "count": int(count),
                "share": float(count / total) if total else 0.0,
            }
            for label, count in ordered.items()
        ],
        "classes_truncated": int(max(class_count - max_classes, 0)),
        "min_class_count": minority,
        "max_class_count": majority,
        "minority_share": float(minority / total) if total else 0.0,
        "imbalance_ratio": float(majority / minority) if minority else None,
        "recommended_folds": int(min(5, minority)) if minority >= 2 else None,
        "stratification_feasible": bool(minority >= 2),
        "scope": "full partition rows; class list capped",
        "interpretation": (
            "Class counts are exact. Stratified k-fold requires every class to appear "
            "in every training fold, so the fold count is capped by the smallest class."
        ),
    }


__all__ = [
    "CLIP_METHODS",
    "ClipError",
    "ImputationError",
    "METHODS",
    "clip_partition",
    "duplicate_analysis",
    "drop_duplicate_rows",
    "encoding_preview",
    "impute_partition",
    "missing_overview",
    "target_balance",
]
