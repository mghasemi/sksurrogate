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


__all__ = [
    "ImputationError",
    "METHODS",
    "impute_partition",
    "missing_overview",
]
