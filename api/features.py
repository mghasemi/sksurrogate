"""Numeric projection of a registered partition for numeric-only analyses.

Registered partitions keep the caller's column types — a churn CSV carries
``Surname``, ``Geography`` and ``Gender`` as object columns. Sensitivity analysis
(SALib sampling + a surrogate regressor) and Pearson-correlation pruning are
numeric procedures: they take means, variances and independent ranges over each
feature, so a string column is not a feature they can estimate. Feeding the raw
matrix raised ``ValueError: could not convert string to float: 'Hargrave'`` from
sklearn's SVR, and the same class of error from ``numpy.corrcoef`` for pruning.

One projection, in one place, that also *reports* what it excluded so the API can
state the exclusion instead of silently dropping data. Numerical detail:

* a column is usable when pandas reports a numeric dtype (booleans included —
  they are 0/1 and have a well-defined range);
* the target must be numeric too: a string-classification target cannot be
  regressed by the sensitivity surrogate, and pretending otherwise would produce
  meaningless weights;
* rows with a non-finite value (NaN or ±inf) anywhere in the numeric block or the
  target are dropped as a complete case — SALib needs finite bounds — and the
  count is reported.

Nothing here decides which *semantically* meaningful features to use: a numeric
identifier such as ``CustomerId`` still participates and will dominate a
variance-based measure. That is a data-modelling choice for the caller (prune the
column or pass a feature subset), and it is stated in the result's limitations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class NumericFeatureError(ValueError):
    """Raised when a partition cannot yield a numeric feature matrix."""


@dataclass
class NumericTrainData:
    """A numeric ``(X, y)`` view of one partition plus what was excluded."""

    columns: list[str]
    X: np.ndarray
    y: np.ndarray
    excluded: list[dict] = field(default_factory=list)
    dropped_rows: int = 0

    @property
    def excluded_names(self) -> list[str]:
        return [entry["column"] for entry in self.excluded]


def numeric_train_xy(frame: pd.DataFrame, target: str) -> NumericTrainData:
    """Project ``frame`` onto its numeric feature columns, keeping ``target``.

    Raises :class:`NumericFeatureError` with an actionable message when the
    projection would be empty (no numeric feature, or a non-numeric target), so
    an endpoint can answer 400 instead of failing inside the fit.
    """
    if target not in frame.columns:
        raise NumericFeatureError("target column %r is not present in the partition" % target)

    feature_columns = [column for column in frame.columns if column != target]
    numeric: list[str] = []
    excluded: list[dict] = []
    for column in feature_columns:
        series = frame[column]
        # ``is_numeric_dtype`` is True for int/float/bool and False for
        # object/category/string/datetime — exactly the line we want.
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series):
            numeric.append(column)
        else:
            excluded.append({"column": str(column), "dtype": str(series.dtype)})

    if not numeric:
        raise NumericFeatureError(
            "Sensitivity and correlation analysis need numeric feature columns, but "
            "this partition has none (non-numeric: %s). Encode or drop those columns, "
            "then re-register the partition."
            % (", ".join(excluded_names(excluded)) or "all columns")
        )

    target_series = frame[target]
    if not pd.api.types.is_numeric_dtype(target_series):
        raise NumericFeatureError(
            "target column %r has dtype %s; sensitivity analysis fits a numeric "
            "surrogate and needs a numeric (encoded) target."
            % (target, target_series.dtype)
        )

    data = frame[numeric].to_numpy(dtype=float, copy=True)
    values = target_series.to_numpy(dtype=float)
    finite = np.isfinite(data).all(axis=1) & np.isfinite(values)
    dropped_rows = int((~finite).sum())
    if not finite.any():
        raise NumericFeatureError(
            "every row of the numeric block has a NaN or infinite value, so there is "
            "nothing to analyse; clean the partition and re-register it."
        )
    return NumericTrainData(
        columns=numeric,
        X=data[finite],
        y=values[finite],
        excluded=excluded,
        dropped_rows=dropped_rows,
    )


def excluded_names(excluded: list[dict]) -> list[str]:
    return [entry["column"] for entry in excluded]


__all__ = ["NumericFeatureError", "NumericTrainData", "excluded_names", "numeric_train_xy"]
