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

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


class NumericFeatureError(ValueError):
    """Raised when a partition cannot yield a numeric feature matrix."""


@dataclass
class NumericTrainData:
    """A numeric ``(X, y)`` view of one partition plus what was excluded.

    ``source`` says where the columns came from: ``"partition_columns"`` for a
    tabular task, ``"temporal_design"`` when the features were generated from a
    declared :class:`~SKSurrogate.TemporalSpec` (see
    :func:`temporal_design_xy`). ``feature_specs`` carries the provenance of each
    generated column in the latter case, so a caller can state that e.g.
    ``sales__lag_4`` is a lag rather than a measured input.
    """

    columns: list[str]
    X: np.ndarray
    y: np.ndarray
    excluded: list[dict] = field(default_factory=list)
    dropped_rows: int = 0
    source: str = "partition_columns"
    feature_specs: list[dict] = field(default_factory=list)
    #: One label per row of ``X`` when the rows have a natural identity (the
    #: origin timestamp of a forecast design). ``label_name`` names that axis.
    row_labels: list[str] = field(default_factory=list)
    label_name: str | None = None

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
        if not feature_columns:
            # Not the same failure as "the columns are strings": after the ignore
            # list removed the temporal/target/ignored columns there is nothing
            # left to analyse. Saying "encode or drop those columns" here would
            # send the caller looking for columns that do not exist.
            raise NumericFeatureError(
                "This partition has no feature columns left: every non-target column "
                "is ignored (or is the temporal time/series column). Register a "
                "partition with feature columns, or, for a time-series task, run the "
                "analysis over its generated forecast design (lag/rolling/calendar "
                "features)."
            )
        raise NumericFeatureError(
            "Sensitivity and correlation analysis need numeric feature columns, but "
            "this partition has none (non-numeric: %s). Encode or drop those columns, "
            "then re-register the partition."
            % ", ".join(excluded_names(excluded))
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


def features_from_json(raw: Any) -> dict | None:
    """Parse a ``features`` JSON request field/query parameter.

    Endpoints that accept ``make_supervised`` builder kwargs take them as JSON.
    ``None`` or a blank string means "use the default design". Anything else that
    is not a JSON object is refused as a :class:`NumericFeatureError`, so callers
    answer 400 with the parse problem instead of silently ignoring the setting.
    """
    if raw is None or not str(raw).strip():
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NumericFeatureError("features must be a JSON object: %s" % exc)
    if not isinstance(parsed, dict):
        raise NumericFeatureError("features must be a JSON object")
    return parsed


def temporal_design_xy(
    frame: pd.DataFrame,
    spec: Any,
    target: str,
    features: Mapping[str, Any] | None = None,
) -> NumericTrainData:
    """Project a temporal partition onto the *generated* forecast design.

    A time-series partition usually has no raw feature columns at all: the time
    (and panel id) columns are excluded by the temporal contract and the target is
    the label, so only generated lags/rollings/calendar fields carry information.
    Sensitivity and correlation analyses therefore run over
    :func:`SKSurrogate.make_supervised`, exactly as the rolling-origin backtest
    does — the same builder, the same shift rule, the same dropped rows.

    :param frame: the registered partition, including its time/series columns.
    :param spec: the task's :class:`~SKSurrogate.TemporalSpec` (mapping or object).
    :param target: label column.
    :param features: ``make_supervised`` builder kwargs (``lags``, ``rolling``,
        ``diffs``, ``pct_changes``, ``calendar``, ``fourier``, ``exogenous``).
        Defaults to one lag at the declared horizon.
    :raises NumericFeatureError: when the design has no usable row.
    """
    from SKSurrogate import TemporalSpec, make_supervised

    resolved = TemporalSpec.from_dict(spec)
    builder_kwargs = dict(features or {})
    builder_kwargs.setdefault("lags", [resolved.horizon])
    design = make_supervised(frame, resolved, target, strategy="direct", **builder_kwargs)
    if not len(design.X):
        raise NumericFeatureError(
            "the forecast design has no usable row: the requested lags/windows are "
            "longer than the available history. Shorten them, add history, or drop "
            "features."
        )
    data = design.X.to_numpy(dtype=float, copy=True)
    values = design.y.to_numpy(dtype=float, copy=True)
    finite = np.isfinite(data).all(axis=1) & np.isfinite(values)
    return NumericTrainData(
        columns=list(design.X.columns),
        X=data[finite],
        y=values[finite],
        excluded=[],
        dropped_rows=int(design.dropped_rows + int((~finite).sum())),
        source="temporal_design",
        feature_specs=[entry.to_dict() for entry in design.feature_specs],
        # The origin timestamp is not a feature — it is the row's identity, and
        # showing it is how a reader sees that the time axis was consumed rather
        # than silently dropped.
        row_labels=[str(value) for value in design.meta["origin_time"].to_numpy()[finite]],
        label_name="origin_time",
    )


__all__ = [
    "NumericFeatureError",
    "NumericTrainData",
    "excluded_names",
    "numeric_train_xy",
    "temporal_design_xy",
]
