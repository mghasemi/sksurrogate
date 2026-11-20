"""Temporal (time-series) data contract for SKSurrogate.

This module is **opt-in**: nothing here runs unless a task explicitly declares a
:class:`TemporalSpec` (via ``mltrack.RegisterData(..., temporal=...)`` or
``mltrack.SetTemporalSpec``). Tabular tasks keep their existing behaviour.

A declared spec answers three questions the rest of the library must not guess:

* **Which column is time** (``time_column``) and, for panel data, **which column
  identifies the series** (``series_id``).
* **What the forecast horizon is** (``horizon``), so feature builders and
  backtesters can reject features that reach into the future.
* **What the expected sampling frequency is** (``freq``), so registration can
  refuse a frame with unordered rows, duplicate timestamps or gaps instead of
  silently feeding a splitter garbage.

The validation report returned by :func:`validate_temporal_frame` is stored in
the task metadata alongside the spec; the control plane returns it from
``GET /api/datasets/{task}/temporal`` so a user can see exactly why an upload was
accepted or rejected.

Only ``pandas`` and ``numpy`` are used here — no new hard dependency.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

#: Task-metadata key under which a task's temporal spec is stored.
TEMPORAL_METADATA_KEY = "temporal_spec"

#: Task-metadata key under which the per-partition validation reports are stored.
TEMPORAL_REPORT_METADATA_KEY = "temporal_validation"

#: Row-order conventions accepted for the time column.
TEMPORAL_ORDERS = ("increasing", "decreasing")

#: How many series are described individually in a validation report.
MAX_SERIES_DETAIL = 50

#: How many offending row labels an error message may list.
MAX_REPORTED_ROWS = 5


class TemporalValidationError(ValueError):
    """Raised when a frame does not satisfy its declared :class:`TemporalSpec`.

    ``report`` carries the partial validation report so callers (and the control
    plane) can show structured diagnostics rather than a single message.
    """

    def __init__(self, message: str, report: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.report = dict(report or {})


@dataclass(frozen=True)
class TemporalSpec:
    """Declarative description of a time-ordered dataset.

    :param time_column: column holding the observation time. Accepts
        datetime-like values (parsed with ``pandas.to_datetime``) or a numeric
        time axis such as an integer step index.
    :param series_id: optional column identifying one of several series in a
        panel. ``None`` means the frame is a single series.
    :param freq: optional expected sampling frequency, as a pandas offset alias
        (``"D"``, ``"h"``, ``"MS"``, ``"15min"``, ...). When given, gaps larger
        than ``gap_tolerance`` periods and off-grid timestamps are rejected.
    :param horizon: forecast horizon in steps; must be >= 1.
    :param order: expected row order — ``"increasing"`` or ``"decreasing"``.
    :param gap_tolerance: how many missing periods per series are tolerated
        before validation fails (only meaningful when ``freq`` is declared).
    """

    time_column: str
    series_id: str | None = None
    freq: str | None = None
    horizon: int = 1
    order: str = "increasing"
    gap_tolerance: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.time_column, str) or not self.time_column.strip():
            raise ValueError("time_column must be a non-empty column name")
        if self.series_id is not None:
            if not isinstance(self.series_id, str) or not self.series_id.strip():
                raise ValueError("series_id must be a non-empty column name or None")
            if self.series_id == self.time_column:
                raise ValueError("series_id and time_column must be different columns")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, (int, np.integer)):
            raise ValueError("horizon must be a positive integer")
        if int(self.horizon) < 1:
            raise ValueError("horizon must be >= 1")
        if self.order not in TEMPORAL_ORDERS:
            raise ValueError(
                "order must be one of %s" % ", ".join(repr(item) for item in TEMPORAL_ORDERS)
            )
        if isinstance(self.gap_tolerance, bool) or not isinstance(
            self.gap_tolerance, (int, np.integer)
        ):
            raise ValueError("gap_tolerance must be a non-negative integer")
        if int(self.gap_tolerance) < 0:
            raise ValueError("gap_tolerance must be >= 0")
        if self.freq is not None:
            if not isinstance(self.freq, str) or not self.freq.strip():
                raise ValueError("freq must be a pandas offset alias such as 'D' or None")
            try:
                to_offset(self.freq)
            except (ValueError, TypeError) as exc:
                raise ValueError("freq %r is not a pandas offset alias: %s" % (self.freq, exc))

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the spec."""
        return {
            "time_column": self.time_column,
            "series_id": self.series_id,
            "freq": self.freq,
            "horizon": int(self.horizon),
            "order": self.order,
            "gap_tolerance": int(self.gap_tolerance),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "TemporalSpec":
        """Build a spec from a dataclass instance, a mapping or ``None``.

        ``None`` is rejected: callers that mean "no temporal role" should not
        call this at all. Unknown keys are refused so a typo in a JSON payload
        fails loudly instead of being ignored.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("temporal spec must be a TemporalSpec or a mapping")
        allowed = {
            "time_column",
            "series_id",
            "freq",
            "horizon",
            "order",
            "gap_tolerance",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError("unknown temporal spec field(s): %s" % ", ".join(unknown))
        if "time_column" not in value:
            raise ValueError("temporal spec requires a time_column")
        kwargs = dict(value)
        if kwargs.get("series_id") == "":
            kwargs["series_id"] = None
        if kwargs.get("freq") == "":
            kwargs["freq"] = None
        return cls(**kwargs)

    @property
    def temporal_columns(self) -> list[str]:
        """Columns that describe *when* / *which series* rather than *what*.

        These are excluded from the feature matrix by default: the time
        information reaches a model through generated features (lags, calendar
        terms), never as a raw scaled epoch number.
        """
        columns = [self.time_column]
        if self.series_id is not None:
            columns.append(self.series_id)
        return columns


def temporal_columns(spec: Any) -> list[str]:
    """Columns a spec marks as temporal metadata (``[]`` when ``spec`` is ``None``)."""
    if spec is None:
        return []
    return TemporalSpec.from_dict(spec).temporal_columns


def infer_freq(values: Iterable[Any]) -> str | None:
    """Infer a pandas frequency alias from a sequence of datetime-like values.

    Returns ``None`` when there are fewer than three distinct values, when the
    values cannot be parsed, or when pandas cannot identify a regular frequency.
    Duplicates are collapsed first, so registering a frame twice does not change
    the answer.
    """
    parsed_series = pd.Series(list(values))
    with warnings.catch_warnings():
        # Mixed/unparseable input is an expected "cannot infer" answer here, not
        # something the caller needs a pandas UserWarning about.
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(parsed_series, errors="coerce").dropna()
    if len(parsed) < 3:
        return None
    unique = pd.DatetimeIndex(parsed.unique()).sort_values()
    if len(unique) < 3:
        return None
    try:
        return pd.infer_freq(unique)
    except (ValueError, TypeError):
        return None


def _time_axis(frame: pd.DataFrame, spec: TemporalSpec) -> dict[str, Any]:
    """Parse the time column into a sortable integer/float axis.

    Two kinds are supported:

    * ``datetime`` — anything ``pandas.to_datetime`` parses (already-datetime
      columns included); the axis is **int64** nanoseconds since the epoch.
      Nanoseconds must not be stored as floats: 2026-era timestamps exceed
      2**53 ns, so a ``float64`` axis silently loses sub-microsecond precision.
    * ``numeric`` — an integer/float step index, used as-is.

    A numeric column is *not* coerced through ``to_datetime``: that would read
    ``0, 1, 2`` as 1970 timestamps and then report a 1-second frequency.
    """
    raw = frame[spec.time_column]
    if pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_bool_dtype(raw):
        values = raw.to_numpy(dtype="float64")
        finite = np.isfinite(values)
        if not finite.all():
            bad = list(frame.index[~finite][:MAX_REPORTED_ROWS])
            raise TemporalValidationError(
                "time column %r has %d non-finite value(s) (first row labels: %s)"
                % (spec.time_column, int((~finite).sum()), bad)
            )
        return {"kind": "numeric", "values": values}

    parsed = pd.to_datetime(raw, errors="coerce")
    missing = parsed.isna()
    if missing.any():
        bad = list(frame.index[missing][:MAX_REPORTED_ROWS])
        raise TemporalValidationError(
            "time column %r has %d missing or unparseable timestamp(s) "
            "(first row labels: %s)" % (spec.time_column, int(missing.sum()), bad)
        )
    values = parsed.to_numpy(dtype="datetime64[ns]").astype("int64")
    return {"kind": "datetime", "values": values}


def _numeric_step(values: np.ndarray) -> float | None:
    """Median positive step of a numeric time axis, or ``None`` when undefined."""
    unique = np.unique(values)
    if len(unique) < 2:
        return None
    deltas = np.diff(unique)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return None
    return float(np.median(deltas))


def _series_report(
    name: Any,
    positions: np.ndarray,
    axis_values: np.ndarray,
    spec: TemporalSpec,
    axis_kind: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Validate one series; return (report, errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    label = "series %r" % (name,) if spec.series_id is not None else "the dataset"

    values = axis_values[positions]
    deltas = np.diff(values)

    def _label(raw_value: Any) -> str:
        if axis_kind == "datetime":
            return str(pd.Timestamp(int(raw_value)))
        return repr(float(raw_value))

    if spec.order == "increasing":
        inversions = np.flatnonzero(deltas < 0)
    else:
        inversions = np.flatnonzero(deltas > 0)
    if inversions.size:
        first = int(inversions[0])
        errors.append(
            "%s: time column is not %s at row %d (previous %s -> next %s); "
            "sort the partition before registering it, or set order='%s'"
            % (
                label,
                spec.order,
                int(positions[first + 1]),
                _label(values[first]),
                _label(values[first + 1]),
                "decreasing" if spec.order == "increasing" else "increasing",
            )
        )

    duplicates = np.flatnonzero(pd.Series(values).duplicated().to_numpy())
    if duplicates.size:
        rows = [int(positions[index]) for index in duplicates[: MAX_REPORTED_ROWS - 1]]
        errors.append(
            "%s: %d duplicate timestamp(s) in %r (first duplicate rows: %s)"
            % (label, duplicates.size, spec.time_column, rows)
        )

    unique_values = np.unique(values)
    regular: bool | None = None
    inferred: str | None = None
    inferred_step: float | None = None
    missing = 0

    if axis_kind == "datetime":
        unique_index = pd.DatetimeIndex(unique_values.astype("datetime64[ns]"))
        if len(unique_index) >= 3:
            try:
                inferred = pd.infer_freq(unique_index)
            except (ValueError, TypeError):
                inferred = None
        if len(deltas) and deltas.size:
            increments = np.unique(deltas[deltas > 0])
            regular = bool(increments.size == 1)
        elif len(unique_values) == 1:
            regular = True
        else:
            regular = None

        offset_alias = spec.freq or inferred
        if offset_alias is not None and len(unique_index) >= 1:
            try:
                grid = pd.date_range(
                    unique_index.min(), unique_index.max(), freq=to_offset(offset_alias)
                )
            except (ValueError, TypeError):
                grid = None
            if grid is not None:
                missing = max(int(len(grid) - len(unique_index)), 0)
                grid_values = set(grid.values)
                off_grid = int(
                    sum(1 for value in unique_index.values if value not in grid_values)
                )
                if off_grid:
                    message = (
                        "%s: %d timestamp(s) are not aligned to the declared frequency %r"
                        % (label, off_grid, offset_alias)
                    )
                    if spec.freq is not None:
                        errors.append(message)
                    else:
                        warnings.append(message)
    else:
        inferred_step = _numeric_step(unique_values)
        if len(deltas):
            increments = np.unique(deltas[deltas > 0])
            regular = bool(increments.size == 1)
        else:
            regular = None
        step = None
        if spec.freq is not None:
            # A numeric axis cannot express a calendar frequency; the declared
            # alias only pins the expected step count between observations.
            step = inferred_step
        elif inferred_step:
            step = inferred_step
        if step and len(unique_values) >= 1:
            span = float(unique_values.max() - unique_values.min())
            expected = int(round(span / step)) + 1
            missing = max(expected - len(unique_values), 0)

    if spec.freq is not None and missing > spec.gap_tolerance:
        errors.append(
            "%s: %d missing period(s) for the declared frequency %r "
            "(gap_tolerance=%d)"
            % (label, missing, spec.freq, spec.gap_tolerance)
        )
    elif spec.freq is None and missing:
        warnings.append(
            "%s: %d missing period(s) relative to the inferred frequency %r"
            % (label, missing, inferred or (inferred_step if inferred_step else "step"))
        )

    length = int(len(positions))
    if length <= spec.horizon:
        warnings.append(
            "%s: only %d observation(s) for horizon=%d; a forecast cannot be "
            "backtested without more history" % (label, length, spec.horizon)
        )
    if regular is False and spec.freq is None:
        warnings.append("%s: the time steps are irregular" % label)

    report = {
        "series": None if name is None else _json_scalar(name),
        "rows": length,
        "inferred_freq": inferred,
        "inferred_step": inferred_step,
        "regular": regular,
        "missing_periods": missing,
    }
    if axis_kind == "datetime" and len(unique_values):
        report["start"] = str(pd.Timestamp(int(unique_values.min())))
        report["end"] = str(pd.Timestamp(int(unique_values.max())))
    elif len(unique_values):
        report["start"] = float(unique_values.min())
        report["end"] = float(unique_values.max())
    return report, errors, warnings


def _json_scalar(value: Any) -> Any:
    """Coerce numpy scalars to plain JSON-serializable Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summarize_series(series_reports, spec, axis_kind):
    """Aggregate per-series reports into the fields every report carries."""
    inferred_counts: dict[str, int] = {}
    for report in series_reports:
        alias = report.get("inferred_freq")
        if alias:
            inferred_counts[alias] = inferred_counts.get(alias, 0) + 1
    inferred_freq = (
        max(inferred_counts, key=lambda alias: inferred_counts[alias])
        if inferred_counts
        else None
    )
    if not inferred_freq and spec.freq is None:
        steps = [report.get("inferred_step") for report in series_reports]
        steps = [step for step in steps if step]
        inferred_freq = None if not steps else "step=%g" % max(steps)

    summary = {
        "series_count": len(series_reports),
        "series": series_reports[:MAX_SERIES_DETAIL],
        "inferred_freq": inferred_freq,
        "missing_periods": int(
            sum(item.get("missing_periods") or 0 for item in series_reports)
        ),
        "series_lengths": {
            str(item["series"]): item["rows"] for item in series_reports[:MAX_SERIES_DETAIL]
        },
    }
    if len(series_reports) > MAX_SERIES_DETAIL:
        summary["series_truncated"] = len(series_reports) - MAX_SERIES_DETAIL
    if axis_kind == "datetime":
        summary["time_range"] = {
            "start": min(item["start"] for item in series_reports),
            "end": max(item["end"] for item in series_reports),
        }
    else:
        summary["time_range"] = {
            "start": min(float(item["start"]) for item in series_reports),
            "end": max(float(item["end"]) for item in series_reports),
        }
    return summary


def validate_temporal_frame(frame: pd.DataFrame, spec: Any) -> dict[str, Any]:
    """Validate ``frame`` against ``spec`` and return a structured report.

    Structural violations raise :class:`TemporalValidationError` (a ``ValueError``
    subclass) so a caller cannot register a frame that would later produce
    meaningless time-ordered splits. Warnings are advisory and stay in the
    report: irregular sampling with no declared frequency, series shorter than
    the horizon, and gaps that were merely inferred rather than declared.

    The report shape is stable and returned by the control plane::

        {"ok": True, "time_kind": "datetime", "rows": 120, "series_count": 2,
         "declared_freq": "D", "inferred_freq": "D", "missing_periods": 0,
         "time_range": {"start": "...", "end": "..."},
         "series": [{"series": "A", "rows": 60, ...}, ...],
         "warnings": [...], "errors": []}
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("validate_temporal_frame expects a pandas DataFrame")
    spec = TemporalSpec.from_dict(spec)

    base: dict[str, Any] = {
        "ok": False,
        "time_column": spec.time_column,
        "series_id": spec.series_id,
        "horizon": int(spec.horizon),
        "order": spec.order,
        "declared_freq": spec.freq,
        "gap_tolerance": int(spec.gap_tolerance),
        "rows": int(len(frame)),
        "warnings": [],
        "errors": [],
    }

    if frame.empty:
        raise TemporalValidationError("cannot validate an empty frame", base)

    missing_columns = [column for column in spec.temporal_columns if column not in frame.columns]
    if missing_columns:
        base["errors"] = ["temporal column(s) not found: %s" % ", ".join(missing_columns)]
        raise TemporalValidationError(
            "temporal column(s) not found in the frame: %s" % ", ".join(missing_columns),
            base,
        )

    if spec.time_column == spec.series_id:  # pragma: no cover - defended by the spec
        raise TemporalValidationError("time_column and series_id must differ", base)

    try:
        axis = _time_axis(frame, spec)
    except TemporalValidationError as exc:
        exc.report.setdefault("errors", [str(exc)])
        raise
    axis_values = axis["values"]

    if spec.series_id is None:
        groups: list[tuple[Any, np.ndarray]] = [
            (None, np.arange(len(frame), dtype="int64"))
        ]
    else:
        series_values = frame[spec.series_id]
        if bool(series_values.isna().any()):
            bad = list(frame.index[series_values.isna()][:MAX_REPORTED_ROWS])
            base["errors"] = ["series_id column %r has missing values" % spec.series_id]
            raise TemporalValidationError(
                "series_id column %r has %d missing value(s) (first row labels: %s)"
                % (spec.series_id, int(series_values.isna().sum()), bad),
                base,
            )
        groups = []
        positions = np.arange(len(frame), dtype="int64")
        for key, sub_positions in pd.Series(positions, index=frame.index).groupby(
            series_values.to_numpy(), sort=False
        ):
            groups.append((key, sub_positions.to_numpy(dtype="int64")))

    series_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for name, positions in groups:
        report, group_errors, group_warnings = _series_report(
            name, positions, axis_values, spec, axis["kind"]
        )
        series_reports.append(report)
        errors.extend(group_errors)
        warnings.extend(group_warnings)

    if errors:
        base.update(_summarize_series(series_reports, spec, axis["kind"]))
        base["time_kind"] = axis["kind"]
        base["errors"] = errors
        base["warnings"] = warnings
        raise TemporalValidationError(errors[0], base)

    summary = _summarize_series(series_reports, spec, axis["kind"])
    report = dict(base)
    report["ok"] = True
    report["time_kind"] = axis["kind"]
    report.update(summary)
    report["warnings"] = warnings
    report["errors"] = []
    return report
