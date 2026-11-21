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
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

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


# ---------------------------------------------------------------------------
# Leakage-safe feature construction (plan phase P2)
# ---------------------------------------------------------------------------

#: Rolling statistics the builder understands.
ROLL_STATS = ("mean", "std", "min", "max", "median", "sum")

#: Calendar features the builder understands (datetime axes only).
CALENDAR_FEATURES = (
    "hour",
    "minute",
    "dow",
    "dom",
    "doy",
    "month",
    "week",
    "is_weekend",
)


class LeakageError(ValueError):
    """Raised when a generated feature depends on observations after its origin."""


@dataclass
class FeatureSpec:
    """Provenance of one generated feature column.

    ``shift`` is the minimum number of steps back from the prediction origin that
    the feature reads. For every target-derived feature it is at least the
    forecast horizon, which is what makes the design leakage-safe by
    construction.
    """

    name: str
    op: str
    column: str | None = None
    shift: int = 0
    window: int | None = None
    params: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "op": self.op,
            "column": self.column,
            "shift": int(self.shift),
            "window": self.window,
            "params": dict(self.params or {}),
        }


@dataclass
class ForecastDesign:
    """A supervised design matrix for a forecasting problem.

    ``X``, ``y`` and ``meta`` share a positional index: row *i* is the origin
    described by ``meta.iloc[i]`` (series, origin time, target time) whose
    feature vector is ``X.iloc[i]`` and whose target is ``y.iloc[i]``.
    """

    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame
    feature_specs: list[FeatureSpec]
    strategy: str
    horizon: int
    dropped_rows: int
    temporal_spec: TemporalSpec
    target_step: int = 0

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    def describe(self) -> dict[str, Any]:
        """JSON-serializable summary for logging into task metadata."""
        return {
            "strategy": self.strategy,
            "horizon": int(self.horizon),
            "target_step": int(self.target_step),
            "rows": int(len(self.X)),
            "dropped_rows": int(self.dropped_rows),
            "features": [spec.to_dict() for spec in self.feature_specs],
            "temporal_spec": self.temporal_spec.to_dict(),
        }


def _normalize_rolling(rolling: Any) -> list[tuple[int, tuple[str, ...]]]:
    """Normalize the ``rolling`` argument to ``[(window, (stats...)), ...]``.

    Accepts ``[{"window": 14, "stats": ["mean", "std"]}, (7, "mean"), 30]``.
    """
    if rolling is None:
        return []
    if isinstance(rolling, (int, np.integer)):
        rolling = [int(rolling)]
    normalized = []
    for entry in rolling:
        if isinstance(entry, (int, np.integer)):
            window, stats = int(entry), ("mean",)
        elif isinstance(entry, Mapping):
            unknown = sorted(set(entry) - {"window", "stats", "min_periods"})
            if unknown:
                raise ValueError(
                    "unknown rolling field(s): %s" % ", ".join(unknown)
                )
            if "window" not in entry:
                raise ValueError("rolling entries require a 'window'")
            window = int(entry["window"])
            stats = entry.get("stats", ("mean",))
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            window, stats = int(entry[0]), entry[1]
        else:
            raise ValueError(
                "rolling entries must be an int, a (window, stats) pair, or a "
                "mapping with 'window'/'stats'"
            )
        if isinstance(stats, str):
            stats = (stats,)
        stats = tuple(stats)
        if window < 1:
            raise ValueError("rolling windows must be >= 1")
        bad = [stat for stat in stats if stat not in ROLL_STATS]
        if bad:
            raise ValueError(
                "unknown rolling statistic(s) %s; expected one of %s"
                % (", ".join(bad), ", ".join(ROLL_STATS))
            )
        normalized.append((window, stats))
    return normalized


class ForecastFeatureBuilder(BaseEstimator, TransformerMixin):
    """Build leakage-safe lag / rolling / calendar features for a temporal dataset.

    Every target-derived feature is shifted by at least the declared forecast
    horizon *within each series*, so a row at origin ``t`` never reads an
    observation after ``t - horizon``. ``center=True`` does not exist in this API.

    The frame passed to :meth:`fit`/:meth:`transform` must still contain the
    target column (lag features are built from it). The control plane's feature
    matrix drops the target, so materialize the design matrix *before*
    registration, or keep the target in the frame you hand to the builder.

    :param spec: :class:`TemporalSpec` or mapping describing time/series/horizon.
    :param target: column to predict; source of every lag/rolling/diff feature.
    :param lags: steps back to read, each must be ``>= horizon``.
    :param rolling: ``[{"window": 14, "stats": ["mean", "std"]}, 7, ...]``;
        windows always end at ``t - horizon``.
    :param diffs: step sizes for ``x[t-h] - x[t-h-k]`` features.
    :param pct_changes: step sizes for relative differences; zero denominators
        produce ``NaN``, never ``inf``.
    :param calendar: calendar fields (datetime axes only).
    :param fourier: ``{"periods": [7, 365.25], "terms": 2}`` — periods are counted
        in time *steps*, and the phase comes from the timestamp itself, so it is
        known at forecast time.
    :param exogenous: extra column names passed through unchanged; they must be
        known at the forecast origin.
    :param dropna: drop rows with an incomplete feature vector or missing target.
    :param min_periods: minimum observations per rolling window (default: the
        window length, so no partial windows leak in early rows).
    """

    def __init__(
        self,
        spec: Any,
        target: str,
        lags: Any = (1,),
        seasonal_lags: Any = (),
        rolling: Any = (),
        diffs: Any = (),
        pct_changes: Any = (),
        calendar: Any = (),
        fourier: Mapping[str, Any] | None = None,
        exogenous: Any = (),
        dropna: bool = True,
        min_periods: int | None = None,
    ):
        self.spec = spec
        self.target = target
        self.lags = lags
        self.seasonal_lags = seasonal_lags
        self.rolling = rolling
        self.diffs = diffs
        self.pct_changes = pct_changes
        self.calendar = calendar
        self.fourier = fourier
        self.exogenous = exogenous
        self.dropna = dropna
        self.min_periods = min_periods

    # -- validation -------------------------------------------------------
    def _resolved(self):
        """Normalize constructor arguments into a validated options mapping.

        A mapping rather than a tuple: every operation added here would otherwise
        shift the unpacking in ``fit`` and ``transform`` at the same time.
        """
        spec = TemporalSpec.from_dict(self.spec)
        lags = tuple(sorted({int(lag) for lag in (self.lags or ())}))
        if any(lag < 1 for lag in lags):
            raise ValueError("lags must be >= 1")
        seasonal_lags = tuple(
            sorted({int(period) for period in (self.seasonal_lags or ())})
        )
        if any(period < 1 for period in seasonal_lags):
            raise ValueError("seasonal_lags periods must be >= 1")
        # A seasonal lag is just a lag with a season-length name; the same
        # horizon rule applies, so it is validated once, here.
        too_short = [lag for lag in lags + seasonal_lags if lag < spec.horizon]
        if too_short:
            raise ValueError(
                "lag(s) %s are shorter than the forecast horizon %d: a lag below "
                "the horizon reads an observation the forecast is supposed to "
                "predict" % (", ".join(str(lag) for lag in too_short), spec.horizon)
            )
        rolling = _normalize_rolling(self.rolling)
        diffs = tuple(sorted({int(step) for step in (self.diffs or ())}))
        if any(step < 1 for step in diffs):
            raise ValueError("diff steps must be >= 1")
        pct_changes = tuple(sorted({int(step) for step in (self.pct_changes or ())}))
        if any(step < 1 for step in pct_changes):
            raise ValueError("pct_change steps must be >= 1")
        calendar = tuple(self.calendar or ())
        bad_calendar = [name for name in calendar if name not in CALENDAR_FEATURES]
        if bad_calendar:
            raise ValueError(
                "unknown calendar feature(s) %s; expected one of %s"
                % (", ".join(bad_calendar), ", ".join(CALENDAR_FEATURES))
            )
        exogenous = tuple(self.exogenous or ())
        fourier = None
        if self.fourier:
            unknown = sorted(set(self.fourier) - {"periods", "terms"})
            if unknown:
                raise ValueError("unknown fourier field(s): %s" % ", ".join(unknown))
            periods = tuple(float(period) for period in self.fourier.get("periods", ()))
            if not periods:
                raise ValueError("fourier requires at least one period")
            if any(period <= 0 for period in periods):
                raise ValueError("fourier periods must be positive")
            terms = int(self.fourier.get("terms", 1))
            if terms < 1:
                raise ValueError("fourier terms must be >= 1")
            fourier = (periods, terms)
        min_periods = self.min_periods
        if min_periods is not None:
            min_periods = int(min_periods)
            if min_periods < 1:
                raise ValueError("min_periods must be >= 1")
        return {
            "spec": spec,
            "lags": lags,
            "seasonal_lags": seasonal_lags,
            "rolling": rolling,
            "diffs": diffs,
            "pct_changes": pct_changes,
            "calendar": calendar,
            "fourier": fourier,
            "exogenous": exogenous,
            "min_periods": min_periods,
        }

    def fit(self, X: Any, y: Any = None):
        """Validate the frame, the columns and the horizon rules; store provenance."""
        options = self._resolved()
        spec = options["spec"]
        frame = self._as_frame(X)
        required = [spec.time_column]
        if spec.series_id is not None:
            required.append(spec.series_id)
        required.append(self.target)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(
                "ForecastFeatureBuilder is missing required column(s): %s"
                % ", ".join(missing)
            )
        missing_exog = [
            column for column in options["exogenous"] if column not in frame.columns
        ]
        if missing_exog:
            raise ValueError(
                "exogenous column(s) not found: %s" % ", ".join(missing_exog)
            )
        if options["calendar"] or options["fourier"] is not None:
            # ``_time_axis`` is the contract's parser: it accepts anything
            # ``pandas.to_datetime`` understands, including the object-dtype ISO
            # strings a CSV produces. Gating on the *raw* dtype rejected exactly
            # that common case while ``transform`` happily parsed it.
            axis = _time_axis(frame, spec)
            if options["calendar"] and axis["kind"] != "datetime":
                raise ValueError(
                    "calendar features require a datetime time column; %r is a numeric "
                    "axis. Store it as a datetime, or drop the calendar fields."
                    % spec.time_column
                )
        # Ordering is the contract's job: fail here rather than build features on
        # a shuffled frame, where every lag silently reads the wrong row.
        validate_temporal_frame(frame, spec)
        self.feature_specs_ = self._feature_specs(options)
        self.n_features_in_ = len(frame.columns)
        self.temporal_spec_ = spec
        return self

    def _feature_specs(self, options):
        spec = options["spec"]
        horizon = spec.horizon
        specs = []
        for lag in options["lags"]:
            specs.append(
                FeatureSpec(
                    name="%s__lag_%d" % (self.target, lag),
                    op="lag",
                    column=self.target,
                    shift=lag,
                )
            )
        for period in options["seasonal_lags"]:
            specs.append(
                FeatureSpec(
                    name="%s__seasonal_lag_%d" % (self.target, period),
                    op="seasonal_lag",
                    column=self.target,
                    shift=period,
                    params={"period": period},
                )
            )
        for window, stats in options["rolling"]:
            for stat in stats:
                specs.append(
                    FeatureSpec(
                        name="%s__roll_%s_%d" % (self.target, stat, window),
                        op="roll_%s" % stat,
                        column=self.target,
                        shift=horizon,
                        window=window,
                    )
                )
        for step in options["diffs"]:
            specs.append(
                FeatureSpec(
                    name="%s__diff_%d" % (self.target, step),
                    op="diff",
                    column=self.target,
                    shift=horizon,
                    params={"step": step},
                )
            )
        for step in options["pct_changes"]:
            specs.append(
                FeatureSpec(
                    name="%s__pct_change_%d" % (self.target, step),
                    op="pct_change",
                    column=self.target,
                    shift=horizon,
                    params={"step": step},
                )
            )
        for name in options["calendar"]:
            specs.append(FeatureSpec(name="cal__%s" % name, op="calendar", params={"field": name}))
        if options["fourier"] is not None:
            periods, terms = options["fourier"]
            for period in periods:
                for term in range(1, terms + 1):
                    for fn in ("sin", "cos"):
                        specs.append(
                            FeatureSpec(
                                name="fourier__%s_%g_%d" % (fn, period, term),
                                op="fourier_%s" % fn,
                                params={"period": period, "term": term},
                            )
                        )
        for column in options["exogenous"]:
            specs.append(
                FeatureSpec(name=column, op="exogenous", column=column, shift=0)
            )
        return specs

    # -- transform --------------------------------------------------------
    @staticmethod
    def _as_frame(X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "ForecastFeatureBuilder works on pandas DataFrames: lag features "
                "need the named target column, which an array cannot carry"
            )
        return X

    def transform(self, X: Any) -> pd.DataFrame:
        """Return the generated feature matrix for ``X`` (target column required)."""
        check_is_fitted(self, "feature_specs_")
        frame = self._as_frame(X)
        spec = self.temporal_spec_
        options = self._resolved()
        if self.target not in frame.columns:
            raise ValueError(
                "transform requires the target column %r: lag features are built "
                "from it" % self.target
            )

        horizon = spec.horizon
        features: dict[str, pd.Series] = {}
        for lag in options["lags"]:
            features["%s__lag_%d" % (self.target, lag)] = self._apply(
                frame, spec, lambda series, lag=lag: series.shift(lag)
            )
        # Readability alias: seasonal_lag(p) is lag(p) with a season name, so it
        # shares the same shift and the same horizon validation.
        for period in options["seasonal_lags"]:
            features["%s__seasonal_lag_%d" % (self.target, period)] = self._apply(
                frame, spec, lambda series, period=period: series.shift(period)
            )
        for window, stats in options["rolling"]:
            periods = (
                options["min_periods"] if options["min_periods"] is not None else window
            )
            for stat in stats:
                features["%s__roll_%s_%d" % (self.target, stat, window)] = self._apply(
                    frame,
                    spec,
                    lambda series, window=window, stat=stat, periods=periods: (
                        series.shift(horizon)
                        .rolling(window, min_periods=periods)
                        .agg(stat)
                    ),
                )
        for step in options["diffs"]:
            features["%s__diff_%d" % (self.target, step)] = self._apply(
                frame,
                spec,
                lambda series, step=step: series.shift(horizon) - series.shift(horizon + step),
            )
        for step in options["pct_changes"]:
            features["%s__pct_change_%d" % (self.target, step)] = self._apply(
                frame, spec, lambda series, step=step: _safe_pct_change(series, horizon, step)
            )
        if options["calendar"] or options["fourier"] is not None:
            time_values = pd.to_datetime(frame[spec.time_column])
            if options["calendar"]:
                features.update(
                    _calendar_features(time_values, options["calendar"])
                )
            if options["fourier"] is not None:
                features.update(
                    _fourier_features(time_values, frame, options["fourier"])
                )
        for column in options["exogenous"]:
            features[column] = pd.to_numeric(frame[column], errors="coerce")

        result = pd.DataFrame(features, index=frame.index)
        if self.dropna:
            result = result.dropna(axis=0, how="any")
        return result

    def build(self, X: Any, target_horizon: int | None = None) -> ForecastDesign:
        """Materialize a supervised :class:`ForecastDesign` from ``frame``.

        :param target_horizon: steps ahead for the *label*, defaulting to the
            declared horizon. The recursive strategy passes ``1`` while the
            declared horizon still governs the minimum lag/feature shift.
        """
        check_is_fitted(self, "feature_specs_")
        frame = self._as_frame(X)
        spec = self.temporal_spec_
        step = int(spec.horizon if target_horizon is None else target_horizon)
        if step < 1:
            raise ValueError("target_horizon must be >= 1")
        features = self.transform(frame)
        target_values, meta = self._target_and_meta(frame, spec, step)

        if self.dropna:
            mask = features.notna().all(axis=1) & target_values.notna()
        else:
            mask = pd.Series(True, index=frame.index)
        X_out = features.loc[mask].reset_index(drop=True)
        y_out = target_values.loc[mask].reset_index(drop=True)
        meta_out = meta.loc[mask].reset_index(drop=True)
        dropped = int(len(frame) - len(X_out))
        return ForecastDesign(
            X=X_out,
            y=y_out,
            meta=meta_out,
            feature_specs=list(self.feature_specs_),
            # A ForecastFeatureBuilder always builds targets `step` steps ahead;
            # make_supervised() sets the strategy label that produced the design.
            strategy="direct",
            horizon=spec.horizon,
            target_step=step,
            dropped_rows=dropped,
            temporal_spec=spec,
        )

    # -- helpers ----------------------------------------------------------
    def _apply(self, frame: pd.DataFrame, spec: TemporalSpec, operation):
        """Apply ``operation`` to the target series, per series when a panel.

        Grouping is what keeps one store's history out of another store's lags;
        the operation itself always shifts at least ``horizon`` steps.
        """
        target = frame[self.target].astype(float)
        if spec.series_id is None:
            return operation(target)
        grouped = frame.groupby(spec.series_id, sort=False)[self.target]
        return grouped.transform(lambda series: operation(series.astype(float)))

    def _target_and_meta(self, frame: pd.DataFrame, spec: TemporalSpec, step: int | None = None):
        """Target values at ``t + step`` plus per-row origin/target times."""
        step = int(spec.horizon if step is None else step)
        time_column = spec.time_column
        if spec.series_id is None:
            target_values = frame[self.target].astype(float).shift(-step)
            target_time = frame[time_column].shift(-step)
            series = pd.Series([None] * len(frame), index=frame.index, dtype=object)
        else:
            grouped = frame.groupby(spec.series_id, sort=False)
            target_values = grouped[self.target].transform(
                lambda values: values.astype(float).shift(-step)
            )
            target_time = grouped[time_column].transform(
                lambda values: values.shift(-step)
            )
            series = frame[spec.series_id]
        meta = pd.DataFrame(
            {
                "series": series,
                "origin_time": frame[time_column],
                "target_time": target_time,
                "horizon": int(spec.horizon),
                "target_step": step,
            },
            index=frame.index,
        )
        return target_values, meta

    # -- serving (plan phase P6) ------------------------------------------

    def serving_twin(self) -> "ForecastFeatureBuilder":
        """A behaviorally identical twin that preserves row positions.

        Serving and recursive walk-forward evaluation must select features by
        position: dropping rows with an incomplete vector would silently shift
        output row *i* away from frame row *i*. The twin keeps every fitted
        attribute and turns ``dropna`` off; the original instance is untouched.
        """
        check_is_fitted(self, "feature_specs_")
        params = dict(self.get_params())
        params["dropna"] = False
        twin = type(self)(**params)
        for attr_name, attr_value in vars(self).items():
            if attr_name.endswith("_"):
                setattr(twin, attr_name, attr_value)
        return twin

    # -- serialization (plan phase P6) ------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialization of this builder's *configuration*.

        The values are the normalized forms ``transform`` actually uses, so
        :meth:`from_dict` on the result reproduces a behaviorally identical
        (unfitted) instance. Nothing fitted is stored: the step keeps no fit
        state beyond its provenance bookkeeping, and serving reuses the trained
        pipeline's own fitted instance anyway.
        """
        rolling = _normalize_rolling(self.rolling)
        fourier = self.fourier
        return {
            "spec": TemporalSpec.from_dict(self.spec).to_dict(),
            "target": str(self.target),
            "lags": [int(lag) for lag in (self.lags or ())],
            "seasonal_lags": [int(period) for period in (self.seasonal_lags or ())],
            "rolling": [
                {"window": int(window), "stats": list(stats)}
                for window, stats in rolling
            ],
            "diffs": [int(step) for step in (self.diffs or ())],
            "pct_changes": [int(step) for step in (self.pct_changes or ())],
            "calendar": list(self.calendar or ()),
            "fourier": None if not fourier else {
                "periods": list(fourier.get("periods", ())),
                "terms": int(fourier.get("terms", 1)),
            },
            "exogenous": list(self.exogenous or ()),
            "min_periods": self.min_periods,
            "dropna": bool(self.dropna),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ForecastFeatureBuilder":
        """Rebuild an unfitted builder from a :meth:`to_dict` mapping.

        Unknown fields are refused and the reconstructed instance is validated
        against its spec (lags vs horizon, ...), so a corrupted or truncated
        bundle block fails loudly at load time rather than mid-forecast.
        """
        if not isinstance(value, Mapping):
            raise TypeError(
                "feature_spec must be a mapping (see ForecastFeatureBuilder.to_dict)"
            )
        allowed = {
            "spec",
            "target",
            "lags",
            "seasonal_lags",
            "rolling",
            "diffs",
            "pct_changes",
            "calendar",
            "fourier",
            "exogenous",
            "min_periods",
            "dropna",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError("unknown feature_spec field(s): %s" % ", ".join(unknown))
        missing = sorted({"spec", "target"} - set(value))
        if missing:
            raise ValueError("feature_spec is missing required field(s): %s" % ", ".join(missing))
        return cls(
            spec=value["spec"],
            target=value["target"],
            lags=tuple(int(lag) for lag in value.get("lags", ())),
            seasonal_lags=tuple(int(period) for period in value.get("seasonal_lags", ())),
            rolling=[dict(entry) for entry in value.get("rolling", ())],
            diffs=tuple(int(step) for step in value.get("diffs", ())),
            pct_changes=tuple(int(step) for step in value.get("pct_changes", ())),
            calendar=tuple(str(name) for name in value.get("calendar", ())),
            fourier=None if not value.get("fourier") else dict(value["fourier"]),
            exogenous=tuple(str(column) for column in value.get("exogenous", ())),
            min_periods=value.get("min_periods"),
            dropna=bool(value.get("dropna", True)),
        )


def _safe_pct_change(series: pd.Series, horizon: int, step: int) -> pd.Series:
    """Relative change from ``t-h-step`` to ``t-h``; zero denominators give NaN."""
    numerator = series.shift(horizon)
    denominator = series.shift(horizon + step)
    safe = denominator.where(denominator != 0)
    return (numerator - denominator) / safe.abs()


def _calendar_features(time_values: pd.Series, fields) -> dict[str, pd.Series]:
    index = time_values.index
    mapping = {
        "hour": time_values.dt.hour,
        "minute": time_values.dt.minute,
        "dow": time_values.dt.dayofweek,
        "dom": time_values.dt.day,
        "doy": time_values.dt.dayofyear,
        "month": time_values.dt.month,
        "week": time_values.dt.isocalendar().week.astype(int),
        "is_weekend": (time_values.dt.dayofweek >= 5).astype(int),
    }
    return {"cal__%s" % name: mapping[name].set_axis(index) for name in fields}


def _fourier_features(time_values, frame, fourier) -> dict[str, pd.Series]:
    """Sine/cosine terms driven by the timestamp, so they are known in advance."""
    periods, terms = fourier
    step_ns = _step_nanoseconds(time_values)
    position = time_values.astype("int64").to_numpy(dtype="float64") / step_ns
    position = position - position.min()
    out = {}
    for period in periods:
        for term in range(1, terms + 1):
            angle = 2.0 * np.pi * term * position / period
            out["fourier__sin_%g_%d" % (period, term)] = pd.Series(
                np.sin(angle), index=frame.index
            )
            out["fourier__cos_%g_%d" % (period, term)] = pd.Series(
                np.cos(angle), index=frame.index
            )
    return out


def _step_nanoseconds(time_values: pd.Series) -> float:
    """Median sampling step of a datetime series, in nanoseconds."""
    unique = np.unique(time_values.astype("int64").to_numpy())
    if len(unique) < 2:
        return 1.0
    deltas = np.diff(unique)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return 1.0
    return float(np.median(deltas))


def make_supervised(
    frame: pd.DataFrame,
    spec: Any,
    target: str,
    strategy: str = "direct",
    **builder_kwargs: Any,
) -> ForecastDesign:
    """Build a supervised design matrix in one call.

    ``strategy="direct"`` predicts the target ``horizon`` steps ahead (one model
    per horizon); ``strategy="recursive"`` predicts the next step only and feeds
    its own prediction back at serve time, so ``horizon`` stays the metadata
    horizon and lags must still respect it.

    Any further keyword is forwarded to :class:`ForecastFeatureBuilder`.
    """
    if strategy not in ("direct", "recursive"):
        raise ValueError("strategy must be 'direct' or 'recursive'")
    resolved = TemporalSpec.from_dict(spec)
    builder = ForecastFeatureBuilder(spec=resolved, target=target, **builder_kwargs)
    builder.fit(frame)
    design = builder.build(frame, target_horizon=1 if strategy == "recursive" else None)
    design.strategy = strategy
    return design


def assert_no_leakage(
    transform: Any,
    frame: pd.DataFrame,
    spec: Any,
    cuts: int = 3,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Empirically verify that ``transform`` reads no post-origin observations.

    The check corrupts every non-temporal column strictly after a cut time (so
    the "future" becomes missing) and rebuilds the features. Rows whose origin is
    at or before that cut must be unchanged — a feature that shifts the future in
    will change value, or disappear because it now reads a missing observation.

    This is a strong structural guard, not a proof: it is exactly the property
    the builder promises, tested by perturbation. Returns a small report.

    :param transform: a fitted :class:`ForecastFeatureBuilder` or any callable
        taking the frame and returning a feature DataFrame.
    :param cuts: how many cut points (from the end of each series) to test.
    :raises LeakageError: on the first feature that changes.
    """
    spec = TemporalSpec.from_dict(spec)
    callable_transform = transform if callable(transform) else transform.transform
    baseline = callable_transform(frame)
    if not isinstance(baseline, pd.DataFrame):
        raise TypeError("assert_no_leakage expects a DataFrame-returning transform")

    temporal = set(spec.temporal_columns)
    corruptible = [column for column in frame.columns if column not in temporal]
    time_values = frame[spec.time_column]
    groups = (
        [frame]
        if spec.series_id is None
        else [sub for _, sub in frame.groupby(spec.series_id, sort=False)]
    )

    checks = 0
    for group in groups:
        ordered = group.sort_values(spec.time_column)
        if len(ordered) <= spec.horizon + 1:
            continue
        candidates = ordered.index[-(spec.horizon + 1):]
        for cut_position in candidates[:cuts]:
            cut_time = frame.loc[cut_position, spec.time_column]
            corrupted = frame.copy()
            after = (time_values > cut_time).to_numpy()
            if not after.any():
                continue
            corrupted.loc[after, corruptible] = np.nan
            rebuilt = callable_transform(corrupted)
            # Only origins the baseline actually produced can be compared: rows
            # dropped by the baseline (incomplete lag vectors) are not evidence.
            eligible = frame.index[(time_values <= cut_time).to_numpy()]
            before = baseline.index.intersection(eligible)
            common = before.intersection(rebuilt.index)
            missing = before.difference(rebuilt.index)
            if len(missing):
                raise LeakageError(
                    "feature rows for origins at or before %r disappeared when the "
                    "future was masked (rows %s): a feature reads post-origin data"
                    % (cut_time, list(missing[:5]))
                )
            left = baseline.loc[common].to_numpy(dtype="float64")
            right = rebuilt.loc[common].to_numpy(dtype="float64")
            if not np.allclose(left, right, atol=tolerance, equal_nan=True, rtol=0.0):
                names = list(baseline.columns)
                differing = [
                    names[index]
                    for index in range(len(names))
                    if not np.allclose(
                        left[:, index], right[:, index], atol=tolerance, equal_nan=True, rtol=0.0
                    )
                ]
                raise LeakageError(
                    "feature(s) %s change when observations after %r are masked"
                    % (", ".join(differing), cut_time)
                )
            checks += 1
    return {"ok": True, "cut_points_checked": checks, "temporal_spec": spec.to_dict()}


# ---------------------------------------------------------------------------
# Rolling-origin cross-validation and backtesting (plan phase P3)
# ---------------------------------------------------------------------------


def rolling_origin_indices(
    n_samples: int,
    n_splits: int = 3,
    horizon: int = 1,
    step: int = 1,
    min_train: int | None = None,
    expanding: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Row-index folds for a rolling-origin (walk-forward) evaluation.

    Each fold trains on observations strictly before an origin and tests on the
    ``horizon`` observations immediately after it, so the test window is always
    the *future* of the training window. Origins move forward by ``step`` between
    folds; the last fold ends at the final observation.

    :param n_samples: number of ordered observations.
    :param n_splits: number of origins to evaluate.
    :param horizon: test-window length in steps; also the forecast horizon.
    :param step: distance between consecutive origins.
    :param min_train: training-window length when ``expanding=False``.
    :param expanding: grow the training window from the first observation
        (default) or slide a fixed ``min_train`` window.
    :raises ValueError: when the series is too short for the requested geometry.
    """
    n_samples, n_splits, horizon, step = (
        int(n_samples),
        int(n_splits),
        int(horizon),
        int(step),
    )
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if step < 1:
        raise ValueError("step must be >= 1")
    last_end = n_samples - horizon
    first_end = last_end - (n_splits - 1) * step
    if first_end < 1:
        raise ValueError(
            "not enough observations for %d origin(s) with horizon=%d and step=%d: "
            "the first origin would leave %d training row(s)"
            % (n_splits, horizon, step, first_end)
        )
    folds = []
    for fold in range(n_splits):
        end = first_end + fold * step
        if expanding:
            start = 0
        else:
            if min_train is None:
                raise ValueError("min_train is required when expanding=False")
            start = end - int(min_train)
            if start < 0:
                raise ValueError(
                    "min_train=%d does not fit before origin %d" % (int(min_train), end)
                )
        folds.append(
            (np.arange(start, end, dtype="int64"), np.arange(end, end + horizon, dtype="int64"))
        )
    return folds


class RollingOriginSplit:
    """Walk-forward splitter with an explicit forecast horizon.

    ``TimeSeriesSplit`` is a fixed-window approximation: its test window must be
    smaller than the training window, which makes long horizons impossible on
    short series. ``RollingOriginSplit`` states the horizon directly and lets
    origins advance by ``step`` instead.

    Compatible with ``cv_to_spec``/``build_cv`` and with scikit-learn's
    ``split``/``get_n_splits`` protocol; listed in
    :data:`SKSurrogate.mltrace.TEMPORAL_CV_SPLITTERS`.
    """

    def __init__(
        self,
        n_splits: int = 3,
        horizon: int = 1,
        step: int = 1,
        min_train: int | None = None,
        expanding: bool = True,
    ):
        self.n_splits = n_splits
        self.horizon = horizon
        self.step = step
        self.min_train = min_train
        self.expanding = expanding

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)`` for each origin."""
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        yield from rolling_origin_indices(
            n_samples,
            n_splits=self.n_splits,
            horizon=self.horizon,
            step=self.step,
            min_train=self.min_train,
            expanding=self.expanding,
        )

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return int(self.n_splits)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            "RollingOriginSplit(n_splits=%r, horizon=%r, step=%r, min_train=%r, "
            "expanding=%r)"
            % (
                self.n_splits,
                self.horizon,
                self.step,
                self.min_train,
                self.expanding,
            )
        )


# ---------------------------------------------------------------------------
# Bundled multi-step inference (plan phase P6)
# ---------------------------------------------------------------------------

#: Format tag of the ``temporal`` block stored in a v2 model-bundle record.
TEMPORAL_BLOCK_FORMAT = 1


@dataclass(frozen=True)
class TemporalServeState:
    """The bundle-side view of a temporal design, enough to serve forecasts.

    A fitted :class:`ForecastFeatureBuilder` is behaviorally defined by its
    constructor arguments (``fit`` only validates and stores provenance), so the
    state persists the *specification* — spec, target column, feature
    configuration, strategy — not any fit artifact. The trained pipeline keeps
    its own fitted builder for prediction; this object owns the serving
    geometry instead: how much history is required, which rows to evaluate and
    what times a forecast covers.

    Build one with :meth:`from_parts`, store it in a bundle record via
    :meth:`to_dict` (the ``temporal`` key of a v2 record) and rebuild it from
    the loaded dict with :meth:`from_dict`.
    """

    spec: Any = None  # TemporalSpec or mapping; normalized in __post_init__
    target: str = ""
    feature_spec: Mapping[str, Any] = field(default_factory=dict)
    strategy: str = "direct"

    def __post_init__(self):
        if self.strategy not in ("direct", "recursive"):
            raise ValueError("strategy must be 'direct' or 'recursive'")
        object.__setattr__(self, "spec", TemporalSpec.from_dict(self.spec))
        # Rebuilding the builder validates feature_spec against the spec (lag >=
        # horizon and friends), so a corrupted block fails here, at load time.
        self.builder()  # noqa: F841

    def __hash__(self):  # frozen dataclass + non-hashable mapping field
        return hash((str(self.spec.to_dict()), str(self.target), self.strategy))

    @classmethod
    def from_parts(
        cls,
        spec: Any,
        target: str,
        builder_or_feature_spec: Any,
        strategy: str = "direct",
    ) -> "TemporalServeState":
        """Build a serve state from a spec, target and a builder or its dict."""
        if isinstance(builder_or_feature_spec, ForecastFeatureBuilder):
            feature_spec = builder_or_feature_spec.to_dict()
        else:
            feature_spec = dict(builder_or_feature_spec)
        return cls(spec=spec, target=target, feature_spec=feature_spec, strategy=strategy)

    def to_dict(self) -> dict[str, Any]:
        """The ``temporal`` block of a v2 model-bundle record."""
        return {
            "format": TEMPORAL_BLOCK_FORMAT,
            "strategy": self.strategy,
            "spec": self.spec.to_dict(),
            "target": self.target,
            "feature_spec": dict(self.feature_spec),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "TemporalServeState":
        """Rebuild a state from the ``temporal`` block of a bundle record."""
        if not isinstance(value, Mapping):
            raise TypeError("the temporal block must be a mapping")
        unknown = sorted(set(value) - {"format", "strategy", "spec", "target", "feature_spec"})
        if unknown:
            raise ValueError("unknown temporal-block field(s): %s" % ", ".join(unknown))
        missing = sorted({"spec", "target", "feature_spec"} - set(value))
        if missing:
            raise ValueError("temporal block is missing required field(s): %s" % ", ".join(missing))
        version = int(value.get("format", TEMPORAL_BLOCK_FORMAT))
        if version != TEMPORAL_BLOCK_FORMAT:
            raise ValueError(
                "unsupported temporal-block format %d (expected %d)"
                % (version, TEMPORAL_BLOCK_FORMAT)
            )
        return cls(
            spec=value["spec"],
            target=str(value["target"]),
            feature_spec=dict(value["feature_spec"]),
            strategy=str(value.get("strategy", "direct")),
        )

    def builder(self) -> ForecastFeatureBuilder:
        """An unfitted instance with the stored configuration (``fit`` is a no-op)."""
        return ForecastFeatureBuilder.from_dict(self.feature_spec)

    @property
    def horizon(self) -> int:
        return self.spec.horizon

    @property
    def exogenous_columns(self) -> tuple[str, ...]:
        return tuple(self.builder()._resolved()["exogenous"])

    def required_length(self) -> int:
        """Minimum number of history observations needed to serve one forecast.

        The count is the smallest ``n`` for which every feature value read at any
        evaluated position is backed by real data, and it depends on strategy as
        well as each term's lookback (``H`` = horizon):

        * ``direct`` evaluates the final ``H`` origins (positions ``t-H+1 .. t``), so a lag or seasonal lag of step ``L`` needs ``n >= H + L``, a rolling window of width ``W`` needs ``n >= 2H + W - 1``, and a difference / percent-change of step ``k`` needs ``n >= 2H + k``;
        * ``recursive`` evaluates from position ``t`` and reads no more than
          ``H`` steps back — beyond that it feeds its own predictions (see
          :func:`serve_forecast`) — so the same terms need only ``n >= L``, ``n >= H + W - 1`` and ``n >= H + k`` respectively.

        The returned value is the maximum of these per-term bounds together with
        a strategy base (``H`` for direct, ``1`` for recursive). Calendar, fourier and exogenous terms read no history at all and add nothing; a target-derived feature is required for the geometry to make sense in practice, but a pure calendar/exogenous bundle serves from ``H`` observations of history under direct and one under recursive (the declared minimums).
        """
        horizon = self.horizon
        options = self.builder()._resolved()
        if self.strategy == "direct":
            needed = [horizon]
            needed.extend(horizon + lag for lag in options["lags"] + options["seasonal_lags"])
            needed.extend(2 * horizon + window - 1 for window, _stats in options["rolling"])
            step_terms = sorted(set(options["diffs"]) | set(options["pct_changes"]))
            needed.extend(2 * horizon + k for k in step_terms)
        else:
            needed = [1]
            needed.extend(lag for lag in options["lags"] + options["seasonal_lags"])
            needed.extend(horizon + window - 1 for window, _stats in options["rolling"])
            step_terms = sorted(set(options["diffs"]) | set(options["pct_changes"]))
            needed.extend(horizon + k for k in step_terms)
        return int(max(needed))

    def validate_history(self, history: Any) -> tuple[pd.DataFrame, Any]:
        """Normalize a single-series history frame and check it can serve.

        Returns ``(frame, series_label)`` where the frame has a clean
        increasing time axis (parsed to datetime when possible), a fresh range
        index and — for a panel spec — exactly one ``series_id`` value. Raises:

        * ``TypeError`` if ``history`` is not a DataFrame;
        * ``ValueError`` with an actionable message on missing columns, a
          non-increasing or duplicated time axis, more than one series, a
          timestamp grid that contradicts the declared frequency, or a history
          shorter than :meth:`required_length`.
        """
        spec = self.spec
        if not isinstance(history, pd.DataFrame):
            raise TypeError("history must be a pandas DataFrame")
        if len(history) == 0:
            raise ValueError("history is empty")

        required = [spec.time_column]
        if spec.series_id is not None:
            required.append(spec.series_id)
        required.append(self.target)
        missing = [column for column in required if column not in history.columns]
        if missing:
            raise ValueError("history is missing column(s): %s" % ", ".join(missing))
        missing_exog = [c for c in self.exogenous_columns if c not in history.columns]
        if missing_exog:
            raise ValueError(
                "exogenous column(s) not found in history: %s" % ", ".join(missing_exog)
            )

        frame = history.copy()
        raw_times = frame[spec.time_column]
        if pd.api.types.is_datetime64_any_dtype(raw_times):
            times = raw_times
        elif pd.api.types.is_numeric_dtype(raw_times):
            # A purely numeric axis is a unit-spaced index (periods, steps),
            # not an epoch encoding: keep it as float so the forecast axis
            # extrapolates in the caller's own units. pd.to_datetime would
            # reinterpret values like 60 as "60 nanoseconds since epoch".
            times = raw_times.astype("float64")
            frame[spec.time_column] = times
        else:
            parsed = pd.to_datetime(frame[spec.time_column], errors="coerce")
            if bool(parsed.notna().all()):
                frame[spec.time_column] = parsed
                times = frame[spec.time_column]
            else:
                numeric = pd.to_numeric(raw_times, errors="coerce")
                if not bool(numeric.notna().all()):
                    raise ValueError(
                        "time column %r holds neither parseable timestamps nor numbers"
                        % spec.time_column
                    )
                frame[spec.time_column] = numeric.astype("float64")
                times = frame[spec.time_column]

        if bool(times.isna().any()):
            raise ValueError("time column %r has missing values" % spec.time_column)
        if not bool(times.is_monotonic_increasing):
            raise ValueError(
                "history time values must be strictly increasing; sort the frame by %r"
                % spec.time_column
            )
        duplicates = int(times.duplicated().sum())
        if duplicates:
            raise ValueError(
                "history has %d duplicate timestamp(s) in %r" % (duplicates, spec.time_column)
            )

        label = None
        if spec.series_id is not None:
            labels = frame[spec.series_id].dropna().unique()
            if len(labels) != 1:
                raise ValueError(
                    "history must contain exactly one %r value; got %d (one forecast per series)"
                    % (spec.series_id, len(labels))
                )
            label = labels[0]

        if spec.freq is not None and pd.api.types.is_datetime64_any_dtype(times):
            inferred = infer_freq(times)
            if inferred is not None:
                declared_name = to_offset(spec.freq).name
                inferred_name = to_offset(inferred).name
                # Alias spellings drift across pandas versions ("15min" vs "T");
                # only refuse when the two aliases disagree even normalized.
                normalize = lambda alias: str(alias).upper().replace("MIN", "T")  # noqa: E731
                if normalize(declared_name) != normalize(inferred_name):
                    raise ValueError(
                        "history timestamps step by %r but the bundle was trained on a "
                        "%r grid; serve it with history at the declared frequency"
                        % (inferred, spec.freq)
                    )

        needed = self.required_length()
        if len(frame) < needed:
            raise ValueError(
                "history has %d observation(s); this bundle needs at least %d for a %s "
                "forecast — extend the history before serving"
                % (len(frame), needed, ("direct" if self.strategy == "direct" else "recursive"))
            )
        return frame.reset_index(drop=True), label


@dataclass
class ForecastResult:
    """One multi-step forecast from a temporal bundle.

    ``means``/``lower``/``upper`` are aligned with ``times`` (the horizon many
    steps after the last history observation). ``lower`` and ``upper`` are
    ``None`` unless the bundle carries an :class:`IntervalCalibration`; when
    they exist, ``interval_note`` says what the offsets were calibrated on.
    """

    series_id: Any
    times: list
    means: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    horizon: int = 0
    strategy: str = "direct"
    interval_note: str | None = None

    def to_frame(self) -> pd.DataFrame:
        """Long-form table: one row per forecast step."""
        lower = (
            self.lower if self.lower is not None else np.full(len(self.means), np.nan)
        )
        upper = (
            self.upper if self.upper is not None else np.full(len(self.means), np.nan)
        )
        return pd.DataFrame(
            {
                "series_id": [self.series_id] * len(self.times),
                "time": list(self.times),
                "mean": np.asarray(self.means, dtype="float64"),
                "lower": np.asarray(lower, dtype="float64"),
                "upper": np.asarray(upper, dtype="float64"),
            }
        )


def _extend_axis(time_values: pd.Series, count: int, freq: str | None = None) -> list[Any]:
    """The next ``count`` values of the history's time axis.

    A datetime axis steps by the declared frequency when one was given; a
    numeric (or irregular, or un-declared) axis extrapolates its median step.
    The synthetic times both build the serving rows and label the result, so
    they must match the axis kind of ``time_values`` exactly.
    """
    if count <= 0:
        return []
    last = time_values.iloc[-1]
    if pd.api.types.is_datetime64_any_dtype(time_values):
        if freq is not None:
            try:
                offset = to_offset(freq)
                return [last + (k + 1) * offset for k in range(count)]
            except (ValueError, TypeError):
                pass  # fall back to the observed step below
        values = pd.to_datetime(time_values).astype("int64").to_numpy(dtype="float64")
    else:
        values = pd.to_numeric(time_values).to_numpy(dtype="float64")
    step = _numeric_step(np.unique(values))
    if not step or step <= 0:
        raise ValueError(
            "cannot extend the forecast axis from %d observation(s) without a declared "
            "frequency" % len(time_values)
        )
    last_value = float(values[-1])
    return [pd.Timestamp(int(last_value + (k + 1) * step)) if isinstance(last, pd.Timestamp) else last_value + (k + 1) * step for k in range(count)]


def recursive_steps(
    transform: Any,
    predictor: Any,
    working: pd.DataFrame,
    target: str,
    horizon: int,
    *,
    future_times: Sequence[Any],
    build_row: Any,
) -> np.ndarray:
    """Forecast ``horizon`` steps one at a time, feeding each prediction back.

    This is the single implementation of recursive multi-step forecasting: the
    serving path (:func:`serve_forecast`) and the rolling-origin backtest
    (``mltrack.backtest(strategy="recursive")``) both call it, so a served
    forecast and its offline evaluation cannot drift apart.

    :param transform: row-preserving feature function, ``frame -> DataFrame``.
    :param predictor: fitted estimator taking a one-row feature frame.
    :param working: history frame, already trimmed to the forecast origin.
    :param target: column that receives each step's prediction.
    :param horizon: number of steps to forecast.
    :param future_times: one timestamp per step, used for the synthetic row and
        for error messages.
    :param build_row: ``(working, step_index, prediction) -> dict`` building the
        synthetic row (time, series label, exogenous values).
    :raises ValueError: on incomplete features, a non-finite prediction, or a
        transform that does not preserve the row count.
    """
    values: list[float] = []
    for step in range(int(horizon)):
        features = transform(working)
        if len(features) != len(working):
            raise ValueError(
                "a preprocessing step changed the row count; only row-preserving "
                "steps can serve a temporal design"
            )
        x_row = features.iloc[-1]
        missing_values = [name for name, value in x_row.items() if pd.isna(value)]
        if missing_values:
            raise ValueError(
                "incomplete feature values at forecast step %d (%s): columns %s are "
                "not backed by the history — supply future exogenous rows or a "
                "longer history"
                % (step + 1, future_times[step], ", ".join(missing_values))
            )
        prediction = float(np.asarray(predictor.predict(x_row.to_frame().T)).ravel()[0])
        if not np.isfinite(prediction):
            raise ValueError(
                "the model produced a non-finite forecast value at step %d (%s); "
                "check the history for gaps or an exogenous value it cannot "
                "extrapolate" % (step + 1, future_times[step])
            )
        row = build_row(working, step, prediction)
        working = pd.concat(
            [working, pd.DataFrame([row], columns=working.columns)], ignore_index=True
        )
        values.append(prediction)
    return np.asarray(values, dtype="float64")


def _serving_steps(bundle: Any):
    """Locate the pipeline's fitted builder and build row-preserving serving steps.

    Returns ``(transformer_pipeline, predictor, builder_index)`` where every
    transformer is reused by reference (fitted state intact) except the
    :class:`ForecastFeatureBuilder` itself, which gets a behaviorally identical
    twin with ``dropna=False``: dropping rows would silently shift row *i* of
    the output away from row *i* of the frame, and serving must select exact
    positions. The original bundle is not mutated.
    """
    from sklearn.pipeline import Pipeline

    model = getattr(bundle, "model", None)
    steps = getattr(model, "steps", None)
    if not isinstance(steps, list) or not steps:
        raise ValueError(
            "a temporal bundle's model must be a scikit-learn pipeline that starts with "
            "the ForecastFeatureBuilder it was trained with"
        )
    for index, (_name, estimator) in enumerate(steps):
        if isinstance(estimator, ForecastFeatureBuilder):
            break
    else:
        raise ValueError(
            "the bundle's pipeline has no ForecastFeatureBuilder step; a tabular model "
            "cannot be served as a forecast"
        )
    serving_steps = []
    for i, (name, estimator) in enumerate(steps):
        if i == index and isinstance(estimator, ForecastFeatureBuilder):
            serving_steps.append((name, estimator.serving_twin()))
        else:
            serving_steps.append((name, estimator))
    transformers = Pipeline(serving_steps[:-1]) if len(serving_steps) > 1 else None
    predictor = serving_steps[-1][1]
    if isinstance(predictor, ForecastFeatureBuilder):
        raise ValueError(
            "the pipeline ends with the feature builder; it must end with an estimator"
        )
    return transformers, predictor, index


def serve_forecast(
    bundle: Any,
    history: pd.DataFrame,
    horizon: int | None = None,
    future_exogenous: Any = None,
) -> ForecastResult:
    """Produce a multi-step forecast from a temporal model bundle.

    :param bundle: a :class:`~SKSurrogate.modelbundle.ModelBundle` whose record
        carries the ``temporal`` block written by plan phase P6 (a v1 tabular
        bundle raises ``ValueError``). The duck-typed contract is only
        ``.model``, ``.temporal`` and ``.interval_calibration``.
    :param history: one series of past observations as a DataFrame with the
        spec's time column, target column (and series_id / exogenous columns
        when declared), strictly increasing in time.
    :param horizon: forecast length in steps; defaults to the design horizon.
        A ``direct`` bundle answers exactly its design horizon and refuses any
        other value; a ``recursive`` bundle accepts any positive horizon — its
        one-step model feeds predictions back into its own features.
    :param future_exogenous: optional frame (or list of dicts) with the time
        column plus every declared exogenous column, for the forecast steps.
        Required when exogenous columns are declared and the strategy is
        recursive; ignored by a direct bundle, which reads only history.

    The predictions reuse the exact pipeline stored in ``bundle.model``, so a
    serving result matches what an offline backtest produced for the same
    origin — down to the floats (see ``tests/test_forecast_inference.py``).
    """
    block = getattr(bundle, "temporal", None)
    if not isinstance(block, Mapping):
        raise ValueError(
            "this bundle has no temporal design; it cannot produce forecasts"
        )
    state = TemporalServeState.from_dict(block)
    spec = state.spec

    if horizon is None:
        requested = int(spec.horizon)
    else:
        if isinstance(horizon, bool):
            raise ValueError("horizon must be a positive integer")
        requested = int(horizon)
        if requested < 1:
            raise ValueError("horizon must be >= 1")
    if state.strategy == "direct" and requested != spec.horizon:
        raise ValueError(
            "a direct-strategy bundle can only forecast its design horizon (%d); "
            "requesting %d is refused rather than answered by extrapolation"
            % (spec.horizon, requested)
        )

    frame, label = state.validate_history(history)
    times_series = frame[spec.time_column]
    synthetic_times = _extend_axis(times_series, requested, freq=spec.freq)

    transformers, predictor, _builder_index = _serving_steps(bundle)
    target = state.target  # TemporalSpec carries no target; it lives on the state

    exog_frame: pd.DataFrame | None = None
    if future_exogenous is not None and state.exogenous_columns:
        if isinstance(future_exogenous, (list, tuple)):
            exog_frame = pd.DataFrame(list(future_exogenous))
        elif isinstance(future_exogenous, pd.DataFrame):
            exog_frame = future_exogenous.copy()
        else:
            raise TypeError("future_exogenous must be a DataFrame or list of dicts")
        missing = [c for c in state.exogenous_columns if c not in exog_frame.columns]
        if missing:
            raise ValueError(
                "future_exogenous is missing column(s): %s" % ", ".join(missing)
            )

    # ``recursive_steps`` is shared with ``mltrack.backtest(strategy="recursive")``,
    # so a served forecast and its walk-forward evaluation cannot diverge.
    if state.strategy == "direct":
        n = len(frame)
        positions = np.arange(n - requested, n, dtype="int64")
        X_all = transformers.transform(frame) if transformers is not None else pd.DataFrame(
            index=frame.index
        )
        if len(X_all) != n:
            raise ValueError(
                "a preprocessing step changed the row count (%d -> %d); only row-preserving "
                "steps can serve a temporal bundle" % (n, len(X_all))
            )
        selected = X_all.iloc[positions]
        bad = []
        for position in positions:
            row_bad = [
                name
                for name, value in selected.iloc[position - positions[0]].items()
                if pd.isna(value)
            ]
            if row_bad:
                bad.append((frame.loc[position, spec.time_column], row_bad))
        if bad:
            detail = "; ".join("%s: %s" % (t, ", ".join(cols)) for t, cols in bad[:3])
            raise ValueError(
                "incomplete feature values at forecast origin(s) — %s; the history has gaps "
                "or ends too close to its last observation" % detail
            )
        predictions = np.asarray(predictor.predict(selected), dtype="float64").ravel()
        if not np.all(np.isfinite(predictions)):
            raise ValueError(
                "the model produced non-finite forecast values; check the history for gaps or an "
                "exogenous value it cannot extrapolate"
            )
    else:

        def _build_row(working: pd.DataFrame, step: int, prediction: float) -> dict:
            row = {column: np.nan for column in working.columns}
            row[spec.time_column] = synthetic_times[step]
            if spec.series_id is not None:
                row[spec.series_id] = label
            row[target] = prediction
            if exog_frame is not None and len(exog_frame):
                match = _exogenous_row(exog_frame, spec.time_column, synthetic_times[step])
                for column in state.exogenous_columns:
                    value = match.get(column) if isinstance(match, Mapping) else np.nan
                    row[column] = (
                        np.nan
                        if value is None or (isinstance(value, float) and pd.isna(value))
                        else value
                    )
            return row

        def _transform(working: pd.DataFrame) -> pd.DataFrame:
            if transformers is None:
                return pd.DataFrame(index=working.index)
            return transformers.transform(working)

        predictions = recursive_steps(
            _transform,
            predictor,
            frame.copy(),
            target,
            requested,
            future_times=synthetic_times,
            build_row=_build_row,
        )

    lower = upper = None
    interval_note = None
    calibration_value = getattr(bundle, "interval_calibration", None)
    if calibration_value is not None:
        from .forecast_intervals import IntervalCalibration  # local: avoid a cycle at import time

        calibration = (
            calibration_value
            if isinstance(calibration_value, IntervalCalibration)
            else IntervalCalibration.from_dict(calibration_value)
        )
        lower, upper = calibration.bounds(predictions)
        interval_note = (
            "intervals apply the calibrated offsets (method=%s, alpha=%s, target_step=%d, "
            "n_residuals=%d) to every step of the forecast; they were fitted on backtest "
            "residuals and do not model error compounding along multi-step paths, so early "
            "steps may be covered more often than nominal"
            % (
                calibration.method,
                calibration.alpha,
                int(calibration.target_step),
                calibration.n_residuals,
            )
        )

    return ForecastResult(
        series_id=label,
        times=list(synthetic_times),
        means=predictions.astype("float64"),
        lower=None if lower is None else np.asarray(lower).ravel(),
        upper=None if upper is None else np.asarray(upper).ravel(),
        horizon=requested,
        strategy=state.strategy,
        interval_note=interval_note,
    )


def _exogenous_row(exog_frame: pd.DataFrame, time_column: str, wanted):
    """The exogenous row whose timestamp equals ``wanted`` (mapping or None).

    Timestamps are compared as values on a common axis so a datetime request
    matches ISO strings and vice versa; numeric axes match numerically.
    """
    parsed_wanted = pd.Timestamp(wanted) if not isinstance(wanted, (int, float)) else wanted
    for index in range(len(exog_frame)):
        candidate = exog_frame.iloc[index][time_column]
        try:
            if pd.api.types.is_datetime64_any_dtype(exog_frame[time_column]) or isinstance(candidate, str):
                if pd.Timestamp(candidate) == (parsed_wanted if isinstance(parsed_wanted, pd.Timestamp) else pd.Timestamp(str(wanted))):
                    return exog_frame.iloc[index].to_dict()
            elif float(candidate) == float(wanted):
                return exog_frame.iloc[index].to_dict()
        except (TypeError, ValueError):
            continue
    return None
