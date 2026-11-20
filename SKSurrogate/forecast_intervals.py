"""Horizon-aware prediction intervals for forecast backtests (plan phase P5).

A point forecast without an interval hides the only thing a planner needs: how wrong
it is likely to be. This module turns the per-origin residuals of a rolling-origin
backtest into a *calibrated* interval that serving can apply to a prediction without
refitting anything.

The construction is split-conformal on the backtest residuals:

* residuals are ``y_true - y_pred`` for origins the model never trained on;
* the interval is ``prediction + [lower_offset, upper_offset]`` where the offsets are
  order statistics of those residuals at level ``alpha``;
* ``method="absolute"`` gives a symmetric interval from the
  :math:`\\lceil (n+1)(1-\\alpha)\\rceil / n` quantile of ``|residual|``;
  ``method="asymmetric"`` uses the empirical ``alpha/2`` and ``1 - alpha/2``
  quantiles, which is what a skewed error distribution needs.

Two honest caveats, stated in the output rather than buried here:

* Coverage is **marginal and approximate**. The finite-sample guarantee assumes
  exchangeable residuals; a walk-forward backtest is ordered, so the empirical
  coverage on held-out origins can drift from nominal. The diagnostics report both,
  with a Wilson interval, instead of asserting the nominal number.
* A short calibration set produces a wide, jagged interval. Below
  ``min_residuals`` (default 10) calibration refuses rather than fabricating a
  quantile from three points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

#: Interval constructions this module implements.
CONFORMAL_METHODS = ("absolute", "asymmetric")


@dataclass
class IntervalCalibration:
    """Residual offsets that turn a point forecast into a calibrated interval."""

    alpha: float
    method: str
    lower_offset: float
    upper_offset: float
    n_residuals: int
    target_step: int = 1
    empirical_coverage: float | None = None
    finite_sample_level: float | None = None
    residual_scale: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def nominal_coverage(self) -> float:
        return float(1.0 - self.alpha)

    def bounds(self, predictions):
        """Return ``(lower, upper)`` arrays for point ``predictions``."""
        values = np.asarray(predictions, dtype="float64").ravel()
        return values + self.lower_offset, values + self.upper_offset

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "nominal_coverage": self.nominal_coverage,
            "method": self.method,
            "lower_offset": self.lower_offset,
            "upper_offset": self.upper_offset,
            "n_residuals": int(self.n_residuals),
            "target_step": int(self.target_step),
            "empirical_coverage": self.empirical_coverage,
            "finite_sample_level": self.finite_sample_level,
            "residual_scale": self.residual_scale,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IntervalCalibration":
        if not isinstance(value, Mapping):
            raise TypeError("interval calibration must be a mapping")
        return cls(
            alpha=float(value["alpha"]),
            method=str(value["method"]),
            lower_offset=float(value["lower_offset"]),
            upper_offset=float(value["upper_offset"]),
            n_residuals=int(value["n_residuals"]),
            target_step=int(value.get("target_step", 1)),
            empirical_coverage=value.get("empirical_coverage"),
            finite_sample_level=value.get("finite_sample_level"),
            residual_scale=value.get("residual_scale"),
            notes=list(value.get("notes") or []),
        )


def conformal_quantile_level(n_residuals: int, alpha: float) -> float:
    """Finite-sample quantile level ``ceil((n+1)(1-alpha)) / n``, capped at 1."""
    n_residuals = int(n_residuals)
    if n_residuals < 1:
        raise ValueError("at least one residual is required")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    level = np.ceil((n_residuals + 1) * (1.0 - float(alpha))) / n_residuals
    return float(min(max(level, 0.0), 1.0))


def calibrate_from_residuals(
    residuals,
    *,
    alpha: float = 0.1,
    method: str = "absolute",
    target_step: int = 1,
    min_residuals: int = 10,
) -> IntervalCalibration:
    """Build an :class:`IntervalCalibration` from held-out residuals.

    :param residuals: ``y_true - y_pred`` for origins the model did not train on.
    :param alpha: miscoverage level; the interval targets ``1 - alpha`` coverage.
    :param method: ``"absolute"`` (symmetric) or ``"asymmetric"``.
    :param target_step: the horizon these residuals belong to (metadata only).
    :param min_residuals: refuse to calibrate from fewer residuals than this.
    """
    values = np.asarray(residuals, dtype="float64").ravel()
    values = values[np.isfinite(values)]
    alpha = float(alpha)
    method = str(method)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if method not in CONFORMAL_METHODS:
        raise ValueError(
            "unknown interval method %r; expected one of %s"
            % (method, ", ".join(CONFORMAL_METHODS))
        )
    min_residuals = int(min_residuals)
    if values.size < max(min_residuals, 1):
        raise ValueError(
            "refusing to calibrate intervals from %d residual(s); at least %d are "
            "required for a meaningful quantile" % (values.size, max(min_residuals, 1))
        )

    notes = []
    ordered = np.sort(values)
    if method == "absolute":
        level = conformal_quantile_level(values.size, alpha)
        offset = float(np.quantile(np.abs(values), level, method="higher"))
        lower_offset, upper_offset = -offset, offset
        if level >= 1.0:
            notes.append(
                "the conformal level reached 1.0: the interval is the full observed "
                "residual range, so coverage is conservative"
            )
    else:
        # Order-statistic quantiles with the conformal indices, clipped to the sample.
        low_index = int(np.floor((values.size + 1) * (alpha / 2.0)))
        high_index = int(np.ceil((values.size + 1) * (1.0 - alpha / 2.0))) - 1
        low_index = min(max(low_index, 0), values.size - 1)
        high_index = min(max(high_index, 0), values.size - 1)
        lower_offset = float(ordered[low_index])
        upper_offset = float(ordered[high_index])
        level = None

    covered = (values >= lower_offset) & (values <= upper_offset)
    calibration = IntervalCalibration(
        alpha=alpha,
        method=method,
        lower_offset=lower_offset,
        upper_offset=upper_offset,
        n_residuals=int(values.size),
        target_step=int(target_step),
        empirical_coverage=float(np.mean(covered)),
        finite_sample_level=level if method == "absolute" else None,
        residual_scale=float(np.mean(np.abs(values))),
        notes=notes,
    )
    if calibration.lower_offset > calibration.upper_offset:  # pragma: no cover - defensive
        raise ValueError("calibration produced an inverted interval")
    return calibration


def calibrate_backtest(result: Mapping[str, Any], *, alpha: float = 0.1, method: str = "absolute", min_residuals: int = 10) -> IntervalCalibration:
    """Calibrate intervals from an ``mltrack.backtest`` result."""
    origins = result.get("origins") or []
    if not origins:
        raise ValueError("the backtest result carries no per-origin records to calibrate from")
    residuals = [float(record["y_true"]) - float(record["y_pred"]) for record in origins]
    target_step = int(origins[0].get("target_step", 1))
    return calibrate_from_residuals(
        residuals,
        alpha=alpha,
        method=method,
        target_step=target_step,
        min_residuals=min_residuals,
    )


def interval_diagnostics(y_true, lower, upper, *, alpha: float = 0.1, horizon=None) -> dict[str, Any]:
    """Coverage/width diagnostics, optionally per horizon step.

    Delegates to :func:`SKSurrogate.mltrace.prediction_interval_diagnostics` (which
    owns the Wilson interval and the Winkler score) and passes ``horizon`` as its
    ``groups`` argument, so a horizon-1 interval and a horizon-12 interval are
    reported separately instead of being averaged into one misleading number.
    """
    from .mltrace import prediction_interval_diagnostics

    report = prediction_interval_diagnostics(
        y_true, lower, upper, alpha=alpha, groups=horizon
    )
    report["horizon_grouped"] = horizon is not None
    return report


def apply_calibration(predictions, calibration: Any):
    """Apply a calibration dict/dataclass to point predictions."""
    if isinstance(calibration, Mapping):
        calibration = IntervalCalibration.from_dict(calibration)
    if not isinstance(calibration, IntervalCalibration):
        raise TypeError("calibration must be an IntervalCalibration or its mapping form")
    return calibration.bounds(predictions)
