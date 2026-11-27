"""Forecast accuracy metrics and naive baselines (plan phase P4).

Point metrics, quantile/interval scores, and the baselines that make a forecast
score interpretable: a model that cannot beat ``seasonal_naive`` on MASE has not
earned the word "forecast".

Two conventions are worth stating because they are easy to get wrong:

* **MASE is scaled by the in-sample series, never by the error of the forecast.**
  The denominator is the mean absolute ``seasonality``-step difference over the
  training history,
  $\\frac{1}{n-m}\\sum_t |y_t - y_{t-m}|$.
  A seasonal-naive forecast scored on that same history therefore has MASE
  exactly 1, which is what makes "worse than naive" a readable statement.
* **sMAPE is reported in percent** with the symmetric denominator
  $|f| + |y|$; a term where both are zero contributes 0 instead of dividing by
  zero. It is a scale-free descriptive measure, not a loss to optimise.

Only NumPy and pandas are used here.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

#: Baselines every temporal backtest can produce.
#: Baseline forecasters every temporal backtest can score. ``mean`` and
#: ``historical_mean_by_series`` are the same computation because a backtest
#: always conditions on one series' own history; the second name exists because
#: a caller coming from a pooled setting expects it stated explicitly.
BASELINE_METHODS = ("naive", "seasonal_naive", "drift", "mean", "historical_mean_by_series")


def _arrays(y_true, y_pred):
    actual = np.asarray(y_true, dtype="float64").ravel()
    predicted = np.asarray(y_pred, dtype="float64").ravel()
    if actual.shape != predicted.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape: %s vs %s"
            % (actual.shape, predicted.shape)
        )
    if actual.size == 0:
        raise ValueError("at least one observation is required")
    return actual, predicted


def _finite_or_nan(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    actual, predicted = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    actual, predicted = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def bias(y_true, y_pred) -> float:
    """Mean signed error ``prediction - actual`` (positive means over-forecast)."""
    actual, predicted = _arrays(y_true, y_pred)
    return float(np.mean(predicted - actual))


def smape(y_true, y_pred, *, as_percent: bool = True) -> float:
    r"""Symmetric mean absolute percentage error, zero-safe.

    .. math::

        \mathrm{sMAPE} = \frac{100}{n} \sum_t
        \frac{2\lvert y_t-\hat y_t\rvert}{\lvert y_t\rvert+\lvert\hat y_t\rvert}

    Terms where both the forecast and the actual are zero contribute 0 rather
    than dividing by zero, so a series with genuine zeros does not produce
    ``inf`` or ``NaN``. Bounded by 100 % per term, unlike MAPE.
    """
    actual, predicted = _arrays(y_true, y_pred)
    denominator = np.abs(actual) + np.abs(predicted)
    numerator = 2.0 * np.abs(predicted - actual)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(denominator == 0.0, 0.0, numerator / np.where(denominator == 0.0, 1.0, denominator))
    value = float(np.mean(terms))
    return 100.0 * value if as_percent else value


def mase_denominator(y_insample, seasonality: int = 1) -> float:
    """Mean absolute ``seasonality``-step difference of an in-sample series.

    This is the MASE scale. Raises :class:`ValueError` when the history is too
    short to estimate it, and returns ``nan`` for a degenerate (zero-difference)
    series — a constant series has no scale, so a scaled error is undefined.
    """
    values = np.asarray(y_insample, dtype="float64").ravel()
    seasonality = int(seasonality)
    if seasonality < 1:
        raise ValueError("seasonality must be >= 1")
    if values.size <= seasonality:
        raise ValueError(
            "MASE needs more than %d in-sample observation(s); got %d"
            % (seasonality, values.size)
        )
    diffs = np.abs(values[seasonality:] - values[:-seasonality])
    if diffs.size == 0:
        raise ValueError("no in-sample differences to scale MASE by")
    # A gap in the history makes some differences undefined; the scale is then
    # estimated from the observed pairs. Ignoring them would silently return nan
    # and disable MASE for the whole series, which is the usual situation for
    # real telemetry with missing days.
    observed = diffs[np.isfinite(diffs)]
    if observed.size == 0:
        raise ValueError("no finite in-sample differences to scale MASE by")
    scale = float(np.mean(observed))
    if scale == 0.0:
        return float("nan")
    return scale


def mase(y_true, y_pred, y_insample, seasonality: int = 1) -> float:
    r"""Mean absolute scaled error against an in-sample scale.

    .. math::

        \mathrm{MASE} = \frac{\frac1n\sum_t \lvert y_t-\hat y_t\rvert}
        {\frac1{T-m}\sum_{s} \lvert y_s - y_{s-m}\rvert}

    with :math:`m` = ``seasonality``. ``y_insample`` must be the training
    history that precedes the forecast (in time order). A seasonal-naive
    forecast scored against its own history gives exactly 1.0: :math:`1` is
    parity with the seasonal-naive scale, above it is worse.
    """
    scale = mase_denominator(y_insample, seasonality)
    actual, predicted = _arrays(y_true, y_pred)
    if not np.isfinite(scale):
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)) / scale)


def rmse_scaled(y_true, y_pred, y_insample, seasonality: int = 1) -> float:
    """RMSE divided by the in-sample mean absolute ``seasonality``-step change."""
    scale = mase_denominator(y_insample, seasonality)
    if not np.isfinite(scale):
        return float("nan")
    return float(rmse(y_true, y_pred) / scale)


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Mean pinball (quantile) loss at ``quantile`` in the open interval (0, 1)."""
    quantile = float(quantile)
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")
    actual, predicted = _arrays(y_true, y_pred)
    residual = actual - predicted
    return float(
        np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual))
    )


def winkler_score(y_true, lower, upper, alpha: float = 0.1) -> float:
    r"""Mean interval (Winkler) score for central intervals at level ``1 - alpha``.

    .. math::

        W = \frac1n\sum_t \Big[(u_t-l_t)
        + \frac{2}{\alpha}(l_t-y_t)\mathbf 1\{y_t<l_t\}
        + \frac{2}{\alpha}(y_t-u_t)\mathbf 1\{y_t>u_t\}\Big]

    Smaller is better: the width plus twice the penalty per :math:`\\alpha`
    for each observation outside the interval. A perfect, narrow interval
    scores near zero.
    """
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    actual, low = _arrays(y_true, lower)
    _, high = _arrays(y_true, upper)
    below = np.maximum(low - actual, 0.0)
    above = np.maximum(actual - high, 0.0)
    scores = (high - low) + (2.0 / alpha) * (below + above)
    return float(np.mean(scores))


def interval_coverage(y_true, lower, upper) -> float:
    """Fraction of observations inside ``[lower, upper]`` (closed interval)."""
    actual, low = _arrays(y_true, lower)
    _, high = _arrays(y_true, upper)
    return float(np.mean((actual >= low) & (actual <= high)))


def interval_width(lower, upper) -> float:
    """Mean interval width ``upper - lower``."""
    low = np.asarray(lower, dtype="float64").ravel()
    high = np.asarray(upper, dtype="float64").ravel()
    if low.shape != high.shape:
        raise ValueError("lower and upper must have the same shape")
    if low.size == 0:
        raise ValueError("at least one interval is required")
    return float(np.mean(high - low))


def _grouped(values, labels) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for value, label in zip(values, labels):
        grouped.setdefault(str(label), []).append(float(value))
    return grouped


def summarize(
    y_true,
    y_pred,
    *,
    y_insample_scale: Any = None,
    seasonality: int = 1,
    quantile: float | None = None,
    lower: Any = None,
    upper: Any = None,
    alpha: float = 0.1,
    series: Any = None,
    horizon: Any = None,
    aggregation: str = "pooled",
) -> dict[str, Any]:
    """Compute the standard forecast metric block for one set of predictions.

    ``aggregation`` selects what ``primary`` reports:

    * ``"pooled"`` (default) — every prediction weighted equally. Honest for a
      single series; misleading across series of different magnitude, which is
      why the per-series block is always returned alongside it.
    * ``"macro"`` — mean of the per-series metrics (each series counts once,
      regardless of length). Requires ``series``.

    ``y_insample_scale`` carries the per-origin MASE denominators when the caller
    has them (a walk-forward backtest does); without it MASE is omitted rather
    than silently computed against the wrong scale.
    """
    actual, predicted = _arrays(y_true, y_pred)
    metrics: dict[str, Any] = {
        "rows": int(actual.size),
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "bias": bias(actual, predicted),
        "smape": smape(actual, predicted),
    }
    errors = np.abs(actual - predicted)
    if y_insample_scale is not None:
        scales = np.asarray(y_insample_scale, dtype="float64").ravel()
        if scales.shape != actual.shape:
            raise ValueError(
                "y_insample_scale must have one value per prediction (%s vs %s)"
                % (scales.shape, actual.shape)
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            per_row = np.where(scales > 0, errors / np.where(scales > 0, scales, 1.0), np.nan)
        metrics["mase"] = (
            float(np.nanmean(per_row)) if np.isfinite(per_row).any() else float("nan")
        )
        metrics["seasonality"] = int(seasonality)
    if quantile is not None:
        metrics["pinball_loss"] = pinball_loss(actual, predicted, quantile)
        metrics["quantile"] = float(quantile)
    if lower is not None and upper is not None:
        metrics["coverage"] = interval_coverage(actual, lower, upper)
        metrics["interval_width"] = interval_width(lower, upper)
        metrics["winkler_score"] = winkler_score(actual, lower, upper, alpha)
        metrics["alpha"] = float(alpha)

    result: dict[str, Any] = {"aggregation": aggregation, "metrics": metrics}
    if series is not None:
        labels = list(series)
        per_series = {}
        for label, indices in _index_groups(labels).items():
            block = {
                "rows": len(indices),
                "mae": mae(actual[indices], predicted[indices]),
                "rmse": rmse(actual[indices], predicted[indices]),
                "bias": bias(actual[indices], predicted[indices]),
                "smape": smape(actual[indices], predicted[indices]),
            }
            if y_insample_scale is not None:
                scales = np.asarray(y_insample_scale, dtype="float64").ravel()[indices]
                errors_block = np.abs(actual[indices] - predicted[indices])
                with np.errstate(divide="ignore", invalid="ignore"):
                    per_row = np.where(
                        scales > 0,
                        errors_block / np.where(scales > 0, scales, 1.0),
                        np.nan,
                    )
                block["mase"] = (
                    float(np.nanmean(per_row)) if np.isfinite(per_row).any() else float("nan")
                )
            per_series[label] = block
        result["per_series"] = per_series
        if aggregation == "macro" and per_series:
            macro = {}
            for key in ("mae", "rmse", "bias", "smape", "mase"):
                values = [block[key] for block in per_series.values() if key in block]
                if len(values) == len(per_series):
                    macro[key] = float(np.nanmean(values))
            macro["rows"] = int(actual.size)
            result["primary"] = macro
    if horizon is not None:
        labels = list(horizon)
        result["per_horizon"] = {
            label: {
                "rows": len(indices),
                "mae": mae(actual[indices], predicted[indices]),
                "rmse": rmse(actual[indices], predicted[indices]),
                "bias": bias(actual[indices], predicted[indices]),
            }
            for label, indices in _index_groups(labels).items()
        }
    result.setdefault("primary", metrics)
    return result


def _index_groups(labels: Sequence[Any]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(str(label), []).append(index)
    return {label: np.asarray(indices, dtype="int64") for label, indices in grouped.items()}


def baseline_prediction(history, method: str, horizon: int, seasonality: int = 1) -> float:
    """One baseline forecast of ``history[-1]``'s series, ``horizon`` steps ahead.

    * ``naive`` — the last observation;
    * ``seasonal_naive`` — the observation ``seasonality`` steps back;
    * ``drift`` — last value plus the average change per step, times ``horizon``;
    * ``mean`` — the arithmetic mean of the history;
    * ``historical_mean_by_series`` — the same mean, named explicitly: every
      call is conditioned on one series' history, so there is no pooled
      cross-series mean to distinguish (plan phase P4.2 vocabulary).

    :param history: ordered observations available *at the forecast origin*.
        Missing values are allowed: every method is defined from the values that
        were actually observed (``naive`` is the most recent observed value).
    """
    values = np.asarray(history, dtype="float64").ravel()
    if values.size == 0:
        raise ValueError("baseline_prediction needs a non-empty history")
    # History with gaps: a baseline is defined from the observations that exist.
    # ``naive`` means "the most recent *observed* value", not "the most recent
    # slot", and the mean family averages observed values only.
    observed = values[np.isfinite(values)]
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    method = str(method)
    if method == "naive":
        if observed.size == 0:
            raise ValueError("naive needs at least one observed value")
        return float(observed[-1])
    if method == "seasonal_naive":
        seasonality = int(seasonality)
        if seasonality < 1:
            raise ValueError("seasonality must be >= 1")
        if values.size < seasonality:
            raise ValueError(
                "seasonal_naive needs at least %d observation(s); got %d"
                % (seasonality, values.size)
            )
        # Seasonal naive at horizon h repeats the value from the same phase one
        # season earlier: y[T + h - m*ceil(h/m)]. For h == m that is the last
        # observation, which is what makes it the canonical seasonal baseline.
        steps_back = seasonality - ((horizon - 1) % seasonality)
        if steps_back > observed.size:
            raise ValueError(
                "seasonal_naive needs %d observed value(s); got %d"
                % (steps_back, observed.size)
            )
        return float(observed[-steps_back])
    if method in ("mean", "historical_mean_by_series"):
        if observed.size == 0:
            raise ValueError("the mean baseline needs at least one observed value")
        return float(np.mean(observed))
    if method == "drift":
        if observed.size < 2:
            if observed.size == 0:
                raise ValueError("the drift baseline needs observed values")
            return float(observed[-1])
        slope = (observed[-1] - observed[0]) / float(observed.size - 1)
        return float(observed[-1] + slope * horizon)
    raise ValueError(
        "unknown baseline method %r; expected one of %s"
        % (method, ", ".join(BASELINE_METHODS))
    )


def forecast_ranking(
    entries: Iterable[Mapping[str, Any]],
    *,
    metric: str = "mase",
    baseline: str = "seasonal_naive",
) -> dict[str, Any]:
    """Rank candidate forecasts and flag the ones that lose to a baseline.

    ``entries`` are mappings with a ``label`` and a ``metrics`` mapping. Entries
    whose ``metric`` is missing or non-finite are listed as unranked rather than
    silently placed last. A candidate with ``metric >= 1`` (``metric`` defaulting
    to MASE) is listed with ``worse_than_baseline: True`` — the caller should not
    report it as the winner, whatever the ordering says.
    """
    ranked = []
    unranked = []
    for entry in entries:
        label = str(entry.get("label"))
        metrics = entry.get("metrics") or {}
        value = metrics.get(metric)
        if value is None or not np.isfinite(float(value)):
            unranked.append({"label": label, "reason": "metric %r is missing or non-finite" % metric})
            continue
        value = float(value)
        # MASE == 1 is exact parity with the seasonal-naive benchmark (that is what
        # the scaling guarantees), so "worse" is strictly greater than 1.
        ranked.append(
            {
                "label": label,
                "value": value,
                "beats_baseline": bool(value < 1.0) if metric == "mase" else None,
                "worse_than_baseline": bool(value > 1.0) if metric == "mase" else None,
                "baseline": baseline if metric == "mase" else None,
            }
        )
    ranked.sort(key=lambda item: item["value"])
    best = ranked[0] if ranked else None
    return {
        "metric": metric,
        "best": best,
        "ranked": ranked,
        "unranked": unranked,
        "winner_beats_baseline": (
            None if best is None or best["beats_baseline"] is None
            else bool(best["beats_baseline"])
        ),
    }
