"""Model explanation and prediction-interval endpoints for one stored bundle.

- ``/{version}/explain``   → permutation importance plus partial dependence for
                             the most important features (bounded, synchronous)
- ``/{version}/intervals`` → split-conformal prediction intervals (regression)

Both work on the task's stored dataset partitions and never refit the stored
bundle: the interval endpoint refits a *clone* on a train slice.
"""

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from SKSurrogate.mltrace import (
    prediction_interval_diagnostics,
    split_conformal_regression_interval,
)

from ..deps import bad_request, finite_or_none, unprocessable
from ..training import get_registered_target, load_partition, problem_family
from .evaluation import _load_bundle

router = APIRouter(prefix="/api/evaluation", tags=["explanations"])

_MAX_ROWS = 1000
_PREVIEW_POINTS = 200


class ExplainRequest(BaseModel):
    partition: str = "train"
    max_rows: int = Field(default=500, ge=20, le=_MAX_ROWS)
    n_repeats: int = Field(default=5, ge=1, le=20)
    top_k: int = Field(default=3, ge=0, le=6)
    random_state: int = 0


class IntervalRequest(BaseModel):
    train_partition: str = "train"
    holdout_partition: str | None = None  # defaults to a slice of the train partition
    alpha: float = Field(default=0.1, gt=0.0, lt=1.0)
    diagnostic_confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    group_by: str | None = None
    max_groups: int = Field(default=20, ge=1, le=50)
    random_state: int = 0


def _features_and_target(task_name, partition):
    target, _ = get_registered_target(task_name)
    frame = load_partition(task_name, partition)
    columns = [column for column in frame.columns if column != target]
    return frame[columns], frame[target].to_numpy()


@router.post("/{task_name}/{model_version}/explain")
def explain_bundle(task_name: str, model_version: str, body: ExplainRequest):
    """Permutation importance (score drop when a feature is shuffled) and PDP curves."""
    from sklearn.inspection import partial_dependence, permutation_importance

    bundle = _load_bundle(task_name, model_version)
    if problem_family(bundle.model) is None:
        raise unprocessable("Explanations need a scikit-learn estimator")
    X, y = _features_and_target(task_name, body.partition)
    source_rows = len(X)
    sampled = source_rows > body.max_rows
    if sampled:
        keep = np.random.RandomState(body.random_state).choice(len(X), body.max_rows, replace=False)
        X, y = X.iloc[keep], y[keep]
    names = list(X.columns)
    values = X.to_numpy()
    try:
        result = permutation_importance(
            bundle.model, values, y, n_repeats=body.n_repeats, random_state=body.random_state
        )
    except Exception as exc:
        raise bad_request("Could not compute importance: %s: %s" % (type(exc).__name__, exc))
    order = np.argsort(-result.importances_mean)
    importance = [
        {
            "feature": names[i],
            "mean": finite_or_none(result.importances_mean[i]),
            "std": finite_or_none(result.importances_std[i]),
        }
        for i in order
    ]
    curves, errors = [], []
    for i in order[: body.top_k]:
        try:
            pd_result = partial_dependence(bundle.model, values, [int(i)], kind="average", grid_resolution=20)
            curves.append(
                {
                    "feature": names[i],
                    "grid": [finite_or_none(v) for v in pd_result["grid_values"][0]],
                    "average": [finite_or_none(v) for v in np.ravel(pd_result["average"][0])],
                }
            )
        except Exception as exc:
            errors.append({"feature": names[i], "detail": "%s: %s" % (type(exc).__name__, exc)})
    return {
        "model_version": model_version,
        "partition": body.partition,
        "rows": int(len(X)),
        "reference_sample": {
            "partition": body.partition,
            "source_rows": int(source_rows),
            "used_rows": int(len(X)),
            "sampling": "random_without_replacement" if sampled else "all_rows",
        },
        "settings": {
            "max_rows": body.max_rows,
            "n_repeats": body.n_repeats,
            "top_k": body.top_k,
            "random_state": body.random_state,
        },
        "importance": importance,
        "partial_dependence": curves,
        "errors": errors,
        "interpretation": {
            "importance": {
                "category": "model_attribution",
                "method": "permutation_importance",
                "score": "estimator.score",
                "causal_effect": False,
            },
            "partial_dependence": {
                "category": "fitted_model_response_profile",
                "reference_partition": body.partition,
                "causal_effect": False,
            },
            "limitations": [
                "Permutation importance measures score sensitivity to shuffled columns, not causal influence; correlated features can share or mask importance.",
                "Partial dependence averages model predictions over reference rows while varying selected features and can include implausible combinations when features are dependent.",
                "Results describe this fitted model on the selected partition and are not estimates of causal effects.",
                "The endpoint does not estimate stability across alternative reference samples.",
            ],
        },
    }


@router.post("/{task_name}/{model_version}/intervals")
def prediction_intervals(task_name: str, model_version: str, body: IntervalRequest):
    """Split-conformal intervals for a regression bundle, checked on held-out rows."""
    bundle = _load_bundle(task_name, model_version)
    family = problem_family(bundle.model)
    if family != "Regression":
        raise unprocessable("Prediction intervals need a regression bundle; %r is %s" % (model_version, family))
    X, y = _features_and_target(task_name, body.train_partition)
    rng = np.random.RandomState(body.random_state)
    order = rng.permutation(len(X))
    if body.holdout_partition:
        X_hold, y_hold = _features_and_target(task_name, body.holdout_partition)
        pool = order
    else:
        cut = max(1, int(0.2 * len(order)))
        pool, hold_index = order[cut:], order[:cut]
        X_hold, y_hold = X.iloc[hold_index], y[hold_index]
    if len(pool) < 10:
        raise bad_request("Need at least 10 training rows to split into fit and calibration sets")
    cut = int(0.7 * len(pool))
    fit_index, cal_index = pool[:cut], pool[cut:]
    group_values = None
    if body.group_by is not None:
        if body.group_by not in X_hold.columns:
            raise bad_request(
                "Subgroup feature %r is not present in the holdout partition"
                % body.group_by
            )
        group_values = X_hold[body.group_by].to_numpy()
    try:
        out = split_conformal_regression_interval(
            bundle.model,
            X.iloc[fit_index].to_numpy(),
            y[fit_index],
            X.iloc[cal_index].to_numpy(),
            y[cal_index],
            X_hold.to_numpy(),
            alpha=body.alpha,
        )
        diagnostics = prediction_interval_diagnostics(
            y_hold,
            out["lower"],
            out["upper"],
            alpha=body.alpha,
            confidence_level=body.diagnostic_confidence_level,
            groups=group_values,
            max_groups=body.max_groups,
        )
    except ValueError as exc:
        raise bad_request(str(exc))
    y_hold = np.asarray(y_hold, dtype=float)
    points = [
        {
            "actual": finite_or_none(y_hold[i]),
            "prediction": finite_or_none(out["predictions"][i]),
            "lower": finite_or_none(out["lower"][i]),
            "upper": finite_or_none(out["upper"][i]),
        }
        for i in range(min(len(y_hold), _PREVIEW_POINTS))
    ]
    return {
        "model_version": model_version,
        "method": out["method"],
        "coverage_assumption": out["coverage_assumption"],
        "alpha": out["alpha"],
        "nominal_coverage": out["coverage_level"],
        "half_width": finite_or_none(out["quantile"]),
        "calibration_size": out["calibration_size"],
        "training_partition": body.train_partition,
        "fit_rows": int(len(fit_index)),
        "calibration_source_partition": body.train_partition,
        "holdout_partition": (
            body.holdout_partition
            if body.holdout_partition
            else "random holdout slice of %s" % body.train_partition
        ),
        "random_state": body.random_state,
        "holdout_rows": int(len(y_hold)),
        "empirical_coverage": diagnostics["empirical_coverage"],
        "coverage_interval": diagnostics["coverage_interval"],
        "mean_interval_width": finite_or_none(diagnostics["mean_interval_width"]),
        "median_interval_width": finite_or_none(diagnostics["median_interval_width"]),
        "finite_interval_count": diagnostics["finite_interval_count"],
        "unbounded_interval_count": diagnostics["unbounded_interval_count"],
        "mean_interval_score": finite_or_none(diagnostics["mean_interval_score"]),
        "interval_score_method": diagnostics["interval_score_method"],
        "diagnostic_assumption": diagnostics["diagnostic_assumption"],
        "group_by": body.group_by,
        "subgroup_coverage": diagnostics["subgroup_coverage"],
        "points": points,
    }
