"""Model explanation and prediction-interval endpoints for one stored bundle.

- ``/{version}/explain``   → permutation importance plus partial dependence for
                             the most important features (bounded, synchronous),
                             with optional descriptive reference-subset
                             stability summaries for both
- ``/{version}/ale``       → accumulated local effects (ALE) for selected
                             numeric features (bounded, synchronous)
- ``/{version}/intervals`` → split-conformal prediction intervals (regression)

Both work on the task's stored dataset partitions and never refit the stored
bundle: the interval endpoint refits a *clone* on a train slice.
"""

from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from SKSurrogate.mltrace import (
    prediction_interval_diagnostics,
    split_conformal_regression_interval,
)

from ..deps import bad_request, finite_or_none, unprocessable, task_access
from ..explanations import (
    DEFAULT_MIN_BIN_FRACTION,
    accumulated_local_effects,
    average_pdp_on_grid,
    descriptive_summary,
    distinct_permutations,
    numeric_bin_edges,
    reference_subsets,
    response_callable,
    row_contribution_summary,
)
from ..training import get_registered_target, load_partition, problem_family
from .evaluation import _load_bundle

router = APIRouter(prefix="/api/evaluation", tags=["explanations"], dependencies=[Depends(task_access)])

_MAX_ROWS = 1000
_PREVIEW_POINTS = 200


class ExplainRequest(BaseModel):
    partition: str = "train"
    max_rows: int = Field(default=500, ge=20, le=_MAX_ROWS)
    n_repeats: int = Field(default=5, ge=1, le=20)
    top_k: int = Field(default=3, ge=0, le=6)
    stability_samples: int = Field(default=1, ge=1, le=5)
    random_state: int = 0


class AleRequest(BaseModel):
    partition: str = "train"
    max_rows: int = Field(default=500, ge=20, le=_MAX_ROWS)
    n_bins: int = Field(default=20, ge=5, le=50)
    top_k: int = Field(default=3, ge=0, le=6)
    features: list[str] | None = None
    stability_samples: int = Field(default=1, ge=1, le=5)
    random_state: int = 0


class RowExplainRequest(BaseModel):
    partition: str = "train"
    row_index: int = Field(default=0, ge=0)
    features: list[str] | None = None
    baseline: Literal["reference_mean", "row"] = "reference_mean"
    baseline_partition: str | None = None
    baseline_row_index: int | None = Field(default=None, ge=0)
    orders: int = Field(default=1, ge=1, le=10)
    reference_max_rows: int = Field(default=500, ge=20, le=_MAX_ROWS)
    random_state: int = 0


class IntervalRequest(BaseModel):
    train_partition: str = "train"
    holdout_partition: str | None = None  # defaults to a slice of the train partition
    alpha: float = Field(default=0.1, gt=0.0, lt=1.0)
    diagnostic_confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    group_by: str | None = None
    max_groups: int = Field(default=20, ge=1, le=50)
    random_state: int = 0


def _features_and_target(task_name, partition) -> tuple[pd.DataFrame, np.ndarray]:
    target, _ = get_registered_target(task_name)
    frame = load_partition(task_name, partition)
    columns = [column for column in frame.columns if column != target]
    return frame[columns], frame[target].to_numpy()


def _json_scalar(value):
    """JSON-safe rendering of one stored feature value (NaN/inf become null)."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return finite_or_none(float(value))
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _aligned_features(frame: pd.DataFrame, names, context) -> pd.DataFrame:
    """Return ``frame[names]`` or a 400 naming the features the partition lacks."""
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise bad_request(
            "%s partition is missing feature(s) %s"
            % (context, ", ".join(repr(name) for name in missing))
        )
    return frame[names]


def _baseline_vector(frame: pd.DataFrame, names) -> tuple[np.ndarray, dict[str, str]]:
    """Reference baseline row: finite mean for numeric columns, mode otherwise."""
    values, sources = [], {}
    for name in names:
        column = frame[name]
        if pd.api.types.is_numeric_dtype(column):
            numeric = column.to_numpy(dtype=float)
            finite = numeric[np.isfinite(numeric)]
            if finite.size == 0:
                raise bad_request(
                    "Feature %r has no finite reference values to build a mean baseline" % name
                )
            values.append(float(finite.mean()))
            sources[name] = "finite_mean"
        else:
            modes = column.astype(str).mode()
            if modes.empty:
                raise bad_request(
                    "Feature %r has no reference values to build a mode baseline" % name
                )
            values.append(modes.iloc[0])
            sources[name] = "most_frequent"
    return np.asarray(values, dtype=object), sources



@router.post("/{task_name}/{model_version}/explain")
def explain_bundle(task_name: str, model_version: str, body: ExplainRequest):
    """Permutation importance (score drop when a feature is shuffled) and PDP curves."""
    from sklearn.inspection import partial_dependence, permutation_importance

    bundle = _load_bundle(task_name, model_version)
    if problem_family(bundle.model) is None:
        raise unprocessable("Explanations need a scikit-learn estimator")
    X, y = _features_and_target(task_name, body.partition)
    source_rows = len(X)
    if body.stability_samples > 1 and source_rows <= body.max_rows:
        raise bad_request(
            "Reference-sample stability needs more source rows than max_rows; "
            "use a smaller max_rows or a partition with more rows."
        )
    X_source, y_source = X, y
    sampled = source_rows > body.max_rows
    sample_rng = np.random.RandomState(body.random_state)
    keep = None
    if sampled:
        keep = sample_rng.choice(len(X), body.max_rows, replace=False)
        X, y = X.iloc[keep], y[keep]
    names = list(X.columns)
    values = X.to_numpy()
    try:
        result = permutation_importance(
            bundle.model, values, y, n_repeats=body.n_repeats, random_state=body.random_state
        )
    except Exception as exc:
        raise bad_request("Could not compute importance: %s: %s" % (type(exc).__name__, exc))
    stability_results = [result]
    extra_subsets = []
    if body.stability_samples > 1:
        extra_subsets = reference_subsets(
            sample_rng,
            source_rows,
            body.max_rows,
            body.stability_samples - 1,
            excluded={tuple(sorted(int(index) for index in keep))} if keep is not None else set(),
        )
    for sample_index, subset in enumerate(extra_subsets):
        try:
            stability_results.append(
                permutation_importance(
                    bundle.model,
                    X_source.iloc[subset].to_numpy(),
                    y_source[subset],
                    n_repeats=body.n_repeats,
                    random_state=body.random_state,
                )
            )
        except Exception as exc:
            raise bad_request(
                "Could not compute importance for reference sample %d: %s: %s"
                % (sample_index + 2, type(exc).__name__, exc)
            )
    reference_sample_stability = None
    if body.stability_samples > 1:
        sample_means = np.vstack([item.importances_mean for item in stability_results])
        top_counts = np.zeros(len(names), dtype=float)
        for sample_mean in sample_means:
            winners = np.flatnonzero(sample_mean == np.max(sample_mean))
            top_counts[winners] += 1.0 / len(winners)
        reference_sample_stability = {
            "sample_count": len(stability_results),
            "sample_size": body.max_rows,
            "permutation_repeats_per_sample": body.n_repeats,
            "sampling": "random_without_replacement",
            "summary": [
                {
                    "feature": names[index],
                    "mean_importance": finite_or_none(float(sample_means[:, index].mean())),
                    "sample_std": finite_or_none(float(sample_means[:, index].std(ddof=1))),
                    "top_rank_frequency": float(top_counts[index] / len(stability_results)),
                }
                for index in np.argsort(-sample_means.mean(axis=0))
            ],
        }
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
    partial_dependence_stability = None
    if body.stability_samples > 1 and curves:
        try:
            response, response_meta = response_callable(bundle.model, problem_family(bundle.model))
        except ValueError as exc:
            partial_dependence_stability = {"skipped_reason": str(exc)}
        else:
            subsets = [keep if sampled else np.arange(source_rows)]
            subsets = [np.asarray(subset) for subset in subsets] + list(extra_subsets)
            base_values = X_source.to_numpy()
            stability_curves = []
            for curve in curves:
                if any(value is None for value in curve["grid"]):
                    stability_curves.append(
                        {"feature": curve["feature"], "skipped_reason": "non_finite_primary_grid"}
                    )
                    continue
                index = names.index(curve["feature"])
                sample_curves = np.vstack(
                    [
                        average_pdp_on_grid(response, base_values[subset], index, curve["grid"])
                        for subset in subsets
                    ]
                )
                summary = descriptive_summary(sample_curves)
                stability_curves.append(
                    {
                        "feature": curve["feature"],
                        "grid": curve["grid"],
                        "mean": [finite_or_none(v) for v in summary["mean"]],
                        "sample_std": [finite_or_none(v) for v in summary["sample_std"]],
                        "min": [finite_or_none(v) for v in summary["min"]],
                        "max": [finite_or_none(v) for v in summary["max"]],
                        "sample_means": [finite_or_none(float(row.mean())) for row in sample_curves],
                    }
                )
            partial_dependence_stability = {
                "sample_count": len(subsets),
                "sample_size": int(len(X)),
                "sampling": "random_without_replacement",
                "grid_source": "primary_curve_grid",
                "response": response_meta,
                "curves": stability_curves,
                "note": (
                    "Each curve is the average partial dependence recomputed with direct model "
                    "calls on one distinct reference subset, evaluated on the primary curve's grid. "
                    "The spread across subsets is descriptive; it is not a confidence interval."
                ),
            }
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
            "stability_samples": body.stability_samples,
            "random_state": body.random_state,
        },
        "importance": importance,
        "reference_sample_stability": reference_sample_stability,
        "partial_dependence": curves,
        "partial_dependence_stability": partial_dependence_stability,
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
                "Reference-sample stability is not estimated by default; when requested, its descriptive sample variation is not a confidence interval or a guarantee of generalization.",
                "Partial-dependence stability curves re-evaluate the same grid on distinct reference subsets; their spread describes reference-sample sensitivity only and is not a confidence band.",
            ],
        },
    }


@router.post("/{task_name}/{model_version}/ale")
def accumulated_local_effects_bundle(task_name: str, model_version: str, body: AleRequest):
    """Accumulated local effects for selected numeric features of one stored bundle.

    See ``docs/mltrace.rst`` (Accumulated local effects) for the contract: numeric
    features only, equal-frequency bins with sparse-bin merging, one scalar model
    response, and descriptive-only reference-subset stability.
    """
    from sklearn.inspection import permutation_importance

    bundle = _load_bundle(task_name, model_version)
    family = problem_family(bundle.model)
    if family is None:
        raise unprocessable("Explanations need a scikit-learn estimator")
    try:
        response, response_meta = response_callable(bundle.model, family)
    except ValueError as exc:
        raise unprocessable(str(exc))

    X, y = _features_and_target(task_name, body.partition)
    source_rows = len(X)
    if body.stability_samples > 1 and source_rows <= body.max_rows:
        raise bad_request(
            "Reference-sample stability needs more source rows than max_rows; "
            "use a smaller max_rows or a partition with more rows."
        )
    names = list(X.columns)
    if body.features is not None:
        unknown = [feature for feature in body.features if feature not in names]
        if unknown:
            raise bad_request(
                "Unknown feature(s) %s; available features are %s"
                % (", ".join(repr(name) for name in unknown), ", ".join(repr(name) for name in names))
            )
        if len(body.features) > 6:
            raise bad_request("At most six features per ALE request")
        selected = list(body.features)
    elif body.top_k == 0:
        selected = []
    else:
        sampled_for_ranking = source_rows > body.max_rows
        if sampled_for_ranking:
            ranking_keep = np.random.RandomState(body.random_state).choice(
                source_rows, body.max_rows, replace=False
            )
            ranking_values, ranking_y = X.iloc[ranking_keep].to_numpy(), y[ranking_keep]
        else:
            ranking_values, ranking_y = X.to_numpy(), y
        try:
            ranking = permutation_importance(
                bundle.model, ranking_values, ranking_y, n_repeats=3, random_state=body.random_state
            )
        except Exception as exc:
            raise bad_request(
                "Could not rank features for ALE selection: %s: %s; pass explicit "
                "`features` instead" % (type(exc).__name__, exc)
            )
        importances = np.asarray(ranking.importances_mean)
        selected = [names[i] for i in np.argsort(-importances)[: body.top_k]]

    values_source = X.to_numpy()
    sampled = source_rows > body.max_rows
    sample_rng = np.random.RandomState(body.random_state)
    keep = None
    if sampled:
        keep = sample_rng.choice(source_rows, body.max_rows, replace=False)
        values = values_source[keep]
    else:
        values = values_source
    subsets = []
    if body.stability_samples > 1:
        subsets = reference_subsets(
            sample_rng,
            source_rows,
            body.max_rows,
            body.stability_samples - 1,
            excluded={tuple(sorted(int(index) for index in keep))} if keep is not None else set(),
        )

    curves, skipped, errors = [], [], []
    for feature in selected:
        index = names.index(feature)
        if not pd.api.types.is_numeric_dtype(X[feature]):
            skipped.append(
                {
                    "feature": feature,
                    "reason": "nominal_categorical_ale_undefined",
                    "detail": (
                        "Accumulated local effects need an ordering of the feature's values; "
                        "a nominal categorical feature has none, so no ALE curve is drawn. "
                        "Partial dependence is still available for this feature."
                    ),
                }
            )
            continue
        try:
            edges, _ = numeric_bin_edges(values[:, index].astype(float), n_bins=body.n_bins)
        except ValueError as exc:
            skipped.append({"feature": feature, "reason": "undefined_bins", "detail": str(exc)})
            continue
        try:
            curve, bin_counts, excluded_rows = accumulated_local_effects(response, values, index, edges)
            entry = {
                "feature": feature,
                "kind": "numeric",
                "grid": [finite_or_none(value) for value in edges],
                "accumulated": [finite_or_none(value) for value in curve],
                "counts": [int(count) for count in bin_counts],
                "excluded_rows": int(excluded_rows),
                "min_bin_fraction": DEFAULT_MIN_BIN_FRACTION,
            }
            if subsets:
                subset_curves = [
                    accumulated_local_effects(response, values_source[subset], index, edges)[0]
                    for subset in subsets
                ]
                summary = descriptive_summary(np.vstack([curve] + subset_curves))
                entry["stability"] = {
                    "sample_count": summary["sample_count"],
                    "sample_size": int(len(values)),
                    "sampling": "random_without_replacement",
                    "mean": [finite_or_none(value) for value in summary["mean"]],
                    "sample_std": [finite_or_none(value) for value in summary["sample_std"]],
                    "min": [finite_or_none(value) for value in summary["min"]],
                    "max": [finite_or_none(value) for value in summary["max"]],
                    "note": (
                        "Each subset is accumulated on the primary reference sample's bin edges "
                        "and centered with its own bin weights, so the spread includes centering "
                        "variation. Descriptive only; not a confidence band."
                    ),
                }
        except Exception as exc:
            errors.append({"feature": feature, "detail": "%s: %s" % (type(exc).__name__, exc)})
            continue
        curves.append(entry)

    return {
        "model_version": model_version,
        "partition": body.partition,
        "response": response_meta,
        "reference_sample": {
            "partition": body.partition,
            "source_rows": int(source_rows),
            "used_rows": int(len(values)),
            "sampling": "random_without_replacement" if sampled else "all_rows",
        },
        "feature_selection": "explicit" if body.features is not None else "permutation_importance_top_k",
        "settings": {
            "max_rows": body.max_rows,
            "n_bins": body.n_bins,
            "top_k": body.top_k,
            "features": list(body.features) if body.features is not None else None,
            "stability_samples": body.stability_samples,
            "random_state": body.random_state,
            "min_bin_fraction": DEFAULT_MIN_BIN_FRACTION,
        },
        "curves": curves,
        "skipped": skipped,
        "errors": errors,
        "interpretation": {
            "accumulated_local_effects": {
                "category": "fitted_model_response_profile",
                "method": "accumulated_local_effects",
                "reference_partition": body.partition,
                "response_output": response_meta["output"],
                "causal_effect": False,
            },
            "limitations": [
                "Accumulated local effects accumulate local model-response differences over observed feature bins; they reduce reliance on off-support predictions relative to partial dependence but still describe a fitted model, not a causal effect.",
                "Correlated features can still make the local differences in one feature reflect another feature's distribution; ALE does not isolate variable importance and does not solve sparse support or conditional-explanation problems.",
                "Nominal categorical features are skipped because ALE requires a feature ordering; the response lists them under `skipped`.",
                "Binary classifier curves are drawn on the labelled scalar response (positive-class probability or decision value); the scale is not comparable across different response outputs or models.",
                "Reference-subset summaries describe sample-to-sample variation only; they are not confidence bands and do not guarantee generalization.",
            ],
        },
    }


@router.post("/{task_name}/{model_version}/explain/row")
def explain_row(task_name: str, model_version: str, body: RowExplainRequest):
    """Ordered-occlusion attribution for one stored row of one bundle.

    Implements the Stage D3 contract in ``docs/mltrace.rst``: each feature is
    credited with the model-response change observed when it is restored from a
    baseline row in a specified order, so the contributions telescope exactly to
    ``f(explained_row) - f(baseline)`` for any estimator. Multiple seeded orders
    describe order-dependence; they are not confidence intervals, and this is a
    model attribution rather than a causal effect.
    """
    bundle = _load_bundle(task_name, model_version)
    family = problem_family(bundle.model)
    if family is None:
        raise unprocessable("Explanations need a scikit-learn estimator")
    try:
        response, response_meta = response_callable(bundle.model, family)
    except ValueError as exc:
        raise unprocessable(str(exc))

    X, _ = _features_and_target(task_name, body.partition)
    if body.row_index >= len(X):
        raise bad_request(
            "row_index %d is outside the %d stored rows of partition %r"
            % (body.row_index, len(X), body.partition)
        )
    names = list(X.columns)
    selected = names if body.features is None else list(body.features)
    unknown = [feature for feature in selected if feature not in names]
    if unknown:
        raise bad_request(
            "Unknown feature(s) %s; available features are %s"
            % (
                ", ".join(repr(name) for name in unknown),
                ", ".join(repr(name) for name in names),
            )
        )
    if not selected:
        raise bad_request("At least one feature is required for a row explanation")
    if len(selected) > 24:
        raise bad_request(
            "At most 24 features per row explanation; pass `features` to select a subset"
        )

    row_values = X.iloc[body.row_index].to_numpy()
    baseline_partition = body.baseline_partition or body.partition
    base_frame, _ = _features_and_target(task_name, baseline_partition)
    base_frame = _aligned_features(base_frame, names, "baseline")
    source_rows = len(base_frame)
    if body.baseline == "row":
        if body.baseline_row_index is None:
            raise bad_request("baseline='row' requires baseline_row_index")
        if body.baseline_row_index >= source_rows:
            raise bad_request(
                "baseline_row_index %d is outside the %d stored rows of baseline partition %r"
                % (body.baseline_row_index, source_rows, baseline_partition)
            )
        baseline_values = np.asarray(base_frame.iloc[body.baseline_row_index].to_numpy())
        baseline_sources = {name: "row" for name in names}
        used_rows = 1
        sampling = "explicit_row"
    else:
        rng = np.random.RandomState(body.random_state)
        sampled = source_rows > body.reference_max_rows
        if sampled:
            keep = rng.choice(source_rows, body.reference_max_rows, replace=False)
            base_frame = base_frame.iloc[keep]
        baseline_values, baseline_sources = _baseline_vector(base_frame, names)
        used_rows = len(base_frame)
        sampling = "random_without_replacement" if sampled else "all_rows"

    positions = [names.index(feature) for feature in selected]
    hybrid = np.asarray(baseline_values).copy()
    for position in positions:
        hybrid[position] = row_values[position]
    baseline_prediction = float(np.ravel(response(np.asarray(baseline_values)[None, :]))[0])
    row_prediction = float(np.ravel(response(np.asarray(row_values)[None, :]))[0])
    hybrid_prediction = float(np.ravel(response(hybrid[None, :]))[0])

    try:
        local_orders = distinct_permutations(
            np.random.RandomState(body.random_state), len(selected), body.orders
        )
    except ValueError as exc:
        raise bad_request(str(exc))
    column_orders = [
        np.asarray([positions[index] for index in order], dtype=int) for order in local_orders
    ]
    summary = row_contribution_summary(
        response, row_values, baseline_values, column_orders, len(names)
    )
    stats = summary["contributions"]
    target_change = hybrid_prediction - baseline_prediction
    max_abs_residual = float(np.max(np.abs(summary["order_totals"] - target_change)))
    tolerance = 1e-6 * max(1.0, abs(target_change), abs(hybrid_prediction))

    contributions = [
        {
            "feature": feature,
            "mean": finite_or_none(float(stats["mean"][position])),
            "sample_std": finite_or_none(float(stats["sample_std"][position])),
            "min": finite_or_none(float(stats["min"][position])),
            "max": finite_or_none(float(stats["max"][position])),
        }
        for feature, position in zip(selected, positions)
    ]
    contributions.sort(key=lambda item: -abs(item["mean"] or 0.0))

    return {
        "model_version": model_version,
        "partition": body.partition,
        "row_index": body.row_index,
        "row_count": int(len(X)),
        "response": response_meta,
        "reference_sample": {
            "partition": baseline_partition,
            "source_rows": int(source_rows),
            "used_rows": int(used_rows),
            "sampling": sampling,
        },
        "baseline": {
            "mode": body.baseline,
            "partition": baseline_partition,
            "row_index": body.baseline_row_index if body.baseline == "row" else None,
            "prediction": finite_or_none(baseline_prediction),
            "values": {
                name: _json_scalar(baseline_values[names.index(name)]) for name in names
            },
            "value_sources": baseline_sources,
        },
        "prediction": finite_or_none(row_prediction),
        "explained_prediction": finite_or_none(hybrid_prediction),
        "explanation_scope": "full_row" if len(selected) == len(names) else "subset_hybrid",
        "explained_features": selected,
        "settings": {
            "orders": body.orders,
            "reference_max_rows": body.reference_max_rows,
            "random_state": body.random_state,
        },
        "orders": {
            "count": len(column_orders),
            "distinct": True,
            "note": (
                "Each order yields an exact telescoping decomposition; the per-feature "
                "sample SD across orders is disclosed order-dependence, not a confidence interval."
            ),
        },
        "contributions": contributions,
        "reconstruction": {
            "baseline_prediction": finite_or_none(baseline_prediction),
            "explained_prediction": finite_or_none(hybrid_prediction),
            "sum_contributions": finite_or_none(float(np.mean(summary["order_totals"]))),
            "target_change": finite_or_none(target_change),
            "max_abs_residual": finite_or_none(max_abs_residual),
            "tolerance": finite_or_none(tolerance),
            "exact_by_construction": True,
        },
        "interpretation": {
            "category": "model_attribution",
            "method": "ordered_occlusion_chain",
            "causal_effect": False,
            "baseline_mode": body.baseline,
            "limitations": [
                "Contributions describe how this fitted model's response changes when each feature is restored from the baseline row; they are not interventions or causal effects.",
                "Restoring features one at a time makes the attribution order-dependent, especially with correlated features or interactions; the per-feature spread across orders measures that dependence and is not a confidence interval.",
                "The baseline defines the reference point: a different baseline partition or row changes every contribution, even though the reconstruction stays exact.",
                "Binary classifier contributions are on the labelled scalar response (positive-class probability or decision value); they are not comparable across response outputs or models.",
                "With a `features` subset the reconstruction explains the hybrid row (listed features at their observed values, the rest at baseline), not the full prediction.",
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
