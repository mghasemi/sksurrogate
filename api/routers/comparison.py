"""Paired model comparison for stored bundles of one task.

One endpoint, ``POST /api/comparison/{task_name}``, compares two to eight
bundles on **the same folds of the same partition** and reports:

* per-model fold scores, mean/SD and rank;
* every model against a control: oriented mean delta, percentile bootstrap
  interval, Wilcoxon signed-rank p (Holm-adjusted across the family), Cliff's
  ordinal effect size and a ``distinguishable`` flag;
* a complexity/cost axis (fitted parameter count, bytes on disk, mean fit time,
  per-row predict latency) and the Pareto front over score/complexity/cost;
* rank stability across folds;
* calibration for classifiers (Brier score and expected calibration error on
  out-of-fold probabilities).

Why refit instead of reading ``bundle.metrics``: stored metrics are aggregates
of *different* folds per bundle, so they cannot be paired. Refitting every
bundle's estimator on identical split indices is what makes the delta a paired
statistic. The price is compute: the endpoint is synchronous and bounded by
``max_rows``/``n_splits``/``repeats``.

Provenance honesty: those estimators were selected by a hyper-parameter search
on this task's data, so the scores are an **exploratory** ranking, not an
unbiased estimate of generalization. That caveat is carried in
``score_provenance`` and ``warnings`` and is not optional.
"""

from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from SKSurrogate.compare import (
    brier_score,
    compare_scores,
    expected_calibration_error,
    fitted_parameter_count,
    pareto_frontier,
    rank_stability,
    simpler_within_tolerance,
)
from SKSurrogate.modelbundle import verify_bundle

from ..config import settings
from ..partitions import resolve_partition
from ..deps import (
    bad_request,
    finite_or_none,
    resolve_bundle,
    task_access,
    unprocessable,
)
from ..training import (
    build_custom_scorer,
    learning_curve_metric_label,
    learning_curve_metric_options,
    problem_family,
)
from .explain import _explain_frame, _explainable_estimator

router = APIRouter(prefix="/api/comparison", tags=["comparison"], dependencies=[Depends(task_access)])

#: Hard caps that keep one synchronous request bounded.
_MAX_MODELS = 8
_MAX_SPLITS = 10
_MAX_REPEATS = 5


class ComparisonRequest(BaseModel):
    """Which bundles to compare and on which folds."""

    model_versions: list[str] = Field(min_length=1, max_length=_MAX_MODELS)
    control: str | None = None
    scoring: str | None = None
    partition: str | None = None
    n_splits: int = Field(default=5, ge=2, le=_MAX_SPLITS)
    repeats: int = Field(default=1, ge=1, le=_MAX_REPEATS)
    random_state: int = 0
    #: Rows are subsampled once, before any model is fitted, so every model is
    #: scored on exactly the same rows. ``None`` uses the whole partition.
    max_rows: int | None = Field(default=2000, ge=100, le=50000)
    include_calibration: bool = True
    n_resamples: int = Field(default=2000, ge=200, le=20000)
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    #: Absolute score margin for the "simpler model within δ of the best" query.
    simpler_tolerance: float = Field(default=0.01, ge=0.0)


def _scorer(scoring, family):
    """Resolve a scorer object and its direction; 400 for an unusable metric."""
    from sklearn.metrics import get_scorer

    if scoring in (None, "", "default"):
        options = learning_curve_metric_options(family)
        scoring = options.get("default")
        if not scoring:
            raise unprocessable(
                "No default scoring metric is available for a %s task; pass `scoring` "
                "explicitly." % (family or "non-estimator")
            )
    if scoring in ("neg_smape",):
        scorer, higher_is_better = build_custom_scorer(scoring)
        return scoring, scorer, bool(higher_is_better)
    try:
        scorer = get_scorer(scoring)
    except ValueError as exc:
        raise bad_request(str(exc))
    return scoring, scorer, bool(getattr(scorer, "_sign", 1) > 0)


def _folds(frame, y, family, *, n_splits, repeats, random_state, temporal):
    """Materialise one list of (train, test) index pairs shared by every model."""
    from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, TimeSeriesSplit

    if temporal:
        # Order matters: a shuffled splitter would train on future rows. Repeats
        # are meaningless for a single time-ordered expansion, so one run is used.
        splitter = TimeSeriesSplit(n_splits=n_splits)
        return list(splitter.split(frame, y)), "TimeSeriesSplit", 1
    if family == "Classification":
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=repeats, random_state=random_state
        )
    else:
        splitter = RepeatedKFold(n_splits=n_splits, n_repeats=repeats, random_state=random_state)
    try:
        return list(splitter.split(frame, y)), type(splitter).__name__, repeats
    except ValueError as exc:
        raise bad_request(
            "Could not build %d fold(s) on %d row(s): %s. Use fewer folds or a larger "
            "partition." % (n_splits, len(y), exc)
        )


def _predict_latency_ms(estimator, values, *, random_state: int) -> float | None:
    """Median per-row prediction time on a bounded random sample."""
    if len(values) == 0:
        return None
    rng = np.random.RandomState(random_state)
    take = min(len(values), 200)
    rows = values if take == len(values) else values[rng.choice(len(values), take, replace=False)]
    try:
        started = time.perf_counter()
        estimator.predict(rows)
        elapsed = time.perf_counter() - started
    except Exception:  # a latency probe must never fail the comparison
        return None
    return float(1000.0 * elapsed / max(1, len(rows)))


def _calibration(estimator, frame, y, splits, *, n_bins=10):
    """Out-of-fold probabilities → Brier score and ECE for a classifier."""
    from sklearn.model_selection import cross_val_predict

    try:
        probabilities = cross_val_predict(
            estimator, frame, y, cv=splits, method="predict_proba", n_jobs=1
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": "%s: %s" % (type(exc).__name__, exc),
            "note": "Calibration needs out-of-fold class probabilities.",
        }
    labels = np.unique(y)
    try:
        return {
            "available": True,
            "brier_score": finite_or_none(brier_score(y, probabilities, labels=labels)),
            "expected_calibration_error": finite_or_none(
                expected_calibration_error(y, probabilities, n_bins=n_bins, labels=labels)["ece"]
            ),
            "n_bins": int(n_bins),
            "labels": [str(label) for label in labels],
            "convention": "multiclass_brier_sum_of_squared_class_errors",
        }
    except ValueError as exc:
        return {"available": False, "reason": str(exc)}


@router.post("/{task_name}")
def compare_bundles(task_name: str, body: ComparisonRequest):
    """Paired comparison of stored model bundles on one shared fold set."""
    from sklearn.model_selection import cross_validate

    if body.control is not None and body.control not in body.model_versions:
        raise bad_request(
            "control %r is not one of the compared model versions %s"
            % (body.control, ", ".join(repr(name) for name in body.model_versions))
        )
    if len(set(body.model_versions)) != len(body.model_versions):
        raise bad_request("model_versions must not repeat a version")

    if len(body.model_versions) < 2:
        raise bad_request(
            "A comparison needs at least two model versions; got %d."
            % len(body.model_versions)
        )

    bundles, families, sizes, integrities = {}, {}, {}, {}
    for version in body.model_versions:
        bundle = resolve_bundle(task_name, model_version=version, strict_dependencies=False)
        estimator = _explainable_estimator(bundle)
        family = problem_family(estimator)
        if family is None:
            raise unprocessable(
                "Model version %r does not carry a scikit-learn estimator, so it cannot "
                "be compared on cross-validated scores." % version
            )
        families[version] = family
        path = settings.task_bundles_dir(task_name) / (
            "%s.bundle" % getattr(bundle, "model_version", version)
        )
        size = None
        integrity = None
        try:
            size = int(path.stat().st_size)
            integrity = verify_bundle(path)
        except OSError:
            pass
        except Exception as exc:  # corrupt checksum: report it, do not crash the comparison
            integrity = {"verified": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        sizes[version] = size
        integrities[version] = integrity
        bundles[version] = (bundle, estimator)

    distinct_families = set(families.values())
    if len(distinct_families) != 1:
        raise bad_request(
            "All compared bundles must belong to the same problem family; got %s."
            % ", ".join("%s=%s" % (name, families[name]) for name in body.model_versions)
        )
    family = distinct_families.pop()

    partition = resolve_partition(task_name, body.partition)
    frame, y, feature_source = _explain_frame(task_name, partition)
    temporal = feature_source == "temporal_design"

    if body.max_rows is not None and len(frame) > body.max_rows:
        rng = np.random.RandomState(body.random_state)
        keep = np.sort(rng.choice(len(frame), body.max_rows, replace=False))
        frame, y = frame.iloc[keep], y[keep]
    splits, splitter_name, effective_repeats = _folds(
        frame,
        y,
        family,
        n_splits=body.n_splits,
        repeats=body.repeats,
        random_state=body.random_state,
        temporal=temporal,
    )

    scoring, scorer, higher_is_better = _scorer(body.scoring, family)
    records = {}
    warnings: list[str] = []
    for version in body.model_versions:
        bundle, estimator = bundles[version]
        try:
            result = cross_validate(
                estimator,
                frame,
                y,
                cv=splits,
                scoring=scorer,
                n_jobs=1,
                error_score=np.nan,
                return_estimator=False,
            )
        except Exception as exc:
            raise unprocessable(
                "Model version %r could not be cross-validated on this task: %s: %s"
                % (version, type(exc).__name__, exc)
            )
        scores = np.asarray(result["test_score"], dtype=float)
        finite = scores[np.isfinite(scores)]
        if finite.size == 0:
            warnings.append(
                "Model %r produced no finite fold score; it is reported but excluded "
                "from the paired tests." % version
            )
        records[version] = {
            "model_version": version,
            "resolved_version": getattr(bundle, "model_version", None),
            "family": families[version],
            "fold_scores": [finite_or_none(value) for value in scores],
            "score_mean": finite_or_none(finite.mean()) if finite.size else None,
            "score_std": finite_or_none(finite.std(ddof=1)) if finite.size > 1 else None,
            "scored_folds": int(finite.size),
            "fit_time_s": finite_or_none(float(np.mean(result["fit_time"]))),
            "predict_time_ms": finite_or_none(
                _predict_latency_ms(estimator, frame, random_state=body.random_state)
            ),
            "parameters": fitted_parameter_count(estimator),
            "disk_bytes": sizes[version],
            "integrity": integrities.get(version),
        }
        if body.include_calibration and family == "Classification":
            records[version]["calibration"] = _calibration(estimator, frame, y, splits)

    available = [version for version in body.model_versions if records[version]["scored_folds"]]
    if len(available) < 2:
        raise unprocessable(
            "Fewer than two models produced finite fold scores, so nothing can be paired. "
            "Inspect the per-model `fold_scores` for the failures."
        )

    scores_by_model = {
        version: [np.nan if value is None else value for value in records[version]["fold_scores"]]
        for version in available
    }
    control = body.control if body.control in available else available[0]
    statistics = compare_scores(
        scores_by_model,
        control=control,
        higher_is_better=higher_is_better,
        alpha=body.alpha,
        n_resamples=body.n_resamples,
        seed=body.random_state,
    )
    stability = rank_stability(scores_by_model, higher_is_better=higher_is_better)

    objectives = {
        version: {
            "score": records[version]["score_mean"],
            "complexity": records[version]["parameters"],
            "cost": (
                None
                if records[version]["fit_time_s"] is None
                else 1000.0 * records[version]["fit_time_s"]
                + (records[version]["predict_time_ms"] or 0.0)
            ),
        }
        for version in available
    }
    pareto = pareto_frontier(objectives, higher_is_better=higher_is_better)
    simpler = simpler_within_tolerance(
        objectives,
        higher_is_better=higher_is_better,
        tolerance=body.simpler_tolerance,
    )
    for record in records.values():
        record["rank"] = stability["models"].get(record["model_version"])

    n_pairs = int(np.sum(~np.isnan(list(scores_by_model[available[0]]))))
    if n_pairs < 6:
        warnings.append(
            "Only %d usable fold(s) are paired; a two-sided Wilcoxon signed-rank test "
            "cannot reach p < 0.05 with fewer than six non-zero pairs, so a "
            "`distinguishable` verdict is unreachable by construction." % n_pairs
        )
    warnings.append(
        "These bundles were selected by a hyper-parameter search on this task. Scores on "
        "the stored partition are an exploratory ranking of already-selected models, not "
        "an unbiased estimate of generalization; use a nested or fresh split for that."
    )

    return {
        "task_name": task_name,
        "partition": partition,
        "feature_source": feature_source,
        "rows": int(len(frame)),
        "scoring": scoring,
        "scoring_label": learning_curve_metric_label(scoring),
        "higher_is_better": bool(higher_is_better),
        "family": family,
        "cv": {
            "splitter": splitter_name,
            "n_splits": int(body.n_splits),
            "repeats": int(effective_repeats),
            "random_state": body.random_state,
            "folds": int(len(splits)),
            "temporal": bool(temporal),
        },
        "settings": {
            "max_rows": body.max_rows,
            "n_resamples": body.n_resamples,
            "confidence": body.confidence,
            "alpha": body.alpha,
            "simpler_tolerance": body.simpler_tolerance,
        },
        "score_provenance": {
            "basis": "cross_validated_refit_on_stored_partition",
            "exploratory": True,
            "selected_on_same_data": True,
            "note": (
                "Each estimator was chosen by a search on this task's data; the same "
                "partition is reused here so the comparison is paired, not unbiased."
            ),
        },
        "models": [records[version] for version in body.model_versions],
        "statistics": statistics,
        "stability": stability,
        "trade_off": {
            "objectives": pareto["objectives"],
            "pareto_frontier": pareto["frontier"],
            "dominated": pareto["dominated"],
            "incomplete": pareto["incomplete"],
            "simpler_within_tolerance": simpler,
        },
        "n_pairs": n_pairs,
        "warnings": warnings,
    }
