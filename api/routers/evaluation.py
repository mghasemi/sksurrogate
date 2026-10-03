"""Evaluation endpoints: per-bundle learning curves across relevant metrics.

The Evaluation page compares each bundle's stored metrics side by side; this
router adds the complementary *learning curve* view — how a chosen metric
evolves as the training set grows. Curves are produced by scikit-learn's
``learning_curve`` on the bundle's own estimator and the task's stored train
partition, using the cross-validation splitter persisted for the task.

Endpoints:

- ``/metrics``          → relevant metrics for the task's problem family (cheap)
- ``/learning-curves``  → the curves themselves for one metric (expensive)
- ``/nested-cv``        → nested cross-validation evaluation as a background job
- ``/{version}/curves`` → diagnostic curves (ROC, calibration, gain, lift), computed
                          server-side without matplotlib; classification bundles only
"""

import sqlite3

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from SKSurrogate import load_bundle, mltrack

from ..config import settings
from ..deps import (
    bad_request,
    finite_or_none,
    not_found,
    open_tracker,
    resolve_bundle,
    unprocessable,
)
from ..jobs import job_manager
from ..training import (
    compute_learning_curve,
    learning_curve_metric_label,
    learning_curve_metric_options,
    problem_family,
)

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

#: Cap on how many bundles are inspected when inferring the task's problem family.
_FAMILY_PROBE_LIMIT = 5


def _bundle_versions(task_name):
    """Unregistered bundle versions for a task, most recent first (matches /api/bundles)."""
    bundles_dir = settings.task_bundles_dir(task_name)
    if not bundles_dir.exists():
        return []
    entries = sorted(bundles_dir.glob("*.bundle"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [path.stem for path in entries]


def _load_bundle(task_name, model_version):
    bundle_path = settings.task_bundles_dir(task_name) / (model_version + ".bundle")
    if not bundle_path.exists():
        raise not_found("No bundle %r found for task %r" % (model_version, task_name))
    return load_bundle(bundle_path, strict_dependencies=False)


def _detect_family(task_name, model_version=None):
    """Infer Classification/Regression from the task's bundles, or ``None`` when unknown."""
    versions = [model_version] if model_version else _bundle_versions(task_name)[:_FAMILY_PROBE_LIMIT]
    for version in versions:
        try:
            bundle = _load_bundle(task_name, version)
        except HTTPException:
            continue
        family = problem_family(getattr(bundle, "model", None))
        if family is not None:
            return family
    return None


@router.get("/{task_name}/metrics")
def evaluation_metrics(task_name: str, model_version: str | None = None):
    """Relevant learning-curve metrics (and their default) for a task.

    Passing ``model_version`` probes just that bundle instead of scanning the
    task's bundles, which keeps the tab bar cheap to render.
    """
    family = _detect_family(task_name, model_version)
    return {"task_name": task_name, **learning_curve_metric_options(family)}


@router.get("/{task_name}/learning-curves")
def evaluation_learning_curves(
    task_name: str,
    scoring: str = "accuracy",
    train_partition: str = "train",
    n_points: int = 5,
    model_versions: str | None = None,
):
    """One cross-validated learning curve per bundle for the requested metric.

    Bundles that cannot produce a curve for this metric (a regression metric on
    a classifier, a non-numeric feature column, too few samples for the stored
    splitter, …) are reported in ``errors`` rather than failing the request, so
    the remaining bundles still render.
    """
    versions = (
        [version.strip() for version in model_versions.split(",") if version.strip()]
        if model_versions
        else _bundle_versions(task_name)
    )

    series = []
    errors = []
    for version in versions:
        bundle = _load_bundle(task_name, version)
        try:
            series.append(
                compute_learning_curve(
                    bundle,
                    task_name=task_name,
                    scoring=scoring,
                    train_partition=train_partition,
                    n_points=n_points,
                )
            )
        except HTTPException as exc:
            errors.append({"model_version": version, "detail": str(exc.detail)})
        except Exception as exc:  # scikit-learn failures (bad metric/estimator mix) are per-bundle
            errors.append({"model_version": version, "detail": "%s: %s" % (type(exc).__name__, exc)})

    return {
        "task_name": task_name,
        "scoring": scoring,
        "scoring_label": learning_curve_metric_label(scoring),
        "train_partition": train_partition,
        "n_points": n_points,
        "series": series,
        "errors": errors,
    }


class NestedCVRequest(BaseModel):
    """Nested cross-validation evaluation parameters.

    Exactly one of ``model_version`` / ``alias`` must be provided — the bundle's
    estimator is cloned per fold, so no fitted state from a previous run leaks in.
    The call is expensive (repeats × outer × inner fits), which is why it runs as
    a background job rather than inline.
    """

    model_version: str | None = None
    alias: str | None = None
    inner_cv: int = Field(default=2, ge=1)
    outer_cv: int = Field(default=2, ge=1)
    train_partition: str = "train"
    validation_partition: str = "validation"
    scoring: str | None = None
    repeats: int = Field(default=1, ge=1)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)


@router.post("/{task_name}/nested-cv")
def run_nested_cv(task_name: str, body: NestedCVRequest):
    """Submit a nested-CV evaluation job for one bundle and return its id immediately.

    The job re-opens the task's tracker on its own (the request context is gone by
    then) and calls ``mltrack.nested_cv_evaluation`` with the bundle's estimator;
    the result — outer/inner scores, per-fold metrics, repeat means, confidence
    interval, and the final validation score when a validation partition exists —
    lands in the job record for the UI to poll.
    """
    if (body.model_version is None) == (body.alias is None):
        raise bad_request("Exactly one of model_version or alias must be provided")

    # Resolve now so a missing bundle fails fast with 404/400, not as a job error.
    resolve_bundle(
        task_name,
        model_version=body.model_version,
        alias=body.alias,
        strict_dependencies=False,
    )

    def _task():
        from sklearn.metrics import get_scorer

        bundle = resolve_bundle(
            task_name,
            model_version=body.model_version,
            alias=body.alias,
            strict_dependencies=False,
        )
        scoring_fn = None
        if body.scoring is not None:
            try:
                scoring_fn = get_scorer(body.scoring)
            except ValueError as exc:
                raise bad_request(str(exc))

        with open_tracker(task_name) as tracker:
            result = tracker.nested_cv_evaluation(
                bundle.model,
                inner_cv=body.inner_cv,
                outer_cv=body.outer_cv,
                train_partition=body.train_partition,
                validation_partition=body.validation_partition,
                scoring=scoring_fn,
                repeats=body.repeats,
                confidence_level=body.confidence_level,
            )

        def _clean(value):
            if isinstance(value, dict):
                return {key: _clean(item) for key, item in value.items()}
            if isinstance(value, list):
                return [_clean(item) for item in value]
            return finite_or_none(value)

        result = _clean(result)
        result["model_version"] = bundle.model_version
        result["task_name"] = task_name
        return result

    job_id = job_manager.submit(task_name, "nested-cv", lambda job_id: _task)
    return {"job_id": job_id, "status": "queued"}


def _positive_scores(model, X_test):
    """Probability (or normalized decision value) of the positive class for ROC/calibration."""
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            return np.asarray(proba[:, 1], dtype=float)
        except Exception:
            pass
    scores = np.asarray(model.decision_function(X_test), dtype=float)
    span = float(scores.max() - scores.min())
    if span <= 0.0:
        raise ValueError("decision function is constant; cannot derive class probabilities")
    return (scores - scores.min()) / span


@router.get("/{task_name}/{model_version}/curves")
def bundle_curves(task_name: str, model_version: str, bins: int = Query(default=10, ge=2, le=50)):
    """Diagnostic curves for one classification bundle, computed without matplotlib.

    The bundle's estimator is evaluated on the 75/25 split cached by
    ``mltrack.split_train`` (the same data the library's own plot methods use).
    ROC and calibration work for any classifier; cumulative gain and lift are
    binary-only — when the test partition has more than two classes they are
    reported in ``errors`` rather than failing the request.
    """
    bundle = _load_bundle(task_name, model_version)
    family = problem_family(bundle.model)
    if family != "Classification":
        raise unprocessable(
            "Diagnostic curves require a classification bundle; %r is %s"
            % (model_version, family or "not a scikit-learn estimator")
        )

    with open_tracker(task_name) as tracker:
        # ``split_train`` falls back to the plain ``data`` table when the tracker's
        # X/y are unset — which API-registered tasks never have (their rows live in
        # per-partition tables). Preload a registered partition so the 75/25 split
        # has data to work with, mirroring what ``LogMetrics`` does.
        splits = tracker.dataset_splits()
        pool_partition = None
        for name in ("train", "validation", "test"):
            if name in splits:
                pool_partition = name
                break
        if pool_partition is None and splits:
            pool_partition = sorted(splits)[0]
        if pool_partition is not None:
            try:
                tracker.get_data(pool_partition)
            except ValueError as exc:
                raise not_found(str(exc))

        try:
            model, _, X_train, X_test, y_train, y_test = tracker.split_train(bundle.model)
        except sqlite3.OperationalError as exc:
            raise not_found("No registered data available for task %r (%s)" % (task_name, exc))
        y_true = np.asarray(y_test)
        classes = list(np.unique(y_true))

        curves = {}
        errors = []

        # ROC — any classifier with predict_proba or a decision function.
        try:
            from sklearn.metrics import auc, roc_curve

            fpr, tpr, _ = roc_curve(y_true, _positive_scores(model, X_test))
            curves["roc"] = {
                "fpr": [finite_or_none(v) for v in fpr.tolist()],
                "tpr": [finite_or_none(v) for v in tpr.tolist()],
                "auc": finite_or_none(auc(fpr, tpr)),
            }
        except Exception as exc:
            errors.append({"curve": "roc", "detail": "%s: %s" % (type(exc).__name__, exc)})

        # Calibration — reliability curve plus the histogram of predicted probabilities.
        try:
            from sklearn.calibration import calibration_curve

            prob_pos = _positive_scores(model, X_test)
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, prob_pos, n_bins=bins
            )
            counts, edges = np.histogram(prob_pos, bins=bins, range=(0.0, 1.0))
            curves["calibration"] = {
                "mean_predicted_value": [finite_or_none(v) for v in mean_predicted_value.tolist()],
                "fraction_of_positives": [finite_or_none(v) for v in fraction_of_positives.tolist()],
                "histogram_counts": [int(count) for count in counts],
                "histogram_edges": [finite_or_none(float(edge)) for edge in edges.tolist()],
            }
        except Exception as exc:
            errors.append({"curve": "calibration", "detail": "%s: %s" % (type(exc).__name__, exc)})

        # Cumulative gain and lift — binary classification only.
        if len(classes) != 2:
            errors.append(
                {
                    "curve": "cumulative_gain",
                    "detail": "Cumulative gain requires exactly 2 classes; found %d" % len(classes),
                }
            )
            errors.append(
                {
                    "curve": "lift",
                    "detail": "Lift requires exactly 2 classes; found %d" % len(classes),
                }
            )
        else:
            try:
                proba = np.asarray(model.predict_proba(X_test))
                prob_pos0, prob_pos1 = proba[:, 0], proba[:, 1]
            except Exception:
                decision = np.asarray(model.decision_function(X_test), dtype=float)
                span = float(decision.max() - decision.min())
                if span <= 0.0:
                    raise unprocessable("decision function is constant; cannot derive class probabilities")
                prob_pos1 = (decision - decision.min()) / span
                prob_pos0 = (decision.max() - decision) / span

            gain_ok = False
            try:
                percentages, gains_class0 = mltrack.cumulative_gain_curve(
                    y_true, prob_pos0, pos_label=classes[0]
                )
                _, gains_class1 = mltrack.cumulative_gain_curve(
                    y_true, prob_pos1, pos_label=classes[1]
                )
                curves["cumulative_gain"] = {
                    "percentages": [finite_or_none(v) for v in percentages.tolist()],
                    "gains_class0": [finite_or_none(v) for v in gains_class0.tolist()],
                    "gains_class1": [finite_or_none(v) for v in gains_class1.tolist()],
                    "class0": str(classes[0]),
                    "class1": str(classes[1]),
                }
                gain_ok = True
            except Exception as exc:
                errors.append(
                    {"curve": "cumulative_gain", "detail": "%s: %s" % (type(exc).__name__, exc)}
                )

            if gain_ok:
                try:
                    lift_percentages = percentages[1:]
                    lift_class0 = gains_class0[1:] / lift_percentages
                    lift_class1 = gains_class1[1:] / lift_percentages
                    curves["lift"] = {
                        "percentages": [finite_or_none(v) for v in lift_percentages.tolist()],
                        "lifts_class0": [finite_or_none(v) for v in lift_class0.tolist()],
                        "lifts_class1": [finite_or_none(v) for v in lift_class1.tolist()],
                        "class0": str(classes[0]),
                        "class1": str(classes[1]),
                    }
                except Exception as exc:
                    errors.append({"curve": "lift", "detail": "%s: %s" % (type(exc).__name__, exc)})

    return {
        "task_name": task_name,
        "model_version": model_version,
        "bins": bins,
        "n_test_samples": int(len(y_true)),
        "classes": [str(cls) for cls in classes],
        **curves,
        "errors": errors,
    }
