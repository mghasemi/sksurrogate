"""Evaluation endpoints: per-bundle learning curves across relevant metrics.

The Evaluation page compares each bundle's stored metrics side by side; this
router adds the complementary *learning curve* view — how a chosen metric
evolves as the training set grows. Curves are produced by scikit-learn's
``learning_curve`` on the bundle's own estimator and the task's stored train
partition, using the cross-validation splitter persisted for the task.

Two endpoints, deliberately split so the page can render the metric tabs
cheaply and only pay for curve computation when a tab is actually opened:

- ``/metrics``          → relevant metrics for the task's problem family (cheap)
- ``/learning-curves``  → the curves themselves for one metric (expensive)
"""

from fastapi import APIRouter, HTTPException

from SKSurrogate import load_bundle

from ..config import settings
from ..deps import not_found
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
