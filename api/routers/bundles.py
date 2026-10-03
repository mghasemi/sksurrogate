"""Model bundle creation & inspection endpoints.

Phase A ships a synchronous "train baseline" convenience endpoint so bundles
can be produced end-to-end without the async AutoML job infrastructure
(Phase C, see docs/ui-plan.md section 8). Fitting logic is shared with the
Experiments and Retraining routers via ``api.training``.

Phase 3 (docs/ui-gap-implementation-plan.md) adds model preservation
(``PreserveModel`` / ``RecoverModel``), a best-model shortcut, MLflow export,
and raw bundle download.
"""

import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import FileResponse

from SKSurrogate import ModelBundle, export_mlflow, load_bundle, save_bundle

from ..config import settings
from ..deps import bad_request, bundle_summary, finite_or_none, not_found, open_tracker
from ..training import fit_baseline_bundle, get_registered_target, load_partition

router = APIRouter(prefix="/api/bundles", tags=["bundles"])


class TrainBaselineRequest(BaseModel):
    estimator: str = "linear_regression"
    train_partition: str = "train"
    validation_partition: str | None = "validation"
    owner: str | None = None
    run_id: str | None = None


@router.post("/{task_name}/train-baseline")
def train_baseline(task_name: str, body: TrainBaselineRequest):
    """Fit a simple scikit-learn baseline pipeline and wrap it into a bundle."""
    bundle = fit_baseline_bundle(
        task_name,
        estimator=body.estimator,
        train_partition=body.train_partition,
        validation_partition=body.validation_partition,
        owner=body.owner,
        run_id=body.run_id,
    )
    bundles_dir = settings.task_bundles_dir(task_name)
    bundles_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = save_bundle(bundle, bundles_dir / (bundle.model_version + ".bundle"))

    return {
        "task_name": task_name,
        "model_version": bundle.model_version,
        "estimator": body.estimator,
        "metrics": bundle.metrics,
        "schema": bundle.schema,
        "bundle_path": str(bundle_path),
    }


@router.get("/{task_name}")
def list_bundles(task_name: str):
    """List unregistered bundles produced for a task, most recent first."""
    bundles_dir = settings.task_bundles_dir(task_name)
    if not bundles_dir.exists():
        return {"task_name": task_name, "bundles": []}
    entries = sorted(bundles_dir.glob("*.bundle"), key=lambda path: path.stat().st_mtime, reverse=True)
    return {"task_name": task_name, "bundles": [path.stem for path in entries]}


# --------------------------------------------------------------------------- #
# Phase 3.1 — model preservation                                              #
# --------------------------------------------------------------------------- #

def _bundle_file(task_name: str, model_version: str):
    """Locate a task's bundle file (unregistered folder first, then registry)."""
    for candidate in (
        settings.task_bundles_dir(task_name) / (model_version + ".bundle"),
        settings.registry_dir / task_name / (model_version + ".bundle"),
    ):
        if candidate.exists():
            return candidate
    raise not_found("No bundle %r found for task %r" % (model_version, task_name))


@router.post("/{task_name}/{model_version}/preserve")
def preserve_bundle(task_name: str, model_version: str):
    """Pickle the bundle's fitted estimator into mltrace's ``saved`` table.

    The estimator is logged first (``LogModel``) when it has no tracking id yet,
    so the snapshot can later be recovered and traced back to a task. Each call
    loads a *fresh* estimator object from the bundle file, so the already-logged
    row is looked up by name: preserving the same model version twice appends a
    new snapshot to one model row instead of piling up duplicate model rows.

    Returns the new pickle row's id plus the model id it belongs to.
    """
    from SKSurrogate.mltrace import MLModel

    bundle_path = _bundle_file(task_name, model_version)
    bundle = load_bundle(bundle_path, strict_dependencies=False)
    with open_tracker(task_name) as tracker:
        if not hasattr(bundle.model, "mltrack_id"):
            logged = (
                MLModel.select()
                .where((MLModel.task_id == tracker.task_id) & (MLModel.name == model_version))
                .first()
            )
            if logged is None:
                tracker.LogModel(bundle.model, name=model_version)
            else:
                bundle.model.mltrack_id = logged.model_id
        tracker.PreserveModel(bundle.model)
        rows = tracker.allPreserved()
    row = rows.iloc[-1]
    return {
        "task_name": task_name,
        "model_version": model_version,
        "pickle_id": int(row["pickle_id"]),
        "model_id": int(row["model_id"]),
        "init_date": str(row["init_date"]),
    }


@router.get("/{task_name}/preserved")
def list_preserved(task_name: str):
    """List all preserved (pickled) model snapshots for a task."""
    if not settings.mltrace_db_path(task_name).exists():
        return {"task_name": task_name, "snapshots": []}
    with open_tracker(task_name) as tracker:
        rows = tracker.allPreserved()
    snapshots = [
        {
            "pickle_id": int(row["pickle_id"]),
            "model_id": int(row["model_id"]),
            "init_date": str(row["init_date"]),
        }
        for _, row in rows.iterrows()
    ]
    return {"task_name": task_name, "snapshots": snapshots}


class RecoverRequest(BaseModel):
    pickle_id: int


@router.post("/{task_name}/recover")
def recover_model(task_name: str, body: RecoverRequest):
    """Recover a preserved snapshot and re-save it as a fresh bundle version.

    ``mltrack.RecoverModel`` looks up the newest pickle **by model id**, not by
    pickle id, so the request's ``pickle_id`` is resolved to its owning model
    first — the two ids only coincide for the very first snapshot of a task, and
    a model may hold several snapshots taken at different points in time.

    The recovered estimator is scored on the task's train partition (when one
    is registered) so the response carries a smoke score; the fitted object
    itself never leaves the server — only the new ``model_version`` does.
    """
    if not settings.mltrace_db_path(task_name).exists():
        raise not_found("No dataset registered for task %r" % task_name)
    with open_tracker(task_name) as tracker:
        # ``allPreserved()`` is a plain SQL projection, so it hands back ints
        # rather than peewee's resolved foreign-key objects.
        snapshots = tracker.allPreserved()
        match = snapshots[snapshots["pickle_id"] == body.pickle_id]
        if match.empty:
            raise not_found(
                "No preserved snapshot with pickle_id %d for task %r" % (body.pickle_id, task_name)
            )
        model_id = int(match.iloc[0]["model_id"])
        try:
            model = tracker.RecoverModel(model_id)
        except IndexError:
            raise not_found(
                "Preserved snapshot %d points at missing model %d for task %r"
                % (body.pickle_id, model_id, task_name)
            )

    # Strictly best-effort: the recovery already succeeded, so a missing or
    # unscorable train partition must never turn into a failed request.
    smoke_score = None
    if settings.task_dataset_dir(task_name).exists():
        try:
            target, _ = get_registered_target(task_name)
            frame = load_partition(task_name, "train")
            if target in frame.columns:
                smoke_score = finite_or_none(float(model.score(frame.drop(columns=[target]), frame[target])))
        except Exception:
            smoke_score = None

    bundle = ModelBundle(
        model,
        task_name=task_name,
        metrics={"recovered_smoke_score": smoke_score} if smoke_score is not None else {},
    )
    bundles_dir = settings.task_bundles_dir(task_name)
    bundles_dir.mkdir(parents=True, exist_ok=True)
    save_bundle(bundle, bundles_dir / (bundle.model_version + ".bundle"))

    return {
        "task_name": task_name,
        "pickle_id": body.pickle_id,
        "model_id": int(model_id),
        "model_type": type(model).__name__,
        "smoke_score": smoke_score,
        "model_version": bundle.model_version,
    }


# --------------------------------------------------------------------------- #
# Phase 3.2 — best-model shortcut                                             #
# --------------------------------------------------------------------------- #

#: Metric columns mltrace's Metrics table actually stores (see the peewee model).
_MLTRACE_METRIC_FIELDS = {
    "accuracy", "auc", "precision", "recall", "f1", "mcc",
    "logloss", "variance", "max_error", "mse", "mae", "r2",
}


def _rank_bundles_by_metric(task_name: str, metric: str):
    """Best ``(value, model_version)`` among the task's bundle files for one metric, or ``(None, None)``.

    Bundles created through this API store their metrics on the bundle file
    itself (``score`` for baselines, ``train_score`` for experiments), so a
    scan over the stored files is the primary source.
    """
    bundles_dir = settings.task_bundles_dir(task_name)
    if not bundles_dir.exists():
        return None, None
    candidates: list[tuple[float, str]] = []
    for path in sorted(bundles_dir.glob("*.bundle")):
        try:
            bundle = load_bundle(path, strict_dependencies=False)
        except Exception:
            continue  # unreadable/corrupt file — skip rather than fail the scan
        value = finite_or_none((bundle.metrics or {}).get(metric))
        if value is not None:
            candidates.append((value, path.stem))
    return max(candidates, key=lambda item: item[0]) if candidates else (None, None)


def _rank_logged_models_by_metric(task_name: str, metric: str):
    """Fallback ranking over mltrace's Metrics table (logged, not bundled, models)."""
    from SKSurrogate.mltrace import MLModel, Metrics, Task

    if metric not in _MLTRACE_METRIC_FIELDS or not settings.mltrace_db_path(task_name).exists():
        return None, None
    with open_tracker(task_name) as tracker:
        rows = (
            Metrics.select()
            .join(MLModel, on=Metrics.model_id == MLModel.model_id)
            .switch(Task, on=Task.task_id == MLModel.task_id)
            .where(Task.task_id == tracker.task_id)
            .order_by(Metrics.__dict__[metric].__dict__["field"].desc())
        )
        if not len(rows):
            return None, None
        row = rows[0]
        # The Metrics table has no model_version column, so the winner is
        # reported by model id.
        return finite_or_none(getattr(row, metric)), "mlmodel-%d" % int(row.model_id)


@router.get("/{task_name}/best")
def best_bundle(task_name: str, metric: str = "score"):
    """Return the task's bundle with the highest value of ``metric``.

    ``metric`` may be a comma-separated preference list (e.g.
    ``score,train_score``), in which case the first metric that ranks at least
    one bundle wins — this lets a caller cover both baseline and experiment
    bundles in a single request instead of chaining 404s.
    """
    metrics = [name.strip() for name in metric.split(",") if name.strip()]
    if not metrics:
        raise bad_request("At least one metric name must be provided")

    for name in metrics:
        best_value, best_version = _rank_bundles_by_metric(task_name, name)
        if best_version is None:
            best_value, best_version = _rank_logged_models_by_metric(task_name, name)
        if best_version is not None:
            return {
                "task_name": task_name,
                "metric": name,
                "metrics_tried": metrics,
                "model_version": best_version,
                "value": best_value,
            }

    raise not_found(
        "No bundle for task %r carries a finite %s metric to rank by"
        % (task_name, "/".join(repr(name) for name in metrics))
    )


# --------------------------------------------------------------------------- #
# Phase 3.3 — MLflow export & bundle download                                 #
# --------------------------------------------------------------------------- #

@router.post("/{task_name}/{model_version}/export-mlflow")
def export_mlflow_bundle(task_name: str, model_version: str):
    """Export a bundle in the standard MLflow directory layout and return it as a zip.

    The export lands under ``exports/{task}/{version}-<timestamp>/`` (a fresh
    directory per call — ``export_mlflow`` refuses existing destinations), is
    zipped, and served back as a download named ``{task}-{version}.mlflow.zip``.
    """
    bundle_path = _bundle_file(task_name, model_version)
    bundle = load_bundle(bundle_path, strict_dependencies=False)

    export_root = settings.root / "exports" / task_name
    export_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = export_root / ("%s-%s" % (model_version, stamp))
    try:
        exported_dir = export_mlflow(bundle, destination)
    except FileExistsError as exc:
        raise bad_request(str(exc))

    zip_path = export_root / ("%s-%s.mlflow.zip" % (task_name, model_version))
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=str(export_root), prefix=".zip-", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(exported_dir.rglob("*")):
                if file.is_file():
                    archive.write(file, arcname=file.relative_to(export_root))
        shutil.move(str(temporary_path), str(zip_path))
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename="%s-%s.mlflow.zip" % (task_name, model_version),
    )


@router.get("/{task_name}/{model_version}/download")
def download_bundle(task_name: str, model_version: str):
    """Serve the raw ``.bundle`` file for a task's model version."""
    bundle_path = _bundle_file(task_name, model_version)
    return FileResponse(
        path=str(bundle_path),
        media_type="application/octet-stream",
        filename=model_version + ".bundle",
    )


# --------------------------------------------------------------------------- #
# Bundle detail (declared last)                                               #
# --------------------------------------------------------------------------- #
# FastAPI matches routes in declaration order, and this two-segment pattern is
# the most permissive one in the router: /{task}/best, /{task}/preserved and
# any future static suffix must be declared ABOVE it or they are swallowed as a
# model_version.

@router.get("/{task_name}/{model_version}")
def get_bundle(task_name: str, model_version: str):
    """Return bundle metadata (schema, metrics, dependencies, audit trail) without the model object."""
    bundle_path = settings.task_bundles_dir(task_name) / (model_version + ".bundle")
    if not bundle_path.exists():
        raise not_found("No bundle %r found for task %r" % (model_version, task_name))
    bundle = load_bundle(bundle_path, strict_dependencies=False)
    return bundle_summary(bundle)
