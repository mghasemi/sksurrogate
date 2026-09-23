"""Dataset registration endpoints.

Wraps ``DataProcess.DataPreprocess`` (type preview) and
``mltrace.mltrack.RegisterData`` (schema fingerprinting) behind plain
file-upload + JSON endpoints, per docs/ui-plan.md section 4.
"""

import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from SKSurrogate import (
    STANDARD_CV_SPLITTERS,
    DataPreprocess,
    build_cv,
    cv_param_defs,
    cv_to_spec,
    default_cv_spec,
)

from ..config import settings
from ..deps import bad_request, not_found, open_tracker

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class SetCVRequest(BaseModel):
    """Cross-validation splitter selection for a task.

    ``spec`` is either an integer fold count (scikit-learn convention: number of
    ``(Stratified)KFold`` folds), or a dictionary with a ``type`` key naming one
    of the standard splitters, e.g. ``{"type": "StratifiedKFold"}``.

    ``params`` optionally carries the splitter's constructor parameters (e.g.
    ``{"n_splits": 5, "shuffle": true}``); they are merged into the spec before
    it is validated and stored in the tracking pipeline.
    """

    spec: dict | int
    params: dict[str, object] | None = None


@router.get("/{task_name}/cv")
def get_cv(task_name: str):
    """Return the standard splitter options and the task's current CV selection."""
    if not settings.mltrace_db_path(task_name).exists():
        raise not_found("No dataset registered for task %r" % task_name)
    with open_tracker(task_name) as tracker:
        stored_spec = tracker.GetCVSpec()
    return {
        "task_name": task_name,
        "options": list(STANDARD_CV_SPLITTERS),
        # Constructor parameters each splitter exposes, so the UI can render
        # editable fields below the dropdown (name/kind/default per parameter).
        "param_defs": {name: cv_param_defs(name) for name in STANDARD_CV_SPLITTERS},
        "current": stored_spec if stored_spec is not None else default_cv_spec(),
        "stored": stored_spec is not None,
    }


@router.put("/{task_name}/cv")
def set_cv(task_name: str, body: SetCVRequest):
    """Persist the task's cross-validation splitter in the tracking pipeline."""
    if not settings.mltrace_db_path(task_name).exists():
        raise not_found("No dataset registered for task %r" % task_name)
    spec = body.spec
    if body.params is not None and isinstance(spec, dict):
        # Merge user-edited constructor parameters into the splitter spec.
        merged = dict(spec)
        merged.update(body.params)
        spec = merged
    try:
        with open_tracker(task_name) as tracker:
            stored_spec = tracker.SetCV(spec)
    except (ValueError, KeyError, TypeError) as exc:
        raise bad_request(str(exc))
    return {"task_name": task_name, "cv": stored_spec}


@router.post("/{task_name}/register")
async def register_dataset(task_name: str, target: str = Form(...), partition: str = Form("train"),
                            file: UploadFile = File(...)):
    """Upload a CSV, deduce its schema, and register it for a task/partition."""
    if not file.filename.lower().endswith(".csv"):
        raise bad_request("Only CSV uploads are supported")
    raw_bytes = await file.read()
    dataset_dir = settings.task_dataset_dir(task_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_dir / (partition + ".csv")
    csv_path.write_bytes(raw_bytes)

    frame = pd.read_csv(csv_path)
    if target not in frame.columns:
        raise bad_request("target column %r not found in the uploaded dataset" % target)

    preview = DataPreprocess(frame)
    preview.deduce_types()

    with open_tracker(task_name) as tracker:
        tracker.RegisterData(frame, target, partition=partition)
        metadata = tracker.GetMetadata()

    return {
        "task_name": task_name,
        "partition": partition,
        "target": target,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dataset_fingerprint": metadata.get("dataset_fingerprint"),
        "dataset_schema": metadata.get("dataset_schema"),
        "deduced_types": preview.pivot_types,
    }


@router.get("/{task_name}")
def get_dataset_metadata(task_name: str):
    """Return the registered schema, fingerprint, and partition history for a task."""
    if not settings.mltrace_db_path(task_name).exists():
        raise not_found("No dataset registered for task %r" % task_name)
    with open_tracker(task_name) as tracker:
        metadata = tracker.GetMetadata()
        splits = tracker.dataset_splits()
    return {
        "task_name": task_name,
        "target_name": metadata.get("target_name"),
        "dataset_fingerprint": metadata.get("dataset_fingerprint"),
        "dataset_columns": metadata.get("dataset_columns"),
        "dataset_schema": metadata.get("dataset_schema"),
        "dataset_feature_count": metadata.get("dataset_feature_count"),
        "partitions": splits,
    }


@router.get("/{task_name}/{partition}/preview")
def preview_dataset(task_name: str, partition: str, limit: int = 20):
    """Return the first ``limit`` rows of a previously uploaded partition."""
    csv_path = settings.task_dataset_dir(task_name) / (partition + ".csv")
    if not csv_path.exists():
        raise not_found("No dataset partition %r stored for task %r" % (partition, task_name))
    frame = pd.read_csv(csv_path, nrows=max(limit, 0))
    return {"task_name": task_name, "partition": partition, "rows": frame.to_dict(orient="records")}
