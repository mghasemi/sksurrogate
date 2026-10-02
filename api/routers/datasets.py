"""Dataset registration endpoints.

Wraps ``DataProcess.DataPreprocess`` (type preview),
``mltrace.mltrack.RegisterData`` (schema fingerprinting), and
``mltrack.validate_data`` (pre-flight schema validation) behind plain
file-upload + JSON endpoints, per docs/ui-plan.md section 4 and
docs/ui-gap-implementation-plan.md Phase 5.1/5.2.
"""

import json
import re
from io import BytesIO
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from SKSurrogate import (
    STANDARD_CV_SPLITTERS,
    DataPreprocess,
    ModelRegistry,
    cv_param_defs,
    default_cv_spec,
    sensitive_feature_report,
)

from ..config import settings
from ..deps import bad_request, finite_or_none as _finite, not_found, open_tracker, unprocessable

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _validate_partition_name(partition):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", partition or ""):
        raise bad_request("partition must contain only letters, numbers, underscores, or hyphens")


def apply_type_overrides(frame, preview, target, overrides=None, binarize_label=False):
    """Apply user-reviewed types through DataPreprocess before registration.

    ``DataPreprocess.set_type`` / ``transform_label_bin`` only record intent in the
    deducer's bookkeeping; they do not change arbitrary feature values. Preserve
    the uploaded feature dtypes and values, and only rewrite the target when the
    user explicitly requests label binarization.

    Returns ``(frame, pivot_types)`` where ``pivot_types`` is the deducer's final
    per-column type map after the overrides.
    """
    frame = frame.copy()
    for column, typ in (overrides or {}).items():
        if column not in frame.columns:
            raise bad_request("type override for unknown column %r" % column)
        try:
            preview.set_type(column, typ)
        except Exception as exc:
            raise bad_request("Invalid type override for column %r: %s" % (column, exc))
    if binarize_label:
        try:
            preview.transform_label_bin(target)
            mapping = preview.mapping.get(target) or {}
            frame[target] = frame[target].map(mapping).astype(float)
        except (TypeError, ValueError, KeyError) as exc:
            raise bad_request("Could not binarize target %r: %s" % (target, exc))
    return frame, preview.pivot_types


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


@router.post("/{task_name}/inspect")
async def inspect_dataset(task_name: str, file: UploadFile = File(...)):
    """Preview a CSV's deduced schema without persisting anything (Phase 5.1).

    Runs ``DataPreprocess.deduce_types`` and the name-based sensitive-column scan
    so the UI can show an editable review table before the user commits the upload
    via ``POST /{task}/register``.
    """
    if not (file.filename or "").lower().endswith(".csv"):
        raise bad_request("Only CSV uploads are supported")
    raw_bytes = await file.read()
    try:
        frame = pd.read_csv(BytesIO(raw_bytes))
    except Exception as exc:
        raise bad_request("Could not parse the uploaded CSV: %s" % exc)

    preview = DataPreprocess(frame)
    preview.deduce_types()
    return {
        "task_name": task_name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "deduced_types": preview.pivot_types,
        # Columns the deducer flagged as label candidates (object dtype, not categorical).
        "target_candidates": list(preview.deduced_types.get("label", [])),
        "sensitive_scan": sensitive_feature_report(frame),
    }


@router.post("/{task_name}/register")
async def register_dataset(task_name: str, target: str = Form(...), partition: str = Form("train"),
                            file: UploadFile = File(...),
                            type_overrides: Annotated[str | None, Form()] = None,
                            binarize_label: Annotated[bool, Form()] = False):
    """Upload a CSV, deduce its schema, and register it for a task/partition.

    ``type_overrides`` (JSON object of column -> type) is applied via
    ``DataPreprocess.set_type`` before registration so the user-reviewed types from
    the inspect step win over the automatic deduction; ``binarize_label`` maps the
    target's unique values to 0..n-1 through ``transform_label_bin`` (Phase 5.1).
    """
    if not (file.filename or "").lower().endswith(".csv"):
        raise bad_request("Only CSV uploads are supported")
    _validate_partition_name(partition)
    raw_bytes = await file.read()
    try:
        frame = pd.read_csv(BytesIO(raw_bytes))
    except Exception as exc:
        raise bad_request("Could not parse the uploaded CSV: %s" % exc)
    if target not in frame.columns:
        raise bad_request("target column %r not found in the uploaded dataset" % target)

    preview = DataPreprocess(frame)
    preview.deduce_types()
    overrides = {}
    if isinstance(type_overrides, str) and type_overrides:
        try:
            overrides = json.loads(type_overrides)
        except (TypeError, ValueError):
            raise bad_request("type_overrides must be a JSON object of column -> type")
        if not isinstance(overrides, dict) or not all(
            isinstance(column, str) and isinstance(typ, str) for column, typ in overrides.items()
        ):
            raise bad_request("type_overrides must be a JSON object of column -> type strings")
    frame, pivot_types = apply_type_overrides(
        frame, preview, target, overrides=overrides or None, binarize_label=binarize_label is True
    )
    dataset_dir = settings.task_dataset_dir(task_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_dir / (partition + ".csv")

    with open_tracker(task_name) as tracker:
        tracker.RegisterData(frame, target, partition=partition)
        tracker.UpdateMetadata({"dataset_deduced_types": pivot_types})
        metadata = tracker.GetMetadata()

    # Persist the materialized target values after successful tracker registration.
    frame.to_csv(csv_path, index=False)

    # Phase 3.4 auto-wiring: record the stored CSV as a registry artifact so the
    # lineage rail and the artifact-cleanup UI can see it. Best-effort — a
    # registry failure must never fail an otherwise successful upload.
    try:
        ModelRegistry(settings.registry_dir).register_dataset(
            task_name,
            metadata.get("dataset_fingerprint"),
            csv_path,
            metadata={"partition": partition, "target": target, "rows": int(len(frame))},
        )
    except Exception:  # pragma: no cover - defensive; registration is auxiliary
        pass

    return {
        "task_name": task_name,
        "partition": partition,
        "target": target,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dataset_fingerprint": metadata.get("dataset_fingerprint"),
        "dataset_schema": metadata.get("dataset_schema"),
        "dataset_deduced_types": metadata.get("dataset_deduced_types"),
        "deduced_types": pivot_types,
        # Name-based PII / sensitive-column scan of the uploaded frame (Phase 2.4).
        "sensitive_scan": sensitive_feature_report(frame),
    }


@router.post("/{task_name}/validate")
async def validate_dataset(
    task_name: str,
    file: Annotated[UploadFile | None, File()] = None,
    partition: Annotated[str | None, Form()] = None,
    missing_columns: Annotated[str, Form()] = "raise",
):
    """Pre-flight schema validation against the registered dataset (Phase 5.2).

    Accepts either a CSV upload or an existing partition name; runs
    ``tracker.validate_data(df, target)`` and returns ``{"valid": true}`` on success
    or a 422 carrying the library's schema-difference message.
    """
    if missing_columns not in ("raise", "ignore", "allow"):
        raise bad_request("missing_columns must be one of: raise, ignore, allow")
    if partition is not None:
        _validate_partition_name(partition)

    frame = None
    source = None
    if file is not None:
        if not (file.filename or "").lower().endswith(".csv"):
            raise bad_request("Only CSV uploads are supported")
        raw_bytes = await file.read()
        try:
            frame = pd.read_csv(BytesIO(raw_bytes))
        except Exception as exc:
            raise bad_request("Could not parse the uploaded CSV: %s" % exc)
        source = "upload"
    elif partition is not None:
        csv_path = settings.task_dataset_dir(task_name) / (partition + ".csv")
        if not csv_path.exists():
            raise not_found("No dataset partition %r stored for task %r" % (partition, task_name))
        frame = pd.read_csv(csv_path)
        source = "partition:%s" % partition

    if frame is None:
        raise bad_request("Provide a CSV upload or an existing partition name")

    with open_tracker(task_name) as tracker:
        metadata = tracker.GetMetadata()
        target = metadata.get("target_name")
        try:
            tracker.validate_data(frame, target, missing_columns=missing_columns)
        except ValueError as exc:
            raise unprocessable(str(exc))
    return {"task_name": task_name, "source": source, "valid": True}


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
        "dataset_deduced_types": metadata.get("dataset_deduced_types"),
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


@router.get("/{task_name}/target-stats")
def target_stats(task_name: str):
    """Descriptive statistics of the registered target column.

    ``mltrack.Stats`` reads the unpartitioned ``data`` table, which API-registered
    tasks do not have (their rows live in per-partition tables), so this endpoint
    computes the same ``describe()`` summary from the stored partitions instead —
    preferring ``train``, then any registered partition. Cheap enough to run inline.
    """
    if not settings.mltrace_db_path(task_name).exists():
        raise not_found("No dataset registered for task %r" % task_name)

    with open_tracker(task_name) as tracker:
        metadata = tracker.GetMetadata()
        target = metadata.get("target_name")
        splits = tracker.dataset_splits()
        if target is None or not splits:
            raise not_found("No dataset registered for task %r" % task_name)

        preferred = ["train", "validation", "test"]
        ordered = [name for name in preferred if name in splits] + sorted(
            set(splits) - set(preferred)
        )
        frame = None
        partition_used = None
        for name in ordered:
            try:
                candidate = tracker.get_dataframe(name)
            except ValueError:
                continue
            if target in candidate.columns:
                frame, partition_used = candidate, name
                break

    if frame is None:
        raise not_found(
            "Target column %r not found in any registered partition of task %r" % (target, task_name)
        )

    summary = frame[target].describe()
    # ``describe()`` yields float counts; keep the exact integer row count instead.
    stats = {key: _finite(value) for key, value in summary.items() if key != "count"}
    return {
        "task_name": task_name,
        "partition": partition_used,
        "target": target,
        "count": int(frame[target].count()),
        **stats,
    }
