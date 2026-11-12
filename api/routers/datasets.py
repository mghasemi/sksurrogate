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
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from pydantic import BaseModel, Field

from SKSurrogate import (
    STANDARD_CV_SPLITTERS,
    DataPreprocess,
    ModelRegistry,
    cv_param_defs,
    dataset_profile,
    default_cv_spec,
    sensitive_feature_report,
)

from ..config import settings
from ..columns import drop_ignored, normalize_ignored_columns, read_ignored_columns
from ..features import NumericFeatureError, numeric_train_xy
from ..preprocessing import (
    CLIP_METHODS,
    METHODS as IMPUTATION_METHODS,
    ClipError,
    ImputationError,
    clip_partition,
    drop_duplicate_rows,
    duplicate_analysis,
    encoding_preview,
    impute_partition,
    missing_overview,
    target_balance,
)
from ..safety import safe_segment
from ..deps import bad_request, current_principal, finite_or_none as _finite, not_found, open_tracker, unprocessable, task_access

router = APIRouter(prefix="/api/datasets", tags=["datasets"], dependencies=[Depends(task_access)])


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


class IgnoredColumnsRequest(BaseModel):
    """Dataset-level column exclusions: names to drop from every modeling path."""

    columns: list[str] = []


class DatasetProfileRequest(BaseModel):
    partition: str = "train"
    target: str | None = None
    features: list[str] | None = None
    group_by: str | None = None
    max_rows: int = Field(default=5000, ge=1, le=10000)
    max_features: int = Field(default=50, ge=1, le=100)
    max_categories: int = Field(default=10, ge=1, le=50)
    random_state: int | None = 0


@router.post("/{task_name}/profile")
def profile_registered_dataset(task_name: str, body: DatasetProfileRequest):
    """Return bounded descriptive statistics and association diagnostics."""
    if not settings.mltrace_db_path(task_name).exists():
        raise not_found("No dataset registered for task %r" % task_name)
    try:
        with open_tracker(task_name) as tracker:
            frame = tracker.get_dataframe(partition=body.partition)
            metadata = tracker.GetMetadata()
        target = body.target or metadata.get("target_name")
        ignored = read_ignored_columns(task_name)
        if ignored:
            frame = drop_ignored(frame, ignored)
        requested_features = body.features
        if requested_features is not None and ignored:
            requested_features = [column for column in requested_features if column not in ignored]
        report = dataset_profile(
            frame,
            target=target,
            features=requested_features,
            group_by=body.group_by,
            max_rows=body.max_rows,
            max_features=body.max_features,
            max_categories=body.max_categories,
            random_state=body.random_state,
        )
    except ValueError as exc:
        raise bad_request(str(exc))
    return {
        "task_name": task_name,
        "partition": body.partition,
        "ignored_columns": ignored,
        "dataset_fingerprint": metadata.get("dataset_partition_history", {})
        .get(body.partition, {})
        .get("fingerprint"),
        **report,
    }


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


@router.put("/{task_name}/ignored-columns")
def set_ignored_columns(task_name: str, body: IgnoredColumnsRequest):
    """Replace the dataset's ignored-column list.

    Ignored columns remain in the stored partitions (so the choice is reversible
    and fingerprints stay valid) but are dropped by experiments, pipeline
    optimization, baselines, sensitivity/correlation analysis, dataset profiles,
    synthetic generation and monitoring drift. The target column cannot be
    ignored, and unknown names are rejected rather than silently ignored.
    """
    if not settings.mltrace_db_path(task_name).exists():
        raise not_found("No dataset registered for task %r" % task_name)
    with open_tracker(task_name) as tracker:
        metadata = tracker.GetMetadata()
        target = metadata.get("target_name")
        known = metadata.get("dataset_columns") or []
        try:
            normalized = normalize_ignored_columns(body.columns, known, target)
        except ValueError as exc:
            raise bad_request(str(exc))
        tracker.UpdateMetadata({"ignored_columns": normalized})
    effective = [
        column
        for column in known
        if column != target and column not in set(normalized)
    ]
    return {
        "task_name": task_name,
        "target": target,
        "ignored_columns": normalized,
        "effective_features": effective,
        "effective_feature_count": len(effective),
    }


@router.post("/{task_name}/inspect")
async def inspect_dataset(task_name: str, file: UploadFile = File(...)):
    """Preview a CSV's deduced schema without persisting anything (Phase 5.1).

    Runs ``DataPreprocess.deduce_types`` and the bounded sensitive-feature scan
    so the UI can show an editable review table before the user commits the upload
    via ``POST /{task}/register``. The scan reports pattern counts, not cell values.
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
                            binarize_label: Annotated[bool, Form()] = False,
                            ignored_columns: Annotated[str | None, Form()] = None):
    """Upload a CSV, deduce its schema, and register it for a task/partition.

    ``type_overrides`` (JSON object of column -> type) is applied via
    ``DataPreprocess.set_type`` before registration so the user-reviewed types from
    the inspect step win over the automatic deduction; ``binarize_label`` maps the
    target's unique values to 0..n-1 through ``transform_label_bin`` (Phase 5.1).
    ``ignored_columns`` (JSON array of names) is stored in the task metadata: the
    columns stay in the stored CSV but are dropped by every training/analysis
    path (see ``api.columns``).
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
    ignored = []
    if isinstance(ignored_columns, str) and ignored_columns:
        try:
            parsed_ignored = json.loads(ignored_columns)
        except (TypeError, ValueError):
            raise bad_request("ignored_columns must be a JSON array of column names")
        try:
            ignored = normalize_ignored_columns(parsed_ignored, frame.columns, target)
        except ValueError as exc:
            raise bad_request(str(exc))
    dataset_dir = settings.task_dataset_dir(task_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_dir / (safe_segment(partition, "partition") + ".csv")

    weights_stored = []
    with open_tracker(task_name) as tracker:
        tracker.RegisterData(frame, target, partition=partition)
        tracker.UpdateMetadata(
            {"dataset_deduced_types": pivot_types, "ignored_columns": ignored}
        )
        metadata = tracker.GetMetadata()
        # Seed the ``weights`` table with the two cheap dataset-level weightings
        # (feature-target Pearson and feature variance) for the numeric columns.
        # ``mltrack.FeatureWeights`` cannot do this for an API task — it reads an
        # unpartitioned ``data`` table that registered tasks do not have — and
        # without a base row set the "Consensus top features" card stays empty
        # until a weighting exists. Sensitivity runs merge their own column on top.
        try:
            projection = numeric_train_xy(frame, target)
        except NumericFeatureError:
            projection = None
        if projection is not None:
            ignored_set = set(ignored)
            feature_columns = [
                column for column in projection.columns if column not in ignored_set
            ]
            data = pd.DataFrame(projection.X, columns=projection.columns)
            if ignored_set:
                data = pd.DataFrame(data[feature_columns])
            values = pd.Series(projection.y)
            if not feature_columns:
                weights_stored = []
            try:
                if feature_columns:
                    tracker.PersistWeights(
                        {
                            "pearson": [data[column].corr(values) for column in feature_columns],
                            "variance": [data[column].var() for column in feature_columns],
                        },
                        features=feature_columns,
                    )
                    weights_stored = ["pearson", "variance"]
            except Exception:  # pragma: no cover - auxiliary; never fail an upload
                weights_stored = []

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
        "ignored_columns": ignored,
        # Return bounded, count-only PII pattern findings for the uploaded frame.
        "sensitive_scan": sensitive_feature_report(frame),
        # Which weighting columns this registration seeded (numeric features only).
        "weights_stored": weights_stored,
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
        csv_path = settings.task_dataset_dir(task_name) / (safe_segment(partition, "partition") + ".csv")
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
        "ignored_columns": metadata.get("ignored_columns") or [],
        "effective_feature_count": len(
            [
                column
                for column in (metadata.get("dataset_columns") or [])
                if column != metadata.get("target_name")
                and column not in set(metadata.get("ignored_columns") or [])
            ]
        ),
        "partitions": splits,
    }


@router.get("/{task_name}/{partition}/preprocessing")
def preprocessing_overview(
    task_name: str,
    partition: str,
    max_rows: int = 5000,
    max_categories: int = 10,
    preview_limit: int = 20,
):
    """Dataset overview for the preprocessing page: types, missingness, quality, preview.

    Reads the stored partition without dropping ignored columns — the page must
    show them and let the user see why they are excluded — and reuses the bounded
    ``dataset_profile`` diagnostics for distributions, duplicate/constant
    features and missingness patterns. The profile blocks that describe the data
    scientifically are forwarded too (column profiles, target/feature
    associations, rank associations, collinearity), with feature pairs capped at
    ``association_limit`` so the payload stays bounded on wide frames; exact
    target counts come from ``preprocessing.target_balance`` over the full
    partition.
    """
    if not settings.mltrace_db_path(task_name).exists():
        raise not_found("No dataset registered for task %r" % task_name)
    csv_path = settings.task_dataset_dir(task_name) / (safe_segment(partition, "partition") + ".csv")
    if not csv_path.exists():
        raise not_found("No dataset partition %r stored for task %r" % (partition, task_name))
    frame = pd.read_csv(csv_path)
    with open_tracker(task_name) as tracker:
        metadata = tracker.GetMetadata()
    target = metadata.get("target_name")
    ignored = metadata.get("ignored_columns") or []
    deduced = metadata.get("dataset_deduced_types") or {}
    if target not in frame.columns:
        raise bad_request(
            "Partition %r has no target column %r" % (partition, target)
        )

    overview = missing_overview(
        frame, target=target, ignored=ignored, deduced_types=deduced
    )
    active = [column for column in frame.columns if column != target and column not in set(ignored)]
    profile_features = active[:100]
    profile = None
    if profile_features:
        try:
            profile = dataset_profile(
                frame,
                target=target,
                features=profile_features,
                max_rows=max_rows,
                max_features=100,
                max_categories=max_categories,
                random_state=0,
            )
        except ValueError as exc:
            raise bad_request(str(exc))

    # Feature-pair associations grow quadratically; keep the endpoint payload
    # bounded and report exactly how many pairs were dropped instead of letting
    # the browser receive a multi-megabyte response on a 100-column frame.
    association_limit = 200
    all_feature_associations = (profile or {}).get("feature_associations") or []
    target_profile = (profile or {}).get("target")
    target_composition = target_balance(
        frame,
        target=target,
        distribution=(target_profile or {}).get("distribution"),
        max_classes=max_categories,
    )

    head = frame.head(max(preview_limit, 0))
    preview_records = head.astype(object).where(pd.notna(head), None).to_dict(orient="records")
    return {
        "task_name": task_name,
        "partition": partition,
        "target": target,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "ignored_columns": ignored,
        "deduced_types": deduced,
        "missing": overview,
        "quality": (profile or {}).get("data_quality"),
        "missingness_patterns": (profile or {}).get("missingness_patterns"),
        "target_profile": target_profile,
        "target_composition": target_composition,
        "row_sampling": (profile or {}).get("row_sampling"),
        "profile_features_truncated": max(len(active) - len(profile_features), 0),
        "column_profiles": (profile or {}).get("columns") or [],
        "feature_count": (profile or {}).get("feature_count"),
        "feature_limit": (profile or {}).get("feature_limit"),
        "target_associations": (profile or {}).get("target_associations") or [],
        "feature_associations": all_feature_associations[:association_limit],
        "feature_associations_truncated": max(
            len(all_feature_associations) - association_limit, 0
        ),
        "association_limit": association_limit,
        "rank_associations": (profile or {}).get("rank_associations"),
        "multicollinearity": (profile or {}).get("multicollinearity"),
        "group_comparisons": (profile or {}).get("group_comparisons"),
        "preview": preview_records,
        "methods": IMPUTATION_METHODS,
        "clip_methods": CLIP_METHODS,
    }


class ImputeRequest(BaseModel):
    """Materialize an imputed copy of one partition.

    ``preview_only`` computes and returns the comparison report without writing
    anything; otherwise the result is saved as ``output_partition`` and
    registered in the task's partition history.
    """

    partition: str = "train"
    method: str
    columns: list[str] | None = None
    params: dict = {}
    output_partition: str | None = None
    preview_only: bool = False
    drop_missing_target: bool = False


@router.post("/{task_name}/impute")
def impute_dataset(task_name: str, body: ImputeRequest):
    """Impute missing values into a NEW partition; the source partition is untouched."""
    if not settings.mltrace_db_path(task_name).exists():
        raise not_found("No dataset registered for task %r" % task_name)
    csv_path = settings.task_dataset_dir(task_name) / (safe_segment(body.partition, "partition") + ".csv")
    if not csv_path.exists():
        raise not_found("No dataset partition %r stored for task %r" % (body.partition, task_name))
    output = body.output_partition or (body.partition + "_imputed")
    _validate_partition_name(output)
    if output == body.partition:
        raise bad_request("output_partition must differ from the source partition")
    output_path = settings.task_dataset_dir(task_name) / (safe_segment(output, "partition") + ".csv")
    if not body.preview_only and output_path.exists():
        raise HTTPException(
            status_code=409,
            detail="Partition %r already exists; choose a new output partition name" % output,
        )

    frame = pd.read_csv(csv_path)
    dataset_dir = settings.task_dataset_dir(task_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = None
    with open_tracker(task_name) as tracker:
        metadata = tracker.GetMetadata()
        target = metadata.get("target_name")
        ignored = metadata.get("ignored_columns") or []
        try:
            result, report = impute_partition(
                frame,
                target=target,
                ignored=ignored,
                method=body.method,
                columns=body.columns,
                params=body.params,
                drop_missing_target=body.drop_missing_target,
            )
        except ImputationError as exc:
            raise bad_request(str(exc))

        if body.preview_only:
            return {
                "task_name": task_name,
                "partition": body.partition,
                "output_partition": output,
                "written": False,
                **report,
            }

        result.to_csv(output_path, index=False)
        tracker.RegisterData(
            result,
            target,
            partition=output,
            source_uri="imputed:%s" % body.partition,
        )
        splits = tracker.dataset_splits()
        fingerprint = (splits.get(output) or {}).get("fingerprint")
        history = list(metadata.get("imputation_history") or [])
        history.append(
            {
                "source_partition": body.partition,
                "output_partition": output,
                "method": report["method"],
                "columns": report["columns"],
                "params": body.params,
                "rows_before": report["rows_before"],
                "rows_after": report["rows_after"],
                "cells_imputed": report["cells_imputed"],
                "drop_missing_target": bool(body.drop_missing_target),
                "actor": current_principal().username,
            }
        )
        tracker.UpdateMetadata({"imputation_history": history[-50:]})
    try:
        ModelRegistry(settings.registry_dir).register_dataset(
            task_name,
            fingerprint,
            output_path,
            metadata={
                "partition": output,
                "source": "imputed",
                "source_partition": body.partition,
                "method": report["method"],
                "columns": report["columns"],
            },
        )
    except Exception:  # pragma: no cover - lineage is auxiliary
        pass
    try:
        from .. import audit

        audit.record(
            "dataset.impute",
            principal=current_principal(),
            task_name=task_name,
            source_partition=body.partition,
            output_partition=output,
            method=report["method"],
            cells_imputed=report["cells_imputed"],
        )
    except Exception:  # pragma: no cover - audit must not fail the write
        pass
    return {
        "task_name": task_name,
        "partition": body.partition,
        "output_partition": output,
        "written": True,
        "csv_path": str(output_path),
        "dataset_fingerprint": fingerprint,
        **report,
    }


def _write_derived_partition(
    task_name,
    tracker,
    frame,
    target,
    *,
    output,
    source_partition,
    kind,
    history_key,
    history_entry,
    artifact_meta=None,
):
    """Write ``frame`` as ``output`` and record lineage for a derived partition.

    Caller must hold the tracker open; this keeps a single SQLite connection per
    request and makes the metadata update atomic with the partition registration.
    """
    output_path = settings.task_dataset_dir(task_name) / (safe_segment(output, "partition") + ".csv")
    frame.to_csv(output_path, index=False)
    tracker.RegisterData(
        frame,
        target,
        partition=output,
        source_uri="%s:%s" % (kind, source_partition),
    )
    splits = tracker.dataset_splits()
    fingerprint = (splits.get(output) or {}).get("fingerprint")
    metadata = tracker.GetMetadata()
    history = list(metadata.get(history_key) or [])
    history.append(
        {
            "source_partition": source_partition,
            "output_partition": output,
            "actor": current_principal().username,
            **history_entry,
        }
    )
    tracker.UpdateMetadata({history_key: history[-50:]})
    try:
        ModelRegistry(settings.registry_dir).register_dataset(
            task_name,
            fingerprint,
            output_path,
            metadata={
                "partition": output,
                "source": kind,
                "source_partition": source_partition,
                **(artifact_meta or {}),
            },
        )
    except Exception:  # pragma: no cover - lineage is auxiliary
        pass
    return output_path, fingerprint


def _dataset_partition_frame(task_name, partition):
    if not settings.mltrace_db_path(task_name).exists():
        raise not_found("No dataset registered for task %r" % task_name)
    csv_path = settings.task_dataset_dir(task_name) / (safe_segment(partition, "partition") + ".csv")
    if not csv_path.exists():
        raise not_found("No dataset partition %r stored for task %r" % (partition, task_name))
    return csv_path, pd.read_csv(csv_path)


def _derived_output(task_name, partition, requested, default_suffix):
    output = requested or (partition + default_suffix)
    _validate_partition_name(output)
    if output == partition:
        raise bad_request("output_partition must differ from the source partition")
    output_path = settings.task_dataset_dir(task_name) / (safe_segment(output, "partition") + ".csv")
    if output_path.exists():
        raise HTTPException(
            status_code=409,
            detail="Partition %r already exists; choose a new output partition name" % output,
        )
    return output, output_path


class ClipRequest(BaseModel):
    """Winsorize numeric features into a NEW partition."""

    partition: str = "train"
    method: str
    columns: list[str] | None = None
    params: dict = {}
    output_partition: str | None = None
    preview_only: bool = False


@router.post("/{task_name}/clip")
def clip_dataset(task_name: str, body: ClipRequest):
    """Clip numeric features at robust bounds; preview or write a new partition."""
    csv_path, frame = _dataset_partition_frame(task_name, body.partition)
    if not body.preview_only:
        output, _output_path = _derived_output(
            task_name, body.partition, body.output_partition, "_clipped"
        )
    else:
        output = body.output_partition or (body.partition + "_clipped")
        _validate_partition_name(output)
    with open_tracker(task_name) as tracker:
        metadata = tracker.GetMetadata()
        target = metadata.get("target_name")
        ignored = metadata.get("ignored_columns") or []
        try:
            result, report = clip_partition(
                frame,
                target=target,
                ignored=ignored,
                method=body.method,
                columns=body.columns,
                params=body.params,
            )
        except ClipError as exc:
            raise bad_request(str(exc))
        if body.preview_only:
            return {
                "task_name": task_name,
                "partition": body.partition,
                "output_partition": output,
                "written": False,
                **report,
            }
        output_path, fingerprint = _write_derived_partition(
            task_name,
            tracker,
            result,
            target,
            output=output,
            source_partition=body.partition,
            kind="clipped",
            history_key="clipping_history",
            history_entry={
                "method": report["method"],
                "columns": report["columns"],
                "params": report["params"],
                "rows": report["rows"],
                "cells_clipped": report["cells_clipped"],
            },
            artifact_meta={
                "method": report["method"],
                "columns": report["columns"],
            },
        )
    return {
        "task_name": task_name,
        "partition": body.partition,
        "output_partition": output,
        "written": True,
        "csv_path": str(output_path),
        "dataset_fingerprint": fingerprint,
        **report,
    }


class DeduplicateRequest(BaseModel):
    """Drop exact feature-duplicate rows into a NEW partition."""

    partition: str = "train"
    columns: list[str] | None = None
    keep: str = "first"
    output_partition: str | None = None
    preview_only: bool = False


@router.post("/{task_name}/deduplicate")
def deduplicate_dataset(task_name: str, body: DeduplicateRequest):
    """Remove exact duplicate rows; always reports the distributional impact."""
    csv_path, frame = _dataset_partition_frame(task_name, body.partition)
    if not body.preview_only:
        output, _output_path = _derived_output(
            task_name, body.partition, body.output_partition, "_deduplicated"
        )
    else:
        output = body.output_partition or (body.partition + "_deduplicated")
        _validate_partition_name(output)
    with open_tracker(task_name) as tracker:
        metadata = tracker.GetMetadata()
        target = metadata.get("target_name")
        ignored = metadata.get("ignored_columns") or []
        try:
            result, report = drop_duplicate_rows(
                frame,
                target=target,
                ignored=ignored,
                columns=body.columns,
                keep=body.keep,
                max_rows=0,
            )
        except ClipError as exc:
            raise bad_request(str(exc))
        if body.preview_only:
            return {
                "task_name": task_name,
                "partition": body.partition,
                "output_partition": output,
                "written": False,
                **report,
            }
        output_path, fingerprint = _write_derived_partition(
            task_name,
            tracker,
            result,
            target,
            output=output,
            source_partition=body.partition,
            kind="deduplicated",
            history_key="deduplication_history",
            history_entry={
                "features": report["features"],
                "keep": report["keep"],
                "rows_before": report["rows_before"],
                "rows_after": report["rows_after"],
                "rows_removed": report["rows_removed"],
                "conflicting_label_groups": report["conflicting_label_groups"],
            },
            artifact_meta={
                "keep": report["keep"],
                "rows_removed": report["rows_removed"],
            },
        )
    return {
        "task_name": task_name,
        "partition": body.partition,
        "output_partition": output,
        "written": True,
        "csv_path": str(output_path),
        "dataset_fingerprint": fingerprint,
        **report,
    }


@router.get("/{task_name}/{partition}/duplicates")
def duplicate_rows(task_name: str, partition: str, max_rows: int = 20000):
    """Describe exact duplicate rows and what removing them would change."""
    _csv_path, frame = _dataset_partition_frame(task_name, partition)
    with open_tracker(task_name) as tracker:
        metadata = tracker.GetMetadata()
    target = metadata.get("target_name")
    ignored = metadata.get("ignored_columns") or []
    try:
        report = duplicate_analysis(
            frame, target=target, ignored=ignored, max_rows=max_rows
        )
    except ClipError as exc:
        raise bad_request(str(exc))
    return {"task_name": task_name, "partition": partition, **report}


@router.get("/{task_name}/{partition}/encoding-preview")
def encoding_preview_route(
    task_name: str,
    partition: str,
    sample_rows: int = 1000,
    max_outputs: int = 40,
):
    """Show the feature matrix the model pipeline builds from this partition."""
    _csv_path, frame = _dataset_partition_frame(task_name, partition)
    with open_tracker(task_name) as tracker:
        metadata = tracker.GetMetadata()
    target = metadata.get("target_name")
    ignored = metadata.get("ignored_columns") or []
    try:
        report = encoding_preview(
            frame,
            target=target,
            ignored=ignored,
            sample_rows=sample_rows,
            max_outputs=max_outputs,
        )
    except ImputationError as exc:
        raise bad_request(str(exc))
    return {"task_name": task_name, "partition": partition, **report}


@router.get("/{task_name}/{partition}/preview")
def preview_dataset(task_name: str, partition: str, limit: int = 20):
    """Return the first ``limit`` rows of a previously uploaded partition."""
    csv_path = settings.task_dataset_dir(task_name) / (safe_segment(partition, "partition") + ".csv")
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
