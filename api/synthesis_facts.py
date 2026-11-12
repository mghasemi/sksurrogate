"""Bounded, computed facts for LLM-assisted project synthesis (Stage D2).

Everything here reads **stored artifacts** — a registered dataset partition, a
job record, a model bundle, the explanation endpoints, and persisted monitoring
alerts — and reduces it to small, aggregated JSON. No raw dataset rows leave
this module: profiles carry summaries, never records; runs carry configuration
and scalar metrics; explanations carry curve ranges and importance means.

The stage contract is ``AI_LAYER02_D2_LLM_SYNTHESIS_SCOPE.md``; the prompt and
citation validation live in :mod:`api.synthesis`.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from fastapi import HTTPException

from SKSurrogate import dataset_profile, load_bundle

from .config import settings
from .columns import drop_ignored, read_ignored_columns
from .safety import safe_segment
from .deps import open_tracker
from .jobs import job_manager
from .routers.explain import (
    AleRequest,
    ExplainRequest,
    accumulated_local_effects_bundle,
    explain_bundle,
)

#: Top-level run-configuration keys worth citing (the raw search space is not).
_RUN_REQUEST_KEYS = (
    "preset",
    "strategy",
    "length",
    "scoring",
    "random_state",
    "cv",
    "n_splits",
    "max_generations",
)


def _sanitize(value: Any) -> Any:
    """Recursively convert NumPy/NaN values into JSON-safe primitives."""
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def _scalar_metrics(metrics: Any) -> dict:
    """Keep only small scalar entries from a metrics mapping."""
    if not isinstance(metrics, dict):
        return {}
    return {
        str(key): _sanitize(value)
        for key, value in list(metrics.items())[:20]
        if isinstance(value, (int, float, bool, np.integer, np.floating))
    }


def profile_fact(task_name: str, partition: str = "train", max_rows: int = 500) -> dict:
    """Bounded dataset profile: counts, column summaries, top associations."""
    with open_tracker(task_name) as tracker:
        frame = tracker.get_dataframe(partition=partition)
        metadata = tracker.GetMetadata()
    frame = drop_ignored(frame, read_ignored_columns(task_name))
    target = metadata.get("target_name")
    report = dataset_profile(
        frame,
        target=target,
        max_rows=max_rows,
        max_features=20,
        max_categories=8,
        random_state=0,
    )
    target_report = report.get("target") or {}
    columns = []
    for column in (report.get("columns") or [])[:20]:
        columns.append(
            {
                "column": column.get("column"),
                "kind": column.get("kind"),
                "missing_rate": column.get("missing_rate"),
                "unique_count": column.get("unique_count"),
                "constant": column.get("constant"),
                "all_missing": column.get("all_missing"),
            }
        )
    return {
        "id": "S1",
        "kind": "dataset_profile",
        "label": "Bounded dataset profile for partition %r" % partition,
        "data": _sanitize(
            {
                "partition": partition,
                "rows": report.get("rows"),
                "sampled_rows": report.get("sampled_rows"),
                "row_sampling": report.get("row_sampling"),
                "feature_count": report.get("feature_count"),
                "target": {
                    "column": target_report.get("column"),
                    "kind": target_report.get("kind"),
                    "missing_rate": target_report.get("missing_rate"),
                },
                "columns": columns,
                "target_associations": (report.get("target_associations") or [])[:5],
                "data_quality": report.get("data_quality"),
                "interpretation": report.get("interpretation"),
            }
        ),
    }


def run_fact(task_name: str, job_id: str) -> dict:
    """Configuration and scalar results of one completed/queued job."""
    record = job_manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="No job %r found" % job_id)
    if record.get("task_name") != task_name:
        raise HTTPException(
            status_code=400,
            detail="Job %r belongs to task %r, not %r"
            % (job_id, record.get("task_name"), task_name),
        )
    request = record.get("request") or {}
    result = record.get("result") or {}
    result_summary = {}
    if isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, (str, int, float, bool)) and len(str(value)) <= 200:
                result_summary[str(key)] = _sanitize(value)
            elif isinstance(value, dict) and key == "metrics":
                result_summary["metrics"] = _scalar_metrics(value)
            elif isinstance(value, list):
                result_summary[str(key) + "_count"] = len(value)
    return {
        "id": "S2",
        "kind": "run",
        "label": "Background job %s (%s, %s)" % (job_id[:8], record.get("kind"), record.get("status")),
        "data": _sanitize(
            {
                "job_id": job_id,
                "job_kind": record.get("kind"),
                "status": record.get("status"),
                "created_at": record.get("created_at"),
                "request": {
                    key: request[key] for key in _RUN_REQUEST_KEYS if key in request
                },
                "result": result_summary,
            }
        ),
    }


def bundle_fact(task_name: str, model_version: str) -> dict:
    """Bundle-level metadata: metrics, schema shape, fingerprint prefix."""
    path = settings.task_bundles_dir(task_name) / (safe_segment(model_version, "model version") + ".bundle")
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="No bundle %r found for task %r" % (model_version, task_name),
        )
    bundle = load_bundle(path, strict_dependencies=False)
    schema = bundle.schema or {}
    return {
        "id": "S3",
        "kind": "model_bundle",
        "label": "Model bundle %s" % model_version[:8],
        "data": _sanitize(
            {
                "model_version": model_version,
                "created_at": bundle.created_at,
                "metrics": _scalar_metrics(bundle.metrics),
                "feature_count": len(schema),
                "feature_names": list(schema)[:40],
                "preprocessing": bundle.preprocessing is not None,
                "dataset_fingerprint_prefix": str(bundle.dataset_fingerprint or "")[:12] or None,
            }
        ),
    }


def explanation_fact(task_name: str, model_version: str, partition: str = "train") -> dict:
    """Bounded global-attribution and ALE summaries from the explanation endpoints."""
    unavailable: list[str] = []
    importance = []
    ale_ranges = []
    skipped = []
    try:
        explained = explain_bundle(
            task_name,
            model_version,
            ExplainRequest(
                partition=partition,
                max_rows=200,
                n_repeats=2,
                top_k=3,
                stability_samples=1,
            ),
        )
        importance = [
            {"feature": item["feature"], "mean": item["mean"], "std": item["std"]}
            for item in (explained.get("importance") or [])[:5]
        ]
    except HTTPException as exc:
        unavailable.append("importance: %s" % exc.detail)
    try:
        ale = accumulated_local_effects_bundle(
            task_name,
            model_version,
            AleRequest(
                partition=partition,
                max_rows=200,
                n_bins=10,
                top_k=3,
                stability_samples=1,
            ),
        )
        for curve in ale.get("curves") or []:
            values = [value for value in curve.get("accumulated") or [] if value is not None]
            if not values:
                continue
            ale_ranges.append(
                {
                    "feature": curve["feature"],
                    "min": min(values),
                    "max": max(values),
                    "bins": len(curve.get("counts") or []),
                    "rows": int(sum(curve.get("counts") or [])),
                }
            )
        skipped = [
            {"feature": item.get("feature"), "reason": item.get("reason")}
            for item in (ale.get("skipped") or [])
        ]
    except HTTPException as exc:
        unavailable.append("ale: %s" % exc.detail)
    return {
        "id": "S4",
        "kind": "explanations",
        "label": "Explainability summaries for partition %r" % partition,
        "data": _sanitize(
            {
                "partition": partition,
                "importance_top": importance,
                "ale_ranges": ale_ranges[:3],
                "skipped_features": skipped,
                "unavailable": unavailable,
                "interpretation": "Model-response descriptions only; not causal effects.",
            }
        ),
    }


def drift_fact(task_name: str, model_version: str | None, limit: int = 10) -> dict:
    """Recent persisted monitoring alerts, filtered to the model version."""
    path = settings.monitor_alerts_path(task_name)
    alerts: list[dict] = []
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    alert = json.loads(line)
                except ValueError:
                    continue
                if model_version and alert.get("version") not in (None, model_version):
                    continue
                alerts.append(alert)
    alerts = alerts[-limit:][::-1]
    return {
        "id": "S5",
        "kind": "drift_alerts",
        "label": "Recent monitoring alerts (%d)" % len(alerts),
        "data": _sanitize(
            {
                "alert_count": len(alerts),
                "alerts": alerts,
                "interpretation": (
                    "Monitoring alerts describe observed drift or inference failures; "
                    "they are not evidence of a causal change in the data-generating process."
                ),
            }
        ),
    }


def gather_facts(
    task_name: str,
    *,
    model_version: str | None = None,
    job_id: str | None = None,
    partition: str = "train",
    include_explanations: bool = True,
    include_drift: bool = False,
) -> list[dict]:
    """Assemble the bounded source list (S1…Sn) for one synthesis request."""
    facts = [profile_fact(task_name, partition=partition)]
    if job_id:
        facts.append(run_fact(task_name, job_id))
    if model_version:
        facts.append(bundle_fact(task_name, model_version))
        if include_explanations:
            facts.append(explanation_fact(task_name, model_version, partition=partition))
        if include_drift:
            facts.append(drift_fact(task_name, model_version))
    elif include_explanations or include_drift:
        raise HTTPException(
            status_code=400,
            detail="model_version is required for explanation or drift sources",
        )
    return facts
