"""Shared FastAPI dependencies and error-mapping helpers."""

import json
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import HTTPException

from SKSurrogate import ModelRegistry, load_bundle, mltrack

from .config import settings


@contextmanager
def open_tracker(task_name):
    """Open a task-scoped ``mltrack`` instance and always close its SQLite handle.

    The splitter persisted for the task (via ``SetCV``) is restored onto the
    tracker automatically by ``mltrack.__init__``, so downstream CV-based
    operations keep using the partitioning method chosen in the UI.
    """
    tracker = mltrack(task_name, db_name=str(settings.mltrace_db_path(task_name)))
    try:
        yield tracker
    finally:
        tracker.close()


def finite_or_none(value):
    """JSON-safe scalar: NaN/inf become ``None`` so JSON responses stay valid."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def not_found(message):
    return HTTPException(status_code=404, detail=message)


def bad_request(message):
    return HTTPException(status_code=400, detail=message)


def unprocessable(message):
    return HTTPException(status_code=422, detail=message)


def resolve_bundle(task_name, *, model_version=None, alias=None, strict_dependencies=False):
    """Load a bundle either from a lifecycle alias in the registry or an unregistered bundle file."""
    if alias is not None:
        try:
            return ModelRegistry(settings.registry_dir).load(
                task_name, alias=alias, strict_dependencies=strict_dependencies
            )
        except KeyError as exc:
            raise not_found(str(exc))
    if model_version is not None:
        bundle_path = settings.task_bundles_dir(task_name) / (model_version + ".bundle")
        if not bundle_path.exists():
            raise not_found("No bundle %r found for task %r" % (model_version, task_name))
        return load_bundle(bundle_path, strict_dependencies=strict_dependencies)
    raise bad_request("Either model_version or alias must be provided")


def append_monitor_record(task_name, model_version, latency_ms, rows, success=True, error=None):
    """Persist one inference-monitor observation as a JSON line, appended atomically."""
    log_path = settings.monitor_log_path(task_name, model_version)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": float(latency_ms),
            "rows": int(rows),
            "success": bool(success),
        }
        if error is not None:
            record["error"] = str(error)
        stream.write(json.dumps(record) + "\n")


def append_monitor_alert(task_name, model_version, alert_type, detail, severity="warning"):
    """Persist a JSON-safe monitoring alert for the Dashboard feed."""
    alert_path = settings.monitor_alerts_path(task_name)
    alert_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": task_name,
        "version": model_version,
        "type": str(alert_type),
        "severity": severity,
        "detail": str(detail),
    }
    with alert_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def append_drift_alerts(task_name, model_version, alerts, thresholds=None):
    """Store report alerts using the stable Dashboard alert shape."""
    if not isinstance(alerts, list):
        raise TypeError("monitoring report alerts must be a list")
    thresholds = thresholds or {}
    for alert in alerts:
        if not isinstance(alert, dict) or not isinstance(alert.get("type"), str):
            raise TypeError("monitoring report alerts must contain a string type")
        details = []
        column = alert.get("column")
        if column is not None:
            details.append(str(column))
        value = alert.get("value")
        if value is not None:
            details.append("value=%s" % value)
        threshold_key = {
            "missingness": "missingness",
            "numeric_drift": "psi",
            "categorical_drift": "categorical",
            "range_drift": "range",
        }.get(alert["type"])
        if threshold_key in thresholds:
            details.append("threshold=%s" % thresholds[threshold_key])
        append_monitor_alert(
            task_name,
            model_version,
            alert["type"],
            ", ".join(details) or alert["type"],
            severity="warning",
        )


def load_monitor_records(task_name, model_version):
    """Return the persisted list of inference-monitor observations for a model version."""
    log_path = settings.monitor_log_path(task_name, model_version)
    if not log_path.exists():
        return []
    with log_path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def bundle_summary(bundle):
    """JSON-safe metadata view of a bundle; never exposes the pickled model object."""
    return {
        "task_name": bundle.task_name,
        "model_version": bundle.model_version,
        "created_at": bundle.created_at,
        "schema": bundle.schema,
        "metrics": bundle.metrics,
        "dependencies": bundle.dependencies,
        "dataset_fingerprint": bundle.dataset_fingerprint,
        "owner": bundle.owner,
        "run_id": bundle.run_id,
        "audit_events": bundle.audit_events,
    }
