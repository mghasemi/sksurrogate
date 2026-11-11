"""Shared FastAPI dependencies and error-mapping helpers."""

import json
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import HTTPException, Request

from SKSurrogate import ModelRegistry, load_bundle, mltrack

from . import acl, auth
from .config import settings
from .safety import contained_path, safe_segment


def current_principal() -> auth.Principal:
    """The identity behind the request being handled (see ``AuthMiddleware``)."""
    return auth.CURRENT_PRINCIPAL.get()


def require_user():
    """FastAPI dependency: an authenticated caller, or HTTP 401.

    While authentication is not enforced (no shared key, no
    ``SKSURROGATE_REQUIRE_AUTH``) every caller is the anonymous principal and the
    dependency passes it through — the open-API behaviour of v1 is preserved, so
    the auth endpoints themselves stay usable in local single-user mode.
    """

    def dependency() -> auth.Principal:
        principal = auth.CURRENT_PRINCIPAL.get()
        if not principal.authenticated and settings.auth_enforced:
            raise HTTPException(status_code=401, detail="Authentication required")
        return principal

    return dependency


def require_admin():
    """FastAPI dependency: an administrator principal, or 401/403."""

    def dependency() -> auth.Principal:
        principal = auth.CURRENT_PRINCIPAL.get()
        if not settings.auth_enforced:
            return principal
        if not principal.authenticated:
            raise HTTPException(status_code=401, detail="Authentication required")
        if not principal.is_admin:
            raise HTTPException(status_code=403, detail="Administrator privileges required")
        return principal

    return dependency


# --------------------------------------------------------------------------- #
# Ownership/sharing enforcement
# --------------------------------------------------------------------------- #
# Router-level dependencies: every task-scoped router declares
# ``dependencies=[Depends(task_access)]`` once, which covers all of its routes
# (including future ones) instead of a per-endpoint audit. The dependency
# resolves the identifier from the matched path and derives the required role
# from the HTTP method: reads need ``viewer``, anything that mutates state needs
# ``editor``. Sharing, deletion and registry promotion stay in the endpoints
# that own those semantics.

_EDIT_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _minimum_role(request: Request) -> str:
    return "editor" if request.method.upper() in _EDIT_METHODS else "viewer"


def task_access(request: Request) -> None:
    """Require view/edit access to the ``task_name`` in the request path.

    A successful *write* by an authenticated user on a task that has no owner
    yet claims it. That single rule replaces per-endpoint bookkeeping: whichever
    authenticated user first creates data under a task name owns it, and later
    writers need an editor grant.
    """
    task_name = request.path_params.get("task_name")
    if task_name is None:
        return
    minimum = _minimum_role(request)
    acl.require_role(acl.TASK, str(task_name), minimum)
    if minimum == "editor":
        acl.claim(acl.TASK, str(task_name))


def project_access(request: Request) -> None:
    """Require view/edit access to the ``project_id`` in the request path."""
    project_id = request.path_params.get("project_id")
    if project_id is None:
        return
    minimum = _minimum_role(request)
    acl.require_role(acl.PROJECT, str(project_id), minimum)
    if minimum == "editor":
        acl.claim(acl.PROJECT, str(project_id))


def leg_for_job(job_id: str):
    """The task or project a job belongs to, as ``(kind, resource_id)``.

    Jobs are task-scoped by construction (``job_manager.list(task_name)``); the
    project link, when present, is recorded as an artifact *after* the job
    finishes, so the task is the authorization source of truth.
    """
    import json

    path = settings.job_record_path(job_id)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    task_name = record.get("task_name")
    if isinstance(task_name, str) and task_name:
        return (acl.TASK, task_name)
    return None


def job_ws_allowed(job_id: str, minimum: str = "viewer") -> bool:
    """WebSocket-safe job authorization check (no ``Request`` object there)."""
    leg = leg_for_job(job_id)
    if leg is None:
        return True  # unknown job: the handler reports "job not found"
    return acl.has_role(leg[0], leg[1], minimum)


def job_access(request: Request) -> None:
    """Require view/edit access to the task owning the ``job_id`` in the path."""
    job_id = request.path_params.get("job_id")
    if job_id is None:
        return
    leg = leg_for_job(str(job_id))
    if leg is None:
        # Unknown job: let the endpoint answer 404 (no existence leak either way).
        return
    acl.require_role(leg[0], leg[1], _minimum_role(request))


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
    """Load a bundle either from a lifecycle alias in the registry or an unregistered bundle file.

    Both the task name and the model version are caller-supplied path segments:
    they are validated (and the resolved path proven to stay inside the task's
    bundle directory) so a request can never load a bundle from elsewhere on
    the host.
    """
    task_name = safe_segment(task_name, "task name")
    if alias is not None:
        try:
            return ModelRegistry(settings.registry_dir).load(
                task_name, alias=alias, strict_dependencies=strict_dependencies
            )
        except KeyError as exc:
            raise not_found(str(exc))
    if model_version is not None:
        bundle_path = contained_path(
            settings.task_bundles_dir(task_name),
            model_version,
            ".bundle",
            field="model version",
        )
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
