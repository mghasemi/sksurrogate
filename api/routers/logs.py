"""Read-only log endpoints for the control plane.

Two views over the same per-job store (:mod:`api.logs`):

* ``GET /api/logs`` browses across jobs with severity/kind/text filters,
* ``GET /api/logs/{job_id}`` tails one job,
* ``WS /api/logs/{job_id}/stream`` pushes new entries incrementally.

Severity filtering accepts either a minimum severity (``level=WARNING`` means
WARNING and above) or an explicit subset (``levels=INFO,WARNING``). Entries are
returned newest first from the HTTP endpoints and oldest first on the stream.
"""

import asyncio
import json
import re

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from .. import logs
from ..config import settings
from ..jobs import job_manager

router = APIRouter(prefix="/api/logs", tags=["logs"])

_PROJECT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def _resolve_filters(level, levels):
    try:
        return logs.parse_levels(level=level, levels=levels)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _project_job_ids(project_id):
    """Job ids referenced by a research project's job artifacts.

    The project store is the source of truth for which jobs belong to a project:
    artifacts of kind ``job`` carry the job id in ``artifact_id``. Resolving them
    here keeps project-scoped log queries working for jobs that were linked
    before (or without) the entry-level ``project_id`` stamp.
    """
    if not isinstance(project_id, str) or not _PROJECT_ID_PATTERN.fullmatch(project_id):
        raise HTTPException(status_code=404, detail="No research project %r found" % project_id)
    path = settings.projects_dir / (project_id + ".json")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No research project %r found" % project_id)
    project = json.loads(path.read_text(encoding="utf-8"))
    job_ids = []
    for artifact in project.get("artifacts", []):
        if artifact.get("kind") == "job" and artifact.get("artifact_id"):
            job_ids.append(str(artifact["artifact_id"]))
    return sorted(set(job_ids))


@router.get("")
def list_logs(
    job_id: str = Query(None, description="Restrict to one job"),
    level: str = Query(None, description="Minimum severity, e.g. WARNING"),
    levels: str = Query(None, description="Explicit subset, e.g. INFO,WARNING"),
    kind: str = Query(None, description="Restrict to one job kind"),
    task_name: str = Query(None, description="Restrict to one dataset/task"),
    project_id: str = Query(None, description="Restrict to a research project's jobs"),
    q: str = Query(None, description="Free-text search over message/logger/task"),
    limit: int = Query(200, ge=1, le=2000),
):
    """Browse captured log entries across jobs, filtered by severity, task and project."""
    allowed = _resolve_filters(level, levels)
    job_ids = None
    if project_id and not job_id:
        job_ids = _project_job_ids(project_id)
    if job_ids is not None and not job_ids:
        # A project with no linked jobs has no log entries; avoid scanning all jobs.
        return {"entries": [], "counts": {name: 0 for name in logs.LEVELS}, "total": 0,
                "levels": allowed, "project_id": project_id, "job_count": 0}
    result = logs.store.query(
        job_id=job_id,
        job_ids=job_ids,
        levels=",".join(allowed),
        query_text=q,
        kind=kind,
        task_name=task_name,
        limit=limit,
    )
    result["levels"] = allowed
    if project_id:
        result["project_id"] = project_id
        result["job_count"] = len(job_ids or [])
    return result


@router.get("/{job_id}")
def job_logs(
    job_id: str,
    level: str = Query(None, description="Minimum severity"),
    levels: str = Query(None, description="Explicit subset"),
    task_name: str = Query(None),
    q: str = Query(None),
    limit: int = Query(500, ge=1, le=2000),
    since_seq: int = Query(None, ge=0, description="Only entries after this sequence"),
):
    """Tail one job's entries (newest first)."""
    if job_manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    allowed = _resolve_filters(level, levels)
    result = logs.store.query(
        job_id=job_id,
        levels=",".join(allowed),
        query_text=q,
        task_name=task_name,
        limit=limit,
        since_seq=since_seq,
    )
    result["job_id"] = job_id
    result["levels"] = allowed
    return result


@router.websocket("/{job_id}/stream")
async def stream_job_logs(websocket: WebSocket, job_id: str):
    """Push new log entries for one job until it terminates."""
    await websocket.accept()
    params = websocket.query_params
    try:
        allowed = set(logs.parse_levels(level=params.get("level"), levels=params.get("levels")))
        since_seq = int(params.get("since_seq") or 0)
    except ValueError as exc:
        await websocket.send_json({"error": str(exc)})
        await websocket.close()
        return

    try:
        while True:
            record = job_manager.get(job_id)
            if record is None:
                await websocket.send_json({"error": "job not found"})
                break
            batch = [
                entry
                for entry in logs.store.entries(job_id)
                if int(entry.get("seq") or 0) > since_seq and entry.get("level") in allowed
            ]
            if batch:
                since_seq = max(int(entry.get("seq") or 0) for entry in batch)
                await websocket.send_json(
                    {"entries": batch, "since_seq": since_seq, "status": record["status"]}
                )
            if record["status"] in ("completed", "failed") and not batch:
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return
    await websocket.close()
