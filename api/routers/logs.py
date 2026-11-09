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

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from .. import logs
from ..jobs import job_manager

router = APIRouter(prefix="/api/logs", tags=["logs"])


def _resolve_filters(level, levels):
    try:
        return logs.parse_levels(level=level, levels=levels)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("")
def list_logs(
    job_id: str = Query(None, description="Restrict to one job"),
    level: str = Query(None, description="Minimum severity, e.g. WARNING"),
    levels: str = Query(None, description="Explicit subset, e.g. INFO,WARNING"),
    kind: str = Query(None, description="Restrict to one job kind"),
    q: str = Query(None, description="Free-text search over message/logger/task"),
    limit: int = Query(200, ge=1, le=2000),
):
    """Browse captured log entries across jobs, filtered by severity and text."""
    allowed = _resolve_filters(level, levels)
    result = logs.store.query(
        job_id=job_id,
        levels=",".join(allowed),
        query_text=q,
        kind=kind,
        limit=limit,
    )
    result["levels"] = allowed
    return result


@router.get("/{job_id}")
def job_logs(
    job_id: str,
    level: str = Query(None, description="Minimum severity"),
    levels: str = Query(None, description="Explicit subset"),
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
