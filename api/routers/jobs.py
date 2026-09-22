"""Job status endpoints: polling and a lightweight WebSocket progress stream."""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..deps import not_found
from ..jobs import job_manager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
def list_jobs(task_name: str | None = None):
    return {"jobs": job_manager.list(task_name)}


@router.get("/{job_id}")
def get_job(job_id: str):
    record = job_manager.get(job_id)
    if record is None:
        raise not_found("No job %r found" % job_id)
    return record


@router.websocket("/{job_id}/stream")
async def stream_job(websocket: WebSocket, job_id: str):
    """Push job status every 500ms until it reaches a terminal state."""
    await websocket.accept()
    try:
        last_status = None
        while True:
            record = job_manager.get(job_id)
            if record is None:
                await websocket.send_json({"error": "job not found"})
                break
            if record["status"] != last_status:
                await websocket.send_json(record)
                last_status = record["status"]
            if record["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return
    await websocket.close()
