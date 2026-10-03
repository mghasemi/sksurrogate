"""Job status endpoints: polling and a lightweight WebSocket progress stream.

API-key enforcement (when ``SKSURROGATE_API_KEY`` is set) happens in the
shared middleware in ``api/main.py``, which also covers this WebSocket route;
the browser UI passes the key as an ``api_key`` query parameter because it
cannot set custom headers on a WS handshake.
"""

import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..config import settings
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


@router.post("/{job_id}/resume")
def resume_job(job_id: str):
    record = job_manager.get(job_id)
    if record is None:
        raise not_found("No job %r found" % job_id)
    if record.get("kind") != "experiment" or record.get("status") != "failed":
        raise HTTPException(status_code=409, detail="Only failed experiment jobs can be resumed")
    resume = record.get("resume")
    if not isinstance(resume, dict) or not isinstance(resume.get("request"), dict):
        raise HTTPException(status_code=409, detail="This experiment does not have saved resume information")
    checkpoint_job_id = resume.get("checkpoint_job_id")
    if not isinstance(checkpoint_job_id, str) or not checkpoint_job_id:
        raise HTTPException(status_code=409, detail="This experiment has no checkpoint reference")
    checkpoint_dir = settings.task_checkpoint_dir(record["task_name"], checkpoint_job_id)
    if not checkpoint_dir.is_dir() or not any(checkpoint_dir.glob("*.eoa")):
        raise HTTPException(status_code=409, detail="No EOA checkpoint is available for this experiment")
    related_jobs = job_manager.list(record["task_name"])
    for related in related_jobs:
        related_resume = related.get("resume")
        if related["job_id"] == job_id or not isinstance(related_resume, dict):
            continue
        if related_resume.get("checkpoint_job_id") == checkpoint_job_id and related["status"] in {
            "queued",
            "running",
            "completed",
        }:
            raise HTTPException(
                status_code=409,
                detail="This checkpoint already has an active or completed resume job",
            )

    from .experiments import resume_experiment

    return resume_experiment(record)


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
