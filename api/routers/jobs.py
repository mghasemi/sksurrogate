"""Job status endpoints: polling, a WebSocket progress stream, and control.

Authentication happens in the shared middleware in ``api/main.py``, which also
covers this WebSocket route; the browser UI passes a key as an
``api_key`` query parameter because it cannot set custom headers on a WS
handshake.

Authorization is per task (see ``api.acl``): the HTTP routes carry
``Depends(job_access)``, which resolves the job record's ``task_name`` and
requires ``viewer``/``editor`` on it, and the list endpoint filters records down
to tasks the caller can see. The WebSocket handler cannot use a
``Request``-typed dependency, so it checks :func:`api.deps.job_ws_allowed`
before accepting the connection.

Liveness comes from :class:`api.jobs.JobManager`, never from the JSON record: a
``running`` record written by a process that has since died is reported as
``interrupted`` and never appears in ``/api/jobs/active``.
"""

import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel

from .. import acl, control
from ..config import settings
from ..deps import current_principal, job_access, job_ws_allowed, not_found, unprocessable
from ..jobs import job_manager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

#: Statuses that mean "the worker pool still owns this job".
ACTIVE_STATUSES = ("queued", "running")

#: Statuses a resume may start from; each needs a saved request and a checkpoint.
RESUMABLE_STATUSES = ("failed", "paused", "interrupted")

#: Job kinds whose work can be resumed from an EOA checkpoint.
RESUMABLE_KINDS = ("experiment", "optimize_pipeline")

#: Fields a job list may expose for the polling indicator. Deliberately drops
#: ``result`` (a finished experiment carries its full evaluation history) and
#: ``resume`` (the whole request body), so a 4-second poll stays tiny.
SUMMARY_FIELDS = (
    "job_id",
    "task_name",
    "display_name",
    "kind",
    "backend",
    "status",
    "created_at",
    "updated_at",
    "heartbeat_at",
    "log_counts",
)


class StopRequest(BaseModel):
    """``pause`` keeps the checkpoint for a later resume; ``cancel`` is final."""

    mode: str = "pause"


class RenameRequest(BaseModel):
    """Replace the job's human label; ``None`` clears it back to the auto name."""

    label: str | None = None


@router.get("")
def list_jobs(task_name: str | None = None):
    """List job records the caller may see (optionally restricted to one task)."""
    jobs = job_manager.list(task_name)
    names = {record.get("task_name") for record in jobs if isinstance(record.get("task_name"), str)}
    visible = set(acl.visible_ids(acl.TASK, names))
    return {"jobs": [record for record in jobs if record.get("task_name") in visible]}


# NOTE: this route must stay declared before ``/{job_id}`` or FastAPI would
# match "active" as a job id.
@router.get("/active")
def list_active_jobs(task_name: str | None = None):
    """Summaries of jobs that are *actually* running in this API process.

    A persisted status is not trusted here: the id must be registered as live by
    the job manager, so a crashed run (or a record copied from another host)
    cannot keep the UI's activity badge alive.
    """
    jobs = job_manager.list(task_name)
    names = {record.get("task_name") for record in jobs if isinstance(record.get("task_name"), str)}
    visible = set(acl.visible_ids(acl.TASK, names))
    active = [
        {field: record.get(field) for field in SUMMARY_FIELDS}
        for record in jobs
        if record.get("status") in ACTIVE_STATUSES
        and record.get("task_name") in visible
        and job_manager.is_live(record.get("job_id"))
    ]
    return {"jobs": active, "count": len(active)}


@router.get("/{job_id}", dependencies=[Depends(job_access)])
def get_job(job_id: str):
    record = job_manager.get(job_id)
    if record is None:
        raise not_found("No job %r found" % job_id)
    if record.get("status") in ACTIVE_STATUSES and not job_manager.is_live(job_id):
        # Self-heal a record left behind by a process that died mid-job.
        record = job_manager.mark_interrupted(job_id) or record
    return record


@router.patch("/{job_id}", dependencies=[Depends(job_access)])
def rename_job(job_id: str, body: RenameRequest):
    """Give a job a human-readable label (or clear it back to the auto name).

    The label is display-only: ``job_id`` stays the identity for routes, project
    artifacts and checkpoints. Naming is a mutation, so it needs ``editor`` on
    the owning task (``job_access`` derives it from the method).
    """
    record = job_manager.rename(job_id, body.label, actor=current_principal().username)
    if record is None:
        raise not_found("No job %r found" % job_id)
    return record


@router.post("/{job_id}/stop", dependencies=[Depends(job_access)])
def stop_job(job_id: str, body: StopRequest | None = None):
    """Ask a live job to stop at its next safe point.

    ``pause`` keeps the checkpoint and leaves the job resumable; ``cancel``
    abandons the run. Stopping is cooperative — a job that is between candidates
    stops promptly, while a job already finishing is unaffected (409). Job kinds
    without a checkpoint can only be cancelled.
    """
    mode = (body.mode if body is not None else "pause") or "pause"
    if mode not in control.STOP_MODES:
        raise unprocessable("mode must be one of %s" % ", ".join(control.STOP_MODES))
    record = job_manager.get(job_id)
    if record is None:
        raise not_found("No job %r found" % job_id)
    if not job_manager.is_live(job_id):
        raise HTTPException(
            status_code=409,
            detail="Job is not running in this API process (it finished, or a restart interrupted it)",
        )
    try:
        updated = job_manager.request_stop(
            job_id, mode, requested_by=current_principal().username
        )
    except KeyError:
        raise HTTPException(status_code=409, detail="Job is not running in this API process")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return updated or record


@router.post("/{job_id}/resume", dependencies=[Depends(job_access)])
def resume_job(job_id: str):
    record = job_manager.get(job_id)
    if record is None:
        raise not_found("No job %r found" % job_id)
    if job_manager.is_live(job_id):
        raise HTTPException(status_code=409, detail="Job is still running; stop it before resuming")
    if record.get("kind") not in RESUMABLE_KINDS or record.get("status") not in RESUMABLE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail="Only failed, paused or interrupted experiment / pipeline jobs can be resumed",
        )
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
        if related_resume.get("checkpoint_job_id") != checkpoint_job_id:
            continue
        # A live job holds the checkpoint; a completed one proves it produced a
        # result. A stale ``running`` record is not live and must not block.
        if related.get("status") == "completed" or job_manager.is_live(related["job_id"]):
            raise HTTPException(
                status_code=409,
                detail="This checkpoint already has an active or completed resume job",
            )

    from .experiments import resume_experiment, resume_optimize_pipeline

    if record.get("kind") == "optimize_pipeline":
        return resume_optimize_pipeline(record)
    return resume_experiment(record)


@router.websocket("/{job_id}/stream")
async def stream_job(websocket: WebSocket, job_id: str):
    """Push job status every 500ms until it reaches a terminal state."""
    if not job_ws_allowed(job_id):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        last_status = None
        while True:
            record = job_manager.get(job_id)
            if record is None:
                await websocket.send_json({"error": "job not found"})
                break
            if record["status"] in ACTIVE_STATUSES and not job_manager.is_live(job_id):
                record = job_manager.mark_interrupted(job_id) or record
            if record["status"] != last_status:
                await websocket.send_json(record)
                last_status = record["status"]
            if record["status"] not in ACTIVE_STATUSES:
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return
    await websocket.close()
