"""Compute-budget settings endpoints.

The API runs background jobs in-process, so the thread/CPU budget is server
state, not a browser preference: values set here are what every subsequently
submitted job obeys (see :mod:`api.compute`). Reads are open to any caller the
rest of the API already serves; writes need an administrator once
authentication is enforced, matching the other server-wide configuration.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import audit, compute, reset
from ..config import settings
from ..deps import bad_request, require_admin, require_user, unprocessable

router = APIRouter(prefix="/api/settings", tags=["settings"])


class ComputeSettings(BaseModel):
    """Complete budget document; a PUT replaces all five values."""

    job_workers: int
    job_threads: int
    blas_threads: int
    dask_workers: int
    dask_threads: int


@router.get("/compute")
def get_compute_settings(principal=Depends(require_user())):
    """Effective compute budget, the host's core count and the defaults."""
    return compute.snapshot()


@router.put("/compute")
def put_compute_settings(body: ComputeSettings, principal=Depends(require_admin())):
    """Persist a new budget; rejects anything that does not fit the core count."""
    try:
        compute.save(body.model_dump())
    except ValueError as exc:
        raise unprocessable(str(exc)) from exc
    snapshot = compute.snapshot()
    audit.record(
        "compute.settings",
        principal=principal,
        values=snapshot["values"],
        cores=snapshot["cores"],
    )
    return snapshot


class ResetRequest(BaseModel):
    """Typed confirmation for the destructive per-user data reset.

    ``confirm`` must be exactly ``reset <username>`` for a signed-in account, or
    ``reset everything`` on an open (single anonymous user) deployment. The
    phrase is checked server-side so an accidental click without typing it
    cannot delete anything.
    """

    confirm: str
    include_shared: bool = False


@router.post("/reset")
def reset_my_data(body: ResetRequest, principal=Depends(require_user())):
    """Erase every task/project the caller owns, with all derived artifacts.

    Never touches accounts, sessions or API keys. Resources shared with other
    users are skipped unless ``include_shared`` is explicitly set. Running jobs
    on a scoped task answer **409** — stop them first.
    """
    expected = (
        "reset everything"
        if not settings.auth_enforced
        else "reset %s" % principal.username
    )
    if body.confirm.strip() != expected:
        raise bad_request(
            "Type %r exactly to confirm the reset" % expected
        )
    if settings.auth_enforced and principal.user_id is None:
        raise bad_request(
            "A per-user reset requires a signed-in account, not the shared API key"
        )
    try:
        return reset.reset_user_data(
            principal, include_shared=body.include_shared
        )
    except reset.LiveJobsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PermissionError as exc:
        raise bad_request(str(exc)) from exc
