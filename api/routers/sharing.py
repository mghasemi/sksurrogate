"""Ownership and sharing endpoints.

The ACL itself lives in :mod:`api.acl`; this module is its HTTP surface:

* ``GET    /api/access/tasks``                     — my tasks and my role on each
* ``GET    /api/access/tasks/{task_name}``         — shares for one task
* ``POST   /api/access/tasks/{task_name}/shares``  — share with a user
* ``DELETE /api/access/tasks/{task_name}/shares/{user_id}``
* the same four for ``projects``
* ``POST   /api/access/adopt``                     — admin: adopt unowned material

Only an ``owner`` (or an administrator) may change the share list. Removing a
grant is idempotent, but demoting the last owner is refused: a resource nobody
owns would fall back to "first writer owns it" and could be stolen.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .. import acl, auth
from ..config import settings
from ..deps import bad_request, not_found, require_admin, require_user

router = APIRouter(prefix="/api/access", tags=["access"])


class ShareRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    role: str = Field(default="viewer", description="viewer | editor | owner")


class AdoptRequest(BaseModel):
    kind: str = Field(default=acl.TASK, description="task | project")
    user_id: Optional[int] = Field(
        default=None, description="account that takes ownership (defaults to the caller)"
    )


def _resolve_user_id(username: str) -> int:
    user = auth.store().get_user_by_username(username)
    if user is None:
        raise not_found("No such user: %r" % username)
    if not user["is_active"]:
        raise bad_request("user %r is disabled" % username)
    return user["user_id"]


def _share(kind: str, resource_id: str, body: ShareRequest, principal) -> dict:
    acl.require_ownership(kind, resource_id, principal)
    try:
        user_id = _resolve_user_id(body.username)
        acl.store().set_role(kind, resource_id, user_id, body.role, granted_by=principal.user_id)
    except acl.AclError as exc:
        raise bad_request(str(exc))
    return acl.shares_payload(kind, resource_id)


def _unshare(kind: str, resource_id: str, user_id: int, principal) -> dict:
    acl.require_ownership(kind, resource_id, principal)
    current = acl.store().role_of(kind, resource_id, user_id)
    if current is None:
        raise not_found("User %d has no grant on %s %r" % (user_id, kind, resource_id))
    if current == "owner":
        owners = [g for g in acl.store().list_grants(kind, resource_id) if g["role"] == "owner"]
        if len(owners) <= 1:
            raise bad_request("cannot remove the last owner; grant another owner first")
    acl.store().remove_role(kind, resource_id, user_id)
    return acl.shares_payload(kind, resource_id)


def _payload(kind: str, resource_id: str, principal) -> dict:
    role = acl.require_view(kind, resource_id, principal)
    payload = acl.shares_payload(kind, resource_id)
    payload["role"] = role
    return payload


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #


@router.get("/tasks")
def list_my_tasks(principal: auth.Principal = Depends(require_user())):
    """Every task the caller can see, with their role — the sidebar's source."""
    if not settings.auth_enforced:
        return {"tasks": [], "auth_enforced": False}
    grants = acl.store().grants_for_user(principal.user_id, acl.TASK) if principal.user_id else []
    by_id = {grant["resource_id"]: grant["role"] for grant in grants}
    shares = []
    for resource_id, role in sorted(by_id.items()):
        shares.append({"task_name": resource_id, "role": role})
    return {"tasks": shares, "auth_enforced": True}


@router.get("/tasks/{task_name}")
def task_shares(task_name: str, principal: auth.Principal = Depends(require_user())):
    return _payload(acl.TASK, task_name, principal)


@router.post("/tasks/{task_name}/shares")
def share_task(task_name: str, body: ShareRequest, principal: auth.Principal = Depends(require_user())):
    return _share(acl.TASK, task_name, body, principal)


@router.delete("/tasks/{task_name}/shares/{user_id}")
def unshare_task(task_name: str, user_id: int, principal: auth.Principal = Depends(require_user())):
    return _unshare(acl.TASK, task_name, user_id, principal)


# --------------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------------- #


@router.get("/projects")
def list_my_projects(principal: auth.Principal = Depends(require_user())):
    if not settings.auth_enforced:
        return {"projects": [], "auth_enforced": False}
    grants = acl.store().grants_for_user(principal.user_id, acl.PROJECT) if principal.user_id else []
    by_id = {grant["resource_id"]: grant["role"] for grant in grants}
    return {
        "projects": [
            {"project_id": resource_id, "role": role}
            for resource_id, role in sorted(by_id.items())
        ],
        "auth_enforced": True,
    }


@router.get("/projects/{project_id}")
def project_share_list(project_id: str, principal: auth.Principal = Depends(require_user())):
    return _payload(acl.PROJECT, project_id, principal)


@router.post("/projects/{project_id}/shares")
def share_project(project_id: str, body: ShareRequest, principal: auth.Principal = Depends(require_user())):
    return _share(acl.PROJECT, project_id, body, principal)


@router.delete("/projects/{project_id}/shares/{user_id}")
def unshare_project(project_id: str, user_id: int, principal: auth.Principal = Depends(require_user())):
    return _unshare(acl.PROJECT, project_id, user_id, principal)


# --------------------------------------------------------------------------- #
# Migration helper
# --------------------------------------------------------------------------- #


@router.post("/adopt")
def adopt_orphans(body: AdoptRequest, principal: auth.Principal = Depends(require_admin())):
    """Give one account ownership of every resource of ``kind`` that has none.

    Run once when a pre-auth deployment switches to authenticated mode: the
    existing datasets, bundles and projects predate accounts, so somebody has to
    own them. Idempotent — resources that already have an owner are skipped.
    """
    if body.kind not in acl.KINDS:
        raise bad_request("kind must be one of: %s" % ", ".join(acl.KINDS))
    target_id = body.user_id or principal.user_id
    if target_id is None:
        raise bad_request("adoption needs an account (the shared API key owns nothing)")
    target = auth.store().get_user(target_id)
    if target is None:
        raise not_found("No such user: %r" % target_id)
    if body.kind == acl.TASK:
        candidates = {
            path.name for path in settings.datasets_dir.glob("*") if path.is_dir()
        } | {path.name for path in settings.bundles_dir.glob("*") if path.is_dir()}
        candidates |= {path.stem for path in settings.mltrace_dir.glob("*.db")}
    else:
        candidates = {path.stem for path in settings.projects_dir.glob("*.json")}
    target_principal = auth.Principal(target["user_id"], target["username"], target["is_admin"], "session")
    adopted = acl.store().adopt_orphans(body.kind, target_principal, candidates)
    return {
        "kind": body.kind,
        "username": target["username"],
        "adopted": adopted,
        "count": len(adopted),
    }
