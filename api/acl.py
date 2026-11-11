"""Ownership and sharing (ACL) for tasks and projects.

Storage in this control plane is keyed by ``task_name``: datasets, bundles,
registry aliases, monitoring records, checkpoints and jobs all hang off a task
directory (see :class:`api.config.Settings`). That makes the **task** the
natural unit of access control — sharing a task shares the data, the models and
every artifact derived from them, which is both what collaborators expect and
one check instead of an audit of 86 routes. Projects are the second unit: they
own themselves and live in their own directory tree.

Roles are ordered — ``owner`` > ``editor`` > ``viewer``:

* ``viewer`` — read anything under the resource (list, inspect, download).
* ``editor`` — additionally register datasets, train/save bundles, promote in
  the registry, submit jobs.
* ``owner`` — additionally share/unshare and delete the resource.

Admins hold ``owner`` on everything. When authentication is not enforced there
is exactly one anonymous user, so the ACL is bypassed entirely and no grants are
written; enabling auth later starts from "everything is unowned" and the first
administrator can adopt the existing material (:meth:`AclStore.adopt_orphans`).
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from fastapi import HTTPException

from . import auth
from .auth import Principal
from .config import settings

#: Resource families that carry ownership.
TASK = "task"
PROJECT = "project"
KINDS = (TASK, PROJECT)

#: Ordered roles: index position is the privilege level.
ROLES = ("viewer", "editor", "owner")
_ROLE_RANK = {role: rank for rank, role in enumerate(ROLES)}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS grants (
    kind         TEXT NOT NULL,
    resource_id  TEXT NOT NULL,
    user_id      INTEGER NOT NULL,
    role         TEXT NOT NULL,
    granted_by   INTEGER,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (kind, resource_id, user_id)
);
CREATE INDEX IF NOT EXISTS grants_user ON grants(user_id);
"""


class AclError(ValueError):
    """Raised for invalid ACL input (unknown kind/role, unknown user)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AclStore:
    """SQLite grant table: ``(kind, resource_id, user_id) -> role``."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    @contextmanager
    def _connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialise(self) -> "AclStore":
        with self._lock, self._connection() as connection:
            connection.executescript(_SCHEMA)
        return self

    # -- grants ------------------------------------------------------------ #

    def set_role(self, kind: str, resource_id: str, user_id: int, role: str, granted_by: Optional[int] = None) -> dict:
        kind, role = _validate(kind, role)
        resource_id = _resource_id(resource_id)
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO grants (kind, resource_id, user_id, role, granted_by, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(kind, resource_id, user_id) DO UPDATE SET role = excluded.role",
                (kind, resource_id, user_id, role, granted_by, _now()),
            )
        return {"kind": kind, "resource_id": resource_id, "user_id": user_id, "role": role}

    def remove_role(self, kind: str, resource_id: str, user_id: int) -> bool:
        kind, _ = _validate(kind, "viewer")
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM grants WHERE kind = ? AND resource_id = ? AND user_id = ?",
                (kind, _resource_id(resource_id), user_id),
            )
            return cursor.rowcount > 0

    def role_of(self, kind: str, resource_id: str, user_id: Optional[int]) -> Optional[str]:
        if user_id is None:
            return None
        with self._connection() as connection:
            row = connection.execute(
                "SELECT role FROM grants WHERE kind = ? AND resource_id = ? AND user_id = ?",
                (kind, _resource_id(resource_id), user_id),
            ).fetchone()
        return row["role"] if row else None

    def has_owner(self, kind: str, resource_id: str) -> bool:
        kind, _ = _validate(kind, "viewer")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM grants WHERE kind = ? AND resource_id = ? AND role = 'owner' LIMIT 1",
                (kind, _resource_id(resource_id)),
            ).fetchone()
        return row is not None

    def list_grants(self, kind: str, resource_id: str) -> list[dict]:
        kind, _ = _validate(kind, "viewer")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM grants WHERE kind = ? AND resource_id = ? ORDER BY role DESC, user_id",
                (kind, _resource_id(resource_id)),
            ).fetchall()
        return [dict(row) for row in rows]

    def grants_for_user(self, user_id: int, kind: Optional[str] = None) -> list[dict]:
        query = "SELECT * FROM grants WHERE user_id = ?"
        params: tuple = (user_id,)
        if kind is not None:
            query += " AND kind = ?"
            params = (user_id, _validate(kind, "viewer")[0])
        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def resource_ids_for_user(self, user_id: int, kind: str) -> set[str]:
        return {grant["resource_id"] for grant in self.grants_for_user(user_id, kind)}

    # -- lifecycle --------------------------------------------------------- #

    def claim(self, kind: str, resource_id: str, principal: Principal) -> dict:
        """Make ``principal`` the owner of an unowned resource; no-op if owned.

        Called on creation paths (registering a dataset into a new task, saving
        a bundle, creating a project). An existing owner is never overwritten,
        so a rename or a re-upload cannot steal someone else's task.
        """
        kind, _ = _validate(kind, "owner")
        resource_id = _resource_id(resource_id)
        if principal.user_id is None:
            return {"kind": kind, "resource_id": resource_id, "user_id": None, "role": "owner"}
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO grants (kind, resource_id, user_id, role, granted_by, created_at) "
                "SELECT ?, ?, ?, 'owner', ?, ? "
                "WHERE NOT EXISTS ("
                "  SELECT 1 FROM grants WHERE kind = ? AND resource_id = ? AND role = 'owner'"
                ")",
                (
                    kind,
                    resource_id,
                    principal.user_id,
                    principal.user_id,
                    _now(),
                    kind,
                    resource_id,
                ),
            )
        return {"kind": kind, "resource_id": resource_id, "user_id": principal.user_id, "role": "owner"}

    def remove_resource(self, kind: str, resource_id: str) -> int:
        kind, _ = _validate(kind, "viewer")
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM grants WHERE kind = ? AND resource_id = ?", (kind, _resource_id(resource_id))
            )
            return cursor.rowcount

    def adopt_orphans(self, kind: str, principal: Principal, resource_ids: Iterable[str]) -> list[str]:
        """Give ``principal`` ownership of every listed resource that has no owner.

        Run by an administrator after switching a single-user deployment to
        authenticated mode: the data predates accounts, so somebody has to own
        it. Idempotent — resources that already have an owner are skipped.
        """
        kind, _ = _validate(kind, "owner")
        adopted = []
        for resource_id in resource_ids:
            resource_id = _resource_id(resource_id)
            if self.has_owner(kind, resource_id):
                continue
            self.claim(kind, resource_id, principal)
            adopted.append(resource_id)
        return sorted(adopted)


def _validate(kind: str, role: str) -> tuple[str, str]:
    if kind not in KINDS:
        raise AclError("unknown resource kind %r (expected one of %s)" % (kind, ", ".join(KINDS)))
    if role not in _ROLE_RANK:
        raise AclError("unknown role %r (expected one of %s)" % (role, ", ".join(ROLES)))
    return kind, role


def _resource_id(resource_id: str) -> str:
    if not isinstance(resource_id, str) or not resource_id.strip():
        raise AclError("resource id must be a non-empty string")
    return resource_id


# --------------------------------------------------------------------------- #
# Module-level store and request-facing helpers
# --------------------------------------------------------------------------- #

_STORE: Optional[AclStore] = None
_STORE_PATH: Optional[Path] = None


def store() -> AclStore:
    """Process-wide ACL store for the configured API home (re-created on root change)."""
    global _STORE, _STORE_PATH
    path = Path(settings.acl_db_path)
    if _STORE is None or _STORE_PATH != path:
        _STORE = AclStore(path).initialise()
        _STORE_PATH = path
    return _STORE


def reset_store() -> None:
    global _STORE, _STORE_PATH
    _STORE = None
    _STORE_PATH = None


def effective_role(kind: str, resource_id: str, principal: Optional[Principal] = None) -> Optional[str]:
    """The caller's role on a resource, or ``None`` when they have no access.

    * auth not enforced — everybody is ``owner`` (v1 single-user behaviour);
    * admin / the static break-glass key — ``owner`` everywhere;
    * an unowned resource — ``owner``: legacy material has no owner until an
      administrator adopts it, and locking users out of their own pre-auth data
      would be worse than the (temporary) lack of segregation;
    * otherwise the stored grant.
    """
    principal = principal or auth.CURRENT_PRINCIPAL.get()
    if not settings.auth_enforced:
        return "owner"
    if principal.is_admin:
        return "owner"
    acl = store()
    role = acl.role_of(kind, resource_id, principal.user_id)
    if role is not None:
        return role
    if not acl.has_owner(kind, resource_id):
        return "owner"
    return None


def has_role(kind: str, resource_id: str, minimum: str, principal: Optional[Principal] = None) -> bool:
    role = effective_role(kind, resource_id, principal)
    return role is not None and _ROLE_RANK[role] >= _ROLE_RANK[minimum]


def require_role(kind: str, resource_id: str, minimum: str = "viewer", principal: Optional[Principal] = None) -> str:
    """Assert at least ``minimum`` access, answering 404/403 like a REST API should.

    No access answers **404**, not 403: telling a stranger that a task exists
    but belongs to somebody else leaks the namespace. Insufficient role on a
    resource the caller *can* see answers 403 with the role they would need.
    """
    role = effective_role(kind, resource_id, principal)
    if role is None:
        raise HTTPException(status_code=404, detail="No such %s: %r" % (kind, resource_id))
    if _ROLE_RANK[role] < _ROLE_RANK[minimum]:
        raise HTTPException(
            status_code=403,
            detail="%s access is required on %s %r (you have %s)"
            % (minimum, kind, resource_id, role),
        )
    return role


def require_view(kind: str, resource_id: str, principal: Optional[Principal] = None) -> str:
    return require_role(kind, resource_id, "viewer", principal)


def require_edit(kind: str, resource_id: str, principal: Optional[Principal] = None) -> str:
    return require_role(kind, resource_id, "editor", principal)


def require_ownership(kind: str, resource_id: str, principal: Optional[Principal] = None) -> str:
    return require_role(kind, resource_id, "owner", principal)


def claim(kind: str, resource_id: str, principal: Optional[Principal] = None) -> None:
    """Record ownership on creation (no-op when auth is off or unowned already)."""
    principal = principal or auth.CURRENT_PRINCIPAL.get()
    if not settings.auth_enforced or principal.user_id is None:
        return
    store().claim(kind, resource_id, principal)


def visible_ids(kind: str, candidates: Iterable[str], principal: Optional[Principal] = None) -> list[str]:
    """Filter ``candidates`` down to the resources the caller may view.

    One query instead of one per candidate: the whole grant set for the caller
    is pulled once and compared in Python (the grant table is small).
    """
    principal = principal or auth.CURRENT_PRINCIPAL.get()
    candidates = list(candidates)
    if not settings.auth_enforced or principal.is_admin:
        return candidates
    acl = store()
    mine = acl.resource_ids_for_user(principal.user_id, kind) if principal.user_id is not None else set()
    owned = set()
    for resource_id in candidates:
        if resource_id in mine:
            continue
        if not acl.has_owner(kind, resource_id):
            owned.add(resource_id)
    return [c for c in candidates if c in mine or c in owned]


def shares_payload(kind: str, resource_id: str) -> dict:
    """Share list joined with public account names, for the sharing UI."""
    grants = store().list_grants(kind, resource_id)
    users = {user["user_id"]: user for user in auth.store().list_users()}
    out = []
    for grant in grants:
        user = users.get(grant["user_id"])
        out.append(
            {
                "user_id": grant["user_id"],
                "username": user["username"] if user else None,
                "is_active": user["is_active"] if user else False,
                "role": grant["role"],
                "granted_by": grant["granted_by"],
                "created_at": grant["created_at"],
            }
        )
    return {"kind": kind, "resource_id": resource_id, "shares": out}


__all__ = [
    "AclError",
    "AclStore",
    "KINDS",
    "PROJECT",
    "ROLES",
    "TASK",
    "claim",
    "effective_role",
    "has_role",
    "require_edit",
    "require_ownership",
    "require_role",
    "require_view",
    "reset_store",
    "shares_payload",
    "store",
    "visible_ids",
]
