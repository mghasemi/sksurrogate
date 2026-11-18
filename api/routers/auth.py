"""Account, session, and API-key endpoints.

This is the identity surface of the control plane: the browser logs in here
(``/api/auth/login``) and receives an ``HttpOnly`` session cookie; programmatic
clients create per-user keys here and send them as ``X-API-Key``/``Bearer``.

Anonymous access is limited to :func:`status` and :func:`login` — the middleware
in :mod:`api.main` enforces that, and every other route in this module requires
an authenticated principal (admins additionally get user management).

Key material is shown once: :func:`create_key` returns the plaintext secret and
later reads return only the prefix, so a database dump cannot be replayed
against the API.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from .. import audit, auth
from ..auth import SESSION_COOKIE
from ..config import settings
from ..deps import bad_request, not_found, require_admin, require_user

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=auth.MAX_PASSWORD_LENGTH)


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=auth.MAX_PASSWORD_LENGTH)
    new_password: str = Field(min_length=auth.MIN_PASSWORD_LENGTH, max_length=auth.MAX_PASSWORD_LENGTH)


class KeyCreate(BaseModel):
    name: str = Field(default="default", min_length=1, max_length=auth.MAX_API_KEY_NAME_LENGTH)


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=auth.MIN_PASSWORD_LENGTH, max_length=auth.MAX_PASSWORD_LENGTH)
    is_admin: bool = False


class UserPatch(BaseModel):
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None
    password: Optional[str] = Field(default=None, min_length=auth.MIN_PASSWORD_LENGTH, max_length=auth.MAX_PASSWORD_LENGTH)


def _public_user(user: dict, *, include_admin: bool = True) -> dict:
    """Account view without the password digest (never leaves the process)."""
    payload = {
        "user_id": user["user_id"],
        "username": user["username"],
        "is_active": user["is_active"],
        "created_at": user["created_at"],
        "updated_at": user["updated_at"],
    }
    if include_admin:
        payload["is_admin"] = user["is_admin"]
    return payload


def _set_session_cookie(response: Response, token: str) -> None:
    """Attach the session cookie.

    ``HttpOnly`` keeps it out of JavaScript (an XSS bug cannot exfiltrate it),
    ``SameSite=Strict`` stops it riding cross-site requests, and ``Secure`` is
    opt-in via ``SKSURROGATE_SESSION_COOKIE_SECURE`` because local development
    runs over plain HTTP.
    """
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_ttl_seconds,
        httponly=True,
        samesite="strict",
        secure=settings.session_cookie_secure,
        path="/",
    )


@router.get("/status")
def status():
    """Public authentication posture, used by the SPA before any login.

    Never reveals account names or counts beyond the boolean "can anyone log
    in yet": the UI needs to know whether to show the login form or the
    bootstrap instructions.
    """
    store = auth.store()
    users = store.count_users()
    return {
        "auth_enforced": bool(settings.auth_enforced),
        "login_available": users > 0,
        "bootstrap_required": bool(settings.auth_enforced) and users == 0,
        "session_ttl_seconds": int(settings.session_ttl_seconds),
    }


@router.post("/login")
def login(body: LoginRequest, request: Request, response: Response):
    """Authenticate with a username/password and start a session.

    Failures are deliberately indistinguishable (same status, same message,
    constant-time hash work) and are throttled per username+client address so
    online guessing costs time.
    """
    client = (request.client.host if request.client else "unknown") or "unknown"
    if auth.THROTTLE.locked(body.username, client):
        audit.record("login", outcome="throttled", username=body.username, client=client)
        return _too_many_attempts()
    user = auth.store().authenticate(body.username, body.password)
    if user is None:
        auth.THROTTLE.record_failure(body.username, client)
        audit.record("login", outcome="denied", username=body.username, client=client)
        return _bad_credentials()
    auth.THROTTLE.reset(body.username, client)
    token = auth.store().create_session(user["user_id"], ttl_seconds=settings.session_ttl_seconds)
    _set_session_cookie(response, token)
    audit.record(
        "login",
        outcome="ok",
        user_id=user["user_id"],
        username=user["username"],
        client=client,
        session_source="cookie",
    )
    return {"user": _public_user(user), "expires_in": int(settings.session_ttl_seconds)}


def _bad_credentials():
    raise HTTPException(status_code=401, detail="Invalid username or password")


def _too_many_attempts():
    raise HTTPException(
        status_code=429, detail="Too many failed login attempts; try again later"
    )


@router.post("/logout")
def logout(request: Request, response: Response):
    """End the current session (idempotent; also clears the cookie)."""
    raw = request.cookies.get(SESSION_COOKIE)
    if raw:
        auth.store().delete_session(raw)
        audit.record("logout", principal=auth.CURRENT_PRINCIPAL.get())
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@router.get("/me")
def me(principal: auth.Principal = Depends(require_user())):
    """Return the caller's identity and how it authenticated."""
    payload = principal.as_dict()
    user = auth.store().get_user(principal.user_id) if principal.user_id is not None else None
    payload["user"] = _public_user(user) if user else None
    payload["auth_enforced"] = bool(settings.auth_enforced)
    return payload


@router.post("/password")
def change_password(body: PasswordChange, request: Request, principal: auth.Principal = Depends(require_user())):
    """Change the caller's own password; other sessions are revoked."""
    if principal.user_id is None:
        raise bad_request("the shared API key has no password to change")
    store = auth.store()
    user = store.get_user(principal.user_id)
    if user is None:
        raise not_found("account no longer exists")
    if not auth.verify_password(user["password_hash"], body.current_password):
        raise bad_request("current password is incorrect")
    current_token = request.cookies.get(SESSION_COOKIE) or ""
    keep = auth.digest_token(current_token) if current_token else None
    store.set_password(principal.user_id, body.new_password, keep_token_hash=keep)
    audit.record("password.change", principal=principal)
    return {"ok": True, "sessions_revoked": True}


# --------------------------------------------------------------------------- #
# API keys (own account)
# --------------------------------------------------------------------------- #


@router.get("/keys")
def list_keys(principal: auth.Principal = Depends(require_user())):
    """List the caller's API keys (prefix only — the secret is not recoverable)."""
    if principal.user_id is None:
        return {"keys": []}
    keys = [key for key in auth.store().list_api_keys(principal.user_id)]
    return {"keys": keys}


@router.post("/keys")
def create_key(body: KeyCreate, principal: auth.Principal = Depends(require_user())):
    """Create an API key; the response is the only time the secret is visible."""
    if principal.user_id is None:
        raise bad_request("the shared API key cannot own per-user API keys")
    secret, record = auth.store().create_api_key(principal.user_id, body.name)
    # Audit the prefix, never the secret.
    audit.record("api_key.create", principal=principal, prefix=record["prefix"], key_id=record["id"])
    return {"key": secret, "record": record}


@router.delete("/keys/{key_id}")
def revoke_key(key_id: int, principal: auth.Principal = Depends(require_user())):
    """Revoke one of the caller's keys (admins may revoke any key)."""
    scope = None if principal.is_admin else principal.user_id
    try:
        record = auth.store().revoke_api_key(key_id, user_id=scope)
    except KeyError as exc:
        raise not_found(str(exc))
    audit.record("api_key.revoke", principal=principal, prefix=record["prefix"], key_id=record["id"])
    return {"revoked": record}


# --------------------------------------------------------------------------- #
# User management (admin)
# --------------------------------------------------------------------------- #


@router.get("/users")
def list_users(principal: auth.Principal = Depends(require_admin())):
    return {"users": [_public_user(user) for user in auth.store().list_users()]}


@router.post("/users", status_code=201)
def create_user(body: UserCreate, principal: auth.Principal = Depends(require_admin())):
    try:
        user = auth.store().create_user(body.username, body.password, is_admin=body.is_admin)
    except auth.UsernameTaken as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except auth.AuthError as exc:
        raise bad_request(str(exc))
    audit.record(
        "user.create", principal=principal, target_user=user["username"], is_admin=user["is_admin"]
    )
    return {"user": _public_user(user)}


@router.patch("/users/{user_id}")
def update_user(user_id: int, body: UserPatch, principal: auth.Principal = Depends(require_admin())):
    store = auth.store()
    if store.get_user(user_id) is None:
        raise not_found("no such user: %r" % user_id)
    try:
        if body.is_active is not None:
            store.set_active(user_id, body.is_active)
        if body.is_admin is not None:
            store.set_admin(user_id, body.is_admin)
        if body.password is not None:
            store.set_password(user_id, body.password)
    except auth.LastAdminError as exc:
        raise bad_request(str(exc))
    except auth.AuthError as exc:
        raise bad_request(str(exc))
    updated = store.get_user(user_id)
    if updated is None:  # deleted concurrently
        raise not_found("no such user: %r" % user_id)
    audit.record(
        "user.update",
        principal=principal,
        target_user=updated["username"],
        is_active=updated["is_active"],
        is_admin=updated["is_admin"],
        password_reset=body.password is not None,
    )
    return {"user": _public_user(updated)}


@router.get("/audit")
def auth_audit(limit: int = 200, principal=Depends(require_admin())):
    """Administrator view of identity events (logins, keys, account changes).

    Newest first, bounded. The file is append-only on disk; this endpoint never
    mutates it. Secrets are never recorded, so entries are safe to inspect.
    """
    events = audit.read(limit)
    return {"events": events, "count": len(events)}


@router.post("/users/{user_id}/logout")
def logout_user(user_id: int, principal: auth.Principal = Depends(require_admin())):
    """Revoke every session of another account."""
    removed = auth.store().delete_user_sessions(user_id)
    audit.record("user.sessions_revoked", principal=principal, target_user_id=user_id, count=removed)
    return {"sessions_revoked": removed}
