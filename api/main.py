"""FastAPI app factory for the SKSurrogate control-plane API.

Run locally with:

    uvicorn api.main:app --reload

See docs/ui-plan.md for the overall architecture and phased delivery plan.
SQLite-only storage is an intentional v1 decision.

Authentication is opt-in and has two layers (see :mod:`api.auth`):

* **Identity** — browser sessions (``HttpOnly`` cookie), per-user API keys
  (``X-API-Key`` / ``Bearer``), or the legacy shared ``SKSURROGATE_API_KEY``,
  which resolves to an admin-equivalent break-glass principal.
* **Enforcement** — with neither a shared key nor ``SKSURROGATE_REQUIRE_AUTH``
  the API is open (local single-user use) and every request runs as the
  anonymous principal. When either is configured, every ``/api`` route except
  ``/api/health``, ``/api/auth/status`` and ``/api/auth/login`` needs a valid
  identity, and the OpenAPI/docs routes are gated with it.
"""

import json
import hmac
import logging
import os
from http.cookies import SimpleCookie
from urllib.parse import parse_qsl, urlsplit

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from . import acl, auth, compute
from .auth import ANONYMOUS, SESSION_COOKIE, STATIC_KEY, Principal
from .config import _flag, settings
from .safety import UnsafeName
from . import logs as log_capture
from .jobs import job_manager
from .routers import (
    auth as auth_router,
    bundles,
    datasets,
    design,
    synthesis,
    evaluation,
    experiments,
    explain,
    inference,
    jobs,
    lineage,
    logs,
    monitoring,
    projects,
    quality_gates,
    registry,
    retraining,
    sensitivity,
    settings as settings_router,
    sharing,
    synthdata,
)

#: Routes an unauthenticated caller may still reach (the login flow itself).
ANONYMOUS_API_PATHS = frozenset({"/api/health", "/api/auth/status", "/api/auth/login"})


def _check_auth_settings() -> None:
    """Validate the auth configuration at startup and bootstrap the first admin.

    Two failure modes matter:

    * No credential at all — the API is open to anyone who can reach the port.
      Fine for local use (log a warning); dangerous once the port is published
      (the container entrypoint binds all interfaces).
    * ``SKSURROGATE_REQUIRE_AUTH`` set with nothing to authenticate *against* —
      the process would be locked out of its own control plane, so refuse to
      start. Any of these satisfies it: ``SKSURROGATE_API_KEY``, a bootstrap
      administrator (``SKSURROGATE_ADMIN_USERNAME``/``_PASSWORD``), or an
      account already present in the auth database.
    """
    # Read the environment here (not ``settings.require_auth``) so a deployment
    # that exports the variable late is still caught; the middleware uses the
    # import-time copy, which is what a running server actually observes.
    require_auth = _flag(os.environ.get("SKSURROGATE_REQUIRE_AUTH"))
    auth.bootstrap_from_env()
    has_users = auth.store().count_users() > 0
    if require_auth and settings.api_key is None and not has_users:
        raise RuntimeError(
            "SKSURROGATE_REQUIRE_AUTH is set but no credential is configured, so the "
            "control plane would lock itself out. Set SKSURROGATE_API_KEY (shared "
            "break-glass key), or bootstrap an administrator with "
            "SKSURROGATE_ADMIN_USERNAME and SKSURROGATE_ADMIN_PASSWORD, or create "
            "one with `python -m api.auth create-admin`."
        )
    if not settings.auth_enforced:
        logging.getLogger("sksurrogate.api").warning(
            "API authentication disabled (no SKSURROGATE_API_KEY, no "
            "SKSURROGATE_REQUIRE_AUTH): every /api endpoint except /api/health is "
            "open to any client that can reach this process. Set "
            "SKSURROGATE_REQUIRE_AUTH and create an account when the port may be "
            "exposed beyond localhost."
        )


_check_auth_settings()


def _cors_origins() -> list[str]:
    """Extra browser origins allowed to call the API, from
    ``SKSURROGATE_API_CORS_ORIGINS`` (comma-separated).

    Local development needs no entry here: the ``allow_origin_regex`` passed to
    ``CORSMiddleware`` below already allows ``localhost`` / ``127.0.0.1`` on any
    port. Starlette matches ``allow_origins`` *exactly*, so a wildcard entry such
    as ``http://localhost:*`` would never match anything — extend the regex for
    patterns, and use this list only for concrete extra origins.
    """
    return list(settings.cors_origins)


class AuthMiddleware:
    """Resolve every HTTP/WebSocket request to a principal and enforce it.

    Implemented as a pure-ASGI middleware (rather than a FastAPI dependency) so
    it covers the WebSocket job stream too, and so no route can accidentally be
    registered without it. Resolution order:

    1. ``X-API-Key`` header or ``Authorization: Bearer`` — the legacy shared
       key (admin break-glass) first, then a per-user API key.
    2. The session cookie (browser UI), stored only as a SHA-256 digest.
    3. On WebSocket handshakes only: an ``api_key`` query parameter, because
       browsers cannot set custom headers on ``new WebSocket(...)``.

    When enforcement is on, unauthenticated requests to ``/api`` (other than the
    login flow) and to the schema/docs routes are rejected — 401 on HTTP,
    close code 1008 on WebSocket. Static SPA assets stay readable so the login
    form can load; the UI is useless without a session because every data route
    is gated.
    """

    #: Non-``/api`` routes that reveal or exercise the API surface.
    PROTECTED_SCHEMA_PATHS = frozenset(
        {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    )

    #: Methods that change state and therefore need CSRF protection when the
    #: credential is a cookie (browsers attach cookies automatically).
    MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

    def __init__(self, app):
        self.app = app

    @staticmethod
    def _header(headers, name: str) -> str:
        target = name.lower().encode("latin-1")
        for key, value in headers or []:
            if key.lower() == target:
                return value.decode("latin-1")
        return ""

    @classmethod
    def _supplied_secret(cls, scope) -> str:
        supplied = cls._header(scope.get("headers"), "x-api-key")
        authorization = cls._header(scope.get("headers"), "authorization")
        if authorization.lower().startswith("bearer "):
            supplied = supplied or authorization[7:].strip()
        if scope["type"] == "websocket":
            for param, value in parse_qsl(scope.get("query_string", b"").decode("latin-1")):
                if param == "api_key":
                    supplied = supplied or value
                    break
        return supplied

    @classmethod
    def _session_token(cls, scope) -> str:
        raw = cls._header(scope.get("headers"), "cookie")
        if not raw:
            return ""
        jar = SimpleCookie()
        try:
            jar.load(raw)
        except Exception:  # malformed Cookie header: treat as no session
            return ""
        morsel = jar.get(SESSION_COOKIE)
        return morsel.value if morsel else ""

    @classmethod
    def _resolve(cls, scope) -> Principal:
        """Identify the caller. Never raises — a bad credential yields ANONYMOUS."""
        if not settings.auth_enforced:
            return ANONYMOUS
        supplied = cls._supplied_secret(scope)
        if supplied:
            expected = settings.api_key
            if expected and hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
                return STATIC_KEY
            user = auth.store().resolve_api_key(supplied)
            if user is not None:
                return Principal(user["user_id"], user["username"], user["is_admin"], "api_key")
        token = cls._session_token(scope)
        if token:
            user = auth.store().resolve_session(token)
            if user is not None:
                return Principal(user["user_id"], user["username"], user["is_admin"], "session")
        return ANONYMOUS

    @staticmethod
    def _csrf_ok(scope) -> bool:
        """Same-origin check for cookie-authenticated mutating requests.

        Browsers always send ``Origin`` on cross-origin and same-origin POST
        requests, so a cookie+POST without one is either a non-browser client
        (which should use an API key) or an attempt to bypass CSRF protection.
        """
        origin = AuthMiddleware._header(scope.get("headers"), "origin")
        host = AuthMiddleware._header(scope.get("headers"), "host")
        if not origin or not host:
            return False
        try:
            parsed = urlsplit(origin)
        except ValueError:
            return False
        return bool(parsed.netloc) and parsed.netloc.lower() == host.lower()

    @staticmethod
    async def _reject(scope, receive, send, detail: str, status: int = 401, code: int = 1008):
        client = scope.get("client") or ("unknown", 0)
        logging.getLogger("sksurrogate.api").warning(
            "Authentication rejected: %s %s from %s (%s)",
            scope.get("method", "websocket"),
            scope.get("path", ""),
            client[0],
            detail,
        )
        if scope["type"] == "http":
            response = JSONResponse(
                status_code=status,
                content={"detail": detail},
                headers={"WWW-Authenticate": 'Cookie, ApiKey realm="sksurrogate"'},
            )
            await response(scope, receive, send)
        else:
            await send({"type": "websocket.close", "code": code})

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        principal = self._resolve(scope)
        scope.setdefault("state", {})["principal"] = principal
        token = auth.CURRENT_PRINCIPAL.set(principal)
        try:
            if not settings.auth_enforced:
                await self.app(scope, receive, send)
                return

            is_api = path.startswith("/api")
            is_schema = path in self.PROTECTED_SCHEMA_PATHS
            if not is_api and not is_schema:
                # Static SPA assets: public so the login form can render. The UI
                # shows nothing without a session because /api is gated.
                await self.app(scope, receive, send)
                return
            if scope["type"] == "http" and path in ANONYMOUS_API_PATHS:
                await self.app(scope, receive, send)
                return
            if not principal.authenticated:
                await self._reject(scope, receive, send, "Authentication required")
                return
            if (
                principal.source == "session"
                and scope["type"] == "http"
                and scope.get("method", "GET").upper() in self.MUTATING_METHODS
                and not self._csrf_ok(scope)
            ):
                await self._reject(
                    scope, receive, send, "Cross-origin session request refused", status=403
                )
                return
            await self.app(scope, receive, send)
        finally:
            auth.CURRENT_PRINCIPAL.reset(token)


#: Backwards-compatible alias (the class was introduced as ``ApiKeyMiddleware``).
ApiKeyMiddleware = AuthMiddleware


def current_principal() -> Principal:
    """The principal of the request being handled (anonymous outside a request)."""
    return auth.CURRENT_PRINCIPAL.get()



app = FastAPI(
    title="SKSurrogate API",
    description="Control-plane API for the SKSurrogate MLOps toolkit.",
    version="0.1.0",
)


@app.exception_handler(UnsafeName)
async def _unsafe_name_handler(request, exc):
    """Map caller-supplied path segments that would escape storage to HTTP 400."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})

# Middleware added first ends up innermost; CORSMiddleware is added last so it
# stays outermost and can decorate the 401 responses with CORS headers.
app.add_middleware(ApiKeyMiddleware)
# The UI is a separate SPA client (see docs/ui-plan.md) and normally calls the API
# *same-origin*, through the dev server's /api proxy — see webui/vite.config.ts —
# so it needs no CORS entry at all. The regex keeps direct localhost/127.0.0.1
# calls working; SKSURROGATE_API_CORS_ORIGINS adds extra origins for non-local
# deployments. Starlette matches allow_origins exactly (no wildcards), and the
# regex is what actually allows local dev origins on any port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
    # Session cookies must ride along when the UI is served from a configured
    # absolute API base; same-origin calls need nothing special.
    allow_credentials=True,
)

app.include_router(auth_router.router)
app.include_router(sharing.router)
app.include_router(datasets.router)
app.include_router(synthdata.router)
app.include_router(bundles.router)
app.include_router(quality_gates.router)
app.include_router(registry.router)
app.include_router(inference.router)
app.include_router(monitoring.router)
app.include_router(projects.router)
app.include_router(sensitivity.router)
app.include_router(experiments.router)
app.include_router(jobs.router)
app.include_router(retraining.router)
app.include_router(lineage.router)
app.include_router(evaluation.router)
app.include_router(explain.router)
app.include_router(design.router)
app.include_router(synthesis.router)
app.include_router(logs.router)
app.include_router(settings_router.router)

# Capture core + control-plane log records into the per-job store. The handler
# only forwards records while a job is bound to the current context, so API
# request handling is not captured.
log_capture.install()

# Pin BLAS/OpenMP to the configured budget before the first request: jobs
# re-apply it at start, but request-path estimators (inference, evaluation)
# share the same process and must not oversubscribe either.
compute.apply_limits()

# Jobs live in this process, so anything persisted as queued/running at start-up
# belongs to a previous process that died: mark it interrupted instead of
# advertising work that no thread is doing.
job_manager.reconcile()


@app.get("/api/health", tags=["health"])
def health():
    from SKSurrogate import __version__

    return {"status": "ok", "version": __version__}


@app.get("/api/tasks", tags=["tasks"])
def list_tasks():
    """Return known task names the caller may see.

    A task is considered known if it has a directory under any of the
    per-task storage areas, an mltrace database file, or at least one
    persisted job record. This powers the task dropdown in the web UI.

    When authentication is enforced the list is filtered to tasks the caller
    owns or has been shared: the dropdown is the first place a user would
    otherwise learn the names of everybody else's datasets.
    """
    names = set()
    for directory in (
        settings.datasets_dir,
        settings.bundles_dir,
        settings.registry_dir,
        settings.monitoring_dir,
        settings.checkpoints_dir,
    ):
        if directory.is_dir():
            names.update(p.name for p in directory.iterdir() if p.is_dir())
    if settings.mltrace_dir.is_dir():
        names.update(p.stem for p in settings.mltrace_dir.glob("*.db"))
    if settings.jobs_dir.is_dir():
        for path in settings.jobs_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            name = record.get("task_name") if isinstance(record, dict) else None
            if isinstance(name, str) and name:
                names.add(name)
    return {"tasks": sorted(acl.visible_ids(acl.TASK, names))}


def _mount_webui() -> None:
    """Serve the built SPA (``webui/dist``) with an index.html fallback.

    Active only when the build exists (production image, or after ``npm run
    build``); in development the Vite dev server serves the UI instead.
    Registered last so it never shadows an ``/api`` route.
    """
    from pathlib import Path

    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    dist = Path(os.environ.get("SKSURROGATE_WEBUI_DIST") or Path(__file__).resolve().parent.parent / "webui" / "dist")
    index = dist / "index.html"
    if not index.is_file():
        return
    root = dist.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = (root / path).resolve()
        if path and candidate.is_file() and root in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index)


_mount_webui()
