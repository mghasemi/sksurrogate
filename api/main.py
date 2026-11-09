"""FastAPI app factory for the SKSurrogate control-plane API.

Run locally with:

    uvicorn api.main:app --reload

See docs/ui-plan.md for the overall architecture and phased delivery plan.
SQLite-only storage is an intentional v1 decision. Authentication is opt-in:
set ``SKSURROGATE_API_KEY`` to require a shared key on every endpoint except
``/api/health``, and ``SKSURROGATE_API_CORS_ORIGINS`` (comma-separated) to
control which browser origins may call the API — by default only local dev
origins are allowed.
"""

import json
import os
from urllib.parse import parse_qsl

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from .config import settings
from . import logs as log_capture
from .routers import (
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
    synthdata,
)


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


class ApiKeyMiddleware:
    """Enforce the shared API key on every ``/api`` scope except ``/api/health``.

    Implemented as a pure-ASGI middleware (rather than a FastAPI dependency) so
    it covers both HTTP routes and the WebSocket job stream — global
    dependencies do not apply to WebSockets. The key is read from an
    ``X-API-Key`` header, a Bearer token, or — for the WS handshake, where
    browsers cannot set custom headers — an ``api_key`` query parameter.
    """

    def __init__(self, app):
        self.app = app

    @staticmethod
    def _header(headers, name: str) -> str:
        target = name.lower().encode("latin-1")
        for key, value in headers or []:
            if key.lower() == target:
                return value.decode("latin-1")
        return ""

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        protected = (
            settings.api_key is not None
            and scope["type"] in ("http", "websocket")
            and path.startswith("/api")
            and path != "/api/health"
        )
        if not protected:
            await self.app(scope, receive, send)
            return

        supplied = self._header(scope.get("headers"), "x-api-key")
        authorization = self._header(scope.get("headers"), "authorization")
        if authorization.lower().startswith("bearer "):
            supplied = supplied or authorization[7:].strip()
        if scope["type"] == "websocket":
            for key, value in parse_qsl(scope.get("query_string", b"").decode("latin-1")):
                if key == "api_key":
                    supplied = supplied or value
                    break

        if supplied != settings.api_key:
            if scope["type"] == "http":
                response = JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
                await response(scope, receive, send)
            else:
                await send({"type": "websocket.close", "code": 1008})
            return
        await self.app(scope, receive, send)


app = FastAPI(
    title="SKSurrogate API",
    description="Control-plane API for the SKSurrogate MLOps toolkit.",
    version="0.1.0",
)

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
)

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

# Capture core + control-plane log records into the per-job store. The handler
# only forwards records while a job is bound to the current context, so API
# request handling is not captured.
log_capture.install()


@app.get("/api/health", tags=["health"])
def health():
    from SKSurrogate import __version__

    return {"status": "ok", "version": __version__}


@app.get("/api/tasks", tags=["tasks"])
def list_tasks():
    """Return known task names discovered across the storage layout.

    A task is considered known if it has a directory under any of the
    per-task storage areas, an mltrace database file, or at least one
    persisted job record. This powers the task dropdown in the web UI.
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
    return {"tasks": sorted(names)}


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
