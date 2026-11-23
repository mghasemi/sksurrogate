"""MCP server exposing the SKSurrogate HTTP control plane as agent tools.

Phase 0 ships three discovery tools only: ``health``, ``list_tasks`` and
``task_overview``. Each tool is one curated endpoint — no computation is
re-implemented here, so every call still passes through the deployment's
authentication, ACL roles and audit trail exactly as if a human had made it.

Configuration comes from the environment (see the package docstring):
``SKSURROGATE_API_URL`` (required) and ``SKSURROGATE_API_KEY`` (optional).

Run with the console script ``sksurrogate-mcp`` (stdio transport, the default)
or ``python -m sksurrogate_mcp.server``.

Error contract: expected API failures (4xx/5xx answers, unreachable host) are
returned as ``{"error": "<message>"}`` payloads rather than raised, so an agent
always receives a structured answer; only unexpected internal exceptions mark
the MCP result itself as an error.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

import httpx

DEFAULT_TIMEOUT_S = 60.0


class ControlPlaneError(RuntimeError):
    """Unreachable deployment or a non-2xx / malformed API answer."""


def _detail(response: httpx.Response) -> str:
    """Bounded human-readable detail for an error response body.

    FastAPI errors are shaped ``{"detail": ...}``; keep the message short so it
    stays useful as a tool-error string in front of an LLM.
    """
    try:
        body = response.json()
    except ValueError:
        text = response.text
    else:
        if isinstance(body, dict) and "detail" in body:
            detail = body["detail"]
            text = detail if isinstance(detail, str) else json.dumps(detail)
        else:
            text = json.dumps(body)
    return text[:1000]


class ControlPlaneClient:
    """Bounded HTTP client bound to one SKSurrogate control-plane deployment."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        if not base_url:
            raise ValueError("base_url is required")
        self.base_url = base_url.rstrip("/")
        # AuthMiddleware (api/main.py) accepts the key as X-API-Key or
        # Authorization: Bearer *** header is the canonical non-browser form.
        headers = {} if api_key in (None, "") else {"X-API-Key": api_key}
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout_s)

    def request(self, method: str, path: str, *, params: Optional[dict] = None) -> Any:
        try:
            response = self._http.request(method, path, params=params)
        except httpx.HTTPError as exc:
            raise ControlPlaneError(
                f"control plane {self.base_url} unreachable: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise ControlPlaneError(
                f"{method.upper()} {path} -> HTTP {response.status_code}: {_detail(response)}"
            )
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise ControlPlaneError(f"{method.upper()} {path} returned a non-JSON body") from exc

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs)


def _call(client: ControlPlaneClient, method: str, path: str) -> Any:
    """Run one control-plane call with the shared error contract.

    Expected API failures (4xx/5xx from the deployment, unreachable host,
    malformed bodies) come back as ``{"error": "<message>"}`` instead of
    raising, so an agent always receives a structured, greppable payload —
    e.g. ``{"error": "GET /api/datasets/foo -> HTTP 404: No dataset registered"}``.
    Only truly unexpected exceptions propagate (the MCP SDK then marks the
    result as an error).
    """
    try:
        data = client.request(method, path)
    except ControlPlaneError as exc:
        return {"error": str(exc)}
    return data if data is not None else {}


def create_server(client: ControlPlaneClient):
    """Build the MCPServer with tools bound to ``client``.

    Kept separate from :func:`main` so tests can construct a server against a
    stubbed client without touching the environment.
    """
    # Imported here (not at module top) so that importing sksurrogate_mcp does
    # not require the optional MCP SDK — only running the server does. Verified
    # against MCP Python SDK 2.x, where FastMCP was renamed to MCPServer.
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        name="sksurrogate",
        title="SKSurrogate control plane",
        description=(
            "Tools for driving a running SKSurrogate deployment: dataset "
            f"registration, pipeline search, evaluation and model lifecycle. Bound to {client.base_url}."
        ),
    )

    @server.tool()
    def health() -> dict:
        """Check connectivity to the SKSurrogate control plane.

        Returns {"status": "ok", "version": "<core version>"}. Call this first
        after any configuration change; a failure here means the base URL or
        credential is wrong, not that one specific tool is broken. On failure
        returns {"error": "<message>"}.
        """
        return _call(client, "GET", "/api/health")

    @server.tool()
    def list_tasks() -> dict:
        """List task names visible to this credential on the deployment.

        A task appears if it has a dataset directory, an mltrace database file,
        or persisted job records; with auth enforced only tasks the caller owns
        or is shared are returned. Read-only. Use the returned names as the
        ``task_name`` argument of other tools. On failure returns
        {"error": "<message>"}.
        """
        return _call(client, "GET", "/api/tasks")

    @server.tool()
    def task_overview(task_name: str) -> dict:
        """Registered schema and metadata for one dataset (no raw rows).

        Returns the registered column schema, target column, declared problem
        type with its provenance, temporal spec, dataset fingerprint, feature
        counts and partition history. Raises HTTP 404 when no dataset is
        registered under ``task_name`` — the tool then returns an
        {"error": ...} payload; call ``list_tasks`` to check names. Row-level
        previews are deliberately not part of this tool.

        Args:
            task_name: Task name as returned by ``list_tasks``.
        """
        return _call(client, "GET", f"/api/datasets/{task_name}")

    return server


def main(argv: Optional[list] = None) -> int:
    base_url = os.environ.get("SKSURROGATE_API_URL", "").strip()
    api_key = os.environ.get("SKSURROGATE_API_KEY") or None
    if not base_url:
        # stderr only: stdout is reserved for the MCP stdio protocol.
        print(
            "sksurrogate-mcp: SKSURROGATE_API_URL is required (e.g. http://127.0.0.1:8099); "
            "optionally set SKSURROGATE_API_KEY for deployments that enforce auth.",
            file=sys.stderr,
        )
        return 2

    client = ControlPlaneClient(base_url, api_key)
    server = create_server(client)
    server.run()  # stdio transport (default)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
