"""Agent-facing MCP server for SKSurrogate's HTTP control plane.

This package is an *external* wrapper: it speaks MCP to the agent (stdio by
default) and plain HTTP to a running SKSurrogate deployment, so no part of
``SKSurrogate/`` or ``api/`` needs to change in order to be driven by an agent
(internal scope document ``MCP_SERVER_SCOPE.md``, decision accepted 2026-11-12).

Configuration is per-process environment variables:

* ``SKSURROGATE_API_URL`` — base URL of a running control plane (required);
* ``SKSURROGATE_API_KEY`` — optional credential sent as the ``X-API-Key``
  header: either a per-user key minted via ``python -m api.auth_cli create-key``
  or the legacy shared static key. Omit it when auth is not enforced.
"""

__version__ = "0.0.1"
