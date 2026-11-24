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

Versioning: there is **one** release, the ``sksurrogate`` distribution. This
subpackage ships inside it, so its version is the core version — never a
separate number (see "Version compatibility" in ``docs/mcp-server.rst``).
"""

from __future__ import annotations

import re
from pathlib import Path


def _distribution_version() -> str:
    """Version of the installed ``sksurrogate`` distribution.

    Single source of truth is ``SKSurrogate.__version__``, surfaced through the
    distribution metadata. Falls back to reading the sibling core package when
    the tree is used straight from a checkout without an installed dist (the
    same regex ``setup.py`` uses), so the value is never a stale literal.
    """
    try:
        from importlib.metadata import version as _dist_version

        return _dist_version("sksurrogate")
    except Exception:  # PackageNotFoundError, or metadata backend trouble
        init_py = Path(__file__).resolve().parents[1] / "SKSurrogate" / "__init__.py"
        try:
            match = re.search(
                r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]',
                init_py.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
        except OSError:
            match = None
        return match.group(1) if match else "0+unknown"


__version__ = _distribution_version()
