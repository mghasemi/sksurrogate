"""Regenerate ``sksurrogate_mcp/openapi_ops.json`` from the core's live spec.

The MCP server embeds this operation list to validate the generic
``api_request`` tool against the routes of the core version it was built with,
and the Phase 3 drift test regenerates it in CI and fails when any curated
tool's hardcoded path disappears from a newer spec.

Run from the repository root::

    SKSURROGATE_API_HOME=$(mktemp -d) \
        env -u PYTHONPATH ./.venv/bin/python scripts/generate_mcp_openapi_snapshot.py

(Importing ``api.main`` instantiates the app against that home; no server is
started and nothing else is touched.)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    # Isolated API home so the import never writes into a real deployment dir.
    os.environ.setdefault("SKSURROGATE_API_HOME", "/tmp/sksurrogate-mcp-snapshot-home")
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from SKSurrogate import __version__
    from api.main import app

    spec = app.openapi()
    operations: list[dict] = []
    for path in sorted(spec.get("paths", {})):
        for method, op in (spec["paths"][path] or {}).items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            operations.append(
                {
                    "method": method.upper(),
                    "path": path,
                    "summary": str(op.get("summary") or "")[:200],
                }
            )

    payload = {
        "generated_from_version": __version__,
        "operation_count": len(operations),
        "operations": operations,
    }
    out_path = _REPO_ROOT / "sksurrogate_mcp" / "openapi_ops.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"wrote {out_path.relative_to(_REPO_ROOT)}: {len(operations)} operations (core {__version__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
