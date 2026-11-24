"""Regenerate — or verify — ``sksurrogate_mcp/openapi_ops.json``.

The MCP server embeds an operation list to validate the generic ``api_request``
tool against the routes of the core version it ships with, and to pin every
curated tool's hardcoded route. Two modes:

``--write`` (default)
    Regenerate the snapshot from the core's live spec. Run from the repository
    root after any control-plane route is added, renamed or removed::

        SKSURROGATE_API_HOME=$(mktemp -d) \\
            env -u PYTHONPATH ./.venv/bin/python scripts/generate_mcp_openapi_snapshot.py

``--check``
    Compare the embedded snapshot with the live spec and exit non-zero on drift
    (added/removed operations or a core-version mismatch), printing the diff and
    the regeneration command. This is the CI guard; the pytest wrapper is
    ``tests/test_mcp_snapshot_drift.py``::

        env -u PYTHONPATH ./.venv/bin/python scripts/generate_mcp_openapi_snapshot.py --check

Importing ``api.main`` instantiates the app against ``SKSURROGATE_API_HOME``;
no server is started and nothing else is touched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SNAPSHOT_PATH = _REPO_ROOT / "sksurrogate_mcp" / "openapi_ops.json"
_DEFAULT_HOME = "/tmp/sksurrogate-mcp-snapshot-home"
_HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete"})


def build_operation_list(spec: dict) -> list[dict]:
    """Flatten an OpenAPI spec into sorted ``{method, path, summary}`` records."""
    operations: list[dict] = []
    for path in sorted(spec.get("paths", {})):
        for method, op in (spec["paths"][path] or {}).items():
            if method not in _HTTP_METHODS:
                continue
            operations.append(
                {
                    "method": method.upper(),
                    "path": path,
                    "summary": str((op or {}).get("summary") or "")[:200],
                }
            )
    return operations


def build_snapshot_payload() -> dict:
    """Build the snapshot payload from the core package installed in this tree.

    Sets ``SKSURROGATE_API_HOME`` to an isolated scratch directory first so the
    import never writes into a real deployment dir.
    """
    os.environ.setdefault("SKSURROGATE_API_HOME", _DEFAULT_HOME)
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from SKSurrogate import __version__
    from api.main import app

    spec = app.openapi()
    operations = build_operation_list(spec)
    return {
        "generated_from_version": __version__,
        "operation_count": len(operations),
        "operations": operations,
    }


def load_embedded_snapshot(path: Path = _SNAPSHOT_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def snapshot_drift(embedded: dict, live: dict) -> list[str]:
    """Human-readable differences between the embedded and live snapshots.

    Compares the route set (method + path), not only the count: a rename keeps
    the count identical but breaks a curated tool. The core version is reported
    separately because it is provenance — a version bump without a route change
    still requires regeneration so the recorded version stays truthful.
    """
    problems: list[str] = []

    embedded_version = str(embedded.get("generated_from_version", "unknown"))
    live_version = str(live.get("generated_from_version", "unknown"))
    if embedded_version != live_version:
        problems.append(
            f"snapshot was generated from core {embedded_version}, "
            f"but this tree is core {live_version}"
        )

    def key(op: dict) -> tuple[str, str]:
        return (str(op.get("method", "")).upper(), str(op.get("path", "")))

    embedded_ops = {key(op): op for op in embedded.get("operations", []) if isinstance(op, dict)}
    live_ops = {key(op): op for op in live.get("operations", []) if isinstance(op, dict)}

    missing = sorted(set(live_ops) - set(embedded_ops))
    stale = sorted(set(embedded_ops) - set(live_ops))
    if missing:
        problems.append("operations in the live spec but missing from the snapshot:")
        problems.extend(f"  + {method} {path}" for method, path in missing)
    if stale:
        problems.append("operations in the snapshot but gone from the live spec:")
        problems.extend(f"  - {method} {path}" for method, path in stale)
    return problems


def write_snapshot(payload: dict, path: Path = _SNAPSHOT_PATH) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write or verify sksurrogate_mcp/openapi_ops.json against the core's live spec."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the embedded snapshot against the live spec; never writes",
    )
    args = parser.parse_args(argv)

    live = build_snapshot_payload()
    version = live["generated_from_version"]
    count = live["operation_count"]

    if not args.check:
        write_snapshot(live)
        print(f"wrote {_SNAPSHOT_PATH.relative_to(_REPO_ROOT)}: {count} operations (core {version})")
        return 0

    try:
        embedded = load_embedded_snapshot()
    except (OSError, ValueError) as exc:
        print(f"cannot read the embedded snapshot: {exc}", file=sys.stderr)
        return 1

    problems = snapshot_drift(embedded, live)
    if not problems:
        print(f"snapshot is current: {count} operations (core {version})")
        return 0

    print("Embedded OpenAPI snapshot DRIFTED from the live control plane:", file=sys.stderr)
    for line in problems:
        print(f"  {line}", file=sys.stderr)
    print(
        "\nRegenerate it and commit the result:\n"
        "  SKSURROGATE_API_HOME=$(mktemp -d) env -u PYTHONPATH "
        "./.venv/bin/python scripts/generate_mcp_openapi_snapshot.py",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
