"""Live-spec drift guard for the embedded MCP OpenAPI snapshot.

Package-level contract (``MCP_SERVER_PLAN.md``, Phase 3): the committed
``sksurrogate_mcp/openapi_ops.json`` must describe the control-plane routes of
the core version that ships beside it. The snapshot backs two guarantees:

* ``api_request`` refuses routes that do not exist in the spec it was built
  against (so agents get a clear error instead of a 404 bounce);
* every curated tool's hardcoded route is pinned (``TOOL_ENDPOINTS``).

Both decay silently if the core renames a route. This module recomputes the
snapshot from the *live* ``api.main:app`` and fails when the two disagree,
printing the added/removed operations and the regeneration command. It reuses
``scripts/generate_mcp_openapi_snapshot.py`` so the check and the generator can
never drift from each other.

Skipped when the API tree is not importable (e.g. a wheel-only install without
``api/``); CI sets ``SKSURROGATE_MCP_REQUIRE_DRIFT_CHECK=1`` so a skip there is
a hard failure instead of a silent hole.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_mcp_openapi_snapshot.py"
SNAPSHOT_PATH = REPO_ROOT / "sksurrogate_mcp" / "openapi_ops.json"

REGENERATE_HINT = (
    "Regenerate the snapshot and commit it:\n"
    "  SKSURROGATE_API_HOME=$(mktemp -d) env -u PYTHONPATH "
    "./.venv/bin/python scripts/generate_mcp_openapi_snapshot.py"
)


def _load_generator():
    """Import the generator script by path (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("_mcp_openapi_generator", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_generator()


@pytest.fixture(scope="module")
def live_payload(generator, tmp_path_factory):
    home = tmp_path_factory.mktemp("mcp-drift-home")
    os.environ.setdefault("SKSURROGATE_API_HOME", str(home))
    try:
        return generator.build_snapshot_payload()
    except ImportError as exc:  # pragma: no cover - depends on install layout
        message = f"cannot build the live OpenAPI spec in this tree: {exc}"
        if os.environ.get("SKSURROGATE_MCP_REQUIRE_DRIFT_CHECK"):
            pytest.fail(message)
        pytest.skip(message)


def test_embedded_snapshot_matches_live_spec(generator, live_payload):
    """Route set (method + path) must match the live control plane exactly."""
    embedded = generator.load_embedded_snapshot(SNAPSHOT_PATH)
    problems = generator.snapshot_drift(embedded, live_payload)
    assert not problems, (
        "sksurrogate_mcp/openapi_ops.json no longer matches api.main:app:\n  "
        + "\n  ".join(problems)
        + "\n\n"
        + REGENERATE_HINT
    )


def test_snapshot_records_the_core_version(generator, live_payload):
    """Provenance: the snapshot must name the core version it came from."""
    from SKSurrogate import __version__

    embedded = generator.load_embedded_snapshot(SNAPSHOT_PATH)
    recorded = embedded.get("generated_from_version")
    assert recorded == __version__, (
        f"snapshot records core {recorded!r} but this tree is {__version__!r}; "
        "the snapshot is regenerated on every core release.\n" + REGENERATE_HINT
    )
    assert live_payload["generated_from_version"] == __version__
    assert embedded.get("operation_count") == len(embedded.get("operations", [])), (
        "snapshot operation_count disagrees with its operations list"
    )


def test_snapshot_drift_reports_route_and_version_changes(generator):
    """Pin the detector itself: additions, removals and version drift are all caught.

    A route *rename* keeps the operation count identical, which is why the
    comparison is on the (method, path) set rather than the count.
    """
    live = {
        "generated_from_version": "0.4.0",
        "operation_count": 2,
        "operations": [
            {"method": "GET", "path": "/api/health"},
            {"method": "GET", "path": "/api/new"},
        ],
    }
    embedded = {
        "generated_from_version": "0.3.1",
        "operation_count": 2,
        "operations": [
            {"method": "GET", "path": "/api/health"},
            {"method": "POST", "path": "/api/old"},
        ],
    }

    problems = generator.snapshot_drift(embedded, live)
    report = "\n".join(problems)
    assert "core 0.3.1" in report and "core 0.4.0" in report
    assert "+ GET /api/new" in report
    assert "- POST /api/old" in report
    # Identical payloads must be clean (no false positives on the live snapshot).
    assert generator.snapshot_drift(live, live) == []
