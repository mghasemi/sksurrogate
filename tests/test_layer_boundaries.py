"""Import-direction lint for the three layers.

The architecture is three layers, in one direction only:

* **Core** — ``SKSurrogate/``: the optimization engine and the scientific ML
  toolkit. Usable without the platform installed (no FastAPI, no web UI).
* **Platform** — ``api/``: the HTTP control plane. It may import the core, never
  the web UI and never the MCP server package.
* **Clients** — ``webui/`` and ``sksurrogate_mcp/``: consumers of the API.

A cyclic import between layers is how a "library" silently becomes an
application: every new feature drags the platform in. This test parses every
module (not just the top-level ``__init__`` — a lazy import inside a function is
still a dependency) and fails on a forbidden edge.

The allowlist is **shrink-only**. Each entry must name the file and the import,
and adding one requires a written reason; the test fails if an entry is stale so
the list cannot rot.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: (layer directory, forbidden top-level packages, human description)
LAYERS = (
    ("SKSurrogate", {"api", "webui", "sksurrogate_mcp"}, "core imports the platform or a client"),
    ("api", {"webui", "sksurrogate_mcp"}, "the control plane imports a client"),
)

#: Deliberate, documented exceptions: ``(relative path, imported module): reason``.
#: Keep this empty unless a reviewed decision requires otherwise.
ALLOWLIST = {}


def _module_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            # A relative import (level > 0) stays inside the layer by definition.
            if node.level == 0 and node.module:
                yield node.module, node.lineno


def _violations():
    found = []
    for directory, forbidden, _ in LAYERS:
        for path in sorted((ROOT / directory).rglob("*.py")):
            for module, line in _module_imports(path):
                top = module.split(".")[0]
                if top in forbidden:
                    found.append((path.relative_to(ROOT), module, line))
    return found


def test_layers_never_import_their_consumers():
    violations = [
        (str(relative), module, line)
        for relative, module, line in _violations()
        if (str(relative), module) not in ALLOWLIST
    ]
    assert not violations, (
        "import-direction violation(s); the core must stay usable without the "
        "platform and the API must not depend on a client:\n"
        + "\n".join("  %s:%s imports %s" % item for item in violations)
    )


def test_allowlist_entries_are_live_and_documented():
    live = {(str(relative), module) for relative, module, _ in _violations()}
    stale = set(ALLOWLIST) - live
    assert not stale, "allowlist entries that no longer match an import: %s" % sorted(stale)
    for (path, module), reason in ALLOWLIST.items():
        assert reason and reason.strip(), "allowlist entry %s (%s) has no reason" % (path, module)


def test_core_does_not_import_the_web_framework():
    """The optimization engine must import without FastAPI/uvicorn installed."""
    forbidden = {"fastapi", "uvicorn", "starlette"}
    violations = []
    for path in sorted((ROOT / "SKSurrogate").rglob("*.py")):
        for module, line in _module_imports(path):
            if module.split(".")[0] in forbidden:
                violations.append((str(path.relative_to(ROOT)), module, line))
    assert not violations, "core modules import the web framework: %s" % violations[:10]
