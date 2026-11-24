====================
Architecture layers
====================

SKSurrogate is three layers with one direction of dependency. The split exists so
the optimization engine stays a **library** that a scientist can install without
the control plane, and so a platform feature can never quietly become a
requirement of the core.

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Layer
     - Contents
     - May import
   * - **Core**
     - ``SKSurrogate/`` — surrogate search, EOA/AML, mean-polynomial and SONC/
       SOS helpers, data preprocessing, model bundles, manifests, evaluation and
       monitoring primitives.
     - Only the core runtime requirements (``requirements-core.txt``) and optional
       extras. **Never** ``api``, ``webui`` or ``sksurrogate_mcp``.
   * - **Platform**
     - ``api/`` — the FastAPI control plane, jobs, auth, storage layout, routers
       and the report/explanations services.
     - The core, and the API requirements in ``requirements-api.txt``. Never the
       web UI or the MCP server package.
   * - **Clients**
     - ``webui/`` (React) and ``sksurrogate_mcp/`` (MCP server).
     - The HTTP API only. They may not be imported by either layer above them.

The core is usable without the platform
=======================================

``import SKSurrogate`` never imports FastAPI, Starlette or uvicorn, and no module
under ``SKSurrogate/`` imports ``api`` or a client. A fit, a bundle and a manifest
work in a plain Python environment with only ``requirements-core.txt`` installed;
the API is a consumer of that library, not its substrate.

The direction is enforced by ``tests/test_layer_boundaries.py``. It parses every
module — including imports inside functions, because a lazy import is still a
dependency — and fails on a forbidden edge. Its allowlist is empty and
shrink-only: a deliberate exception must name the file, the import and a written
reason, and a stale entry fails the test.

Packaging follows the same split
================================

The published wheel is the **core**. Heavier capabilities are extras
(``pip install "sksurrogate[api]"``, ``[benchmark]``, ``[mcp]``, ``[xgboost]``,
…), and a missing optional backend raises an actionable message naming the extra
that installs it. The control plane ships as a separate workspace package
(``requirements-api.txt``) and the web UI is built separately.

The import-direction test is the guard against the common drift: a core feature
that reaches for ``api.config.settings`` to find a directory, or an explanation
helper that imports a router. If that helper belongs to the platform, it moves;
if it belongs to the core, it takes its inputs as arguments.

Deferred work
=============

The full package split (``sksurrogate-core`` / ``sksurrogate-extensions`` /
``sksurrogate-platform`` distributions) is deferred. This cycle only enforces the
import direction and documents the boundaries; moving files would make every
reliability diff unreviewable.
