MCP server for agents
=====================

SKSurrogate ships an optional **Model Context Protocol (MCP) server** so AI coding
agents can drive a running instance over its HTTP control plane — register data,
launch experiments and wait on them, analyse models, manage the promotion
lifecycle and keep research records. The server is *external*: it wraps a live
API with per-user key auth and makes **zero changes to ``SKSurrogate/`` or
``api/``** (v1 constraint).

Install
-------

The MCP runtime lives in an opt-in extra so the core install stays lean:

.. code-block:: console

   pip install sksurrogate[mcp]          # or: pip install -e ".[mcp]" from source

This pulls ``httpx`` and the official ``mcp`` SDK (2.x) into your environment;
the package itself is importable without them — only *running* the server needs
the extra. The console script is ``sksurrogate-mcp``.

Configure
---------

Two environment variables, mirroring the control plane's own key auth:

.. list-table::
   :header-rows: 1

   * - Variable
     - Meaning
   * - ``SKSURROGATE_API_URL``
     - Base URL of a running instance (e.g. ``http://localhost:8080``)
   * - ``SKSURROGATE_API_KEY``
     - A **per-user** API key, sent as the ``X-API-Key`` header. Never an admin
       session. Create one with ``python -m api.auth_cli create-key --username <name>``.

Every agent action lands in the audit trail under that key's owner — the same
accountability applies to tools as to manual API calls.

Run (stdio)
-----------

The server speaks MCP over stdio, the standard transport for local agents:

.. code-block:: console

   SKSURROGATE_API_URL=http://localhost:8080 \
   SKSURROGATE_API_KEY=*** \
   sksurrogate-mcp

A generic stdio client configuration looks like this:

.. code-block:: json

   {
     "mcpServers": {
       "sksurrogate": {
         "command": "sksurrogate-mcp",
         "env": {
           "SKSURROGATE_API_URL": "http://localhost:8080",
           "SKSURROGATE_API_KEY": "***"
         }
       }
     }
   }

Client configuration
--------------------

Any MCP client that can launch a stdio server works; the two below are the ones
exercised in practice. Two notes apply to all of them:

* Point ``command`` at the **absolute path** of the console script in the
  environment where the extra is installed
  (``/path/to/venv/bin/sksurrogate-mcp``). GUI applications do not inherit your
  shell ``PATH``, and "command not found" is the most common setup failure.
* Raise the per-tool-call timeout to at least the largest ``job_wait`` cap you
  intend to use (``job_wait`` allows up to 900 s). Clients default to about
  120 s and would cut a long wait off mid-poll; the job itself keeps running
  server-side, so a later ``job_wait``/``job_diagnose`` still collects it.

Hermes
~~~~~~

Hermes reads MCP servers from ``mcp_servers`` in ``~/.hermes/config.yaml``:

.. code-block:: yaml

   mcp_servers:
     sksurrogate:
       command: "/home/me/.venvs/sk/bin/sksurrogate-mcp"
       env:
         SKSURROGATE_API_URL: "http://localhost:8080"
         SKSURROGATE_API_KEY: "sk-..."
       timeout: 900        # >= the largest job_wait cap you use

The stdio launcher passes a *filtered* environment (``PATH``, ``HOME``,
``USER``, ``LANG``, ``TERM``, …) plus exactly the ``env`` block — exporting the
two variables in your shell is not enough, they must be listed here. Restart
Hermes after editing; the tools appear as ``mcp_sksurrogate_<tool>`` (for
example ``mcp_sksurrogate_job_wait``).

Claude Desktop
~~~~~~~~~~~~~~

Claude Desktop reads ``mcpServers`` from ``claude_desktop_config.json`` —
macOS ``~/Library/Application Support/Claude/``, Windows ``%APPDATA%\Claude\``.
Use Settings → Developer → Edit Config so the file is created and opened for
you, then restart the app:

.. code-block:: json

   {
     "mcpServers": {
       "sksurrogate": {
         "command": "/Users/me/.venvs/sk/bin/sksurrogate-mcp",
         "env": {
           "SKSURROGATE_API_URL": "http://localhost:8080",
           "SKSURROGATE_API_KEY": "sk-..."
         }
       }
     }
   }

The server's stderr (including the startup error when ``SKSURROGATE_API_URL``
is unset) lands in ``mcp-server-sksurrogate.log`` in the client's log
directory — macOS ``~/Library/Logs/Claude/``, Windows
``%APPDATA%\Claude\logs\``.

Security notes
--------------

* **Per-user key only.** Never an admin session or password, and not the legacy
  shared static key: the shared key owns no keys, is not attributable to a
  person and bypasses per-user scoping.
* **Everything is audited.** Each tool call becomes an API request recorded in
  the audit trail under the key owner — treat an MCP connection as that
  person's activity, including ``promote_model`` and dataset mutations.
* **Grant the minimum.** A key carries its account's permissions. Use a
  dedicated account whose ACL role matches the job (``viewer`` for read-only
  exploration, ``editor`` when the agent must run experiments or analysis, and
  avoid ``owner`` unless automated promotion is genuinely wanted).
* **The secret is plaintext in the client config.** Both client formats store it
  as an environment string: ``chmod 600`` the file, never commit it, and do not
  paste it into shared logs or issue trackers. Mint with
  ``python -m api.auth_cli create-key --username <name>`` (the secret is shown
  once); revoke with ``DELETE /api/auth/keys/{key_id}``.
* **Transport.** The key travels as an HTTP header; keep the control plane on
  loopback or behind TLS whenever the network is not trusted.
* **No privilege escalation.** The MCP server runs as your OS user and can only
  do what the key can do; it adds no permissions of its own, and it never chains
  irreversible steps together.

Tool catalog
------------

Twenty-nine curated tools grouped by workflow stage, plus one escape hatch (thirty tools
in all). The list stays deliberately tight — anything beyond it is reachable through
``api_request`` and graduates to a curated tool only if agents actually need it repeatedly.

Discovery and data
~~~~~~~~~~~~~~~~~~~

- ``health`` — instance status/version; ``list_tasks``, ``task_overview`` — what exists where
- ``experiment_options`` — presets, components and scoring options in one merged view
- ``register_dataset`` (multipart upload), ``dataset_preflight``, ``impute_missing``,
  ``set_ignored_columns``, ``dataset_preview`` (capped sample only)

Experiments and jobs
~~~~~~~~~~~~~~~~~~~~

- ``run_experiment``, ``optimize_pipeline`` — submit returns a job handle immediately
- ``job_wait(job_id, timeout_s)`` — bounded poll loop *inside* the server (≈5 s cadence;
  default cap 120 s, ceiling 900 s), so agents never hand-poll per turn. While waiting it
  emits MCP progress notifications distilled from the live log: trial count, phase and best
  score seen so far.
- ``job_diagnose`` — failure rollup for a dead job; ``stop_job``, ``resume_job``
- Submit tools refuse to start a duplicate while a live job of the same kind runs on the
  task (idempotency guard); the refusal names the existing job id

Analysis
~~~~~~~~

- ``sensitivity_analysis``, ``top_features`` — model behaviour summaries
- ``explain_model``, ``explain_row`` — ALE-based explanations; row-level intervals use a
  reference sample bounded by the bundle's row count, with clamped inputs and clear errors
- ``evaluation_metrics``, ``nested_cv``

Lifecycle
~~~~~~~~~

- ``registry_summary``, ``quality_gate_check``, ``promote_model`` (staging → production) —
  promotion stays an explicit call; nothing auto-chains
- ``predict``, ``predict_batch`` — against the staged or promoted model via alias/version

Research records
~~~~~~~~~~~~~~~~

- ``project_create``, ``project_update``, ``project_note`` — append-only note semantics kept:
  notes are never edited, only added (human-review posture)

Escape hatch
~~~~~~~~~~~~

- ``api_request(method, path, body)`` — the full control plane (150+ operations) is reachable
  without tool sprawl. Paths and methods are validated against an embedded OpenAPI snapshot
  (regenerated by ``scripts/generate_mcp_openapi_snapshot.py``); a unit test asserts every
  curated tool's hardcoded route exists in the snapshot, and the drift test
  ``tests/test_mcp_snapshot_drift.py`` fails the build when the snapshot and the live
  control plane disagree (see *Version compatibility* below).

Error contract and safety
-------------------------

Expected API failures do **not** raise: tools return structured payloads such as
``{"error": "<detail>", "status_code": 409}`` (4xx detail strings preserved), and an
unreachable host degrades to the same payload shape — MCP tool exceptions would otherwise
strip the message, which is not agent-useful. Malformed *inputs* are rejected locally with a
field-level error before anything hits the API.

Mutating tools state their effect in their descriptions; expensive or irreversible actions
(impute materialization, promotion, resets) are explicit calls only — the server never
chains them automatically. Row-level data export is deliberately absent (the D2 contract);
``dataset_preview`` returns capped samples only.

Version compatibility and pinning
---------------------------------

**One release, one version.** The MCP server ships inside the ``sksurrogate``
distribution, so its version *is* the core version:
``sksurrogate_mcp.__version__`` resolves the installed distribution metadata, and there is
no separate MCP release to keep in sync. Pin and upgrade them together: ``pip install
sksurrogate[mcp]`` installs the latest release; to match a specific deployment, pin it to
the same version as the core package, e.g. ``pip install "sksurrogate[mcp]==<core-version>"``.

The ``[mcp]`` extra declares the runtime floors: ``httpx>=0.27`` and ``mcp>=2.3,<3``. The
SDK upper bound is deliberate — the server imports the 2.x
``mcp.server.mcpserver.MCPServer`` API (SDK 2.0 renamed ``FastMCP``), so an SDK 3.x rename
needs a code change rather than a silent upgrade.

The embedded route snapshot (``sksurrogate_mcp/openapi_ops.json``) is generated from the
core version it ships beside, and CI pins that in two places:

* ``tests/test_mcp_snapshot_drift.py`` rebuilds the spec from the live ``api.main:app``
  and fails on any added, renamed or removed route — counting operations is not enough,
  since a rename keeps the count identical — or on a core-version mismatch. It skips only
  where the API tree is unimportable (a wheel-only install); CI sets
  ``SKSURROGATE_MCP_REQUIRE_DRIFT_CHECK=1`` so a skip there is a failure.
* ``scripts/generate_mcp_openapi_snapshot.py --check`` runs the same comparison from the
  shell and is a CI step of its own, so the failure reads as a route diff plus the
  regeneration command.

Running an older server against a newer core therefore degrades predictably: ``api_request``
refuses routes it does not know, and a curated tool whose route disappeared returns the
API's 404 as an ordinary ``{"error": ..., "status_code": 404}`` payload. Keep client and
API on the same release to avoid that entirely.

Maintaining the snapshot (maintainers)
--------------------------------------

Regenerate after any control-plane route change **and on every core version bump** (the file
records which core it came from), committing it with the change::

    SKSURROGATE_API_HOME=$(mktemp -d) \
        env -u PYTHONPATH ./.venv/bin/python scripts/generate_mcp_openapi_snapshot.py
    env -u PYTHONPATH ./.venv/bin/python scripts/generate_mcp_openapi_snapshot.py --check

The release checklist in ``release.rst`` includes this step.

Testing
-------

``tests/test_mcp_server.py`` covers the tool registry, job-state transitions against a stubbed
control plane (reality-shaped job records), duplicate-live-job refusal, input validation for
every Phase-2 tool and two real-MCP-client stdio round-trips (skipped when the SDK is not
installed). ``tests/test_mcp_snapshot_drift.py`` pins packaging hygiene: the embedded route
snapshot must match the live ``api.main:app`` (any added, renamed or removed route fails) and
must name the core version it was generated from. No network access or solver runs are needed.
