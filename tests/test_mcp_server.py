"""Tests for the sksurrogate_mcp package (Phases 0-2: core loop, analysis,
lifecycle, research records).

The control plane is a local stateful HTTP stub (``ThreadingHTTPServer``,
same fake-provider pattern as ``tests/test_api_synthesis.py``): no network,
no solver. The MCP tool functions are exercised directly against it — their
logic and the shared error contract do not depend on the optional SDK — and
two tests additionally require the ``mcp`` extra (skipped without it): the
tool registry built through MCPServer and one full stdio round-trip with a
real client session.

Run from a clean environment::

    env -u PYTHONPATH ./.venv/bin/python -m pytest tests/test_mcp_server.py -q
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Stub control plane (stateful; scripted job transitions)
# ---------------------------------------------------------------------------


class _StubState:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.active: list[dict] = []
        self.datasets: dict[str, dict] = {
            "demo": {
                "task_name": "demo",
                "target_name": "y",
                "problem_type": "regression",
                "effective_feature_count": 1,
                "partitions": {"train": {"rows": 40}},
            }
        }
        self.requests: list[dict] = []
        self.last_multipart_body = b""
        # Phase 2 state: registry bundles per task + project records / notes.
        self.registry_bundles: dict[str, list[dict]] = {}
        self.projects: dict[str, dict] = {}
        self.project_notes: dict[str, list[dict]] = {}

    def seed_job(
        self,
        job_id: str,
        script: list[str],
        *,
        kind: str = "experiment",
        task_name: str = "demo",
        result=None,
        error: str | None = None,
    ) -> None:
        self.jobs[job_id] = {
            "job_id": job_id,
            "task_name": task_name,
            "kind": kind,
            "script": list(script),
            "polls": 0,
            "result": result if result is not None else {"best_score": -12.5},
            "error": error,
        }


class _StubHandler(BaseHTTPRequestHandler):
    server_version = "mcp-stub/1"

    def log_message(self, format: str, *args) -> None:  # keep test output clean
        pass

    @property
    def state(self) -> _StubState:
        return self.server.state  # type: ignore[attr-defined]

    # -- plumbing ---------------------------------------------------------
    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record(self, method: str, parsed, json_body=None) -> dict:
        entry = {
            "method": method,
            "path": parsed.path,
            "query": {k: v[-1] for k, v in urllib.parse.parse_qs(parsed.query).items()},
        }
        if json_body is not None:
            entry["json_body"] = json_body
        self.state.requests.append(entry)
        return entry

    def _job_record(self, job_id: str):
        state = self.state
        job = state.jobs.get(job_id)
        if job is None:
            return {"detail": f"unknown job id: {job_id!r}"}
        script = job["script"]
        index = min(job["polls"], len(script) - 1)
        status = script[index]
        job["polls"] += 1
        record = {
            "job_id": job_id,
            "task_name": job["task_name"],
            "kind": job["kind"],
            "backend": "local",
            "status": status,
            "created_at": "2026-11-23T00:00:00Z",
            # The real control plane always carries both fields (null while live);
            # mirror that so tool guards can't mistake a live record for an error.
            "result": None,
            "error": None,
        }
        if status == "completed":
            record["result"] = job["result"]
        elif status in ("failed", "interrupted"):
            record["error"] = job["error"] or (
                "Traceback (most recent call last):\n  stub failure\nValueError: boom"
            )
        return record

    def _route(self, method: str) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        body = self._read_body()
        path_parts = [p for p in parsed.path.split("/") if p]
        state = self.state

        try:
            json_body = json.loads(body.decode("utf-8")) if body else {}
        except ValueError:
            json_body = None

        # -- explicit GET stubs ---------------------------------------------
        if method == "GET" and path_parts == ["api", "health"]:
            self._record(method, parsed)
            return self._send_json({"status": "ok", "version": "0.3.1-stub"})
        if method == "GET" and path_parts == ["api", "tasks"]:
            self._record(method, parsed)
            return self._send_json({"tasks": sorted(state.datasets)})
        if method == "GET" and len(path_parts) == 3 and path_parts[:2] == ["api", "datasets"]:
            task_name = path_parts[2]
            self._record(method, parsed)
            meta = state.datasets.get(task_name)
            if meta is None:
                return self._send_json({"detail": f"no dataset registered for task {task_name!r}"}, 404)
            return self._send_json(meta)
        if method == "GET" and len(path_parts) == 4 and path_parts[:2] == ["api", "datasets"] and path_parts[3] == "preflight":
            task_name = path_parts[2]
            self._record(method, parsed)
            meta = state.datasets.get(task_name)
            if meta is None:
                return self._send_json({"detail": f"no dataset registered for task {task_name!r}"}, 404)
            scanned = int((urllib.parse.parse_qs(parsed.query).get("max_rows") or ["5000"])[0])
            return self._send_json(
                {"task_name": task_name, "advisory": True, "rows_scanned": min(scanned, 40), "warnings": []}
            )
        if method == "GET" and len(path_parts) == 5 and path_parts[:2] == ["api", "datasets"] and path_parts[4] == "preview":
            task_name, partition = path_parts[2], path_parts[3]
            self._record(method, parsed)
            meta = state.datasets.get(task_name) or {}
            if partition not in (meta.get("partitions") or {}):
                return self._send_json({"detail": f"unknown partition {partition!r} for task {task_name!r}"}, 404)
            limit = int((urllib.parse.parse_qs(parsed.query).get("limit") or ["20"])[0])
            rows = [{"x": float(i), "y": float(2 * i)} for i in range(min(limit, 5))]
            return self._send_json({"partition": partition, "columns": ["x", "y"], "rows": rows, "row_count": len(rows)})
        if method == "GET" and len(path_parts) == 3 and path_parts[:2] == ["api", "experiments"] and path_parts[2] in {
            "presets", "components", "scoring-options"}:
            self._record(method, parsed)
            return self._send_json({"stub": path_parts[2], "value": 42})
        if method == "GET" and len(path_parts) >= 3 and path_parts[:2] == ["api", "jobs"] and path_parts[2] != "active":
            job_id = path_parts[2]
            self._record(method, parsed)
            return self._send_json(self._job_record(job_id), status=200 if job_id in state.jobs else 404)
        if method == "GET" and len(path_parts) == 3 and path_parts[:2] == ["api", "jobs"] and path_parts[2] == "active":
            self._record(method, parsed)
            task_filter = (urllib.parse.parse_qs(parsed.query).get("task_name") or [None])[0]
            jobs = [j for j in state.active if task_filter is None or j.get("task_name") == task_filter]
            return self._send_json({"jobs": jobs, "count": len(jobs)})
        if method == "GET" and path_parts[:2] == ["api", "logs"] and len(path_parts) == 3:
            job_id = path_parts[2]
            limit = int((urllib.parse.parse_qs(parsed.query).get("limit") or ["100"])[0])
            self._record(method, parsed)
            entries = [
                {"ts": f"t{i}", "level": "info", "message": f"trial {i} finished"} for i in range(min(limit, 8))
            ] + [{"ts": "t9", "level": "error", "message": "boom at the end"}]
            return self._send_json(
                {"entries": entries[-limit:], "counts": {"info": len(entries) - 1, "error": 1}, "total": len(entries)}
            )

        # -- mutating routes --------------------------------------------------
        if method == "POST" and path_parts[:2] == ["api", "experiments"] and len(path_parts) == 4:
            task_name, action = path_parts[2], path_parts[3]
            self._record(method, parsed)
            prefix = "stub-run-job" if action == "run" else f"stub-opt-{task_name}"
            return self._send_json(
                {"job_id": prefix, "display_name": f"{action} {task_name}", "status": "queued"}, 202
            )
        if method == "POST" and len(path_parts) == 4 and path_parts[:2] == ["api", "jobs"] and path_parts[3] in {"stop", "resume"}:
            job_id = path_parts[2]
            self._record(method, parsed)
            if job_id not in state.jobs:
                return self._send_json({"detail": f"job {job_id!r} is not running"}, 409)
            if path_parts[3] == "stop":
                mode = (json_body or {}).get("mode", "pause")
                return self._send_json(
                    {"job_id": job_id, "status": "paused" if mode == "pause" else "interrupted"}
                )
            new_id = f"{job_id}-resume"
            state.jobs[new_id] = dict(state.jobs[job_id], job_id=new_id, polls=0)
            return self._send_json({"job_id": new_id, "display_name": f"resume {job_id}", "status": "queued"})
        if method == "POST" and len(path_parts) == 4 and path_parts[:2] == ["api", "datasets"] and path_parts[3] in {"impute", "register"}:
            task_name = path_parts[2]
            self._record(method, parsed)
            if path_parts[3] == "register":
                state.last_multipart_body = body
                return self._send_json({"task_name": task_name, "partition": "train", "rows": 40})
            out_partition = (json_body or {}).get("output_partition") or f"train_imputed"
            preview_only = bool((json_body or {}).get("preview_only"))
            return self._send_json(
                {
                    "task_name": task_name,
                    "output_partition": None if preview_only else out_partition,
                    "written": not preview_only,
                }
            )
        if method == "PUT" and len(path_parts) == 4 and path_parts[:2] == ["api", "datasets"] and path_parts[3] == "ignored-columns":
            task_name = path_parts[2]
            self._record(method, parsed)
            columns = (json_body or {}).get("columns")
            if not isinstance(columns, list):
                return self._send_json({"detail": "columns must be a list"}, 400)
            meta = state.datasets.get(task_name) or {}
            features = int(meta.get("effective_feature_count") or 1) - len(columns)
            return self._send_json(
                {"task_name": task_name, "ignored_columns": columns, "effective_features": max(features, 0)}
            )

        # -- Phase 2 routes ----------------------------------------------------
        if method == "GET" and len(path_parts) == 4 and path_parts[:2] == ["api", "sensitivity"] and path_parts[3] == "top-features":
            task_name = path_parts[2]
            self._record(method, parsed, json_body)
            meta = state.datasets.get(task_name)
            if meta is None:
                return self._send_json({"detail": f"no dataset registered for task {task_name!r}"}, 404)
            num = int((urllib.parse.parse_qs(parsed.query).get("num") or ["10"])[0])
            features = [{"name": "x", "weight": 0.9}] * min(num, 3)
            return self._send_json(
                {"available": True, "method": "sobol", "num_requested": num, "features": features}
            )
        if method == "GET" and len(path_parts) == 4 and path_parts[:2] == ["api", "evaluation"] and path_parts[3] == "metrics":
            task_name = path_parts[2]
            self._record(method, parsed, json_body)
            meta = state.datasets.get(task_name)
            if meta is None:
                return self._send_json({"detail": f"no dataset registered for task {task_name!r}"}, 404)
            model_version = (urllib.parse.parse_qs(parsed.query).get("model_version") or [None])[0]
            return self._send_json(
                {
                    "task_name": task_name,
                    "problem_type": meta.get("problem_type"),
                    "model_version": model_version,
                    "metric_options": ["neg_mean_absolute_error", "neg_root_mean_squared_error"],
                    "default_metric": "neg_mean_absolute_error",
                }
            )
        if method == "GET" and len(path_parts) == 4 and path_parts[:2] == ["api", "registry"] and path_parts[3] == "summary":
            task_name = path_parts[2]
            self._record(method, parsed, json_body)
            meta = state.datasets.get(task_name)
            if meta is None:
                return self._send_json({"detail": f"no dataset registered for task {task_name!r}"}, 404)
            bundles = list(state.registry_bundles.get(task_name, []))
            return self._send_json({"task_name": task_name, "bundles": bundles})
        if method == "GET" and len(path_parts) == 3 and path_parts[:2] == ["api", "projects"]:
            project_id = path_parts[2]
            self._record(method, parsed, json_body)
            record = state.projects.get(project_id)
            if record is None:
                return self._send_json({"detail": f"no project with id {project_id!r}"}, 404)
            return self._send_json(dict(record))

        if method == "POST" and len(path_parts) == 4 and path_parts[:2] == ["api", "sensitivity"] and path_parts[3] == "run":
            task_name = path_parts[2]
            self._record(method, parsed, json_body)
            return self._send_json(
                {"job_id": f"stub-sens-{task_name}", "status": "queued", "echo_body": json_body or {}}, 202
            )
        if method == "POST" and len(path_parts) == 4 and path_parts[:2] == ["api", "evaluation"] and path_parts[3] == "nested-cv":
            task_name = path_parts[2]
            self._record(method, parsed, json_body)
            return self._send_json(
                {"job_id": f"stub-nested-{task_name}", "status": "queued", "echo_body": json_body or {}}, 202
            )
        if method == "POST" and len(path_parts) == 5 and path_parts[:2] == ["api", "evaluation"] and path_parts[4] == "explain":
            model_version = path_parts[3]
            self._record(method, parsed, json_body)
            return self._send_json({"model_version": model_version, "echo_body": json_body or {}})
        if method == "POST" and len(path_parts) == 6 and path_parts[:2] == ["api", "evaluation"] and path_parts[4:6] == ["explain", "row"]:
            model_version = path_parts[3]
            self._record(method, parsed, json_body)
            return self._send_json({"model_version": model_version, "echo_body": json_body or {}})
        if method == "POST" and len(path_parts) == 4 and path_parts[:2] == ["api", "registry"] and path_parts[3] == "promote":
            self._record(method, parsed, json_body)
            body = json_body or {}
            return self._send_json(
                {
                    "model_version": body.get("model_version"),
                    "state": body.get("state"),
                    "approvers": body.get("approvers", []),
                    "required_approvals": body.get("required_approvals", 1),
                }
            )
        if method == "POST" and len(path_parts) == 5 and path_parts[:2] == ["api", "quality-gates"] and path_parts[4] == "check":
            model_version = path_parts[3]
            self._record(method, parsed, json_body)
            body = json_body or {}
            return self._send_json({"passed": True, "checks": [], "model_version": model_version, "echo_body": body})
        if method == "POST" and len(path_parts) == 4 and path_parts[:2] == ["api", "inference"]:
            task_name, action = path_parts[2], path_parts[3]
            self._record(method, parsed, json_body)
            body = json_body or {}
            if action == "predict":
                rows = body.get("rows")
                if not isinstance(rows, list):
                    return self._send_json({"detail": "rows must be a list of objects"}, 400)
                return self._send_json(
                    {
                        "count": len(rows),
                        "predictions": [1.5] * len(rows),
                        "alias": body.get("alias"),
                        "model_version": body.get("model_version"),
                    }
                )
            if action == "predict-batch":
                return self._send_json(
                    {
                        "status": "ok",
                        "partition": body.get("partition"),
                        "alias": body.get("alias"),
                        "model_version": body.get("model_version"),
                    }
                )
        if method == "POST" and path_parts == ["api", "projects"]:
            self._record(method, parsed, json_body)
            body = json_body or {}
            name = body.get("name")
            for existing in state.projects.values():
                if existing.get("name") == name:
                    return self._send_json({"detail": f"project named {name!r} already exists"}, 409)
            project_id = "%032x" % (len(state.projects) + 1)
            new_record: dict[str, object] = {"project_id": project_id}
            for key in ("name", "description"):
                if body.get(key) is not None:
                    new_record[key] = body[key]
            new_record["research_questions"] = list(body.get("research_questions") or [])
            new_record["artifacts"] = list(body.get("artifacts") or [])
            state.projects[project_id] = new_record
            return self._send_json(new_record)
        if method == "POST" and len(path_parts) == 4 and path_parts[:2] == ["api", "projects"] and path_parts[3] == "notes":
            project_id = path_parts[2]
            self._record(method, parsed, json_body)
            record = state.projects.get(project_id)
            if record is None:
                return self._send_json({"detail": f"no project with id {project_id!r}"}, 404)
            body = json_body or {}
            note: dict[str, object] = {"note_id": "n%032d" % (len(state.project_notes.get(project_id, [])) + 1)}
            note.update(body)
            state.project_notes.setdefault(project_id, []).append(note)
            return self._send_json({"project_id": project_id, **note})

        if method == "PUT" and len(path_parts) == 3 and path_parts[:2] == ["api", "projects"]:
            project_id = path_parts[2]
            self._record(method, parsed, json_body)
            record = state.projects.get(project_id)
            if record is None:
                return self._send_json({"detail": f"no project with id {project_id!r}"}, 404)
            body = json_body or {}
            for key in ("name", "description"):
                if body.get(key) is not None:
                    record[key] = body[key]
            if body.get("research_questions") is not None:
                record["research_questions"] = list(body["research_questions"])
            if body.get("artifacts") is not None:
                record["artifacts"] = list(body["artifacts"])
            return self._send_json(dict(record))

        # -- fallback: faithful to FastAPI's unknown-route behaviour ----------
        self._record(method, parsed)
        self._send_json({"detail": f"stub has no route for {method} {parsed.path}"}, 404)

    def do_GET(self) -> None:  # noqa: N802
        self._route("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._route("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self._route("PUT")


class StubControlPlane:
    """Context-managed stub server on an ephemeral localhost port."""

    def __init__(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self.state = _StubState()
        self.server.state = self.state  # type: ignore[attr-defined]
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"

    def start(self) -> "StubControlPlane":
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def stub():
    server = StubControlPlane().start()
    yield server
    server.stop()


@pytest.fixture()
def client(stub):
    from sksurrogate_mcp.server import ControlPlaneClient

    return ControlPlaneClient(stub.base_url, "stub-key")


# ---------------------------------------------------------------------------
# Tool registry (no SDK required)
# ---------------------------------------------------------------------------


EXPECTED_TOOL_NAMES = {
    # Phase 0 — discovery
    "health",
    "list_tasks",
    "task_overview",
    # Phase 1 — core loop
    "experiment_options",
    "register_dataset",
    "dataset_preflight",
    "impute_missing",
    "set_ignored_columns",
    "dataset_preview",
    "run_experiment",
    "optimize_pipeline",
    "job_wait",
    "job_diagnose",
    "stop_job",
    "resume_job",
    # Phase 2 — analysis
    "sensitivity_analysis",
    "top_features",
    "explain_model",
    "explain_row",
    "evaluation_metrics",
    "nested_cv",
    # Phase 2 — lifecycle
    "registry_summary",
    "promote_model",
    "quality_gate_check",
    "predict",
    "predict_batch",
    # Phase 2 — research records
    "project_create",
    "project_update",
    "project_note",
    # Escape hatch (Phase 1)
    "api_request",
}


def test_tool_registry_is_complete_and_unique():
    from sksurrogate_mcp.server import TOOLS, TOOL_ENDPOINTS

    names = [fn.__name__ for fn in TOOLS]
    assert sorted(names) == sorted(EXPECTED_TOOL_NAMES), names
    assert len(set(names)) == len(names)
    # Every curated tool has a documented endpoint mapping (drift-test anchor, G3).
    # api_request is exempt: it is the escape hatch validated against the whole
    # embedded spec snapshot, not one fixed route.
    for name in names:
        if name == "api_request":
            continue
        entries = TOOL_ENDPOINTS.get(name)
        assert entries, f"TOOL_ENDPOINTS missing entry for {name}"


def test_package_version_tracks_the_core_release():
    """One distribution, one version: sksurrogate_mcp ships inside sksurrogate.

    The value is resolved from the installed distribution metadata (with a
    source-tree fallback), so it can never drift as a hardcoded literal.
    """
    import sksurrogate_mcp
    from SKSurrogate import __version__

    assert sksurrogate_mcp.__version__ == __version__


def test_snapshot_covers_every_curated_endpoint():
    """The embedded OpenAPI snapshot must contain each curated tool's route."""
    from sksurrogate_mcp.server import TOOL_ENDPOINTS, _load_openapi_snapshot

    loaded = _load_openapi_snapshot()
    assert isinstance(loaded, tuple), f"snapshot load failed: {loaded!r}"
    operations, _version_note = loaded
    for tool_name, entries in TOOL_ENDPOINTS.items():
        for method, path in entries:
            template_segments = [s for s in path.split("/") if s]
            matched = any(
                str(op.get("method", "")).upper() == method
                and _segments_match([s for s in str(op["path"]).split("/") if s], template_segments)
                for op in operations
            )
            assert matched, f"{tool_name}: {method} {path} missing from embedded snapshot"


def _segments_match(wanted: list[str], literal: list[str]) -> bool:
    """Template match used by the drift check ({param} accepts one segment)."""
    if len(wanted) != len(literal):
        return False
    return all(
        w == l or (w.startswith("{") and w.endswith("}") and bool(l)) for w, l in zip(wanted, literal)
    )


def test_client_error_carries_status_code():
    from sksurrogate_mcp.server import ControlPlaneClient, ControlPlaneError

    # Port 1 on loopback: nothing listens → unreachable-host path (no status).
    client = ControlPlaneClient("http://127.0.0.1:1", timeout_s=0.5)
    with pytest.raises(ControlPlaneError) as excinfo:
        client.get("/api/health")
    assert "unreachable" in str(excinfo.value), excinfo.value
    assert excinfo.value.status_code is None, excinfo.value


# ---------------------------------------------------------------------------
# Discovery tools
# ---------------------------------------------------------------------------


def test_health_and_list_tasks(client):
    from sksurrogate_mcp.server import health, list_tasks

    payload = health(client)
    assert payload == {"status": "ok", "version": "0.3.1-stub"}, payload
    tasks = list_tasks(client)["tasks"]
    assert tasks == ["demo"], tasks


def test_task_overview_and_404_contract(client):
    from sksurrogate_mcp.server import task_overview

    meta = task_overview(client, "demo")
    assert meta["task_name"] == "demo" and meta["target_name"] == "y", meta

    missing = task_overview(client, "ghost")
    assert isinstance(missing, dict) and "error" in missing, missing
    assert "404" in missing["error"], missing


def test_experiment_options_merges_three_endpoints(client):
    from sksurrogate_mcp.server import experiment_options

    payload = experiment_options(client)
    assert payload["presets"] == {"stub": "presets", "value": 42}, payload
    assert payload["components"]["stub"] == "components", payload
    assert payload["scoring_options"]["stub"] == "scoring-options", payload


# ---------------------------------------------------------------------------
# Data tools
# ---------------------------------------------------------------------------


def test_register_dataset_sends_multipart_with_fields(client, stub, tmp_path):
    from sksurrogate_mcp.server import register_dataset

    csv_file = tmp_path / "train.csv"
    csv_file.write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
    payload = register_dataset(
        client, task_name="demo", target="y", csv_path=str(csv_file), ignored_columns=["noise"]
    )
    assert "error" not in payload and payload.get("task_name") == "demo", payload
    raw = stub.state.last_multipart_body
    assert b"x,y\n1,2" in raw, "CSV file bytes must ride as a multipart part"
    assert b'name="target"' in raw, f"form fields missing from multipart body: {raw[:200]!r}"


def test_register_dataset_rejects_missing_file(client):
    from sksurrogate_mcp.server import register_dataset

    payload = register_dataset(client, task_name="demo", target="y", csv_path="/does/not/exist.csv")
    assert "error" in payload and "not found locally" in payload["error"], payload


def test_preflight_passes_query_params(client, stub):
    from sksurrogate_mcp.server import dataset_preflight

    payload = dataset_preflight(client, "demo", partition="train", max_rows=1234)
    assert isinstance(payload, dict) and "error" not in payload, payload
    # rows_scanned proves the max_rows query parameter reached the stub.
    assert payload.get("advisory") is True and payload.get("rows_scanned") == 40, payload
    preflight_calls = [r for r in stub.state.requests if r["path"] == "/api/datasets/demo/preflight"]
    assert preflight_calls, "preflight request never hit the wire"
    last = preflight_calls[-1]
    assert last["query"].get("partition") == "train" and last["query"].get("max_rows") == "1234", last


def test_set_ignored_columns_round_trip_and_validation(client):
    from sksurrogate_mcp.server import set_ignored_columns

    payload = set_ignored_columns(client, "demo", ["noise"])
    assert payload.get("ignored_columns") == ["noise"], payload
    bad = set_ignored_columns(client, "demo", "not-a-list")  # type: ignore[arg-type]
    assert isinstance(bad, dict) and "error" in bad, bad


def test_impute_missing_default_and_preview_only(client):
    from sksurrogate_mcp.server import impute_missing

    written = impute_missing(client, "demo", method="simple")
    # G6: the tool surfaces the NEW partition name; callers must not assume it.
    assert written.get("output_partition") == "train_imputed" and written.get("written"), written

    preview = impute_missing(client, "demo", method="simple", preview_only=True)
    assert preview.get("written") is False and preview.get("output_partition") in (None, ""), preview


def test_dataset_preview_resolves_default_partition_and_clamps_limit(client, stub):
    from sksurrogate_mcp.server import dataset_preview

    payload = dataset_preview(client, "demo", limit=10**6)
    assert "error" not in payload and payload.get("partition") == "train", payload
    preview_calls = [r for r in stub.state.requests if r["path"] == "/api/datasets/demo/train/preview"]
    assert preview_calls, "default partition was not resolved to 'train'"
    assert preview_calls[-1]["query"].get("limit") == "100", "G6 cap: limit must clamp to 100"

    missing = dataset_preview(client, "ghost", partition="train")
    assert isinstance(missing, dict) and "404" in missing["error"], missing


# ---------------------------------------------------------------------------
# Experiment submission + duplicate guard (G4)
# ---------------------------------------------------------------------------


def test_run_experiment_submits_and_records_body(client, stub):
    from sksurrogate_mcp.server import run_experiment

    payload = run_experiment(
        client, "demo", preset="regression", max_generations=7, train_partition="train_imputed"
    )
    assert payload.get("job_id") == "stub-run-job" and payload.get("status") == "queued", payload
    runs = [r for r in stub.state.requests if r["path"] == "/api/experiments/demo/run"]
    assert len(runs) == 1, runs


def test_optimize_pipeline_requires_sequence_and_refuses_duplicates(client, stub):
    from sksurrogate_mcp.server import optimize_pipeline

    bad = optimize_pipeline(client, "demo", seq=[])
    assert "error" in bad and "non-empty" in bad["error"], bad

    ok = optimize_pipeline(
        client, "demo", seq=["sklearn.ensemble.RandomForestRegressor"], scoring="neg_mean_absolute_error"
    )
    assert str(ok.get("job_id", "")).startswith("stub-opt-"), ok

    stub.state.active.append(
        {"job_id": "live-opt", "task_name": "demo", "kind": "optimize_pipeline", "status": "running"}
    )
    refused = optimize_pipeline(client, "demo", seq=["sklearn.linear_model.LinearRegression"])
    assert "error" in refused and refused.get("duplicate_job_id") == "live-opt", refused


def test_run_experiment_refuses_duplicate_live_job(client, stub):
    from sksurrogate_mcp.server import run_experiment

    stub.state.active.append(
        {"job_id": "live-exp-1", "task_name": "demo", "kind": "experiment", "status": "queued"}
    )
    refused = run_experiment(client, "demo", preset="regression")
    assert "error" in refused and refused.get("duplicate_job_id") == "live-exp-1", refused

    # A non-live (completed) record of the same kind must NOT block resubmission.
    stub.state.active[0]["status"] = "completed"
    allowed = run_experiment(client, "demo", preset="regression")
    assert allowed.get("job_id") == "stub-run-job", allowed


# ---------------------------------------------------------------------------
# Job handling (G1 bounded wait, G5 diagnose rollup)
# ---------------------------------------------------------------------------


def test_job_wait_transitions_to_completed(client, stub):
    import asyncio

    from sksurrogate_mcp.server import job_wait

    stub.state.seed_job("job-done", ["queued", "running", "completed"])
    payload = asyncio.run(job_wait(client, "job-done", timeout_s=10, poll_interval_s=0.2))
    assert payload.get("status") == "completed" and not payload.get("timed_out"), payload
    assert payload.get("result") == {"best_score": -12.5}, payload


def test_job_wait_times_out_on_nonterminal_status(client, stub):
    import asyncio
    import time

    from sksurrogate_mcp.server import job_wait

    stub.state.seed_job("job-slow", ["queued", "running"])  # never terminal
    started = time.monotonic()
    payload = asyncio.run(job_wait(client, "job-slow", timeout_s=1.0, poll_interval_s=0.3))
    elapsed = time.monotonic() - started
    assert payload.get("timed_out") is True and payload.get("status") == "running", payload
    assert 0.8 <= elapsed < 5.0, f"wait loop must respect its budget; took {elapsed:.2f}s"


def test_job_wait_unknown_job_returns_404_payload(client):
    import asyncio

    from sksurrogate_mcp.server import job_wait

    payload = asyncio.run(job_wait(client, "nope", timeout_s=1))
    assert "error" in payload and "404" in payload["error"], payload
    assert payload.get("status_code") == 404, payload


def test_job_diagnose_failed_job_rollup_shape(client, stub):
    from sksurrogate_mcp.server import job_diagnose

    traceback = "\n".join(f'File "line{i}.py", line {i}' for i in range(800)) + "\nValueError: kaboom at the end"
    assert len(traceback) > 4000
    stub.state.seed_job("job-fail", ["failed"], error=traceback)
    payload = job_diagnose(client, "job-fail")

    assert "error" not in payload, payload
    assert payload["job"]["status"] == "failed", payload
    # Long tracebacks are bounded to their tail (LLM-friendly), marker included.
    error_text = payload["job"]["error"]
    assert error_text.endswith("ValueError: kaboom at the end"), error_text[-120:]
    assert "[traceback truncated to its tail]" in error_text, len(error_text)
    assert len(error_text) <= 4200, len(error_text)
    # Log rollup present with counts and recent entries.
    assert payload["log_counts"].get("error") == 1, payload["log_counts"]
    assert isinstance(payload["recent_logs"], list) and payload["recent_logs"], payload["recent_logs"]
    assert any(e.get("level") == "error" for e in payload["recent_logs"])


def test_stop_and_resume_jobs(client, stub):
    from sksurrogate_mcp.server import resume_job, stop_job

    bad_mode = stop_job(client, "whatever", mode="yolo")
    assert "error" in bad_mode and "pause" in bad_mode["error"], bad_mode

    stub.state.seed_job("job-pause", ["running"])
    stopped = stop_job(client, "job-pause", mode="pause")
    assert stopped.get("status") == "paused", stopped
    resumed = resume_job(client, "job-pause")
    assert resumed.get("job_id") == "job-pause-resume" and resumed.get("status") == "queued", resumed

    missing = stop_job(client, "ghost-job")
    assert isinstance(missing, dict) and "409" in missing["error"], missing


# ---------------------------------------------------------------------------
# api_request escape hatch (snapshot-validated)
# ---------------------------------------------------------------------------


def test_api_request_allows_known_get_and_validates_shape(client, stub):
    from sksurrogate_mcp.server import api_request

    ok = api_request(client, "GET", "/api/health")
    assert ok.get("status") == "ok", ok

    unknown_route = api_request(client, "GET", "/api/definitely/not/a/route")
    assert isinstance(unknown_route, dict) and "error" in unknown_route, unknown_route
    assert "snapshot" in unknown_route["error"], unknown_route
    # And it must not have hit the wire.
    assert not [r for r in stub.state.requests if r["path"] == "/api/definitely/not/a/route"]

    bad_method = api_request(client, "TRACE", "/api/health")
    assert "error" in bad_method and "method must be one of" in bad_method["error"], bad_method

    bad_path = api_request(client, "GET", "/nope")
    assert "error" in bad_path and "/api/" in bad_path["error"], bad_path

    get_with_body = api_request(client, "get", "/api/health", body={"x": 1})
    assert "error" in get_with_body and "body is only allowed" in get_with_body["error"], get_with_body


def test_api_request_passes_json_bodies_through(client):
    from sksurrogate_mcp.server import api_request

    payload = api_request(client, "POST", "/api/experiments/demo/run", body={"preset": "regression"})
    assert payload.get("job_id") == "stub-run-job", payload


# ---------------------------------------------------------------------------
# Phase 2 — analysis tools (server-side bound clamps + G4 duplicate guards)
# ---------------------------------------------------------------------------


def test_sensitivity_analysis_validates_method_and_clamps(client, stub):
    from sksurrogate_mcp.server import sensitivity_analysis

    bad = sensitivity_analysis(client, "demo", method="fourier")
    assert "error" in bad and "method must be one of" in bad["error"], bad
    assert not [r for r in stub.state.requests if "/run" in r["path"]], (
        "an invalid method must be refused before any request goes out"
    )

    ok = sensitivity_analysis(client, "demo", n_features_to_select=0)  # clamped up to the floor of 1
    assert ok.get("job_id") == "stub-sens-demo" and ok.get("status") == "queued", ok
    assert ok["echo_body"] == {"method": "sobol", "n_features_to_select": 1}, ok

    with_partition = sensitivity_analysis(
        client, "demo", method="morris", train_partition="train_imputed"
    )
    assert with_partition["echo_body"]["train_partition"] == "train_imputed", with_partition


def test_sensitivity_analysis_refuses_duplicate_live_job(client, stub):
    from sksurrogate_mcp.server import sensitivity_analysis

    stub.state.active.append(
        {
            "job_id": "live-sens",
            "kind": "sensitivity",
            "status": "running",
            "task_name": "demo",
            "progress_message": None,
            "error": None,
        }
    )
    refused = sensitivity_analysis(client, "demo")
    assert refused.get("duplicate_job_id") == "live-sens" and "A live sensitivity job already exists" in refused["error"], (
        refused
    )


def test_top_features_clamps_num_and_passes_through(client):
    from sksurrogate_mcp.server import top_features

    result = top_features(client, "demo", num=0)  # clamped up to the floor of 1
    assert result.get("available") is True and result["num_requested"] == 1, result


def test_top_features_404_becomes_error_payload(client):
    from sksurrogate_mcp.server import top_features

    err = top_features(client, "ghost-task")
    assert err["status_code"] == 404 and "error" in err and "is_error" not in err, err


def test_explain_model_clamps_to_server_bounds(client):
    from sksurrogate_mcp.server import explain_model

    result = explain_model(
        client, "demo", "v1", max_rows=1, n_repeats=999, top_k=7, partition="train"
    )
    assert result["echo_body"] == {
        "max_rows": 20,      # server-side floor (min 20)
        "n_repeats": 20,     # server-side ceiling (max 20)
        "top_k": 6,          # server-side ceiling (max 6)
        "partition": "train",
    }, result


def test_explain_row_validates_baseline_and_clamps(client):
    from sksurrogate_mcp.server import explain_row

    bad = explain_row(client, "demo", "v1", baseline="median")
    assert "error" in bad and "baseline must be one of" in bad["error"], bad

    ok = explain_row(
        client,
        "demo",
        "v1",
        row_index=-3,  # negative index clamped into range -> 0 (row_count stubbed to 5)
        features=["x"],
        baseline="row",
        partition="train",
        baseline_row_index=4,
        orders=99,  # server-side ceiling (max 10)
    )
    assert ok["echo_body"] == {
        "row_index": 0,
        "features": ["x"],
        "baseline": "row",
        "partition": "train",
        "baseline_row_index": 4,
        "orders": 10,
    }, ok


def test_evaluation_metrics_query_param_optional(client):
    from sksurrogate_mcp.server import evaluation_metrics

    with_mv = evaluation_metrics(client, "demo", model_version="v1")
    assert with_mv.get("model_version") == "v1" and "metric_options" in with_mv, with_mv

    without_mv = evaluation_metrics(client, "demo")
    assert without_mv.get("model_version") is None, without_mv


def test_evaluation_metrics_404_for_unknown_task(client):
    from sksurrogate_mcp.server import evaluation_metrics

    err = evaluation_metrics(client, "ghost-task")
    assert err["status_code"] == 404 and "error" in err, err


def test_nested_cv_requires_exactly_one_model_selector(client):
    from sksurrogate_mcp.server import nested_cv

    both = nested_cv(client, "demo", alias="staging", model_version="v1")
    assert "error" in both and "exactly one of" in both["error"], both

    neither = nested_cv(client, "demo")
    assert "error" in neither and "exactly one of" in neither["error"], neither


def test_nested_cv_clamps_folds_and_returns_job_handle(client):
    from sksurrogate_mcp.server import nested_cv

    ok = nested_cv(client, "demo", alias="staging", inner_cv=99, outer_cv=1, repeats=99)
    assert ok.get("job_id") == "stub-nested-demo" and ok.get("status") == "queued", ok
    assert ok["echo_body"] == {
        "inner_cv": 10,   # server-side bounds: min 2, max 10
        "outer_cv": 2,    # server-side bounds: min 2, max 5
        "repeats": 5,     # server-side bounds: min 1, max 5
        "alias": "staging",
    }, ok


def test_nested_cv_refuses_duplicate_live_job(client, stub):
    from sksurrogate_mcp.server import nested_cv

    # The job store labels this kind with a hyphen — the guard must match it exactly.
    stub.state.active.append(
        {
            "job_id": "live-nested",
            "kind": "nested-cv",
            "status": "running",
            "task_name": "demo",
            "progress_message": None,
            "error": None,
        }
    )
    refused = nested_cv(client, "demo", alias="staging")
    assert refused.get("duplicate_job_id") == "live-nested" and "A live nested_cv job already exists" in refused["error"], (
        refused
    )


# ---------------------------------------------------------------------------
# Phase 2 — lifecycle (registry / promotion / quality gate / inference)
# ---------------------------------------------------------------------------


def test_registry_summary_lists_bundles_and_404s_unknown_tasks(client, stub):
    from sksurrogate_mcp.server import registry_summary

    stub.state.registry_bundles["demo"] = [{"model_version": "v1", "state": "candidate"}]

    summary = registry_summary(client, "demo")
    assert [b["model_version"] for b in summary["bundles"]] == ["v1"], summary

    err = registry_summary(client, "ghost-task")
    assert err["status_code"] == 404 and "error" in err, err


def test_promote_model_validates_state_and_forwards_fields(client):
    from sksurrogate_mcp.server import promote_model

    bad = promote_model(client, "demo", "v1", state="release")
    assert "error" in bad and "state must be one of" in bad["error"], bad

    ok = promote_model(client, "demo", "v1", state="staging", approvers=["mehdi"])
    assert ok == {
        "model_version": "v1",
        "state": "staging",
        "approvers": ["mehdi"],
        "required_approvals": 1,
    }, ok


def test_quality_gate_check_forwards_optional_thresholds(client):
    from sksurrogate_mcp.server import quality_gate_check

    minimal = quality_gate_check(client, "demo", "v1")
    assert minimal["passed"] is True and minimal["echo_body"] == {}, minimal

    with_gates = quality_gate_check(
        client,
        "demo",
        "v1",
        metric_thresholds={"neg_mean_absolute_error": -0.5},
        max_bundle_size_bytes=1_000_000,
    )
    assert with_gates["echo_body"] == {
        "metric_thresholds": {"neg_mean_absolute_error": -0.5},
        "max_bundle_size_bytes": 1_000_000,
    }, with_gates


def test_predict_requires_exactly_one_model_selector_and_passes_rows(client):
    from sksurrogate_mcp.server import predict

    both = predict(client, "demo", rows=[{"x": 1.0}], alias="staging", model_version="v1")
    assert "error" in both and "exactly one of" in both["error"], both

    ok = predict(client, "demo", rows=[{"x": 1.0}, {"x": 2.0}], alias="staging")
    assert ok.get("count") == 2 and ok.get("alias") == "staging" and len(ok["predictions"]) == 2, ok


def test_predict_batch_forwards_partition_and_selector(client):
    from sksurrogate_mcp.server import predict_batch

    ok = predict_batch(client, "demo", partition="test", model_version="v1")
    assert (
        ok.get("status") == "ok" and ok.get("partition") == "test" and ok.get("model_version") == "v1"
    ), ok


# ---------------------------------------------------------------------------
# Phase 2 — research records (append-only, human-review semantics kept)
# ---------------------------------------------------------------------------


def test_project_create_sends_full_record(client):
    from sksurrogate_mcp.server import project_create

    ok = project_create(
        client,
        name="Sobol audit",
        description="why does x dominate?",
        research_questions=["Q1"],
        artifacts=[{"artifact_id": "a1"}],
    )
    assert isinstance(ok.get("project_id"), str) and len(ok["project_id"]) == 32, ok
    assert ok["name"] == "Sobol audit" and ok["artifacts"] == [{"artifact_id": "a1"}], ok


def test_project_create_409_on_duplicate_name_is_an_error_payload(client):
    from sksurrogate_mcp.server import project_create

    first = project_create(client, name="Only one")
    assert "error" not in first
    dup = project_create(client, name="Only one")
    assert dup["status_code"] == 409 and "already exists" in dup["error"], dup


def test_project_update_preserves_untouched_fields_via_read_modify_write(client, stub):
    from sksurrogate_mcp.server import project_create, project_update

    created = project_create(
        client, name="Sobol audit", description="old description", research_questions=["Q1"]
    )
    project_id = created["project_id"]

    updated = project_update(client, project_id, description="new description")
    assert "error" not in updated, updated
    # The PUT body must carry the untouched fields back (read-modify-write).
    put_requests = [r for r in stub.state.requests if r["method"] == "PUT"]
    assert len(put_requests) == 1, put_requests
    sent = put_requests[0]["json_body"]
    assert sent == {
        "name": "Sobol audit",            # preserved by read-modify-write
        "description": "new description",
        "research_questions": ["Q1"],     # preserved
        "artifacts": [],                  # preserved (empty list)
    }, sent


def test_project_update_404s_for_unknown_project(client):
    from sksurrogate_mcp.server import project_update

    err = project_update(client, "%032x" % 999, description="nope")
    assert err["status_code"] == 404 and "error" in err, err


def test_project_note_posts_full_record_and_rejects_bad_source(client, stub):
    from sksurrogate_mcp.server import project_create, project_note

    created = project_create(client, name="Sobol audit")
    project_id = created["project_id"]

    bad = project_note(client, project_id, text="t", source="invented-source")
    assert "error" in bad and "source must be one of" in bad["error"], bad
    assert not [r for r in stub.state.requests if r["path"].endswith("/notes")]

    ok = project_note(
        client,
        project_id,
        text="x dominates (Sobol S1=0.9)",
        source="generated",
        model="gpt-x",
        citations=["f1"],
        computed_facts=[{"id": "f1", "value": 0.9}],
    )
    assert ok.get("note_id") and ok["project_id"] == project_id, ok
    note_requests = [r for r in stub.state.requests if r["path"].endswith("/notes")]
    sent = note_requests[0]["json_body"]
    assert sent == {
        "text": "x dominates (Sobol S1=0.9)",
        "source": "generated",
        "model": "gpt-x",
        "citations": ["f1"],
        "computed_facts": [{"id": "f1", "value": 0.9}],
    }, sent


def test_project_note_404s_for_unknown_project(client):
    from sksurrogate_mcp.server import project_note

    err = project_note(client, "%032x" % 999, text="t", source="human")
    assert err["status_code"] == 404 and "error" in err, err


# ---------------------------------------------------------------------------
# SDK-backed checks (optional extra; skipped when the [mcp] extra is absent)
# ---------------------------------------------------------------------------


def test_mcp_sdk_tool_registry(stub):
    pytest.importorskip("mcp.server.mcpserver")
    from sksurrogate_mcp.server import ControlPlaneClient, create_server

    server = create_server(ControlPlaneClient(stub.base_url))
    result = asyncio.run(server.list_tools())
    # mcp SDK builds differ: some return a ListToolsResult, others a bare list.
    tools_list = getattr(result, "tools", None) or list(result)
    names = sorted(t.name for t in tools_list)
    assert names == sorted(EXPECTED_TOOL_NAMES), names
    for tool in tools_list:
        assert tool.description, f"{tool.name} has no description"
    by_name = {t.name: t for t in tools_list}
    run_schema = getattr(by_name["run_experiment"], "inputSchema", None) or getattr(
        by_name["run_experiment"], "input_schema", None
    )
    props = (run_schema or {}).get("properties", {})
    assert "task_name" in props and "preset" in props, list(props)


def test_stdio_roundtrip_with_real_client(stub):
    """One full agent-style session over stdio: registry + live job_wait."""
    pytest.importorskip("mcp.client.stdio")
    import sys as _sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    stub.state.seed_job("smoke", ["queued", "completed"])

    env = {**os.environ, "SKSURROGATE_API_URL": stub.base_url, "PYTHONPATH": str(REPO_ROOT)}
    params = StdioServerParameters(
        command=_sys.executable, args=["-m", "sksurrogate_mcp.server"], env=env
    )

    async def drive() -> None:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                names = sorted(t.name for t in tools.tools)
                assert names == sorted(EXPECTED_TOOL_NAMES), names

                health_payload = json.loads(_text(await session.call_tool("health", {})))
                assert health_payload.get("status") == "ok", health_payload

                wait_result = await session.call_tool(
                    "job_wait", {"job_id": "smoke", "timeout_s": 15, "poll_interval_s": 0.4}
                )
                waited = json.loads(_text(wait_result))
                assert not getattr(wait_result, "is_error", False), _text(wait_result)
                assert waited.get("status") == "completed" and not waited.get("timed_out"), waited

                escape = await session.call_tool(
                    "api_request", {"method": "GET", "path": "/api/health"}
                )
                escaped = json.loads(_text(escape))
                assert escaped.get("version") == "0.3.1-stub", escaped

    asyncio.run(drive())


def _text(result) -> str:
    return "".join(block.text for block in result.content if getattr(block, "type", "") == "text")


def test_progress_summary_distills_trials_phases_and_best_scores():
    """job_wait's G2 rollup: newest-first entries; bools are not trials; non-finite scores ignored."""
    from sksurrogate_mcp.server import _progress_summary

    payload = {
        "entries": [
            {"level": "INFO", "message": "plain text lines carry no structure"},
            {"trial": 7, "phase": "refit", "score": -138.2},
            {"trial": 6, "phase": "search", "score": -140.9},
            "a scalar entry is ignored",
            {"message": "no structure here"},
        ]
    }
    assert _progress_summary(payload) == {
        "max_trial": 7,
        "best_score": -138.2,
        "phase": "refit",
    }

    # bools are not trials and non-finite scores never win best_score
    tricky = {"entries": [{"trial": True, "score": float("inf")}, {"message": "x"}]}
    assert _progress_summary(tricky) == {}

    # malformed payloads degrade to no progress rather than raising
    assert _progress_summary(None) == {}
    assert _progress_summary("not a dict") == {}
    assert _progress_summary({"entries": []}) == {}
