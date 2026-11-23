"""Phase 1 tests for the sksurrogate_mcp package (core loop tools).

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

    def _record(self, method: str, parsed) -> dict:
        entry = {
            "method": method,
            "path": parsed.path,
            "query": {k: v[-1] for k, v in urllib.parse.parse_qs(parsed.query).items()},
        }
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
    "health",
    "list_tasks",
    "task_overview",
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
    from sksurrogate_mcp.server import job_wait

    stub.state.seed_job("job-done", ["queued", "running", "completed"])
    payload = job_wait(client, "job-done", timeout_s=10, poll_interval_s=0.2)
    assert payload.get("status") == "completed" and not payload.get("timed_out"), payload
    assert payload.get("result") == {"best_score": -12.5}, payload


def test_job_wait_times_out_on_nonterminal_status(client, stub):
    import time

    from sksurrogate_mcp.server import job_wait

    stub.state.seed_job("job-slow", ["queued", "running"])  # never terminal
    started = time.monotonic()
    payload = job_wait(client, "job-slow", timeout_s=1.0, poll_interval_s=0.3)
    elapsed = time.monotonic() - started
    assert payload.get("timed_out") is True and payload.get("status") == "running", payload
    assert 0.8 <= elapsed < 5.0, f"wait loop must respect its budget; took {elapsed:.2f}s"


def test_job_wait_unknown_job_returns_404_payload(client):
    from sksurrogate_mcp.server import job_wait

    payload = job_wait(client, "nope", timeout_s=1)
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
