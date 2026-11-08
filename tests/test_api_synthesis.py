"""API tests for Stage D2 — opt-in LLM-assisted project synthesis.

The provider is exercised with a local stub HTTP server implementing the
OpenAI-compatible chat-completions contract; no test touches the network or a
real model. Assertions cover: disabled-by-default behaviour, citation
enforcement (unknown/uncited drafts), bounded aggregate-only prompts, provider
error mapping, and generated-note provenance.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("api-synthesis")
    patcher = pytest.MonkeyPatch()
    patcher.setattr(settings, "root", root)
    settings.ensure()
    yield TestClient(app)
    patcher.undo()


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch):
    for name in (
        "SKSURROGATE_LLM_BASE_URL",
        "SKSURROGATE_LLM_MODEL",
        "SKSURROGATE_LLM_API_KEY",
        "SKSURROGATE_LLM_TIMEOUT",
        "SKSURROGATE_LLM_MAX_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def stub_provider():
    """Local HTTP server standing in for an OpenAI-compatible endpoint."""
    captured = []
    state = {
        "content": "Temperature dominates the response [S1]; the bundle reports one metric [S3].",
        "status": 200,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server API
            length = int(self.headers.get("content-length", 0))
            raw = self.rfile.read(length).decode("utf-8")
            captured.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("authorization"),
                    "body": json.loads(raw),
                }
            )
            if state["status"] != 200:
                payload = b"provider exploded"
                self.send_response(state["status"])
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            payload = json.dumps(
                {"choices": [{"message": {"content": state["content"]}}]}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):  # keep test output clean
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "base_url": "http://127.0.0.1:%d/v1" % server.server_port,
            "captured": captured,
            "state": state,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def synthesis_task(client):
    rng = np.random.RandomState(5)
    n = 120
    values = rng.normal(size=(n, 3))
    frame = pd.DataFrame(values, columns=["temperature", "pressure", "flow"])
    frame["y"] = (
        1.5 * values[:, 0] - 0.9 * values[:, 1] + 0.2 * values[:, 0] * values[:, 2]
        + 0.1 * rng.normal(size=n)
    )
    response = client.post(
        "/api/datasets/synth/register",
        data={"target": "y", "partition": "train"},
        files={"file": ("t.csv", frame.to_csv(index=False), "text/csv")},
    )
    assert response.status_code == 200, response.text
    trained = client.post(
        "/api/bundles/synth/train-baseline",
        json={"estimator": "random_forest_regressor", "validation_partition": None},
    )
    assert trained.status_code == 200, trained.text
    return {"task": "synth", "version": trained.json()["model_version"], "frame": frame}


@pytest.fixture()
def project(client):
    created = client.post(
        "/api/projects",
        json={"name": "Synthesis project", "description": "D2 test", "research_questions": []},
    )
    assert created.status_code == 200, created.text
    return created.json()["project_id"]


def _wait(client, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = client.get("/api/jobs/" + job_id).json()
        if record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.3)
    raise AssertionError("synthesis job did not finish")


def _configure(monkeypatch, stub):
    monkeypatch.setenv("SKSURROGATE_LLM_BASE_URL", stub["base_url"])
    monkeypatch.setenv("SKSURROGATE_LLM_MODEL", "stub-model")
    monkeypatch.setenv("SKSURROGATE_LLM_API_KEY", "test-key")


def test_status_reports_unconfigured_without_leaking_secrets(client):
    payload = client.get("/api/synthesis/status").json()
    assert payload["enabled"] is False
    assert payload["provider"]["configured"] is False
    assert payload["provider"]["model"] is None
    assert "api_key" not in json.dumps(payload)
    assert "SKSURROGATE_LLM_BASE_URL" in payload["instructions"]


def test_submit_is_refused_when_unconfigured(client, project, synthesis_task):
    response = client.post(
        "/api/synthesis/projects/%s" % project,
        json={"task_name": synthesis_task["task"], "model_version": synthesis_task["version"]},
    )
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


def test_happy_path_returns_cited_draft_and_bounded_prompt(
    client, monkeypatch, stub_provider, project, synthesis_task
):
    _configure(monkeypatch, stub_provider)
    response = client.post(
        "/api/synthesis/projects/%s" % project,
        json={
            "task_name": synthesis_task["task"],
            "model_version": synthesis_task["version"],
            "include_explanations": True,
            "instruction": "Emphasize the measured feature effects.",
        },
    )
    assert response.status_code == 200, response.text
    job = _wait(client, response.json()["job_id"])
    assert job["status"] == "completed", job.get("error")
    result = job["result"]

    assert result["draft"]["citations"] == ["S1", "S3"]
    assert result["draft"]["text"].startswith("Temperature dominates")
    assert result["draft"]["word_count"] > 0
    assert [source["id"] for source in result["sources"]] == ["S1", "S3", "S4"]
    assert result["computed_facts"][0]["kind"] == "dataset_profile"
    assert result["computed_facts"][0]["data"]["rows"] == 120
    assert result["computed_facts"][2]["kind"] == "explanations"
    assert result["provider"]["model"] == "stub-model"
    assert result["interpretation"]["category"] == "generated_interpretation"
    assert result["interpretation"]["causal_effect"] is False

    # The provider request is well-formed, authenticated, and carries aggregates only.
    assert len(stub_provider["captured"]) == 1
    call = stub_provider["captured"][0]
    assert call["path"] == "/v1/chat/completions"
    assert call["authorization"] == "Bearer test-key"
    assert call["body"]["model"] == "stub-model"
    assert call["body"]["temperature"] == 0
    prompt = json.dumps(call["body"]["messages"])
    assert "Emphasize the measured feature effects." in prompt
    assert "train.csv" not in prompt
    # Source facts must be aggregate structures, never row payloads.
    user_message = call["body"]["messages"][1]["content"]
    facts_json = user_message.split("Write the project note", 1)[1]
    facts_json = facts_json[facts_json.index("{"): facts_json.index("\n\nAdditional user instruction")]
    facts = json.loads(facts_json)["sources"]
    for fact in facts:
        assert "values" not in fact["data"]
        assert "records" not in fact["data"]
        assert "frame" not in fact["data"]
    assert len(prompt) < 60000


def test_unknown_citation_fails_the_job(
    client, monkeypatch, stub_provider, project, synthesis_task
):
    _configure(monkeypatch, stub_provider)
    stub_provider["state"]["content"] = "A claim with invented provenance [S9]."
    response = client.post(
        "/api/synthesis/projects/%s" % project,
        json={"task_name": synthesis_task["task"], "model_version": synthesis_task["version"]},
    )
    job = _wait(client, response.json()["job_id"])
    assert job["status"] == "failed"
    assert "unknown source" in job["error"]


def test_uncited_draft_fails_the_job(client, monkeypatch, stub_provider, project, synthesis_task):
    _configure(monkeypatch, stub_provider)
    stub_provider["state"]["content"] = "Temperature matters, apparently."
    response = client.post(
        "/api/synthesis/projects/%s" % project,
        json={"task_name": synthesis_task["task"], "model_version": synthesis_task["version"]},
    )
    job = _wait(client, response.json()["job_id"])
    assert job["status"] == "failed"
    assert "cites no sources" in job["error"]


def test_provider_http_error_is_reported(
    client, monkeypatch, stub_provider, project, synthesis_task
):
    _configure(monkeypatch, stub_provider)
    stub_provider["state"]["status"] = 500
    response = client.post(
        "/api/synthesis/projects/%s" % project,
        json={"task_name": synthesis_task["task"], "model_version": synthesis_task["version"]},
    )
    job = _wait(client, response.json()["job_id"])
    assert job["status"] == "failed"
    assert "HTTP 500" in job["error"]


def test_generated_note_requires_and_persists_provenance(client, project):
    body = {
        "text": "Temperature dominates the response [S1].",
        "source": "generated",
        "citations": ["S1"],
        "computed_facts": [{"id": "S1", "kind": "dataset_profile", "data": {"rows": 120}}],
    }
    missing = client.post("/api/projects/%s/notes" % project, json=body)
    assert missing.status_code == 422
    assert "provider model" in missing.text

    wrong = client.post(
        "/api/projects/%s/notes" % project,
        json={**body, "model": "stub-model", "citations": ["S9"]},
    )
    assert wrong.status_code == 422
    assert "no matching computed fact" in wrong.text

    accepted = client.post(
        "/api/projects/%s/notes" % project,
        json={
            **body,
            "model": "stub-model",
            "sources": [{"id": "S1", "kind": "dataset_profile", "label": "profile"}],
            "synthesis_job_id": "abc123",
        },
    )
    assert accepted.status_code == 200, accepted.text
    note = accepted.json()["notes"][-1]
    assert note["source"] == "generated"
    assert note["model"] == "stub-model"
    assert note["citations"] == ["S1"]
    assert note["computed_facts"][0]["id"] == "S1"
    assert note["synthesis_job_id"] == "abc123"

    human = client.post("/api/projects/%s/notes" % project, json={"text": "A human note."})
    assert human.status_code == 200, human.text
    human_note = human.json()["notes"][-1]
    assert human_note["source"] == "human"
    assert "citations" not in human_note
