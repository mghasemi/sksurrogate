"""Regression tests for the opt-in shared-key API authentication middleware.

``SKSURROGATE_API_KEY`` is resolved into ``settings.api_key`` at import time, and
:class:`api.main.ApiKeyMiddleware` reads that attribute on every request — so
these tests patch ``settings.api_key`` directly via monkeypatch instead of
re-importing the app (mirroring how :mod:`tests.test_api_synthesis` redirects
the storage root).

Coverage: default-open behaviour, HTTP 401 for missing/wrong keys in both header
styles (``X-API-Key`` and ``Authorization: Bearer``), the health-endpoint
exemption, WebSocket handshake policy on the job stream, and the SPA fallback
staying reachable without a key.
"""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import app, _check_auth_settings  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("api-key-auth")
    patcher = pytest.MonkeyPatch()
    patcher.setattr(settings, "root", root)
    settings.ensure()
    yield TestClient(app)
    patcher.undo()


@pytest.fixture(autouse=True)
def key_unset_by_default(monkeypatch):
    """Pin the v1 default: no key configured means an open API.

    Every test that needs authentication enabled opts in through the ``key``
    fixture below, which runs after this one and overrides the attribute.
    """
    monkeypatch.setattr(settings, "api_key", None)


@pytest.fixture()
def key(monkeypatch):
    """Enable shared-key auth for a single test."""
    monkeypatch.setattr(settings, "api_key", "unit-test-api-key")
    return "unit-test-api-key"


def test_open_when_no_key_configured(client):
    res = client.get("/api/tasks")
    assert res.status_code == 200
    assert isinstance(res.json().get("tasks"), list)


def test_missing_key_rejected_with_401(client, key):
    res = client.get("/api/tasks")
    assert res.status_code == 401
    assert res.json()["detail"] == "Authentication required"
    assert res.headers.get("www-authenticate")


def test_x_api_key_header_accepted(client, key):
    res = client.get("/api/tasks", headers={"X-API-Key": key})
    assert res.status_code == 200


def test_wrong_x_api_key_rejected(client, key):
    res = client.get("/api/tasks", headers={"X-API-Key": "not-the-key"})
    assert res.status_code == 401


def test_bearer_token_accepted_and_wrong_bearer_rejected(client, key):
    ok = client.get("/api/tasks", headers={"Authorization": f"Bearer {key}"})
    assert ok.status_code == 200

    bad = client.get("/api/tasks", headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401


def test_health_endpoint_exempt_from_key(client, key):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_websocket_handshake_rejected_without_key(client, key):
    with pytest.raises(WebSocketDisconnect) as excinfo:
        # websocket_connect is a context-manager factory: the handshake (and its
        # rejection) happens on __enter__.
        with client.websocket_connect("/api/jobs/no-such-job/stream"):
            pass  # not reached — entering raises WebSocketDisconnect
    # 1008 = policy violation (the middleware's rejection code for bad keys).
    assert excinfo.value.code == 1008


def test_websocket_stream_accepted_with_key_query_param(client, key):
    url = f"/api/jobs/no-such-job/stream?api_key={key}"
    with client.websocket_connect(url) as ws:
        # Auth passed; the endpoint then reports the missing job and closes.
        assert ws.receive_json() == {"error": "job not found"}


def test_websocket_wrong_key_query_param_rejected(client, key):
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/api/jobs/no-such-job/stream?api_key=wrong"):
            pass  # not reached — entering raises WebSocketDisconnect
    assert excinfo.value.code == 1008


def test_spa_fallback_not_gated_by_key(client, key):
    """Static UI assets and the index fallback must stay reachable without a
    key — only ``/api`` scopes are protected.

    ``/`` serves the SPA only when ``webui/dist`` has been built (it is not
    built in CI's test job), so both 200 (built UI) and 404 (no dist) are
    correct here. What matters is that the middleware never answers 401/403
    for a non-``/api`` path.
    """
    res = client.get("/")
    assert res.status_code in (200, 404), res.status_code


# --- Startup validation (SKSURROGATE_REQUIRE_AUTH) --------------------------
# _check_auth_settings runs at api.main import time; the unit tests below call
# it directly with patched settings/env, and the subprocess pins at the bottom
# exercise the real module-level path in fresh interpreters.


def test_no_key_without_requirement_warns(client, caplog):
    """Default configuration logs a one-line startup warning."""
    with caplog.at_level(logging.WARNING, logger="sksurrogate.api"):
        _check_auth_settings()
    assert any("authentication disabled" in r.message.lower() for r in caplog.records)


def test_key_configured_does_not_warn(client, key, caplog):
    with caplog.at_level(logging.WARNING, logger="sksurrogate.api"):
        _check_auth_settings()
    assert not [r for r in caplog.records if r.name == "sksurrogate.api"]


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_require_auth_without_key_refuses_startup(client, monkeypatch, value):
    monkeypatch.setenv("SKSURROGATE_REQUIRE_AUTH", value)
    with pytest.raises(RuntimeError, match="SKSURROGATE_API_KEY"):
        _check_auth_settings()


def test_require_auth_falsy_value_only_warns(client, monkeypatch, caplog):
    monkeypatch.setenv("SKSURROGATE_REQUIRE_AUTH", "0")
    with caplog.at_level(logging.WARNING, logger="sksurrogate.api"):
        _check_auth_settings()  # no raise — warning only
    assert any("authentication disabled" in r.message.lower() for r in caplog.records)


def test_require_auth_with_key_passes(client, key, monkeypatch):
    monkeypatch.setenv("SKSURROGATE_REQUIRE_AUTH", "true")
    _check_auth_settings()  # no exception expected


def _import_api_main(env_extra: dict) -> subprocess.CompletedProcess:
    """Import ``api.main`` in a fresh interpreter with only the given
    SKSURROGATE_* variables set (host env values are stripped so the test is
    deterministic).

    The API home is always redirected to a temporary directory: whether the
    startup check passes depends on the account table, so inheriting the
    developer's ``var/sksurrogate-api`` would make this test order-dependent.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("SKSURROGATE_")}
    env.update(env_extra)
    if "SKSURROGATE_API_HOME" not in env:
        env["SKSURROGATE_API_HOME"] = tempfile.mkdtemp(prefix="sksurrogate-auth-probe-")
    return subprocess.run(
        [sys.executable, "-c", "import api.main"],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_subprocess_import_refused_when_require_auth_and_no_key():
    res = _import_api_main({"SKSURROGATE_REQUIRE_AUTH": "1"})
    assert res.returncode != 0
    assert "SKSURROGATE_API_KEY" in res.stderr


def test_subprocess_import_ok_with_require_auth_and_key():
    res = _import_api_main(
        {"SKSURROGATE_REQUIRE_AUTH": "true", "SKSURROGATE_API_KEY": "subprocess-key"}
    )
    assert res.returncode == 0, res.stderr


# --- Schema/documentation gating -------------------------------------------


def test_schema_and_docs_open_when_no_key(client):
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_schema_and_docs_gated_when_key_configured(client, key):
    for path in ("/openapi.json", "/docs", "/redoc"):
        assert client.get(path).status_code == 401, path
        assert client.get(path, headers={"X-API-Key": key}).status_code == 200, path


# --- Failed-authentication audit -------------------------------------------


def test_failed_auth_is_logged_without_the_supplied_secret(client, key, caplog):
    supplied = "wrong-secret-value"
    with caplog.at_level(logging.WARNING, logger="sksurrogate.api"):
        res = client.get("/api/tasks", headers={"X-API-Key": supplied})
    assert res.status_code == 401
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "Authentication rejected" in messages
    assert supplied not in messages
