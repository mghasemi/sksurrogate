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

import pytest
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402


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
    assert res.json()["detail"] == "Invalid or missing API key"


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
    key — only ``/api`` scopes are protected."""
    res = client.get("/")
    assert res.status_code == 200
