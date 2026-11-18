"""Smoke sweep: every parameter-less ``GET /api`` route answers without a 500.

This exists because a router edit can introduce a ``NameError`` (a missing
import, e.g. ``acl`` referenced by a list filter) that no unit test notices when
the affected endpoint happens to lack one: the endpoint only runs when a browser
or client calls it. Enumerating the app's own routes means a missing import in a
list endpoint fails here instead of in the UI.

Both postures are exercised: the open (single-user) API, and an enforced API
where the shared key resolves to an admin principal, so each route runs its ACL
code path rather than short-circuiting on "auth disabled".

Routes that need path parameters cannot be called generically; they are covered
by the endpoint-specific suites.
"""

import pytest
from fastapi.testclient import TestClient

from api.config import settings
from api.main import app

#: Statuses a read-only, parameter-less endpoint may legitimately return.
ALLOWED = {200, 204, 404, 409, 422, 503}


def _parameterless_api_get_routes():
    """Every ``GET /api/...`` without path parameters, from the OpenAPI schema.

    Reading the schema rather than ``app.routes`` is deliberate: included routers
    are nested objects in current FastAPI, so walking ``app.routes`` only sees the
    routes registered directly on the application.
    """
    seen = []
    for path, operations in app.openapi()["paths"].items():
        if "get" not in operations:
            continue
        if not path.startswith("/api") or "{" in path:
            continue
        if path in {"/api/health", "/api/auth/status", "/api/auth/login"}:
            continue  # public by design and covered elsewhere
        seen.append(path)
    return sorted(set(seen))


ROUTES = _parameterless_api_get_routes()


@pytest.fixture()
def open_client(tmp_path, monkeypatch):
    """Open mode: one anonymous user, no enforcement, no filtering."""
    monkeypatch.setattr(settings, "root", tmp_path)
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "require_auth", False)
    settings.ensure()
    from api import acl, auth

    auth.reset_store()
    acl.reset_store()
    return TestClient(app)


@pytest.fixture()
def enforced_client(tmp_path, monkeypatch):
    """Enforced mode with the shared key, which resolves to an admin principal."""
    monkeypatch.setattr(settings, "root", tmp_path)
    monkeypatch.setattr(settings, "api_key", "smoke-shared-key")
    monkeypatch.setattr(settings, "require_auth", True)
    settings.ensure()
    from api import acl, auth

    auth.reset_store()
    acl.reset_store()
    return TestClient(app, headers={"X-API-Key": "smoke-shared-key"})


def test_route_enumeration_is_not_empty():
    # Guard against the sweep silently checking nothing (e.g. a routing refactor).
    assert len(ROUTES) >= 12, ROUTES


@pytest.mark.parametrize("path", ROUTES)
def test_open_mode_route_has_no_server_error(open_client, path):
    response = open_client.get(path)
    assert response.status_code != 500, (path, response.status_code, response.text[:500])
    assert response.status_code in ALLOWED, (path, response.status_code, response.text[:200])


@pytest.mark.parametrize("path", ROUTES)
def test_enforced_mode_route_has_no_server_error(enforced_client, path):
    response = enforced_client.get(path)
    assert response.status_code != 500, (path, response.status_code, response.text[:500])
    assert response.status_code in ALLOWED, (path, response.status_code, response.text[:200])


def test_jobs_list_returns_records_and_is_filterable(open_client):
    """The endpoint that produced a live 500: it must also be exercised for data."""
    response = open_client.get("/api/jobs")
    assert response.status_code == 200
    assert isinstance(response.json().get("jobs"), list)
    assert open_client.get("/api/jobs", params={"task_name": "no-such-task"}).json()["jobs"] == []
