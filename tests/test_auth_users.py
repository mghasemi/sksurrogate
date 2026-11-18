"""Tests for the user/session/API-key identity layer (``api.auth``).

Two layers are covered:

* **Store** — hashing, username uniqueness, session expiry, key revocation,
  account deactivation, last-admin protection, and the login throttle. These
  need no HTTP stack and are the security-critical part.
* **HTTP** — the middleware's enforcement contract: anonymous rejection,
  cookie login + CSRF, per-user API keys, admin-only user management, the
  legacy shared key still resolving to an admin principal, and the
  backwards-compatible open mode.

Auth state is file-backed (``settings.auth_db_path``), so every fixture points
``settings.root`` at a temporary directory and drops the cached store — the same
pattern :mod:`tests.test_api_key_auth` uses for ``settings.api_key``.
"""

import logging

import pytest
from fastapi.testclient import TestClient

from api import audit, auth
from api.config import settings
from api.main import app, _check_auth_settings

ORIGIN = {"Origin": "http://testserver", "Host": "testserver"}
ADMIN = ("admin", "admin-password-1")
ANALYST = ("analyst", "analyst-password-1")


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A fresh auth database per test, with enforcement off unless opted in."""
    monkeypatch.setattr(settings, "root", tmp_path)
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "require_auth", False)
    settings.ensure()
    auth.reset_store()
    audit.clear()
    auth.THROTTLE._failures.clear()
    yield auth.store()
    auth.reset_store()


@pytest.fixture()
def enforced(store, monkeypatch):
    """Auth enforced with one administrator account already present."""
    monkeypatch.setattr(settings, "require_auth", True)
    store.create_user(*ADMIN, is_admin=True)
    return store


@pytest.fixture()
def client(enforced):
    return TestClient(app)


def _login(client, credentials=ADMIN):
    return client.post(
        "/api/auth/login",
        json={"username": credentials[0], "password": credentials[1]},
        headers=ORIGIN,
    )


# --------------------------------------------------------------------------- #
# Store: passwords and accounts
# --------------------------------------------------------------------------- #


def test_password_digest_roundtrip_and_rejection(store):
    digest = auth.hash_password("hunter2-hunter2")
    assert digest.startswith("scrypt$")
    assert "hunter2" not in digest
    assert auth.verify_password(digest, "hunter2-hunter2")
    assert not auth.verify_password(digest, "hunter2-hunter3")
    assert not auth.verify_password("garbage", "hunter2-hunter2")
    assert not auth.verify_password(digest, "")


def test_short_password_rejected(store):
    with pytest.raises(auth.AuthError):
        auth.hash_password("short")


def test_username_uniqueness_is_case_insensitive(store):
    store.create_user("Mehdi", "a-password-1")
    with pytest.raises(auth.UsernameTaken):
        store.create_user("mehdi", "a-password-2")
    assert store.get_user_by_username("MEHDI")["username"] == "Mehdi"


@pytest.mark.parametrize("bad", ["", "ab", "has space", "../evil", "x" * 65, "-leading"])
def test_username_policy(store, bad):
    with pytest.raises(auth.AuthError):
        store.create_user(bad, "a-password-1")


def test_authenticate_returns_none_for_unknown_user_without_raising(store):
    assert store.authenticate("ghost", "whatever-1") is None


def test_authenticate_refuses_inactive_account(store):
    user = store.create_user("dormant", "a-password-1")
    store.set_active(user["user_id"], False)
    assert store.authenticate("dormant", "a-password-1") is None


# --------------------------------------------------------------------------- #
# Store: sessions
# --------------------------------------------------------------------------- #


def test_session_resolves_and_is_stored_hashed(store, tmp_path):
    user = store.create_user(*ADMIN, is_admin=True)
    token = store.create_session(user["user_id"], ttl_seconds=60)
    resolved = store.resolve_session(token)
    assert resolved is not None and resolved["username"] == ADMIN[0]
    assert store.resolve_session("not-a-token") is None

    import sqlite3

    with sqlite3.connect(str(settings.auth_db_path)) as connection:
        stored = [row[0] for row in connection.execute("SELECT token_hash FROM sessions")]
    assert token not in stored
    assert auth.digest_token(token) in stored


def test_expired_session_is_rejected_and_deleted(store):
    user = store.create_user(*ADMIN, is_admin=True)
    token = store.create_session(user["user_id"], ttl_seconds=-1)
    assert store.resolve_session(token) is None
    assert store.list_api_keys(user["user_id"]) == []


def test_delete_session_logs_out(store):
    user = store.create_user(*ADMIN, is_admin=True)
    token = store.create_session(user["user_id"], ttl_seconds=60)
    store.delete_session(token)
    assert store.resolve_session(token) is None


def test_deactivating_account_drops_sessions(store):
    user = store.create_user(*ANALYST)
    token = store.create_session(user["user_id"], ttl_seconds=60)
    store.set_active(user["user_id"], False)
    assert store.resolve_session(token) is None


# --------------------------------------------------------------------------- #
# Store: API keys
# --------------------------------------------------------------------------- #


def test_api_key_is_returned_once_and_stored_hashed(store):
    user = store.create_user(*ADMIN, is_admin=True)
    secret, record = store.create_api_key(user["user_id"], "ci")
    assert secret.startswith(auth.API_KEY_PREFIX)
    assert record["prefix"] == secret[: len(auth.API_KEY_PREFIX) + 6]
    assert secret[len(auth.API_KEY_PREFIX) + 6 :] not in str(record)

    import sqlite3

    with sqlite3.connect(str(settings.auth_db_path)) as connection:
        stored = [row[0] for row in connection.execute("SELECT key_hash FROM api_keys")]
    assert secret not in stored
    assert store.resolve_api_key(secret)["username"] == ADMIN[0]
    assert store.resolve_api_key("sksk_bogus") is None
    assert store.resolve_api_key("") is None


def test_revoked_key_no_longer_resolves(store):
    user = store.create_user(*ADMIN, is_admin=True)
    secret, record = store.create_api_key(user["user_id"], "ci")
    store.revoke_api_key(record["id"])
    assert store.resolve_api_key(secret) is None


def test_key_cannot_be_revoked_by_another_user(store):
    owner = store.create_user(*ADMIN, is_admin=True)
    other = store.create_user(*ANALYST)
    _secret, record = store.create_api_key(owner["user_id"], "ci")
    with pytest.raises(KeyError):
        store.revoke_api_key(record["id"], user_id=other["user_id"])


def test_key_of_deactivated_user_stops_working(store):
    user = store.create_user(*ANALYST)
    secret, _ = store.create_api_key(user["user_id"], "ci")
    store.set_active(user["user_id"], False)
    assert store.resolve_api_key(secret) is None


# --------------------------------------------------------------------------- #
# Store: admin invariants and throttling
# --------------------------------------------------------------------------- #


def test_last_admin_cannot_be_disabled_or_demoted(store):
    admin = store.create_user(*ADMIN, is_admin=True)
    with pytest.raises(auth.LastAdminError):
        store.set_active(admin["user_id"], False)
    with pytest.raises(auth.LastAdminError):
        store.set_admin(admin["user_id"], False)


def test_second_admin_allows_demotion(store):
    first = store.create_user(*ADMIN, is_admin=True)
    store.create_user("second-admin", "another-password-1", is_admin=True)
    assert store.set_admin(first["user_id"], False)["is_admin"] is False


def test_login_throttle_locks_after_repeated_failures(store):
    throttle = auth.LoginThrottle(max_failures=3, window_seconds=60)
    assert not throttle.locked("mehdi", "10.0.0.1")
    for _ in range(3):
        throttle.record_failure("mehdi", "10.0.0.1")
    assert throttle.locked("mehdi", "10.0.0.1")
    assert not throttle.locked("mehdi", "10.0.0.2")
    throttle.reset("mehdi", "10.0.0.1")
    assert not throttle.locked("mehdi", "10.0.0.1")


def test_bootstrap_from_env_creates_admin_once(store, monkeypatch):
    monkeypatch.setenv("SKSURROGATE_ADMIN_USERNAME", "root")
    monkeypatch.setenv("SKSURROGATE_ADMIN_PASSWORD", "root-password-1")
    created = auth.bootstrap_from_env()
    assert created["is_admin"] is True
    assert auth.bootstrap_from_env() is None
    assert store.count_users() == 1


# --------------------------------------------------------------------------- #
# HTTP: enforcement
# --------------------------------------------------------------------------- #


def test_status_is_public_and_reports_login_available(client):
    res = client.get("/api/auth/status")
    assert res.status_code == 200
    assert res.json()["auth_enforced"] is True
    assert res.json()["login_available"] is True
    assert res.json()["bootstrap_required"] is False


def test_health_stays_public(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_anonymous_api_request_is_rejected(client):
    res = client.get("/api/tasks")
    assert res.status_code == 401
    assert res.json()["detail"] == "Authentication required"


def test_login_sets_httponly_strict_cookie(client):
    res = _login(client)
    assert res.status_code == 200
    cookie = res.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "samesite=strict" in cookie.lower()
    assert auth.SESSION_COOKIE in cookie


def test_wrong_password_is_rejected_without_cookie(client):
    res = client.post(
        "/api/auth/login", json={"username": ADMIN[0], "password": "wrong-password"}, headers=ORIGIN
    )
    assert res.status_code == 401
    assert res.json()["detail"] == "Invalid username or password"
    assert auth.SESSION_COOKIE not in res.cookies


def test_session_cookie_authorises_api_calls(client):
    _login(client)
    me = client.get("/api/auth/me").json()
    assert me["username"] == ADMIN[0]
    assert me["source"] == "session"
    assert client.get("/api/tasks").status_code == 200


def test_logout_clears_the_session(client):
    _login(client)
    assert client.post("/api/auth/logout", headers=ORIGIN).status_code == 200
    assert client.get("/api/tasks").status_code == 401


def test_cookie_authenticated_mutation_needs_same_origin(client):
    _login(client)
    assert client.post("/api/auth/keys", json={"name": "ci"}).status_code == 403
    assert client.post("/api/auth/keys", json={"name": "ci"}, headers=ORIGIN).status_code == 200
    bad = client.post(
        "/api/auth/keys", json={"name": "ci"}, headers={"Origin": "http://evil.example", "Host": "testserver"}
    )
    assert bad.status_code == 403


def test_per_user_api_key_authorises_api_calls(client):
    _login(client)
    created = client.post("/api/auth/keys", json={"name": "ci"}, headers=ORIGIN).json()
    secret = created["key"]

    other = TestClient(app)
    assert other.get("/api/tasks").status_code == 401
    assert other.get("/api/tasks", headers={"X-API-Key": secret}).status_code == 200
    me = other.get("/api/auth/me", headers={"Authorization": f"Bearer {secret}"}).json()
    assert me["username"] == ADMIN[0]
    assert me["source"] == "api_key"


def test_api_key_is_listed_without_secret(client):
    _login(client)
    secret = client.post("/api/auth/keys", json={"name": "ci"}, headers=ORIGIN).json()["key"]
    listed = client.get("/api/auth/keys").json()["keys"]
    assert len(listed) == 1
    assert listed[0]["prefix"] == secret[: len(auth.API_KEY_PREFIX) + 6]
    assert secret not in str(listed)


def test_revoked_key_is_rejected(client):
    _login(client)
    created = client.post("/api/auth/keys", json={"name": "ci"}, headers=ORIGIN).json()
    assert client.delete(f"/api/auth/keys/{created['record']['id']}", headers=ORIGIN).status_code == 200
    other = TestClient(app)
    assert other.get("/api/tasks", headers={"X-API-Key": created["key"]}).status_code == 401


def test_legacy_shared_key_still_resolves_to_admin_principal(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "legacy-shared-key")
    other = TestClient(app)
    assert other.get("/api/tasks", headers={"X-API-Key": "wrong"}).status_code == 401
    me = other.get("/api/auth/me", headers={"X-API-Key": "legacy-shared-key"}).json()
    assert me["is_admin"] is True
    assert me["source"] == "static"
    assert me["user"] is None


# --------------------------------------------------------------------------- #
# HTTP: accounts and keys management
# --------------------------------------------------------------------------- #


def test_non_admin_cannot_manage_users(client):
    _login(client)
    client.post(
        "/api/auth/users", json={"username": ANALYST[0], "password": ANALYST[1]}, headers=ORIGIN
    )
    client.post("/api/auth/logout", headers=ORIGIN)
    _login(client, ANALYST)
    assert client.get("/api/auth/users").status_code == 403
    assert client.post(
        "/api/auth/users", json={"username": "third", "password": "third-password-1"}, headers=ORIGIN
    ).status_code == 403


def test_admin_creates_and_deactivates_user(client):
    _login(client)
    created = client.post(
        "/api/auth/users", json={"username": ANALYST[0], "password": ANALYST[1]}, headers=ORIGIN
    )
    assert created.status_code == 201
    user_id = created.json()["user"]["user_id"]

    analyst = TestClient(app)
    _login(analyst, ANALYST)
    assert analyst.get("/api/tasks").status_code == 200

    assert client.patch(
        f"/api/auth/users/{user_id}", json={"is_active": False}, headers=ORIGIN
    ).status_code == 200
    assert analyst.get("/api/tasks").status_code == 401
    assert _login(analyst, ANALYST).status_code == 401


def test_duplicate_username_is_a_conflict(client):
    _login(client)
    client.post("/api/auth/users", json={"username": ANALYST[0], "password": ANALYST[1]}, headers=ORIGIN)
    dupe = client.post(
        "/api/auth/users", json={"username": "ANALYST", "password": ANALYST[1]}, headers=ORIGIN
    )
    assert dupe.status_code == 409


def test_password_change_revokes_other_sessions(client):
    second = TestClient(app)
    _login(client)
    _login(second)
    changed = client.post(
        "/api/auth/password",
        json={"current_password": ADMIN[1], "new_password": "rotated-password-1"},
        headers=ORIGIN,
    )
    assert changed.status_code == 200
    assert second.get("/api/tasks").status_code == 401
    assert client.get("/api/tasks").status_code == 200
    assert client.post("/api/auth/password", json={"current_password": "nope", "new_password": "whatever-1"}, headers=ORIGIN).status_code == 400


def test_login_is_throttled_after_repeated_failures(client, monkeypatch):
    monkeypatch.setattr(auth.THROTTLE, "max_failures", 2)
    bad = {"username": ADMIN[0], "password": "wrong-password"}
    for _ in range(2):
        assert client.post("/api/auth/login", json=bad, headers=ORIGIN).status_code == 401
    locked = client.post("/api/auth/login", json=bad, headers=ORIGIN)
    assert locked.status_code == 429
    # The real password is refused while the throttle is engaged.
    assert _login(client).status_code == 429


def test_failed_login_is_logged_without_the_supplied_secret(client, caplog):
    supplied = "do-not-log-this-1"
    with caplog.at_level(logging.WARNING, logger="sksurrogate.api"):
        client.post("/api/auth/login", json={"username": ADMIN[0], "password": supplied}, headers=ORIGIN)
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert supplied not in messages
    assert all(supplied not in str(record.args) for record in caplog.records)


# --------------------------------------------------------------------------- #
# Startup validation and open mode
# --------------------------------------------------------------------------- #


def test_bootstrap_env_satisfies_require_auth(store, monkeypatch):
    monkeypatch.setattr(settings, "require_auth", True)
    monkeypatch.setenv("SKSURROGATE_REQUIRE_AUTH", "true")
    monkeypatch.setenv("SKSURROGATE_ADMIN_USERNAME", "root")
    monkeypatch.setenv("SKSURROGATE_ADMIN_PASSWORD", "root-password-1")
    _check_auth_settings()  # no raise: bootstrap credentials exist
    assert store.count_users() == 1


def test_open_mode_keeps_anonymous_access(store, monkeypatch):
    monkeypatch.setattr(settings, "require_auth", False)
    monkeypatch.setattr(settings, "api_key", None)
    open_client = TestClient(app)
    assert open_client.get("/api/tasks").status_code == 200
    assert open_client.get("/openapi.json").status_code == 200
    me = open_client.get("/api/auth/me").json()
    assert me["username"] == "anonymous"
    assert me["source"] == "anonymous"
    assert open_client.get("/api/auth/status").json()["auth_enforced"] is False


# --------------------------------------------------------------------------- #
# Audit trail
# --------------------------------------------------------------------------- #


def test_successful_login_is_audited_without_the_password(client, tmp_path):
    _login(client)
    events = audit.read()
    assert events and events[0]["event"] == "login"
    assert events[0]["outcome"] == "ok"
    assert events[0]["username"] == ADMIN[0]
    raw = (tmp_path / "logs" / "auth.jsonl").read_text(encoding="utf-8")
    assert ADMIN[1] not in raw
    assert auth.SESSION_COOKIE not in raw


def test_denied_login_is_audited_without_the_supplied_secret(client, tmp_path):
    supplied = "wrong-secret-never-log-1"
    client.post(
        "/api/auth/login", json={"username": ADMIN[0], "password": supplied}, headers=ORIGIN
    )
    events = audit.read()
    assert events[0]["event"] == "login" and events[0]["outcome"] == "denied"
    assert supplied not in (tmp_path / "logs" / "auth.jsonl").read_text(encoding="utf-8")


def test_throttled_login_is_audited(client, monkeypatch):
    monkeypatch.setattr(auth.THROTTLE, "max_failures", 1)
    bad = {"username": ADMIN[0], "password": "wrong-password"}
    client.post("/api/auth/login", json=bad, headers=ORIGIN)
    client.post("/api/auth/login", json=bad, headers=ORIGIN)
    outcomes = [event["outcome"] for event in audit.read()]
    assert "throttled" in outcomes


def test_key_creation_audits_the_prefix_only(client, tmp_path):
    _login(client)
    secret = client.post("/api/auth/keys", json={"name": "ci"}, headers=ORIGIN).json()["key"]
    event = audit.read()[0]
    assert event["event"] == "api_key.create"
    assert event["prefix"] == secret[: len(auth.API_KEY_PREFIX) + 6]
    assert secret not in (tmp_path / "logs" / "auth.jsonl").read_text(encoding="utf-8")


def test_account_changes_are_audited(client):
    _login(client)
    client.post(
        "/api/auth/users", json={"username": ANALYST[0], "password": ANALYST[1]}, headers=ORIGIN
    )
    assert audit.read()[0]["event"] == "user.create"
    assert audit.read()[0]["target_user"] == ANALYST[0]


def test_audit_endpoint_is_admin_only(client):
    _login(client)
    assert client.get("/api/auth/audit").status_code == 200
    client.post(
        "/api/auth/users", json={"username": ANALYST[0], "password": ANALYST[1]}, headers=ORIGIN
    )
    client.post("/api/auth/logout", headers=ORIGIN)
    _login(client, ANALYST)
    assert client.get("/api/auth/audit").status_code == 403


def test_audit_read_is_bounded_and_skips_malformed_lines(store, tmp_path):
    path = tmp_path / "logs" / "auth.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            ['{"event": "a", "outcome": "ok"}', "not json", '{"event": "b", "outcome": "ok"}']
        )
        + "\n",
        encoding="utf-8",
    )
    events = audit.read(limit=10)
    assert [event["event"] for event in events] == ["b", "a"]
    assert len(audit.read(limit=1)) == 1
    assert audit.read(limit=0) != []
