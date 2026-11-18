"""Tests for ownership and sharing (``api.acl`` and ``/api/access``).

The contract under test:

* the **task** is the ACL unit, with ordered roles ``viewer`` < ``editor`` <
  ``owner``;
* unowned (pre-auth) resources stay open so a single-user deployment keeps
  working after auth is switched on;
* a stranger sees 404, not 403 — the namespace itself is not leaked;
* only an owner may change the share list, and the last owner cannot be removed;
* every task-scoped router enforces access once, via its router dependency.
"""

import pytest
from fastapi.testclient import TestClient

from api import acl, auth
from api.config import settings
from api.main import app
from api.safety import UnsafeName  # noqa: F401  (imported for parity with the API surface)

ORIGIN = {"Origin": "http://testserver", "Host": "testserver"}


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "require_auth", False)
    settings.ensure()
    auth.reset_store()
    acl.reset_store()
    auth.THROTTLE._failures.clear()
    yield acl.store()
    auth.reset_store()
    acl.reset_store()


@pytest.fixture()
def enforced(store, monkeypatch):
    """Auth on, with an administrator plus two ordinary accounts."""
    monkeypatch.setattr(settings, "require_auth", True)
    auth.store().create_user("root", "root-password-1", is_admin=True)
    auth.store().create_user("alice", "alice-password-1")
    auth.store().create_user("bob", "bob-password-1")
    return store


def _client_for(username, password):
    client = TestClient(app)
    response = client.post(
        "/api/auth/login", json={"username": username, "password": password}, headers=ORIGIN
    )
    assert response.status_code == 200, response.text
    return client


@pytest.fixture()
def alice(enforced):
    return _client_for("alice", "alice-password-1")


@pytest.fixture()
def bob(enforced):
    return _client_for("bob", "bob-password-1")


@pytest.fixture()
def root_client(enforced):
    return _client_for("root", "root-password-1")


def put(client, url):
    """Cookie-authenticated ``PUT`` with the same-origin header the CSRF check wants."""
    return client.put(url, json={"method": "kfold", "params": {}}, headers=ORIGIN)


def _make_task(name):
    """Create the on-disk footprints ``/api/tasks`` discovers."""
    (settings.datasets_dir / name).mkdir(parents=True, exist_ok=True)
    (settings.bundles_dir / name).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


def test_roles_roundtrip_and_removal(store):
    store.set_role(acl.TASK, "iris", 7, "editor", granted_by=1)
    assert store.role_of(acl.TASK, "iris", 7) == "editor"
    assert store.role_of(acl.TASK, "iris", 8) is None
    store.set_role(acl.TASK, "iris", 7, "owner")
    assert store.role_of(acl.TASK, "iris", 7) == "owner"
    assert store.has_owner(acl.TASK, "iris") is True
    assert store.remove_role(acl.TASK, "iris", 7) is True
    assert store.has_owner(acl.TASK, "iris") is False


def test_unknown_kind_and_role_are_rejected(store):
    with pytest.raises(acl.AclError):
        store.set_role("dataset", "iris", 1, "viewer")
    with pytest.raises(acl.AclError):
        store.set_role(acl.TASK, "iris", 1, "admin")


def test_claim_never_overwrites_an_existing_owner(store):
    owner = auth.Principal(1, "alice", False, "session")
    other = auth.Principal(2, "bob", False, "session")
    store.claim(acl.TASK, "iris", owner)
    store.claim(acl.TASK, "iris", other)
    assert store.role_of(acl.TASK, "iris", 1) == "owner"
    assert store.role_of(acl.TASK, "iris", 2) is None


def test_adopt_orphans_skips_owned_resources(store):
    admin = auth.Principal(1, "root", True, "session")
    store.claim(acl.TASK, "owned", admin)
    adopted = store.adopt_orphans(acl.TASK, admin, ["owned", "legacy-a", "legacy-b"])
    assert adopted == ["legacy-a", "legacy-b"]
    assert store.has_owner(acl.TASK, "legacy-a")


def test_effective_role_matrix(store, monkeypatch):
    monkeypatch.setattr(settings, "require_auth", True)
    alice = auth.Principal(1, "alice", False, "session")
    admin = auth.Principal(9, "root", True, "session")

    # Unowned legacy resource: everybody authenticated may use it.
    assert acl.effective_role(acl.TASK, "legacy", alice) == "owner"
    # Once owned, a stranger has no access and an admin is still owner.
    store.claim(acl.TASK, "legacy", admin)
    assert acl.effective_role(acl.TASK, "legacy", alice) is None
    assert acl.effective_role(acl.TASK, "legacy", admin) == "owner"
    # A grant restores the exact role.
    store.set_role(acl.TASK, "legacy", 1, "viewer")
    assert acl.effective_role(acl.TASK, "legacy", alice) == "viewer"
    # Auth off: the ACL is bypassed entirely.
    monkeypatch.setattr(settings, "require_auth", False)
    assert acl.effective_role(acl.TASK, "legacy", auth.ANONYMOUS) == "owner"


def test_require_role_answers_404_for_strangers_and_403_for_weak_roles(store, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(settings, "require_auth", True)

    owner = auth.Principal(1, "alice", False, "session")
    viewer = auth.Principal(2, "bob", False, "session")
    stranger = auth.Principal(3, "carol", False, "session")
    store.claim(acl.TASK, "iris", owner)
    store.set_role(acl.TASK, "iris", 2, "viewer")

    with pytest.raises(HTTPException) as denied:
        acl.require_role(acl.TASK, "iris", "editor", viewer)
    assert denied.value.status_code == 403
    with pytest.raises(HTTPException) as hidden:
        acl.require_role(acl.TASK, "iris", "viewer", stranger)
    assert hidden.value.status_code == 404


def test_visible_ids_filters_owned_resources(store, monkeypatch):
    monkeypatch.setattr(settings, "require_auth", True)
    owner = auth.Principal(1, "alice", False, "session")
    bob = auth.Principal(2, "bob", False, "session")
    store.claim(acl.TASK, "mine", owner)
    store.claim(acl.TASK, "theirs", bob)
    assert acl.visible_ids(acl.TASK, ["mine", "theirs", "legacy"], owner) == ["mine", "legacy"]
    store.set_role(acl.TASK, "theirs", 1, "viewer")
    assert acl.visible_ids(acl.TASK, ["mine", "theirs"], owner) == ["mine", "theirs"]


# --------------------------------------------------------------------------- #
# HTTP: isolation and sharing
# --------------------------------------------------------------------------- #


def test_new_task_is_claimed_by_its_first_writer(alice, bob, root_client):
    _make_task("iris")
    # A write attempt on an unowned task claims it, then the endpoint answers 404
    # (no dataset registered yet) — the claim must not depend on success.
    assert put(alice, "/api/datasets/iris/cv").status_code in (200, 400, 404, 422)
    alice_id = auth.store().get_user_by_username("alice")["user_id"]
    assert acl.store().has_owner(acl.TASK, "iris")
    assert acl.store().role_of(acl.TASK, "iris", alice_id) == "owner"

    assert "iris" in alice.get("/api/tasks").json()["tasks"]
    assert "iris" not in bob.get("/api/tasks").json()["tasks"]


def test_stranger_gets_404_not_403(alice, bob):
    _make_task("iris")
    put(alice, "/api/datasets/iris/cv")
    hidden = bob.get("/api/datasets/iris")
    assert hidden.status_code == 404
    assert hidden.json()["detail"] == "No such task: 'iris'"


def test_share_grants_view_then_edit(alice, bob):
    _make_task("iris")
    put(alice, "/api/datasets/iris/cv")

    shared = alice.post(
        "/api/access/tasks/iris/shares", json={"username": "bob", "role": "viewer"}, headers=ORIGIN
    )
    assert shared.status_code == 200, shared.text
    assert "iris" in bob.get("/api/tasks").json()["tasks"]
    assert bob.get("/api/access/tasks/iris").json()["role"] == "viewer"

    # A viewer may read but not mutate; the write gate answers 403 (not 404).
    assert bob.get("/api/datasets/iris").status_code == 404  # no dataset registered yet
    assert put(bob, "/api/datasets/iris/cv").status_code == 403
    assert bob.post("/api/access/tasks/iris/shares", json={"username": "bob", "role": "owner"}, headers=ORIGIN).status_code == 403

    upgraded = alice.post(
        "/api/access/tasks/iris/shares", json={"username": "bob", "role": "editor"}, headers=ORIGIN
    )
    assert upgraded.status_code == 200
    assert put(bob, "/api/datasets/iris/cv").status_code != 403


def test_only_an_owner_may_share(alice, bob, root_client):
    _make_task("iris")
    put(alice, "/api/datasets/iris/cv")
    alice.post("/api/access/tasks/iris/shares", json={"username": "bob", "role": "editor"}, headers=ORIGIN)
    assert bob.post(
        "/api/access/tasks/iris/shares", json={"username": "root", "role": "viewer"}, headers=ORIGIN
    ).status_code == 403
    # Administrators hold owner-equivalent access everywhere.
    assert root_client.post(
        "/api/access/tasks/iris/shares", json={"username": "root", "role": "owner"}, headers=ORIGIN
    ).status_code == 200


def test_last_owner_cannot_be_removed(alice):
    _make_task("iris")
    put(alice, "/api/datasets/iris/cv")
    shares = alice.get("/api/access/tasks/iris").json()["shares"]
    owner_id = next(share["user_id"] for share in shares if share["role"] == "owner")
    removed = alice.delete(f"/api/access/tasks/iris/shares/{owner_id}", headers=ORIGIN)
    assert removed.status_code == 400
    assert "last owner" in removed.json()["detail"]


def test_sharing_unknown_user_is_404(alice):
    _make_task("iris")
    put(alice, "/api/datasets/iris/cv")
    assert alice.post(
        "/api/access/tasks/iris/shares", json={"username": "ghost", "role": "viewer"}, headers=ORIGIN
    ).status_code == 404


def test_adopt_orphans_endpoint(alice, root_client, bob):
    _make_task("legacy-task")
    # Unowned material is open until adoption, so both users still see it.
    assert "legacy-task" in bob.get("/api/tasks").json()["tasks"]

    alice_id = auth.store().get_user_by_username("alice")["user_id"]
    adopted = root_client.post(
        "/api/access/adopt", json={"kind": "task", "user_id": alice_id}, headers=ORIGIN
    )
    assert adopted.status_code == 200, adopted.text
    assert "legacy-task" in adopted.json()["adopted"]
    # After adoption only the owner (alice, id 2 in this fixture ordering) and
    # admins see it.
    assert "legacy-task" in alice.get("/api/tasks").json()["tasks"]
    assert "legacy-task" not in bob.get("/api/tasks").json()["tasks"]

    # Idempotent.
    again = root_client.post("/api/access/adopt", json={"kind": "task"}, headers=ORIGIN)
    assert again.json()["count"] == 0


def test_projects_are_isolated_and_shareable(alice, bob):
    created = alice.post(
        "/api/projects", json={"name": "Alice project", "description": "", "research_questions": []}, headers=ORIGIN
    )
    assert created.status_code in (200, 201), created.text
    project_id = created.json()["project_id"]

    assert project_id not in [p["project_id"] for p in bob.get("/api/projects").json()["projects"]]
    assert bob.get(f"/api/projects/{project_id}").status_code == 404

    shared = alice.post(
        f"/api/access/projects/{project_id}/shares",
        json={"username": "bob", "role": "viewer"},
        headers=ORIGIN,
    )
    assert shared.status_code == 200, shared.text
    assert project_id in [p["project_id"] for p in bob.get("/api/projects").json()["projects"]]
    assert bob.get(f"/api/projects/{project_id}").status_code == 200


def test_open_mode_has_no_acl(tmp_path, monkeypatch):
    """Auth off: one anonymous user, so nothing is filtered or refused."""
    monkeypatch.setattr(settings, "root", tmp_path)
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "require_auth", False)
    settings.ensure()
    auth.reset_store()
    acl.reset_store()
    client = TestClient(app)

    _make_task("solo")
    assert client.get("/api/tasks").json()["tasks"] == ["solo"]
    assert acl.effective_role(acl.TASK, "solo", auth.ANONYMOUS) == "owner"
