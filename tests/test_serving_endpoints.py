"""Serving-pin endpoints: declare which model a task serves, then score through it.

The contract under test (documented in ``docs/inference.rst``, "Serving a pinned
model"):

* ``POST /api/serving/{task}`` pins a model, ``DELETE`` unpins it; an unpinned
  task answers ``active: false`` on the status read and 409 on the serving
  endpoint — never a silently different model;
* ``resolve_mode="version"`` freezes the exact version (a later promotion leaves
  it alone) while ``resolve_mode="alias"`` follows the registry pointer
  (a later promotion changes what the *same* request returns). The differing
  predictions after one ``promote`` is the assertion that separates the modes;
* the served path predicts and records monitoring through the same core as
  ``/api/inference``, so its monitor log is keyed to the served version;
* ``GET /api/serving`` is a list surface: the router-level ``task_access``
  dependency cannot guard it (no task in the path), so it filters through the
  ACL like the other list routes;
* caller-supplied segments (task name, model identifier) stay inside the storage
  root, and an unresolvable pin is refused at declaration time (404).
"""

from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from api import acl, auth
from api.config import settings
from api.deps import load_monitor_records
from api.main import app
from api.safety import UnsafeName
from SKSurrogate import ModelBundle, ModelRegistry, save_bundle

ORIGIN = {"Origin": "http://testserver", "Host": "testserver"}
TASK = "serving-demo"
FEATURES = {"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0, 4.0]}
ROWS = {"rows": [{"a": 1.0, "b": 2.0}]}


def _bundle(shift):
    """A trivial linear model whose predictions are ``a + shift``."""
    frame = pd.DataFrame(FEATURES)
    target = [value + shift for value in frame["a"]]
    return ModelBundle(
        LinearRegression().fit(frame, target),
        task_name=TASK,
        schema={"a": {"dtype": "float64"}, "b": {"dtype": "float64"}},
        metrics={"r2": 1.0},
    )


def _open_root(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "require_auth", False)
    settings.ensure()
    auth.reset_store()
    acl.reset_store()
    auth.THROTTLE._failures.clear()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Open API on a throwaway root, with two registered versions (shift 0 / 100)."""
    _open_root(tmp_path, monkeypatch)
    registry = ModelRegistry(settings.registry_dir)
    first = registry.register(_bundle(0.0))
    second = registry.register(_bundle(100.0))
    registry.promote(TASK, first, "production")
    yield SimpleNamespace(
        registry=registry, first=first, second=second, client=TestClient(app)
    )
    auth.reset_store()
    acl.reset_store()


def _pin(client, **body):
    return client.post(f"/api/serving/{TASK}", json=body, headers=ORIGIN)


def _predict(client, **body):
    payload = {**ROWS, **body}
    return client.post(f"/api/serving/{TASK}/predict", json=payload, headers=ORIGIN)


def test_status_is_inactive_before_any_pin(env):
    response = env.client.get(f"/api/serving/{TASK}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["active"] is False
    assert body["model_version"] is None
    assert body["started_at"] is None
    assert body["history"] == []


def test_predict_before_pinning_is_409(env):
    response = _predict(env.client)
    assert response.status_code == 409, response.text
    assert "not serving" in response.json()["detail"]


def test_pin_by_version_and_predict_round_trip(env):
    pinned = _pin(env.client, model_version=env.first, note="rollout")
    assert pinned.status_code == 200, pinned.text
    state = pinned.json()
    assert state["active"] is True
    assert state["resolve_mode"] == "version"
    assert state["requested"] == env.first
    assert state["model_version"] == env.first
    assert state["note"] == "rollout"
    # Anonymous single-user API: the actor is recorded verbatim.
    assert state["started_by"] == "anonymous"

    response = _predict(env.client, request_id="served-1")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_version"] == env.first
    assert body["resolve_mode"] == "version"
    assert body["requested"] == env.first
    assert body["request_id"] == "served-1"
    assert body["predictions"] == pytest.approx([1.0])
    assert body["metrics"]["rows"] == 1


def test_pinned_alias_follows_promotion(env):
    """The distinguishing assertion: one ``promote`` changes the served model."""
    assert _pin(env.client, alias="production").status_code == 200
    first = _predict(env.client).json()
    assert first["model_version"] == env.first
    assert first["predictions"] == pytest.approx([1.0])

    env.registry.promote(TASK, env.second, "production")

    second = _predict(env.client).json()
    assert second["model_version"] == env.second
    assert second["predictions"] == pytest.approx([101.0])
    # The pin itself is unchanged — it still names the alias, not a version.
    status = env.client.get(f"/api/serving/{TASK}").json()
    assert status["resolve_mode"] == "alias"
    assert status["requested"] == "production"
    assert status["model_version"] == env.second


def test_pinned_version_stays_frozen_across_promotion(env):
    assert _pin(env.client, model_version=env.first).status_code == 200
    env.registry.promote(TASK, env.second, "production")
    body = _predict(env.client).json()
    assert body["model_version"] == env.first
    assert body["predictions"] == pytest.approx([1.0])


def test_unpin_stops_serving_and_a_second_unpin_is_409(env):
    assert _pin(env.client, model_version=env.first).status_code == 200
    stopped = env.client.delete(f"/api/serving/{TASK}", headers=ORIGIN)
    assert stopped.status_code == 200, stopped.text
    assert stopped.json()["active"] is False
    assert _predict(env.client).status_code == 409
    assert env.client.delete(f"/api/serving/{TASK}", headers=ORIGIN).status_code == 409


def test_start_requires_exactly_one_identifier(env):
    assert _pin(env.client).status_code == 400
    both = _pin(env.client, model_version=env.first, alias="production")
    assert both.status_code == 400
    assert "Exactly one" in both.json()["detail"]


def test_start_rejects_an_unresolvable_model(env):
    missing = _pin(env.client, model_version="no-such-version")
    assert missing.status_code == 404, missing.text
    unknown_alias = _pin(env.client, alias="no-such-alias")
    assert unknown_alias.status_code == 404, unknown_alias.text
    # Nothing was pinned by the failed attempts.
    assert env.client.get(f"/api/serving/{TASK}").json()["active"] is False


def test_unsafe_segments_are_rejected_and_nothing_outside_the_root_is_loaded(env):
    """A traversing identifier must 400 at the segment guard, never load a bundle.

    ``contained_path(bundles/<task>, '../../evil', '.bundle')`` resolves to
    ``<root>/evil.bundle``, so a real bundle is parked there: if the guard were
    missing, the pin would load it (``joblib`` load is RCE-shaped) and succeed.
    """
    save_bundle(_bundle(0.0), settings.root / "evil.bundle")

    traversal_model = _pin(env.client, model_version="../../evil")
    assert traversal_model.status_code == 400, traversal_model.text
    # ``%2E%2E`` decodes to a ``..`` path segment, which reaches the endpoint
    # (``..%2Fevil`` is normalized away by the client before routing).
    traversal_task = env.client.post(
        "/api/serving/%2E%2E", json={"model_version": env.first}, headers=ORIGIN
    )
    assert traversal_task.status_code == 400, traversal_task.text
    with pytest.raises(UnsafeName):
        settings.task_serving_path("..")
    assert env.client.get(f"/api/serving/{TASK}").json()["active"] is False


def test_history_records_the_pin_sequence(env):
    assert _pin(env.client, model_version=env.first).status_code == 200
    assert _pin(env.client, alias="production").status_code == 200
    assert env.client.delete(f"/api/serving/{TASK}", headers=ORIGIN).status_code == 200
    history = env.client.get(f"/api/serving/{TASK}").json()["history"]
    assert [entry["event"] for entry in history] == ["start", "replace", "stop"]
    assert history[0]["model_version"] == env.first
    assert history[1]["requested"] == "production"
    assert all(entry["actor"] == "anonymous" for entry in history)


def test_served_predictions_are_monitored_under_the_served_version(env):
    assert _pin(env.client, model_version=env.first).status_code == 200
    assert _predict(env.client).status_code == 200
    records = load_monitor_records(TASK, env.first)
    assert len(records) == 1
    assert records[0]["rows"] == 1
    assert records[0]["success"] is True


def test_serving_list_reports_only_active_pins(env):
    assert env.client.get("/api/serving").json() == {"serving": [], "count": 0}
    assert _pin(env.client, model_version=env.first).status_code == 200
    listing = env.client.get("/api/serving").json()
    assert listing["count"] == 1
    assert listing["serving"][0]["task_name"] == TASK
    assert listing["serving"][0]["model_version"] == env.first

    env.client.delete(f"/api/serving/{TASK}", headers=ORIGIN)
    assert env.client.get("/api/serving").json()["count"] == 0
    # An inactive pin still has a status document (the UI shows "not serving").
    assert env.client.get(f"/api/serving/{TASK}").json()["active"] is False


def test_partition_input_is_projected_onto_the_bundle_schema(env):
    """Stored partitions carry the target column; it must be dropped, not sent."""
    dataset_dir = settings.task_dataset_dir(TASK)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({**FEATURES, "target": [0.0, 1.0, 2.0, 3.0]})
    frame.to_csv(dataset_dir / "holdout.csv", index=False)
    assert _pin(env.client, model_version=env.first).status_code == 200
    response = _predict(env.client, rows=None, partition="holdout")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ignored_columns"] == ["target"]
    assert body["metrics"]["rows"] == 4


def test_serving_list_filters_through_the_acl(tmp_path, monkeypatch):
    """Owned pins are hidden from a stranger; the task namespace is not leaked."""
    _open_root(tmp_path, monkeypatch)
    registry = ModelRegistry(settings.registry_dir)
    version = registry.register(_bundle(0.0))
    monkeypatch.setattr(settings, "require_auth", True)
    auth.store().create_user("alice", "alice-password-1")
    auth.store().create_user("bob", "bob-password-1")

    def client_for(username, password):
        client = TestClient(app)
        login = client.post(
            "/api/auth/login",
            json={"username": username, "password": password},
            headers=ORIGIN,
        )
        assert login.status_code == 200, login.text
        return client

    alice, bob = client_for("alice", "alice-password-1"), client_for(
        "bob", "bob-password-1"
    )
    # A successful write claims ownership of the task for the authenticated user.
    assert _pin(alice, model_version=version).status_code == 200
    assert alice.get(f"/api/serving/{TASK}").json()["active"] is True
    assert bob.get("/api/serving").json() == {"serving": [], "count": 0}
    assert bob.get(f"/api/serving/{TASK}").status_code == 404
    assert bob.post(f"/api/serving/{TASK}/predict", json=ROWS, headers=ORIGIN).status_code == 404
    auth.reset_store()
    acl.reset_store()
