"""Experiment-design endpoints: the suggest-next contract the UI depends on.

The page parses the pasted observations itself and then posts them; this file
pins what the API guarantees for good and bad payloads, so a UI regression shows
up as a failing contract instead of a 422 in the browser:

* two observations are the minimum (a surrogate cannot extrapolate from one),
* sample/value counts and row arity must agree,
* bounds must be finite with low < high,
* variable names must be unique.
"""

import importlib.util

import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    return TestClient(app)


VARIABLES = [{"name": "x1", "low": 0.0, "high": 1.0}, {"name": "x2", "low": 0.0, "high": 1.0}]


def _suggest(client, **overrides):
    body = {
        "variables": VARIABLES,
        "samples": [[0.2, 0.7], [0.8, 0.1], [0.5, 0.5]],
        "values": [1.35, 0.42, 0.9],
        "n_candidates": 200,
    }
    body.update(overrides)
    return client.post("/api/design/suggest-next", json=body)


def test_latin_hypercube_is_seeded_and_in_bounds(client):
    body = {"variables": VARIABLES, "n_samples": 8, "random_state": 3}
    first = client.post("/api/design/latin-hypercube", json=body)
    assert first.status_code == 200, first.text
    payload = first.json()
    assert payload["columns"] == ["x1", "x2"]
    assert len(payload["points"]) == 8
    for x1, x2 in payload["points"]:
        assert 0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0
    assert first.json() == client.post("/api/design/latin-hypercube", json=body).json()


def test_suggest_next_returns_a_bounded_candidate(client):
    response = _suggest(client)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["columns"] == ["x1", "x2"]
    result = payload["result"]
    # A suggestion is a point inside the box plus the surrogate's prediction.
    assert len(result["candidate"]) == 2
    assert all(0.0 <= value <= 1.0 for value in result["candidate"])
    assert result["acquisition"] == "lower_confidence_bound"
    assert result["observations_used"] == 3
    assert result["candidate_pool_size"] == 200


def test_single_observation_is_rejected_by_the_schema(client):
    """One run is not enough; the page turns this into its own message."""
    response = _suggest(client, samples=[[0.2, 0.7]], values=[1.35])
    assert response.status_code == 422
    locations = {tuple(item["loc"]) for item in response.json()["detail"]}
    assert ("body", "samples") in locations
    assert ("body", "values") in locations


def test_sample_and_value_counts_must_agree(client):
    response = _suggest(client, samples=[[0.2, 0.7], [0.8, 0.1], [0.5, 0.5]], values=[1.35, 0.42])
    assert response.status_code == 400
    assert "one scalar per sample" in response.json()["detail"]


def test_sample_rows_must_match_the_variable_count(client):
    response = _suggest(client, samples=[[0.2], [0.8]], values=[1.0, 2.0])
    assert response.status_code == 400
    assert "shape" in response.json()["detail"]


@pytest.mark.parametrize("bounds", [(1.0, 1.0), (2.0, 1.0)])
def test_bounds_must_be_ordered(client, bounds):
    variables = [{"name": "x1", "low": bounds[0], "high": bounds[1]}]
    response = _suggest(client, variables=variables, samples=[[0.2], [0.8]], values=[1.0, 2.0])
    assert response.status_code == 400
    assert "lower < upper" in response.json()["detail"]


def test_duplicate_variable_names_are_rejected(client):
    variables = [{"name": "x", "low": 0.0, "high": 1.0}, {"name": "x", "low": 0.0, "high": 1.0}]
    response = _suggest(client, variables=variables)
    assert response.status_code == 400
    assert "unique" in response.json()["detail"]
