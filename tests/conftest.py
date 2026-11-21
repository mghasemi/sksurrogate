"""Shared pytest configuration.

Only one fixture lives here, and it exists because of an upstream quirk: openml
0.15 stores a *masked* value when the API key comes from ``OPENML_API_KEY`` (the
key object prints as ``2dc6…b902b`` and is 10 characters long), so the client
sends an unmasked-looking token the server rejects with
``code 103: Authentication failed``. Assigning the environment value directly to
``openml.config.apikey`` restores the real key.

Tests that touch OpenML (``tests/test_cspace_cli.py``) therefore need the key in
the environment; elsewhere this fixture is a no-op and openml is never imported
unless it is already installed.
"""

import importlib.util
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _openml_api_key():
    """Unmask ``OPENML_API_KEY`` for the OpenML client, when one is provided."""
    key = os.environ.get("OPENML_API_KEY")
    if not key or importlib.util.find_spec("openml") is None:
        yield
        return
    import openml

    # Only override the masked env-derived value; an explicitly configured key
    # (e.g. via ``~/.openml/config``) is left alone.
    if "$" not in key and len(getattr(openml.config, "apikey", "") or "") < len(key):
        openml.config.apikey = key
    yield
