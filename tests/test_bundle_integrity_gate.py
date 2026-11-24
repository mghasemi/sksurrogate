"""Serving-path bundle integrity: checksum gate and strict-mode refusal.

`resolve_bundle(verify=True)` is used by every prediction route. A checksum
mismatch is always refused (409); a missing manifest sidecar is refused only in
strict mode (``SKSURROGATE_STRICT_BUNDLES=1``), so legacy bundles keep serving
in a non-strict deployment.
"""

import importlib.util

import numpy as np
import pandas as pd
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


def _trained_bundle(client, task="integrity-demo", rows=60, seed=4):
    rng = np.random.RandomState(seed)
    a = rng.normal(size=rows)
    frame = pd.DataFrame({"a": a, "b": rng.normal(size=rows), "y": 2.0 * a})
    registered = client.post(
        "/api/datasets/%s/register" % task,
        data={"target": "y", "partition": "train"},
        files={"file": ("t.csv", frame.to_csv(index=False), "text/csv")},
    )
    assert registered.status_code == 200, registered.text
    trained = client.post(
        "/api/bundles/%s/train-baseline" % task,
        json={"estimator": "linear_regression", "validation_partition": None},
    )
    assert trained.status_code == 200, trained.text
    return task, trained.json()["model_version"], frame


def _predict(client, task, version):
    return client.post(
        "/api/inference/%s/predict" % task,
        json={"model_version": version, "rows": [{"a": 1.0, "b": 0.5}]},
    )


def test_strict_mode_refuses_a_bundle_without_a_sidecar(client, monkeypatch):
    task, version, _ = _trained_bundle(client)
    monkeypatch.delenv("SKSURROGATE_STRICT_BUNDLES", raising=False)
    assert _predict(client, task, version).status_code == 200

    sidecar = (
        settings.task_bundles_dir(task) / ("%s.bundle.manifest.json" % version)
    )
    assert sidecar.exists()
    sidecar.unlink()

    monkeypatch.setenv("SKSURROGATE_STRICT_BUNDLES", "1")
    refused = _predict(client, task, version)
    assert refused.status_code == 409, refused.text
    assert "strict mode" in refused.json()["detail"]
    assert "SKSURROGATE_STRICT_BUNDLES=0" in refused.json()["detail"]

    # The same artifact still serves in a non-strict deployment: the gate is the
    # policy, not corruption.
    monkeypatch.delenv("SKSURROGATE_STRICT_BUNDLES", raising=False)
    assert _predict(client, task, version).status_code == 200


def test_a_corrupted_bundle_is_refused_even_without_strict_mode(client, monkeypatch):
    task, version, _ = _trained_bundle(client, task="integrity-corrupt")
    monkeypatch.delenv("SKSURROGATE_STRICT_BUNDLES", raising=False)
    bundle_path = settings.task_bundles_dir(task) / ("%s.bundle" % version)
    raw = bytearray(bundle_path.read_bytes())
    raw[-1] ^= 0x01
    bundle_path.write_bytes(bytes(raw))

    refused = _predict(client, task, version)
    assert refused.status_code == 409, refused.text
    assert "checksum mismatch" in refused.json()["detail"]
