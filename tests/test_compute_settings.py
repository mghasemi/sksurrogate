"""Tests for the compute budget (api/compute.py) and its API surface.

The budget is what stops the job pool and joblib's fold pool from multiplying
into ``4 x cpu_count`` workers. These tests pin the arithmetic, the validation
messages, persistence, the executor resize and the wiring into a submitted job
and into the AML search.
"""

import importlib.util
import json
import os
import time

import pytest

if importlib.util.find_spec("httpx") is None:
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api import compute, training  # noqa: E402
from api.config import settings  # noqa: E402
from api.jobs import job_manager  # noqa: E402
from api.main import app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    compute.reset_executor()
    return TestClient(app)


@pytest.fixture()
def small_host(monkeypatch):
    """Pretend the process may only use 8 cores, so budgets are easy to reason about."""
    monkeypatch.setattr(compute, "cpu_count", lambda: 8)


def _register(client, task, rows=60):
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(0)
    frame = pd.DataFrame(rng.normal(size=(rows, 3)), columns=["a", "b", "c"])
    frame["y"] = (frame["a"] + 0.3 * rng.normal(size=rows) > 0).astype(int)
    response = client.post(
        "/api/datasets/%s/register" % task,
        data={"target": "y", "partition": "train"},
        files={"file": ("t.csv", frame.to_csv(index=False), "text/csv")},
    )
    assert response.status_code == 200, response.text


def _wait(job_id, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = job_manager.get(job_id)
        if record and record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.05)
    raise AssertionError("job did not finish")


# --------------------------------------------------------------------------- #
# Budget arithmetic and validation
# --------------------------------------------------------------------------- #


def test_defaults_reserve_one_blas_thread_per_job_slot():
    values = compute.defaults(cores=32)
    assert values["job_workers"] == 4
    assert values["blas_threads"] == 1
    assert values["job_threads"] == 28
    used = values["job_workers"] * values["blas_threads"] + values["job_threads"]
    assert used == 32


def test_defaults_shrink_on_small_hosts():
    assert compute.defaults(cores=1) == {
        "job_workers": 1,
        "job_threads": 1,
        "blas_threads": 1,
        "dask_workers": 1,
        "dask_threads": 1,
    }
    assert compute.defaults(cores=2)["job_workers"] == 2
    assert compute.defaults(cores=2)["job_threads"] == 1


def test_validate_rejects_a_budget_above_the_core_count():
    values = compute.defaults(cores=8) | {"job_threads": 8}
    with pytest.raises(ValueError) as excinfo:
        compute.validate(values, cores=8)
    message = str(excinfo.value)
    assert "exceeds the 8 available cores" in message
    assert "reduce job_threads to at most 4" in message


def test_validate_rejects_single_values_above_the_core_count():
    values = compute.defaults(cores=8) | {"blas_threads": 9, "job_workers": 1}
    with pytest.raises(ValueError, match="blas_threads = 9 exceeds the 8 cores"):
        compute.validate(values, cores=8)


def test_validate_rejects_non_integer_and_boolean_values():
    values = compute.defaults(cores=8)
    with pytest.raises(ValueError, match="job_workers must be an integer"):
        compute.validate(values | {"job_workers": "4"}, cores=8)
    with pytest.raises(ValueError, match="job_workers must be an integer"):
        compute.validate(values | {"job_workers": True}, cores=8)
    with pytest.raises(ValueError, match="job_threads must be at least 1"):
        compute.validate(values | {"job_threads": 0}, cores=8)


def test_validate_rejects_dask_oversubscription():
    values = compute.defaults(cores=8) | {"dask_workers": 4, "dask_threads": 4}
    with pytest.raises(ValueError, match="dask_workers x dask_threads = 4 x 4 = 16"):
        compute.validate(values, cores=8)


def test_save_and_load_round_trip(client):
    values = compute.defaults(cores=compute.cpu_count()) | {"job_workers": 1, "job_threads": 1}
    compute.save(values)
    assert compute.current() == values
    stored = json.loads(settings.compute_settings_path.read_text())
    assert stored == values
    # The executor was rebuilt for the new width.
    assert compute.job_executor()._max_workers == 1


def test_load_falls_back_when_the_file_no_longer_fits(client, small_host, monkeypatch):
    settings.compute_settings_path.write_text(
        json.dumps({"job_workers": 4, "job_threads": 28, "blas_threads": 1,
                    "dask_workers": 4, "dask_threads": 1})
    )
    # 32 cores' worth of budget on an 8-core host must not be trusted.
    assert compute.current() == compute.defaults(cores=8)


def test_load_ignores_corrupt_json(client, small_host):
    settings.compute_settings_path.write_text("{not json")
    assert compute.current() == compute.defaults(cores=8)


# --------------------------------------------------------------------------- #
# Executor and thread limits
# --------------------------------------------------------------------------- #


def test_job_executor_is_rebuilt_when_the_width_changes(client, small_host):
    first = compute.job_executor()
    assert first._max_workers == 4
    compute.save(compute.defaults(cores=8) | {"job_workers": 1, "job_threads": 1})
    second = compute.job_executor()
    assert second is not first
    assert second._max_workers == 1


def test_apply_limits_pins_thread_environment(client, small_host):
    values = compute.defaults(cores=8) | {"blas_threads": 2, "job_workers": 1, "job_threads": 1}
    compute.save(values)
    applied = compute.apply_limits()
    assert applied["OMP_NUM_THREADS"] == "2"
    import os

    for name in compute.THREAD_ENV_VARS:
        assert os.environ[name] == "2"


def test_snapshot_reports_the_budget(client, small_host):
    payload = compute.snapshot()
    assert payload["cores"] == 8
    assert payload["budget"]["used"] == 8
    assert payload["budget"]["available"] == 8
    assert "4 jobs x 1 BLAS thread(s) + 4 fold workers = 8 of 8 cores" in payload["budget"]["summary"]


def test_dask_cluster_is_constructed_with_the_budget(client, small_host, monkeypatch):
    pytest.importorskip("distributed")
    recorded = {}

    def fake_cluster(**kwargs):
        recorded.update(kwargs)
        return "cluster"

    # Patch the binding api.compute actually imports. Patching the top-level
    # ``distributed`` module leaks here: dask.distributed caches resolved
    # attributes in its own module dict, where monkeypatch cannot undo them.
    monkeypatch.setattr("dask.distributed.LocalCluster", fake_cluster)
    assert compute.dask_cluster() == "cluster"
    values = compute.current()
    assert recorded["n_workers"] == values["dask_workers"]
    assert recorded["threads_per_worker"] == values["dask_threads"]


# --------------------------------------------------------------------------- #
# API surface
# --------------------------------------------------------------------------- #


def test_get_compute_settings_reports_cores_and_defaults(client):
    payload = client.get("/api/settings/compute").json()
    assert payload["cores"] == compute.cpu_count()
    assert payload["values"] == payload["defaults"]
    assert set(payload["values"]) == set(compute.KEYS)


def test_put_compute_settings_rejects_an_over_budget_document(client):
    payload = client.get("/api/settings/compute").json()
    bad = dict(payload["values"], job_threads=payload["cores"])
    response = client.put("/api/settings/compute", json=bad)
    assert response.status_code == 422, response.text
    assert "exceeds the" in response.json()["detail"]


def test_put_compute_settings_persists_a_valid_budget(client):
    payload = client.get("/api/settings/compute").json()
    good = dict(
        payload["values"],
        job_workers=1,
        job_threads=1,
        blas_threads=1,
        dask_workers=1,
        dask_threads=1,
    )
    response = client.put("/api/settings/compute", json=good)
    assert response.status_code == 200, response.text
    assert response.json()["values"] == good
    assert client.get("/api/settings/compute").json()["values"] == good
    assert json.loads(settings.compute_settings_path.read_text()) == good


def test_put_compute_settings_requires_every_field(client):
    response = client.put("/api/settings/compute", json={"job_workers": 1})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Enforcement
# --------------------------------------------------------------------------- #


def test_submitted_job_records_and_applies_the_budget(client, small_host):
    compute.save(compute.defaults(cores=8) | {"blas_threads": 3, "job_workers": 1, "job_threads": 1})
    job_id = job_manager.submit("compute-task", "test", lambda _jid: (lambda: {"ok": True}))
    record = _wait(job_id)
    assert record["status"] == "completed", record["error"]
    assert record["compute"] == compute.limits()
    assert record["compute"]["blas_threads"] == 3
    assert record["compute_applied"]["OMP_NUM_THREADS"] == "3"
    # A count of 4 jobs in a 1-slot budget would be a bug in the executor wiring.
    assert compute.job_executor()._max_workers == 1


def test_fold_workers_come_from_the_compute_budget(client, small_host, tmp_path, monkeypatch):
    _register(client, "compute-fold")
    compute.save(
        compute.defaults(cores=8) | {"job_workers": 2, "blas_threads": 1, "job_threads": 3}
    )
    recorded = {}

    class StubAML:
        def __init__(self, **kwargs):
            recorded.update(kwargs)
            raise RuntimeError("stop after construction")

    monkeypatch.setattr(training, "AML", StubAML)
    with pytest.raises(RuntimeError, match="stop after construction"):
        training.fit_pipeline_bundle(
            "compute-fold",
            seq=["sklearn.tree.DecisionTreeClassifier"],
            checkpoint_dir=tmp_path / "ckpt",
        )
    assert recorded["n_jobs"] == 3

    # An explicit override wins (used by callers that know better than the budget).
    recorded.clear()
    with pytest.raises(RuntimeError, match="stop after construction"):
        training.fit_pipeline_bundle(
            "compute-fold",
            seq=["sklearn.tree.DecisionTreeClassifier"],
            checkpoint_dir=tmp_path / "ckpt",
            n_jobs=7,
        )
    assert recorded["n_jobs"] == 7


def test_experiment_bundle_uses_the_same_fold_width(client, small_host, tmp_path, monkeypatch):
    _register(client, "compute-exp")
    compute.save(
        compute.defaults(cores=8) | {"job_workers": 2, "blas_threads": 1, "job_threads": 2}
    )
    recorded = {}

    class StubAML:
        def __init__(self, **kwargs):
            recorded.update(kwargs)
            raise RuntimeError("stop after construction")

    monkeypatch.setattr(training, "AML", StubAML)
    with pytest.raises(RuntimeError, match="stop after construction"):
        training.fit_experiment_bundle(
            "compute-exp",
            config={"sklearn.tree.DecisionTreeClassifier": {}},
            checkpoint_dir=tmp_path / "ckpt",
        )
    assert recorded["n_jobs"] == 2


def test_job_pool_never_exceeds_the_configured_width(client, small_host):
    """One slot means one job at a time, no matter how many are submitted."""
    import threading

    compute.save(compute.defaults(cores=8) | {"job_workers": 1, "job_threads": 1})
    lock = threading.Lock()
    active = 0
    peak = 0

    def factory(_job_id):
        def run():
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.3)
            with lock:
                active -= 1
            return {"ok": True}

        return run

    ids = [job_manager.submit("pool-task", "test", factory) for _ in range(3)]
    for job_id in ids:
        record = _wait(job_id)
        assert record["status"] == "completed", record["error"]
    assert peak == 1


@pytest.mark.skipif(
    os.environ.get("SKSURROGATE_SKIP_DASK_CLUSTER") == "1",
    reason="real Dask cluster test disabled by environment",
)
def test_real_dask_cluster_runs_a_task_within_the_budget(client, small_host):
    pytest.importorskip("distributed")
    compute.save(
        compute.defaults(cores=8) | {"dask_workers": 1, "dask_threads": 1, "job_workers": 1, "job_threads": 1}
    )
    cluster = compute.dask_cluster()
    assert cluster is not None
    try:
        client_handle = cluster.get_client()
        try:
            assert len(cluster.workers) == 1
            worker = next(iter(cluster.workers.values()))
            assert worker.nthreads == 1
            assert client_handle.submit(lambda: 6 * 7).result(timeout=60) == 42
        finally:
            client_handle.close()
    finally:
        cluster.close()
