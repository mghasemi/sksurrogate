"""Regression tests for benchmark dataset loading (Stage A, AI_LAYER02_REMAINING_PLAN).

The warm cache in ``benchmarks.datasets`` must be honored *without* the optional
``openml`` dependency, so that cached CSVs load in slim environments (CI-mirror
images, production Docker image) that do not install the ``[benchmark]`` extra.
A genuine download still fails with the actionable install hint.

The tests are hermetic: a temp spec file replaces ``benchmarks/datasets.json``
and a temp directory replaces the cache, so no real dataset or network access
is involved. Blocking ``openml`` is done by setting ``sys.modules["openml"] =
None``, which makes any ``import openml`` raise ``ImportError`` — i.e., it
simulates an environment where the package is not installed.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest


def _write_spec(tmp_path, monkeypatch) -> None:
    """Point ``benchmarks.datasets`` at a temp spec with one synthetic OpenML set."""
    # 12 rows x 4 numeric features + target (last column) with exactly 3 classes,
    # matching the pinning that load_dataset verifies against.
    spec = {"datasets": {"synthetic-set": {"did": 9999, "n": 12, "d": 4, "classes": 3}}}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr("benchmarks.datasets._SPEC_PATH", spec_path)
    monkeypatch.setenv("SKSURROGATE_BENCH_CACHE", str(tmp_path / "cache"))


def _write_cached_csv() -> pd.DataFrame:
    """A warm-cache CSV consistent with the synthetic spec's pinning."""
    rng = np.random.default_rng(0)
    features = rng.uniform(size=(12, 4)).round(3)
    target = np.array([0, 1, 2] * 4, dtype=float)  # exactly 3 distinct values
    df = pd.DataFrame(features)
    df.columns = pd.Index([f"feature{i}" for i in range(4)])
    df["class"] = target
    return df


def test_cached_dataset_loads_without_openml(tmp_path, monkeypatch):
    """A warm-cache CSV loads through load_dataset with openml unavailable."""
    _write_spec(tmp_path, monkeypatch)
    from benchmarks import datasets as ds

    cache_dir = ds.cache_dir()  # also creates the directory
    _write_cached_csv().to_csv(cache_dir / "synthetic-set.csv", index=False)

    # Simulate an environment without openml installed. If the code path still
    # imports it before checking the cache, this raises and the test fails.
    monkeypatch.setitem(sys.modules, "openml", None)

    X, y = ds.load_dataset("synthetic-set")

    assert X.shape == (12, 4)
    assert list(X.columns) == [f"feature{i}" for i in range(4)]
    assert set(np.asarray(y.unique())) == {0.0, 1.0, 2.0}


def test_cache_miss_without_openml_raises_actionable_error(tmp_path, monkeypatch):
    """A real download (cache miss) still fails with the install hint."""
    _write_spec(tmp_path, monkeypatch)
    from benchmarks import datasets as ds

    ds.cache_dir()  # create an *empty* cache directory
    monkeypatch.setitem(sys.modules, "openml", None)

    with pytest.raises(ImportError, match="pip install sksurrogate\\[benchmark\\]"):
        ds.load_dataset("synthetic-set")


def test_download_path_still_used_on_cache_miss(tmp_path, monkeypatch):
    """With openml available and no cache, the download branch runs and caches."""
    _write_spec(tmp_path, monkeypatch)
    from benchmarks import datasets as ds

    calls: list[int] = []

    class _FakeDataset:
        def get_data(self, dataset_format=None):  # noqa: ANN001 - mirrors openml-py API
            return _write_cached_csv(), None, None, []

    class _FakeOpenml:
        datasets = type("datasets", (), {"get_dataset": staticmethod(lambda did: calls.append(did) or _FakeDataset())})

    monkeypatch.setitem(sys.modules, "openml", _FakeOpenml)

    X, y = ds.load_dataset("synthetic-set")

    assert calls == [9999]  # downloaded by the pinned did
    assert (ds.cache_dir() / "synthetic-set.csv").exists()  # result cached for next time
    assert X.shape == (12, 4)


def test_unknown_dataset_name_still_rejected(tmp_path, monkeypatch):
    """The spec round-trip does not mask the unknown-name guard."""
    _write_spec(tmp_path, monkeypatch)
    from benchmarks import datasets as ds

    with pytest.raises(ValueError, match="Unknown benchmark dataset"):
        ds.load_dataset("does-not-exist")
