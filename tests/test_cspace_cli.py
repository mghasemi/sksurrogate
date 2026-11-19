"""CLI-level integration test for the C-space track (DESIGN-cspace.md §9).

Exercises the real pipeline end to end — dataset load, split, arm dispatch,
per-cell JSON, crash-resilient jsonl, summary.csv, and the manifest — at a budget
small enough for CI. Runner semantics are covered in test_cspace_runner.py.
"""

import json

import pandas as pd
import pytest

# The C-space harness needs the optional [benchmark] extra — openml for the pinned
# datasets and optuna for the TPE arm. The base "Tests + smoke" CI job installs
# only base deps, so skip these there; the benchmark-smoke job installs
# .[benchmark] and runs them.
pytest.importorskip("openml")
pytest.importorskip("optuna")

from benchmarks.run_cspace import run_suite


def test_run_suite_writes_every_artefact(tmp_path):
    aggregate = run_suite(
        protocol="B",
        datasets=["ionosphere", "galaxy3"],
        methods=["random-cspace", "optuna-tpe-cspace"],
        seeds=[0],
        budget=4,
        wall=None,
        out_dir=tmp_path,
    )

    # summary.csv is non-empty and every cell reports ok
    assert not aggregate.empty
    assert len(aggregate) == 4  # 2 datasets x 1 seed x 2 arms
    assert set(aggregate["status"]) == {"ok"}
    assert set(aggregate["protocol"]) == {"B"}
    on_disk = pd.read_csv(tmp_path / "summary.csv")
    assert len(on_disk) == 4
    assert on_disk["final_test_accuracy"].notna().all()

    # per-cell JSON + jsonl + manifest
    cells = sorted(path.name for path in tmp_path.glob("*__*__seed*.json"))
    assert len(cells) == 4
    lines = (tmp_path / "results.jsonl").read_text().strip().splitlines()
    assert len(lines) == 4
    for line in lines:
        payload = json.loads(line)
        assert payload["status"] == "ok"
        assert payload["dataset"] in {"ionosphere", "galaxy3"}

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["track"] == "cspace"
    assert manifest["n_records"] == 4
    assert manifest["n_errors"] == 0
    assert manifest["machine"]["packages"]["thread_policy"].startswith("threadpoolctl")
    assert "sha" in manifest["git"] and "dirty" in manifest["git"]


def test_a_failing_arm_is_recorded_not_dropped(tmp_path):
    """A broken arm must appear in summary.csv with status=error (never silently skipped)."""

    def explode(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    import benchmarks.run_cspace as cli

    original = cli.METHODS.get("random-cspace")
    cli.METHODS["random-cspace"] = explode
    try:
        aggregate = run_suite(
            protocol="B",
            datasets=["ionosphere"],
            methods=["random-cspace"],
            seeds=[0],
            budget=2,
            wall=None,
            out_dir=tmp_path,
        )
    finally:
        cli.METHODS["random-cspace"] = original

    assert len(aggregate) == 1
    assert aggregate.iloc[0]["status"] == "error"
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["n_errors"] == 1
