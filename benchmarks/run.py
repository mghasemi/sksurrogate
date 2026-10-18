"""Benchmark runner CLI (DESIGN.md section 7).

Single-command reproduction of the full suite::

    python -m benchmarks.run                      # all datasets, budget 30, seeds 1-7
    python -m benchmarks.run --dataset ionosphere --budget 8 --seeds 2
    python -m benchmarks.run --smoke              # CI smoke subset (fast)

Raw results land in ``benchmarks/results/<UTC date>/``: one JSON per
(method, dataset, seed) run plus a crash-resilient ``results.jsonl`` append log
and an aggregated long-format CSV. Raw data is archived to Zenodo; aggregate
tables and figures are promoted into the docs (see plot.py).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .datasets import dataset_names, load_dataset
from .methods import METHODS, RunRecord

#: CI smoke subset (DESIGN.md section 8): must finish in well under 3 minutes.
SMOKE = {"datasets": ["ionosphere"], "budget": 8, "seeds": 2, "n_estimators": 10}


def _parse_datasets(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return dataset_names()
    names = [name.strip() for name in raw.split(",") if name.strip()]
    known = set(dataset_names())
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}. Known: {sorted(known)}")
    return names


def _parse_methods(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [n for n in names if n not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown method(s): {unknown}. Known: {sorted(METHODS)}")
    return names


def _machine_info() -> dict:
    """Machine + environment description for wall-clock claims (DESIGN.md section 10, risk #4)."""
    cpu_model = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    def _ver(name: str) -> str:
        try:
            from importlib.metadata import version

            return version(name)
        except Exception:  # noqa: BLE001 - optional dependency missing is fine

            return "n/a"

    repo = Path(__file__).resolve().parent.parent
    git_sha, git_dirty = "", ""
    try:
        git_sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        git_dirty = bool(dirty)
    except Exception:  # noqa: BLE001 - not a git checkout or git missing

        pass

    return {
        "machine": {
            "platform": platform.platform(),
            "cpu_model": cpu_model,
            "vcpus": os.cpu_count(),
            "python": platform.python_version(),
            "packages": {k: _ver(v) for k, v in [
                ("numpy", "numpy"), ("pandas", "pandas"), ("scikit-learn", "scikit-learn"),
                ("scipy", "scipy"), ("openml", "openml"), ("optuna", "optuna"),
            ]},
        },
        "git": {"sha": git_sha, "dirty": git_dirty},
    }


def run_suite(
    datasets: list[str],
    methods: list[str],
    budget: int,
    seeds: int,
    n_estimators: int = 50,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the full matrix and persist results. Returns the aggregate DataFrame."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "benchmarks" / "results" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Machine/environment manifest (DESIGN.md section 10, risk #4): wall-clock claims are only
    # meaningful alongside the machine description; evaluation-count metrics stay primary.
    import json as _json

    manifest = {
        "stamp": stamp,
        "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "config": {
            "datasets": list(datasets),
            "methods": list(methods),
            "budget": budget,
            "seeds": seeds,
            "n_estimators": n_estimators,
        },
    }
    manifest.update(_machine_info())
    (out_dir / "run_manifest.json").write_text(
        _json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    jsonl_path = out_dir / "results.jsonl"
    rows: list[dict] = []

    for dataset in datasets:
        X, y = load_dataset(dataset)
        print(f"[{dataset}] loaded n={len(X)} d={X.shape[1]} classes={y.nunique()}", flush=True)

        for seed_idx in range(seeds):
            seed = seed_idx  # fixed seeds 0..seeds-1: reproducible across runs/machines
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.20, random_state=seed, stratify=y
            )

            for method_name in methods:
                record = None
                started_line = f"[{dataset} seed={seed}] {method_name}"
                print(started_line + " ...", end="", flush=True)
                try:
                    record = METHODS[method_name](Xtr, ytr, Xte, yte, budget, seed, n_estimators)
                except Exception as exc:  # noqa: BLE001 - one bad run must not kill the suite
                    record = RunRecord(
                        method=method_name,
                        dataset=dataset,
                        seed=seed,
                        budget=budget,
                        n_estimators=n_estimators,
                        status="error",
                        error=f"{exc.__class__.__name__}: {exc}",
                    )
                    traceback.print_exc()

                record.dataset = dataset
                with open(jsonl_path, "a", encoding="utf-8") as fh:
                    fh.write(record.to_json().replace("\n", "") + "\n")
                (out_dir / f"{record.method}__{dataset}__seed{seed}.json").write_text(
                    record.to_json() + "\n", encoding="utf-8"
                )

                if record.status == "ok":
                    tail = (
                        f"done: {record.n_evals} evals, best CV={record.final_cv_best:.4f}, "
                        f"test acc={record.final_test_accuracy:.4f}"
                    )
                else:
                    tail = f"{record.status}: {(record.error or '')[:120]}"
                print(tail, flush=True)

                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": method_name,
                        "budget": budget,
                        "n_estimators": n_estimators,
                        "status": record.status,
                        "n_evals": record.n_evals,
                        "final_cv_best": record.final_cv_best,
                        "final_test_accuracy": record.final_test_accuracy,
                        "final_test_balanced": record.final_test_balanced,
                        "total_wall_s": record.total_wall_s,
                    }
                )

    aggregate = pd.DataFrame(rows)
    if not aggregate.empty:
        aggregate.to_csv(out_dir / "summary.csv", index=False)

    print(f"\nResults written to {out_dir}")
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SKSurrogate benchmark suite runner")
    parser.add_argument("--dataset", default="all", help="'all' or comma-separated names (default: all)")
    parser.add_argument("--methods", default=",".join(METHODS), help="comma-separated method names")
    parser.add_argument("--budget", type=int, default=30, help="model evaluations per search (default 30)")
    parser.add_argument("--seeds", type=int, default=7, help="number of seeds, fixed 0..N-1 (default 7)")
    parser.add_argument("--n-estimators", type=int, default=50, help="RF trees per candidate (default 50)")
    parser.add_argument("--smoke", action="store_true", help="CI smoke subset: ionosphere only, budget 8, 2 seeds, RF n_estimators=10")
    args = parser.parse_args(argv)

    if args.smoke:
        datasets = SMOKE["datasets"]
        budget = SMOKE["budget"]
        seeds = SMOKE["seeds"]
        n_estimators = SMOKE["n_estimators"]
        print(f"[smoke] {dict(SMOKE)}")
    else:
        datasets = _parse_datasets(args.dataset)
        budget, seeds, n_estimators = args.budget, args.seeds, args.n_estimators

    np.seterr(all="ignore")
    aggregate = run_suite(datasets, _parse_methods(args.methods), budget, seeds, n_estimators)

    hard_errors = 0
    if not aggregate.empty:
        with pd.option_context("display.width", 120):
            print(aggregate.pivot_table(
                index="dataset", columns="method", values="final_test_accuracy", aggfunc="mean"
            ).round(4))
        bad = aggregate[aggregate["status"].isin(["error"])]
        hard_errors = len(bad)
        skipped = aggregate[aggregate["status"] == "skipped_archived"]["method"].unique().tolist()
        for name in skipped:
            print(f"note: method '{name}' skipped (archived dependency) — recorded as data")

    return 1 if hard_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
