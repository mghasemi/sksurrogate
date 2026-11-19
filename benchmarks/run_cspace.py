"""CLI for the pipeline-space benchmark (DESIGN-cspace.md §3, §9).

Three protocols, one command each::

    # primary: equal wall-clock
    python -m benchmarks.run_cspace --protocol A --wall 120 --seeds 7

    # secondary: exactly 40 candidate evaluations
    python -m benchmarks.run_cspace --protocol B --budget 40 --seeds 7

    # system level (FLAML / AutoGluon / SRS) -- separate table, never pooled
    python -m benchmarks.run_cspace --protocol C --wall 120 --seeds 7

    # CI gate
    python -m benchmarks.run_cspace --smoke

Outputs land in ``benchmarks/results/cspace-<protocol>-<UTC stamp>/``::

    run_manifest.json   protocol, matrix, machine, package versions, git SHA + dirty
    results.jsonl       crash-resilient append log (one record per line)
    <method>__<dataset>__seed<k>.json
    summary.csv         long-format aggregate

Raw directories are archived to Zenodo; only aggregates and figures are committed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

from benchmarks.cspace_methods import METHODS, PROTOCOLS, train_test_split_frame
from benchmarks.datasets import dataset_names, load_dataset

#: CI gate (DESIGN-cspace.md §9): must finish in well under 3 minutes.
SMOKE = {
    "protocol": "B",
    "datasets": ["ionosphere"],
    "seeds": 1,
    "budget": 8,
    "wall": 15,
    "methods": ["random-cspace", "optuna-tpe-cspace", "srs-cspace", "srs-nosurr-cspace"],
}


def _machine_info() -> dict:
    """Machine + package manifest (wall-clock claims are hardware-dependent)."""
    cpu_model = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    def version(name: str) -> str:
        try:
            from importlib.metadata import version as dist_version

            return dist_version(name)
        except Exception:  # noqa: BLE001 - optional dependency
            return "n/a"

    repo = Path(__file__).resolve().parent.parent
    git_sha, git_dirty = "", None
    try:
        git_sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        )
    except Exception:  # noqa: BLE001 - not a checkout
        pass

    packages = {
        name: version(name)
        for name in (
            "numpy", "pandas", "scikit-learn", "scipy", "SKSurrogate", "optuna",
            "flaml", "autogluon.tabular", "threadpoolctl",
        )
    }
    # Thread pinning happens inside each fit (threadpoolctl limits=1, see
    # benchmarks.cspace_methods._one_thread); raw env vars would read as "unpinned"
    # and misrepresent the measured policy, so record the policy itself.
    packages["thread_policy"] = "threadpoolctl limits=1 around every fit; n_jobs=1"
    packages["threads_env"] = {
        key: os.environ.get(key)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    return {
        "machine": {
            "platform": platform.platform(),
            "cpu_model": cpu_model,
            "vcpus": os.cpu_count(),
            "python": platform.python_version(),
            "packages": packages,
        },
        "git": {"sha": git_sha, "dirty": git_dirty},
    }


def _quiet_known_benign() -> None:
    """Silence warnings that are expected across this space, by exact pattern.

    Each of these fires on legitimate configurations and would otherwise flood the
    log of an 8-hour run (and hide real failures):
    * ``QuantileTransformer`` with fewer samples than its default ``n_quantiles``;
    * ``SelectKBest`` on a dataset with a constant feature (ionosphere has one);
    * the resulting zero-variance divide warning;
    * L-BFGS/GP hyperparameter convergence notices on small evaluation budgets.
    """
    import warnings

    patterns = (
        r"n_quantiles \(\d+\) is greater than the total number of samples",
        r"Features \[.*\] are constant",
        r"invalid value encountered in divide",
        r"lbfgs failed to converge",
        r"The optimal value found for dimension",
        r"Variable(s) .* are constant",
    )
    for pattern in patterns:
        warnings.filterwarnings("ignore", message=pattern)


def _parse_list(raw: str, known: list[str], what: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(known)
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in known]
    if unknown:
        raise SystemExit(f"Unknown {what}: {unknown}. Known: {sorted(known)}")
    return names


def run_suite(
    protocol: str,
    datasets: list[str],
    methods: list[str],
    seeds: list[int],
    budget: int | None,
    wall: float | None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "benchmarks" / "results" / f"cspace-{protocol}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "track": "cspace",
        "protocol": protocol,
        "datasets": datasets,
        "methods": methods,
        "seeds": seeds,
        "budget_evals": budget,
        "budget_wall_s": wall,
        "started": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "protocol_note": {
            "A": "equal wall-clock — primary",
            "B": "equal candidate evaluations — secondary",
            "C": "system level — separate table, never pooled with A/B",
        }[protocol],
    }
    manifest.update(_machine_info())
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    jsonl = out_dir / "results.jsonl"
    rows: list[dict] = []
    for name in datasets:
        X, y = load_dataset(name)
        for seed in seeds:
            Xtr, Xte, ytr, yte = train_test_split_frame(X, y, seed)
            for method in methods:
                runner = METHODS[method]
                started = time.perf_counter()
                try:
                    record = runner(Xtr, ytr, Xte, yte, budget=budget, wall=wall, seed=seed)
                    payload = json.loads(record.to_json())
                    payload["dataset"] = name
                except Exception as exc:  # noqa: BLE001 - keep the record, never drop it
                    payload = {
                        "method": method, "dataset": name, "seed": seed,
                        "status": "error",
                        "error": f"{exc.__class__.__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=3),
                        "total_wall_s": time.perf_counter() - started,
                        "extra": {"protocol": protocol},
                    }
                payload.update({"protocol": protocol, "budget_evals": budget, "budget_wall_s": wall})
                rows.append(payload)
                with open(jsonl, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, default=str) + "\n")
                (out_dir / f"{method}__{name}__seed{seed}.json").write_text(
                    json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
                )
                score = payload.get("final_test_accuracy")
                print(
                    f"{name:<10} seed={seed} {method:<18} "
                    f"{'acc=%.4f' % score if score is not None else 'FAILED'} "
                    f"wall={payload.get('total_wall_s', float('nan')):.1f}s "
                    f"evals={payload.get('n_evals')} "
                    f"engaged={payload.get('extra', {}).get('engaged_at')}",
                    flush=True,
                )

    summary_rows = []
    for payload in rows:
        extra = payload.get("extra") or {}
        summary_rows.append(
            {
                "protocol": protocol,
                "dataset": payload.get("dataset"),
                "seed": payload.get("seed"),
                "method": payload.get("method"),
                "status": payload.get("status"),
                "n_evals": payload.get("n_evals"),
                "final_cv_best": payload.get("final_cv_best"),
                "final_test_accuracy": payload.get("final_test_accuracy"),
                "final_test_balanced": payload.get("final_test_balanced"),
                "total_wall_s": payload.get("total_wall_s"),
                "invalid_rate": extra.get("invalid_rate"),
                "duplicate_rate": extra.get("duplicate_rate"),
                "engaged_at": extra.get("engaged_at"),
                "n_surface_fits": extra.get("n_surface_fits"),
                "best_clf": (payload.get("best_params") or {}).get("clf"),
                "best_prep": (payload.get("best_params") or {}).get("prep"),
                "best_reduce": (payload.get("best_params") or {}).get("reduce"),
            }
        )
    aggregate = pd.DataFrame(summary_rows)
    aggregate.to_csv(out_dir / "summary.csv", index=False)

    manifest["finished"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    manifest["n_records"] = len(rows)
    manifest["n_errors"] = sum(1 for payload in rows if payload.get("status") != "ok")
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir} ({len(rows)} records, {manifest['n_errors']} errors)")
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", choices=["A", "B", "C"], default="A")
    parser.add_argument("--dataset", default="all", help="'all' or comma-separated names")
    parser.add_argument("--methods", default=None, help="comma-separated; default = the protocol's arm set")
    parser.add_argument("--seeds", type=int, default=7, help="number of seeds, fixed 0..N-1")
    parser.add_argument("--budget", type=int, default=None, help="candidate evaluations (Protocol B)")
    parser.add_argument("--wall", type=float, default=None, help="seconds per method (Protocols A/C)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="CI subset: ionosphere, 1 seed, B=8")
    args = parser.parse_args(argv)
    _quiet_known_benign()

    if args.smoke:
        out = args.out or Path("benchmarks/results") / (
            "cspace-smoke-" + _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        run_suite(
            SMOKE["protocol"], SMOKE["datasets"], SMOKE["methods"],
            list(range(SMOKE["seeds"])), SMOKE["budget"], SMOKE["wall"], out,
        )
        return 0

    budget, wall = args.budget, args.wall
    if args.protocol == "B":
        budget = budget if budget is not None else 40
        wall = None
    else:
        wall = wall if wall is not None else 120.0
        budget = None if args.protocol == "A" else None

    datasets = _parse_list(args.dataset, dataset_names(), "dataset(s)")
    methods = (
        _parse_list(args.methods, list(METHODS), "method(s)")
        if args.methods
        else PROTOCOLS[args.protocol]
    )
    run_suite(args.protocol, datasets, methods, list(range(args.seeds)), budget, wall, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
