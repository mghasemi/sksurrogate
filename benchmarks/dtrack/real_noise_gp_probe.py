"""Compare surrogate surfaces on real cross-validation search objectives.

Run with ``python -m benchmarks.dtrack.real_noise_gp_probe``. This bounded
study complements the synthetic regression probe; it does not by itself
justify changing the library's default surrogate.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split

from benchmarks.datasets import load_dataset
from benchmarks.methods import (
    METHODS,
    RunRecord,
    _run_srs,
)
from benchmarks.run import _machine_info

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "benchmarks" / "results" / "dtrack"


def _run_noise_aware_gp(Xtr, ytr, Xte, yte, budget: int, seed: int, n_estimators: int):
    return _run_srs(
        Xtr,
        ytr,
        Xte,
        yte,
        budget,
        seed,
        n_estimators,
        task_tag="srsnoisegp",
        method_name="sksurrogate-gp-white",
        regressor_factory=lambda: GaussianProcessRegressor(
            kernel=RBF() + WhiteKernel(),
            random_state=seed,
        ),
    )


def _run_rf_surrogate(Xtr, ytr, Xte, yte, budget: int, seed: int, n_estimators: int):
    return _run_srs(
        Xtr,
        ytr,
        Xte,
        yte,
        budget,
        seed,
        n_estimators,
        task_tag="srsrf",
        method_name="sksurrogate-rf",
        regressor_factory=lambda: RandomForestRegressor(
            n_estimators=100,
            random_state=seed,
        ),
    )


METHODS_TO_COMPARE = (
    ("hilbert", METHODS["sksurrogate"]),
    ("gp", METHODS["sksurrogate-gp"]),
    ("gp-white", _run_noise_aware_gp),
    ("random-forest", _run_rf_surrogate),
    ("srs-no-surrogate", METHODS["srs-no-surrogate"]),
    ("random", METHODS["random"]),
)

SKLEARN_DATASETS = {
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
    "wine": load_wine,
}


def _load_probe_dataset(dataset: str):
    if dataset == "galaxy3":
        return load_dataset(dataset)
    if dataset not in SKLEARN_DATASETS:
        raise ValueError(f"Unknown probe dataset {dataset!r}")
    bunch = SKLEARN_DATASETS[dataset](as_frame=True)
    return bunch.data, bunch.target


def _summarize(records: list[RunRecord]) -> dict[str, dict[str, float | int]]:
    summary = {}
    for name, _runner in METHODS_TO_COMPARE:
        rows = [record for record in records if record.method == name and record.status == "ok"]
        if not rows:
            continue
        overheads = []
        for row in rows:
            evaluation_time = sum(
                float(item.get("wall_s") or 0.0) for item in row.records
            )
            total = row.total_wall_s or 0.0
            overheads.append(max(total - evaluation_time, 0.0) / total if total else 0.0)
        summary[name] = {
            "runs": len(rows),
            "median_best_cv": float(statistics.median(
                row.final_cv_best for row in rows if row.final_cv_best is not None
            )),
            "median_test_accuracy": float(statistics.median(
                row.final_test_accuracy
                for row in rows if row.final_test_accuracy is not None
            )),
            "median_search_wall_s": float(statistics.median(
                row.total_wall_s for row in rows if row.total_wall_s is not None
            )),
            "median_overhead_fraction": float(statistics.median(overheads)),
        }
    return summary


def run_probe(
    budget: int = 30,
    seeds=range(7),
    n_estimators: int = 10,
    dataset: str = "galaxy3",
) -> dict:
    """Run equal-budget search arms on Galaxy3 or a bundled sklearn dataset."""
    if budget < 21:
        raise ValueError(
            "budget must be at least 21 so the three-parameter surrogate arms engage"
        )
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1")
    X, y = _load_probe_dataset(dataset)
    records: list[RunRecord] = []
    started = time.perf_counter()
    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=seed,
            stratify=y,
        )
        for method, runner in METHODS_TO_COMPARE:
            record = runner(Xtr, ytr, Xte, yte, budget, seed, n_estimators)
            record.method = method
            record.dataset = dataset
            if record.status != "ok" or record.n_evals != budget:
                raise RuntimeError(
                    f"{method} seed={seed} did not complete its budget: "
                    f"status={record.status}, n_evals={record.n_evals}, error={record.error}"
                )
            records.append(record)

    return {
        "config": {
            "dataset": dataset,
            "rows": int(len(X)),
            "features": int(X.shape[1]),
            "budget": budget,
            "seeds": list(seeds),
            "n_estimators": n_estimators,
            "split": "stratified 80/20 by seed",
            "cv": "stratified 5-fold",
        },
        "machine": _machine_info(),
        "elapsed_s": time.perf_counter() - started,
        "summary": _summarize(records),
        "records": [dataclasses.asdict(record) for record in records],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["galaxy3", *SKLEARN_DATASETS],
        default="galaxy3",
        help="local benchmark or bundled scikit-learn dataset",
    )
    parser.add_argument("--budget", type=int, default=30, help="candidate evaluations per arm (>=21)")
    parser.add_argument("--seeds", type=int, default=7, help="number of fixed seeds, 0..N-1")
    parser.add_argument("--n-estimators", type=int, default=10, help="trees per tuned RF candidate")
    parser.add_argument("--out", type=Path, help="JSON results path (default: dataset-specific file)")
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    if args.budget < 21:
        parser.error("--budget must be at least 21 so the surrogate arms engage")
    if args.n_estimators < 1:
        parser.error("--n-estimators must be at least 1")

    result = run_probe(
        budget=args.budget,
        seeds=range(args.seeds),
        n_estimators=args.n_estimators,
        dataset=args.dataset,
    )
    out_path = args.out or DEFAULT_OUT_DIR / f"real_noise_gp_probe_{args.dataset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.dataset} real-workload probe (B={args.budget}, seeds={args.seeds})")
    for method, metrics in result["summary"].items():
        print(
            f"{method:18s} CV={metrics['median_best_cv']:.4f} "
            f"test={metrics['median_test_accuracy']:.4f} "
            f"wall={metrics['median_search_wall_s']:.3f}s "
            f"overhead={metrics['median_overhead_fraction']:.1%}"
        )
    print(f"wrote {len(result['records'])} runs to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
