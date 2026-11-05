"""Profile candidate-evaluation and surrogate-search overhead on a fixed split.

Run with ``python -m benchmarks.dtrack.profile_search_overhead``. The reported
cProfile cumulative times are diagnostic and should not be treated as
unprofiled wall-clock benchmarks.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path

from sklearn.model_selection import train_test_split

from benchmarks.datasets import load_dataset
from benchmarks.methods import METHODS

PROFILE_METHODS = (
    ("hilbert", METHODS["sksurrogate"]),
    ("gp", METHODS["sksurrogate-gp"]),
    ("no-surrogate", METHODS["srs-no-surrogate"]),
)


def profile_search(
    dataset: str = "galaxy3",
    budget: int = 30,
    seed: int = 0,
    n_estimators: int = 10,
) -> dict:
    if budget < 21:
        raise ValueError("budget must be at least 21 so the surrogate arms engage")
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1")

    X, y = load_dataset(dataset)
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )
    results = []
    for name, runner in PROFILE_METHODS:
        profiler = cProfile.Profile()
        record = profiler.runcall(
            runner, Xtr, ytr, Xte, yte, budget, seed, n_estimators
        )
        stats = pstats.Stats(profiler)
        functions = [
            {
                "file": Path(filename).name,
                "line": line,
                "function": function,
                "calls": call_data[1],
                "total_seconds": call_data[2],
                "cumulative_seconds": call_data[3],
            }
            for (filename, line, function), call_data in stats.stats.items()
            if filename.endswith(("structsearch.py", "NpyProximation.py"))
            or "gaussian_process" in filename
        ]
        functions.sort(key=lambda item: item["cumulative_seconds"], reverse=True)

        evaluation_seconds = sum(
            float(item.get("wall_s") or 0.0) for item in record.records
        )
        total_seconds = record.total_wall_s or 0.0
        results.append(
            {
                "method": name,
                "surrogate": record.extra.get("surrogate"),
                "surrogate_source": record.extra.get("surrogate_source"),
                "n_evals": record.n_evals,
                "search_wall_seconds": total_seconds,
                "candidate_evaluation_seconds": evaluation_seconds,
                "residual_overhead_fraction": (
                    max(total_seconds - evaluation_seconds, 0.0) / total_seconds
                    if total_seconds
                    else 0.0
                ),
                "profile_functions": functions[:15],
            }
        )

    return {
        "config": {
            "dataset": dataset,
            "budget": budget,
            "seed": seed,
            "n_estimators": n_estimators,
            "split": "stratified 80/20",
            "cv": "stratified 5-fold",
        },
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="galaxy3")
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=10)
    args = parser.parse_args(argv)
    if args.budget < 21:
        parser.error("--budget must be at least 21 so the surrogate arms engage")
    if args.n_estimators < 1:
        parser.error("--n-estimators must be at least 1")

    report = profile_search(
        dataset=args.dataset,
        budget=args.budget,
        seed=args.seed,
        n_estimators=args.n_estimators,
    )
    print(
        f"Component profile: {args.dataset}, budget={args.budget}, "
        f"seed={args.seed}, RF({args.n_estimators})"
    )
    for result in report["results"]:
        print(
            f"\n{result['method']} ({result['surrogate']}): "
            f"evals={result['n_evals']}, wall={result['search_wall_seconds']:.3f}s, "
            f"candidate_cv={result['candidate_evaluation_seconds']:.3f}s, "
            f"residual={result['residual_overhead_fraction']:.1%}"
        )
        for item in result["profile_functions"]:
            print(
                f"  {item['cumulative_seconds']:.3f}s cumulative "
                f"({item['total_seconds']:.3f}s self), "
                f"{item['calls']} calls: "
                f"{item['file']}:{item['line']} {item['function']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
