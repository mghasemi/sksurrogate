"""Gate G3 — pre-launch fit gate for the pipeline C-space (DESIGN-cspace.md §8).

Every configuration in the frozen space must *fit and predict* on every pinned
dataset before a full run may start. This is the check that caught arrhythmia's
408 missing cells, which silently invalidated three estimator families and would
have turned the benchmark into a comparison of failure-handling rather than of
search quality.

The gate is single-fit (train split only), not 5-fold: its purpose is to catch
configurations that cannot run at all. Fold-level behaviour is exercised by the
pilot and the full runs.

Usage::

    python -m benchmarks.gate_cspace_fit                       # all datasets
    python -m benchmarks.gate_cspace_fit --dataset ionosphere   # one
    python -m benchmarks.gate_cspace_fit --stride 50            # sampled (fast)

Exit code 0 only if every (config, dataset) pair fits. Failures are written to a
JSON report so they can be triaged without re-running.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from benchmarks.cspace import CARDINALITY, canonical_key, enumerate_space, to_sklearn
from benchmarks.datasets import dataset_names, load_dataset

__all__ = ["run_gate"]


def run_gate(datasets: list[str], stride: int = 1, seed: int = 0) -> dict:
    """Fit every (config, dataset) pair once; return a failure report."""
    from sklearn.model_selection import train_test_split

    space = enumerate_space()[::stride]
    report: dict = {
        "cardinality": CARDINALITY,
        "configs_checked": len(space),
        "stride": stride,
        "seed": seed,
        "datasets": {},
        "failures": [],
    }
    for name in datasets:
        X, y = load_dataset(name)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        started = time.perf_counter()
        failures = 0
        for cfg in space:
            try:
                model = to_sklearn(cfg, random_state=seed)
                model.fit(Xtr, ytr)
                prediction = model.predict(Xte)
                if prediction.shape != (Xte.shape[0],):
                    raise AssertionError(f"prediction shape {prediction.shape}")
            except Exception as exc:  # noqa: BLE001 - every failure is data
                failures += 1
                report["failures"].append(
                    {
                        "dataset": name,
                        "config": cfg,
                        "key": str(canonical_key(cfg)),
                        "error": f"{exc.__class__.__name__}: {exc}"[:400],
                    }
                )
        report["datasets"][name] = {
            "n": int(len(Xtr)),
            "d": int(Xtr.shape[1]),
            "configs": len(space),
            "failures": failures,
            "wall_s": round(time.perf_counter() - started, 1),
        }
        print(
            f"{name:<12} {len(space) - failures}/{len(space)} fit  "
            f"({failures} failures, {report['datasets'][name]['wall_s']}s)",
            flush=True,
        )
    report["ok"] = not report["failures"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--stride", type=int, default=1, help="check every Nth config (1 = all)")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/gate_cspace_fit.json"))
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")
    datasets = dataset_names() if args.dataset == "all" else [name.strip() for name in args.dataset.split(",")]
    report = run_gate(datasets, stride=args.stride)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n{'PASS' if report['ok'] else 'FAIL'} — wrote {args.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
