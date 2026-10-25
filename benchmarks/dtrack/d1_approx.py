"""D1 — Characterize HilbertRegressor approximation behavior & small-n stability.

Runs the custom Hilbert-space regressor on closed-form synthetic objectives and
measures, for each (objective family, dimension, degree):

1. **Convergence rate vs sample size** — RMSE on a dense test grid as the number
   of training points ``n`` grows (log-log slope estimates the rate).
2. **Stability at small n** — mean +/- std RMSE over repeated draws at small ``n``,
   including how often the fit fails (singular Gram matrix / NaN coefficients).
3. **Noise robustness** — RMSE degradation as Gaussian noise is added to the
   objective evaluations at fixed ``n``.

Results land in ``benchmarks/results/dtrack/d1/`` (raw JSON per cell + a long-format
CSV aggregate). Raw dirs are git-ignored; aggregates are promoted into docs.

Usage::

    python -m benchmarks.dtrack.d1_approx            # full matrix
    python -m benchmarks.dtrack.d1_approx --smoke    # tiny subset (fast)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .objectives import DIMS, FAMILIES, make_objective, sample_domain, test_grid

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_ROOT = REPO_ROOT / "benchmarks" / "results" / "dtrack" / "d1"

#: sample sizes for the convergence sweep
CONV_N = [10, 20, 50, 100, 200, 500]
#: small-n stability sweep
SMALL_N = [10, 20, 30]
SMALL_REPS = 20
#: noise levels (std of Gaussian noise, relative to unit-scale response)
NOISE_SIGMAS = [0.0, 0.01, 0.05, 0.1]
NOISE_N = 100
NOISE_REPS = 10
#: dense test grid size
TEST_N = 2000


def _fit_rmse(reg, X, y, Xtest, ytest):
    """Fit ``reg`` on (X, y), return RMSE on the test grid, or None on failure."""
    try:
        reg.fit(X, y)
        pred = reg.predict(Xtest)
        if not np.all(np.isfinite(pred)):
            return None
        return float(np.sqrt(np.mean((pred - ytest) ** 2)))
    except Exception:
        return None


def _run_cell(family, d, deg, n, sigma, seed):
    """One (family, d, deg, n, sigma, seed) cell -> dict or None."""
    f, name = make_objective(family, d)
    rng = np.random.default_rng(seed)
    X = sample_domain(n, d, rng)
    y = f(X) + (sigma * rng.standard_normal(n) if sigma > 0 else 0.0)
    Xtest = test_grid(TEST_N, d, np.random.default_rng(12345))
    ytest = f(Xtest)
    from SKSurrogate import HilbertRegressor

    reg = HilbertRegressor(deg=deg)
    rmse = _fit_rmse(reg, X, y, Xtest, ytest)
    return {
        "family": family, "dim": d, "deg": deg, "n": n, "sigma": sigma, "seed": seed,
        "rmse": rmse, "ok": rmse is not None,
    }


def run_d1(smoke: bool = False) -> list[dict]:
    rows: list[dict] = []
    conv_n = CONV_N if not smoke else [10, 50]
    small_n = SMALL_N if not smoke else [10]
    small_reps = SMALL_REPS if not smoke else 3
    noise_sigmas = NOISE_SIGMAS if not smoke else [0.0, 0.1]
    noise_reps = NOISE_REPS if not smoke else 2
    degs = [1, 2, 3]

    for family in FAMILIES:
        for d in DIMS:
            # 1) convergence sweep (single seed per n)
            for n in conv_n:
                for deg in degs:
                    rows.append(_run_cell(family, d, deg, n, 0.0, seed=0))
            # 2) small-n stability (repeated draws)
            for n in small_n:
                for deg in degs:
                    for rep in range(small_reps):
                        rows.append(_run_cell(family, d, deg, n, 0.0, seed=rep))
            # 3) noise robustness (fixed n, repeated draws)
            for sigma in noise_sigmas:
                for deg in degs:
                    for rep in range(noise_reps):
                        rows.append(_run_cell(family, d, deg, NOISE_N, sigma, seed=rep))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D1 Hilbert approximation study")
    ap.add_argument("--smoke", action="store_true", help="tiny fast subset")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    rows = run_d1(smoke=args.smoke)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    import csv

    with open(OUT_ROOT / "d1_aggregate.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT_ROOT / "d1_manifest.json").write_text(
        json.dumps({"smoke": args.smoke, "n_cells": len(rows),
                    "elapsed_s": round(time.perf_counter() - t0, 1)}, indent=2),
        encoding="utf-8",
    )

    # quick summary: mean RMSE by (family, dim, deg) at the largest n
    import pandas as pd

    df = pd.DataFrame(rows)
    ok = df[df["ok"]]
    if not ok.empty:
        big = ok[ok["n"] == ok["n"].max()]
        print("Mean RMSE at largest n, by (family, dim, deg):")
        print(big.groupby(["family", "dim", "deg"])["rmse"].mean().round(4).to_string())
    fail_rate = 1.0 - df["ok"].mean()
    print(f"\nfit-failure rate across all cells: {fail_rate:.3f}")
    print(f"wrote {len(rows)} cells -> {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
