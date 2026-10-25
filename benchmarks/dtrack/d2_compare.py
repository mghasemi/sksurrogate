"""D2 — Head-to-head surrogate comparison (Hilbert vs GP vs RF) + regime map.

Fits three surrogate *regressors* on the same training samples drawn from the
closed-form synthetic objectives and measures RMSE on a dense test grid:

- ``HilbertRegressor(deg=3)``  (the custom Hilbert-space method)
- ``GaussianProcessRegressor`` (RBF kernel, stock sklearn)
- ``RandomForestRegressor``   (100 trees, stock sklearn)

The regime grid spans objective family x dimension x sample size x noise. For
each cell the mean RMSE over repeated draws is recorded, and the winning
surrogate is flagged — producing the "regime map" (which surrogate wins where).

TPE is not a point-predictor regressor (it is a density-ratio sampler), so it is
not in this regressor table; it enters the study as a search baseline in the
write-up (evals-to-target, cf. the B4 benchmark on real datasets).

Results land in ``benchmarks/results/dtrack/d2/`` (raw JSON per cell + long-format
CSV aggregate). Raw dirs are git-ignored; aggregates are promoted into docs.

Usage::

    python -m benchmarks.dtrack.d2_compare            # full matrix
    python -m benchmarks.dtrack.d2_compare --smoke    # tiny subset (fast)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .objectives import DIMS, FAMILIES, make_objective, sample_domain, test_grid

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_ROOT = REPO_ROOT / "benchmarks" / "results" / "dtrack" / "d2"

#: regime grid
N_SMALL, N_MED, N_LARGE = 20, 100, 500
N_LEVELS = [N_SMALL, N_MED, N_LARGE]
NOISE_LEVELS = [0.0, 0.05]
REPS = 10
TEST_N = 2000

SURROGATES = ["hilbert", "gp", "rf"]


def _make_regressor(name: str):
    if name == "hilbert":
        from SKSurrogate import HilbertRegressor

        return HilbertRegressor(deg=3)
    if name == "gp":
        from sklearn.gaussian_process import GaussianProcessRegressor

        return GaussianProcessRegressor()
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=100, random_state=0)
    raise ValueError(name)


def _fit_rmse(reg, X, y, Xtest, ytest):
    try:
        reg.fit(X, y)
        pred = reg.predict(Xtest)
        if not np.all(np.isfinite(pred)):
            return None
        return float(np.sqrt(np.mean((pred - ytest) ** 2)))
    except Exception:
        return None


def run_d2(smoke: bool = False) -> list[dict]:
    rows: list[dict] = []
    n_levels = N_LEVELS if not smoke else [N_SMALL, N_MED]
    noise_levels = NOISE_LEVELS if not smoke else [0.0, 0.05]
    reps = REPS if not smoke else 2

    for family in FAMILIES:
        for d in DIMS:
            f, _name = make_objective(family, d)
            Xtest = test_grid(TEST_N, d, np.random.default_rng(12345))
            ytest = f(Xtest)
            for n in n_levels:
                for sigma in noise_levels:
                    for rep in range(reps):
                        rng = np.random.default_rng(rep)
                        X = sample_domain(n, d, rng)
                        y = f(X) + (sigma * rng.standard_normal(n) if sigma > 0 else 0.0)
                        for sname in SURROGATES:
                            reg = _make_regressor(sname)
                            rmse = _fit_rmse(reg, X, y, Xtest, ytest)
                            rows.append({
                                "family": family, "dim": d, "n": n, "sigma": sigma,
                                "rep": rep, "surrogate": sname, "rmse": rmse,
                                "ok": rmse is not None,
                            })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D2 surrogate head-to-head")
    ap.add_argument("--smoke", action="store_true", help="tiny fast subset")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    rows = run_d2(smoke=args.smoke)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    import csv

    with open(OUT_ROOT / "d2_aggregate.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT_ROOT / "d2_manifest.json").write_text(
        json.dumps({"smoke": args.smoke, "n_cells": len(rows),
                    "elapsed_s": round(time.perf_counter() - t0, 1)}, indent=2),
        encoding="utf-8",
    )

    import pandas as pd

    df = pd.DataFrame(rows)
    ok = df[df["ok"]]
    if not ok.empty:
        # regime map: mean RMSE by (family, dim, n, sigma, surrogate)
        g = ok.groupby(["family", "dim", "n", "sigma", "surrogate"])["rmse"].mean().reset_index()
        # winner per cell
        g["winner"] = g.groupby(["family", "dim", "n", "sigma"])["rmse"].transform("min")
        g["is_winner"] = np.isclose(g["rmse"], g["winner"])
        print("Regime map (mean RMSE; * = winner):")
        for (fam, d, n, sig), sub in g.groupby(["family", "dim", "n", "sigma"]):
            line = f"{fam:8s} d={d} n={n:4d} sig={sig:.2f}: "
            for sname in SURROGATES:
                row = sub[sub["surrogate"] == sname]
                if row.empty:
                    continue
                v = row["rmse"].iloc[0]
                star = "*" if row["is_winner"].iloc[0] else " "
                line += f"{sname}={v:.4f}{star}  "
            print(line)
    fail_rate = 1.0 - df["ok"].mean()
    print(f"\nfit-failure rate across all cells: {fail_rate:.3f}")
    print(f"wrote {len(rows)} cells -> {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
