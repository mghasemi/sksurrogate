#!/usr/bin/env python3
"""Measured evidence for the two /concepts/ pages, produced by the shipped API.

Nothing here is asserted from prose: every number printed is computed by
SKSurrogate's own functions on scikit-learn's diabetes dataset (442 rows,
10 features, public, no download).

    python scripts/concepts_evidence.py                 # human-readable
    python scripts/concepts_evidence.py --json          # machine-readable for the page build
"""
import json
import sys
import tempfile

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

from SKSurrogate import mltrack
from SKSurrogate.mltrace import (
    prediction_interval_diagnostics,
    split_conformal_regression_interval,
)


def split_data(seed=0):
    d = load_diabetes()
    X, y = d.data, d.target
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    a, b = idx[:280], idx[280:360]          # train / calibration
    c = idx[360:]                            # held-out test
    return X, y, a, b, c


def nested_run(estimator, repeats=5, seed=0):
    """Drive the shipped nested_cv_evaluation through a real task."""
    X, y, a, b, c = split_data(seed)
    import pandas as pd
    cols = [f"x{i}" for i in range(X.shape[1])]
    train = pd.DataFrame(X[a], columns=cols); train["target"] = y[a]
    # nested_cv_evaluation reserves a validation partition; give it the held-out rows
    val = pd.DataFrame(X[c], columns=cols); val["target"] = y[c]
    with tempfile.NamedTemporaryFile(suffix=".db") as db:
        t = mltrack("concepts-evidence", db_name=db.name)
        t.RegisterData(train, "target", partition="train")
        t.RegisterData(val, "target", partition="validation")
        r = t.nested_cv_evaluation(estimator, inner_cv=3, outer_cv=5,
                                   train_partition="train",
                                   validation_partition="validation",
                                   repeats=repeats)
    return r


def conformal_block(alpha, n_cal, seed=0):
    X, y, a, b, c = split_data(seed)
    b = b[:n_cal]
    est = Ridge(alpha=1.0)
    out = split_conformal_regression_interval(
        est, X[a], y[a], X[b], y[b], X[c], alpha=alpha)
    diag = prediction_interval_diagnostics(
        y[c], out["lower"], out["upper"], alpha=alpha)
    return out, diag


def main():
    result = {}

    print("=" * 78)
    print("NESTED CV  —  nested_cv_evaluation, diabetes, inner=3 outer=5 repeats=5")
    print("=" * 78)
    for name, est in (("Ridge(alpha=1.0)", Ridge(alpha=1.0)),
                      ("KNeighborsRegressor(n_neighbors=9)", KNeighborsRegressor(n_neighbors=9))):
        r = nested_run(est)
        key = {"Ridge(alpha=1.0)": "ridge", "KNeighborsRegressor(n_neighbors=9)": "knn"}[name]
        row = {
            "outer_scores": [round(float(s), 4) for s in r["outer_scores"]],
            "mean_outer_score": round(float(r["mean_outer_score"]), 4),
            "final_validation_score": round(float(r["final_validation_score"]), 4),
        }
        ci = r.get("confidence_interval")
        if ci:
            row["confidence_interval"] = {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                                          for k, v in ci.items()}
        if r.get("repeated_outer_scores"):
            row["repeated_outer_scores"] = [round(float(s), 4) for s in r["repeated_outer_scores"]]
        result[key] = row
        print(f"\n{name}")
        print(f"  outer fold scores      {row['outer_scores']}")
        print(f"  mean outer score       {row['mean_outer_score']}")
        print(f"  repeated (5x) means    {row.get('repeated_outer_scores')}")
        print(f"  confidence interval    {row.get('confidence_interval')}")
        print(f"  final validation score {row['final_validation_score']}")

    print()
    print("=" * 78)
    print("SPLIT CONFORMAL — Ridge(alpha=1.0), diabetes, held-out n=82")
    print("=" * 78)
    for alpha, n_cal in ((0.10, 80), (0.20, 80), (0.05, 80), (0.01, 40), (0.01, 80)):
        out, diag = conformal_block(alpha, n_cal)
        key = f"alpha{alpha}_ncal{n_cal}"
        q = out.get("quantile", out.get("q"))
        row = {
            "alpha": alpha,
            "nominal_coverage": round(1 - alpha, 3),
            "n_calibration": n_cal,
            "quantile": (round(float(q), 4) if q is not None and np.isfinite(q) else "inf"),
            "empirical_coverage": round(float(diag["empirical_coverage"]), 4)
            if diag.get("empirical_coverage") is not None else None,
            "coverage_interval": diag.get("coverage_interval"),
            "mean_width": (round(float(diag["mean_interval_width"]), 3)
                           if diag.get("mean_interval_width") is not None else None),
            "median_width": (round(float(diag["median_interval_width"]), 3)
                             if diag.get("median_interval_width") is not None else None),
            "mean_winkler": (round(float(diag["mean_interval_score"]), 3)
                             if diag.get("mean_interval_score") is not None else None),
            "covered": f"{diag.get('covered_count')}/{diag.get('sample_size')}",
            "unbounded": diag.get("unbounded_interval_count"),
        }
        result[key] = row
        print(f"\nalpha={alpha}  n_cal={n_cal}  nominal={1-alpha:.0%}")
        print(f"  correction q           {row['quantile']}")
        print(f"  empirical coverage     {row['empirical_coverage']}  interval {row['coverage_interval']}")
        print(f"  mean / median width    {row['mean_width']} / {row['median_width']}")
        print(f"  mean Winkler           {row['mean_winkler']}")

    if "--json" in sys.argv:
        print("\nJSON:\n" + json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
