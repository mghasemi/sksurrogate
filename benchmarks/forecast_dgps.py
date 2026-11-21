"""Time-series DGPs with known structure for forecast-backtest benchmarking (plan P8.3).

Companion to ``benchmarks/dgps.py`` (which covers static tabular DGPs). Each
generator returns a *frame* ready for ``mltrack.RegisterData(..., temporal=...)``
plus the temporal spec, and :func:`evaluate` scores a lag model against the
naive baselines on identical rolling origins — the same protocol the control
plane's backtest route runs.

The point of the suite is calibration, not ranking: a forecaster that cannot
beat ``naive`` on a random walk is not a forecaster, so ``easy_naive`` exists as
a *must-lose-check* — the naive baseline should win there, and a model that
"wins" on a driftless random walk is reading noise.

Measured, not assumed: at horizon 1 the naive baseline is strong even on
``random_walk_drift``, and a lag-linear model fitted on *levels* can lose to it
(its one-step slope is estimated on a trending series and overshoots). The
report therefore records the full MAE ranking rather than presupposing that the
model wins; the drift DGP's role is that a drift-aware baseline (``drift``) must
beat ``naive`` there, which is the actual, testable statement about the data.

Design rules mirror ``benchmarks/dgps.py``:

- every draw is seeded via ``numpy.random.default_rng(seed)`` (bit-reproducible
  within a NumPy version);
- the harness is descriptive — point estimates, no significance claims;
- report records carry an explicit schema version.

Run directly::

    python -m benchmarks.forecast_dgps --dgps random_walk_drift seasonal_weekly --out DIR
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from SKSurrogate import TemporalSpec, mltrack

#: Report schema version for the JSON written by :func:`evaluate_all`.
REPORT_SCHEMA_VERSION = "sksurrogate.forecast-dgp-validation.v1"

#: Baseline methods scored on every DGP.
BASELINES = ("naive", "seasonal_naive", "drift", "mean")

HORIZON = 1
N_OBS = 240
SEED = 0
#: Rolling origins per DGP. Kept large enough that naive MASE is estimated
#: near its theoretical 1.0 — with four origins the sampling noise alone
#: (sd(|N(0,1)|) ≈ 0.6) moves MAE by ~30%.
N_SPLITS = 40


def _time_index(n: int) -> np.ndarray:
    return np.arange(n, dtype=float)


def random_walk_drift(n: int = N_OBS, *, mu: float = 0.3, sigma: float = 1.0, seed: int = SEED):
    """Random walk with positive drift: ``v_t = v_{t-1} + mu + eps_t``.

    ``naive`` is systematically biased by ``mu``; a model that learns the drift
    (or differs the series) can beat it.
    """
    rng = np.random.default_rng(seed)
    steps = mu + rng.normal(scale=sigma, size=n)
    values = np.cumsum(steps)
    frame = pd.DataFrame({"time": _time_index(n), "value": values})
    return frame, {"time_column": "time", "horizon": HORIZON}


def seasonal_weekly(n: int = N_OBS, *, slope: float = 0.05, sigma: float = 0.4, seed: int = SEED):
    """Trend + weekly seasonality (period 7) + noise."""
    rng = np.random.default_rng(seed)
    t = _time_index(n)
    values = slope * t + 2.0 * np.sin(2 * np.pi * t / 7.0) + rng.normal(scale=sigma, size=n)
    frame = pd.DataFrame({"time": t, "value": values})
    return frame, {"time_column": "time", "horizon": HORIZON}


def seasonal_daily(n: int = N_OBS * 2, *, slope: float = 0.01, sigma: float = 0.3, seed: int = SEED):
    """Trend + daily seasonality (period 24) + noise."""
    rng = np.random.default_rng(seed)
    t = _time_index(n)
    values = slope * t + 1.5 * np.sin(2 * np.pi * t / 24.0) + rng.normal(scale=sigma, size=n)
    frame = pd.DataFrame({"time": t, "value": values})
    return frame, {"time_column": "time", "horizon": HORIZON}


def panel_5series(n: int = N_OBS, *, series: int = 5, sigma: float = 0.5, seed: int = SEED):
    """Five independent AR(1) series on a shared step axis (a panel)."""
    rng = np.random.default_rng(seed)
    t = _time_index(n)
    frames = []
    for index in range(series):
        phi = 0.6 + 0.05 * index
        level = 10.0 + 2.0 * index
        values = np.empty(n)
        values[0] = level
        for step in range(1, n):
            values[step] = level + phi * (values[step - 1] - level) + rng.normal(scale=sigma)
        frames.append(pd.DataFrame({"series": f"s{index}", "time": t, "value": values}))
    frame = pd.concat(frames, ignore_index=True)
    return frame, {"time_column": "time", "series_id": "series", "horizon": HORIZON}


def easy_naive(n: int = N_OBS, *, sigma: float = 1.0, seed: int = SEED):
    """Driftless random walk — ``naive`` is the Bayes-optimal point forecast.

    A model that reports a lower MAE here than ``naive`` is fitting noise, which
    is exactly why this DGP is part of the suite.
    """
    rng = np.random.default_rng(seed)
    values = np.cumsum(rng.normal(scale=sigma, size=n))
    frame = pd.DataFrame({"time": _time_index(n), "value": values})
    return frame, {"time_column": "time", "horizon": HORIZON}


GENERATORS: Mapping[str, Callable[..., tuple[pd.DataFrame, dict]]] = {
    "random_walk_drift": random_walk_drift,
    "seasonal_weekly": seasonal_weekly,
    "seasonal_daily": seasonal_daily,
    "panel_5series": panel_5series,
    "easy_naive": easy_naive,
}

KNOWN_DGPS: tuple[str, ...] = tuple(GENERATORS)


def _lag_design_for(spec: Mapping[str, Any], horizon: int) -> dict:
    """Feature kwargs a lag model needs: lags at least as long as the horizon."""
    return {"lags": [horizon, 2 * horizon]}


def evaluate(
    name: str,
    *,
    horizon: int = HORIZON,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    **generator_kwargs: Any,
) -> dict:
    """Score a lag-:math:`h` linear model against the naive baselines on one DGP.

    Returns a JSON-safe record with pooled metrics per method and a
    ``naive_wins`` verdict (naive MAE strictly below the model's).
    """
    if name not in GENERATORS:
        raise ValueError(
            "unknown forecast DGP %r; known DGPs are: %s" % (name, ", ".join(KNOWN_DGPS))
        )
    frame, spec = GENERATORS[name](seed=seed, **generator_kwargs)
    spec = dict(spec)
    spec["horizon"] = horizon

    with tempfile.TemporaryDirectory(prefix="forecast-dgp-") as tmp:
        tracker = mltrack("forecast-dgp", db_name=str(Path(tmp) / "task.db"))
        try:
            tracker.RegisterData(
                frame, "value", partition="train", temporal=TemporalSpec.from_dict(spec)
            )
            result = tracker.backtest(
                LinearRegression(),
                horizon=horizon,
                n_splits=n_splits,
                step=horizon,
                feature_kwargs=_lag_design_for(spec, horizon),
                baselines=list(BASELINES),
                seasonality=7 if name == "seasonal_weekly" else 1,
            )
        finally:
            tracker.close()

    pooled = result["pooled"]
    method_metrics = {
        "model": {"mae": pooled.get("mae"), "rmse": pooled.get("rmse"), "mase": pooled.get("mase")},
    }
    for method, metrics in (result.get("baselines") or {}).items():
        method_metrics[method] = {
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "mase": metrics.get("mase"),
        }

    naive_mae = (method_metrics.get("naive") or {}).get("mae")
    model_mae = method_metrics["model"]["mae"]
    naive_wins = (
        naive_mae is not None
        and model_mae is not None
        and float(naive_mae) < float(model_mae)
    )
    scored: dict[str, float] = {}
    for method_name, metrics in method_metrics.items():
        value = metrics.get("mae")
        if value is not None:
            scored[method_name] = float(value)
    best_method = min(scored, key=lambda m: scored[m]) if scored else None
    method_mae_ranking = sorted(scored, key=lambda m: scored[m])
    return {
        "dgp": name,
        "horizon": int(horizon),
        "rows": int(len(frame)),
        "series": sorted(str(value) for value in frame["series"].unique())
        if "series" in frame.columns
        else None,
        "origins": int(len(result.get("origins") or [])),
        "methods": method_metrics,
        "naive_wins": bool(naive_wins),
        "best_method": best_method,
        "method_mae_ranking": method_mae_ranking,
        "spec": spec,
    }


def evaluate_all(
    dgps: Sequence[str] = KNOWN_DGPS,
    *,
    horizon: int = HORIZON,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    out_dir: str | Path | None = None,
) -> dict:
    """Run every requested DGP and write one versioned JSON record.

    The ``easy_naive`` verdict is reported as ``expectations.easy_naive_naive_wins``
    so a regression that breaks the check is visible in the artifact, not only in
    the test suite.
    """
    records = {
        name: evaluate(name, horizon=horizon, n_splits=n_splits, seed=seed) for name in dgps
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"horizon": int(horizon), "n_splits": int(n_splits), "seed": int(seed)},
        "results": records,
        "expectations": {
            "easy_naive_naive_wins": bool(records.get("easy_naive", {}).get("naive_wins", False))
            if "easy_naive" in records
            else None,
        },
    }
    if out_dir is not None:
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "forecast_dgp_validation.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a lag model against naive baselines on known forecast DGPs (plan P8.3)."
    )
    parser.add_argument("--dgps", nargs="+", choices=list(KNOWN_DGPS), default=list(KNOWN_DGPS))
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument(
        "--out",
        default=str(Path("benchmarks/results/forecast-dgps") / stamp),
        help="output directory for forecast_dgp_validation.json",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = evaluate_all(
        dgps=args.dgps,
        horizon=args.horizon,
        n_splits=args.n_splits,
        seed=args.seed,
        out_dir=args.out,
    )

    print(f"{'dgp':<20} {'model mae':>10} {'naive mae':>10} {'model mase':>11} {'naive wins':>11}")
    for name, record in report["results"].items():
        model = record["methods"]["model"]
        naive = record["methods"].get("naive", {})
        print(
            f"{name:<20} {_fmt(model.get('mae')):>10} {_fmt(naive.get('mae')):>10} "
            f"{_fmt(model.get('mase')):>11} {str(record['naive_wins']):>11}"
        )
    verdict = report["expectations"]["easy_naive_naive_wins"]
    if verdict is not None:
        print(f"easy_naive: naive baseline wins → {verdict}")
    print(f"report: {report.get('report_path')}")
    return 0


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
