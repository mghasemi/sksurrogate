"""D4 readiness assessment: acquisition-strategy comparison for sequential design.

This is an **analysis harness**, not a production acquisition layer. It compares
the shipped Gaussian-process LCB policy (``SKSurrogate.design.suggest_next_point``)
against expected improvement (EI), probability of improvement (PI), and a
noise-aware LCB variant on the synthetic objectives used by the existing
active-learning benchmark.

Design rules (see ``AI_LAYER02_D4_READINESS_ASSESSMENT.md``):

- **Equal evaluation budgets.** Every strategy starts from the same seeded Latin
  hypercube design and receives the same number of sequential objective
  evaluations per family/seed/noise setting.
- **Paired seeds.** Candidate-pool seeds and observation-noise draws are derived
  from the same run seed, so differences are not confounded by sampling.
- **Descriptive only.** Summaries are means/medians of regret across a fixed set
  of seeds on synthetic functions; no significance or generalization claim is
  made, and no real experiment is executed.
- **The shipped policy is exercised, not re-implemented.** ``lcb`` calls
  ``suggest_next_point`` directly; the alternative acquisitions use the same
  kernel and candidate-pool construction so the comparison isolates the
  acquisition rule (and, for ``lcb_noise_aware``, the noise model).

Run directly::

    python -m benchmarks.design_readiness --out DIR
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from SKSurrogate import latin_hypercube, suggest_next_point
from benchmarks.dtrack.objectives import make_objective, test_grid

#: Report schema version for the readiness JSON.
REPORT_SCHEMA_VERSION = "sksurrogate.design-readiness.v1"

#: Objectives from the existing active-learning benchmark (2-D, domain [-1,1]^2).
DEFAULT_FAMILIES: tuple[str, ...] = ("trig", "mixed", "rational")
BOUNDS: list[list[float]] = [[-1.0, 1.0], [-1.0, 1.0]]
DIMENSION = 2

#: Acquisition policies under test. ``lcb`` is the shipped implementation.
DEFAULT_STRATEGIES: tuple[str, ...] = (
    "random",
    "lcb",
    "ei",
    "pi",
    "lcb_noise_aware",
)

_LENGTH_SCALE = 1.0
_BETA = 2.0
_SD_FLOOR = 1e-12


def _candidate_pool(seed: int, n_candidates: int, observed: np.ndarray) -> np.ndarray:
    """Seeded Latin-hypercube candidates, excluding coordinates already observed.

    Mirrors ``SKSurrogate.design.suggest_next_point`` so EI/PI see the same pool
    the shipped LCB policy sees for the same step seed.
    """
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=DIMENSION, seed=int(seed))
    unit = sampler.random(n=int(n_candidates))
    points = qmc.scale(unit, np.asarray(BOUNDS)[:, 0], np.asarray(BOUNDS)[:, 1])
    observed_unit = (observed - np.asarray(BOUNDS)[:, 0]) / (
        np.asarray(BOUNDS)[:, 1] - np.asarray(BOUNDS)[:, 0]
    )
    distinct = ~np.any(
        np.all(
            np.isclose(
                unit[:, np.newaxis, :],
                observed_unit[np.newaxis, :, :],
                rtol=0.0,
                atol=1e-12,
            ),
            axis=2,
        ),
        axis=1,
    )
    return points[distinct]


def _fit_surrogate(
    samples: np.ndarray,
    values: np.ndarray,
    *,
    noise_aware: bool,
    seed: int,
) -> GaussianProcessRegressor:
    """Fit the same Matern-5/2 surrogate as the shipped policy, optionally noise-aware."""
    kernel = Matern(
        length_scale=np.full(DIMENSION, _LENGTH_SCALE),
        length_scale_bounds="fixed",
        nu=2.5,
    )
    if noise_aware:
        kernel = kernel + WhiteKernel(
            noise_level=1e-4,
            noise_level_bounds=(1e-8, 1e-1),
        )
        optimizer: str | None = "fmin_l_bfgs_b"
    else:
        optimizer = None
    unit = (samples - np.asarray(BOUNDS)[:, 0]) / (
        np.asarray(BOUNDS)[:, 1] - np.asarray(BOUNDS)[:, 0]
    )
    surrogate = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        optimizer=optimizer,
        n_restarts_optimizer=0,
    )
    return surrogate.fit(unit, values)


def propose(
    strategy: str,
    samples: np.ndarray,
    values: np.ndarray,
    *,
    n_candidates: int,
    step_seed: int,
    fit_seed: int,
) -> tuple[np.ndarray, dict]:
    """Propose one point under ``strategy``.

    ``lcb`` delegates to the shipped ``suggest_next_point`` (including its beta
    and metadata). ``random`` draws a uniform point from the same step seed.
    ``ei``/``pi``/``lcb_noise_aware`` rank the same candidate pool with the same
    surrogate family.
    """
    if strategy == "random":
        point = np.random.default_rng(step_seed).uniform(-1.0, 1.0, size=DIMENSION)
        return point, {"acquisition": "uniform_random"}

    if strategy == "lcb":
        result = suggest_next_point(
            samples,
            values,
            BOUNDS,
            n_candidates=n_candidates,
            beta=_BETA,
            random_state=step_seed,
        )
        return np.asarray(result["candidate"], dtype=float), {
            "acquisition": result["acquisition"],
            "acquisition_value": result["acquisition_value"],
            "predicted_mean": result["predicted_mean"],
            "predicted_std": result["predicted_std"],
        }

    noise_aware = strategy == "lcb_noise_aware"
    candidates = _candidate_pool(step_seed, n_candidates, samples)
    if len(candidates) == 0:
        raise RuntimeError("candidate pool contains no points distinct from observations")
    surrogate = _fit_surrogate(samples, values, noise_aware=noise_aware, seed=fit_seed)
    unit = (candidates - np.asarray(BOUNDS)[:, 0]) / (
        np.asarray(BOUNDS)[:, 1] - np.asarray(BOUNDS)[:, 0]
    )
    mean, std = surrogate.predict(unit, return_std=True)
    std = np.maximum(std, _SD_FLOOR)

    if strategy in ("lcb", "lcb_noise_aware"):
        acquisition = mean - _BETA * std
        index = int(np.argmin(acquisition))
        label = "lower_confidence_bound" if strategy == "lcb" else "lower_confidence_bound_noise_aware"
    else:
        best = float(np.min(values))
        improvement = best - mean
        z = improvement / std
        if strategy == "ei":
            acquisition = improvement * norm.cdf(z) + std * norm.pdf(z)
            index = int(np.argmax(acquisition))
            label = "expected_improvement"
        elif strategy == "pi":
            acquisition = norm.cdf(z)
            index = int(np.argmax(acquisition))
            label = "probability_of_improvement"
        else:  # pragma: no cover - guarded by the strategy whitelist
            raise ValueError("unknown acquisition strategy %r" % strategy)

    return candidates[index].copy(), {
        "acquisition": label,
        "acquisition_value": float(acquisition[index]),
        "predicted_mean": float(mean[index]),
        "predicted_std": float(std[index]),
    }


def _best_curve(values: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(np.asarray(values, dtype=float))


def run_assessment(
    *,
    families: Sequence[str] = DEFAULT_FAMILIES,
    strategies: Sequence[str] = DEFAULT_STRATEGIES,
    seeds: int = 5,
    budget: int = 20,
    initial: int = 6,
    n_candidates: int = 256,
    noise_levels: Sequence[float] = (0.0, 0.05),
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the paired equal-budget comparison and return a versioned report."""
    unknown = sorted(set(strategies) - set(DEFAULT_STRATEGIES))
    if unknown:
        raise ValueError("unknown strategy name(s): %s" % ", ".join(unknown))
    if not strategies:
        raise ValueError("at least one strategy is required")
    for name, value, minimum, maximum in (
        ("seeds", seeds, 1, 50),
        ("budget", budget, 4, 60),
        ("initial", initial, 2, 60),
        ("n_candidates", n_candidates, 16, 10000),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError("%s must be an integer between %d and %d" % (name, minimum, maximum))
    if initial >= budget:
        raise ValueError("initial must be smaller than budget")
    for level in noise_levels:
        if not np.isfinite(level) or level < 0.0:
            raise ValueError("noise levels must be finite and non-negative")

    started = time.time()
    grid = test_grid(4096, DIMENSION, np.random.default_rng(2026))
    records: list[dict] = []
    run_summaries: list[dict] = []

    for family in families:
        objective, _ = make_objective(family, DIMENSION)
        grid_values = objective(grid)
        optimum = float(np.min(grid_values))
        for noise_level in noise_levels:
            for seed in range(seeds):
                initial_points = latin_hypercube(BOUNDS, initial, random_state=seed)
                initial_true = np.asarray(objective(initial_points), dtype=float)
                for strategy in strategies:
                    noise_rng = np.random.default_rng(seed * 7919 + 17)
                    samples = initial_points.copy()
                    true_values = initial_true.copy()
                    observed_values = (
                        true_values
                        if noise_level == 0.0
                        else true_values + noise_rng.normal(0.0, noise_level, size=len(true_values))
                    )
                    for iteration in range(initial):
                        records.append(
                            {
                                "family": family,
                                "noise_level": float(noise_level),
                                "seed": seed,
                                "strategy": strategy,
                                "iteration": iteration,
                                "phase": "initial",
                                "x": initial_points[iteration].round(12).tolist(),
                                "y_true": float(true_values[iteration]),
                                "y_observed": float(observed_values[iteration]),
                            }
                        )

                    for iteration in range(initial, budget):
                        step_seed = seed * 1009 + iteration
                        point, metadata = propose(
                            strategy,
                            samples,
                            observed_values,
                            n_candidates=n_candidates,
                            step_seed=step_seed,
                            fit_seed=step_seed,
                        )
                        true_value = float(objective(point.reshape(1, -1))[0])
                        observed = (
                            true_value
                            if noise_level == 0.0
                            else true_value + float(noise_rng.normal(0.0, noise_level))
                        )
                        samples = np.vstack([samples, point.reshape(1, -1)])
                        true_values = np.append(true_values, true_value)
                        observed_values = np.append(observed_values, observed)
                        records.append(
                            {
                                "family": family,
                                "noise_level": float(noise_level),
                                "seed": seed,
                                "strategy": strategy,
                                "iteration": iteration,
                                "phase": "sequential",
                                "x": point.round(12).tolist(),
                                "y_true": true_value,
                                "y_observed": observed,
                                **metadata,
                            }
                        )

                    curve = _best_curve(true_values)
                    run_summaries.append(
                        {
                            "family": family,
                            "noise_level": float(noise_level),
                            "seed": seed,
                            "strategy": strategy,
                            "budget": budget,
                            "initial": initial,
                            "optimum": optimum,
                            "final_regret": float(curve[-1] - optimum),
                            "regret_auc": float(np.mean(curve - optimum)),
                            "best_observed": float(np.min(observed_values)),
                        }
                    )

    summaries: list[dict] = []
    groups = sorted(
        {(row["strategy"], row["family"], row["noise_level"]) for row in run_summaries}
    )
    for strategy, family, noise_level in groups:
        rows = [
            row
            for row in run_summaries
            if row["strategy"] == strategy
            and row["family"] == family
            and row["noise_level"] == noise_level
        ]
        summaries.append(
            {
                "strategy": strategy,
                "family": family,
                "noise_level": noise_level,
                "runs": len(rows),
                "mean_final_regret": float(np.mean([row["final_regret"] for row in rows])),
                "median_final_regret": float(np.median([row["final_regret"] for row in rows])),
                "mean_regret_auc": float(np.mean([row["regret_auc"] for row in rows])),
            }
        )

    paired: list[dict] = []
    for family in families if "lcb" in strategies else ():
        for noise_level in noise_levels:
            for strategy in strategies:
                if strategy == "lcb":
                    continue
                wins = losses = ties = 0
                for seed in range(seeds):
                    candidate = next(
                        row
                        for row in run_summaries
                        if row["strategy"] == strategy
                        and row["family"] == family
                        and row["noise_level"] == noise_level
                        and row["seed"] == seed
                    )
                    baseline = next(
                        row
                        for row in run_summaries
                        if row["strategy"] == "lcb"
                        and row["family"] == family
                        and row["noise_level"] == noise_level
                        and row["seed"] == seed
                    )
                    if candidate["final_regret"] < baseline["final_regret"]:
                        wins += 1
                    elif candidate["final_regret"] > baseline["final_regret"]:
                        losses += 1
                    else:
                        ties += 1
                paired.append(
                    {
                        "strategy": strategy,
                        "baseline": "lcb",
                        "family": family,
                        "noise_level": noise_level,
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                    }
                )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "families": list(families),
            "strategies": list(strategies),
            "budget": budget,
            "initial": initial,
            "n_candidates": n_candidates,
            "seeds": seeds,
            "noise_levels": [float(level) for level in noise_levels],
            "bounds": BOUNDS,
            "dimension": DIMENSION,
            "baseline_acquisition": "lower_confidence_bound (shipped suggest_next_point, beta=2)",
            "surrogate": "Matern(nu=2.5), fixed unit length scale, normalize_y",
            "noise_aware_surrogate": "Matern(nu=2.5) + WhiteKernel(noise optimized)",
            "regret": "best true objective evaluated minus dense-grid optimum",
            "observation_noise": "Gaussian with the stated standard deviation, added to reported y",
        },
        "run_summaries": run_summaries,
        "summaries": summaries,
        "paired_vs_lcb": paired,
        "records": records,
        "wall_seconds": round(time.time() - started, 2),
        "note": (
            "Synthetic readiness evidence only: fixed objectives, fixed surrogate "
            "family, synthetic Gaussian noise, and no real experimental cost model. "
            "No significance or generalization claim is made, and no production "
            "acquisition code ships from this harness."
        ),
    }

    if out_dir is not None:
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "design_readiness.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--initial", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.05])
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    report = run_assessment(
        families=tuple(args.families),
        strategies=tuple(args.strategies),
        seeds=args.seeds,
        budget=args.budget,
        initial=args.initial,
        n_candidates=args.candidates,
        noise_levels=tuple(args.noise),
        out_dir=args.out,
    )
    print(
        "%-18s %-9s %-6s %10s %10s"
        % ("strategy", "family", "noise", "mean regret", "median")
    )
    for row in report["summaries"]:
        print(
            "%-18s %-9s %-6.3g %10.4f %10.4f"
            % (
                row["strategy"],
                row["family"],
                row["noise_level"],
                row["mean_final_regret"],
                row["median_final_regret"],
            )
        )
    if args.out:
        print("report written to:", Path(args.out) / "design_readiness.json")
    else:
        print("wall seconds:", report["wall_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
