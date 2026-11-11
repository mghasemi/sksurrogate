"""Compare LHS initialization and GP-LCB suggestions with random sampling.

Run a small reproducible smoke probe with::

    python -m benchmarks.doe_active_learning --smoke

The objective calls in this benchmark are synthetic and local. The library's
candidate proposal helper never calls an objective itself.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from SKSurrogate import latin_hypercube, suggest_next_point
from benchmarks.dtrack.objectives import make_objective, test_grid


_FAMILIES = ("trig", "mixed", "rational")
_BOUNDS = [(-1.0, 1.0), (-1.0, 1.0)]


def _surrogate():
    return GaussianProcessRegressor(
        kernel=Matern(
            length_scale=np.ones(len(_BOUNDS)),
            length_scale_bounds="fixed",
            nu=2.5,
        ),
        alpha=1.0e-8,
        normalize_y=True,
        optimizer=None,
        random_state=0,
    )


def _best_curve(values):
    return np.minimum.accumulate(np.asarray(values, dtype=float))


def run_diabetes_benchmark(*, seeds=5, budget=12, initial=4, n_candidates=128):
    """Compare equal-budget policies tuning Ridge alpha on fixed-fold diabetes CV."""
    for name, value, minimum, maximum in (
        ("seeds", seeds, 1, 100),
        ("budget", budget, 4, 100),
        ("initial", initial, 2, 100),
        ("n_candidates", n_candidates, 16, 10000),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError("%s must be an integer between %d and %d" % (name, minimum, maximum))
    if initial >= budget:
        raise ValueError("initial must be smaller than budget")

    data = load_diabetes()
    folds = KFold(n_splits=5, shuffle=True, random_state=2026)
    bounds = [(-4.0, 4.0)]

    def objective(points):
        scores = []
        for (log_alpha,) in points:
            estimator = make_pipeline(
                StandardScaler(), Ridge(alpha=10.0 ** float(log_alpha))
            )
            score = cross_val_score(
                estimator,
                data.data,
                data.target,
                cv=folds,
                scoring="neg_mean_squared_error",
            )
            scores.append(float(-np.mean(score)))
        return np.asarray(scores)

    reference_grid = np.linspace(bounds[0][0], bounds[0][1], 301).reshape(-1, 1)
    reference_optimum = float(np.min(objective(reference_grid)))
    records = []
    for seed in range(seeds):
        active_samples = latin_hypercube(bounds, initial, random_state=seed)
        active_values = objective(active_samples)
        random_samples = np.random.default_rng(seed).uniform(
            bounds[0][0], bounds[0][1], size=(initial, 1)
        )
        random_values = objective(random_samples)
        active_best = list(active_values)
        random_best = list(random_values)

        for step in range(initial, budget):
            proposal = suggest_next_point(
                active_samples,
                active_values,
                bounds,
                n_candidates=n_candidates,
                random_state=seed * 1009 + step,
            )
            point = proposal["candidate"].reshape(1, -1)
            value = objective(point)[0]
            active_samples = np.vstack([active_samples, point])
            active_values = np.append(active_values, value)
            active_best.append(value)

            point = np.random.default_rng(seed * 1009 + step).uniform(
                bounds[0][0], bounds[0][1], size=(1, 1)
            )
            value = objective(point)[0]
            random_samples = np.vstack([random_samples, point])
            random_values = np.append(random_values, value)
            random_best.append(value)

        records.append(
            {
                "seed": seed,
                "budget": budget,
                "active_final_regret": float(min(active_best) - reference_optimum),
                "random_final_regret": float(min(random_best) - reference_optimum),
            }
        )
    return {
        "protocol": {
            "dataset": "scikit-learn diabetes regression dataset",
            "objective": "five-fold shuffled-CV mean squared error for standardized Ridge",
            "fold_seed": 2026,
            "tuned_parameter": "log10(Ridge alpha)",
            "bounds": bounds,
            "seeds": seeds,
            "total_evaluation_budget": budget,
            "initial_design_size": initial,
            "candidate_pool_size": n_candidates,
            "reference_optimum": "minimum over a shared 301-point log-alpha grid",
        },
        "summary": {
            "runs": len(records),
            "active_final_regret_wins": int(
                sum(r["active_final_regret"] < r["random_final_regret"] for r in records)
            ),
            "mean_active_final_regret": float(
                np.mean([r["active_final_regret"] for r in records])
            ),
            "mean_random_final_regret": float(
                np.mean([r["random_final_regret"] for r in records])
            ),
        },
        "records": records,
        "interpretation": (
            "A single real-data model-tuning task, not a physical experiment "
            "or evidence of general superiority."
        ),
    }


def run_breast_cancer_benchmark(*, seeds=5, budget=12, initial=4, n_candidates=128):
    """Compare equal-budget policies tuning logistic-regression C on fixed-fold CV."""
    for name, value, minimum, maximum in (
        ("seeds", seeds, 1, 100),
        ("budget", budget, 4, 100),
        ("initial", initial, 2, 100),
        ("n_candidates", n_candidates, 16, 10000),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError("%s must be an integer between %d and %d" % (name, minimum, maximum))
    if initial >= budget:
        raise ValueError("initial must be smaller than budget")

    data = load_breast_cancer()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    bounds = [(-3.0, 3.0)]

    def objective(points):
        scores = []
        for (log_c,) in points:
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=10.0 ** float(log_c),
                    solver="liblinear",
                    max_iter=1000,
                ),
            )
            score = cross_val_score(
                estimator,
                data.data,
                data.target,
                cv=folds,
                scoring="neg_log_loss",
            )
            scores.append(float(-np.mean(score)))
        return np.asarray(scores)

    reference_grid = np.linspace(bounds[0][0], bounds[0][1], 121).reshape(-1, 1)
    reference_value = float(np.min(objective(reference_grid)))
    records = []
    for seed in range(seeds):
        active_samples = latin_hypercube(bounds, initial, random_state=seed)
        active_values = objective(active_samples)
        random_samples = np.random.default_rng(seed).uniform(
            bounds[0][0], bounds[0][1], size=(initial, 1)
        )
        random_values = objective(random_samples)
        active_best = list(active_values)
        random_best = list(random_values)

        for step in range(initial, budget):
            proposal = suggest_next_point(
                active_samples,
                active_values,
                bounds,
                n_candidates=n_candidates,
                random_state=seed * 1009 + step,
            )
            point = proposal["candidate"].reshape(1, -1)
            value = objective(point)[0]
            active_samples = np.vstack([active_samples, point])
            active_values = np.append(active_values, value)
            active_best.append(value)

            point = np.random.default_rng(seed * 1009 + step).uniform(
                bounds[0][0], bounds[0][1], size=(1, 1)
            )
            value = objective(point)[0]
            random_samples = np.vstack([random_samples, point])
            random_values = np.append(random_values, value)
            random_best.append(value)

        records.append(
            {
                "seed": seed,
                "budget": budget,
                "active_final_grid_delta": float(min(active_best) - reference_value),
                "random_final_grid_delta": float(min(random_best) - reference_value),
            }
        )
    return {
        "protocol": {
            "dataset": "scikit-learn breast cancer classification dataset",
            "objective": "five-fold stratified-CV log loss for standardized logistic regression",
            "fold_seed": 2026,
            "tuned_parameter": "log10(LogisticRegression C)",
            "bounds": bounds,
            "seeds": seeds,
            "total_evaluation_budget": budget,
            "initial_design_size": initial,
            "candidate_pool_size": n_candidates,
            "reference_value": "minimum over a shared 121-point log-C grid",
        },
        "summary": {
            "runs": len(records),
            "active_final_grid_delta_wins": int(
                sum(
                    r["active_final_grid_delta"] < r["random_final_grid_delta"]
                    for r in records
                )
            ),
            "mean_active_final_grid_delta": float(
                np.mean([r["active_final_grid_delta"] for r in records])
            ),
            "mean_random_final_grid_delta": float(
                np.mean([r["random_final_grid_delta"] for r in records])
            ),
        },
        "records": records,
        "interpretation": (
            "A single fixed-fold classification tuning task. The grid reference "
            "is approximate; negative deltas mean a policy found a lower value."
        ),
    }


def run_benchmark(*, seeds=10, budget=20, initial=6, n_candidates=256):
    """Return paired design and optimization results on three 2-D objectives."""
    for name, value, minimum, maximum in (
        ("seeds", seeds, 1, 100),
        ("budget", budget, 4, 100),
        ("initial", initial, 2, 100),
        ("n_candidates", n_candidates, 16, 10000),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError("%s must be an integer between %d and %d" % (name, minimum, maximum))
    if initial >= budget:
        raise ValueError("initial must be smaller than budget")

    records = []
    test_points = test_grid(4096, 2, np.random.default_rng(2026))
    for family in _FAMILIES:
        objective, _ = make_objective(family, 2)
        test_values = objective(test_points)
        for seed in range(seeds):
            lhs_initial = latin_hypercube(_BOUNDS, initial, random_state=seed)
            random_initial = np.random.default_rng(seed).uniform(
                -1.0, 1.0, size=(initial, 2)
            )
            lhs_values = objective(lhs_initial)
            random_values = objective(random_initial)

            lhs_model = _surrogate().fit(lhs_initial, lhs_values)
            random_model = _surrogate().fit(random_initial, random_values)
            lhs_rmse = float(
                np.sqrt(np.mean((lhs_model.predict(test_points) - test_values) ** 2))
            )
            random_rmse = float(
                np.sqrt(
                    np.mean((random_model.predict(test_points) - test_values) ** 2)
                )
            )

            active_samples = lhs_initial.copy()
            active_values = lhs_values.copy()
            random_objective_values = list(random_values)
            active_best = [float(value) for value in active_values]
            random_best = [float(value) for value in random_objective_values]
            for step in range(initial, budget):
                proposal = suggest_next_point(
                    active_samples,
                    active_values,
                    _BOUNDS,
                    n_candidates=n_candidates,
                    random_state=seed * 1009 + step,
                )
                point = proposal["candidate"]
                value = float(objective(point.reshape(1, -1))[0])
                active_samples = np.vstack([active_samples, point])
                active_values = np.append(active_values, value)
                active_best.append(value)

                random_point = np.random.default_rng(seed * 1009 + step).uniform(
                    -1.0, 1.0, size=2
                )
                random_value = float(objective(random_point.reshape(1, -1))[0])
                random_objective_values.append(random_value)
                random_best.append(random_value)

            active_curve = _best_curve(active_best)
            random_curve = _best_curve(random_best)
            optimum = float(np.min(test_values))
            records.append(
                {
                    "family": family,
                    "seed": seed,
                    "budget": budget,
                    "initial": initial,
                    "lhs_initial_rmse": lhs_rmse,
                    "random_initial_rmse": random_rmse,
                    "active_final_regret": float(active_curve[-1] - optimum),
                    "random_final_regret": float(random_curve[-1] - optimum),
                    "active_regret_auc": float(np.mean(active_curve - optimum)),
                    "random_regret_auc": float(np.mean(random_curve - optimum)),
                }
            )

    summaries = []
    for family in _FAMILIES:
        rows = [row for row in records if row["family"] == family]
        summaries.append(
            {
                "family": family,
                "runs": len(rows),
                "mean_initial_rmse_lhs": float(
                    np.mean([row["lhs_initial_rmse"] for row in rows])
                ),
                "mean_initial_rmse_random": float(
                    np.mean([row["random_initial_rmse"] for row in rows])
                ),
                "active_final_regret_wins": int(
                    sum(
                        row["active_final_regret"] < row["random_final_regret"]
                        for row in rows
                    )
                ),
                "mean_active_final_regret": float(
                    np.mean([row["active_final_regret"] for row in rows])
                ),
                "mean_random_final_regret": float(
                    np.mean([row["random_final_regret"] for row in rows])
                ),
                "mean_active_regret_auc": float(
                    np.mean([row["active_regret_auc"] for row in rows])
                ),
                "mean_random_regret_auc": float(
                    np.mean([row["random_regret_auc"] for row in rows])
                ),
            }
        )
    return {
        "protocol": {
            "families": list(_FAMILIES),
            "dimension": 2,
            "bounds": _BOUNDS,
            "seeds": seeds,
            "total_evaluation_budget": budget,
            "initial_design_size": initial,
            "candidate_pool_size": n_candidates,
            "test_grid_size": len(test_points),
            "active_policy": "Latin-hypercube initialization then GP lower-confidence bound",
            "baseline": "uniform random sampling at the same objective-evaluation budget",
            "known_optimum": "minimum over a shared 4096-point test grid",
        },
        "summary": summaries,
        "records": records,
        "interpretation": (
            "Synthetic benchmark evidence only; it does not establish value on "
            "physical experiments or domain-constrained scientific workloads."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run a small two-seed probe")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--initial", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=256)
    args = parser.parse_args(argv)
    synthetic = run_benchmark(
        seeds=2 if args.smoke else args.seeds,
        budget=10 if args.smoke else args.budget,
        initial=4 if args.smoke else args.initial,
        n_candidates=64 if args.smoke else args.candidates,
    )
    diabetes = run_diabetes_benchmark(
        seeds=2 if args.smoke else 5,
        budget=8 if args.smoke else 12,
        initial=3 if args.smoke else 4,
        n_candidates=64 if args.smoke else 128,
    )
    breast_cancer = run_breast_cancer_benchmark(
        seeds=2 if args.smoke else 5,
        budget=8 if args.smoke else 12,
        initial=3 if args.smoke else 4,
        n_candidates=64 if args.smoke else 128,
    )
    result = {
        "synthetic": synthetic,
        "diabetes_model_tuning": diabetes,
        "breast_cancer_model_tuning": breast_cancer,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
