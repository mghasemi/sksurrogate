"""Reproducible probe of surrogate regression with and without observation noise.

Run with ``python -m benchmarks.dtrack.noise_gp_probe``. This small synthetic
comparison is a diagnostic, not sufficient evidence to change search defaults.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from .objectives import make_objective, sample_domain, test_grid


def run_probe(
    seeds=range(10),
    dimensions=(1, 2, 3),
    train_size=40,
    test_size=400,
    noise_levels=(0.0, 0.3),
):
    """Return repeated RMSE measurements on smooth trigonometric surfaces."""
    results = {}
    for dimension in dimensions:
        objective, _ = make_objective("trig", dimension)
        x_test = test_grid(test_size, dimension, np.random.default_rng(12345))
        y_test = objective(x_test)
        dimension_results = results[dimension] = {
            noise_std: {"rbf_gp": [], "noise_aware_gp": [], "random_forest": []}
            for noise_std in noise_levels
        }

        for seed in seeds:
            rng = np.random.default_rng(seed)
            x_train = sample_domain(train_size, dimension, rng)
            y_train = objective(x_train)
            noise = rng.standard_normal(train_size)
            for noise_std, model_results in dimension_results.items():
                observed_y = y_train + noise_std * noise
                estimators = {
                    "rbf_gp": GaussianProcessRegressor(random_state=seed),
                    "noise_aware_gp": GaussianProcessRegressor(
                        kernel=RBF() + WhiteKernel(), random_state=seed
                    ),
                    "random_forest": RandomForestRegressor(
                        n_estimators=100, random_state=seed
                    ),
                }
                for name, estimator in estimators.items():
                    estimator.fit(x_train, observed_y)
                    predictions = estimator.predict(x_test)
                    rmse = float(np.sqrt(np.mean((predictions - y_test) ** 2)))
                    model_results[name].append(rmse)

    return results


def main():
    results = run_probe()
    print("Surrogate RMSE against noise-free trigonometric surfaces")
    for dimension, dimension_results in results.items():
        for noise_std, model_results in dimension_results.items():
            print("\nDimension: %d; training observation-noise SD: %.1f" % (dimension, noise_std))
            print("| surrogate | mean RMSE | seed SD |")
            print("|---|---:|---:|")
            for name, values in model_results.items():
                print("| %s | %.4f | %.4f |" % (name, np.mean(values), np.std(values)))


if __name__ == "__main__":
    main()
