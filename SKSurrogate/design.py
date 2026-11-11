"""Bounded candidate generation for sequential experimental design."""

import numpy as np


_MAX_DESIGN_ROWS = 10000
_MAX_SURROGATE_OBSERVATIONS = 500
_MAX_DESIGN_DIMENSIONS = 100
_GP_LENGTH_SCALE = 1.0


def _validated_bounds(bounds):
    """Return finite, strictly increasing numeric bounds as a float array."""
    try:
        values = np.asarray(bounds, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("bounds must be a sequence of finite (lower, upper) pairs") from exc
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or values.shape[0] < 1
        or values.shape[0] > _MAX_DESIGN_DIMENSIONS
    ):
        raise ValueError(
            "bounds must contain between 1 and %d (lower, upper) pairs"
            % _MAX_DESIGN_DIMENSIONS
        )
    if not np.all(np.isfinite(values)) or np.any(values[:, 0] >= values[:, 1]):
        raise ValueError("each bound must be finite and have lower < upper")
    return values


def _validated_seed(random_state):
    if random_state is not None and (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, (int, np.integer))
    ):
        raise ValueError("random_state must be an integer or None")
    return int(random_state) if random_state is not None else None


def latin_hypercube(bounds, n_samples, *, random_state=0):
    """Generate a seeded Latin-hypercube design over continuous numeric bounds.

    The returned array has shape ``(n_samples, n_dimensions)``. Every
    dimension contains exactly one point in each of its equal-width strata.
    This function only proposes candidates; it does not evaluate an objective
    or execute physical experiments.
    """
    from scipy.stats import qmc

    limits = _validated_bounds(bounds)
    if (
        isinstance(n_samples, (bool, np.bool_))
        or not isinstance(n_samples, (int, np.integer))
        or n_samples < 1
        or n_samples > _MAX_DESIGN_ROWS
    ):
        raise ValueError("n_samples must be an integer between 1 and %d" % _MAX_DESIGN_ROWS)
    seed = _validated_seed(random_state)
    sampler = qmc.LatinHypercube(d=len(limits), seed=seed)
    unit_design = sampler.random(n=int(n_samples))
    return qmc.scale(unit_design, limits[:, 0], limits[:, 1])


def suggest_next_point(
    samples,
    values,
    bounds,
    *,
    n_candidates=1000,
    beta=2.0,
    random_state=0,
):
    """Recommend one continuous-domain point using Gaussian-process LCB.

    ``samples`` and ``values`` are previously observed input/output pairs for
    a scalar objective to minimize. The function fits a Gaussian-process
    surrogate and selects the lowest lower-confidence-bound value from a
    seeded Latin-hypercube candidate pool. It returns prediction and sampling
    metadata as well as the candidate; it never evaluates the true objective.

    This is a bounded prototype, not a constrained optimizer. Only finite,
    continuous numeric domains are supported; domain-specific feasibility,
    physical constraints, and operational safety must be reviewed by the user.
    """
    from scipy.stats import qmc
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    limits = _validated_bounds(bounds)
    seed = _validated_seed(random_state)
    if (
        isinstance(n_candidates, (bool, np.bool_))
        or not isinstance(n_candidates, (int, np.integer))
        or n_candidates < 1
        or n_candidates > _MAX_DESIGN_ROWS
    ):
        raise ValueError(
            "n_candidates must be an integer between 1 and %d" % _MAX_DESIGN_ROWS
        )
    if isinstance(beta, (bool, np.bool_)) or not isinstance(
        beta, (int, float, np.integer, np.floating)
    ):
        raise ValueError("beta must be a finite positive number")
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be a finite positive number")

    try:
        observed = np.asarray(samples, dtype=float)
        responses = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("samples and values must contain finite numeric data") from exc
    if observed.ndim == 1 and len(limits) == 1:
        observed = observed.reshape(-1, 1)
    if observed.ndim != 2 or observed.shape[1] != len(limits):
        raise ValueError("samples must have shape (n_observations, n_dimensions)")
    if observed.shape[0] < 2 or observed.shape[0] > _MAX_SURROGATE_OBSERVATIONS:
        raise ValueError(
            "samples must contain between 2 and %d observations"
            % _MAX_SURROGATE_OBSERVATIONS
        )
    if len(responses) != len(observed):
        raise ValueError("values must contain one scalar per sample")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(responses)):
        raise ValueError("samples and values must be finite")
    if np.any(observed < limits[:, 0]) or np.any(observed > limits[:, 1]):
        raise ValueError("samples must lie within bounds")

    span = limits[:, 1] - limits[:, 0]
    observed_unit = (observed - limits[:, 0]) / span
    sampler = qmc.LatinHypercube(d=len(limits), seed=seed)
    candidates_unit = sampler.random(n=int(n_candidates))
    candidates = qmc.scale(candidates_unit, limits[:, 0], limits[:, 1])
    not_observed = ~np.any(
        np.all(
            np.isclose(
                candidates_unit[:, np.newaxis, :],
                observed_unit[np.newaxis, :, :],
                rtol=0.0,
                atol=1.0e-12,
            ),
            axis=2,
        ),
        axis=1,
    )
    candidates_unit = candidates_unit[not_observed]
    candidates = candidates[not_observed]
    if not len(candidates):
        raise ValueError("candidate pool contains no points distinct from observed samples")

    surrogate = GaussianProcessRegressor(
        kernel=Matern(
            length_scale=np.full(len(limits), _GP_LENGTH_SCALE),
            length_scale_bounds="fixed",
            nu=2.5,
        ),
        alpha=1.0e-8,
        normalize_y=True,
        optimizer=None,
        random_state=seed,
    )
    surrogate.fit(observed_unit, responses)
    predicted_mean, predicted_std = surrogate.predict(candidates_unit, return_std=True)
    acquisition = predicted_mean - beta * predicted_std
    selected_index = int(np.argmin(acquisition))
    return {
        "candidate": candidates[selected_index].copy(),
        "predicted_mean": float(predicted_mean[selected_index]),
        "predicted_std": float(predicted_std[selected_index]),
        "acquisition_value": float(acquisition[selected_index]),
        "acquisition": "lower_confidence_bound",
        "surrogate_kernel": "Matern(nu=2.5)",
        "kernel_length_scale": [_GP_LENGTH_SCALE] * len(limits),
        "beta": beta,
        "observations_used": int(len(observed)),
        "candidate_pool_size": int(len(candidates)),
        "random_state": seed,
        "optimization_direction": "minimize",
        "interpretation": (
            "Surrogate-based candidate only; the objective was not evaluated "
            "and physical or operational constraints were not checked."
        ),
    }
