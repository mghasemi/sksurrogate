"""Synthetic objectives for the D-track Hilbert-space surrogate study.

Closed-form functions on the box ``[-1, 1]^d`` so that approximation error is
measurable against a dense test grid. Each objective is a **vectorized** callable
``f(X) -> (n,)`` for ``X`` an ``(n, d)`` array.

Families (D1/D2 regime axis "objective type"):
- ``poly``    : low-degree polynomial  -> Hilbert (polynomial basis) should be near-exact
- ``trig``    : trigonometric         -> Fourier-type, smooth but not polynomial
- ``mixed``   : smooth non-polynomial (Gaussian bump * linear)
- ``rational``: non-polynomial with a pole-free rational form (harder for polynomials)

All objectives are normalized so the response has comparable scale across
families (roughly O(1)), which makes noise levels and RMSE comparable.
"""

from __future__ import annotations

import numpy as np

DOMAIN = (-1.0, 1.0)


def make_objective(family: str, d: int):
    """Return ``(f, name)`` for a synthetic objective of dimension ``d``.

    ``f`` is vectorized: ``f(X) -> (n,)`` for ``X`` of shape ``(n, d)``.
    """
    if family == "poly":
        def f(X):
            X = np.asarray(X, dtype=float)
            if d == 1:
                return 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 0] ** 2 + 0.5 * X[:, 0] ** 3
            if d == 2:
                return (1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 0] ** 2 + 0.5 * X[:, 0] ** 3
                        + 1.5 * X[:, 1] - 0.5 * X[:, 1] ** 2 + 0.25 * X[:, 0] * X[:, 1])
            # d == 3
            return (1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 0] ** 2 + 0.5 * X[:, 0] ** 3
                    + 1.5 * X[:, 1] - 0.5 * X[:, 1] ** 2 + 0.25 * X[:, 0] * X[:, 1]
                    + 0.75 * X[:, 2] - 0.25 * X[:, 2] ** 2 + 0.1 * X[:, 0] * X[:, 1] * X[:, 2])

    elif family == "trig":
        def f(X):
            X = np.asarray(X, dtype=float)
            s = np.zeros(X.shape[0])
            for i in range(d):
                s += np.sin(2.0 * np.pi * X[:, i]) + 0.5 * np.cos(2.0 * np.pi * X[:, i])
            return s / d

    elif family == "mixed":
        def f(X):
            X = np.asarray(X, dtype=float)
            bump = np.exp(-np.sum(X ** 2, axis=1))
            lin = 1.0 + 0.5 * np.sum(X, axis=1)
            return bump * lin

    elif family == "rational":
        def f(X):
            X = np.asarray(X, dtype=float)
            return 1.0 / (1.0 + np.sum(X ** 2, axis=1))

    else:
        raise ValueError(f"unknown objective family {family!r}")

    return f, f"{family}_d{d}"


def sample_domain(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` points uniformly in ``[-1,1]^d``."""
    return rng.uniform(-1.0, 1.0, size=(n, d))


def test_grid(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """A dense quasi-uniform test grid (deterministic per seed) for RMSE."""
    # Latin-hypercube-ish: stratified uniform, deterministic given rng.
    pts = np.empty((n, d))
    for i in range(d):
        u = (np.arange(n) + rng.random(n)) / n
        pts[:, i] = -1.0 + 2.0 * u
    rng.shuffle(pts)
    return pts


FAMILIES = ["poly", "trig", "mixed", "rational"]
DIMS = [1, 2, 3]
