"""Synthetic benchmark functions with known global minima (benchmark suite v2).

This module supplies the deterministic objective layer of the v2 synthetic
benchmark protocol described in ``benchmarks/DESIGN.md``: six standard
single-objective test functions, each carrying its own search domain, its
published global minimum, and at least one exact global minimizer. Downstream
code (search runners, acquisition studies, regret curves) consumes the
registry through :func:`get_function` and never needs to know a formula.

Contract
--------

- Every entry of :data:`FUNCTIONS` is a frozen :class:`SyntheticFunction`
  whose ``bounds`` are the *inclusive* search box, shape ``(dim, 2)``, and
  whose ``minimizers`` are points of that box. ``function(m)`` reproduces
  ``function.minimum`` to floating-point accuracy; the module smoke test
  (``python -m benchmarks.synth``) asserts that agreement at ``1e-6``.
- All six functions are minimization problems. Function values are returned as
  plain ``float`` so that regret arithmetic is unambiguous.
- Branin and the two Hartmann functions have a fixed input dimension;
  Ackley, Rosenbrock and Levy follow the customary convention of being
  dimension-parametric, with ``dimension=2`` as the default. Passing a
  ``dimension`` to a fixed-dimension function is an error rather than a
  silent no-op, because a mismatched ``dim`` would otherwise produce a
  function that cannot be evaluated on its own declared bounds.
- The optima below are the standard constants from the optimization literature
  (Branin 0.397887 at three points; Hartmann-3 -3.86278; Hartmann-6 -3.32237;
  Ackley/Rosenbrock/Levy 0 at the origin resp. the all-ones vector). Minimizer
  coordinates are quoted exactly as published; they are truncated decimals, so
  ``f(minimizer)`` reproduces ``minimum`` to about ``2e-10`` rather than
  bit-for-bit — three orders of magnitude inside the ``1e-6`` smoke tolerance.

Noisy evaluations
-----------------

:func:`noisy` wraps a benchmark function in an additive Gaussian observation
model. The noise scale is *absolute* and fixed per function::

    sigma = relative_sigma * max(1.0, abs(function.minimum))

The ``max(1.0, ...)`` floor is what the design contract calls the "signal":
it keeps the perturbation meaningful for functions whose optimum is zero
(Ackley, Rosenbrock, Levy), where a purely relative-to-``f_min`` scale would
vanish. Noise is therefore homoscedastic in ``x``, which is the usual
assumption for an expensive-experiment surrogate model; it does not scale with
the local value of ``f``. For the zero-minimum functions the default
``relative_sigma=0.02`` corresponds to ``sigma = 0.02``, i.e. small next to the
O(1)-O(10) value range those functions take inside their boxes.

A :class:`NoisyFunction` owns a ``numpy.random.Generator`` seeded with
``seed`` that **advances on every evaluation**, so repeated evaluation of the
same point returns different observations (which is the point of a noisy
benchmark). Determinism is per wrapper: two wrappers constructed with the same
function, ``relative_sigma`` and ``seed`` produce identical sequences, while a
single long-lived wrapper produces one fixed sequence. :meth:`NoisyFunction.reset`
rewinds the generator so a run can be repeated in-process. This "advance per
evaluation, reproducible per seed" behaviour is the documented contract; a
generator that *re-seeded per evaluation* would make the noise a deterministic
function of ``x`` and defeat the purpose.

Run directly::

    python -m benchmarks.synth
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

_TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Raw objective formulas (standard published definitions)
# ---------------------------------------------------------------------------
#
# Each implementation takes ``(dim, x)`` with ``x`` already validated to a
# 1-D float array of length ``dim``. ``dim`` is unused by the fixed-dimension
# functions, which keeps one uniform signature for the dispatch table.

# Branin-Hoo (a.k.a. Branin RCOS), bounds [-5, 10] x [0, 15].
_BRANIN_A = 1.0
_BRANIN_B = 5.1 / (4.0 * np.pi**2)
_BRANIN_C = 5.0 / np.pi
_BRANIN_R = 6.0
_BRANIN_S = 10.0
_BRANIN_T = 1.0 / (8.0 * np.pi)


def _branin(dim: int, x: np.ndarray) -> float:
    residual = x[1] - _BRANIN_B * x[0] ** 2 + _BRANIN_C * x[0] - _BRANIN_R
    return float(
        _BRANIN_A * residual**2
        + _BRANIN_S * (1.0 - _BRANIN_T) * np.cos(x[0])
        + _BRANIN_S
    )


# Hartmann-3, bounds [0, 1]^3.
_HARTMANN3_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_HARTMANN3_A = np.array(
    [
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
    ]
)
_HARTMANN3_P = 1e-4 * np.array(
    [
        [3689.0, 1170.0, 2673.0],
        [4699.0, 4387.0, 7470.0],
        [1091.0, 8732.0, 5547.0],
        [381.0, 5743.0, 8828.0],
    ]
)

# Hartmann-6, bounds [0, 1]^6.
_HARTMANN6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_HARTMANN6_A = np.array(
    [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ]
)
_HARTMANN6_P = 1e-4 * np.array(
    [
        [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
        [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
        [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
        [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
    ]
)


def _hartmann(alpha: np.ndarray, a: np.ndarray, p: np.ndarray, x: np.ndarray) -> float:
    inner = (a * (x[np.newaxis, :] - p) ** 2).sum(axis=1)
    return float(-(alpha * np.exp(-inner)).sum())


def _hartmann3(dim: int, x: np.ndarray) -> float:
    return _hartmann(_HARTMANN3_ALPHA, _HARTMANN3_A, _HARTMANN3_P, x)


def _hartmann6(dim: int, x: np.ndarray) -> float:
    return _hartmann(_HARTMANN6_ALPHA, _HARTMANN6_A, _HARTMANN6_P, x)


# Ackley, bounds [-32.768, 32.768]^d (conventionally, so that 32.768 is a
# power-of-two multiple that covers several periods of the cosine term).
_ACKLEY_BOUND = 32.768


def _ackley(dim: int, x: np.ndarray) -> float:
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
        - np.exp(np.mean(np.cos(_TWO_PI * x)))
        + 20.0
        + np.e
    )


def _rosenbrock(dim: int, x: np.ndarray) -> float:
    return float(
        np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
    )


def _levy(dim: int, x: np.ndarray) -> float:
    w = 1.0 + (x - 1.0) / 4.0
    head = np.sin(np.pi * w[0]) ** 2
    middle = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
    tail = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(_TWO_PI * w[-1]) ** 2)
    return float(head + middle + tail)


_IMPLEMENTATIONS: Mapping[str, Callable[[int, np.ndarray], float]] = {
    "branin": _branin,
    "hartmann3": _hartmann3,
    "hartmann6": _hartmann6,
    "ackley": _ackley,
    "rosenbrock": _rosenbrock,
    "levy": _levy,
}


# ---------------------------------------------------------------------------
# Registry metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SyntheticFunction:
    """A synthetic minimization problem with a known global optimum.

    Parameters
    ----------
    name:
        Registry key, e.g. ``"hartmann6"``. The value is also the dispatch key
        for the objective formula, so instances are normally obtained from
        :func:`get_function` rather than constructed by hand.
    dim:
        Input dimension; ``bounds`` and every minimizer have this length.
    bounds:
        Search box, shape ``(dim, 2)`` with ``bounds[:, 0] <= bounds[:, 1]``.
    minimum:
        Global minimum value in the box.
    minimizers:
        One or more global minimizers, each a length-``dim`` tuple of floats.

    Equality is identity-based (``eq=False``) because ``bounds`` is an ndarray
    and ``minimizers`` is a tuple of tuples: a field-wise ``==`` would either
    raise on the array or silently compare arrays elementwise. Registry
    instances are singletons, so identity is the meaningful comparison.
    """

    name: str
    dim: int
    bounds: np.ndarray
    minimum: float
    minimizers: tuple[tuple[float, ...], ...]

    def __call__(self, x) -> float:
        """Evaluate the objective at ``x`` (sequence or array of length ``dim``)."""
        z = np.asarray(x, dtype=float).reshape(-1)
        if z.size != self.dim:
            raise ValueError(
                f"{self.name}: expected a point of dimension {self.dim}, got {z.size}"
            )
        impl = _IMPLEMENTATIONS.get(self.name)
        if impl is None:  # pragma: no cover - defensive; only registry names dispatch
            raise KeyError(
                f"no implementation registered for {self.name!r}; "
                f"known names: {', '.join(sorted(_IMPLEMENTATIONS))}"
            )
        return impl(self.dim, z)


@dataclass(frozen=True)
class _FixedSpec:
    """Internal metadata template for a fixed-dimension benchmark function."""

    dim: int
    bounds: np.ndarray
    minimum: float
    minimizers: tuple[tuple[float, ...], ...]


# Minimizer coordinates are the published values (Branin's third point is
# 3*pi, quoted as 9.42478). ``minimum`` is the value at the true stationary
# point; the difference from ``f(published minimizer)`` is <= 2.4e-10.
_FIXED_SPECS: Mapping[str, _FixedSpec] = {
    "branin": _FixedSpec(
        dim=2,
        bounds=np.array([[-5.0, 10.0], [0.0, 15.0]]),
        minimum=0.39788735772973816,
        minimizers=(
            (-np.pi, 12.275),
            (np.pi, 2.275),
            (9.42478, 2.475),
        ),
    ),
    "hartmann3": _FixedSpec(
        dim=3,
        bounds=np.array([[0.0, 1.0]] * 3),
        minimum=-3.862779787332663,
        minimizers=((0.114614, 0.555649, 0.852547),),
    ),
    "hartmann6": _FixedSpec(
        dim=6,
        bounds=np.array([[0.0, 1.0]] * 6),
        minimum=-3.3223680114155147,
        minimizers=((0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),),
    ),
}

#: Dimension-parametric functions: name -> (default dim, box, minimizer value).
_FLEXIBLE_SPECS: Mapping[str, tuple[int, float, float, float]] = {
    # name: (default_dim, lower, upper, minimizer_coordinate)
    "ackley": (2, -_ACKLEY_BOUND, _ACKLEY_BOUND, 0.0),
    "rosenbrock": (2, -5.0, 10.0, 1.0),
    "levy": (2, -10.0, 10.0, 1.0),
}

_MINIMUM_DIM = 2


def _checked_dimension(name: str, dimension: int | None, default: int) -> int:
    dim = default if dimension is None else int(dimension)
    if dim < _MINIMUM_DIM:
        raise ValueError(f"{name}: dimension must be >= {_MINIMUM_DIM}, got {dim}")
    return dim


def get_function(name: str, *, dimension: int | None = None) -> SyntheticFunction:
    """Return the benchmark function registered under ``name``.

    Parameters
    ----------
    name:
        One of :data:`FUNCTIONS` (``branin``, ``hartmann3``, ``hartmann6``,
        ``ackley``, ``rosenbrock``, ``levy``).
    dimension:
        Optional input dimension for the dimension-parametric functions
        (``ackley``, ``rosenbrock``, ``levy``); default is 2, the customary
        two-variable form. Supplying it for a fixed-dimension function
        (``branin``, ``hartmann3``, ``hartmann6``) is a ``ValueError`` unless
        it equals the function's own dimension.

    Raises
    ------
    KeyError
        For an unknown name; the message lists the known names.
    """
    fixed = _FIXED_SPECS.get(name)
    if fixed is not None:
        if dimension is not None and int(dimension) != fixed.dim:
            raise ValueError(
                f"{name}: fixed dimension is {fixed.dim}, got dimension={dimension}"
            )
        return SyntheticFunction(
            name=name,
            dim=fixed.dim,
            bounds=fixed.bounds.copy(),
            minimum=float(fixed.minimum),
            minimizers=tuple(tuple(float(v) for v in m) for m in fixed.minimizers),
        )

    flexible = _FLEXIBLE_SPECS.get(name)
    if flexible is None:
        known = ", ".join(sorted(FUNCTIONS))
        raise KeyError(f"unknown synthetic function {name!r}; known names: {known}")

    default_dim, lower, upper, optimum_coordinate = flexible
    dim = _checked_dimension(name, dimension, default_dim)
    return SyntheticFunction(
        name=name,
        dim=dim,
        bounds=np.tile(np.array([lower, upper], dtype=float), (dim, 1)),
        minimum=0.0,
        minimizers=(tuple([float(optimum_coordinate)] * dim),),
    )


#: The canonical registry: one singleton per benchmark function at its default
#: dimension. Keys are exactly the names accepted by :func:`get_function`.
FUNCTIONS: dict[str, SyntheticFunction] = {
    name: get_function(name)
    for name in ("branin", "hartmann3", "hartmann6", "ackley", "rosenbrock", "levy")
}


# ---------------------------------------------------------------------------
# Noisy observations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoisyFunction:
    """Additive-Gaussian wrapper around a :class:`SyntheticFunction`.

    ``y_obs(x) = f(x) + eps``, ``eps ~ N(0, sigma^2)`` with
    ``sigma = relative_sigma * max(1.0, abs(f.minimum))``. The internal
    generator advances once per call, so repeated evaluations of one point
    differ; two wrappers sharing ``(function, relative_sigma, seed)`` replay
    the identical sequence. :meth:`reset` rewinds the generator.
    """

    function: SyntheticFunction
    relative_sigma: float = 0.02
    seed: int = 0
    _rng: np.random.Generator = field(
        default=None, init=False, repr=False, compare=False
    )  # type: ignore[assignment]  # replaced in __post_init__

    def __post_init__(self) -> None:
        if not np.isfinite(self.relative_sigma) or self.relative_sigma < 0.0:
            raise ValueError(
                f"relative_sigma must be finite and non-negative, got {self.relative_sigma!r}"
            )
        object.__setattr__(self, "_rng", np.random.default_rng(int(self.seed)))

    # -- metadata pass-through ------------------------------------------------
    @property
    def name(self) -> str:
        return self.function.name

    @property
    def dim(self) -> int:
        return self.function.dim

    @property
    def bounds(self) -> np.ndarray:
        return self.function.bounds

    @property
    def minimum(self) -> float:
        return self.function.minimum

    @property
    def minimizers(self) -> tuple[tuple[float, ...], ...]:
        return self.function.minimizers

    @property
    def noise_scale(self) -> float:
        """Standard deviation of the additive noise, ``relative_sigma * signal``."""
        return float(self.relative_sigma * max(1.0, abs(self.function.minimum)))

    # -- evaluation -----------------------------------------------------------
    def __call__(self, x) -> float:
        clean = self.function(x)
        return float(clean + self._rng.normal(0.0, self.noise_scale))

    def reset(self) -> "NoisyFunction":
        """Rewind the generator; returns ``self`` for chaining."""
        object.__setattr__(self, "_rng", np.random.default_rng(int(self.seed)))
        return self


def noisy(
    function: SyntheticFunction,
    *,
    relative_sigma: float = 0.02,
    seed: int = 0,
) -> NoisyFunction:
    """Return an additive-Gaussian observation model for ``function``.

    The wrap is deterministic for a fixed ``seed``: wrapper construction always
    starts from ``numpy.random.default_rng(seed)``, and the generator advances
    by exactly one draw per evaluation, so the ``k``-th observation is the same
    for every wrapper built with the same arguments.
    """
    return NoisyFunction(function=function, relative_sigma=relative_sigma, seed=seed)


# ---------------------------------------------------------------------------
# Regret
# ---------------------------------------------------------------------------


def simple_regret(function, value) -> float:
    """Simple regret for minimization: ``value - function.minimum``.

    Non-negative at any point of the box (up to floating-point noise) and zero
    at a global minimizer. Accepts either a :class:`SyntheticFunction` or a
    :class:`NoisyFunction` (both expose ``.minimum``).
    """
    return float(value) - float(function.minimum)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke() -> int:
    """Verify every registry entry reproduces its stated minimum at 1e-6."""
    failures: list[str] = []
    for name, fn in FUNCTIONS.items():
        print(f"{name:<12} dim={fn.dim:<2} minimum={fn.minimum: .8f}")
        for minimizer in fn.minimizers:
            observed = fn(minimizer)
            if abs(observed - fn.minimum) > 1e-6:
                failures.append(
                    f"{name}: f({minimizer}) = {observed!r}, "
                    f"expected {fn.minimum!r} (|diff| = {abs(observed - fn.minimum):.3e})"
                )
    if failures:
        print("\nSMOKE FAILED")
        for message in failures:
            print("  " + message)
        return 1
    print("\nSMOKE OK: all minimizers reproduce their stated minimum within 1e-6")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
