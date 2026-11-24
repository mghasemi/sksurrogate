"""Tests for benchmarks/synth.py — deterministic synthetic benchmarks (v2 suite).

The contract under test is the one stated in the module docstring: every
registry entry is a bounded minimization problem whose published minimizer
reproduces its published global minimum, regret is measured against that
minimum, and the noisy wrapper is an additive-Gaussian observation model that
is reproducible for a fixed seed while advancing once per evaluation.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks import synth

REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_NAMES = ("branin", "hartmann3", "hartmann6", "ackley", "rosenbrock", "levy")

# Branin and the Hartmann functions are quoted to 4-6 decimals in the
# literature, so their published minimizers only match the published minimum
# at 1e-4; the smooth origin/ones-optimum functions are exact at 1e-5.
MINIMUM_TOL = {
    "branin": 1e-4,
    "hartmann3": 1e-4,
    "hartmann6": 1e-4,
    "ackley": 1e-5,
    "rosenbrock": 1e-5,
    "levy": 1e-5,
}


class TestRegistry:
    def test_registry_keys_are_exactly_the_six_benchmarks(self):
        assert set(synth.FUNCTIONS) == set(ALL_NAMES)

    def test_entries_are_frozen_dataclasses_with_required_fields(self):
        import dataclasses

        for fn in synth.FUNCTIONS.values():
            assert isinstance(fn, synth.SyntheticFunction)
            assert dataclasses.is_dataclass(fn)
            assert {f.name for f in dataclasses.fields(fn)} >= {
                "name",
                "dim",
                "bounds",
                "minimum",
                "minimizers",
            }
            with pytest.raises(dataclasses.FrozenInstanceError):
                fn.dim = fn.dim + 1  # type: ignore[misc]

    def test_metadata_types(self):
        for name, fn in synth.FUNCTIONS.items():
            assert fn.name == name
            assert isinstance(fn.dim, int) and fn.dim >= 2
            assert isinstance(fn.bounds, np.ndarray)
            assert fn.bounds.dtype == np.float64
            assert isinstance(fn.minimum, float)
            assert isinstance(fn.minimizers, tuple) and fn.minimizers
            for minimizer in fn.minimizers:
                assert isinstance(minimizer, tuple)
                assert len(minimizer) == fn.dim
                assert all(isinstance(v, float) for v in minimizer)

    def test_call_returns_python_float(self):
        fn = synth.FUNCTIONS["rosenbrock"]
        value = fn((0.3, -1.2))
        assert isinstance(value, float)

    def test_wrong_input_dimension_is_rejected(self):
        fn = synth.FUNCTIONS["hartmann6"]
        with pytest.raises(ValueError, match="dimension 6"):
            fn((0.5, 0.5))
        with pytest.raises(ValueError, match="dimension 3"):
            synth.FUNCTIONS["hartmann3"]((0.5, 0.5, 0.5, 0.5))


class TestKnownMinima:
    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_every_minimizer_hits_the_stated_minimum(self, name):
        fn = synth.get_function(name)
        for minimizer in fn.minimizers:
            assert fn(minimizer) == pytest.approx(
                fn.minimum, abs=MINIMUM_TOL[name]
            )

    def test_stated_minima_match_published_values(self):
        expected = {
            "branin": 0.397887,
            "hartmann3": -3.86278,
            "hartmann6": -3.32237,
            "ackley": 0.0,
            "rosenbrock": 0.0,
            "levy": 0.0,
        }
        for name, minimum in expected.items():
            assert synth.FUNCTIONS[name].minimum == pytest.approx(minimum, abs=1e-5)

    def test_dimensions_and_multiplicity_of_minimizers(self):
        assert synth.FUNCTIONS["branin"].dim == 2
        assert len(synth.FUNCTIONS["branin"].minimizers) == 3
        assert synth.FUNCTIONS["hartmann3"].dim == 3
        assert synth.FUNCTIONS["hartmann6"].dim == 6
        for name in ("ackley", "rosenbrock", "levy"):
            assert synth.FUNCTIONS[name].dim == 2

    def test_values_off_the_minimizer_are_strictly_larger(self):
        for name, fn in synth.FUNCTIONS.items():
            corner = tuple(float(bounds[0]) for bounds in fn.bounds)
            assert fn(corner) > fn.minimum + 1e-3


class TestBounds:
    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_bounds_shape_and_ordering(self, name):
        fn = synth.get_function(name)
        assert fn.bounds.shape == (fn.dim, 2)
        assert np.all(fn.bounds[:, 0] < fn.bounds[:, 1])

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_minimizers_lie_inside_the_box(self, name):
        fn = synth.get_function(name)
        for minimizer in fn.minimizers:
            point = np.asarray(minimizer)
            assert np.all(point >= fn.bounds[:, 0])
            assert np.all(point <= fn.bounds[:, 1])

    def test_published_boxes(self):
        np.testing.assert_allclose(
            synth.FUNCTIONS["branin"].bounds, [[-5.0, 10.0], [0.0, 15.0]]
        )
        np.testing.assert_allclose(synth.FUNCTIONS["hartmann3"].bounds, [[0.0, 1.0]] * 3)
        np.testing.assert_allclose(synth.FUNCTIONS["hartmann6"].bounds, [[0.0, 1.0]] * 6)
        np.testing.assert_allclose(
            synth.FUNCTIONS["ackley"].bounds, [[-32.768, 32.768]] * 2
        )
        np.testing.assert_allclose(synth.FUNCTIONS["rosenbrock"].bounds, [[-5.0, 10.0]] * 2)
        np.testing.assert_allclose(synth.FUNCTIONS["levy"].bounds, [[-10.0, 10.0]] * 2)

    def test_bounds_are_not_shared_between_calls(self):
        a = synth.get_function("levy", dimension=3)
        a.bounds[0, 0] = -1e9
        b = synth.get_function("levy", dimension=3)
        assert b.bounds[0, 0] == -10.0
        assert synth.FUNCTIONS["levy"].bounds[0, 0] == -10.0


class TestDimensionOverride:
    def test_ackley_override_changes_dim_and_bounds(self):
        fn = synth.get_function("ackley", dimension=5)
        assert fn.dim == 5
        assert fn.bounds.shape == (5, 2)
        np.testing.assert_allclose(fn.bounds, [[-32.768, 32.768]] * 5)
        assert fn.minimizers == ((0.0,) * 5,)
        assert fn(fn.minimizers[0]) == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize("name", ("ackley", "rosenbrock", "levy"))
    def test_override_is_honoured_for_every_parametric_function(self, name):
        fn = synth.get_function(name, dimension=4)
        assert fn.dim == 4
        assert fn.bounds.shape == (4, 2)
        assert len(fn.minimizers[0]) == 4
        assert fn(fn.minimizers[0]) == pytest.approx(fn.minimum, abs=1e-5)

    def test_default_dimension_is_two(self):
        for name in ("ackley", "rosenbrock", "levy"):
            assert synth.get_function(name).dim == 2
            assert synth.get_function(name, dimension=None).dim == 2

    def test_fixed_dimension_functions_reject_a_mismatch(self):
        with pytest.raises(ValueError, match="fixed dimension"):
            synth.get_function("branin", dimension=3)
        with pytest.raises(ValueError, match="fixed dimension"):
            synth.get_function("hartmann6", dimension=2)
        # The matching dimension is accepted and is a no-op.
        assert synth.get_function("branin", dimension=2).dim == 2

    def test_too_small_dimension_is_rejected(self):
        with pytest.raises(ValueError, match="dimension must be >= 2"):
            synth.get_function("rosenbrock", dimension=1)


class TestUnknownName:
    def test_unknown_name_raises_key_error_listing_known_names(self):
        with pytest.raises(KeyError) as excinfo:
            synth.get_function("rastrigin")
        message = str(excinfo.value)
        for name in ALL_NAMES:
            assert name in message


class TestSimpleRegret:
    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_regret_is_zero_at_every_minimizer(self, name):
        fn = synth.get_function(name)
        for minimizer in fn.minimizers:
            assert synth.simple_regret(fn, fn(minimizer)) == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_regret_is_positive_away_from_the_minimizer(self, name):
        fn = synth.get_function(name)
        corner = tuple(float(b[0]) for b in fn.bounds)
        assert synth.simple_regret(fn, fn(corner)) > 0.0

    def test_regret_is_value_minus_minimum(self):
        fn = synth.FUNCTIONS["ackley"]
        assert synth.simple_regret(fn, 2.5) == pytest.approx(2.5 - fn.minimum)
        assert synth.simple_regret(fn, -1.0) == pytest.approx(-1.0 - fn.minimum)

    def test_regret_accepts_a_noisy_wrapper(self):
        wrapped = synth.noisy(synth.FUNCTIONS["levy"], seed=2)
        assert synth.simple_regret(wrapped, wrapped.minimum) == pytest.approx(0.0)
        assert wrapped.minimum == synth.FUNCTIONS["levy"].minimum


class TestNoisy:
    def test_wrapper_exposes_metadata(self):
        fn = synth.FUNCTIONS["hartmann3"]
        wrapped = synth.noisy(fn, relative_sigma=0.05, seed=13)
        assert wrapped.function is fn
        assert wrapped.relative_sigma == 0.05
        assert wrapped.seed == 13
        assert wrapped.name == fn.name
        assert wrapped.dim == fn.dim
        np.testing.assert_array_equal(wrapped.bounds, fn.bounds)
        assert wrapped.minimum == fn.minimum
        assert wrapped.minimizers == fn.minimizers

    def test_noise_scale_uses_the_function_minimum(self):
        # |f_min| < 1 => floor of 1.0; |f_min| > 1 => |f_min|.
        ackley = synth.noisy(synth.FUNCTIONS["ackley"], relative_sigma=0.02)
        hartmann = synth.noisy(synth.FUNCTIONS["hartmann3"], relative_sigma=0.02)
        assert ackley.noise_scale == pytest.approx(0.02)
        assert hartmann.noise_scale == pytest.approx(0.02 * abs(-3.862779787332663))

    def test_same_seed_replays_the_identical_sequence(self):
        fn = synth.FUNCTIONS["ackley"]
        points = [tuple(0.5 * i for _ in range(2)) for i in range(1, 9)]
        first = synth.noisy(fn, seed=7)
        second = synth.noisy(fn, seed=7)
        seq_first = [first(p) for p in points]
        seq_second = [second(p) for p in points]
        assert seq_first == seq_second

    def test_different_seeds_differ(self):
        fn = synth.FUNCTIONS["ackley"]
        point = (0.25, -0.75)
        a = synth.noisy(fn, seed=0)
        b = synth.noisy(fn, seed=1)
        assert [a(point) for _ in range(5)] != [b(point) for _ in range(5)]

    def test_generator_advances_once_per_evaluation(self):
        fn = synth.FUNCTIONS["ackley"]
        point = (0.25, -0.75)
        wrapped = synth.noisy(fn, seed=3)
        draws = [wrapped(point) for _ in range(6)]
        # The noise is not a deterministic function of x ...
        assert len(set(draws)) == len(draws)
        # ... and replaying the wrapper reproduces the same draws in order.
        replay = synth.noisy(fn, seed=3)
        assert [replay(point) for _ in range(6)] == draws

    def test_reset_rewinds_the_sequence(self):
        point = (0.25, -0.75)
        wrapped = synth.noisy(synth.FUNCTIONS["levy"], seed=5)
        first = [wrapped(point) for _ in range(4)]
        assert wrapped.reset() is wrapped
        assert [wrapped(point) for _ in range(4)] == first

    def test_noise_matches_the_documented_gaussian_scale(self):
        fn = synth.FUNCTIONS["ackley"]  # minimum 0 => noise_scale = relative_sigma
        wrapped = synth.noisy(fn, relative_sigma=0.2, seed=11)
        point = (1.0, 0.5)
        clean = fn(point)
        residuals = np.array([wrapped(point) - clean for _ in range(4000)])
        assert residuals.mean() == pytest.approx(0.0, abs=0.005)
        assert residuals.std() == pytest.approx(wrapped.noise_scale, rel=0.1)

    def test_zero_sigma_is_exact(self):
        fn = synth.FUNCTIONS["rosenbrock"]
        wrapped = synth.noisy(fn, relative_sigma=0.0, seed=0)
        point = (0.9, 0.8)
        assert wrapped(point) == fn(point)

    def test_negative_sigma_is_rejected(self):
        with pytest.raises(ValueError, match="relative_sigma"):
            synth.noisy(synth.FUNCTIONS["levy"], relative_sigma=-0.1)


class TestCliSmoke:
    def test_module_smoke_exits_zero(self):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        proc = subprocess.run(
            [sys.executable, "-m", "benchmarks.synth"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        for name in ALL_NAMES:
            assert name in proc.stdout
        assert "SMOKE OK" in proc.stdout

    def test_smoke_fails_loudly_on_a_broken_minimum(self, monkeypatch, capsys):
        broken = synth.SyntheticFunction(
            name="ackley",
            dim=2,
            bounds=np.array([[-32.768, 32.768]] * 2),
            minimum=1.0,  # deliberately wrong
            minimizers=((0.0, 0.0),),
        )
        monkeypatch.setitem(synth.FUNCTIONS, "ackley", broken)
        assert synth._smoke() == 1
        captured = capsys.readouterr()
        assert "SMOKE FAILED" in captured.out
