from unittest.mock import patch

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import WhiteKernel

from benchmarks.dtrack.real_noise_gp_probe import (
    _load_probe_dataset,
    _run_noise_aware_gp,
    run_probe,
)


def test_real_probe_requires_budget_for_surrogate_engagement():
    with pytest.raises(ValueError, match="at least 21"):
        run_probe(budget=20)


def test_real_probe_requires_at_least_one_seed():
    with pytest.raises(ValueError, match="at least one seed"):
        run_probe(seeds=[])


def test_real_probe_loads_bundled_sklearn_datasets_without_network():
    X, y = _load_probe_dataset("wine")

    assert X.shape == (178, 13)
    assert y.nunique() == 3


def test_noise_aware_gp_arm_uses_white_kernel_and_requested_seed():
    captured = {}

    def fake_run_srs(*args, **kwargs):
        captured.update(kwargs)
        return object()

    with patch("benchmarks.dtrack.real_noise_gp_probe._run_srs", side_effect=fake_run_srs):
        _run_noise_aware_gp(
            np.empty((1, 1)),
            np.empty(1),
            np.empty((1, 1)),
            np.empty(1),
            budget=30,
            seed=4,
            n_estimators=10,
        )

    regressor = captured["regressor_factory"]()
    assert isinstance(regressor.kernel.k2, WhiteKernel)
    assert regressor.random_state == 4
    assert captured["method_name"] == "sksurrogate-gp-white"
