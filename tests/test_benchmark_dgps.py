"""Tests for benchmarks/dgps.py — Stage D1 controlled synthetic DGPs (R6).

Covers seed determinism, ground-truth structure of each DGP at large n, the
validation harness's report schema and JSON output, and the CLI entry point.
All statistical assertions are deliberately loose: they pin *structure*
(recovered coefficients, empirical correlations, residual variance tracking,
shift degradation) with wide margins, not tight sampling estimates.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from benchmarks import dgps


class TestSampleDgp:
    def test_determinism_same_seed_identical(self):
        a = dgps.sample_dgp("interactions", n_train=64, n_test=32, seed=11)
        b = dgps.sample_dgp("interactions", n_train=64, n_test=32, seed=11)
        pd.testing.assert_frame_equal(a.X_train, b.X_train)
        pd.testing.assert_frame_equal(a.X_test, b.X_test)
        pd.testing.assert_frame_equal(a.X_iid, b.X_iid)
        np.testing.assert_array_equal(a.y_train, b.y_train)
        np.testing.assert_array_equal(a.y_test, b.y_test)
        np.testing.assert_array_equal(a.y_iid, b.y_iid)

    def test_different_seeds_differ(self):
        a = dgps.sample_dgp("correlated", n_train=64, seed=0)
        b = dgps.sample_dgp("correlated", n_train=64, seed=1)
        assert not np.allclose(a.X_train.to_numpy(), b.X_train.to_numpy())

    def test_unknown_dgp_lists_known_names(self):
        with pytest.raises(ValueError, match="unknown DGP"):
            dgps.sample_dgp("nope")
        with pytest.raises(ValueError) as exc:
            dgps.validate(dgps=("nope",))
        for name in dgps.KNOWN_DGPS:
            assert name in str(exc.value)

    def test_split_shapes_and_columns(self):
        s = dgps.sample_dgp("heteroscedastic", n_train=50, n_test=25, p=6, seed=3)
        assert list(s.X_train.columns) == [f"x{i}" for i in range(6)]
        assert s.X_train.shape == (50, 6) and s.y_train.shape == (50,)
        assert s.X_test.shape == (25, 6) and s.y_test.shape == (25,)
        assert s.X_iid.shape == (25, 6) and s.y_iid.shape == (25,)
        assert np.isfinite(s.y_train).all()

    def test_shift_test_distribution_is_actually_shifted(self):
        s = dgps.sample_dgp("shift", n_train=400, n_test=1000, seed=0)
        # Train features centered at 0; test features centered at +2 on every column.
        assert abs(s.X_train.to_numpy().mean(axis=0).max()) < 0.15
        shift_means = s.X_test.to_numpy().mean(axis=0)
        assert np.all(np.abs(shift_means - dgps._SHIFT_MU) < 0.15)
        # In-distribution reference stays at the training distribution.
        assert abs(s.X_iid.to_numpy().mean(axis=0).max()) < 0.15


class TestGroundTruthStructure:
    def test_correlated_empirical_pairwise_correlations(self):
        s = dgps.sample_dgp("correlated", n_train=4000, n_test=100, seed=5)
        X = s.X_train.to_numpy()
        corr = np.corrcoef(X.T)
        assert abs(corr[0, 1] - 0.9) < 0.08
        assert abs(corr[3, 4] - 0.9) < 0.08
        # Off-block pairs stay near independence at this n (loose band).
        assert abs(corr[0, 3]) < 0.12

    def test_interactions_structure_recovery(self):
        s = dgps.sample_dgp("interactions", n_train=2000, n_test=500, seed=9)
        checks = dgps._ground_truth_checks(s)
        # The expanded design recovers gamma = 1.5 closely at this sample size.
        assert abs(checks["interaction_coef_abs_error"]) < 0.15
        # A plain linear model cannot represent the interaction: its test RMSE
        # must exceed that of the correctly-expanded one (descriptive, but a
        # structural gap of O(1) here — keep a generous margin).
        assert checks["rmse_plain_ridge_test"] > checks["rmse_expanded_ridge_test"] * 1.25

    def test_heteroscedastic_residuals_track_predicted_variance(self):
        s = dgps.sample_dgp("heteroscedastic", n_train=4000, n_test=2000, seed=2)
        checks = dgps._ground_truth_checks(s)
        corr = checks["corr_squared_residuals_vs_predicted_variance"]
        # Squared residuals of a linear fit must visibly track sigma(x)^2.
        assert 0.3 < corr <= 1.0

    def test_correlated_coef_deviation_reported(self):
        s = dgps.sample_dgp("correlated", n_train=2000, n_test=100, seed=4)
        checks = dgps._ground_truth_checks(s)
        assert 0.0 < checks["coef_mean_abs_deviation"] < 0.5
        assert checks["coef_max_abs_deviation"] >= checks["coef_mean_abs_deviation"]


class TestValidate:
    def test_shift_degradation_for_linear_model(self):
        s = dgps.sample_dgp("shift", n_train=400, n_test=800, seed=7)
        model = dgps._make_method("ridge", 7).fit(s.X_train.to_numpy(), s.y_train)
        rmse_test = dgps._rmse(s.y_test, model.predict(s.X_test.to_numpy()))
        rmse_iid = dgps._rmse(s.y_iid, model.predict(s.X_iid.to_numpy()))
        # The missing x2**2 curvature degrades a linear fit under the shift.
        assert (rmse_test / rmse_iid) >= 1.3

    def test_report_structure_and_json_output(self, tmp_path):
        report = dgps.validate(
            dgps=("interactions", "shift"),
            methods=("ridge",),
            n_train=200,
            n_test=100,
            seeds=(0,),
            out_dir=tmp_path,
        )
        assert report["schema"] == dgps.REPORT_SCHEMA_VERSION
        for name in ("interactions", "shift"):
            entry = report["results"][name]
            row = entry["per_seed"]["0"]
            assert row["truth"]["kind"] in {"interactions", "shift"}
            m = row["methods"]["ridge"]
            assert np.isfinite(m["rmse_test"]) and np.isfinite(m["degradation_ratio"])
        # Ground-truth probes present where defined.
        assert "interaction_coef_abs_error" in report["results"]["interactions"]["per_seed"]["0"]["ground_truth_checks"]

        written = json.loads((tmp_path / "dgp_validation.json").read_text(encoding="utf-8"))
        assert written["schema"] == dgps.REPORT_SCHEMA_VERSION
        assert set(written["seeds"]) == {0}
        assert written["methods"]["ridge"] == "Ridge"

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="unknown method"):
            dgps.validate(dgps=("shift",), methods=("nope",))

    def test_default_panel_instantiable(self):
        for name in dgps.DEFAULT_METHODS:
            est = dgps._make_method(name, seed=0)
            assert est is not None


class TestCli:
    def test_main_writes_report_and_returns_zero(self, tmp_path, capsys):
        rc = dgps.main(
            [
                "--dgps", "correlated",
                "--methods", "ridge",
                "--n-train", "150",
                "--n-test", "80",
                "--seeds", "0",
                "--out", str(tmp_path),
            ]
        )
        assert rc == 0
        written = json.loads((tmp_path / "dgp_validation.json").read_text(encoding="utf-8"))
        assert written["schema"] == dgps.REPORT_SCHEMA_VERSION
        out = capsys.readouterr().out
        assert "correlated" in out and "ridge" in out
