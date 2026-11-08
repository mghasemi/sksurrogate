"""Tests for benchmarks/design_readiness.py — D4 readiness assessment (R7).

Covers strategy validation, equal-budget record completeness, seeded
determinism, observation-noise recording, and the CLI report contract. These are
structural pins on synthetic evidence; they make no performance claims.
"""

from __future__ import annotations

import json

import pytest

from benchmarks import design_readiness


def _tiny(**overrides):
    params: dict = dict(
        families=("trig",),
        strategies=("random", "lcb", "ei", "pi", "lcb_noise_aware"),
        seeds=1,
        budget=6,
        initial=3,
        n_candidates=32,
        noise_levels=(0.0,),
    )
    params.update(overrides)
    return design_readiness.run_assessment(**params)


class TestValidation:
    def test_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="unknown strategy"):
            design_readiness.run_assessment(families=("trig",), strategies=("ucb",))
        with pytest.raises(ValueError, match="unknown strategy"):
            design_readiness.run_assessment(
                families=("trig",), strategies=("ei", "mystery")
            )

    def test_empty_strategy_list_is_rejected(self):
        with pytest.raises(ValueError, match="at least one strategy"):
            design_readiness.run_assessment(families=("trig",), strategies=())

    def test_budget_guards(self):
        with pytest.raises(ValueError, match="initial must be smaller"):
            _tiny(initial=6, budget=6)
        with pytest.raises(ValueError, match="seeds must be"):
            _tiny(seeds=0)
        with pytest.raises(ValueError, match="noise levels must be"):
            _tiny(noise_levels=(-0.1,))


class TestReport:
    def test_equal_budget_records_and_schema(self):
        report = _tiny()
        assert report["schema"] == design_readiness.REPORT_SCHEMA_VERSION
        strategies = report["protocol"]["strategies"]
        budget = report["protocol"]["budget"]
        assert len(report["run_summaries"]) == len(strategies)
        # Every strategy receives exactly `budget` evaluations (initial + sequential).
        assert len(report["records"]) == len(strategies) * budget
        for strategy in strategies:
            rows = [row for row in report["records"] if row["strategy"] == strategy]
            assert [row["iteration"] for row in rows] == list(range(budget))
            assert rows[0]["phase"] == "initial" and rows[-1]["phase"] == "sequential"
            assert all(
                -1.0 <= value <= 1.0 for row in rows for value in row["x"]
            ), (strategy, rows[0]["x"])
        for summary in report["run_summaries"]:
            assert summary["final_regret"] >= 0.0
            assert summary["regret_auc"] >= 0.0
            assert summary["runs"] if "runs" in summary else True
        assert report["paired_vs_lcb"]

    def test_observation_noise_is_recorded_and_changes_y(self):
        report = _tiny(noise_levels=(0.0, 0.05))
        assert {row["noise_level"] for row in report["records"]} == {0.0, 0.05}
        clean = [row for row in report["records"] if row["noise_level"] == 0.0]
        noisy = [row for row in report["records"] if row["noise_level"] == 0.05]
        assert all(row["y_observed"] == row["y_true"] for row in clean)
        assert any(row["y_observed"] != row["y_true"] for row in noisy)

    def test_seeded_determinism(self):
        first = _tiny()
        second = _tiny()
        assert first["summaries"] == second["summaries"]
        assert first["records"] == second["records"]

    def test_noise_aware_strategy_metadata(self):
        report = _tiny(strategies=("lcb_noise_aware",), noise_levels=(0.05,))
        sequential = [row for row in report["records"] if row["phase"] == "sequential"]
        assert sequential
        assert all(
            row["acquisition"] == "lower_confidence_bound_noise_aware"
            for row in sequential
        )

    def test_random_baseline_is_uniform_metadata(self):
        report = _tiny(strategies=("random",))
        sequential = [row for row in report["records"] if row["phase"] == "sequential"]
        assert all(row["acquisition"] == "uniform_random" for row in sequential)
        assert "acquisition_value" not in sequential[0]


class TestCli:
    def test_main_writes_versioned_report(self, tmp_path, capsys):
        code = design_readiness.main(
            [
                "--families",
                "trig",
                "--strategies",
                "random",
                "lcb",
                "ei",
                "--seeds",
                "1",
                "--budget",
                "5",
                "--initial",
                "3",
                "--candidates",
                "32",
                "--noise",
                "0",
                "--out",
                str(tmp_path),
            ]
        )
        assert code == 0
        written = json.loads(
            (tmp_path / "design_readiness.json").read_text(encoding="utf-8")
        )
        assert written["schema"] == design_readiness.REPORT_SCHEMA_VERSION
        assert written["protocol"]["budget"] == 5
        output = capsys.readouterr().out
        assert "strategy" in output and "trig" in output
