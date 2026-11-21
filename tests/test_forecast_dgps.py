"""Phase P8.3 tests: forecast DGP benchmark suite.

The suite's job is calibration, so the tests pin the two claims the plan makes:
the naive baseline wins on the deliberately easy DGP, and the report carries a
schema version plus an explicit expectation verdict a future regression can
read. The remaining DGPs are exercised for shape (panel membership, rankings),
not for a particular winner — their numbers are descriptive.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.forecast_dgps import (
    KNOWN_DGPS,
    REPORT_SCHEMA_VERSION,
    evaluate,
    evaluate_all,
)


class TestForecastDGPs:
    def test_naive_wins_on_the_easy_dgp(self):
        record = evaluate("easy_naive")
        assert record["naive_wins"] is True
        model_mae = record["methods"]["model"]["mae"]
        naive_mae = record["methods"]["naive"]["mae"]
        assert naive_mae < model_mae
        # With 40 origins, naive MASE is estimated near its theoretical 1.0
        # (its errors and the in-sample scale come from the same distribution).
        assert 0.6 <= record["methods"]["naive"]["mase"] <= 1.6

    def test_ranking_is_consistent_with_the_metrics(self):
        record = evaluate("easy_naive")
        ranking = record["method_mae_ranking"]
        assert record["best_method"] == ranking[0]
        maes = [record["methods"][name]["mae"] for name in ranking]
        assert maes == sorted(maes)

    def test_panel_dgp_reports_every_series(self):
        record = evaluate("panel_5series", n_splits=6)
        assert record["series"] == [f"s{index}" for index in range(5)]
        # One record per (series, origin).
        assert record["origins"] == 5 * 6

    def test_unknown_dgp_is_refused(self):
        with pytest.raises(ValueError, match="unknown forecast DGP"):
            evaluate("no-such-process")

    def test_evaluate_all_writes_a_versioned_report(self, tmp_path):
        report = evaluate_all(
            dgps=["easy_naive", "seasonal_weekly"], n_splits=6, out_dir=tmp_path
        )
        assert report["schema_version"] == REPORT_SCHEMA_VERSION
        assert report["expectations"]["easy_naive_naive_wins"] is True
        path = tmp_path / "forecast_dgp_validation.json"
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert stored["schema_version"] == REPORT_SCHEMA_VERSION
        assert set(stored["results"]) == {"easy_naive", "seasonal_weekly"}
        # The payload is JSON-safe end to end (no numpy scalars leaking through).
        json.dumps(stored)

    def test_every_known_dgp_evaluates(self):
        report = evaluate_all(dgps=list(KNOWN_DGPS), n_splits=4)
        assert set(report["results"]) == set(KNOWN_DGPS)
        for name, record in report["results"].items():
            assert record["methods"]["model"]["mae"] is not None, name
            assert record["methods"]["naive"]["mae"] is not None, name
