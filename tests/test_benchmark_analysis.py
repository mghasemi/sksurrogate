"""Analysis-layer pins for the v2.0 paired statistics and wall-clock table.

`benchmarks.stats` is tested on hand-computed fixtures; this file tests the
analyzer wiring: a record's best-so-far score at a wall-clock budget, the wall
table it produces, and the paired table that collapses per-dataset rows into the
aggregate row every v2.0 claim must cite.
"""

import numpy as np
import pandas as pd
import pytest

from benchmarks.analyze_surrogates import (
    build_wall_table,
    paired_stats,
    wall_best,
)


def _record(method, dataset, seed, scores, wall_s):
    return {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "budget": 60,
        "final_test_accuracy": None,
        "records": [
            {"k": k, "cv_score": score, "wall_s": wall}
            for k, (score, wall) in enumerate(zip(scores, wall_s))
        ],
    }


def test_wall_best_respects_the_cumulative_candidate_time():
    record = _record(
        "random", "ionosphere", 0,
        scores=[0.50, 0.60, None, 0.55],
        wall_s=[1.0, 2.0, 3.0, 1.0],
    )
    assert wall_best(record, 1.5) == 0.50
    assert wall_best(record, 3.5) == 0.60
    assert wall_best(record, 100.0) == 0.60


def test_wall_best_is_undefined_without_per_evaluation_timing():
    record = _record("skopt", "ionosphere", 0, scores=[0.5], wall_s=[None])
    assert wall_best(record, 10.0) is None


def test_build_wall_table_keeps_every_seed():
    records = [
        _record("random", "ionosphere", seed, [0.5, 0.7], [1.0, 1.0])
        for seed in range(3)
    ]
    table = build_wall_table(records, 10.0)
    assert len(table) == 3
    assert set(table.columns) == {"dataset", "seed", "method", "wall_best_cv", "total_wall_s"}
    assert (table["wall_best_cv"] == 0.7).all()


def test_paired_stats_detects_a_constant_shift_and_reports_the_aggregate():
    rows = []
    for dataset in ("a", "b"):
        for seed in range(4):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "method": "sksurrogate-gp",
                    "final_cv_best": 0.80,
                }
            )
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "method": "random",
                    "final_cv_best": 0.70,
                }
            )
    table = pd.DataFrame(rows)
    paired = paired_stats(table, "final_cv_best", "sksurrogate-gp")
    aggregate = paired[paired["scope"] == "aggregate"].iloc[0]
    assert aggregate["method"] == "random"
    assert aggregate["n_pairs"] == 8
    assert aggregate["mean_delta"] == pytest.approx(-0.10)
    assert aggregate["ci_low"] == pytest.approx(-0.10)
    assert aggregate["ci_high"] == pytest.approx(-0.10)
    assert aggregate["wins"] == 0 and aggregate["losses"] == 8
    assert aggregate["cliffs_delta"] == pytest.approx(-1.0)
    assert bool(aggregate["distinguishable"]) is True
    assert "p_holm" in paired.columns
    # Two per-dataset rows plus one aggregate row.
    assert set(paired["scope"]) == {"a", "b", "aggregate"}


def test_paired_stats_returns_empty_when_the_baseline_is_absent():
    table = pd.DataFrame(
        [
            {"dataset": "a", "seed": 0, "method": "random", "final_cv_best": 0.7},
            {"dataset": "a", "seed": 1, "method": "random", "final_cv_best": 0.8},
        ]
    )
    assert paired_stats(table, "final_cv_best", "sksurrogate-gp").empty
