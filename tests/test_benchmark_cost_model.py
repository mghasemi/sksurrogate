"""Cost-model arithmetic pins (no timing; timing is measured by the CLI)."""

from benchmarks.cost_model import estimate_matrix, render_markdown


def _surrogates():
    return {
        3: {"fit_s": 0.01, "acquisition_s": 0.02, "n_points": 30, "dim": 3},
        10: {"fit_s": 0.02, "acquisition_s": 0.05, "n_points": 30, "dim": 10},
        30: {"fit_s": 0.03, "acquisition_s": 0.10, "n_points": 30, "dim": 30},
    }


def test_estimate_scales_with_evaluation_cost_and_budget():
    cheap = estimate_matrix(
        {("ionosphere", "rf"): 0.1, ("ionosphere", "svc"): 0.01},
        _surrogates(),
        datasets=["ionosphere"],
        budgets=(30,),
        seeds=2,
        arms=6,
        surrogate_arms=3,
    )
    expensive = estimate_matrix(
        {("ionosphere", "rf"): 0.2, ("ionosphere", "svc"): 0.01},
        _surrogates(),
        datasets=["ionosphere"],
        budgets=(30,),
        seeds=2,
        arms=6,
        surrogate_arms=3,
    )
    assert expensive["real_matrix_s"] > cheap["real_matrix_s"]
    assert cheap["total_s"] > cheap["real_matrix_s"]
    assert {row["family"] for row in cheap["rows"]} == {"rf", "svc"}


def test_high_dimensional_block_only_counts_measured_dimensions():
    one_dim = estimate_matrix(
        {("ionosphere", "rf"): 0.1},
        {3: _surrogates()[3]},
        datasets=["ionosphere"],
        budgets=(30,),
        seeds=1,
    )
    three_dims = estimate_matrix(
        {("ionosphere", "rf"): 0.1},
        _surrogates(),
        datasets=["ionosphere"],
        budgets=(30,),
        seeds=1,
    )
    assert one_dim["high_dim_s"] == 0.0
    assert three_dims["high_dim_s"] > 0.0


def test_rendered_document_carries_the_gate_and_measured_tables():
    evaluation = {("ionosphere", "rf"): 0.1}
    estimate = estimate_matrix(
        evaluation, _surrogates(), datasets=["ionosphere"], budgets=(30,), seeds=1
    )
    document = render_markdown(evaluation, _surrogates(), estimate, ["ionosphere"])
    assert "Benchmark v2.0 cost model" in document
    assert "Decision: _(pending)_" in document
    assert "seconds / evaluation" in document
    assert "total (serial)" in document
