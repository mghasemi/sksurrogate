"""Executable form of the search contract (docs/search-contract.rst).

Property-based tests over the *state machine* rather than over one fixture: whatever
budget/dimension/min_evals combination the strategy produces, the counters, the
termination reason and the evaluation accounting must stay consistent.

Reference: docs/search-contract.rst invariants I1-I6 and I10.
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from SKSurrogate.structsearch import SurrogateSearch

REASONS = {
    "n_iter",
    "max_evals",
    "time_limit",
    "max_iter_no_progress",
    "acquisition_failed",
    "stopped",
}


def quadratic(point):
    return float(sum((component - 0.25) ** 2 for component in point))


def make_search(dim, budget, **kwargs):
    search = SurrogateSearch(
        quadratic,
        x0=[0.0] * dim,
        bounds=[(-1.0, 1.0)] * dim,
        max_iter=budget,
        min_evals=kwargs.pop("min_evals", 2),
        regressor=kwargs.pop("regressor", None),
        task_name=None,  # no checkpoint files in a property test
        random_state=0,
        **kwargs,
    )
    return search


SMALL = settings(
    max_examples=25,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@SMALL
@given(budget=st.integers(min_value=1, max_value=10), dim=st.integers(min_value=1, max_value=3))
def test_i4_budget_is_spent_exactly(budget, dim):
    """I4: with max_evals unset, n_evaluations equals n_iter exactly."""
    search = make_search(dim, budget)
    search()
    assert len(search.evaluated) == budget
    assert len(search.sampling_sources_) == budget
    assert search.summary_["budget"]["evaluations"] == budget
    assert search.termination_reason == "n_iter"


@SMALL
@given(
    budget=st.integers(min_value=2, max_value=10),
    cap=st.integers(min_value=1, max_value=9),
)
def test_i3_max_evals_is_a_hard_cap(budget, cap):
    """I3: n_evaluations never exceeds max_evals, and the cap is the stated reason."""
    search = make_search(2, budget, max_evals=cap)
    search()
    assert len(search.evaluated) == min(cap, budget)
    assert len(search.evaluated) <= cap
    assert search.termination_reason in ("max_evals", "n_iter")
    if cap < budget:
        assert search.termination_reason == "max_evals"


@SMALL
@given(budget=st.integers(min_value=1, max_value=8))
def test_i6_termination_reason_is_in_the_documented_vocabulary(budget):
    """I6: the reason names the constraint that bound the run."""
    search = make_search(2, budget)
    search()
    assert search.termination_reason in REASONS


@SMALL
@given(budget=st.integers(min_value=1, max_value=8), dim=st.integers(min_value=1, max_value=3))
def test_i1_i2_one_objective_call_per_recorded_evaluation(budget, dim):
    """I1/I2: every recorded evaluation is exactly one objective invocation."""
    calls = []

    def counted(point):
        calls.append(tuple(np.asarray(point).tolist()))
        return quadratic(point)

    search = SurrogateSearch(
        counted,
        x0=[0.0] * dim,
        bounds=[(-1.0, 1.0)] * dim,
        max_iter=budget,
        min_evals=2,
        task_name=None,
        random_state=0,
    )
    search()
    assert len(calls) == len(search.evaluated)
    assert len(search.sampling_sources_) == len(search.evaluated)


@SMALL
@given(budget=st.integers(min_value=2, max_value=8))
def test_i5_a_round_without_an_evaluation_is_charged_to_no_progress(budget):
    """I5: rounds that produce no evaluation still advance the no-progress counter."""
    search = make_search(2, budget)
    search()
    evaluations = len(search.evaluated)
    # Every round after the initial candidate either evaluated (one entry) or was
    # charged to NumIterNoProg; so the shortfall is bounded by the counter.
    shortfall = search.n_rounds - (evaluations - 1)
    assert shortfall >= 0
    assert search.NumIterNoProg >= shortfall


def test_acquisition_failure_spends_no_evaluation(monkeypatch):
    """A round whose local optimizer fails costs a round, never an evaluation."""
    from types import SimpleNamespace

    search = make_search(2, 5, max_itr_no_prog=10 ** 9)
    monkeypatch.setattr(
        search,
        "_SurrogateSearch__round_optim",
        lambda *args, **kwargs: SimpleNamespace(success=False, x=None, status=1, fun=None),
    )

    search()

    # With min_evals=2 the first round still explores (one point known), then every
    # acquisition fails: two evaluations, then nothing but failed rounds.
    assert len(search.evaluated) == 2
    assert search.termination_reason == "acquisition_failed"
    assert search.surrogate_info_["local_optimization_failures"] >= 1
    # Every failed round was charged to no-progress rather than to the budget: the
    # non-evaluating rounds are exactly the shortfall between rounds and evaluations.
    non_evaluating_rounds = search.n_rounds - (len(search.evaluated) - 1)
    assert search.NumIterNoProg == non_evaluating_rounds


def test_min_evals_above_budget_warns_and_never_fires():
    """A surface that cannot fire is reported, not silently ignored."""
    from sklearn.datasets import make_regression
    from sklearn.linear_model import Ridge

    from SKSurrogate.structsearch import SurrogateRandomCV

    X, y = make_regression(n_samples=60, n_features=2, noise=1.0, random_state=0)
    search = SurrogateRandomCV(
        Ridge(),
        param_distributions={"alpha": (0.01, 10.0)},
        cv=2,
        n_jobs=1,
        max_iter=3,
        min_evals=4,
        random_state=0,
    )
    with pytest.warns(UserWarning, match="can never fire"):
        search.fit(X, y)
    assert search.surrogate_info_["min_evals"] == 4
    assert search.surrogate_info_["min_evals_source"] == "user"
    assert len(search.evaluation_history_) == 3
    assert search.n_surrogate_fits_ == 0
