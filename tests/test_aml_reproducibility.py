"""The AML search must reproduce itself at a fixed ``random_state``.

Master derives every stream the search needs from one ``random_state``
(:func:`SKSurrogate.structsearch.make_rng_streams`: proposals, sampling, restarts), so
these are guard tests rather than the fix they started as: the search used to draw its
start point from the *global* ``random`` module, which the same seed did not control, so
identical runs produced different start points, different candidates and different scores,
and no reported metric was reproducible.
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression

from SKSurrogate import AML


def _search(seed=7, max_evals=3, n_samples=180):
    X, y = make_regression(n_samples=n_samples, n_features=6, noise=8.0, random_state=0)
    search = AML(
        preset="regression",
        scoring="neg_root_mean_squared_error",
        max_evals=max_evals,
        time_limit=600,
        cv=3,
        random_state=seed,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X, y)
    return search


def test_search_is_reproducible_at_a_fixed_seed():
    first = _search().to_audit_dict()["best_candidate"]
    second = _search().to_audit_dict()["best_candidate"]

    assert first["pipeline"] == second["pipeline"]
    assert first["score"] == pytest.approx(second["score"], rel=1e-12)


def test_repeated_searches_evaluate_identical_trials():
    first = [record["score"] for record in _search().evaluation_history_]
    second = [record["score"] for record in _search().evaluation_history_]

    assert len(first) == len(second)
    assert first == pytest.approx(second, rel=1e-12)


def test_selected_estimator_carries_the_seed():
    """An unseeded winner would make every later refit non-reproducible too."""
    search = _search()
    estimator = search.best_estimator_

    seeded = {
        name: value
        for name, value in estimator.get_params(deep=True).items()
        if name == "random_state" or name.endswith("__random_state")
    }
    assert seeded, "the selected estimator exposes no random_state"
    assert all(value is not None for value in seeded.values())
    assert set(seeded.values()) == {search.random_state}
