"""I7: one ``random_state`` reproduces one search, in-process and across processes.

The pre-0.6 code drew its initial candidate from Python's *global* ``random``, so
two fits with ``random_state=0`` diverged and nothing downstream (a benchmark cell,
an experiment record, a bug report) could be reproduced. These tests pin the fix.

Reference: docs/search-contract.rst invariants I7 and I8.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from SKSurrogate.structsearch import SurrogateRandomCV, make_rng_streams

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE = {"max_depth": [3, 5, 7, 9], "min_samples_leaf": [0.01, 0.05, 0.1]}

#: Runs one small search and prints its candidate sequence as JSON. Executed in a
#: child process so the check cannot be satisfied by in-process state.
_CHILD = """
import json, os, sys
sys.path.insert(0, os.getcwd())
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from SKSurrogate.structsearch import SurrogateRandomCV

X, y = make_classification(
    n_samples=120, n_features=6, n_informative=3, random_state=0
)
search = SurrogateRandomCV(
    RandomForestClassifier(n_estimators=5, random_state=0),
    param_distributions=%r,
    cv=2,
    n_jobs=1,
    max_iter=5,
    min_evals=3,
    random_state=0,
)
search.fit(X, y)
print(json.dumps([[record["params"], record["score"]] for record
                  in search.evaluation_history_], sort_keys=True))
""" % (SPACE,)


def sequence(search):
    return [(record["params"], record["score"]) for record in search.evaluation_history_]


def fit_inline(cv=2):
    X, y = make_classification(
        n_samples=120, n_features=6, n_informative=3, random_state=0
    )
    search = SurrogateRandomCV(
        RandomForestClassifier(n_estimators=5, random_state=0),
        param_distributions=SPACE,
        cv=cv,
        n_jobs=1,
        max_iter=5,
        min_evals=3,
        random_state=0,
    )
    search.fit(X, y)
    return search


def test_same_seed_reproduces_the_candidate_sequence_in_process():
    assert sequence(fit_inline()) == sequence(fit_inline())


def test_different_seed_changes_the_candidate_sequence():
    first = fit_inline()
    X, y = make_classification(
        n_samples=120, n_features=6, n_informative=3, random_state=0
    )
    other = SurrogateRandomCV(
        RandomForestClassifier(n_estimators=5, random_state=0),
        param_distributions=SPACE,
        cv=2,
        n_jobs=1,
        max_iter=5,
        min_evals=3,
        random_state=1,
    )
    other.fit(X, y)
    assert sequence(first) != sequence(other)


def test_same_seed_reproduces_across_processes():
    env = dict(os.environ)
    # The repo root must be importable in the child: the shared venv's editable
    # install points at the main checkout, so a bare child would measure other code.
    env["PYTHONPATH"] = REPO_ROOT
    env.pop("PYTHONPATH_LEAK", None)
    runs = [
        subprocess.run(
            [sys.executable, "-c", _CHILD],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    payload = json.loads(runs[0])
    assert len(payload) == 5


def test_fitting_does_not_consume_global_random_state():
    """I8: no global ``random``/``numpy.random`` state is read or written."""
    import random as pyrandom

    pyrandom.seed(1234)
    np.random.seed(1234)
    python_before = pyrandom.getstate()
    numpy_before = np.random.get_state()[1].copy()

    fit_inline()

    assert pyrandom.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before)


def test_streams_are_independent_and_reproducible():
    first = make_rng_streams(0)
    second = make_rng_streams(0)
    assert set(first) == {"init_candidate", "sampling"}
    assert np.array_equal(
        first["init_candidate"].uniform(0, 1, 5), second["init_candidate"].uniform(0, 1, 5)
    )
    assert np.array_equal(
        first["sampling"].uniform(0, 1, 5), second["sampling"].uniform(0, 1, 5)
    )
    # The two streams are different sequences: the starting point and the sampling
    # path must not be able to perturb each other.
    assert not np.array_equal(
        make_rng_streams(0)["init_candidate"].uniform(0, 1, 5),
        make_rng_streams(0)["sampling"].uniform(0, 1, 5),
    )


@pytest.mark.parametrize("random_state", [0, 7, None, np.random.default_rng(3), np.random.RandomState(3)])
def test_random_state_accepts_the_scikit_learn_types(random_state):
    streams = make_rng_streams(random_state)
    assert all(hasattr(stream, "uniform") for stream in streams.values())


def test_random_state_rejects_unsupported_types():
    with pytest.raises(TypeError, match="random_state must be"):
        make_rng_streams("not-a-seed")
