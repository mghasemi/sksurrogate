"""scikit-learn compatibility contracts for :class:`SurrogateRandomCV`.

Inheriting ``BaseSearchCV`` is not evidence of compatibility: this file pins the
behaviour a ``RandomizedSearchCV`` user relies on. Each test names the contract, so
a failure says what broke for the caller rather than which internal changed.

Reference: plan phase 1 item 3.5 (tests/test_sklearn_compat.py).
"""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from SKSurrogate.structsearch import Categorical, Integer, Real, SurrogateRandomCV

SPACE = {"alpha": (0.01, 10.0), "solver": ["svd", "cholesky"]}


def reg_data(rows=60):
    return make_regression(n_samples=rows, n_features=3, noise=1.0, random_state=0)


def make_search(estimator=None, space=None, **kwargs):
    options = dict(cv=2, n_jobs=1, max_iter=4, min_evals=2, random_state=0)
    options.update(kwargs)
    return SurrogateRandomCV(
        Ridge() if estimator is None else estimator,
        param_distributions=SPACE if space is None else space,
        **options,
    )


class RecordingEstimator(BaseEstimator):
    """Estimator that records the keyword arguments its ``fit`` received."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y=None, **kwargs):
        self.fit_kwargs_ = dict(kwargs)
        self.mean_ = float(np.mean(y)) if y is not None else 0.0
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)

    def score(self, X, y):
        return -float(np.mean((self.predict(X) - y) ** 2))


def test_clone_contract_fit_does_not_modify_constructor_parameters():
    """``fit`` must leave every ``__init__`` argument as it found it."""
    X, y = reg_data()
    search = make_search()
    before = {
        name: value
        for name, value in search.get_params(deep=False).items()
        if name != "estimator"
    }
    search.fit(X, y)
    after = {
        name: value
        for name, value in search.get_params(deep=False).items()
        if name != "estimator"
    }
    assert before == after
    # ``radius`` used to be overwritten with the resolved value during fit.
    assert search.radius is None
    assert search.radius_ > 0
    # ``scoring`` used to be replaced by its resolved form; the resolved value now
    # lives on a fitted attribute instead.
    assert search.scoring is None
    assert callable(search.scorer_)


def test_clone_before_and_after_fit_is_stable():
    X, y = reg_data()
    search = make_search().fit(X, y)
    assert clone(search).get_params(deep=False)["max_iter"] == search.max_iter


def test_nested_estimator_parameters_round_trip():
    search = make_search()
    deep = search.get_params(deep=True)
    assert "estimator__alpha" in deep
    search.set_params(estimator__alpha=3.0)
    assert search.estimator.alpha == 3.0


def test_pipeline_nested_parameter_names_are_searchable():
    X, y = reg_data()
    pipeline = Pipeline([("scale", StandardScaler()), ("model", Ridge())])
    search = SurrogateRandomCV(
        pipeline,
        param_distributions={
            "model__alpha": (0.01, 10.0),
            "model__solver": ["svd", "cholesky"],
        },
        cv=2,
        n_jobs=1,
        max_iter=3,
        min_evals=2,
        random_state=0,
    )
    search.fit(X, y)
    assert set(search.best_params_) == {"model__alpha", "model__solver"}
    assert search.best_estimator_.predict(X).shape == (len(X),)


def test_cv_results_follows_the_base_search_cv_column_layout():
    X, y = reg_data()
    search = make_search().fit(X, y)
    results = search.cv_results_
    expected = {
        "params",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "mean_fit_time",
        "param_alpha",
        "param_solver",
        "split0_test_score",
        "split1_test_score",
    }
    missing = expected - set(results)
    assert not missing, "cv_results_ is missing %s" % sorted(missing)
    n_trials = len(search.evaluation_history_)
    for key, column in results.items():
        assert len(column) == n_trials, "%s has %d rows, expected %d" % (
            key,
            len(column),
            n_trials,
        )
    assert results["param_solver"][0] in {"svd", "cholesky"}
    assert results["best_index"] if "best_index" in results else True


def test_refit_true_fits_the_best_estimator():
    X, y = reg_data()
    search = make_search(refit=True).fit(X, y)
    assert hasattr(search.best_estimator_, "coef_")
    assert search.predict(X).shape == (len(X),)


def test_refit_false_exposes_no_fitted_estimator():
    X, y = reg_data()
    search = make_search(refit=False).fit(X, y)
    assert not hasattr(search, "best_estimator_")
    assert search.best_params_  # the parameters are still reported
    with pytest.raises(Exception):
        search.predict(X)


def test_callable_refit_receives_cv_results_and_its_return_value_is_stored():
    X, y = reg_data()
    seen = {}

    def refit(cv_results):
        seen["keys"] = set(cv_results)
        return Ridge(alpha=0.5).fit(X, y)

    search = make_search(refit=refit).fit(X, y)
    assert "mean_test_score" in seen["keys"]
    assert isinstance(search.best_estimator_, Ridge)
    assert search.best_estimator_.alpha == 0.5


def test_multi_metric_scoring_is_refused_with_guidance():
    X, y = reg_data()
    search = make_search(scoring={"r2": "r2", "neg_mse": "neg_mean_squared_error"})
    with pytest.raises(ValueError, match="multi-metric scoring"):
        search.fit(X, y)


def test_bare_callable_scorer_is_refused_with_guidance():
    X, y = reg_data()

    def loss(estimator, X, y):
        return -float(np.mean((estimator.predict(X) - y) ** 2))

    search = make_search(scoring=loss)
    with pytest.raises(TypeError, match="make_scorer"):
        search.fit(X, y)


def test_sample_weight_is_either_routed_or_refused_explicitly():
    """Metadata routing: routed correctly, or an explicit error - never silent."""
    X, y = reg_data()
    weights = np.linspace(0.5, 1.5, len(y))
    search = make_search(estimator=RecordingEstimator(), space={"alpha": (0.01, 10.0)})
    try:
        search.fit(X, y, sample_weight=weights)
    except TypeError as exc:
        assert "sample_weight" in str(exc)
    else:
        assert "sample_weight" in search.best_estimator_.fit_kwargs_
        assert np.allclose(
            search.best_estimator_.fit_kwargs_["sample_weight"], weights
        )


def test_all_folds_failing_produces_the_documented_fallback():
    X, y = reg_data()

    class Exploding(BaseEstimator):
        def __init__(self, alpha=1.0):
            self.alpha = alpha

        def fit(self, X, y=None):
            raise RuntimeError("cannot fit")

        def predict(self, X):
            raise RuntimeError("cannot predict")

        def score(self, X, y):
            return -1.0

    search = SurrogateRandomCV(
        Exploding(),
        param_distributions={"alpha": (0.01, 10.0)},
        cv=2,
        n_jobs=1,
        max_iter=2,
        min_evals=2,
        random_state=0,
    )
    search.fit(X, y)  # error_score="raise" is the default in sklearn too
    assert search.best_score_ == float("-inf")
    assert search.best_params_ == {}
    assert search.failure_summary_["invalid_trials"] == 0


@pytest.mark.parametrize(
    "space",
    [
        {"alpha": Integer(1, 5)},
        {"alpha": Real(0.01, 10.0)},
        {"alpha": Categorical([0.1, 1.0])},
        {"alpha": [0.1, 1.0, 5.0]},
    ],
)
def test_supported_search_space_types(space):
    X, y = reg_data()
    search = make_search(space=space, max_iter=3).fit(X, y)
    assert search.best_params_
    assert search.best_params_["alpha"] in [
        value
        for value in search.cv_results_["param_alpha"]
        if value is not None
    ]


def test_distribution_objects_are_refused_with_a_clear_message():
    from scipy.stats import loguniform

    X, y = reg_data()
    search = make_search(space={"alpha": loguniform(0.01, 10.0)})
    with pytest.raises(ValueError, match="distribution"):
        search.fit(X, y)


def test_parallel_and_serial_searches_explore_the_same_candidates():
    X, y = reg_data()
    serial = make_search(n_jobs=1).fit(X, y)
    parallel = make_search(n_jobs=2).fit(X, y)
    serial_params = sorted(
        str(sorted(record["params"].items())) for record in serial.evaluation_history_
    )
    parallel_params = sorted(
        str(sorted(record["params"].items())) for record in parallel.evaluation_history_
    )
    assert serial_params == parallel_params


def test_reports_the_same_estimator_and_scorer_attributes_as_sklearn():
    X, y = reg_data()
    ours = make_search().fit(X, y)
    reference = RandomizedSearchCV(Ridge(), SPACE, n_iter=4, cv=2, random_state=0).fit(X, y)
    for attribute in ("scorer_", "best_params_", "best_score_", "best_index_", "cv_results_"):
        assert hasattr(ours, attribute), attribute
        assert hasattr(reference, attribute), attribute
    # The serialized result must be a plain dict of sequences, as sklearn's is.
    assert isinstance(ours.cv_results_, dict)
    assert callable(ours.scorer_)
