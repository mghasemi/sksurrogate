import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from SKSurrogate.structsearch import Categorical, SurrogateRandomCV, SurrogateSearch


class FailingRegressor:
    def fit(self, X, y):
        raise RuntimeError("intentional surrogate fit failure")

    def predict(self, X):
        return np.zeros(len(X))


class NonFiniteRegressor:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), np.nan)


def test_surrogate_search_reports_selected_custom_regressor_and_fit_history(tmp_path):
    search = SurrogateSearch(
        lambda point: (point[0] - 0.25) ** 2,
        x0=[0.0],
        bounds=[(-1.0, 1.0)],
        max_iter=3,
        min_evals=1,
        max_evals=3,
        regressor=LinearRegression(),
        task_name=str(tmp_path / "diagnostics"),
        random_state=0,
    )

    search()

    assert search.surrogate_info_["source"] == "user"
    assert search.surrogate_info_["regressor"] == "sklearn.linear_model._base.LinearRegression"
    assert search.surrogate_info_["automatic_fallback"] is None
    assert search.surrogate_info_["fit_attempts"] >= 1
    assert search.surrogate_fit_history_[0]["fit_status"] == "complete"
    assert search.summary_["surrogate"] is search.surrogate_info_


def test_surrogate_fit_error_is_recorded_and_propagated_without_fallback(tmp_path):
    search = SurrogateSearch(
        lambda point: point[0] ** 2,
        x0=[0.0],
        bounds=[(-1.0, 1.0)],
        max_iter=3,
        min_evals=1,
        max_evals=3,
        regressor=FailingRegressor(),
        task_name=str(tmp_path / "failed-fit"),
        random_state=0,
    )

    with pytest.raises(RuntimeError, match="intentional surrogate fit failure"):
        search()

    assert search.surrogate_info_["fit_attempts"] == 1
    assert search.surrogate_info_["fit_failures"] == 1
    assert search.surrogate_info_["automatic_fallback"] is None
    assert search.surrogate_fit_history_[0]["failure_stage"] == "fit"
    assert search.surrogate_fit_history_[0]["failure_type"] == "RuntimeError"


def test_non_finite_surrogate_prediction_is_recorded_and_propagated(tmp_path):
    search = SurrogateSearch(
        lambda point: point[0] ** 2,
        x0=[0.0],
        bounds=[(-1.0, 1.0)],
        max_iter=3,
        min_evals=1,
        max_evals=3,
        regressor=NonFiniteRegressor(),
        task_name=str(tmp_path / "non-finite-prediction"),
        random_state=0,
    )

    with pytest.raises(ValueError, match="exactly one finite value"):
        search()

    assert search.surrogate_info_["prediction_failures"] == 1
    assert search.surrogate_fit_history_[0]["failure_stage"] == "predict"
    assert search.surrogate_fit_history_[0]["failure_type"] == "ValueError"


def test_surrogate_random_cv_exposes_default_regressor_diagnostics(tmp_path):
    search = SurrogateRandomCV(
        LogisticRegression(max_iter=100),
        {"C": Categorical([0.1, 1.0])},
        cv=2,
        n_jobs=1,
        max_iter=1,
        min_evals=1,
        max_evals=1,
        refit=False,
        warm_start=False,
        Continue=False,
        task_name=str(tmp_path / "default"),
    )

    search.fit(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1]))

    assert search.surrogate_info_["source"] == "default"
    assert search.surrogate_info_["regressor"] == (
        "sklearn.gaussian_process._gpr.GaussianProcessRegressor"
    )
    assert search.surrogate_info_["automatic_fallback"] is None
    assert search.summary_["surrogate"] is search.surrogate_info_
