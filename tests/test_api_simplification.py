import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge

from SKSurrogate import AML, SurrogateRandomCV
from SKSurrogate.structsearch import Categorical, Integer, Real, as_space


def _data():
    rng = np.random.RandomState(0)
    X = rng.rand(40, 3)
    return X, X.sum(axis=1)


def test_sklearn_style_names_and_clone():
    X, y = _data()
    search = SurrogateRandomCV(
        Ridge(),
        param_distributions={"alpha": (0.1, 10.0), "solver": ["svd", "cholesky"]},
        n_iter=3,
        min_evals=2,
        n_jobs=1,
    )
    fitted = clone(search).fit(X, y)
    assert set(fitted.best_params_) == {"alpha", "solver"}
    assert fitted.predict(X[:2]).shape == (2,)


def test_legacy_names_still_work():
    X, y = _data()
    search = SurrogateRandomCV(Ridge(), {"alpha": (0.1, 10.0)}, max_iter=3, min_evals=2, n_jobs=1)
    assert search.fit(X, y).max_iter_ == 3


def test_conflicting_aliases_rejected():
    with pytest.raises(ValueError):
        SurrogateRandomCV(Ridge(), params={"alpha": (0.1, 1.0)}, param_distributions={"alpha": (0.1, 2.0)})
    with pytest.raises(ValueError):
        SurrogateRandomCV(Ridge())


def test_as_space_shorthand():
    assert isinstance(as_space((1, 5)), Integer)
    assert isinstance(as_space((0.1, 5)), Real)
    assert isinstance(as_space(["a", "b"]), Categorical)
    real = Real(0.0, 1.0)
    assert as_space(real) is real


@pytest.mark.parametrize("kwargs", [{}, {"preset": "classification"}, {"preset": "regression"}])
def test_aml_clone_and_repr(kwargs):
    aml = AML(**kwargs)
    skip = {"config", "surrogates"}  # hold objects without __eq__
    keep = lambda d: {k: v for k, v in d.items() if k not in skip}
    assert keep(clone(aml).get_params()) == keep(aml.get_params())
    assert repr(aml).startswith("AML(")


def test_sampling_string_shortcut():
    from SKSurrogate import BoxSample
    from SKSurrogate.structsearch import CompactSample, resolve_sampling

    assert resolve_sampling("box") is BoxSample
    assert resolve_sampling("Compact") is CompactSample
    assert resolve_sampling(BoxSample) is BoxSample
    with pytest.raises(ValueError):
        resolve_sampling("nope")
    X, y = _data()
    search = SurrogateRandomCV(
        Ridge(), param_distributions={"alpha": (0.1, 10.0)}, n_iter=3, min_evals=2, n_jobs=1, sampling="box"
    )
    assert "alpha" in search.fit(X, y).best_params_


def test_aml_fit_strategy_and_stacking():
    aml = AML(preset="classification", stacking={"cv": 3, "probs": False})
    opts = aml._stacking_options()
    assert opts["cv"] == 3 and opts["probs"] is False and opts["res"] is True
    with pytest.raises(ValueError):
        AML(preset="classification", stacking={"bogus": 1})._stacking_options()
    with pytest.raises(ValueError):
        aml.fit([[0.0], [1.0]], [0, 1], strategy="nope")
    assert clone(aml).get_params()["stacking"] == {"cv": 3, "probs": False}


def test_aml_fit_eoa_strategy_dispatches(monkeypatch):
    aml = AML(preset="classification")
    seen = {}
    monkeypatch.setattr(AML, "eoa_fit", lambda self, X, y, **kw: seen.update(kw) or self)
    assert aml.fit([[0.0]], [0], strategy="eoa", max_generations=2) is aml
    assert seen == {"max_generations": 2}


def test_bundle_save_load_methods(tmp_path):
    from SKSurrogate import ModelBundle

    X, y = _data()
    bundle = ModelBundle(Ridge().fit(X, y), task_name="t")
    path = bundle.save(tmp_path / "b.joblib")
    loaded = ModelBundle.load(path, strict_dependencies=False)
    assert loaded.task_name == "t"
    assert np.allclose(loaded.model.predict(X), bundle.model.predict(X))


def test_subgroup_precision_is_not_recall():
    from SKSurrogate import subgroup_performance_report

    y_true = [1, 0, 0, 0, 1, 1]
    y_pred = [1, 1, 1, 0, 1, 0]
    groups = ["a"] * 6
    report = subgroup_performance_report(y_true, y_pred, groups, metric="precision")
    assert report["groups"]["a"]["precision"] == pytest.approx(2 / 4)
    report = subgroup_performance_report(y_true, y_pred, groups, metric="recall")
    assert report["groups"]["a"]["recall"] == pytest.approx(2 / 3)


def test_best_score_follows_sklearn_sign():
    X, y = _data()
    search = SurrogateRandomCV(
        Ridge(), param_distributions={"alpha": (0.1, 10.0)}, n_iter=3, min_evals=2, n_jobs=1,
        scoring="neg_mean_squared_error",
    ).fit(X, y)
    assert search.best_score_ <= 0
    assert search.best_score_ == pytest.approx(-search.best_estimator_score)


def test_cv_results_sklearn_keys():
    X, y = _data()
    search = SurrogateRandomCV(
        Ridge(), param_distributions={"alpha": (0.1, 10.0)}, n_iter=4, min_evals=2, n_jobs=1,
        scoring="neg_mean_squared_error",
    ).fit(X, y)
    res = search.cv_results_
    assert max(res["mean_test_score"]) == pytest.approx(search.best_score_)
    assert res["rank_test_score"][search.best_index_] == 1
    assert res["mean_test_score"][search.best_index_] == pytest.approx(-res["loss"][search.best_index_])


def test_aml_keeps_best_stage_when_budget_exhausted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.RandomState(0)
    X = rng.rand(60, 4)
    y = (X[:, 0] > 0.5).astype(int)
    from SKSurrogate import Categorical, Real

    config = {
        "sklearn.linear_model.LogisticRegression": {
            "C": Real(0.01, 10.0),
            "solver": Categorical(["lbfgs", "liblinear"]),
        }
    }
    aml = AML(config=config, length=1, verbose=0, max_evals=8, min_random_evals=3, n_jobs=1, cv=3)
    aml.fit(X, y)
    assert np.isfinite(aml.best_estimator_score)
    assert aml.predict(X).shape == (60,)


def test_stacking_estimator_set_params_is_standard():
    from sklearn.linear_model import LogisticRegression

    from SKSurrogate import StackingEstimator

    st = StackingEstimator(LogisticRegression())
    st.set_params(cv=3, n_jobs=2, estimator__C=0.5)
    assert (st.cv, st.n_jobs, st.estimator.C) == (3, 2, 0.5)
    with pytest.raises(ValueError):
        st.set_params(bogus=1)


def test_aml_second_fit_does_not_resume_stale_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from SKSurrogate import Real

    rng = np.random.RandomState(0)
    X = rng.rand(60, 4)
    config = {"sklearn.linear_model.Ridge": {"alpha": Real(0.01, 10.0)}}
    for y in (X[:, 0] * 3, -X[:, 1] * 50):
        aml = AML(
            config=config, length=1, verbose=0, max_evals=6, min_random_evals=3, n_jobs=1, cv=3,
            scoring="neg_mean_squared_error",
        ).fit(X, y)
        assert np.isfinite(aml.best_estimator_score)
