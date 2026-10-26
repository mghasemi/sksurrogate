"""Regression tests for the scikit-learn ``BaseSearchCV`` contract.

These pin the behaviour that the README promises ("drop-in replacement for
``BayesSearchCV``") and that silently broke against scikit-learn 1.9:

* ``best_params_`` / ``best_index_`` / ``multimetric_`` / ``n_splits_`` exist
  after ``fit`` and ``score`` / ``predict`` work through ``BaseSearchCV``;
* ``best_score_`` follows the sklearn convention (higher is better) and equals
  the mean CV score of ``best_params_`` recomputed independently -- the internal
  minimisation must never appear as a sign on the public attribute;
* ``cv_results_["mean_test_score"]`` carries scores, not costs;
* the ``params`` constructor argument is not mutated by ``fit`` (``clone``
  contract), while list/tuple search spaces are accepted as documented.
"""

import unittest

import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

from SKSurrogate import Categorical, Integer, Real, SurrogateRandomCV


def _data(n=300):
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X[:n], y[:n], test_size=0.25, stratify=y[:n], random_state=0
    )
    return X_train, X_test, y_train, y_test


def _search(params, **kwargs):
    settings = dict(max_iter=10, min_evals=6, cv=3, n_jobs=1, random_state=0)
    settings.update(kwargs)
    return SurrogateRandomCV(
        RandomForestClassifier(n_estimators=20, random_state=0), params, **settings
    )


class TestBaseSearchCVContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = _data()
        cls.params = {"max_depth": (3, 12), "criterion": ["gini", "entropy"]}
        cls.search = _search(cls.params)
        cls.search.fit(cls.X_train, cls.y_train)

    def test_best_params_attribute_exists(self):
        self.assertTrue(hasattr(self.search, "best_params_"))
        self.assertEqual(set(self.search.best_params_), {"max_depth", "criterion"})

    def test_best_params_match_best_estimator(self):
        fitted = self.search.best_estimator_.get_params()
        for name, value in self.search.best_params_.items():
            self.assertEqual(fitted[name], value)

    def test_best_score_is_a_score_not_a_cost(self):
        """best_score_ equals the mean CV score of best_params_ (no sign flip)."""
        manual = cross_val_score(
            clone(self.search.estimator).set_params(**self.search.best_params_),
            self.X_train,
            self.y_train,
            cv=3,
            n_jobs=1,
        ).mean()
        self.assertAlmostEqual(self.search.best_score_, manual, places=10)
        self.assertGreater(self.search.best_score_, 0.0)
        self.assertLessEqual(self.search.best_score_, 1.0)
        self.assertAlmostEqual(self.search.best_estimator_score, manual, places=10)

    def test_search_score_matches_best_estimator_score(self):
        self.assertAlmostEqual(
            self.search.score(self.X_test, self.y_test),
            self.search.best_estimator_.score(self.X_test, self.y_test),
            places=12,
        )

    def test_predict_delegates_to_best_estimator(self):
        np.testing.assert_array_equal(
            self.search.predict(self.X_test),
            self.search.best_estimator_.predict(self.X_test),
        )

    def test_base_search_cv_attributes_are_present(self):
        self.assertFalse(self.search.multimetric_)
        self.assertEqual(self.search.n_splits_, 3)
        self.assertIsInstance(self.search.best_index_, int)
        self.assertGreaterEqual(self.search.refit_time_, 0.0)
        self.assertEqual(self.search.best_index_, self.search._trial_index(self.search.best_params_))

    def test_cv_results_use_score_convention(self):
        scores = [s for s in self.search.cv_results_["mean_test_score"] if s is not None]
        self.assertTrue(scores)
        self.assertAlmostEqual(max(scores), self.search.best_score_, places=10)
        # accuracy-like scores are non-negative; a cost flip would make them all <= 0
        self.assertTrue(all(s >= 0.0 for s in scores))

    def test_history_records_score_and_cost(self):
        complete = [e for e in self.search.evaluation_history_ if e["status"] == "complete"]
        self.assertTrue(complete)
        for entry in complete:
            self.assertAlmostEqual(entry["score"], -entry["cost"], places=12)

    def test_pareto_frontier_maximises_score(self):
        frontier = self.search.pareto_frontier()
        self.assertTrue(frontier)
        best = max(frontier, key=lambda item: item["score"])
        self.assertAlmostEqual(best["score"], self.search.best_score_, places=12)


class TestSearchSpaceCoercion(unittest.TestCase):
    def test_params_argument_is_not_mutated(self):
        params = {"max_depth": (3, 12), "criterion": ["gini", "entropy"]}
        X_train, _, y_train, _ = _data(n=200)
        search = _search(params)
        search.fit(X_train, y_train)
        # clone() must still see what the user passed in
        self.assertIsInstance(search.params["max_depth"], tuple)
        self.assertEqual(search.params["criterion"], ["gini", "entropy"])
        self.assertIsInstance(search.params_["max_depth"], Integer)
        self.assertIsInstance(search.params_["criterion"], Categorical)
        self.assertEqual(clone(search).params["criterion"], ["gini", "entropy"])

    def test_two_element_numeric_pair_is_an_interval(self):
        X_train, _, y_train, _ = _data(n=200)
        search = SurrogateRandomCV(
            LogisticRegression(max_iter=500, random_state=0),
            {"C": (0.01, 10.0)},
            max_iter=6,
            min_evals=4,
            cv=3,
            n_jobs=1,
            random_state=0,
        )
        search.fit(X_train, y_train)
        self.assertGreaterEqual(search.best_params_["C"], 0.01)
        self.assertLessEqual(search.best_params_["C"], 10.0)
        self.assertIsInstance(search.params_["C"], Real)

    def test_domain_objects_still_accepted(self):
        X_train, _, y_train, _ = _data(n=200)
        search = _search({"max_depth": Integer(3, 10), "criterion": Categorical(["gini"])})
        search.fit(X_train, y_train)
        self.assertEqual(search.best_params_["criterion"], "gini")

    def test_empty_domain_is_rejected(self):
        X_train, _, y_train, _ = _data(n=200)
        search = _search({"criterion": []})
        with self.assertRaises(ValueError):
            search.fit(X_train, y_train)

    def test_bad_domain_type_is_rejected_with_actionable_message(self):
        X_train, _, y_train, _ = _data(n=200)
        search = _search({"criterion": 3.5})
        with self.assertRaises(ValueError) as ctx:
            search.fit(X_train, y_train)
        self.assertIn("criterion", str(ctx.exception))


class TestAmlScoreConvention(unittest.TestCase):
    """AML delegates to SurrogateRandomCV.search and must keep the same units."""

    @classmethod
    def setUpClass(cls):
        from collections import OrderedDict

        from SKSurrogate import AML

        # ``__new__`` bypasses __init__ (which builds a default estimator
        # catalogue and may need optional third-party libraries): these tests
        # exercise ordering/units only.
        cls.aml = AML.__new__(AML)
        cls.aml.models = OrderedDict(
            [
                (("weak",), (None, 0.41)),
                (("strong",), (None, 0.93)),
                (("mid",), (None, 0.62)),
            ]
        )
        cls.aml.evaluation_history_ = [
            {"pipeline": ("weak",), "score": 0.41, "duration": 1.0},
            {"pipeline": ("strong",), "score": 0.93, "duration": 2.0},
            {"pipeline": ("mid",), "score": 0.62, "duration": 0.5},
        ]

    def test_get_top_returns_highest_scores_first(self):
        top = self.aml.get_top(2)
        self.assertEqual(list(top), [("strong",), ("mid",)])
        self.assertEqual(top[("strong",)][1], 0.93)

    def test_pareto_frontier_does_not_flip_scores(self):
        frontier = {tuple(point["pipeline"]): point["score"] for point in self.aml.pareto_frontier()}
        self.assertEqual(frontier[("strong",)], 0.93)
        self.assertEqual(frontier[("mid",)], 0.62)


class TestScorerConvention(unittest.TestCase):
    def test_negative_scorer_keeps_sklearn_sign(self):
        X_train, _, y_train, _ = _data(n=240)
        from sklearn.linear_model import Ridge

        search = SurrogateRandomCV(
            Ridge(),
            {"alpha": [0.1, 1.0, 10.0]},
            scoring="neg_mean_squared_error",
            max_iter=6,
            min_evals=4,
            cv=3,
            n_jobs=1,
            random_state=0,
        )
        search.fit(X_train, y_train.astype(float))
        manual = cross_val_score(
            clone(search.estimator).set_params(**search.best_params_),
            X_train,
            y_train.astype(float),
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=1,
        ).mean()
        self.assertAlmostEqual(search.best_score_, manual, places=10)
        self.assertLessEqual(search.best_score_, 0.0)


if __name__ == "__main__":
    unittest.main()
