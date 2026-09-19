import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from SKSurrogate import AML, DataPreprocess, StackingEstimator, mltrack, np2df
from SKSurrogate.eoa import UniformRand
from SKSurrogate.sensapprx import SensAprx
from SKSurrogate.structsearch import Real, SurrogateRandomCV, SurrogateSearch


class TestOptimizedPaths(unittest.TestCase):
    def test_sensitivity_reduction_matches_group_means(self):
        X = np.array([[0, 1], [0, 1], [1, 0]], dtype=float)
        y = np.array([2.0, 4.0, 8.0])
        transformer = SensAprx(reduce=True)

        transformer._avg_fucn(X, y)

        np.testing.assert_array_equal(transformer.domain, [[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(transformer.probs, [3.0, 8.0])

    def test_preprocess_and_stacking_shapes(self):
        frame = pd.DataFrame(
            {
                "category": ["a", "b", "a", "b"],
                "integer": ["1", "2", "3", "4"],
                "decimal": ["1.5", "2.5", "3.5", "4.5"],
            }
        )
        processed = DataPreprocess(frame).encode()
        self.assertEqual(processed["integer"].tolist(), [1, 2, 3, 4])

        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        stacked = StackingEstimator(RandomForestClassifier(n_estimators=3, random_state=1))
        stacked.fit(X, y)
        self.assertEqual(stacked.transform(X).shape, (4, 4))

    def test_stacking_fit_transform_uses_out_of_fold_predictions(self):
        X = np.arange(24, dtype=float).reshape(12, 2)
        y = np.array([0, 1] * 6)
        stacked = StackingEstimator(
            KNeighborsClassifier(n_neighbors=1),
            cv=KFold(n_splits=3, shuffle=False),
            decision=False,
        )

        features = stacked.fit_transform(X, y)

        self.assertEqual(features.shape, (12, 5))
        self.assertTrue(np.any(features[:, -1] != stacked.estimator.predict(X)))

    def test_mltrace_reuses_plot_split_and_probability_metrics(self):
        X, y = make_classification(n_samples=80, n_features=4, random_state=2)
        frame = np2df(np.column_stack((X, y)), ["f0", "f1", "f2", "f3", "target"])()
        cv = ShuffleSplit(n_splits=2, test_size=0.25, random_state=2)

        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("test", db_name=database.name, cv=cv)
            tracker.RegisterData(frame, "target")
            tracker.UpdateMetadata({"dataset": {"rows": 80}, "seed": 2})
            metadata = tracker.GetMetadata()
            self.assertEqual(metadata["dataset"]["rows"], 80)
            self.assertEqual(metadata["seed"], 2)
            model = tracker.LogModel(RandomForestClassifier(n_estimators=5, random_state=2), "rf")
            metrics = tracker.LogMetrics(model)
            tracker.plot_roc_curve(model, "rf")
            first_split = tracker._split_cache[model.mltrack_id]
            tracker.plot_lift_curve(model, title="rf")
            second_split = tracker._split_cache[model.mltrack_id]

            self.assertIs(first_split[2], second_split[2])
            self.assertIsNotNone(metrics["logloss"])
            self.assertEqual(len(tracker.allPlots(model.mltrack_id)), 2)

    def test_uniform_parent_sampling_is_unique(self):
        class Reference:
            population = [("a",), ("b",), ("c",), ("d",)]
            population_size = 4
            num_parents = 3
            evals = {item: None for item in population}

        from collections import OrderedDict

        parents = UniformRand()(Reference())
        self.assertIsInstance(parents, OrderedDict)
        self.assertEqual(len(parents), 3)
        self.assertEqual(len(set(parents)), 3)

    def test_aml_estimator_surface(self):
        automl = AML(config={}, random_state=11, time_limit=1.0, max_evals=3, verbose=0)
        self.assertEqual(automl.get_params()["random_state"], 11)
        with self.assertRaises(RuntimeError):
            automl.predict([[0.0]])

        automl.best_estimator_ = LinearRegression().fit([[0.0], [1.0]], [0.0, 1.0])
        np.testing.assert_allclose(automl.predict([[2.0]]), [2.0])
        self.assertEqual(automl.score([[0.0], [1.0]], [0.0, 1.0]), 1.0)
        automl.set_params(verbose=0)
        self.assertIsNone(automl.best_estimator_)

    def test_surrogate_search_evaluation_budget(self):
        evaluations = []

        def objective(point):
            evaluations.append(point[0])
            return point[0] ** 2

        with tempfile.TemporaryDirectory() as directory:
            search = SurrogateSearch(
                objective,
                x0=[0.0],
                bounds=[(-1.0, 1.0)],
                max_iter=100,
                min_evals=2,
                max_evals=3,
                task_name=directory + "/search",
            )
            search()
        self.assertLessEqual(len(evaluations), 3)

    def test_surrogate_trial_metadata_and_frontier(self):
        X, y = make_classification(
            n_samples=30,
            n_features=4,
            n_informative=2,
            n_redundant=0,
            random_state=4,
        )
        search = SurrogateRandomCV(
            LogisticRegression(max_iter=100),
            {"C": Real(0.1, 1.0)},
            cv=3,
            n_jobs=1,
            max_iter=1,
            min_evals=2,
            max_evals=1,
            refit=False,
        )
        search.fit(X, y)

        self.assertEqual(len(search.cv_results_["params"]), len(search.evaluation_history_))
        self.assertIn("duration", search.cv_results_)
        self.assertIn("error", search.cv_results_)
        self.assertTrue(search.pareto_frontier())


if __name__ == "__main__":
    unittest.main()
