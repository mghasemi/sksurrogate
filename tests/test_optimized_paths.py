import tempfile
import unittest
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold, GroupKFold
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

    def test_dataset_fingerprint_and_schema_are_stable_and_sensitive(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            frame_a = pd.DataFrame(
                {
                    "feature_a": [1.0, 2.0, 3.0],
                    "feature_b": ["x", "y", "x"],
                    "target": [0, 1, 0],
                }
            )
            frame_b = frame_a.copy()
            frame_c = frame_a.copy()
            frame_c.loc[0, "feature_a"] = 9.0

            for frame in (frame_a, frame_b):
                tracker = mltrack("fingerprint-test", db_name=database.name)
                tracker.RegisterData(frame, "target")
                metadata = tracker.GetMetadata()
                self.assertIn("dataset_fingerprint", metadata)
                self.assertIn("dataset_schema", metadata)
                self.assertEqual(len(metadata["dataset_fingerprint"]), 64)

            tracker = mltrack("fingerprint-test", db_name=database.name)
            tracker.RegisterData(frame_a, "target")
            fingerprint_a = tracker.GetMetadata()["dataset_fingerprint"]
            tracker.RegisterData(frame_c, "target")
            fingerprint_c = tracker.GetMetadata()["dataset_fingerprint"]
            self.assertNotEqual(fingerprint_a, fingerprint_c)
            schema = tracker.GetMetadata()["dataset_schema"]
            self.assertIn("feature_a", schema)
            self.assertIn("target", schema)

    def test_dataset_schema_validation_rejects_missing_and_reordered_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("schema-validation", db_name=database.name)
            reference = pd.DataFrame(
                {
                    "feature_a": [1.0, 2.0, 3.0],
                    "feature_b": ["x", "y", "x"],
                    "target": [0, 1, 0],
                }
            )
            tracker.RegisterData(reference, "target")

            tracker.validate_data(reference)
            with self.assertRaises(ValueError):
                tracker.validate_data(reference[["feature_a", "target"]])
            with self.assertRaises(ValueError):
                tracker.validate_data(reference[["target", "feature_b", "feature_a"]])

            prediction_data = reference[["feature_a", "feature_b"]].copy()
            tracker.validate_prediction_data(prediction_data)
            with self.assertRaises(ValueError):
                tracker.validate_prediction_data(reference[["feature_a", "target"]])

    def test_dataset_validation_policy_handles_missing_and_unknown_categories(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("schema-policy", db_name=database.name)
            reference = pd.DataFrame(
                {
                    "feature_a": [1.0, 2.0, 3.0],
                    "feature_b": ["x", "y", "x"],
                    "target": [0, 1, 0],
                }
            )
            tracker.RegisterData(reference, "target")

            with self.assertRaises(ValueError):
                tracker.validate_data(reference[["feature_a", "target"]])

            self.assertTrue(
                tracker.validate_data(
                    reference[["feature_a", "target"]],
                    missing_columns="ignore",
                    target="target",
                )
            )

            unknown_category = reference.copy()
            unknown_category.loc[0, "feature_b"] = "z"
            with self.assertRaises(ValueError):
                tracker.validate_data(unknown_category)
            self.assertTrue(
                tracker.validate_data(unknown_category, unknown_categories="ignore")
            )

    def test_dataset_provenance_and_partition_metadata_are_recorded(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("provenance-test", db_name=database.name)
            frame = pd.DataFrame(
                {
                    "feature_a": [1.0, 2.0, 3.0],
                    "feature_b": ["x", "y", "x"],
                    "target": [0, 1, 0],
                }
            )

            tracker.RegisterData(frame, "target", source_uri="s3://bucket/train.csv", partition="train")
            metadata = tracker.GetMetadata()
            self.assertEqual(metadata["dataset_source_uri"], "s3://bucket/train.csv")
            self.assertEqual(metadata["dataset_partition"], "train")
            self.assertIn("dataset_ingestion_timestamp", metadata)
            self.assertEqual(metadata["dataset_feature_count"], 2)
            self.assertEqual(metadata["dataset_rows"], 3)

    def test_dataset_partition_history_tracks_train_and_validation_sets(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("partition-history", db_name=database.name)
            train = pd.DataFrame(
                {
                    "feature_a": [1.0, 2.0, 3.0],
                    "feature_b": ["x", "y", "x"],
                    "target": [0, 1, 0],
                }
            )
            validation = pd.DataFrame(
                {
                    "feature_a": [4.0, 5.0],
                    "feature_b": ["y", "x"],
                    "target": [1, 0],
                }
            )

            tracker.RegisterData(train, "target", partition="train", source_uri="train.csv")
            tracker.RegisterData(validation, "target", partition="validation", source_uri="validation.csv")
            metadata = tracker.GetMetadata()

            self.assertEqual(metadata["dataset_partition"], "validation")
            self.assertIn("train", metadata["dataset_partition_history"])
            self.assertIn("validation", metadata["dataset_partition_history"])
            self.assertEqual(metadata["dataset_partition_history"]["train"]["source_uri"], "train.csv")
            self.assertEqual(metadata["dataset_partition_history"]["validation"]["source_uri"], "validation.csv")

    def test_explicit_split_data_is_retrievable_by_partition(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("explicit-splits", db_name=database.name)
            train = pd.DataFrame(
                {
                    "feature_a": [1.0, 2.0, 3.0],
                    "feature_b": ["x", "y", "x"],
                    "target": [0, 1, 0],
                }
            )
            validation = pd.DataFrame(
                {
                    "feature_a": [4.0, 5.0],
                    "feature_b": ["y", "x"],
                    "target": [1, 0],
                }
            )

            tracker.RegisterData(train, "target", partition="train")
            tracker.RegisterData(validation, "target", partition="validation")

            X_train, y_train = tracker.get_data("train")
            X_validation, y_validation = tracker.get_data("validation")
            self.assertEqual(X_train.shape[0], 3)
            self.assertEqual(X_validation.shape[0], 2)
            self.assertListEqual(y_train.tolist(), [0, 1, 0])
            self.assertListEqual(y_validation.tolist(), [1, 0])

            split_map = tracker.dataset_splits()
            self.assertIn("train", split_map)
            self.assertIn("validation", split_map)
            self.assertEqual(split_map["validation"]["rows"], 2)

    def test_nested_cv_evaluation_uses_train_split_for_selection_and_validation_for_final_score(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("nested-cv", db_name=database.name)
            train = pd.DataFrame(
                {
                    "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    "feature_b": ["x", "y", "x", "y", "x", "y"],
                    "target": [0, 1, 0, 1, 0, 1],
                }
            )
            validation = pd.DataFrame(
                {
                    "feature_a": [7.0, 8.0],
                    "feature_b": ["x", "y"],
                    "target": [0, 1],
                }
            )
            tracker.RegisterData(train, "target", partition="train")
            tracker.RegisterData(validation, "target", partition="validation")

            results = tracker.nested_cv_evaluation(
                LogisticRegression(max_iter=200),
                inner_cv=2,
                outer_cv=2,
                train_partition="train",
                validation_partition="validation",
            )
            self.assertIn("outer_scores", results)
            self.assertIn("mean_outer_score", results)
            self.assertIn("final_validation_score", results)
            self.assertGreaterEqual(results["final_validation_score"], 0.0)
            self.assertLessEqual(results["mean_outer_score"], 1.0)

    def test_nested_cv_evaluation_supports_group_kfold_configuration(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("grouped-cv", db_name=database.name)
            train = pd.DataFrame(
                {
                    "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    "feature_b": ["x", "y", "x", "y", "x", "y"],
                    "group_id": [0, 0, 1, 1, 2, 2],
                    "target": [0, 1, 0, 1, 0, 1],
                }
            )
            validation = pd.DataFrame(
                {
                    "feature_a": [7.0, 8.0],
                    "feature_b": ["x", "y"],
                    "group_id": [3, 3],
                    "target": [0, 1],
                }
            )
            tracker.RegisterData(train, "target", partition="train")
            tracker.RegisterData(validation, "target", partition="validation")

            results = tracker.nested_cv_evaluation(
                LogisticRegression(max_iter=200),
                inner_cv=GroupKFold(n_splits=2),
                outer_cv=GroupKFold(n_splits=2),
                train_partition="train",
                validation_partition="validation",
                groups=train["group_id"].to_numpy(),
            )
            self.assertIn("outer_scores", results)
            self.assertIn("mean_outer_score", results)
            self.assertIn("final_validation_score", results)
            self.assertGreaterEqual(len(results["outer_scores"]), 1)
            self.assertLessEqual(results["mean_outer_score"], 1.0)

    def test_nested_cv_evaluation_stores_fold_level_predictions_and_metrics(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("fold-metrics", db_name=database.name)
            train = pd.DataFrame(
                {
                    "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    "feature_b": ["x", "y", "x", "y", "x", "y"],
                    "target": [0, 1, 0, 1, 0, 1],
                }
            )
            validation = pd.DataFrame(
                {
                    "feature_a": [7.0, 8.0],
                    "feature_b": ["x", "y"],
                    "target": [0, 1],
                }
            )
            tracker.RegisterData(train, "target", partition="train")
            tracker.RegisterData(validation, "target", partition="validation")

            results = tracker.nested_cv_evaluation(
                LogisticRegression(max_iter=200),
                inner_cv=2,
                outer_cv=2,
                train_partition="train",
                validation_partition="validation",
            )
            self.assertIn("fold_metrics", results)
            self.assertEqual(len(results["fold_metrics"]), 2)
            self.assertIn("inner_scores", results["fold_metrics"][0])
            self.assertIn("outer_predictions", results["fold_metrics"][0])
            self.assertIn("outer_score", results["fold_metrics"][0])

    def test_nested_cv_evaluation_supports_repeated_cv_and_reports_confidence_interval(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("repeat-cv", db_name=database.name)
            train = pd.DataFrame(
                {
                    "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    "feature_b": ["x", "y", "x", "y", "x", "y", "x", "y"],
                    "target": [0, 1, 0, 1, 0, 1, 0, 1],
                }
            )
            validation = pd.DataFrame(
                {
                    "feature_a": [9.0, 10.0],
                    "feature_b": ["x", "y"],
                    "target": [0, 1],
                }
            )
            tracker.RegisterData(train, "target", partition="train")
            tracker.RegisterData(validation, "target", partition="validation")

            results = tracker.nested_cv_evaluation(
                LogisticRegression(max_iter=200),
                inner_cv=2,
                outer_cv=2,
                train_partition="train",
                validation_partition="validation",
                repeats=3,
                confidence_level=0.95,
            )
            self.assertIn("repeat_count", results)
            self.assertEqual(results["repeat_count"], 3)
            self.assertIn("confidence_interval", results)
            self.assertIn("lower", results["confidence_interval"])
            self.assertIn("upper", results["confidence_interval"])
            self.assertIn("repeated_outer_scores", results)
            self.assertEqual(len(results["repeated_outer_scores"]), 3)

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

    def test_aml_records_budget_termination_summary(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y = np.array([0, 1, 0, 1], dtype=int)
        automl = AML(
            config={
                "sklearn.linear_model.LogisticRegression": {
                    "C": Real(0.1, 1.0),
                }
            },
            length=1,
            cv=2,
            random_state=0,
            time_limit=0.0,
            max_evals=1,
            verbose=0,
        )

        automl.fit(X, y)

        self.assertEqual(automl.termination_reason, "time_limit")
        self.assertEqual(automl.summary_["termination_reason"], "time_limit")
        self.assertIn("budget", automl.summary_)

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

    def test_surrogate_search_records_budget_termination_reason(self):
        search = SurrogateSearch(
            lambda z: (z[0] - 2.5) ** 2,
            x0=[0.0],
            bounds=[(0.0, 5.0)],
            max_iter=25,
            min_evals=2,
            time_limit=0.0,
            random_state=0,
        )

        search()

        self.assertEqual(search.termination_reason, "time_limit")
        self.assertEqual(search.summary_["termination_reason"], "time_limit")
        self.assertIn("budget", search.summary_)

    def test_surrogate_search_reports_no_progress_early_stop(self):
        def objective(point):
            return 1.0 if point[0] < 0.0 else 0.0

        search = SurrogateSearch(
            objective,
            x0=[0.0],
            bounds=[(-1.0, 1.0)],
            max_iter=5,
            min_evals=2,
            max_itr_no_prog=0,
            random_state=0,
        )

        search()

        self.assertEqual(search.termination_reason, "max_iter_no_progress")
        self.assertEqual(search.summary_["termination_reason"], "max_iter_no_progress")
        self.assertIn("budget", search.summary_)

    def test_aml_cpu_limit_caps_parallelism(self):
        automl = AML(config={}, n_jobs=-1, cpu_limit=1, verbose=0)
        self.assertEqual(automl.n_jobs, 1)
        self.assertEqual(automl.cpu_limit, 1)

    def test_aml_memory_limit_is_recorded(self):
        automl = AML(config={}, memory_limit="512MiB", verbose=0)
        self.assertEqual(automl.memory_limit, "512MiB")
        self.assertIn("memory_limit", automl.get_params())

    def test_surrogate_cv_prunes_unpromising_partial_trials(self):
        from sklearn.base import BaseEstimator, ClassifierMixin

        class AlwaysZeroClassifier(BaseEstimator, ClassifierMixin):
            _estimator_type = "classifier"

            def __init__(self, alpha=0.0):
                self.alpha = alpha

            def fit(self, X, y):
                self.n_features_in_ = X.shape[1]
                return self

            def score(self, X, y):
                return 0.0

        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y = np.array([0, 1, 0, 1], dtype=int)

        search = SurrogateRandomCV(
            AlwaysZeroClassifier(),
            {"alpha": Real(0.0, 1.0)},
            cv=2,
            n_jobs=1,
            max_iter=1,
            min_evals=1,
            prune_threshold=0.0,
            min_partial_folds=1,
            refit=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            search.fit(X, y)

        self.assertIn("pruned", [item["status"] for item in search.evaluation_history_])


if __name__ == "__main__":
    unittest.main()
