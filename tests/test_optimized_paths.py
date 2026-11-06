import tempfile
import unittest
import warnings
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from SKSurrogate import (
    AML,
    assert_bundle_quality,
    BundleQualityGateError,
    check_bundle_quality,
    DataPreprocess,
    FoldLocalDataPreprocessor,
    HilbertRegressor,
    DeploymentApprovalGate,
    DaskExecutionBackend,
    LocalProcessExecutionBackend,
    ModelBundle,
    ModelRegistry,
    RetrainingJob,
    SchemaValidationError,
    StackingEstimator,
    export_mlflow,
    predict_batch,
    dataset_profile,
    drift_report,
    InferenceMonitor,
    delayed_label_performance,
    prediction_distribution_report,
    loss_report,
    load_bundle,
    latin_hypercube,
    mltrack,
    np2df,
    prediction_interval_diagnostics,
    save_bundle,
    sensitive_feature_report,
    fairness_report,
    subgroup_performance_report,
    split_conformal_regression_interval,
    split_conformal_quantile_interval,
    suggest_next_point,
)
from SKSurrogate.inference import main as batch_main
from SKSurrogate.inference import create_app
from SKSurrogate.eoa import UniformRand
from SKSurrogate.sensapprx import SensAprx
from SKSurrogate.structsearch import Categorical, Real, SurrogateRandomCV, SurrogateSearch


class MeanRecordingRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.fit_X_ = np.asarray(X).copy()
        self.fit_y_ = np.asarray(y).copy()
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class TestOptimizedPaths(unittest.TestCase):
    def test_latin_hypercube_is_seeded_bounded_and_stratified(self):
        bounds = [(-2.0, 3.0), (10.0, 20.0), (0.25, 1.25)]
        design = latin_hypercube(bounds, 12, random_state=31)
        repeated = latin_hypercube(bounds, 12, random_state=31)

        self.assertEqual(design.shape, (12, 3))
        np.testing.assert_array_equal(design, repeated)
        self.assertTrue(np.all(design >= np.array([-2.0, 10.0, 0.25])))
        self.assertTrue(np.all(design <= np.array([3.0, 20.0, 1.25])))
        for column, (lower, upper) in enumerate(bounds):
            strata = np.floor((design[:, column] - lower) / (upper - lower) * 12)
            self.assertEqual(sorted(strata.tolist()), list(range(12)))

    def test_latin_hypercube_rejects_invalid_design_requests(self):
        for bounds in ([], [(1.0, 1.0)], [(2.0, 1.0)], [(0.0, np.inf)], [(0.0, 1.0, 2.0)]):
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):
                    latin_hypercube(bounds, 4)
        for n_samples in (0, -1, 10001, True, 1.5):
            with self.subTest(n_samples=n_samples):
                with self.assertRaisesRegex(ValueError, "n_samples"):
                    latin_hypercube([(0.0, 1.0)], n_samples)
        with self.assertRaisesRegex(ValueError, "random_state"):
            latin_hypercube([(0.0, 1.0)], 2, random_state=True)

    def test_active_learning_recommends_a_bounded_repeatable_unevaluated_point(self):
        bounds = [(-1.0, 1.0), (-2.0, 2.0)]
        samples = np.array(
            [
                [-1.0, -2.0],
                [-1.0, 2.0],
                [1.0, -2.0],
                [1.0, 2.0],
                [0.0, 0.0],
                [0.0, 1.0],
            ]
        )
        values = np.sum((samples - np.array([0.25, -0.5])) ** 2, axis=1)
        proposal = suggest_next_point(
            samples,
            values,
            bounds,
            n_candidates=128,
            random_state=17,
        )
        repeated = suggest_next_point(
            samples,
            values,
            bounds,
            n_candidates=128,
            random_state=17,
        )

        self.assertEqual(proposal["acquisition"], "lower_confidence_bound")
        self.assertEqual(proposal["optimization_direction"], "minimize")
        self.assertEqual(proposal["surrogate_kernel"], "Matern(nu=2.5)")
        self.assertEqual(proposal["kernel_length_scale"], [1.0, 1.0])
        self.assertEqual(proposal["observations_used"], len(samples))
        self.assertEqual(proposal["candidate_pool_size"], 128)
        self.assertIn("not evaluated", proposal["interpretation"])
        self.assertTrue(np.all(proposal["candidate"] >= [-1.0, -2.0]))
        self.assertTrue(np.all(proposal["candidate"] <= [1.0, 2.0]))
        np.testing.assert_array_equal(proposal["candidate"], repeated["candidate"])
        self.assertFalse(np.any(np.all(samples == proposal["candidate"], axis=1)))

    def test_active_learning_does_not_emit_gp_optimizer_convergence_warnings(self):
        samples = np.array(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]
        )
        values = np.sin(samples[:, 0] * 6.0) + samples[:, 1] ** 2
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            proposal = suggest_next_point(
                samples,
                values,
                [(-1.0, 1.0), (-1.0, 1.0)],
                n_candidates=128,
                random_state=4,
            )
        self.assertEqual(proposal["kernel_length_scale"], [1.0, 1.0])

    def test_active_learning_validates_observations_and_bounds(self):
        bounds = [(0.0, 1.0)]
        samples = np.array([[0.1], [0.9]])
        values = np.array([1.0, 0.5])
        invalid = (
            (samples[:1], values),
            (samples, values[:1]),
            (np.array([[0.1, 0.2], [0.3, 0.4]]), values),
            (np.array([[0.1], [np.nan]]), values),
            (np.array([[0.1], [1.1]]), values),
        )
        for bad_samples, bad_values in invalid:
            with self.subTest(samples=bad_samples, values=bad_values):
                with self.assertRaises(ValueError):
                    suggest_next_point(bad_samples, bad_values, bounds)
        for kwargs in (
            {"n_candidates": 0},
            {"beta": 0.0},
            {"beta": np.inf},
            {"random_state": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    suggest_next_point(samples, values, bounds, **kwargs)

    def test_dataset_profile_summarizes_mixed_data_and_associations(self):
        frame = pd.DataFrame(
            {
                "signal": [1.0, 2.0, 3.0, 4.0, np.inf],
                "redundant": [2.0, 4.0, 6.0, 8.0, np.inf],
                "group": ["a", "a", "b", "b", None],
                "target": ["low", "low", "high", "high", "high"],
            }
        )

        report = dataset_profile(frame, target="target", max_categories=5)

        self.assertEqual(report["rows"], 5)
        self.assertEqual(report["sampled_rows"], 5)
        self.assertEqual(report["target"]["distribution"]["unique"], 2)
        self.assertEqual(report["target"]["distribution"]["top"][0]["value"], "high")
        numeric = next(item for item in report["columns"] if item["column"] == "signal")
        self.assertEqual(numeric["distribution"]["non_finite_count"], 1)
        self.assertEqual(numeric["distribution"]["count"], 4)
        pair = next(
            item
            for item in report["feature_associations"]
            if {item["feature_a"], item["feature_b"]} == {"signal", "redundant"}
        )
        self.assertEqual(pair["method"], "pearson")
        self.assertAlmostEqual(pair["value"], 1.0)
        target_group = next(
            item for item in report["target_associations"] if item["feature"] == "group"
        )
        self.assertEqual(target_group["method"], "cramers_v")
        self.assertIn("not establish causation", report["interpretation"])

    def test_dataset_profile_reports_robust_statistics_and_sampled_data_quality(self):
        frame = pd.DataFrame(
            {
                "value": [1.0, 2.0, 2.0, 3.0, 100.0, 2.0],
                "duplicate_value": [1.0, 2.0, 2.0, 3.0, 100.0, 2.0],
                "constant": ["same"] * 6,
                "all_missing": [None] * 6,
            }
        )

        report = dataset_profile(frame)

        numeric = next(item for item in report["columns"] if item["column"] == "value")
        self.assertAlmostEqual(numeric["distribution"]["median_absolute_deviation"], 0.5)
        self.assertEqual(numeric["distribution"]["iqr_outlier_count"], 1)
        self.assertEqual(numeric["distribution"]["iqr_outlier_rate"], 1 / 6)
        self.assertIsNotNone(numeric["distribution"]["skewness"])
        self.assertIsNotNone(numeric["distribution"]["excess_kurtosis"])
        quality = report["data_quality"]
        self.assertEqual(quality["duplicate_rows"]["count"], 2)
        self.assertEqual(quality["duplicate_rows"]["scope"], "sampled rows and selected features")
        self.assertEqual(quality["constant_features"], ["constant"])
        self.assertEqual(quality["all_missing_features"], ["all_missing"])
        self.assertEqual(
            quality["duplicate_features"],
            [{"feature": "duplicate_value", "duplicate_of": "value"}],
        )

    def test_dataset_profile_reports_bounded_missingness_patterns(self):
        frame = pd.DataFrame(
            {
                "first": [None, None, 1.0, 1.0, 1.0],
                "second": [1.0, 1.0, None, None, 1.0],
                "target": [0, 1, 0, 1, 1],
            }
        )

        report = dataset_profile(frame, target="target", max_categories=1)

        patterns = report["missingness_patterns"]
        self.assertEqual(patterns["patterns"][0]["missing_features"], ["first"])
        self.assertEqual(patterns["patterns"][0]["count"], 2)
        self.assertAlmostEqual(patterns["patterns"][0]["proportion"], 0.4)
        self.assertEqual(patterns["other_count"], 3)
        self.assertEqual(patterns["pattern_limit"], 1)

    def test_dataset_profile_extreme_finite_values_remain_json_safe(self):
        report = dataset_profile(
            pd.DataFrame({"value": [1.0e308, 1.0e308, 1.0e308, 1.0e308]})
        )

        distribution = report["columns"][0]["distribution"]
        self.assertEqual(distribution["mean"], 1.0e308)
        self.assertEqual(distribution["std"], 0.0)
        json.dumps(report, allow_nan=False)

    def test_dataset_profile_computes_bounded_descriptive_group_comparisons(self):
        frame = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B", "C", None, "A"],
                "measurement": [1.0, 3.0, 10.0, np.inf, 100.0, 8.0, 5.0],
                "target": [2.0, 4.0, 11.0, 12.0, 101.0, 9.0, 6.0],
            }
        )

        report = dataset_profile(
            frame,
            target="target",
            group_by="group",
            max_categories=1,
        )

        comparison = report["group_comparisons"]
        self.assertEqual(comparison["method"], "descriptive_numeric_distribution_by_group")
        self.assertEqual(comparison["missing_group_count"], 1)
        self.assertEqual(comparison["group_limit"], 1)
        self.assertEqual(
            [group["group"] for group in comparison["comparisons"][0]["groups"]],
            ["A", "Other (combined)"],
        )
        self.assertEqual(
            [item["column"] for item in comparison["comparisons"]],
            ["measurement", "target"],
        )
        self.assertEqual(comparison["comparisons"][0]["groups"][0]["count"], 3)
        self.assertEqual(comparison["comparisons"][0]["groups"][0]["row_count"], 3)
        self.assertEqual(comparison["comparisons"][0]["groups"][1]["count"], 3)
        self.assertEqual(comparison["comparisons"][0]["groups"][1]["row_count"], 3)
        self.assertEqual(
            comparison["comparisons"][0]["groups"][1]["non_finite_count"], 1
        )
        self.assertIn("no hypothesis test", comparison["interpretation"])
        json.dumps(report, allow_nan=False)

    def test_dataset_profile_reports_rank_associations_for_monotonic_pairs(self):
        frame = pd.DataFrame(
            {
                "linear": [1.0, 2.0, 3.0, 4.0, np.inf],
                "monotonic": [1.0, 4.0, 9.0, 16.0, 25.0],
                "constant": [1.0] * 5,
                "target": [2.0, 3.0, 5.0, 8.0, 10.0],
            }
        )

        report = dataset_profile(frame, target="target")

        monotonic = [
            item
            for item in report["rank_associations"]["features"]
            if {item["feature_a"], item["feature_b"]} == {"linear", "monotonic"}
        ]
        self.assertEqual(len(monotonic), 2)
        self.assertEqual({item["method"] for item in monotonic}, {"spearman", "kendall_tau"})
        for item in monotonic:
            self.assertAlmostEqual(item["value"], 1.0)
            self.assertEqual(item["n"], 4)
        target_rank = next(
            item
            for item in report["rank_associations"]["target"]
            if item["feature"] == "linear"
        )
        self.assertEqual(target_rank["method"], "spearman")
        self.assertEqual(target_rank["n"], 4)
        self.assertIn("do not detect all nonlinear dependence", report["rank_associations"]["interpretation"])

    def test_dataset_profile_reports_multicollinearity_without_nonfinite_json(self):
        frame = pd.DataFrame(
            {
                "signal": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "duplicate_signal": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
                "independent": [0.0, 1.0, 4.0, 2.0, 5.0, 3.0],
            }
        )

        report = dataset_profile(frame)

        diagnostic = report["multicollinearity"]
        self.assertEqual(diagnostic["complete_case_rows"], 6)
        self.assertIsNone(diagnostic["condition_number"])
        self.assertEqual(diagnostic["condition_number_reason"], "rank_deficient")
        for column in ("signal", "duplicate_signal"):
            result = next(item for item in diagnostic["vif"] if item["feature"] == column)
            self.assertIsNone(result["value"])
            self.assertEqual(result["reason"], "perfect_or_near_linear_dependence")
        json.dumps(report, allow_nan=False)

    def test_dataset_profile_validates_group_column_selection_and_type(self):
        frame = pd.DataFrame(
            {"category": ["A", "B"], "value": [1.0, 2.0], "target": [3.0, 4.0]}
        )

        with self.assertRaisesRegex(ValueError, "included in the selected features"):
            dataset_profile(frame, target="target", features=["value"], group_by="category")
        with self.assertRaisesRegex(ValueError, "must be categorical"):
            dataset_profile(frame, target="target", group_by="value")

    def test_dataset_profile_sampling_is_deterministic_and_wide_data_requires_selection(self):
        frame = pd.DataFrame(
            {
                "feature": np.arange(100),
                "other": np.arange(100) % 4,
                "target": np.arange(100) % 2,
            }
        )
        first = dataset_profile(frame, target="target", max_rows=20, random_state=17)
        second = dataset_profile(frame, target="target", max_rows=20, random_state=17)

        self.assertEqual(first["sampled_rows"], 20)
        self.assertTrue(first["row_sampling"]["truncated"])
        self.assertEqual(first["columns"], second["columns"])
        self.assertEqual(first["target_associations"], second["target_associations"])
        with self.assertRaisesRegex(ValueError, "select features explicitly"):
            dataset_profile(frame, target="target", max_features=1)
        selected = dataset_profile(
            frame,
            target="target",
            features=["other"],
            max_features=1,
        )
        self.assertEqual(selected["feature_count"], 1)

    def test_dataset_profile_rejects_bad_inputs_and_supports_small_frames(self):
        frame = pd.DataFrame({"value": [1.0], "target": [None]})

        report = dataset_profile(frame, target="target")

        self.assertEqual(report["target"]["distribution"]["count"], 0)
        self.assertEqual(report["target"]["distribution"]["unique"], 0)
        self.assertEqual(report["target"]["distribution"]["top"], [])
        self.assertEqual(report["target"]["missing_rate"], 1.0)
        with self.assertRaisesRegex(ValueError, "target column"):
            dataset_profile(frame, target="missing")
        with self.assertRaisesRegex(ValueError, "between 1 and 10000"):
            dataset_profile(frame, max_rows=10001)

    def test_drift_report_separates_schema_quality_and_distribution_changes(self):
        reference = pd.DataFrame(
            {"amount": np.arange(100, dtype=float), "segment": ["a", "b"] * 50}
        )
        current = pd.DataFrame(
            {"amount": np.arange(100, dtype=float) + 100.0, "segment": ["a"] * 100}
        )

        report = drift_report(
            reference,
            current,
            model_version="production-v1",
            reference_dataset_fingerprint="train-v1",
            current_dataset_fingerprint="live-v2",
        )

        self.assertEqual(report["model_version"], "production-v1")
        self.assertEqual(report["reference_dataset_fingerprint"], "train-v1")
        self.assertEqual(report["schema_changes"]["missing_columns"], [])
        self.assertEqual(report["data_quality"]["missingness"]["amount"]["delta"], 0.0)
        self.assertGreater(report["drift"]["numeric"]["amount"]["value"], 0.0)
        self.assertGreater(report["drift"]["categorical"]["segment"]["value"], 0.0)
        self.assertTrue(report["alerts"])

    def test_drift_report_reports_schema_changes_separately(self):
        reference = pd.DataFrame({"feature": [1.0, 2.0], "category": ["a", "b"]})
        current = pd.DataFrame({"category": ["a", "c"], "new_feature": [1, 2]})

        report = drift_report(reference, current)

        self.assertEqual(report["schema_changes"]["missing_columns"], ["feature"])
        self.assertEqual(report["schema_changes"]["extra_columns"], ["new_feature"])
        self.assertNotIn("feature", report["drift"]["numeric"])

    def test_monitoring_reports_range_and_prediction_drift(self):
        reference = pd.DataFrame({"feature": np.arange(100, dtype=float)})
        current = pd.DataFrame({"feature": np.arange(100, dtype=float) + 100.0})

        report = drift_report(reference, current, range_threshold=0.5)
        prediction_report = prediction_distribution_report(
            np.zeros(100), np.ones(100), model_version="model-v1"
        )

        self.assertEqual(report["drift"]["range"]["feature"]["out_of_range_rate"], 1.0)
        self.assertTrue(any(alert["type"] == "range_drift" for alert in report["alerts"]))
        self.assertEqual(prediction_report["model_version"], "model-v1")
        self.assertTrue(prediction_report["alerts"])

    def test_monitoring_updates_delayed_label_metrics_and_runtime_summary(self):
        performance = delayed_label_performance([0, 1, 1], [0, 0, 1], model_version="model-v1")
        monitor = InferenceMonitor("model-v1")
        monitor.record(10.0, 100)
        monitor.record(30.0, 50, success=False)
        summary = monitor.summary()

        self.assertEqual(performance["metrics"]["accuracy"], 2 / 3)
        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["error_rate"], 0.5)
        self.assertEqual(summary["rows"], 150)
        self.assertEqual(summary["throughput_rows_per_second"], 3750.0)

    def test_phase8_bundle_tracks_audit_events_and_ownership(self):
        model = DummyClassifier(strategy="most_frequent").fit(
            np.array([[0.0], [1.0]]), np.array([0, 1])
        )
        bundle = ModelBundle(
            model,
            owner="alice",
            run_id="run-42",
            sensitive_features=["ssn"],
            metadata={"api_key": "super-secret"},
        )

        bundle.record_audit_event("training", dataset_fingerprint="fp-1", status="ok")

        self.assertEqual(bundle.owner, "alice")
        self.assertEqual(bundle.run_id, "run-42")
        self.assertEqual(bundle.sensitive_features, ["ssn"])
        self.assertEqual(bundle.audit_events[0]["event_type"], "training")
        self.assertEqual(bundle.audit_events[0]["details"]["api_key"], "[REDACTED]")
        self.assertEqual(bundle.audit_events[0]["details"]["dataset_fingerprint"], "fp-1")

    def test_phase8_registry_records_promotions_and_audit_history(self):
        model = DummyClassifier(strategy="most_frequent").fit(
            np.array([[0.0], [1.0]]), np.array([0, 1])
        )
        bundle = ModelBundle(model, task_name="governance-demo", owner="alice")

        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(directory)
            model_version = registry.register(bundle)
            registry.promote("governance-demo", model_version, "staging")
            audit = registry.audit_log("governance-demo")

        self.assertEqual(audit[0]["event_type"], "register")
        self.assertTrue(any(event["event_type"] == "promote" and event["to"] == "staging" for event in audit))

    def test_sensitive_feature_report_flags_pii_and_sensitive_columns(self):
        frame = pd.DataFrame({"email": ["a@example.com", "b@example.com"], "ssn": ["123", "456"], "amount": [1.0, 2.0]})

        report = sensitive_feature_report(frame, sensitive_features=["ssn"])

        self.assertIn("email", report["pii_columns"])
        self.assertIn("ssn", report["sensitive_columns"])
        self.assertTrue(report["warnings"])

    def test_sensitive_feature_report_scans_bounded_values_without_returning_them(self):
        private_email = "person@example.org"
        private_ssn = "123-45-6789"
        private_phone = "(212) 555-0199"
        frame = pd.DataFrame(
            {
                0: [private_email, "not-an-email", "later@example.org"],
                "reference": [private_ssn, "invoice 1234567890", "000-00-0000"],
                "comment": [private_phone, "order 2125550199", "555-55-5555"],
            }
        )

        report = sensitive_feature_report(frame, max_scan_rows=2)

        findings = {
            item["column"]: item["pattern_counts"]
            for item in report["value_findings"]
        }
        self.assertEqual(findings[0], {"email": 1})
        self.assertEqual(findings["reference"], {"ssn": 1})
        self.assertEqual(findings["comment"], {"phone": 1})
        self.assertEqual(
            report["value_scan"],
            {"rows_scanned": 2, "row_limit": 2, "truncated": True},
        )
        rendered_report = repr(report)
        for private_value in (private_email, private_ssn, private_phone):
            self.assertNotIn(private_value, rendered_report)

        false_positive_frame = pd.DataFrame(
            {"Unnamed: 0": ["order 1234567890", "ordinary text", None]}
        )
        false_positive_report = sensitive_feature_report(false_positive_frame)
        self.assertEqual(false_positive_report["value_findings"], [])
        self.assertEqual(false_positive_report["pii_columns"], [])

        with self.assertRaisesRegex(ValueError, "between 1 and 10000"):
            sensitive_feature_report(frame, max_scan_rows=10001)

    def test_registry_retention_deletes_artifacts_by_task_model_and_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(directory)
            dataset_path = Path(directory) / "dataset.csv"
            prediction_path = Path(directory) / "prediction.csv"
            dataset_path.write_text("feature\n1\n2\n", encoding="utf-8")
            prediction_path.write_text("prediction\n0\n1\n", encoding="utf-8")

            registry.register_dataset(
                "retention-demo",
                "dataset-fp-1",
                dataset_path,
                metadata={"rows": 2},
            )
            registry.register_prediction(
                "retention-demo",
                "model-v1",
                "dataset-fp-1",
                prediction_path,
                prediction_id="pred-1",
            )

            self.assertTrue((Path(directory) / "retention-demo" / "datasets" / "dataset-fp-1.json").exists())
            self.assertTrue((Path(directory) / "retention-demo" / "predictions" / "model-v1" / "dataset-fp-1" / "pred-1.csv").exists())

            deleted = registry.delete_artifacts(
                task_name="retention-demo",
                model_version="model-v1",
                dataset_fingerprint="dataset-fp-1",
            )

            self.assertTrue(deleted["datasets"])
            self.assertTrue(deleted["predictions"])
            self.assertFalse((Path(directory) / "retention-demo" / "datasets" / "dataset-fp-1.json").exists())
            self.assertFalse((Path(directory) / "retention-demo" / "predictions" / "model-v1" / "dataset-fp-1" / "pred-1.csv").exists())

    def test_phase9_local_backend_persists_trial_state_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = LocalProcessExecutionBackend(directory, task_name="phase9-demo")

            trial_id = backend.submit_trial(
                {"value": 2},
                fn=lambda params: params["value"] * 3,
                worker_id="worker-a",
            )

            self.assertEqual(backend.get_trial(trial_id)["status"], "queued")

            backend.mark_running(trial_id, worker_id="worker-b")
            backend.mark_failed(trial_id, error="crash-before-result")

            resumed = LocalProcessExecutionBackend(directory, task_name="phase9-demo").resume_incomplete_trials()
            self.assertEqual(len(resumed), 1)
            self.assertEqual(resumed[0]["trial_id"], trial_id)

            result = resumed[0]["fn"](resumed[0]["params"])
            self.assertEqual(result, 6)
            backend.mark_completed(trial_id, result=result)
            self.assertEqual(backend.get_trial(trial_id)["status"], "completed")
            self.assertEqual(backend.get_trial(trial_id)["result"], 6)

    def test_phase9_local_backend_executes_pending_trials_with_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = LocalProcessExecutionBackend(directory, task_name="phase9-worker")

            success_id = backend.submit_trial({"value": 3}, fn=lambda params: params["value"] * 2)
            failed_id = backend.submit_trial({"value": 0}, fn=lambda params: 1 / params["value"])

            executed = backend.execute_pending(worker_id="worker-c")

            self.assertEqual(len(executed), 2)
            self.assertEqual(backend.get_trial(success_id)["status"], "completed")
            self.assertEqual(backend.get_trial(success_id)["result"], 6)
            self.assertEqual(backend.get_trial(failed_id)["status"], "failed")
            self.assertIn("ZeroDivisionError", backend.get_trial(failed_id)["error"])

    def test_phase9_local_backend_claims_and_lists_completed_results(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = LocalProcessExecutionBackend(directory, task_name="phase9-queue")

            pending_id = backend.submit_trial({"value": 7}, fn=lambda params: params["value"] + 1)

            self.assertEqual(backend.claim_trial(pending_id, worker_id="worker-d")["status"], "running")
            self.assertIsNone(backend.claim_trial(pending_id, worker_id="worker-e"))

            backend.mark_completed(pending_id, result=8, worker_id="worker-d")
            result_rows = backend.list_results()

            self.assertEqual(len(result_rows), 1)
            self.assertEqual(result_rows[0]["result"], 8)
            self.assertEqual(result_rows[0]["worker_id"], "worker-d")

    def test_phase9_local_backend_claims_oldest_pending_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = LocalProcessExecutionBackend(directory, task_name="phase9-order")

            first = backend.submit_trial({"value": 1}, fn=lambda params: params["value"])
            second = backend.submit_trial({"value": 2}, fn=lambda params: params["value"])

            first_claim = backend.claim_next_trial(worker_id="worker-x")
            self.assertEqual(first_claim["trial_id"], first)
            self.assertEqual(backend.list_pending()[0]["trial_id"], second)
            self.assertEqual(len(backend.list_pending()), 1)

    def test_phase9_dask_backend_dispatches_and_persists_callback_result(self):
        class ImmediateFuture:
            def __init__(self, value):
                self.value = value

            def add_done_callback(self, callback):
                callback(self)

            def result(self):
                return self.value

        class FakeClient:
            def submit(self, fn, params):
                return ImmediateFuture(fn(params))

        with tempfile.TemporaryDirectory() as directory:
            backend = DaskExecutionBackend(directory, task_name="phase9-dask", client=FakeClient())
            trial_id = backend.submit_trial({"value": 4}, fn=lambda params: params["value"] * 5)

            scheduled = backend.execute_pending(worker_id="dask-worker")

            self.assertEqual(scheduled[0]["trial_id"], trial_id)
            self.assertEqual(backend.get_trial(trial_id)["status"], "completed")
            self.assertEqual(backend.get_trial(trial_id)["result"], 20)
            self.assertEqual(backend.get_trial(trial_id)["worker_id"], "dask-worker")

    def test_phase9_ci_quality_gates_validate_schema_metrics_and_size(self):
        model = DummyClassifier(strategy="most_frequent").fit(
            np.array([[0.0], [1.0]]), np.array([0, 1])
        )
        bundle = ModelBundle(
            model,
            schema={"columns": ["feature"]},
            metrics={"accuracy": 0.92},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = save_bundle(bundle, Path(directory) / "model.bundle")
            report = check_bundle_quality(
                path,
                expected_schema={"columns": ["feature"]},
                metric_thresholds={"accuracy": 0.90},
                max_bundle_size_bytes=path.stat().st_size,
            )
            self.assertTrue(report["passed"])
            self.assertTrue(assert_bundle_quality(path, metric_thresholds={"accuracy": 0.90})["passed"])

            failed = check_bundle_quality(
                path,
                expected_schema={"columns": ["other"]},
                metric_thresholds={"accuracy": 0.95},
                max_bundle_size_bytes=1,
            )
            self.assertFalse(failed["passed"])
            with self.assertRaises(BundleQualityGateError):
                assert_bundle_quality(path, metric_thresholds={"accuracy": 0.95})

    def test_phase9_retraining_entry_points_register_and_promote_bundles(self):
        model = DummyClassifier(strategy="most_frequent").fit(
            np.array([[0.0], [1.0]]), np.array([0, 1])
        )
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(directory)
            calls = []

            def trainer(context):
                calls.append(context)
                return ModelBundle(model, metrics={"accuracy": 1.0})

            job = RetrainingJob(
                trainer,
                registry,
                "phase9-retraining",
                promotion_state="staging",
            )
            skipped = job.run_scheduled(lambda context: context["due"], context={"due": False})
            result = job.run_on_data(lambda context: context["new_data"], context={"new_data": True})

            self.assertEqual(skipped["status"], "skipped")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["promotion_state"], "staging")
            self.assertTrue(result["quality_report"]["passed"])
            self.assertEqual(calls, [{"new_data": True}])
            self.assertEqual(registry.audit_log("phase9-retraining")[-1]["to"], "staging")

    def test_phase9_retraining_quality_threshold_blocks_registration_and_promotion(self):
        model = DummyClassifier(strategy="most_frequent").fit(
            np.array([[0.0], [1.0]]), np.array([0, 1])
        )
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(directory)
            job = RetrainingJob(
                lambda context: ModelBundle(model, metrics={"accuracy": 0.7}),
                registry,
                "quality-gated-retraining",
                promotion_state="staging",
                quality_metric_thresholds={"accuracy": 0.8},
            )

            with self.assertRaisesRegex(ValueError, "failed quality gates"):
                job.run(context={})
            self.assertFalse((Path(directory) / "quality-gated-retraining").exists())

    def test_phase9_deployment_gate_requires_approval_and_preserves_rollback_audit(self):
        model = DummyClassifier(strategy="most_frequent").fit(
            np.array([[0.0], [1.0]]), np.array([0, 1])
        )
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(directory)
            first = registry.register(ModelBundle(model, task_name="phase9-deploy"))
            second = registry.register(ModelBundle(model, task_name="phase9-deploy"))
            gate = DeploymentApprovalGate(registry, required_approvals=2)

            with self.assertRaises(PermissionError):
                gate.promote("phase9-deploy", first, "production", approvers=["alice"])
            with self.assertRaises(ValueError):
                gate.promote(
                    "phase9-deploy",
                    first,
                    "production",
                    approvers=["alice", "bob"],
                    quality_report={"passed": False},
                )
            with self.assertRaisesRegex(ValueError, "quality report is required"):
                gate.promote(
                    "phase9-deploy",
                    first,
                    "production",
                    approvers=["alice", "bob"],
                )

            promoted = gate.promote(
                "phase9-deploy",
                first,
                "production",
                approvers=["alice", "bob", "alice"],
                quality_report={"passed": True},
            )
            gate.promote(
                "phase9-deploy",
                second,
                "production",
                approvers=["carol", "dana"],
                quality_report={"passed": True},
            )
            rollback = gate.rollback("phase9-deploy", "production", model_version=first)
            audit = registry.audit_log("phase9-deploy")

            self.assertEqual(promoted["status"], "promoted")
            self.assertEqual(registry.load_version("phase9-deploy", first).model_version, first)
            self.assertEqual(rollback["status"], "rolled_back")
            self.assertEqual(sum(event["event_type"] == "approval" for event in audit), 4)
            self.assertEqual(audit[-1]["event_type"], "rollback")

    def test_group_fairness_report_detects_selection_gap(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 0, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        report = fairness_report(y_true, y_pred, groups)

        self.assertAlmostEqual(report["groups"]["A"]["selection_rate"], 0.5)
        self.assertAlmostEqual(report["groups"]["B"]["selection_rate"], 0.25)
        self.assertGreater(report["fairness"]["demographic_parity_gap"], 0.0)
        self.assertGreater(report["fairness"]["equal_opportunity_gap"], 0.0)

    def test_subgroup_performance_report_compares_accuracy_by_group(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 0, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        report = subgroup_performance_report(y_true, y_pred, groups, metric="accuracy")

        self.assertAlmostEqual(report["groups"]["A"]["accuracy"], 1.0)
        self.assertAlmostEqual(report["groups"]["B"]["accuracy"], 0.75)
        self.assertAlmostEqual(report["fairness_gap"], 0.25)

    def test_batch_prediction_validates_schema_and_preserves_model_metadata(self):
        model = DummyClassifier(strategy="most_frequent").fit(
            np.array([[0.0], [1.0], [2.0]]), np.array([0, 1, 0])
        )
        bundle = ModelBundle(
            model,
            model_version="batch-v1",
            schema={"feature": {"dtype": "float64", "categorical_values": []}},
            dependencies={},
        )
        frame = pd.DataFrame({"feature": np.array([3.0, 4.0])})

        result = predict_batch(bundle, frame, request_id="request-1")

        self.assertListEqual(result["prediction"].tolist(), [0, 0])
        self.assertEqual(result["model_version"].unique().tolist(), ["batch-v1"])
        self.assertEqual(result["request_id"].unique().tolist(), ["request-1"])
        self.assertEqual(result.attrs["inference_metrics"]["rows"], 2)
        self.assertGreaterEqual(result.attrs["inference_metrics"]["latency_ms"], 0.0)
        self.assertGreaterEqual(
            result.attrs["inference_metrics"]["throughput_rows_per_second"], 0.0
        )

        with self.assertRaises(SchemaValidationError) as context:
            predict_batch(bundle, pd.DataFrame({"wrong": [3.0]}))
        self.assertIn("missing columns", context.exception.errors[0])

    def test_batch_prediction_writes_csv_atomically(self):
        model = DummyClassifier(strategy="most_frequent").fit(
            np.array([[0.0], [1.0]]), np.array([0, 1])
        )
        bundle = ModelBundle(model, schema={"feature": {"dtype": "float64"}}, dependencies={})
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "predictions.csv"
            predict_batch(bundle, pd.DataFrame({"feature": [2.0]}), output_path=output_path)
            output = pd.read_csv(output_path)

        self.assertEqual(output.loc[0, "prediction"], 0)
        self.assertIn("request_id", output)

    def test_batch_prediction_cli_uses_same_inference_path(self):
        model = DummyClassifier(strategy="most_frequent").fit(
            np.array([[0.0], [1.0]]), np.array([0, 1])
        )
        bundle = ModelBundle(model, model_version="cli-v1", dependencies={})
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = Path(directory) / "model.bundle"
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "output.csv"
            save_bundle(bundle, bundle_path)
            pd.DataFrame({"feature": [2.0]}).to_csv(input_path, index=False)

            self.assertEqual(
                batch_main(
                    [
                        "--bundle", str(bundle_path),
                        "--input", str(input_path),
                        "--output", str(output_path),
                        "--request-id", "cli-request",
                        "--no-strict-dependencies",
                    ]
                ),
                0,
            )
            output = pd.read_csv(output_path)

        self.assertEqual(output.loc[0, "model_version"], "cli-v1")
        self.assertEqual(output.loc[0, "request_id"], "cli-request")

    def test_http_adapter_exposes_health_readiness_and_prediction(self):
        from io import BytesIO
        from wsgiref.util import setup_testing_defaults

        model = DummyClassifier(strategy="most_frequent").fit(
            np.array([[0.0], [1.0]]), np.array([0, 1])
        )
        application = create_app(
            ModelBundle(
                model,
                model_version="http-v1",
                schema={"feature": {"dtype": "float64"}},
                dependencies={},
            ),
            strict_dependencies=False,
        )

        def request(method, path, payload=None):
            environment = {}
            setup_testing_defaults(environment)
            environment.update({"REQUEST_METHOD": method, "PATH_INFO": path})
            encoded = json.dumps(payload).encode("utf-8") if payload is not None else b""
            environment["CONTENT_LENGTH"] = str(len(encoded))
            environment["wsgi.input"] = BytesIO(encoded)
            status = []
            body = b"".join(application(environment, lambda value, headers: status.append(value)))
            return status[0], json.loads(body)

        health_status, health = request("GET", "/health")
        ready_status, ready = request("GET", "/ready")
        predict_status, prediction = request(
            "POST", "/predict", {"data": [{"feature": 2.0}], "request_id": "http-request"}
        )

        self.assertEqual(health_status, "200 OK")
        self.assertEqual(health["status"], "ok")
        self.assertEqual(ready_status, "200 OK")
        self.assertEqual(ready["model_version"], "http-v1")
        self.assertEqual(predict_status, "200 OK")
        self.assertEqual(prediction["predictions"], [0])
        self.assertEqual(prediction["request_id"], "http-request")
        self.assertEqual(prediction["metrics"]["rows"], 1)
        self.assertGreaterEqual(prediction["metrics"]["latency_ms"], 0.0)

    def test_model_bundle_round_trip_preserves_predictions_and_metadata(self):
        model = LogisticRegression(max_iter=100).fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
        bundle = ModelBundle(
            model,
            task_name="bundle-test",
            schema={"feature": {"dtype": "float64"}},
            metrics={"accuracy": 1.0},
            dataset_fingerprint="dataset-1",
            config_version="config-1",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_bundle(bundle, Path(directory) / "model.bundle")
            loaded = load_bundle(path, expected_schema=bundle.schema)

        np.testing.assert_array_equal(bundle.predict([[0.0], [3.0]]), loaded.predict([[0.0], [3.0]]))
        self.assertEqual(loaded.dataset_fingerprint, "dataset-1")
        self.assertEqual(loaded.config_version, "config-1")

    def test_model_bundle_rejects_incompatible_schema(self):
        bundle = ModelBundle(
            DummyClassifier(strategy="most_frequent").fit(np.array([[0.0]]), np.array([1])),
            schema={"a": "float"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_bundle(bundle, Path(directory) / "model.bundle")
            with self.assertRaisesRegex(ValueError, "schema is incompatible"):
                load_bundle(path, expected_schema={"b": "float"})

    def test_model_bundle_exports_mlflow_compatible_layout(self):
        model = DummyClassifier(strategy="most_frequent").fit(
            np.array([[0.0], [1.0]]), np.array([0, 1])
        )
        bundle = ModelBundle(model, task_name="mlflow-test", model_version="v1", dependencies={})
        with tempfile.TemporaryDirectory() as directory:
            export_path = export_mlflow(bundle, Path(directory) / "exported-model")
            self.assertTrue((export_path / "MLmodel").is_file())
            self.assertTrue((export_path / "model.pkl").is_file())
            self.assertTrue((export_path / "bundle.json").is_file())
            exported_model = __import__("joblib").load(export_path / "model.pkl")

        np.testing.assert_array_equal(
            model.predict(np.array([[0.0], [1.0]])),
            exported_model.predict(np.array([[0.0], [1.0]])),
        )

    def test_model_registry_supports_versions_aliases_and_rollback_history(self):
        model = DummyClassifier(strategy="most_frequent").fit(
            np.array([[0.0], [1.0]]), np.array([0, 1])
        )
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(directory)
            first = ModelBundle(model, task_name="registry-test", model_version="v1", dependencies={})
            second = ModelBundle(model, task_name="registry-test", model_version="v2", dependencies={})
            registry.register(first)
            registry.register(second)
            self.assertEqual(
                registry.load("registry-test", "latest", strict_dependencies=False).model_version,
                "v2",
            )
            registry.promote("registry-test", "v1", "production")
            registry.promote("registry-test", "v2", "production")
            registry.rollback("registry-test", "production", "v1")
            loaded = registry.load("registry-test", "production", strict_dependencies=False)
            history_length = len(registry.history("registry-test"))

        self.assertEqual(loaded.model_version, "v1")
        self.assertEqual(history_length, 3)

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

    def test_fold_local_preprocessor_learns_only_from_fit_partition(self):
        frame = pd.DataFrame(
            {
                "number": [1.0, np.nan, 3.0, 5.0, 999.0, 1000.0],
                "category": ["a", "b", "c", "d", "heldout", "heldout"],
                "date": pd.to_datetime(
                    [
                        "2020-01-01",
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-04",
                        "2030-01-01",
                        "2030-01-02",
                    ]
                ),
                "text": pd.Series(
                    [
                        "alpha beta",
                        "alpha gamma",
                        "beta gamma",
                        "alpha beta",
                        "heldouttoken",
                        "heldouttoken",
                    ],
                    dtype="string",
                ),
            }
        )
        train = frame.iloc[:4]
        heldout = frame.iloc[4:]
        preprocessor = FoldLocalDataPreprocessor().fit(train)

        fitted = preprocessor.transformer_
        numeric_imputer = fitted.named_transformers_["numeric"].named_steps["imputer"]
        self.assertEqual(numeric_imputer.statistics_[0], 3.0)
        categories = fitted.named_transformers_["categorical"].named_steps[
            "onehot"
        ].categories_[0]
        self.assertNotIn("heldout", categories)
        date_imputer = fitted.named_transformers_["datetime"].named_steps["imputer"]
        self.assertLess(date_imputer.statistics_[0], 1_893_456_000.0)
        text_vocabulary = fitted.named_transformers_["text_3"].named_steps[
            "tfidf"
        ].vocabulary_
        self.assertNotIn("heldouttoken", text_vocabulary)
        self.assertEqual(preprocessor.transform(heldout).shape[0], 2)

        with self.assertRaisesRegex(ValueError, "missing=.*number"):
            preprocessor.transform(heldout.drop(columns=["number"]))

    def test_fold_local_preprocessor_works_inside_cross_validation_pipeline(self):
        from sklearn.model_selection import cross_val_score

        X = pd.DataFrame(
            {
                "number": np.arange(12, dtype=float),
                "category": ["a", "b", "c"] * 4,
                "text": pd.Series(
                    ["alpha beta", "beta gamma", "alpha gamma"] * 4,
                    dtype="string",
                ),
            }
        )
        y = np.array([0, 1] * 6)
        pipeline = Pipeline(
            [
                ("preprocess", FoldLocalDataPreprocessor(scale_numeric=True)),
                ("classifier", LogisticRegression(max_iter=200)),
            ]
        )

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=KFold(n_splits=3, shuffle=False),
            error_score="raise",
        )

        self.assertEqual(scores.shape, (3,))

    def test_hilbert_regressor_rejects_rank_deficient_basis(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-1.0, 1.0, size=(10, 3))
        y = np.sin(X[:, 0]) + X[:, 1] ** 2 - X[:, 2]

        with self.assertRaisesRegex(ValueError, "linearly dependent"):
            HilbertRegressor(deg=3).fit(X, y)

    def test_hilbert_regressor_fits_full_rank_basis(self):
        X = np.array([[-1.0], [0.0], [1.0]])
        y = 2.0 * X[:, 0] + 1.0
        model = HilbertRegressor(deg=1).fit(X, y)

        np.testing.assert_allclose(model.predict(X), y, atol=1e-10)

    def test_aml_fits_categorical_encoder_within_each_cv_fold(self):
        from unittest.mock import patch
        from category_encoders.one_hot import OneHotEncoder

        X = np.array(
            [["a"], ["b"], ["a"], ["b"], ["c"], ["d"], ["c"], ["d"]],
            dtype=object,
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        fitted_categories = []
        original_fit = OneHotEncoder.fit

        def record_fit(encoder, values, target=None, **kwargs):
            fitted_categories.append(set(np.asarray(values)[:, 0]))
            return original_fit(encoder, values, target, **kwargs)

        automl = AML(
            config={"sklearn.linear_model.LogisticRegression": {}},
            cat_cols=[0],
            cv=KFold(n_splits=2, shuffle=False),
            n_jobs=1,
            verbose=0,
        )
        with patch.object(OneHotEncoder, "fit", new=record_fit):
            automl.fit(X, y)

        self.assertEqual(automl.num_features_upper_bound, X.shape[1] + X.shape[0])
        self.assertIn({"c", "d"}, fitted_categories)
        self.assertIn({"a", "b"}, fitted_categories)
        self.assertIn({"a", "b", "c", "d"}, fitted_categories)
        np.testing.assert_array_equal(automl.predict(X), y)
        unseen_prediction = automl.predict(np.array([["unseen"]], dtype=object))
        self.assertEqual(unseen_prediction.shape, (1,))

    def test_aml_preserves_dataframe_names_and_accepts_named_categorical_columns(self):
        X = pd.DataFrame(
            {"measurement_a": [0.0, 0.1, 0.9, 1.0], "measurement_b": [1.0, 0.8, 0.2, 0.0]}
        )
        y = np.array([0, 0, 1, 1])
        named_model = AML(
            config={"sklearn.linear_model.LogisticRegression": {}},
            length=1,
            cv=2,
            n_jobs=1,
            verbose=0,
        ).fit(X, y)

        estimator = named_model.best_estimator_.steps[-1][1]
        self.assertEqual(list(estimator.feature_names_in_), list(X.columns))

        categorical = pd.DataFrame(
            {
                "segment": ["a", "b", "a", "b", "c", "d", "c", "d"],
                "measurement": [0.0, 1.0, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            }
        )
        categorical_model = AML(
            config={"sklearn.linear_model.LogisticRegression": {}},
            cat_cols=["segment"],
            length=1,
            cv=KFold(n_splits=2, shuffle=False),
            n_jobs=1,
            verbose=0,
        ).fit(categorical, np.array([0, 1, 0, 1, 0, 1, 0, 1]))
        self.assertEqual(len(categorical_model.predict(categorical)), len(categorical))

    def test_aml_eoa_fits_categorical_encoder_within_each_cv_fold(self):
        from unittest.mock import patch
        from category_encoders.one_hot import OneHotEncoder

        X = np.array(
            [["a"], ["b"], ["a"], ["b"], ["c"], ["d"], ["c"], ["d"]],
            dtype=object,
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        fitted_categories = []
        original_fit = OneHotEncoder.fit

        def record_fit(encoder, values, target=None, **kwargs):
            fitted_categories.append(set(np.asarray(values)[:, 0]))
            return original_fit(encoder, values, target, **kwargs)

        class ImmediateEOA:
            def __init__(self, population, fitness, **kwargs):
                self.population = population
                self.fitness = fitness

            def __call__(self):
                self.fitness(self.population[:1])

        automl = AML(
            config={"sklearn.linear_model.LogisticRegression": {}},
            cat_cols=[0],
            cv=KFold(n_splits=2, shuffle=False),
            n_jobs=1,
            verbose=0,
        )
        with patch.object(OneHotEncoder, "fit", new=record_fit):
            with patch("SKSurrogate.eoa.EOA", ImmediateEOA):
                automl.eoa_fit(X, y)

        self.assertEqual(automl.num_features_upper_bound, X.shape[1] + X.shape[0])
        self.assertIn({"c", "d"}, fitted_categories)
        self.assertIn({"a", "b"}, fitted_categories)
        self.assertIn({"a", "b", "c", "d"}, fitted_categories)
        np.testing.assert_array_equal(automl.predict(X), y)

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
            tracker.plot_calibration_curve(model, "rf", fig_index=97, bins=5)
            calibration_split = tracker._split_cache[model.mltrack_id]
            tracker.plot_lift_curve(model, title="rf")
            second_split = tracker._split_cache[model.mltrack_id]

            self.assertIs(first_split[2], second_split[2])
            self.assertIs(first_split[2], calibration_split[2])
            self.assertIsNotNone(metrics["logloss"])
            self.assertEqual(len(tracker.allPlots(model.mltrack_id)), 3)
            import matplotlib.pyplot as plt

            plt.close("all")

    def test_mltrace_snake_case_aliases_preserve_legacy_methods(self):
        X, y = make_classification(n_samples=80, n_features=4, random_state=3)
        frame = np2df(np.column_stack((X, y)), ["f0", "f1", "f2", "f3", "target"])()

        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("snake-case-aliases", db_name=database.name)
            tracker.register_data(frame, "target")
            weights = tracker.feature_weights(weights=("variance",))
            model = tracker.log_model(DummyClassifier(strategy="most_frequent"), "dummy")

            self.assertIn("variance", weights.columns)
            self.assertEqual(len(tracker.all_models()), 1)
            self.assertTrue(tracker.top_features(2))
            self.assertEqual(tracker.get_metadata(), tracker.GetMetadata())
            tracker.update_metadata({"alias_test": True})
            self.assertTrue(tracker.get_metadata()["alias_test"])
            self.assertEqual(tracker.set_cv(2), tracker.get_cv_spec())
            self.assertEqual(len(tracker.all_tasks()), 1)
            tracker.preserve_model(model)
            self.assertEqual(len(tracker.all_preserved()), 1)
            self.assertEqual(type(tracker.recover_model(model.mltrack_id)), type(model))
            self.assertIs(mltrack.register_data, mltrack.RegisterData)
            self.assertIs(mltrack.log_model, mltrack.LogModel)
            self.assertIs(mltrack.feature_weights, mltrack.FeatureWeights)
            self.assertIs(mltrack.top_features, mltrack.TopFeatures)
            self.assertIs(mltrack.update_metadata, mltrack.UpdateMetadata)
            self.assertIs(mltrack.get_metadata, mltrack.GetMetadata)
            self.assertIs(mltrack.set_cv, mltrack.SetCV)
            self.assertIs(mltrack.get_cv_spec, mltrack.GetCVSpec)
            self.assertIs(mltrack.update_task, mltrack.UpdateTask)
            self.assertIs(mltrack.update_model, mltrack.UpdateModel)
            self.assertIs(mltrack.log_metrics, mltrack.LogMetrics)
            self.assertIs(mltrack.load_model, mltrack.LoadModel)
            self.assertIs(mltrack.all_models, mltrack.allModels)
            self.assertIs(mltrack.all_tasks, mltrack.allTasks)
            self.assertIs(mltrack.all_preserved, mltrack.allPreserved)
            self.assertIs(mltrack.preserve_model, mltrack.PreserveModel)
            self.assertIs(mltrack.recover_model, mltrack.RecoverModel)
            self.assertIs(mltrack.all_plots, mltrack.allPlots)
            self.assertIs(mltrack.load_plot, mltrack.LoadPlot)
            self.assertIs(mltrack.retrieve_weights, mltrack.RetrieveWeights)
            self.assertIs(mltrack.get_best, mltrack.getBest)
            self.assertIs(mltrack.stats, mltrack.Stats)
            tracker.close()

    def test_mltrace_reports_and_plots_single_target_regression_diagnostics(self):
        X = np.linspace(-2.0, 2.0, 80).reshape(-1, 1)
        y = 1.5 * X[:, 0] - 0.3 + np.sin(X[:, 0]) * 0.1
        frame = pd.DataFrame({"feature": X[:, 0], "target": y})

        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("regression-diagnostics", db_name=database.name)
            tracker.RegisterData(frame, "target")
            model = tracker.LogModel(LinearRegression(), "linear-regression")

            summary = tracker.regression_diagnostics(model)
            first_split = tracker._split_cache[model.mltrack_id]
            expected_predictions = model.predict(first_split[3])
            expected_residuals = first_split[5] - expected_predictions
            import matplotlib.pyplot as plt

            tracker.plot_regression_diagnostics(model, title="Test diagnostics")
            second_split = tracker._split_cache[model.mltrack_id]

            self.assertEqual(summary["n_samples"], len(first_split[3]))
            self.assertGreaterEqual(summary["mean_absolute_error"], 0.0)
            self.assertGreaterEqual(summary["root_mean_squared_error"], 0.0)
            self.assertAlmostEqual(
                summary["root_mean_squared_error"],
                np.sqrt(np.mean(expected_residuals**2)),
            )
            self.assertAlmostEqual(
                summary["residual_mean"], np.mean(expected_residuals)
            )
            self.assertAlmostEqual(
                summary["residual_std"], np.std(expected_residuals, ddof=1)
            )
            self.assertEqual(len(plt.gcf().axes), 4)
            self.assertIs(first_split[2], second_split[2])
            self.assertEqual(len(tracker.allPlots(model.mltrack_id)), 1)
            plt.close("all")

    def test_mltrace_permutation_importance_is_bounded_and_repeatable(self):
        rng = np.random.default_rng(23)
        signal = rng.normal(size=120)
        noise = rng.normal(size=120)
        frame = pd.DataFrame(
            {"signal": signal, "noise": noise, "target": 4.0 * signal + 0.05 * noise}
        )

        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("permutation-importance", db_name=database.name)
            tracker.RegisterData(frame, "target")
            model = tracker.LogModel(LinearRegression(), "linear-regression")
            result = tracker.permutation_importance(
                model,
                features=["signal", "noise"],
                max_samples=30,
                n_repeats=3,
                random_state=9,
            )
            repeated = tracker.permutation_importance(
                model,
                features=["signal", "noise"],
                max_samples=30,
                n_repeats=3,
                random_state=9,
            )

            self.assertEqual(result["reference"]["source"], "cached_test_split")
            self.assertEqual(result["reference"]["sampled_rows"], 30)
            self.assertEqual(result["reference"]["max_samples"], 30)
            self.assertEqual(result["reference"]["random_state"], 9)
            self.assertEqual(result["selected_features"], ["signal", "noise"])
            self.assertEqual(
                result["importances"]["feature"].tolist(),
                repeated["importances"]["feature"].tolist(),
            )
            np.testing.assert_allclose(
                result["importances"]["importance_mean"],
                repeated["importances"]["importance_mean"],
            )
            self.assertEqual(
                len(result["importances"].iloc[0]["importance_samples"]), 3
            )
            self.assertEqual(
                result["importances"].iloc[0]["feature"], "signal"
            )

            with self.assertRaisesRegex(ValueError, "exceeding max_features"):
                tracker.permutation_importance(model, max_features=1)
            with self.assertRaisesRegex(ValueError, "Unknown feature name"):
                tracker.permutation_importance(model, features=["unknown"])

            tracker.RegisterData(frame.iloc[:20], "target", partition="test")
            partition_result = tracker.permutation_importance(
                model,
                reference_partition="test",
                features=["signal"],
                max_samples=10,
                random_state=4,
            )
            self.assertEqual(partition_result["reference"]["source"], "partition")
            self.assertEqual(partition_result["reference"]["partition"], "test")
            self.assertEqual(partition_result["reference"]["available_rows"], 20)
            self.assertEqual(partition_result["reference"]["sampled_rows"], 10)

    def test_mltrace_plots_permutation_importance_and_partial_dependence_ice(self):
        rng = np.random.default_rng(31)
        signal = rng.normal(size=100)
        noise = rng.normal(size=100)
        frame = pd.DataFrame(
            {"signal": signal, "noise": noise, "target": 2.5 * signal + noise}
        )

        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("model-explanations", db_name=database.name)
            tracker.RegisterData(frame, "target")
            model = tracker.LogModel(LinearRegression(), "linear-regression")
            import matplotlib.pyplot as plt

            tracker.plot_permutation_importance(
                model,
                max_samples=35,
                n_repeats=2,
                random_state=3,
            )
            self.assertEqual(len(plt.gcf().axes), 1)
            tracker.plot_partial_dependence(
                model,
                ["signal"],
                kind="both",
                max_samples=35,
                ice_samples=8,
                grid_resolution=8,
                random_state=3,
            )
            tracker.plot_partial_dependence(
                model,
                [("signal", "noise")],
                kind="average",
                max_samples=35,
                grid_resolution=5,
                random_state=3,
            )

            plots = tracker.allPlots(model.mltrack_id)
            self.assertEqual(len(plots), 3)
            self.assertEqual(plots["title"].tolist().count("partial dependence"), 2)
            saved_partial_dependence = tracker.LoadPlot(plots.iloc[1]["plot_id"])
            self.assertGreater(len(saved_partial_dependence.axes), 0)
            self.assertGreaterEqual(
                max(len(axis.lines) for axis in saved_partial_dependence.axes), 2
            )
            with self.assertRaisesRegex(ValueError, "ICE is supported only"):
                tracker.plot_partial_dependence(
                    model,
                    [("signal", "noise")],
                    kind="both",
                    max_samples=20,
                )
            plt.close("all")

    def test_partial_dependence_can_leave_perfectly_correlated_feature_support(self):
        from sklearn.inspection import partial_dependence

        class DifferenceRegressor(RegressorMixin, BaseEstimator):
            def fit(self, X, y):
                self.n_features_in_ = np.asarray(X).shape[1]
                return self

            def predict(self, X):
                values = np.asarray(X)
                return np.square(values[:, 0] - values[:, 1])

        values = np.linspace(-1.0, 1.0, 101)
        reference = pd.DataFrame({"first": values, "second": values})
        model = DifferenceRegressor().fit(reference, np.zeros(len(reference)))

        observed_predictions = model.predict(reference)
        profile = partial_dependence(
            model,
            reference,
            [0],
            method="brute",
            grid_resolution=9,
        )

        self.assertEqual(float(np.max(observed_predictions)), 0.0)
        self.assertGreater(float(np.max(profile["average"])), 1.0)

    def test_mltrace_rejects_classifiers_for_regression_diagnostics(self):
        X, y = make_classification(n_samples=80, n_features=4, random_state=2)
        frame = np2df(np.column_stack((X, y)), ["f0", "f1", "f2", "f3", "target"])()

        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("classification-regression-diagnostics", db_name=database.name)
            tracker.RegisterData(frame, "target")
            model = tracker.LogModel(
                RandomForestClassifier(n_estimators=5, random_state=2), "classifier"
            )

            with self.assertRaisesRegex(ValueError, "require a regression estimator"):
                tracker.regression_diagnostics(model)

    def test_mltrace_plots_multiclass_roc_with_ovr_and_micro_average(self):
        X, y = make_classification(
            n_samples=120,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=4,
        )
        frame = np2df(
            np.column_stack((X, y)),
            ["f0", "f1", "f2", "f3", "f4", "f5", "target"],
        )()

        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("multiclass-roc", db_name=database.name)
            tracker.RegisterData(frame, "target")
            model = tracker.LogModel(
                RandomForestClassifier(n_estimators=5, random_state=4), "multiclass"
            )
            import matplotlib.pyplot as plt

            tracker.plot_roc_curve(model, label="RF")
            axes = plt.gcf().axes

            self.assertEqual(len(axes), 1)
            labels = [line.get_label() for line in axes[0].lines]
            self.assertEqual(len(labels), 5)
            self.assertTrue(any("micro-average" in label for label in labels))
            self.assertTrue(any("AUC=" in label for label in labels))
            self.assertEqual(len(tracker.allPlots(model.mltrack_id)), 1)
            plt.close("all")

    def test_mltrace_plots_multiclass_calibration_one_vs_rest(self):
        X, y = make_classification(
            n_samples=120,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=5,
        )
        frame = np2df(
            np.column_stack((X, y)),
            ["f0", "f1", "f2", "f3", "f4", "f5", "target"],
        )()

        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("multiclass-calibration", db_name=database.name)
            tracker.RegisterData(frame, "target")
            model = tracker.LogModel(
                RandomForestClassifier(n_estimators=5, random_state=5), "multiclass"
            )
            import matplotlib.pyplot as plt

            tracker.plot_calibration_curve(model, "RF", fig_index=98, bins=5)
            axes = plt.figure(98).axes

            self.assertEqual(len(axes), 2)
            self.assertEqual(len(axes[0].lines), 4)
            self.assertGreater(len(axes[1].patches), 0)
            self.assertEqual(len(tracker.allPlots(model.mltrack_id)), 1)
            plt.close("all")

    def test_mltrace_compares_repeated_cv_scores_with_paired_bootstrap(self):
        reference = {"repeated_outer_scores": [0.70, 0.80, 0.75, 0.65]}
        candidate = {"repeated_outer_scores": [0.75, 0.78, 0.80, 0.70]}

        result = mltrack.compare_repeated_cv_results(
            reference, candidate, n_resamples=2000, random_state=17
        )
        repeated_result = mltrack.compare_repeated_cv_results(
            reference, candidate, n_resamples=2000, random_state=17
        )
        differences = np.array([0.05, -0.02, 0.05, 0.05])

        self.assertEqual(result, repeated_result)
        self.assertEqual(result["repeat_count"], 4)
        self.assertEqual(result["comparability"]["status"], "unverified")
        self.assertIn("evaluation_protocol", result["comparability"]["unverified_fields"])
        self.assertAlmostEqual(result["mean_difference"], float(np.mean(differences)))
        self.assertAlmostEqual(
            result["cohens_dz"],
            float(np.mean(differences) / np.std(differences, ddof=1)),
        )
        interval = result["confidence_interval"]
        self.assertLessEqual(interval["lower"], result["mean_difference"])
        self.assertGreaterEqual(interval["upper"], result["mean_difference"])

    def test_mltrace_paired_bootstrap_covers_seeded_iid_score_shift(self):
        rng = np.random.default_rng(2026)
        true_shift = 0.08
        covered = 0
        observed_differences = []

        for _ in range(100):
            reference = rng.normal(0.7, 0.15, 30)
            candidate = reference + true_shift + rng.normal(0.0, 0.06, 30)
            result = mltrack.compare_repeated_cv_results(
                {"repeated_outer_scores": reference.tolist()},
                {"repeated_outer_scores": candidate.tolist()},
                n_resamples=1000,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            interval = result["confidence_interval"]
            covered += interval["lower"] <= true_shift <= interval["upper"]
            observed_differences.append(result["mean_difference"])

        self.assertGreaterEqual(covered / 100, 0.90)
        self.assertLess(covered / 100, 1.0)
        self.assertAlmostEqual(float(np.mean(observed_differences)), true_shift, delta=0.02)

    def test_mltrace_paired_comparison_handles_constant_differences(self):
        result = mltrack.compare_repeated_cv_results(
            {"repeated_outer_scores": [0.70, 0.80, 0.75]},
            {"repeated_outer_scores": [0.80, 0.90, 0.85]},
            n_resamples=200,
            random_state=17,
        )

        self.assertAlmostEqual(result["mean_difference"], 0.1)
        self.assertIsNone(result["cohens_dz"])
        self.assertAlmostEqual(result["confidence_interval"]["lower"], 0.1)
        self.assertAlmostEqual(result["confidence_interval"]["upper"], 0.1)

    def test_mltrace_repeated_cv_comparison_rejects_mismatched_protocols(self):
        protocol = {
            "train_fingerprint": "train-a",
            "validation_fingerprint": "validation-a",
            "validation_used": True,
            "inner_cv": {"type": "n_splits", "value": 2},
            "outer_cv": {"type": "n_splits", "value": 2},
            "repeat_count": 2,
            "integer_split_seed": 42,
            "preprocessing": "none",
            "scoring": {"type": "sklearn_scorer", "verified": True},
            "group_fingerprint": None,
            "model_selection": "fixed_estimator_no_hyperparameter_search",
        }
        reference = {
            "repeated_outer_scores": [0.7, 0.8],
            "evaluation_protocol": protocol,
        }
        aligned_candidate = {
            "repeated_outer_scores": [0.75, 0.85],
            "evaluation_protocol": dict(protocol),
        }
        aligned_result = mltrack.compare_repeated_cv_results(
            reference, aligned_candidate, n_resamples=100, random_state=2
        )
        self.assertEqual(aligned_result["comparability"]["status"], "verified")

        candidate = {
            "repeated_outer_scores": [0.75, 0.85],
            "evaluation_protocol": {
                **protocol,
                "train_fingerprint": "train-b",
            },
        }

        with self.assertRaisesRegex(ValueError, "train_fingerprint"):
            mltrack.compare_repeated_cv_results(
                reference, candidate, n_resamples=100, random_state=2
            )

    def test_mltrace_paired_repeated_cv_comparison_rejects_unpaired_inputs(self):
        with self.assertRaisesRegex(ValueError, "at least two repeat scores"):
            mltrack.compare_repeated_cv_results(
                {"repeated_outer_scores": [0.7, 0.8]},
                {"repeated_outer_scores": [0.75]},
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            mltrack.compare_repeated_cv_results(
                {"repeated_outer_scores": [0.7, 0.8]},
                {"repeated_outer_scores": [0.75, 0.85]},
                n_resamples=0,
            )

    def test_split_conformal_regression_uses_disjoint_training_and_calibration_data(self):
        X_train = np.array([[0.0], [1.0]])
        y_train = np.array([1.0, 3.0])
        X_calibration = np.array([[10.0], [11.0], [12.0]])
        y_calibration = np.array([2.0, 3.0, 7.0])
        X_test = np.array([[20.0], [21.0]])

        result = split_conformal_regression_interval(
            MeanRecordingRegressor(),
            X_train,
            y_train,
            X_calibration,
            y_calibration,
            X_test,
            alpha=0.25,
        )

        np.testing.assert_array_equal(result["estimator"].fit_X_, X_train)
        np.testing.assert_array_equal(result["estimator"].fit_y_, y_train)
        np.testing.assert_array_equal(result["predictions"], [2.0, 2.0])
        np.testing.assert_array_equal(result["lower"], [-3.0, -3.0])
        np.testing.assert_array_equal(result["upper"], [7.0, 7.0])
        self.assertEqual(result["quantile"], 5.0)
        self.assertEqual(result["coverage_level"], 0.75)
        self.assertIn("exchangeability", result["coverage_assumption"])

    def test_prediction_interval_diagnostics_report_coverage_uncertainty_and_scores(self):
        result = prediction_interval_diagnostics(
            np.array([0.0, 2.0, 4.0, 10.0]),
            np.array([-1.0, 1.0, 1.0, 0.0]),
            np.array([1.0, 3.0, 3.0, 8.0]),
            alpha=0.2,
            confidence_level=0.95,
        )

        self.assertEqual(result["sample_size"], 4)
        self.assertEqual(result["covered_count"], 2)
        self.assertEqual(result["nominal_coverage"], 0.8)
        self.assertEqual(result["empirical_coverage"], 0.5)
        self.assertEqual(result["coverage_interval"]["method"], "Wilson score interval")
        self.assertLessEqual(result["coverage_interval"]["lower"], 0.5)
        self.assertGreaterEqual(result["coverage_interval"]["upper"], 0.5)
        self.assertEqual(result["mean_interval_width"], 3.5)
        self.assertEqual(result["median_interval_width"], 2.0)
        self.assertEqual(result["mean_interval_score"], 11.0)
        self.assertIn("independent held-out", result["diagnostic_assumption"])

    def test_prediction_interval_diagnostics_flag_unbounded_intervals(self):
        result = prediction_interval_diagnostics(
            np.array([2.0]),
            np.array([-np.inf]),
            np.array([np.inf]),
            alpha=0.1,
        )

        self.assertEqual(result["empirical_coverage"], 1.0)
        self.assertEqual(result["unbounded_interval_count"], 1)
        self.assertIsNone(result["mean_interval_width"])
        self.assertIsNone(result["mean_interval_score"])

    def test_prediction_interval_diagnostics_report_bounded_subgroup_coverage(self):
        result = prediction_interval_diagnostics(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0] * 5,
            [2.0] * 5,
            groups=["A", "A", "B", "B", None],
            max_groups=1,
        )

        subgroups = result["subgroup_coverage"]
        self.assertEqual(subgroups["missing_group_count"], 1)
        self.assertEqual(subgroups["other_group_count"], 2)
        self.assertEqual([item["group"] for item in subgroups["groups"]], ["A", "Other (combined)"])
        self.assertEqual(subgroups["groups"][0]["empirical_coverage"], 1.0)
        self.assertEqual(subgroups["groups"][1]["empirical_coverage"], 0.0)
        self.assertIn("does not imply conditional", subgroups["interpretation"])

    def test_prediction_interval_diagnostics_reject_malformed_intervals(self):
        with self.assertRaisesRegex(ValueError, "ordered bounds"):
            prediction_interval_diagnostics([0.0], [1.0], [0.0])
        with self.assertRaisesRegex(ValueError, "equal length"):
            prediction_interval_diagnostics([0.0, 1.0], [0.0], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            prediction_interval_diagnostics([0.0], [0.0], [1.0], groups=["A", "B"])

    def test_split_conformal_regression_corrects_small_calibration_quantile(self):
        result = split_conformal_regression_interval(
            MeanRecordingRegressor(),
            np.array([[0.0], [1.0]]),
            np.array([1.0, 3.0]),
            np.array([[10.0], [11.0]]),
            np.array([2.0, 3.0]),
            np.array([[20.0]]),
            alpha=0.1,
        )

        self.assertTrue(np.isinf(result["quantile"]))
        self.assertTrue(np.isneginf(result["lower"][0]))
        self.assertTrue(np.isposinf(result["upper"][0]))

    def test_split_conformal_quantile_interval_calibrates_disjoint_partitions(self):
        from sklearn.dummy import DummyRegressor

        X_train = np.array([[0.0], [1.0]])
        y_train = np.array([10.0, 20.0])
        X_calibration = np.arange(7.0, 14.0).reshape(-1, 1)
        y_calibration = np.array([-1.0, 0.2, 0.5, 1.5, 0.0, 1.0, 0.4])
        X_test = np.array([[20.0], [21.0]])

        result = split_conformal_quantile_interval(
            DummyRegressor(strategy="constant", constant=0.25),
            DummyRegressor(strategy="constant", constant=0.75),
            X_train,
            y_train,
            X_calibration,
            y_calibration,
            X_test,
            alpha=0.25,
        )

        np.testing.assert_array_equal(result["lower_estimator"].predict(X_test), [0.25, 0.25])
        np.testing.assert_array_equal(result["upper_estimator"].predict(X_test), [0.75, 0.75])
        np.testing.assert_array_equal(result["lower"], [-0.5, -0.5])
        np.testing.assert_array_equal(result["upper"], [1.5, 1.5])
        self.assertEqual(result["correction"], 0.75)
        self.assertEqual(result["coverage_level"], 0.75)
        self.assertEqual(result["calibration_size"], 7)
        self.assertIn("exchangeability", result["coverage_assumption"])

    def test_split_conformal_quantile_interval_fits_only_on_training_partition(self):
        X_train = np.array([[0.0], [1.0]])
        y_train = np.array([1.0, 3.0])
        result = split_conformal_quantile_interval(
            MeanRecordingRegressor(),
            MeanRecordingRegressor(),
            X_train,
            y_train,
            np.array([[10.0], [11.0], [12.0]]),
            np.array([2.0, 3.0, 7.0]),
            np.array([[20.0]]),
            alpha=0.25,
        )

        for key in ("lower_estimator", "upper_estimator"):
            np.testing.assert_array_equal(result[key].fit_X_, X_train)
            np.testing.assert_array_equal(result[key].fit_y_, y_train)

    def test_split_conformal_quantile_interval_runs_with_sklearn_quantile_regressors(self):
        from sklearn.linear_model import QuantileRegressor

        X = np.arange(30.0).reshape(-1, 1)
        y = 2.0 * X[:, 0] + np.sin(X[:, 0])
        result = split_conformal_quantile_interval(
            QuantileRegressor(quantile=0.1, alpha=0.0),
            QuantileRegressor(quantile=0.9, alpha=0.0),
            X[:15],
            y[:15],
            X[15:25],
            y[15:25],
            X[25:],
            alpha=0.2,
        )

        self.assertEqual(result["lower"].shape, (5,))
        self.assertEqual(result["upper"].shape, (5,))
        self.assertTrue(np.all(result["lower"] <= result["upper"]))
        self.assertEqual(result["lower_estimator"].quantile, 0.1)
        self.assertEqual(result["upper_estimator"].quantile, 0.9)

    def test_split_conformal_quantile_interval_orders_bounds_and_can_contract(self):
        from sklearn.dummy import DummyRegressor

        result = split_conformal_quantile_interval(
            DummyRegressor(strategy="constant", constant=8.0),
            DummyRegressor(strategy="constant", constant=2.0),
            np.array([[0.0], [1.0]]),
            np.array([0.0, 1.0]),
            np.arange(7.0).reshape(-1, 1),
            np.array([3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0]),
            np.array([[10.0]]),
            alpha=0.25,
        )

        self.assertEqual(result["correction"], -1.0)
        np.testing.assert_array_equal(result["lower"], [3.0])
        np.testing.assert_array_equal(result["upper"], [7.0])
        np.testing.assert_array_equal(result["empty_interval_mask"], [False])

    def test_split_conformal_quantile_interval_reports_empty_prediction_sets(self):
        from sklearn.base import BaseEstimator, RegressorMixin

        class XValueRegressor(RegressorMixin, BaseEstimator):
            def __init__(self, value):
                self.value = value

            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.asarray(X)[:, 0] * self.value

        result = split_conformal_quantile_interval(
            XValueRegressor(2.0),
            XValueRegressor(8.0),
            np.array([[1.0], [1.1]]),
            np.array([4.0, 8.0]),
            np.array([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]),
            np.full(7, 5.0),
            np.array([[0.1]]),
            alpha=0.25,
        )

        self.assertGreater(result["lower"][0], result["upper"][0])
        np.testing.assert_array_equal(result["empty_interval_mask"], [True])

    def test_split_conformal_quantile_interval_uses_infinite_correction_for_small_calibration(self):
        from sklearn.dummy import DummyRegressor

        result = split_conformal_quantile_interval(
            DummyRegressor(strategy="constant", constant=0.0),
            DummyRegressor(strategy="constant", constant=1.0),
            np.array([[0.0], [1.0]]),
            np.array([0.0, 1.0]),
            np.array([[2.0], [3.0]]),
            np.array([0.5, 0.5]),
            np.array([[4.0]]),
            alpha=0.1,
        )

        self.assertTrue(np.isinf(result["correction"]))
        self.assertTrue(np.isneginf(result["lower"][0]))
        self.assertTrue(np.isposinf(result["upper"][0]))

    def test_split_conformal_quantile_interval_validates_alpha_and_target_shape(self):
        from sklearn.dummy import DummyRegressor

        arguments = (
            DummyRegressor(),
            DummyRegressor(),
            np.array([[0.0], [1.0]]),
            np.array([0.0, 1.0]),
            np.array([[2.0], [3.0]]),
            np.array([0.5, 0.5]),
            np.array([[4.0]]),
        )
        with self.assertRaisesRegex(ValueError, "alpha must be between 0 and 1"):
            split_conformal_quantile_interval(*arguments, alpha=float("nan"))
        with self.assertRaisesRegex(ValueError, "single-target"):
            split_conformal_quantile_interval(
                *arguments[:3],
                np.array([[0.0, 1.0], [1.0, 2.0]]),
                *arguments[4:],
            )

    def test_mltrace_warns_when_eager_preprocessing_is_enabled(self):
        frame = pd.DataFrame(
            {"feature": [1.0, 2.0, 3.0, 4.0], "target": [0.0, 1.0, 0.0, 1.0]}
        )
        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("eager-preprocessing-warning", db_name=database.name, encode=True)
            with self.assertWarnsRegex(UserWarning, "fits preprocessing on the complete"):
                tracker.RegisterData(frame, "target")

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

    def test_unseen_float_values_are_not_treated_as_categories(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("float-schema", db_name=database.name)
            reference = pd.DataFrame(
                {"x": [0.1, 0.2, 0.3, 0.4], "label": ["a", "b", "a", "b"], "target": [0.0, 1.0, 0.0, 1.0]}
            )
            tracker.RegisterData(reference, "target")
            fresh = reference.copy()
            fresh["x"] = [0.15, 0.25, 0.35, 0.45]
            self.assertTrue(tracker.validate_data(fresh))
            fresh.loc[0, "label"] = "z"
            with self.assertRaises(ValueError):
                tracker.validate_data(fresh)

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

    def test_nested_cv_evaluation_preprocesses_mixed_features_within_each_fold(self):
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
                    "feature_b": ["validation-only", "y"],
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
            self.assertEqual(results["preprocessing"], "fold_local")
            self.assertEqual(
                results["evaluation_protocol"]["train_fingerprint"],
                tracker.dataset_splits()["train"]["fingerprint"],
            )
            self.assertTrue(results["evaluation_protocol"]["validation_used"])
            self.assertGreaterEqual(results["final_validation_score"], 0.0)
            self.assertLessEqual(results["mean_outer_score"], 1.0)

    def test_nested_cv_fold_local_pipeline_preserves_accuracy_fallback(self):
        class ModePredictor(BaseEstimator):
            def fit(self, X, y):
                self.value_ = pd.Series(y).mode().iloc[0]
                return self

            def predict(self, X):
                return np.repeat(self.value_, len(X))

        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("nested-cv-score-fallback", db_name=database.name)
            train = pd.DataFrame(
                {
                    "category": ["a", "b", "a", "b", "c", "a"],
                    "target": [0, 1, 0, 1, 0, 1],
                }
            )
            validation = pd.DataFrame(
                {"category": ["unseen", "b"], "target": [0, 1]}
            )
            tracker.RegisterData(train, "target", partition="train")
            tracker.RegisterData(validation, "target", partition="validation")

            results = tracker.nested_cv_evaluation(
                ModePredictor(),
                inner_cv=2,
                outer_cv=2,
                train_partition="train",
                validation_partition="validation",
            )

        self.assertEqual(results["preprocessing"], "fold_local")
        self.assertGreaterEqual(results["final_validation_score"], 0.0)

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
            self.assertEqual(results["preprocessing"], "fold_local")

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

    def test_aml_task_presets_are_bounded_and_fit_with_task_scoring(self):
        x_class, y_class = make_classification(
            n_samples=48,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            random_state=8,
        )
        x_reg = np.linspace(-2.0, 2.0, 48).reshape(-1, 1)
        y_reg = 3.0 * x_reg[:, 0] + np.sin(x_reg[:, 0])

        with tempfile.TemporaryDirectory() as checkpoint:
            classification = AML(
                preset="classification",
                check_point=checkpoint + "/",
                cv=2,
                min_random_evals=2,
                max_evals=5,
                n_jobs=1,
                random_state=8,
                verbose=0,
            )
            regression = AML(
                preset="regression",
                check_point=checkpoint + "/",
                cv=2,
                min_random_evals=2,
                max_evals=5,
                time_limit=30,
                n_jobs=1,
                random_state=8,
                verbose=0,
            )

            self.assertEqual(classification.length, 1)
            self.assertEqual(regression.length, 1)
            self.assertEqual(classification.scoring, "accuracy")
            self.assertEqual(regression.scoring, "neg_root_mean_squared_error")
            self.assertEqual(
                set(classification.config),
                {
                    "sklearn.ensemble.RandomForestClassifier",
                    "sklearn.ensemble.ExtraTreesClassifier",
                },
            )
            self.assertEqual(
                set(regression.config),
                {
                    "sklearn.ensemble.RandomForestRegressor",
                    "sklearn.ensemble.ExtraTreesRegressor",
                },
            )

            classification.fit(x_class, y_class)
            regression.fit(x_reg, y_reg)

            self.assertEqual(len(classification.predict(x_class[:3])), 3)
            self.assertEqual(len(regression.predict(x_reg[:3])), 3)
            self.assertEqual(regression.summary_["budget"]["max_evals"], 5)
            self.assertEqual(regression.summary_["budget"]["time_limit"], 30)

        preset = AML(preset="classification")
        self.assertEqual(preset.max_evals, 20)
        self.assertIsNone(preset.get_params()["config"])
        preset.set_params(preset="regression")
        self.assertEqual(preset.scoring, "neg_root_mean_squared_error")
        self.assertEqual(preset.config_types["sklearn.ensemble.RandomForestRegressor"], "regressor")
        preset.set_params(scoring="neg_mean_absolute_error")
        self.assertEqual(preset.scoring, "neg_mean_absolute_error")

        legacy = AML(config={})
        legacy.set_params(preset="classification")
        self.assertEqual(legacy.length, 1)
        self.assertEqual(legacy.max_evals, 20)
        self.assertEqual(legacy.scoring, "accuracy")

    def test_aml_presets_preprocess_dataframes_without_manual_cat_cols(self):
        n_samples = 36
        X = pd.DataFrame(
            {
                "signal": np.linspace(-2.0, 2.0, n_samples),
                "category": ["a", "b", "c"] * 12,
                "text": pd.Series(
                    ["warm sample", "cool sample", "stable sample"] * 12,
                    dtype="string",
                ),
            }
        )
        X.loc[::7, "signal"] = np.nan
        y_classification = np.array([0, 1] * (n_samples // 2))
        y_regression = (
            X["signal"].fillna(0.0).to_numpy()
            + X["category"].map({"a": 0.0, "b": 1.0, "c": 2.0}).to_numpy()
        )

        with tempfile.TemporaryDirectory() as checkpoint:
            for preset, target in (
                ("classification", y_classification),
                ("regression", y_regression),
            ):
                with self.subTest(preset=preset):
                    preset_checkpoint = Path(checkpoint, preset)
                    preset_checkpoint.mkdir()
                    automl = AML(
                        preset=preset,
                        check_point=str(preset_checkpoint) + "/",
                        cv=2,
                        min_random_evals=2,
                        max_evals=2,
                        n_jobs=1,
                        random_state=8,
                        verbose=0,
                    ).fit(X, target)

                    self.assertIn("preprocess", automl.best_estimator_.named_steps)
                    predictions = automl.predict(X)
                    self.assertEqual(len(predictions), n_samples)
                    self.assertTrue(np.all(np.isfinite(predictions)))

    def test_aml_preset_retains_explicit_cat_cols_path(self):
        X = pd.DataFrame(
            {
                "category": ["a", "b", "a", "b"] * 6,
                "measurement": np.linspace(0.0, 1.0, 24),
            }
        )
        y = np.array([0, 1] * 12)

        with tempfile.TemporaryDirectory() as checkpoint:
            automl = AML(
                preset="classification",
                cat_cols=["category"],
                check_point=checkpoint + "/",
                cv=2,
                min_random_evals=2,
                max_evals=2,
                n_jobs=1,
                random_state=8,
                verbose=0,
            ).fit(X, y)

        self.assertIn("categorical_encoder", automl.best_estimator_.named_steps)
        self.assertNotIn("preprocess", automl.best_estimator_.named_steps)
        self.assertEqual(len(automl.predict(X)), len(y))

    def test_aml_rejects_all_failed_candidate_pipelines(self):
        X = pd.DataFrame({"category": ["a", "b", "a", "b"] * 4})
        y = np.array([0, 1] * 8)
        automl = AML(
            config={"sklearn.linear_model.LogisticRegression": {"C": Real(0.1, 1.0)}},
            cv=2,
            min_random_evals=2,
            max_evals=2,
            n_jobs=1,
            random_state=8,
            verbose=0,
        )

        with self.assertRaisesRegex(
            ValueError, "All candidate pipelines failed to produce a finite CV score"
        ) as error:
            automl.fit(X, y)

        self.assertIn("could not convert string to float", str(error.exception))

    def test_aml_preset_rejects_unknown_or_conflicting_configuration(self):
        with self.assertRaisesRegex(ValueError, "preset must be"):
            AML(preset="fast")
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            AML(config={}, preset="classification")
        legacy = AML(config={})
        self.assertEqual(legacy.length, 5)
        self.assertEqual(legacy.max_evals, None)
        self.assertEqual(legacy.scoring, "accuracy")

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

    def test_surrogate_cv_never_refits_a_forbidden_combination(self):
        # In sklearn >= 1.8 the 'penalty' axis is gone: L1 (l1_ratio=1) with
        # solver='lbfgs' raises, so a refit of that banned cell would fail; the
        # search must fall back to (or skip) a feasible trial instead.
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([0, 1, 0, 1, 0, 1])
        search = SurrogateRandomCV(
            LogisticRegression(max_iter=100),
            {
                "solver": Categorical(["lbfgs", "liblinear"]),
                "l1_ratio": Categorical([0.0, 1.0]),
                "C": Real(0.1, 1.0),
            },
            forbidden=({"solver": "lbfgs", "l1_ratio": 1.0},),
            cv=2,
            n_jobs=1,
            max_iter=2,
            min_evals=1,
            max_evals=2,
            refit=True,
        )

        search.fit(X, y)

        best = search.best_estimator_.get_params()
        self.assertNotEqual((best["solver"], best["l1_ratio"]), ("lbfgs", 1.0))
        self.assertTrue(
            all(
                not (item["params"]["solver"] == "lbfgs" and item["params"]["l1_ratio"] == 1.0)
                for item in search.evaluation_history_
                if item["status"] == "complete"
            )
        )

    def test_surrogate_cv_rejects_unknown_search_parameters_before_evaluation(self):
        search = SurrogateRandomCV(
            LogisticRegression(max_iter=100),
            {"missing_parameter": Real(0.1, 1.0)},
            cv=2,
            max_iter=1,
            min_evals=1,
            refit=False,
        )

        with self.assertRaisesRegex(ValueError, "Unknown search parameter"):
            search.fit([[0.0], [1.0]], [0, 1])

    def test_surrogate_cv_supports_log_scaled_numeric_parameters(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 1, 0, 1])
        search = SurrogateRandomCV(
            LogisticRegression(max_iter=100),
            {"C": Real(1.0e-3, 1.0e3, scale="log")},
            cv=2,
            n_jobs=1,
            max_iter=1,
            min_evals=1,
            max_evals=1,
            refit=False,
        )

        search.fit(X, y)

        values = [item["params"]["C"] for item in search.evaluation_history_]
        self.assertTrue(all(1.0e-3 <= value <= 1.0e3 for value in values))
        self.assertGreater(search.best_estimator_.get_params()["C"], 0.0)

    def test_surrogate_cv_clamps_optimizer_category_coordinates(self):
        search = SurrogateRandomCV(
            LogisticRegression(max_iter=100),
            {"solver": Categorical(["lbfgs", "liblinear"])},
            cv=2,
            n_jobs=1,
            max_iter=1,
            min_evals=1,
            max_evals=1,
            refit=False,
        )

        search.fit([[0.0], [1.0], [2.0], [3.0]], [0, 1, 0, 1])

        self.assertIn(search.best_estimator_.get_params()["solver"], {"lbfgs", "liblinear"})

    def test_surrogate_cv_rejects_invalid_log_scaled_bounds(self):
        search = SurrogateRandomCV(
            LogisticRegression(max_iter=100),
            {"C": Real(0.0, 1.0, scale="log")},
            cv=2,
            max_iter=1,
            min_evals=1,
            refit=False,
        )

        with self.assertRaisesRegex(ValueError, "finite positive bounds"):
            search.fit([[0.0], [1.0]], [0, 1])

    def test_surrogate_cv_omits_inactive_conditional_parameters(self):
        search = SurrogateRandomCV(
            LogisticRegression(max_iter=100),
            {
                "solver": Categorical(["lbfgs"]),
                "l1_ratio": Real(0.1, 0.9),
            },
            conditional={"l1_ratio": {"solver": "liblinear"}},
            cv=2,
            n_jobs=1,
            max_iter=1,
            min_evals=1,
            max_evals=1,
            refit=False,
        )

        search.fit([[0.0], [1.0], [2.0], [3.0]], [0, 1, 0, 1])

        self.assertTrue(search.evaluation_history_)
        self.assertTrue(all("l1_ratio" not in item["params"] for item in search.evaluation_history_))

    def test_surrogate_cv_supports_parameter_free_estimators(self):
        search = SurrogateRandomCV(
            DummyClassifier(strategy="most_frequent"),
            {},
            cv=2,
            n_jobs=1,
            max_iter=1,
            min_evals=1,
            max_evals=1,
            refit=False,
        )

        search.fit([[0.0], [1.0], [2.0], [3.0]], [0, 1, 0, 1])

        self.assertEqual(search.evaluation_history_[0]["params"], {})
        self.assertEqual(search.evaluation_history_[0]["status"], "complete")

    def test_aml_routes_wrapped_estimator_parameters_through_stacking_estimator(self):
        with tempfile.TemporaryDirectory() as directory:
            automl = AML(
                config={
                    "sklearn.linear_model.LogisticRegression": {"C": Real(0.1, 1.0)},
                    "sklearn.linear_model.LinearRegression": {},
                },
                cv=2,
                n_jobs=1,
                verbose=0,
                check_point=directory + "/",
                max_evals=1,
                min_random_evals=1,
            )
            automl.add_surrogate(LinearRegression(), 1)

            model, _ = automl.optimize_pipeline(
                (
                    "sklearn.linear_model.LogisticRegression",
                    "sklearn.linear_model.LinearRegression",
                ),
                np.array([[0.0], [1.0], [2.0], [3.0]]),
                np.array([0.0, 1.0, 0.0, 1.0]),
            )

            self.assertIn("stp_0__estimator__C", model.get_params())

    def test_surrogate_cv_rejects_zero_feature_pipeline_before_fitting(self):
        pipeline = Pipeline(
            [
                ("selector", VarianceThreshold(threshold=1.0)),
                ("model", LogisticRegression(max_iter=100)),
            ]
        )
        search = SurrogateRandomCV(
            pipeline,
            {},
            cv=2,
            n_jobs=1,
            max_iter=1,
            min_evals=1,
            max_evals=1,
            refit=False,
        )

        search.fit(np.zeros((4, 2)), [0, 1, 0, 1])

        self.assertTrue(search.evaluation_history_)
        self.assertEqual(search.evaluation_history_[0]["status"], "invalid")
        self.assertEqual(
            search.evaluation_history_[0]["failure_records"][0]["type"],
            "ZeroFeaturePipeline",
        )

    def test_surrogate_cv_rejects_malformed_pipeline_before_evaluation(self):
        pipeline = Pipeline(
            [
                ("transformer", VarianceThreshold()),
                ("model", object()),
            ]
        )
        search = SurrogateRandomCV(
            pipeline,
            {},
            cv=2,
            n_jobs=1,
            max_iter=1,
            min_evals=1,
            max_evals=1,
            refit=False,
        )

        with self.assertRaisesRegex(ValueError, "final step 'model' must implement fit"):
            search.fit(np.ones((4, 2)), [0, 1, 0, 1])

        self.assertEqual(search.evaluation_history_, [])

    def test_surrogate_cv_records_structured_fit_failures(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 1, 0, 1])
        search = SurrogateRandomCV(
            LogisticRegression(max_iter=100),
            {"solver": Categorical(["invalid_solver"])},
            cv=2,
            n_jobs=1,
            max_iter=1,
            min_evals=1,
            error_score=np.nan,
            refit=False,
        )

        search.fit(X, y)

        trial = search.evaluation_history_[0]
        self.assertEqual(trial["status"], "failed")
        self.assertTrue(trial["failure_records"])
        failure = trial["failure_records"][0]
        self.assertEqual(failure["type"], "FitError")
        self.assertIn("invalid", failure["message"])
        self.assertEqual(failure["parameters"], {"solver": "invalid_solver"})
        self.assertEqual(search.cv_results_["failure_records"][0], trial["failure_records"])
        self.assertEqual(
            search.failure_summary_["by_estimator"]["LogisticRegression"]["failed_trials"],
            len(search.evaluation_history_),
        )

    def test_surrogate_cv_rejects_forbidden_combinations_before_fitting(self):
        # The single cell of this space is the invalid (lbfgs, L1) combination;
        # every candidate must be rejected before any fit attempt.
        search = SurrogateRandomCV(
            LogisticRegression(max_iter=100),
            {
                "solver": Categorical(["lbfgs"]),
                "l1_ratio": Categorical([1.0]),
            },
            forbidden=({"solver": "lbfgs", "l1_ratio": 1.0},),
            cv=2,
            max_iter=1,
            min_evals=1,
            refit=False,
        )

        search.fit([[0.0], [1.0], [2.0], [3.0]], [0, 1, 0, 1])

        self.assertTrue(search.evaluation_history_)
        self.assertTrue(all(item["status"] == "invalid" for item in search.evaluation_history_))
        self.assertEqual(search.failure_summary_["invalid_trials"], len(search.evaluation_history_))
        self.assertEqual(
            search.evaluation_history_[0]["failure_records"][0]["type"],
            "ForbiddenParameterCombination",
        )

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
