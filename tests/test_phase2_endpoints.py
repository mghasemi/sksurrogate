"""Tests for the Phase 2 UI-gap endpoints (docs/ui-gap-implementation-plan.md).

Covers the monitoring-completion surfaces with one happy path and one error path
each:

- ``POST /api/monitoring/{task}/{version}/prediction-drift``   (2.2)
- ``POST /api/monitoring/{task}/{version}/subgroup-performance`` (2.3)
- ``POST /api/monitoring/{task}/sensitive-scan``               (2.4)
- ``POST /api/datasets/{task}/register`` -> ``sensitive_scan`` (2.4, wiring a)

Router functions are called directly (they are plain functions) against a
temporary storage root, with the small Galaxy3 dataset under ``data/``
registered as ``train`` / ``validation`` partitions.
"""

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import HTTPException

from SKSurrogate import ModelBundle, save_bundle

from api.config import settings
from api.deps import open_tracker
from api.routers import datasets as datasets_router
from api.routers import monitoring as monitoring_router
from api.routers.monitoring import PredictionSource

#: Small binary-classification dataset shipped with the repo (1600 rows, 21 columns).
GALAXY3_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "Galaxy3-[GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz].tabular"
)

TASK = "phase2-endpoints"
MODEL_VERSION = "classifier-v1"
TARGET = "target"


class TestPhase2Endpoints(unittest.TestCase):
    """Happy paths and error paths for the Phase 2 monitoring endpoints."""

    @classmethod
    def setUpClass(cls):
        cls._original_root = settings.root
        cls._frame = pd.read_csv(GALAXY3_PATH, sep="\t")

    @classmethod
    def tearDownClass(cls):
        settings.root = cls._original_root

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        settings.root = Path(self._tmp.name)
        settings.ensure()

        train = self._frame.iloc[:1200].reset_index(drop=True)
        validation = self._frame.iloc[1200:].reset_index(drop=True)
        with open_tracker(TASK) as tracker:
            tracker.RegisterData(train, TARGET, partition="train")
            tracker.RegisterData(validation, TARGET, partition="validation")

        # ``register_dataset`` writes the CSV before calling ``RegisterData``; the
        # prediction-source resolver reads those CSVs, so mirror that here.
        dataset_dir = settings.task_dataset_dir(TASK)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(dataset_dir / "train.csv", index=False)
        validation.to_csv(dataset_dir / "validation.csv", index=False)

        bundle = ModelBundle(
            None,
            task_name=TASK,
            model_version=MODEL_VERSION,
            schema={},
            dependencies={},
        )
        save_bundle(bundle, settings.task_bundles_dir(TASK) / (MODEL_VERSION + ".bundle"))

        # Deterministic reference / current prediction series that actually drift.
        rng = np.random.default_rng(0)
        self._reference = rng.normal(0.0, 1.0, size=300).round(4).tolist()
        self._current = rng.normal(0.6, 1.0, size=300).round(4).tolist()

        # A partition with tell-tale PII / sensitive column names for the scan.
        self._sensitive_partition = "sensitive"
        sensitive_frame = pd.DataFrame(
            {
                "user_email": ["a@x.com", "b@x.com"] * 5,
                "customer_name": ["ann", "bob"] * 5,
                "api_key": ["k1", "k2"] * 5,
                "age": list(range(10)),
                TARGET: [0, 1] * 5,
            }
        )
        with open_tracker(TASK) as tracker:
            tracker.RegisterData(sensitive_frame, TARGET, partition=self._sensitive_partition)
        sensitive_frame.to_csv(dataset_dir / (self._sensitive_partition + ".csv"), index=False)

    def tearDown(self):
        self._tmp.cleanup()

    # ------------------------------------------------- 2.2 prediction distribution

    def test_prediction_drift_with_inline_arrays(self):
        body = monitoring_router.PredictionDriftRequest(
            reference_source=PredictionSource(values=self._reference),
            current_source=PredictionSource(values=self._current),
            threshold=0.2,
            bins=10,
        )

        report = monitoring_router.check_prediction_drift(TASK, MODEL_VERSION, body)

        self.assertEqual(report["prediction_column"], "prediction")
        self.assertIn("prediction", report["drift"]["numeric"])
        psi = report["drift"]["numeric"]["prediction"]["value"]
        self.assertIsNotNone(psi)
        # A ~0.6 sigma mean shift should be flagged against a 0.2 threshold.
        self.assertGreater(psi, 0.2)
        self.assertTrue(any(alert["type"] == "numeric_drift" for alert in report["alerts"]))
        self.assertEqual(report["thresholds"]["psi"], 0.2)

    def test_prediction_drift_with_partition_source_uses_target_column(self):
        body = monitoring_router.PredictionDriftRequest(
            reference_source=PredictionSource(partition="train"),
            current_source=PredictionSource(partition="validation"),
        )

        report = monitoring_router.check_prediction_drift(TASK, MODEL_VERSION, body)

        self.assertEqual(report["prediction_column"], "prediction")
        self.assertEqual(report["data_quality"]["reference_rows"], 1200)
        self.assertEqual(report["data_quality"]["current_rows"], 400)

    def test_prediction_drift_reads_persisted_inference_artifact(self):
        artifact_dir = settings.task_predictions_dir(TASK) / MODEL_VERSION
        artifact_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "prediction": self._current,
                "model_version": MODEL_VERSION,
                "request_id": "req-1",
            }
        ).to_csv(artifact_dir / "req-1.csv", index=False)

        body = monitoring_router.PredictionDriftRequest(
            reference_source=PredictionSource(values=self._reference),
            current_source=PredictionSource(request_id="req-1"),
        )

        report = monitoring_router.check_prediction_drift(TASK, MODEL_VERSION, body)

        self.assertEqual(report["data_quality"]["current_rows"], 300)

    def test_prediction_drift_rejects_two_sources_in_one_pick(self):
        body = monitoring_router.PredictionDriftRequest(
            reference_source=PredictionSource(values=self._reference, partition="train"),
            current_source=PredictionSource(values=self._current),
        )

        with self.assertRaises(HTTPException) as ctx:
            monitoring_router.check_prediction_drift(TASK, MODEL_VERSION, body)

        self.assertEqual(ctx.exception.status_code, 400)

    def test_prediction_drift_missing_artifact_is_not_found(self):
        body = monitoring_router.PredictionDriftRequest(
            reference_source=PredictionSource(values=self._reference),
            current_source=PredictionSource(request_id="never-written"),
        )

        with self.assertRaises(HTTPException) as ctx:
            monitoring_router.check_prediction_drift(TASK, MODEL_VERSION, body)

        self.assertEqual(ctx.exception.status_code, 404)

    # ------------------------------------------------------ 2.3 subgroup performance

    def test_subgroup_performance_with_inline_groups(self):
        y_true = [0, 1] * 50
        y_pred = ([0, 1] * 25) + ([1, 1] * 25)
        groups = (["a"] * 50) + (["b"] * 50)

        body = monitoring_router.SubgroupPerformanceRequest(
            y_true_source=PredictionSource(values=y_true),
            y_pred_source=PredictionSource(values=y_pred),
            inline_groups=groups,
            metric="accuracy",
        )

        report = monitoring_router.check_subgroup_performance(TASK, MODEL_VERSION, body)

        self.assertEqual(report["metric"], "accuracy")
        self.assertEqual(set(report["groups"]), {"a", "b"})
        self.assertEqual(report["groups"]["a"]["count"], 50)
        self.assertAlmostEqual(report["groups"]["a"]["accuracy"], 1.0)
        self.assertAlmostEqual(report["groups"]["b"]["accuracy"], 0.5)
        self.assertAlmostEqual(report["fairness_gap"], 0.5)

    def test_subgroup_performance_supports_loss_metric(self):
        body = monitoring_router.SubgroupPerformanceRequest(
            y_true_source=PredictionSource(values=[0, 1, 1, 0]),
            y_pred_source=PredictionSource(values=[0, 0, 1, 1]),
            inline_groups=["a", "a", "b", "b"],
            metric="loss",
        )

        report = monitoring_router.check_subgroup_performance(TASK, MODEL_VERSION, body)

        self.assertEqual(report["metric"], "loss")
        self.assertAlmostEqual(report["groups"]["a"]["loss"], 0.5)
        self.assertAlmostEqual(report["groups"]["b"]["loss"], 0.5)
        self.assertAlmostEqual(report["fairness_gap"], 0.0)

    def test_subgroup_performance_groups_from_partition_column(self):
        # Group by a stored column via the ``column`` override instead of inline values.
        body = monitoring_router.SubgroupPerformanceRequest(
            y_true_source=PredictionSource(partition=self._sensitive_partition),
            y_pred_source=PredictionSource(partition=self._sensitive_partition),
            groups_source=PredictionSource(partition=self._sensitive_partition, column="age"),
            metric="accuracy",
        )

        report = monitoring_router.check_subgroup_performance(TASK, MODEL_VERSION, body)

        self.assertEqual(report["metric"], "accuracy")
        # ``age`` runs 0..9, so every row is its own group.
        self.assertEqual(len(report["groups"]), 10)
        self.assertEqual(sum(g["count"] for g in report["groups"].values()), 10)

    def test_subgroup_performance_keeps_categorical_group_labels(self):
        # Groups must not be numerically coerced: non-numeric labels survive verbatim.
        body = monitoring_router.SubgroupPerformanceRequest(
            y_true_source=PredictionSource(partition=self._sensitive_partition),
            y_pred_source=PredictionSource(partition=self._sensitive_partition),
            groups_source=PredictionSource(partition=self._sensitive_partition, column="customer_name"),
        )

        report = monitoring_router.check_subgroup_performance(TASK, MODEL_VERSION, body)

        self.assertEqual(set(report["groups"]), {"ann", "bob"})

    def test_subgroup_performance_accepts_categorical_inline_group_values(self):
        # Inline group labels arrive through ``groups_source.values`` in the UI, so the
        # source model must accept strings (not only floats) or the request 422s.
        body = monitoring_router.SubgroupPerformanceRequest(
            y_true_source=PredictionSource(values=[0, 1, 0, 1]),
            y_pred_source=PredictionSource(values=[0, 1, 1, 1]),
            groups_source=PredictionSource(values=["female", "female", "male", "male"]),
        )

        report = monitoring_router.check_subgroup_performance(TASK, MODEL_VERSION, body)

        self.assertEqual(set(report["groups"]), {"female", "male"})
        self.assertAlmostEqual(report["groups"]["female"]["accuracy"], 1.0)
        self.assertAlmostEqual(report["groups"]["male"]["accuracy"], 0.5)

    def test_subgroup_performance_rejects_unknown_metric(self):
        body = monitoring_router.SubgroupPerformanceRequest(
            y_true_source=PredictionSource(values=[0, 1]),
            y_pred_source=PredictionSource(values=[0, 1]),
            inline_groups=["a", "a"],
            metric="auc",
        )

        with self.assertRaises(HTTPException) as ctx:
            monitoring_router.check_subgroup_performance(TASK, MODEL_VERSION, body)

        self.assertEqual(ctx.exception.status_code, 400)

    def test_subgroup_performance_requires_groups(self):
        body = monitoring_router.SubgroupPerformanceRequest(
            y_true_source=PredictionSource(values=[0, 1]),
            y_pred_source=PredictionSource(values=[0, 1]),
        )

        with self.assertRaises(HTTPException) as ctx:
            monitoring_router.check_subgroup_performance(TASK, MODEL_VERSION, body)

        self.assertEqual(ctx.exception.status_code, 400)

    def test_subgroup_performance_rejects_length_mismatch(self):
        body = monitoring_router.SubgroupPerformanceRequest(
            y_true_source=PredictionSource(values=[0, 1, 1]),
            y_pred_source=PredictionSource(values=[0, 1]),
            inline_groups=["a", "a", "a"],
        )

        with self.assertRaises(HTTPException) as ctx:
            monitoring_router.check_subgroup_performance(TASK, MODEL_VERSION, body)

        self.assertEqual(ctx.exception.status_code, 400)

    # ------------------------------------------------------------ 2.4 sensitive scan

    def test_sensitive_scan_flags_pii_and_sensitive_columns(self):
        body = monitoring_router.SensitiveScanRequest(partition=self._sensitive_partition)

        report = monitoring_router.scan_sensitive(TASK, body)

        self.assertIn("user_email", report["pii_columns"])
        self.assertIn("customer_name", report["pii_columns"])
        self.assertIn("api_key", report["sensitive_columns"])
        self.assertNotIn("age", report["pii_columns"])
        self.assertTrue(report["warnings"])

    def test_sensitive_scan_accepts_explicit_columns(self):
        body = monitoring_router.SensitiveScanRequest(
            partition=self._sensitive_partition,
            columns=["age", TARGET],
        )

        report = monitoring_router.scan_sensitive(TASK, body)

        self.assertEqual(report["pii_columns"], [])
        self.assertEqual(report["sensitive_columns"], [])
        self.assertEqual(report["warnings"], [])

    def test_sensitive_scan_adds_explicit_sensitive_features(self):
        body = monitoring_router.SensitiveScanRequest(
            partition=self._sensitive_partition,
            columns=["age"],
            sensitive_features=["age"],
        )

        report = monitoring_router.scan_sensitive(TASK, body)

        self.assertEqual(report["sensitive_columns"], ["age"])

    def test_sensitive_scan_requires_a_source(self):
        with self.assertRaises(HTTPException) as ctx:
            monitoring_router.scan_sensitive(TASK, monitoring_router.SensitiveScanRequest())

        self.assertEqual(ctx.exception.status_code, 400)

    def test_sensitive_scan_rejects_unknown_column(self):
        body = monitoring_router.SensitiveScanRequest(partition=self._sensitive_partition, columns=["nope"])

        with self.assertRaises(HTTPException) as ctx:
            monitoring_router.scan_sensitive(TASK, body)

        self.assertEqual(ctx.exception.status_code, 400)

    # -------------------------------------------------- 2.4 dataset-register wiring

    def test_register_dataset_returns_sensitive_scan(self):
        import asyncio
        from fastapi import UploadFile

        csv_bytes = (
            "user_email,customer_name,api_key,age,target\n"
            + "\n".join("a%d@x.com,name%d,k%d,%d,%d" % (i, i, i, i % 5, i % 2) for i in range(20))
            + "\n"
        ).encode()

        upload = UploadFile(file=io.BytesIO(csv_bytes), filename="upload.csv")

        response = asyncio.run(datasets_router.register_dataset(TASK, TARGET, "uploaded", upload))

        scan = response["sensitive_scan"]
        self.assertIn("user_email", scan["pii_columns"])
        self.assertIn("customer_name", scan["pii_columns"])
        self.assertIn("api_key", scan["sensitive_columns"])
        self.assertTrue(scan["warnings"])


if __name__ == "__main__":
    unittest.main()
