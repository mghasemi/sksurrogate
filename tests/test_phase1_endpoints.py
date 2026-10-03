"""Tests for the Phase 1 UI-gap endpoints (docs/ui-gap-implementation-plan.md).

Covers the four new API surfaces with one happy path and one error path each:

- ``POST /api/evaluation/{task}/nested-cv``      (background job)
- ``GET  /api/evaluation/{task}/{version}/curves``
- ``GET  /api/datasets/{task}/target-stats``
- ``GET  /api/sensitivity/{task}/top-features``

The router functions are called directly (they are plain functions) against a
temporary storage root, with the small Galaxy3 dataset under ``data/`` registered
as ``train`` / ``validation`` partitions — matching the plan's testing notes.
Matplotlib is never imported: the endpoints compute arrays, not figures.
"""

import tempfile
import time
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from fastapi import HTTPException
from sklearn.linear_model import LinearRegression, LogisticRegression

from SKSurrogate import ModelBundle, save_bundle

from api.config import settings
from api.deps import open_tracker
from api.jobs import job_manager
from api.routers import datasets as datasets_router
from api.routers import evaluation as evaluation_router
from api.routers import sensitivity as sensitivity_router

#: Small binary-classification dataset shipped with the repo (1600 rows, 21 columns).
GALAXY3_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "Galaxy3-[GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz].tabular"
)

TASK = "phase1-endpoints"
CLASSIFIER_VERSION = "classifier-v1"
REGRESSOR_VERSION = "regressor-v1"
TARGET = "target"

#: How long to wait for a background job before failing the test.
JOB_TIMEOUT_SECONDS = 120.0


def _wait_for_job(job_id, timeout=JOB_TIMEOUT_SECONDS):
    """Poll the persisted job record until it reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = job_manager.get(job_id)
        if record is not None and record["status"] in {"completed", "failed"}:
            return record
        time.sleep(0.25)
    raise AssertionError("job %s did not finish within %.0fs" % (job_id, timeout))


class TestPhase1Endpoints(unittest.TestCase):
    """Happy paths and error paths for the Phase 1 evaluation/dataset/sensitivity endpoints."""

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

        # Register two partitions so the split_train-based curves have data and the
        # validation partition lets nested CV report a final_validation_score.
        train = self._frame.iloc[:1200].reset_index(drop=True)
        validation = self._frame.iloc[1200:].reset_index(drop=True)
        with open_tracker(TASK) as tracker:
            tracker.RegisterData(train, TARGET, partition="train")
            tracker.RegisterData(validation, TARGET, partition="validation")

        self._save_bundle(
            LogisticRegression(max_iter=300).fit(
                train.drop(columns=[TARGET]).values, train[TARGET].values
            ),
            CLASSIFIER_VERSION,
        )
        self._save_bundle(
            LinearRegression().fit(
                train.drop(columns=[TARGET]).values, train[TARGET].values
            ),
            REGRESSOR_VERSION,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _save_bundle(self, model, version):
        bundle = ModelBundle(
            model,
            task_name=TASK,
            model_version=version,
            schema={},
            dependencies={},
        )
        save_bundle(bundle, settings.task_bundles_dir(TASK) / (version + ".bundle"))

    def _write_weights(self, frame):
        with open_tracker(TASK) as tracker:
            frame.to_sql("weights", tracker.conn, if_exists="replace", index=False)

    # ---------------------------------------------------------------- 1.3 datasets

    def test_target_stats_summarizes_registered_target(self):
        expected = self._frame.iloc[:1200][TARGET].describe()

        stats = datasets_router.target_stats(TASK)

        self.assertEqual(stats["task_name"], TASK)
        self.assertEqual(stats["target"], TARGET)
        # ``train`` is preferred over the other registered partitions.
        self.assertEqual(stats["partition"], "train")
        self.assertEqual(stats["count"], 1200)
        for key in ("mean", "std", "min", "25%", "50%", "75%", "max"):
            self.assertAlmostEqual(stats[key], float(expected[key]), places=6, msg=key)

    def test_target_stats_falls_back_when_train_is_missing(self):
        # Leave only a ``test`` partition: ``train`` and ``validation`` are absent, so
        # the endpoint must fall back to the one partition that exists.
        with open_tracker(TASK) as tracker:
            tracker.RegisterData(self._frame, TARGET, partition="test")
        with open_tracker(TASK) as tracker:
            tracker.conn.execute("DROP TABLE data_train")
            tracker.conn.execute("DROP TABLE data_validation")
            tracker.conn.commit()

        stats = datasets_router.target_stats(TASK)

        self.assertEqual(stats["partition"], "test")
        self.assertEqual(stats["count"], len(self._frame))

    def test_target_stats_raises_for_unregistered_task(self):
        with self.assertRaises(HTTPException) as ctx:
            datasets_router.target_stats("never-registered")

        self.assertEqual(ctx.exception.status_code, 404)

    # ------------------------------------------------------------ 1.4 sensitivity

    def test_top_features_reports_unavailable_without_stored_weightings(self):
        payload = sensitivity_router.get_top_features(TASK, num=5)

        self.assertFalse(payload["available"])
        self.assertEqual(payload["features"], [])
        self.assertIn("sensitivity", payload["hint"].lower())

    def test_top_features_ranks_consensus_across_weightings(self):
        features = [column for column in self._frame.columns if column != TARGET]
        # Both weighting columns lead with features[0], so it is the only feature in
        # both top-2s and must rank first; the runners-up differ per weighting.
        self._write_weights(
            pd.DataFrame(
                {
                    "feature": features,
                    "pearson": [10.0, 9.0] + [0.0] * (len(features) - 2),
                    "variance": [10.0, 0.0, 9.0] + [0.0] * (len(features) - 3),
                }
            )
        )

        payload = sensitivity_router.get_top_features(TASK, num=2)

        self.assertTrue(payload["available"])
        self.assertEqual(payload["weightings"], 2)
        self.assertEqual(payload["num"], 2)
        self.assertEqual(payload["features"][0], [features[0], 2])

        counts = dict(payload["features"])
        # The pearson runner-up and the variance runner-up each appear once.
        self.assertEqual(counts[features[1]], 1)
        self.assertEqual(counts[features[2]], 1)
        # ``pearson`` uniquely also counts the bottom of its ranking (mltrack's
        # documented both-ends behaviour), so a tail feature is picked up too.
        self.assertIn(features[-1], counts)
        # The consensus list is sorted by descending appearance count.
        self.assertEqual([count for _, count in payload["features"]], sorted(
            [count for _, count in payload["features"]], reverse=True
        ))

    def test_top_features_raises_for_unregistered_task(self):
        with self.assertRaises(HTTPException) as ctx:
            sensitivity_router.get_top_features("never-registered", num=5)

        self.assertEqual(ctx.exception.status_code, 404)

    # ------------------------------------------------------------ 1.2 curves

    def test_bundle_curves_returns_roc_calibration_gain_and_lift(self):
        payload = evaluation_router.bundle_curves(TASK, CLASSIFIER_VERSION, bins=5)

        self.assertEqual(payload["model_version"], CLASSIFIER_VERSION)
        self.assertEqual(payload["bins"], 5)
        self.assertEqual(len(payload["classes"]), 2)
        self.assertGreater(payload["n_test_samples"], 0)
        self.assertEqual(payload["errors"], [])

        roc = payload["roc"]
        self.assertGreaterEqual(roc["auc"], 0.0)
        self.assertLessEqual(roc["auc"], 1.0)
        self.assertEqual(len(roc["fpr"]), len(roc["tpr"]))
        self.assertEqual(roc["fpr"][0], 0.0)
        self.assertEqual(roc["fpr"][-1], 1.0)
        self.assertEqual(roc["tpr"][-1], 1.0)

        calibration = payload["calibration"]
        self.assertEqual(len(calibration["mean_predicted_value"]), len(calibration["fraction_of_positives"]))
        # Histogram is fixed at ``bins`` buckets over (0, 1) and counts every test row.
        self.assertEqual(len(calibration["histogram_counts"]), 5)
        self.assertEqual(sum(calibration["histogram_counts"]), payload["n_test_samples"])

        gain = payload["cumulative_gain"]
        self.assertEqual(gain["percentages"][0], 0.0)
        self.assertAlmostEqual(gain["percentages"][-1], 1.0)
        self.assertAlmostEqual(gain["gains_class0"][-1], 1.0)
        self.assertAlmostEqual(gain["gains_class1"][-1], 1.0)
        self.assertEqual(len(gain["percentages"]), len(gain["gains_class0"]))
        self.assertEqual(len(gain["percentages"]), len(gain["gains_class1"]))

        # Lift drops the leading (0, 0) point then divides gain by percentage.
        lift = payload["lift"]
        self.assertEqual(len(lift["percentages"]), len(gain["percentages"]) - 1)
        self.assertEqual(len(lift["percentages"]), len(lift["lifts_class0"]))
        self.assertEqual(len(lift["percentages"]), len(lift["lifts_class1"]))
        self.assertAlmostEqual(lift["percentages"][-1], 1.0)

    def test_bundle_curves_rejects_regression_bundle(self):
        with self.assertRaises(HTTPException) as ctx:
            evaluation_router.bundle_curves(TASK, REGRESSOR_VERSION)

        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("classification", ctx.exception.detail)

    def test_bundle_curves_returns_404_for_unknown_bundle(self):
        with self.assertRaises(HTTPException) as ctx:
            evaluation_router.bundle_curves(TASK, "does-not-exist")

        self.assertEqual(ctx.exception.status_code, 404)

    # ------------------------------------------------------------ 1.1 nested CV

    def test_nested_cv_runs_as_background_job(self):
        submitted = evaluation_router.run_nested_cv(
            TASK,
            evaluation_router.NestedCVRequest(
                model_version=CLASSIFIER_VERSION,
                inner_cv=2,
                outer_cv=2,
                repeats=1,
                confidence_level=0.95,
            ),
        )

        self.assertEqual(submitted["status"], "queued")
        record = _wait_for_job(submitted["job_id"])
        self.assertEqual(record["status"], "completed", record["error"])
        self.assertEqual(record["kind"], "nested-cv")

        result = record["result"]
        self.assertEqual(result["task_name"], TASK)
        self.assertEqual(result["model_version"], CLASSIFIER_VERSION)
        self.assertEqual(result["repeat_count"], 1)
        self.assertEqual(len(result["outer_scores"]), 2)
        self.assertEqual(len(result["fold_metrics"]), 2)
        self.assertEqual(len(result["repeated_outer_scores"]), 1)
        self.assertIsNotNone(result["mean_outer_score"])

        interval = result["confidence_interval"]
        self.assertEqual(interval["confidence_level"], 0.95)
        self.assertLessEqual(interval["lower"], result["mean_outer_score"])
        self.assertGreaterEqual(interval["upper"], result["mean_outer_score"])

        # A validation partition was registered, so the pooled score must be present.
        self.assertIsNotNone(result["final_validation_score"])

        for fold in result["fold_metrics"]:
            self.assertEqual(len(fold["outer_predictions"]), len(fold["outer_truth"]))

    def test_nested_cv_requires_exactly_one_bundle_selector(self):
        with self.assertRaises(HTTPException) as ctx:
            evaluation_router.run_nested_cv(TASK, evaluation_router.NestedCVRequest())

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("model_version", ctx.exception.detail)

        with self.assertRaises(HTTPException) as ctx_both:
            evaluation_router.run_nested_cv(
                TASK,
                evaluation_router.NestedCVRequest(
                    model_version=CLASSIFIER_VERSION, alias="production"
                ),
            )

        self.assertEqual(ctx_both.exception.status_code, 400)

    def test_nested_cv_returns_404_for_unknown_bundle(self):
        with self.assertRaises(HTTPException) as ctx:
            evaluation_router.run_nested_cv(
                TASK, evaluation_router.NestedCVRequest(model_version="nope")
            )

        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
