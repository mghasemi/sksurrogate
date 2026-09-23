"""Tests for the Phase 3 model-lifecycle endpoints (docs/ui-gap-implementation-plan.md).

Covers the four new API surfaces with one happy path and one error path each:

- ``POST /api/bundles/{task}/{version}/preserve`` / ``GET .../preserved`` / ``POST .../recover``
- ``GET  /api/bundles/{task}/best``
- ``POST /api/bundles/{task}/{version}/export-mlflow`` / ``GET .../download``
- ``GET  /api/registry/{task}/artifacts`` / ``DELETE /api/registry/{task}/artifacts``

The router functions are called directly (they are plain functions) against a
temporary storage root, with the small Galaxy3 dataset under ``data/`` registered
as ``train`` / ``validation`` partitions — matching the phase 1 test pattern.
"""

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from fastapi import HTTPException
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from SKSurrogate import ModelBundle, ModelRegistry, save_bundle

from api.config import settings
from api.deps import open_tracker
from api.routers import bundles as bundles_router
from api.routers import registry as registry_router

#: Small binary-classification dataset shipped with the repo (1600 rows, 21 columns).
GALAXY3_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "Galaxy3-[GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz].tabular"
)

TASK = "phase3-endpoints"
TARGET = "target"
GOOD_VERSION = "good-v1"
WEAK_VERSION = "weak-v1"


class TestPhase3Endpoints(unittest.TestCase):
    """Happy paths and error paths for the Phase 3 lifecycle/artifact endpoints."""

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

        self._train = self._frame.iloc[:1200].reset_index(drop=True)
        self._validation = self._frame.iloc[1200:].reset_index(drop=True)
        dataset_dir = settings.task_dataset_dir(TASK)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        self._train.to_csv(dataset_dir / "train.csv", index=False)
        self._validation.to_csv(dataset_dir / "validation.csv", index=False)
        with open_tracker(TASK) as tracker:
            tracker.RegisterData(self._train, TARGET, partition="train")
            tracker.RegisterData(self._validation, TARGET, partition="validation")
            self._fingerprint = tracker.GetMetadata()["dataset_fingerprint"]

        X = self._train.drop(columns=[TARGET])
        y = self._train[TARGET]

        # A strong and a deliberately weak model so "best" has a unique winner.
        self._save_bundle(
            RandomForestClassifier(n_estimators=25, random_state=0).fit(X, y),
            GOOD_VERSION,
            score=0.91,
        )
        self._save_bundle(LogisticRegression(max_iter=200).fit(X, y), WEAK_VERSION, score=0.40)

    def tearDown(self):
        self._tmp.cleanup()

    def _save_bundle(self, model, version, *, score=None, train_score=None):
        metrics = {}
        if score is not None:
            metrics["score"] = float(score)
        if train_score is not None:
            metrics["train_score"] = float(train_score)
        bundle = ModelBundle(
            model,
            task_name=TASK,
            model_version=version,
            metrics=metrics,
            dataset_fingerprint=self._fingerprint,
            schema={column: {"dtype": str(self._train[column].dtype)} for column in self._train.columns if column != TARGET},
            dependencies={},
        )
        save_bundle(bundle, settings.task_bundles_dir(TASK) / (version + ".bundle"))

    # ----------------------------------------------------------------- 3.2 best

    def test_best_picks_the_highest_scoring_bundle(self):
        payload = bundles_router.best_bundle(TASK, metric="score")

        self.assertEqual(payload["task_name"], TASK)
        self.assertEqual(payload["metric"], "score")
        self.assertEqual(payload["metrics_tried"], ["score"])
        self.assertEqual(payload["model_version"], GOOD_VERSION)
        self.assertAlmostEqual(payload["value"], 0.91, places=6)

    def test_best_prefers_the_first_metric_that_ranks_anything(self):
        estimator = self._bundle_model(WEAK_VERSION)
        self._save_bundle(estimator, "weak-v2", score=None, train_score=0.99)

        payload = bundles_router.best_bundle(TASK, metric="score,train_score")

        # ``score`` ranks something, so it wins outright — even though
        # ``train_score`` holds a higher (but not comparable) number.
        self.assertEqual(payload["metric"], "score")
        self.assertEqual(payload["model_version"], GOOD_VERSION)
        self.assertEqual(payload["metrics_tried"], ["score", "train_score"])

    def test_best_falls_through_to_the_next_metric(self):
        # Replace every ``score``-carrying bundle with an experiment-style one
        # (``train_score`` only) so the preference list has to walk on.
        estimator = self._bundle_model(GOOD_VERSION)
        for version in (GOOD_VERSION, WEAK_VERSION):
            (settings.task_bundles_dir(TASK) / (version + ".bundle")).unlink()
        self._save_bundle(estimator, "exp-v1", score=None, train_score=0.77)

        payload = bundles_router.best_bundle(TASK, metric="score,train_score")

        self.assertEqual(payload["metric"], "train_score")
        self.assertEqual(payload["model_version"], "exp-v1")
        self.assertAlmostEqual(payload["value"], 0.77, places=6)

    def test_best_raises_when_no_bundle_carries_the_metric(self):
        with self.assertRaises(HTTPException) as ctx:
            bundles_router.best_bundle(TASK, metric="train_score")

        self.assertEqual(ctx.exception.status_code, 404)

    # ------------------------------------------------------------ 3.1 preserve

    def test_preserve_and_list_and_recover_roundtrip(self):
        preserved = bundles_router.preserve_bundle(TASK, GOOD_VERSION)

        self.assertEqual(preserved["task_name"], TASK)
        self.assertEqual(preserved["model_version"], GOOD_VERSION)
        self.assertGreater(preserved["pickle_id"], 0)
        self.assertGreater(preserved["model_id"], 0)

        # A second snapshot of the *same* logged model: pickle ids advance while
        # the model id stays put, which is exactly the case a naive
        # ``RecoverModel(pickle_id)`` call gets wrong.
        second = bundles_router.preserve_bundle(TASK, GOOD_VERSION)
        self.assertEqual(second["model_id"], preserved["model_id"])
        self.assertNotEqual(second["pickle_id"], preserved["pickle_id"])

        listed = bundles_router.list_preserved(TASK)
        self.assertEqual(
            {snap["pickle_id"] for snap in listed["snapshots"]},
            {preserved["pickle_id"], second["pickle_id"]},
        )

        recovered = bundles_router.recover_model(TASK, bundles_router.RecoverRequest(pickle_id=second["pickle_id"]))

        self.assertEqual(recovered["pickle_id"], second["pickle_id"])
        self.assertEqual(recovered["model_id"], preserved["model_id"])
        self.assertEqual(recovered["model_type"], type(self._bundle_model(GOOD_VERSION)).__name__)
        # A fresh version is minted rather than overwriting the source bundle.
        self.assertNotEqual(recovered["model_version"], GOOD_VERSION)
        self.assertIsNotNone(recovered["smoke_score"])
        self.assertTrue(
            (settings.task_bundles_dir(TASK) / (recovered["model_version"] + ".bundle")).exists()
        )

    def test_list_preserved_is_empty_for_unregistered_task(self):
        payload = bundles_router.list_preserved("never-registered")

        self.assertEqual(payload["snapshots"], [])

    def test_preserve_raises_for_unknown_bundle(self):
        with self.assertRaises(HTTPException) as ctx:
            bundles_router.preserve_bundle(TASK, "does-not-exist")

        self.assertEqual(ctx.exception.status_code, 404)

    def test_recover_raises_for_unknown_pickle_id(self):
        with self.assertRaises(HTTPException) as ctx:
            bundles_router.recover_model(TASK, bundles_router.RecoverRequest(pickle_id=4242))

        self.assertEqual(ctx.exception.status_code, 404)

    # -------------------------------------------------------------- 3.3 export

    def test_export_mlflow_returns_a_zip_with_the_expected_layout(self):
        response = bundles_router.export_mlflow_bundle(TASK, GOOD_VERSION)

        payload = Path(response.path).read_bytes()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = set(archive.namelist())

        # The zip roots each entry at the export directory itself.
        self.assertTrue(any(name.endswith("/MLmodel") for name in names), names)
        self.assertTrue(any(name.endswith("/model.pkl") for name in names), names)
        self.assertTrue(any(name.endswith("/bundle.json") for name in names), names)
        self.assertEqual(response.media_type, "application/zip")

    def test_download_serves_the_raw_bundle_file(self):
        response = bundles_router.download_bundle(TASK, GOOD_VERSION)

        self.assertEqual(response.media_type, "application/octet-stream")
        self.assertTrue(Path(response.path).exists())
        self.assertEqual(Path(response.filename).suffix, ".bundle")

    def test_download_raises_for_unknown_version(self):
        with self.assertRaises(HTTPException) as ctx:
            bundles_router.download_bundle(TASK, "nope")

        self.assertEqual(ctx.exception.status_code, 404)

    # ----------------------------------------------------- 3.4 artifact cleanup

    def _register_artifacts(self):
        registry = ModelRegistry(settings.registry_dir)
        csv_path = settings.task_dataset_dir(TASK) / "train.csv"
        registry.register_dataset(
            TASK, self._fingerprint, csv_path, metadata={"partition": "train", "rows": len(self._train)}
        )
        prediction_path = settings.task_predictions_dir(TASK) / GOOD_VERSION / "req-1.csv"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text("prediction\n0\n1\n", encoding="utf-8")
        registry.register_prediction(
            TASK, GOOD_VERSION, self._fingerprint, prediction_path, prediction_id="req-1"
        )
        return registry

    def test_artifacts_lists_datasets_and_predictions_with_sizes(self):
        self._register_artifacts()

        payload = registry_router.registry_artifacts(TASK)

        self.assertEqual(payload["dataset_count"], 1)
        self.assertEqual(payload["prediction_count"], 1)
        self.assertEqual(payload["datasets"][0]["dataset_fingerprint"], self._fingerprint)
        self.assertEqual(payload["predictions"][0]["model_version"], GOOD_VERSION)
        self.assertGreater(payload["total_bytes"], 0)

    def test_delete_artifacts_scoped_by_model_version_and_dataset(self):
        self._register_artifacts()

        payload = registry_router.delete_registry_artifacts(
            TASK, model_version=GOOD_VERSION, dataset_fingerprint=self._fingerprint
        )

        self.assertEqual(payload["deleted"], {"datasets": 1, "predictions": 1})
        # The artifact files themselves are gone from disk.
        self.assertEqual(registry_router.registry_artifacts(TASK)["total_bytes"], 0)

    def test_delete_artifacts_requires_at_least_one_selector(self):
        self._register_artifacts()

        with self.assertRaises(HTTPException) as ctx:
            registry_router.delete_registry_artifacts(TASK)

        self.assertEqual(ctx.exception.status_code, 422)

    def test_artifacts_raises_for_unregistered_task(self):
        with self.assertRaises(HTTPException) as ctx:
            registry_router.registry_artifacts("never-registered")

        self.assertEqual(ctx.exception.status_code, 404)

    # ----------------------------------------------------------------- helpers

    def _bundle_model(self, version):
        from SKSurrogate import load_bundle

        return load_bundle(settings.task_bundles_dir(TASK) / (version + ".bundle"), strict_dependencies=False).model


class TestAutoWiring(unittest.TestCase):
    """The registry auto-wiring added in Phase 3.4 records what the other routers produce."""

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

    def tearDown(self):
        self._tmp.cleanup()

    def test_inference_batch_registers_a_prediction_artifact(self):
        """A batch run leaves a prediction artifact registered against the task."""
        from api.routers import inference as inference_router

        train = self._frame.iloc[:1200].reset_index(drop=True)
        with open_tracker(TASK) as tracker:
            tracker.RegisterData(train, TARGET, partition="train")
            fingerprint = tracker.GetMetadata()["dataset_fingerprint"]
        dataset_dir = settings.task_dataset_dir(TASK)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(dataset_dir / "train.csv", index=False)

        model = LogisticRegression(max_iter=200).fit(train.drop(columns=[TARGET]), train[TARGET])
        bundle = ModelBundle(
            model,
            task_name=TASK,
            model_version=GOOD_VERSION,
            metrics={"score": 0.5},
            dataset_fingerprint=fingerprint,
            schema={column: {"dtype": str(train[column].dtype)} for column in train.columns if column != TARGET},
            dependencies={},
        )
        save_bundle(bundle, settings.task_bundles_dir(TASK) / (GOOD_VERSION + ".bundle"))

        body = inference_router.PredictRequest(
            model_version=GOOD_VERSION, partition="train", request_id="auto-wire-1"
        )
        result = inference_router.predict_batch_endpoint(TASK, body)

        self.assertEqual(result["request_id"], "auto-wire-1")
        artifacts = registry_router.registry_artifacts(TASK)
        self.assertEqual(artifacts["prediction_count"], 1)
        self.assertEqual(artifacts["predictions"][0]["prediction_id"], "auto-wire-1")
        self.assertEqual(artifacts["predictions"][0]["model_version"], GOOD_VERSION)


if __name__ == "__main__":
    unittest.main()
