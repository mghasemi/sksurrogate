"""Tests for Phase 5 dataset schema review, validation, and synthesis endpoints."""

import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile

from api.config import settings
from api.deps import open_tracker
from api.routers import datasets as datasets_router
from api.routers import inference as inference_router
from api.routers import synthdata as synthdata_router
from api.routers.synthdata import GenerateSyntheticRequest


TASK = "phase5-endpoints"


def _upload(frame, filename="sample.csv"):
    stream = io.BytesIO(frame.to_csv(index=False).encode("utf-8"))
    return UploadFile(filename=filename, file=stream)


class TestPhase5Endpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_root = settings.root

    @classmethod
    def tearDownClass(cls):
        settings.root = cls._original_root

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        settings.root = Path(self._tmp.name)
        settings.ensure()
        self.frame = pd.DataFrame(
            {
                "measurement": [0.5, 1.5, 2.5, 3.5],
                "category": ["a", "b", "a", "b"],
                "target": ["no", "yes", "no", "yes"],
            }
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _register(self, *, overrides=None, binarize=False):
        return asyncio.run(
            datasets_router.register_dataset(
                TASK,
                target="target",
                partition="train",
                file=_upload(self.frame),
                type_overrides=json.dumps(overrides) if overrides is not None else None,
                binarize_label=binarize,
            )
        )

    def test_inspect_does_not_persist_and_returns_schema_scan(self):
        response = asyncio.run(datasets_router.inspect_dataset(TASK, _upload(self.frame)))

        self.assertEqual(response["rows"], len(self.frame))
        self.assertEqual(response["columns"], list(self.frame.columns))
        self.assertIn("target", response["deduced_types"])
        self.assertFalse(settings.mltrace_db_path(TASK).exists())
        self.assertFalse(settings.task_dataset_dir(TASK).exists())

    def test_register_applies_reviewed_types_and_binarizes_target(self):
        response = self._register(overrides={"measurement": "int64", "target": "label"}, binarize=True)

        self.assertEqual(response["deduced_types"]["measurement"], "int64")
        self.assertEqual(response["deduced_types"]["target"], "label")
        self.assertEqual(response["dataset_deduced_types"], response["deduced_types"])
        with open_tracker(TASK) as tracker:
            registered = tracker.get_dataframe("train")
            metadata = tracker.GetMetadata()
        self.assertEqual(sorted(registered["target"].unique().tolist()), [0.0, 1.0])
        self.assertEqual(metadata["dataset_deduced_types"]["measurement"], "int64")
        self.assertTrue((settings.task_dataset_dir(TASK) / "train.csv").exists())

    def test_register_rejects_unknown_override_column_without_persisting(self):
        with self.assertRaises(HTTPException) as ctx:
            self._register(overrides={"missing": "float64"})

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertFalse(settings.mltrace_db_path(TASK).exists())
        self.assertFalse(settings.task_dataset_dir(TASK).exists())

    def test_validate_upload_and_report_schema_difference(self):
        self._register()
        valid = asyncio.run(datasets_router.validate_dataset(TASK, file=_upload(self.frame)))
        self.assertTrue(valid["valid"])
        self.assertEqual(valid["source"], "upload")

        mismatched = self.frame.drop(columns=["category"])
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                datasets_router.validate_dataset(TASK, file=_upload(mismatched))
            )
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("missing columns", ctx.exception.detail)

    def test_validate_partition_and_preflight_helper(self):
        self._register()
        result = asyncio.run(
            datasets_router.validate_dataset(TASK, file=None, partition="train")
        )
        self.assertTrue(result["valid"])
        self.assertIsNone(inference_router._preflight_error(TASK, self.frame[["measurement", "category"]]))
        error = inference_router._preflight_error(TASK, self.frame[["measurement"]])
        self.assertIn("schema mismatch", error)

    def test_generate_list_and_download_synthetic_data(self):
        self._register()
        result = synthdata_router.generate_synthetic(
            TASK,
            GenerateSyntheticRequest(
                num=5,
                partition="train",
                distribution_type="marginal",
                type_overrides={"measurement": "real", "category": "cat", "target": "cat"},
            ),
        )
        self.assertEqual(result["rows"], 5)
        self.assertEqual(result["columns"], list(self.frame.columns))
        self.assertEqual(len(result["preview"]), 5)
        self.assertTrue(Path(result["path"]).exists())

        listing = synthdata_router.list_synthetic(TASK)
        self.assertEqual(len(listing["files"]), 1)
        self.assertEqual(listing["files"][0]["rows"], 5)
        response = synthdata_router.download_synthetic(TASK, Path(result["path"]).name)
        self.assertEqual(response.filename, Path(result["path"]).name)

    def test_synthetic_generation_rejects_bad_inputs(self):
        with self.assertRaises(HTTPException) as ctx:
            synthdata_router.generate_synthetic(TASK, GenerateSyntheticRequest(num=0))
        self.assertEqual(ctx.exception.status_code, 400)

        with self.assertRaises(HTTPException) as ctx:
            synthdata_router.generate_synthetic(TASK, GenerateSyntheticRequest(num=1))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_joint_synthetic_generation_supports_one_row(self):
        self._register()
        result = synthdata_router.generate_synthetic(
            TASK,
            GenerateSyntheticRequest(
                num=1,
                partition="train",
                distribution_type="joint",
                type_overrides={"measurement": "real", "category": "cat", "target": "cat"},
            ),
        )

        self.assertEqual(result["rows"], 1)
        self.assertEqual(len(result["preview"]), 1)
        self.assertEqual(result["columns"], list(self.frame.columns))

if __name__ == "__main__":
    unittest.main()
