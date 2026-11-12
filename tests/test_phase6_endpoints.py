"""Focused tests for job resume metadata and execution backend selection."""

import tempfile
import time
import unittest
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
import numpy as np
import pandas as pd
from pydantic import ValidationError
from sklearn.dummy import DummyClassifier

from SKSurrogate import ModelBundle, ModelRegistry, mltrack
from api.config import settings
from api.jobs import JobManager, job_manager
from api.routers import experiments as experiments_router
from api.routers import jobs as jobs_router
from api.routers import retraining as retraining_router
from api.routers.experiments import RunExperimentRequest
from api.routers.retraining import (
    BaselineTrainerConfig,
    ExperimentTrainerConfig,
    RunRetrainingRequest,
    TriggerRetrainingRequest,
)


def _wait_for_job(job_id, timeout=10, manager=job_manager):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = manager.get(job_id)
        if record and record["status"] in {"completed", "failed"}:
            return record
        time.sleep(0.02)
    raise AssertionError("job %s did not finish" % job_id)


class TestPhase6JobEndpoints(unittest.TestCase):
    def setUp(self):
        self._original_root = settings.root
        self._tmp = tempfile.TemporaryDirectory()
        settings.root = Path(self._tmp.name)
        settings.ensure()

    def tearDown(self):
        settings.root = self._original_root
        self._tmp.cleanup()

    def test_experiment_generation_field_accepts_legacy_and_canonical_names(self):
        experiment_schema = RunExperimentRequest.model_json_schema(mode="validation")
        trainer_schema = ExperimentTrainerConfig.model_json_schema(mode="validation")
        self.assertIn("max_generations", experiment_schema["properties"])
        self.assertNotIn("max_generation", experiment_schema["properties"])
        self.assertIn("max_generations", trainer_schema["properties"])
        self.assertNotIn("max_generation", trainer_schema["properties"])

        experiment = RunExperimentRequest(config={}, max_generations=5)
        legacy_experiment = RunExperimentRequest(config={}, max_generation=6)
        trainer = ExperimentTrainerConfig(config={}, max_generations=7)
        legacy_trainer = ExperimentTrainerConfig(config={}, max_generation=8)
        retraining = RunRetrainingRequest(
            trainer={
                "kind": "experiment",
                "config": {},
                "max_generation": 9,
            }
        )
        self.assertEqual(experiment.max_generations, 5)
        self.assertEqual(legacy_experiment.max_generations, 6)
        self.assertEqual(trainer.max_generations, 7)
        self.assertEqual(legacy_trainer.max_generations, 8)
        self.assertEqual(retraining.trainer.max_generations, 9)

        with self.assertRaises(ValidationError):
            RunExperimentRequest(
                config={},
                max_generations=5,
                max_generation=6,
            )
        with self.assertRaises(ValidationError):
            ExperimentTrainerConfig(
                config={},
                max_generations=5,
                max_generation=6,
            )

    def test_failed_experiment_resumes_with_original_checkpoint_and_run_id(self):
        body = RunExperimentRequest(config={}, run_id=None)
        calls = []
        fitted = SimpleNamespace(model_version="resumed-model", metrics={"train_score": 0.75})

        def fit(*args, **kwargs):
            _ = args
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("interrupted")
            return fitted, [], [], []

        with patch.object(experiments_router, "fit_experiment_bundle", side_effect=fit), patch.object(
            experiments_router, "save_bundle", return_value=Path("resumed.bundle")
        ):
            submitted = experiments_router.run_experiment("resume-task", body)
            failed = _wait_for_job(submitted["job_id"])
            self.assertEqual(failed["status"], "failed")

            original_checkpoint = settings.task_checkpoint_dir("resume-task", submitted["job_id"])
            original_checkpoint.mkdir(parents=True)
            (original_checkpoint / "search.eoa").write_bytes(b"checkpoint")

            resumed = jobs_router.resume_job(submitted["job_id"])
            completed = _wait_for_job(resumed["job_id"])

        self.assertEqual(completed["status"], "completed", completed["error"])
        self.assertEqual(calls[1]["checkpoint_dir"], original_checkpoint)
        self.assertEqual(calls[1]["run_id"], submitted["job_id"])
        self.assertEqual(completed["resume"]["checkpoint_job_id"], submitted["job_id"])
        self.assertEqual(completed["resume"]["run_id"], submitted["job_id"])

    def test_resume_rejects_non_failed_or_non_experiment_jobs(self):
        for status, kind in (("running", "experiment"), ("failed", "retraining")):
            job_id = "invalid-%s-%s" % (status, kind)
            job_manager._write(
                {
                    "job_id": job_id,
                    "task_name": "resume-task",
                    "kind": kind,
                    "status": status,
                }
            )
            with self.assertRaises(HTTPException) as ctx:
                jobs_router.resume_job(job_id)
            self.assertEqual(ctx.exception.status_code, 409)

    def test_resume_requires_a_checkpoint_file(self):
        job_id = "without-checkpoint"
        checkpoint_id = "old-checkpoint"
        job_manager._write(
            {
                "job_id": job_id,
                "task_name": "resume-task",
                "kind": "experiment",
                "status": "failed",
                "resume": {
                    "request": {"config": {}, "backend": "local"},
                    "checkpoint_job_id": checkpoint_id,
                    "run_id": "run",
                },
            }
        )
        settings.task_checkpoint_dir("resume-task", checkpoint_id).mkdir(parents=True)
        with self.assertRaises(HTTPException) as ctx:
            jobs_router.resume_job(job_id)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("checkpoint", ctx.exception.detail)

    def test_resume_prevents_concurrent_use_of_the_same_checkpoint(self):
        checkpoint_id = "original-checkpoint"
        checkpoint = settings.task_checkpoint_dir("resume-task", checkpoint_id)
        checkpoint.mkdir(parents=True)
        (checkpoint / "search.eoa").write_bytes(b"checkpoint")
        resume = {
            "request": {"config": {}, "backend": "local"},
            "checkpoint_job_id": checkpoint_id,
            "run_id": "run",
        }
        job_manager._write(
            {
                "job_id": "failed-source",
                "task_name": "resume-task",
                "kind": "experiment",
                "status": "failed",
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2025-01-01T00:00:00+00:00",
                "resume": resume,
            }
        )
        # A *completed* resume proves the checkpoint already produced a result.
        # A stale ``running`` record is deliberately not enough any more: liveness
        # comes from the job manager, not from the persisted status.
        job_manager._write(
            {
                "job_id": "active-resume",
                "task_name": "resume-task",
                "kind": "experiment",
                "status": "completed",
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2025-01-01T00:00:00+00:00",
                "resume": resume,
            }
        )
        with self.assertRaises(HTTPException) as ctx:
            jobs_router.resume_job("failed-source")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("already", ctx.exception.detail)

    def test_dask_is_rejected_cleanly_when_optional_dependency_is_missing(self):
        body = RunExperimentRequest(config={}, backend="dask")
        with patch.object(experiments_router, "dask_available", return_value=False):
            with self.assertRaises(HTTPException) as ctx:
                experiments_router.run_experiment("resume-task", body)
        self.assertEqual(ctx.exception.status_code, 503)

        retraining = RunRetrainingRequest(
            trainer=BaselineTrainerConfig(kind="baseline"),
            backend="dask",
        )
        with patch.object(retraining_router, "dask_available", return_value=False):
            with self.assertRaises(HTTPException) as ctx:
                retraining_router.run_retraining("resume-task", retraining)
        self.assertEqual(ctx.exception.status_code, 503)

    def test_external_retraining_trigger_is_idempotent_and_quality_gated(self):
        model = DummyClassifier(strategy="most_frequent").fit(
            np.array([[0.0], [1.0]]), np.array([0, 1])
        )
        body = TriggerRetrainingRequest(
            trainer=BaselineTrainerConfig(kind="baseline"),
            trigger="scheduled",
            event_id="nightly-2026-11-05",
            occurred_at=datetime.now(timezone.utc),
            metric_thresholds={"accuracy": 0.8},
        )

        with patch.object(
            retraining_router,
            "_make_trainer",
            return_value=lambda context: ModelBundle(
                model, metrics={"accuracy": 0.9}
            ),
        ):
            submitted = retraining_router.trigger_retraining("trigger-task", body)
            duplicate = retraining_router.trigger_retraining("trigger-task", body)
            completed = _wait_for_job(submitted["job_id"])

        self.assertTrue(submitted["duplicate"] is False)
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(duplicate["job_id"], submitted["job_id"])
        self.assertEqual(completed["status"], "completed", completed["error"])
        self.assertTrue(completed["result"]["quality_report"]["passed"])
        self.assertEqual(
            completed["trigger_event"]["event_id"], "nightly-2026-11-05"
        )
        audit = ModelRegistry(settings.registry_dir).audit_log("trigger-task")
        self.assertTrue(any(event.get("to") == "staging" for event in audit))

        conflicting = body.model_copy(update={"metric_thresholds": {"accuracy": 0.95}})
        with self.assertRaises(HTTPException) as ctx:
            retraining_router.trigger_retraining("trigger-task", conflicting)
        self.assertEqual(ctx.exception.status_code, 409)

    def test_job_manager_deduplicates_concurrent_trigger_submissions(self):
        manager = JobManager()

        def submit(_):
            return manager.submit_idempotent(
                "concurrent-trigger-task",
                "retraining",
                "data:train:fingerprint",
                "request-fingerprint",
                lambda job_id: lambda: {"job_id": job_id},
                metadata_factory=lambda job_id: {"trigger_event": {"event_id": "upload-1"}},
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(submit, range(16)))

        job_ids = {job_id for job_id, _ in results}
        created_count = sum(created for _, created in results)
        self.assertEqual(len(job_ids), 1)
        self.assertEqual(created_count, 1)
        completed = _wait_for_job(next(iter(job_ids)), manager=manager)
        self.assertEqual(completed["status"], "completed", completed["error"])

    def test_external_scheduled_trigger_records_not_due_without_training(self):
        body = TriggerRetrainingRequest(
            trainer=BaselineTrainerConfig(kind="baseline"),
            trigger="scheduled",
            event_id="not-due-2026-11-05",
            occurred_at=datetime.now(timezone.utc),
            due=False,
            metric_thresholds={"accuracy": 0.8},
        )
        trainer_calls = []

        def make_trainer(*args):
            _ = args
            return lambda context: trainer_calls.append(context)

        with patch.object(retraining_router, "_make_trainer", side_effect=make_trainer):
            submitted = retraining_router.trigger_retraining("not-due-task", body)
            completed = _wait_for_job(submitted["job_id"])

        self.assertEqual(completed["status"], "completed", completed["error"])
        self.assertEqual(completed["result"]["status"], "skipped")
        self.assertEqual(completed["result"]["reason"], "trigger_not_satisfied")
        self.assertEqual(trainer_calls, [])

    def test_external_data_trigger_requires_fresh_registered_partition_identity(self):
        frame = pd.DataFrame(
            {"feature": np.arange(8, dtype=float), "target": np.arange(8, dtype=float)}
        )
        tracker = mltrack(
            "data-trigger-task",
            db_name=str(settings.mltrace_db_path("data-trigger-task")),
        )
        try:
            tracker.RegisterData(frame, "target", partition="train")
            fingerprint = tracker.dataset_splits()["train"]["fingerprint"]
        finally:
            tracker.close()

        body = TriggerRetrainingRequest(
            trainer=BaselineTrainerConfig(kind="baseline"),
            trigger="on_data",
            event_id="dataset-upload-8",
            occurred_at=datetime.now(timezone.utc),
            data_partition="train",
            data_fingerprint=fingerprint,
            metric_thresholds={"r2": 0.0},
        )
        with patch.object(
            retraining_router,
            "_make_trainer",
            return_value=lambda context: {"unexpected": True},
        ):
            wrong_fingerprint = body.model_copy(
                update={"data_fingerprint": "stale-fingerprint"}
            )
            with self.assertRaises(HTTPException) as mismatch:
                retraining_router.trigger_retraining(
                    "data-trigger-task", wrong_fingerprint
                )
            self.assertEqual(mismatch.exception.status_code, 409)

            old_event = body.model_copy(
                update={"occurred_at": datetime.now(timezone.utc) - timedelta(minutes=11)}
            )
            with self.assertRaises(HTTPException) as stale:
                retraining_router.trigger_retraining("data-trigger-task", old_event)
            self.assertEqual(stale.exception.status_code, 409)

        with self.assertRaises(ValidationError):
            TriggerRetrainingRequest(
                trainer=BaselineTrainerConfig(kind="baseline"),
                trigger="on_data",
                event_id="unsafe-production",
                occurred_at=datetime.now(timezone.utc),
                data_partition="train",
                data_fingerprint=fingerprint,
                promotion_state="production",
                metric_thresholds={"r2": 0.0},
            )

        model = DummyClassifier(strategy="most_frequent").fit(
            np.array([[0.0], [1.0]]), np.array([0, 1])
        )
        with patch.object(
            retraining_router,
            "_make_trainer",
            return_value=lambda context: ModelBundle(
                model, metrics={"r2": 0.5}
            ),
        ):
            submitted = retraining_router.trigger_retraining("data-trigger-task", body)
            retry = body.model_copy(update={"event_id": "upload-retry"})
            duplicate = retraining_router.trigger_retraining(
                "data-trigger-task", retry
            )
            completed = _wait_for_job(submitted["job_id"])

        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(duplicate["job_id"], submitted["job_id"])
        self.assertEqual(completed["status"], "completed", completed["error"])
        self.assertEqual(
            completed["result"]["context"]["data_fingerprint"], fingerprint
        )

    def test_scoring_options_report_available_backends(self):
        with patch.object(experiments_router, "dask_available", return_value=False):
            self.assertEqual(experiments_router.scoring_options()["backends"], ["local"])
        with patch.object(experiments_router, "dask_available", return_value=True):
            self.assertEqual(experiments_router.scoring_options()["backends"], ["local", "dask"])

    def test_dask_job_manager_dispatches_and_records_result(self):
        class FakeCluster:
            """Stand-in for the LocalCluster that api.jobs builds from the budget."""

            def __init__(self):
                self.closed = False

            def get_client(self):
                return SimpleNamespace(close=lambda: None)

            def close(self):
                self.closed = True

        class FakeExecutionBackend:
            def __init__(self, root, task_name, client=None):
                _ = (root, task_name)
                self.client = client
                self._function: Callable[[object], object] | None = None
                self._params: object = None
                self._status = "queued"
                self._result: object = None

            def submit_trial(self, params, fn):
                self._params = params
                self._function = fn
                return "trial"

            def execute_pending(self, limit):
                _ = limit
                assert self._function is not None
                self._result = self._function(self._params)
                self._status = "completed"

            def get_trial(self, trial_id):
                _ = trial_id
                return {"status": self._status, "result": self._result}

        manager = JobManager()
        cluster = FakeCluster()
        with patch("SKSurrogate.execution.DaskExecutionBackend", FakeExecutionBackend), patch(
            "api.jobs.compute.dask_cluster", return_value=cluster
        ):
            job_id = manager.submit(
                "dask-task",
                "experiment",
                lambda job_id: self._result_factory(job_id),
                backend="dask",
            )
            record = _wait_for_job(job_id, manager=manager)
        self.assertEqual(record["status"], "completed", record["error"])
        self.assertEqual(record["backend"], "dask")
        self.assertEqual(record["result"], {"value": 42})
        # The job owns its cluster and must release it when it finishes.
        self.assertTrue(cluster.closed)

    @staticmethod
    def _result_factory(job_id):
        _ = job_id
        return lambda: {"value": 42}


if __name__ == "__main__":
    unittest.main()
