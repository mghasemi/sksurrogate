"""Focused tests for job resume metadata and execution backend selection."""

import tempfile
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from api.config import settings
from api.jobs import JobManager, job_manager
from api.routers import experiments as experiments_router
from api.routers import jobs as jobs_router
from api.routers import retraining as retraining_router
from api.routers.experiments import RunExperimentRequest
from api.routers.retraining import BaselineTrainerConfig, RunRetrainingRequest


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
        job_manager._write(
            {
                "job_id": "active-resume",
                "task_name": "resume-task",
                "kind": "experiment",
                "status": "running",
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

    def test_scoring_options_report_available_backends(self):
        with patch.object(experiments_router, "dask_available", return_value=False):
            self.assertEqual(experiments_router.scoring_options()["backends"], ["local"])
        with patch.object(experiments_router, "dask_available", return_value=True):
            self.assertEqual(experiments_router.scoring_options()["backends"], ["local", "dask"])

    def test_dask_job_manager_dispatches_and_records_result(self):
        class FakeExecutionBackend:
            def __init__(self, root, task_name):
                _ = (root, task_name)
                self.client = SimpleNamespace(close=lambda: None)
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
        with patch("SKSurrogate.execution.DaskExecutionBackend", FakeExecutionBackend):
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

    @staticmethod
    def _result_factory(job_id):
        _ = job_id
        return lambda: {"value": 42}


if __name__ == "__main__":
    unittest.main()
