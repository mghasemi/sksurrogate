"""A minimal persisted background-job manager.

Long-running work (AML/EOA searches, sensitivity analysis) is submitted here
instead of running inline in the request/response cycle (see docs/ui-plan.md
section 5). Jobs run on a thread pool in this process and their status is
persisted as JSON so ``GET /api/jobs/{id}`` keeps working across restarts;
this intentionally stays file-based rather than pulling in Celery/Redis.
"""

import importlib
import json
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .config import settings

_executor = ThreadPoolExecutor(max_workers=4)
_write_lock = threading.Lock()


def _now():
    return datetime.now(timezone.utc).isoformat()


def dask_available():
    """Return whether the optional Dask distributed client can be imported."""
    try:
        importlib.import_module("dask.distributed")
    except (ImportError, ModuleNotFoundError):
        return False
    return True


class JobManager:
    """Submits callables to a thread pool and persists their lifecycle as JSON."""

    def _write(self, record):
        path = settings.job_record_path(record["job_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".json.tmp")
        with _write_lock:
            temp_path.write_text(json.dumps(record, default=str), encoding="utf-8")
            temp_path.replace(path)

    def get(self, job_id):
        path = settings.job_record_path(job_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, task_name=None):
        records = [
            json.loads(path.read_text(encoding="utf-8")) for path in settings.jobs_dir.glob("*.json")
        ]
        if task_name is not None:
            records = [record for record in records if record["task_name"] == task_name]
        return sorted(records, key=lambda record: record["created_at"], reverse=True)

    def submit(self, task_name, kind, fn_factory, *, backend="local", metadata_factory=None):
        """Run ``fn_factory(job_id)()`` in the background; result must be a JSON-safe dict."""
        job_id = uuid.uuid4().hex
        record = {
            "job_id": job_id,
            "task_name": task_name,
            "kind": kind,
            "backend": backend,
            "status": "queued",
            "created_at": _now(),
            "updated_at": _now(),
            "result": None,
            "error": None,
        }
        if metadata_factory is not None:
            record.update(metadata_factory(job_id))
        self._write(record)

        def _run():
            record["status"] = "running"
            record["updated_at"] = _now()
            self._write(record)
            try:
                fn = fn_factory(job_id)
                if backend == "dask":
                    from SKSurrogate.execution import DaskExecutionBackend

                    execution = DaskExecutionBackend(
                        settings.jobs_dir / "execution",
                        task_name=job_id,
                    )
                    try:
                        def invoke(_params):
                            _ = _params
                            return fn()

                        trial_id = execution.submit_trial(None, invoke)
                        execution.execute_pending(limit=1)
                        while True:
                            trial = execution.get_trial(trial_id)
                            if trial["status"] == "completed":
                                record["result"] = trial["result"]
                                break
                            if trial["status"] == "failed":
                                detail = trial.get("traceback") or trial.get("error") or "Unknown Dask task failure"
                                raise RuntimeError(detail)
                            time.sleep(0.1)
                    finally:
                        execution.client.close()
                else:
                    record["result"] = fn()
                record["status"] = "completed"
            except Exception:
                record["status"] = "failed"
                record["error"] = traceback.format_exc()
            record["updated_at"] = _now()
            self._write(record)

        _executor.submit(_run)
        return job_id


job_manager = JobManager()
