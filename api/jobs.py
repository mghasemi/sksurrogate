"""A minimal persisted background-job manager.

Long-running work (AML/EOA searches, sensitivity analysis) is submitted here
instead of running inline in the request/response cycle (see docs/ui-plan.md
section 5). Jobs run on a thread pool in this process and their status is
persisted as JSON so ``GET /api/jobs/{id}`` keeps working across restarts;
this intentionally stays file-based rather than pulling in Celery/Redis.
"""

import json
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .config import settings

_executor = ThreadPoolExecutor(max_workers=4)
_write_lock = threading.Lock()


def _now():
    return datetime.now(timezone.utc).isoformat()


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

    def submit(self, task_name, kind, fn_factory):
        """Run ``fn_factory(job_id)()`` in the background; result must be a JSON-safe dict."""
        job_id = uuid.uuid4().hex
        record = {
            "job_id": job_id,
            "task_name": task_name,
            "kind": kind,
            "status": "queued",
            "created_at": _now(),
            "updated_at": _now(),
            "result": None,
            "error": None,
        }
        self._write(record)

        def _run():
            record["status"] = "running"
            record["updated_at"] = _now()
            self._write(record)
            try:
                record["result"] = fn_factory(job_id)()
                record["status"] = "completed"
            except Exception:
                record["status"] = "failed"
                record["error"] = traceback.format_exc()
            record["updated_at"] = _now()
            self._write(record)

        _executor.submit(_run)
        return job_id


job_manager = JobManager()
