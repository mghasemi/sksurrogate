"""Execution backends for distributed and resumable search workloads."""

from __future__ import annotations

import base64
import json
import os
import pickle
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import cloudpickle
except ImportError:  # pragma: no cover - fallback for minimal environments
    cloudpickle = None


class ExecutionBackend:
    """Base interface for execution backends used by long-running searches."""

    def submit_trial(self, params: Any, fn, worker_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        raise NotImplementedError

    def get_trial(self, trial_id: str):
        raise NotImplementedError

    def mark_running(self, trial_id: str, worker_id: Optional[str] = None):
        raise NotImplementedError

    def mark_completed(self, trial_id: str, result: Any = None, worker_id: Optional[str] = None):
        raise NotImplementedError

    def mark_failed(self, trial_id: str, error: Optional[str] = None, traceback: Optional[str] = None, worker_id: Optional[str] = None):
        raise NotImplementedError

    def execute_pending(self, worker_id: Optional[str] = None, limit: Optional[int] = None):
        raise NotImplementedError

    def claim_trial(self, trial_id: str, worker_id: Optional[str] = None):
        raise NotImplementedError

    def claim_next_trial(self, worker_id: Optional[str] = None):
        raise NotImplementedError

    def list_pending(self, *, task_name: Optional[str] = None):
        raise NotImplementedError

    def list_results(self, *, status: Optional[str] = "completed", task_name: Optional[str] = None):
        raise NotImplementedError

    def resume_incomplete_trials(self):
        raise NotImplementedError


class LocalProcessExecutionBackend(ExecutionBackend):
    """Simple filesystem-backed backend for local workers and resumable jobs.

    Trial state is persisted as JSON so a worker process can resume incomplete
    work after a restart without losing the original parameter set or callable.
    """

    def __init__(self, root, task_name="default"):
        self.root = Path(root)
        self.task_name = task_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / f"{task_name}.execution_state.json"
        self._lock = threading.RLock()
        self._state = self._load_state()

    @staticmethod
    def _now_iso():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _encode_callable(fn):
        if cloudpickle is not None:
            payload = cloudpickle.dumps(fn)
        else:
            payload = pickle.dumps(fn)
        return base64.b64encode(payload).decode("latin1")

    @staticmethod
    def _decode_callable(blob):
        payload = base64.b64decode(blob.encode("latin1"))
        if cloudpickle is not None:
            return cloudpickle.loads(payload)
        return pickle.loads(payload)

    @staticmethod
    def _json_default(value):
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except TypeError:
                pass
        if hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool, type(None), dict, list, tuple)):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def _load_state(self):
        if not self.state_path.exists():
            return {}
        with self.state_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        return dict(data or {})

    def _save_state(self):
        tmp_path = self.state_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as stream:
            json.dump(self._state, stream, indent=2, sort_keys=True, default=self._json_default)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, self.state_path)

    def _restore_record(self, record):
        if not isinstance(record, dict):
            return record
        restored = dict(record)
        if "_callable_blob" in restored:
            restored["fn"] = self._decode_callable(restored["_callable_blob"])
            restored.pop("_callable_blob", None)
        return restored

    def _persist_record(self, trial_id, record):
        self._state[trial_id] = {
            **record,
            "trial_id": trial_id,
            "task_name": self.task_name,
            "updated_at": self._now_iso(),
        }
        if "fn" in self._state[trial_id] and callable(self._state[trial_id]["fn"]):
            self._state[trial_id]["_callable_blob"] = self._encode_callable(self._state[trial_id]["fn"])
            self._state[trial_id].pop("fn", None)
        self._save_state()

    def submit_trial(self, params: Any, fn, worker_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        if not callable(fn):
            raise TypeError("fn must be callable")
        trial_id = uuid.uuid4().hex
        record = {
            "trial_id": trial_id,
            "task_name": self.task_name,
            "status": "queued",
            "params": params,
            "fn": fn,
            "worker_id": worker_id,
            "metadata": {} if metadata is None else dict(metadata),
            "created_at": self._now_iso(),
            "updated_at": self._now_iso(),
        }
        with self._lock:
            self._persist_record(trial_id, record)
        return trial_id

    def get_trial(self, trial_id: str):
        with self._lock:
            record = self._state.get(trial_id)
            if record is None:
                raise KeyError(f"Unknown trial: {trial_id}")
            restored = self._restore_record(record)
            restored.pop("_callable_blob", None)
            return restored

    def mark_running(self, trial_id: str, worker_id: Optional[str] = None):
        with self._lock:
            record = self._state.get(trial_id)
            if record is None:
                raise KeyError(f"Unknown trial: {trial_id}")
            record["status"] = "running"
            record["worker_id"] = worker_id or record.get("worker_id")
            record["updated_at"] = self._now_iso()
            self._save_state()
            return self._restore_record(record)

    def mark_completed(self, trial_id: str, result: Any = None, worker_id: Optional[str] = None):
        with self._lock:
            record = self._state.get(trial_id)
            if record is None:
                raise KeyError(f"Unknown trial: {trial_id}")
            record["status"] = "completed"
            record["result"] = result
            record["worker_id"] = worker_id or record.get("worker_id")
            record["updated_at"] = self._now_iso()
            self._save_state()
            return self._restore_record(record)

    def mark_failed(self, trial_id: str, error: Optional[str] = None, traceback: Optional[str] = None, worker_id: Optional[str] = None):
        with self._lock:
            record = self._state.get(trial_id)
            if record is None:
                raise KeyError(f"Unknown trial: {trial_id}")
            record["status"] = "failed"
            record["error"] = error
            record["traceback"] = traceback
            record["worker_id"] = worker_id or record.get("worker_id")
            record["updated_at"] = self._now_iso()
            self._save_state()
            return self._restore_record(record)

    def execute_pending(self, worker_id: Optional[str] = None, limit: Optional[int] = None):
        """Execute queued trials in a recoverable, persisted loop.

        Each pending item is marked as running before execution, and the result is
        persisted as completed or failed. The method returns a list of record
        snapshots describing what was executed.
        """
        with self._lock:
            pending = [
                (trial_id, record)
                for trial_id, record in self._state.items()
                if record.get("status") in {"queued", "running"}
            ]
            if limit is not None:
                pending = pending[:limit]

            executed = []
            for trial_id, record in pending:
                fn = self._restore_record(record).get("fn")
                if fn is None:
                    self.mark_failed(trial_id, error="Missing callable for pending trial", worker_id=worker_id)
                    executed.append(self.get_trial(trial_id))
                    continue

                record["status"] = "running"
                record["worker_id"] = worker_id or record.get("worker_id")
                record["updated_at"] = self._now_iso()
                self._save_state()

                try:
                    result = fn(record.get("params"))
                except Exception as exc:
                    import traceback

                    message = f"{exc.__class__.__name__}: {exc}"
                    record["status"] = "failed"
                    record["error"] = message
                    record["traceback"] = traceback.format_exc()
                    record["worker_id"] = worker_id or record.get("worker_id")
                    record["updated_at"] = self._now_iso()
                    self._save_state()
                    executed.append(self._restore_record(record))
                    continue

                record["status"] = "completed"
                record["result"] = result
                record["worker_id"] = worker_id or record.get("worker_id")
                record["updated_at"] = self._now_iso()
                self._save_state()
                executed.append(self._restore_record(record))

            return executed

    def claim_trial(self, trial_id: str, worker_id: Optional[str] = None):
        with self._lock:
            record = self._state.get(trial_id)
            if record is None:
                raise KeyError(f"Unknown trial: {trial_id}")
            if record.get("status") not in {"queued", "running"}:
                return None
            if record.get("status") == "running" and record.get("worker_id") not in {None, worker_id}:
                return None
            record["status"] = "running"
            record["worker_id"] = worker_id or record.get("worker_id")
            record["updated_at"] = self._now_iso()
            self._save_state()
            return self._restore_record(record)

    def claim_next_trial(self, worker_id: Optional[str] = None):
        with self._lock:
            pending = [
                (trial_id, record)
                for trial_id, record in self._state.items()
                if record.get("status") == "queued"
            ]
            if not pending:
                return None
            ordered = sorted(pending, key=lambda item: item[1].get("created_at", ""))
            trial_id, record = ordered[0]
            record["status"] = "running"
            record["worker_id"] = worker_id
            record["updated_at"] = self._now_iso()
            self._save_state()
            return self._restore_record(record)

    def list_pending(self, *, task_name: Optional[str] = None):
        with self._lock:
            pending = []
            for trial_id, record in sorted(self._state.items(), key=lambda item: item[1].get("created_at", "")):
                if record.get("status") != "queued":
                    continue
                if task_name is not None and record.get("task_name") != task_name:
                    continue
                restored = self._restore_record(record)
                restored.pop("_callable_blob", None)
                pending.append(restored)
            return pending

    def list_results(self, *, status: Optional[str] = "completed", task_name: Optional[str] = None):
        with self._lock:
            results = []
            for trial_id, record in self._state.items():
                if record.get("status") != status:
                    continue
                if task_name is not None and record.get("task_name") != task_name:
                    continue
                restored = self._restore_record(record)
                restored.pop("_callable_blob", None)
                if "result" in restored:
                    results.append(restored)
            return results

    def resume_incomplete_trials(self):
        with self._lock:
            incomplete = []
            for trial_id, record in self._state.items():
                status = record.get("status")
                if status in {"queued", "running", "failed", "timed_out"}:
                    recovered = self._restore_record(record)
                    recovered.pop("_callable_blob", None)
                    incomplete.append(recovered)
            return incomplete

    def list_trials(self, status: Optional[str] = None):
        with self._lock:
            trials = [self._restore_record(record) for record in self._state.values()]
            if status is not None:
                trials = [trial for trial in trials if trial.get("status") == status]
            return trials


class DaskExecutionBackend(ExecutionBackend):
    """Optional Dask adapter backed by the durable local trial store."""

    def __init__(self, root, task_name="default", client=None, address=None):
        if client is None:
            try:
                from dask.distributed import Client
            except ImportError as exc:  # pragma: no cover - depends on optional package
                raise ImportError(
                    "DaskExecutionBackend requires dask[distributed] or an injected client"
                ) from exc
            client = Client(address) if address is not None else Client()
        self.client = client
        self.local = LocalProcessExecutionBackend(root, task_name=task_name)
        self._futures = {}

    def submit_trial(self, params: Any, fn, worker_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        return self.local.submit_trial(params, fn, worker_id=worker_id, metadata=metadata)

    def get_trial(self, trial_id: str):
        return self.local.get_trial(trial_id)

    def mark_running(self, trial_id: str, worker_id: Optional[str] = None):
        return self.local.mark_running(trial_id, worker_id=worker_id)

    def mark_completed(self, trial_id: str, result: Any = None, worker_id: Optional[str] = None):
        return self.local.mark_completed(trial_id, result=result, worker_id=worker_id)

    def mark_failed(self, trial_id: str, error: Optional[str] = None, traceback: Optional[str] = None, worker_id: Optional[str] = None):
        return self.local.mark_failed(trial_id, error=error, traceback=traceback, worker_id=worker_id)

    def _record_future_result(self, trial_id, future):
        worker_id = self.get_trial(trial_id).get("worker_id")
        try:
            self.mark_completed(trial_id, result=future.result(), worker_id=worker_id)
        except Exception as exc:
            import traceback

            self.mark_failed(
                trial_id,
                error=f"{exc.__class__.__name__}: {exc}",
                traceback=traceback.format_exc(),
                worker_id=worker_id,
            )
        finally:
            self._futures.pop(trial_id, None)

    def execute_pending(self, worker_id: Optional[str] = None, limit: Optional[int] = None):
        scheduled = []
        while limit is None or len(scheduled) < limit:
            trial = self.local.claim_next_trial(worker_id=worker_id)
            if trial is None:
                break
            future = self.client.submit(trial["fn"], trial.get("params"))
            self._futures[trial["trial_id"]] = future
            future.add_done_callback(
                lambda completed, trial_id=trial["trial_id"]: self._record_future_result(
                    trial_id, completed
                )
            )
            scheduled.append(self.get_trial(trial["trial_id"]))
        return scheduled

    def claim_trial(self, trial_id: str, worker_id: Optional[str] = None):
        return self.local.claim_trial(trial_id, worker_id=worker_id)

    def claim_next_trial(self, worker_id: Optional[str] = None):
        return self.local.claim_next_trial(worker_id=worker_id)

    def list_pending(self, *, task_name: Optional[str] = None):
        return self.local.list_pending(task_name=task_name)

    def list_results(self, *, status: Optional[str] = "completed", task_name: Optional[str] = None):
        return self.local.list_results(status=status, task_name=task_name)

    def resume_incomplete_trials(self):
        return self.local.resume_incomplete_trials()

    def list_trials(self, status: Optional[str] = None):
        return self.local.list_trials(status=status)


__all__ = ["ExecutionBackend", "LocalProcessExecutionBackend", "DaskExecutionBackend"]
