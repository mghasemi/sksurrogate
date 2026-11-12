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
from datetime import datetime, timezone

from . import compute, control, logs
from .config import settings
from .safety import normalize_label

# ``_write_lock`` is re-entrant because the heartbeat thread checks the record's
# status and writes it under the same lock.
_write_lock = threading.RLock()
_idempotency_lock = threading.Lock()


def _now():
    return datetime.now(timezone.utc).isoformat()


def auto_display_name(record):
    """Readable fallback name for a job that has no user-chosen label.

    Machine ids (``job_id`` is a uuid4 hex) are the canonical key and must stay
    stable, but they are a poor way to *track* work: this name is the human
    handle and is always derivable from fields every record already carries.
    """
    kind = str(record.get("kind") or "job").replace("_", " ")
    task = record.get("task_name") or "unassigned task"
    created = record.get("created_at")
    try:
        when = datetime.fromisoformat(str(created)).astimezone().strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        when = str(created or "unknown time")
    return "%s · %s · %s" % (kind, task, when)


def display_name(record):
    """The user-facing name: a custom ``label`` when set, else :func:`auto_display_name`."""
    label = record.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    return auto_display_name(record)


def public_record(record):
    """Add the derived ``display_name`` to a persisted record (never stored twice)."""
    return {**record, "display_name": display_name(record)}


def dask_available():
    """Return whether the optional Dask distributed client can be imported."""
    try:
        importlib.import_module("dask.distributed")
    except (ImportError, ModuleNotFoundError):
        return False
    return True


class JobManager:
    """Submits callables to a thread pool and persists their lifecycle as JSON.

    ``_live`` is the authoritative answer to "is this job actually running?": a
    job is live only while its control object is registered by this process, so
    a JSON record left at ``running`` by a crashed API can never be mistaken for
    active work (see :meth:`reconcile` and ``/api/jobs/active``).
    """

    def __init__(self):
        self._live = {}
        self._live_lock = threading.RLock()

    # -- liveness ------------------------------------------------------
    def is_live(self, job_id):
        with self._live_lock:
            return str(job_id) in self._live

    def live_job_ids(self):
        with self._live_lock:
            return list(self._live)

    def control_for(self, job_id):
        with self._live_lock:
            return self._live.get(str(job_id))

    def _register(self, control_obj):
        with self._live_lock:
            self._live[control_obj.job_id] = control_obj
        return control_obj

    def _unregister(self, job_id):
        with self._live_lock:
            self._live.pop(job_id, None)

    def request_stop(self, job_id, mode, requested_by=None):
        """Ask a live job to stop at its next safe point.

        Raises ``KeyError`` when the job is not live in this process (finished,
        or a stale record from a previous run) and ``RuntimeError`` when the
        kind has no checkpoint to resume from, which the router surfaces as 409.
        """
        control_obj = self.control_for(job_id)
        if control_obj is None:
            raise KeyError(job_id)
        if mode == "pause" and not self._pause_supported(job_id):
            raise RuntimeError(
                "This job kind cannot be paused mid-flight; stop it instead"
            )
        control_obj.request(mode, requested_by=requested_by)
        record = self.get(job_id)
        if record is not None:
            record["stop_requested_at"] = control_obj.requested_at
            record["stop_mode"] = mode
            record["stop_requested_by"] = requested_by
            record["updated_at"] = _now()
            self._write(record)
        return record

    def _pause_supported(self, job_id):
        """Pausing needs a checkpoint plus a saved resume request."""
        record = self.get(job_id)
        if not isinstance(record, dict):
            return False
        return bool(record.get("resumable")) and isinstance(record.get("resume"), dict)

    def mark_interrupted(self, job_id, reason=None):
        """Persist a stale ``queued``/``running`` record as ``interrupted``."""
        record = self.get(job_id)
        if record is None:
            return None
        if record.get("status") not in ("queued", "running") or self.is_live(job_id):
            return record
        record["status"] = "interrupted"
        record["interrupted_at"] = _now()
        record["updated_at"] = _now()
        if not record.get("error"):
            record["error"] = reason or (
                "The API process that owned this job exited before it finished; "
                "the job is not running."
            )
        self._write(record)
        return record

    def reconcile(self):
        """At startup, correct every record left ``queued``/``running`` by a dead process."""
        corrected = []
        try:
            paths = list(settings.jobs_dir.glob("*.json"))
        except OSError:
            return corrected
        for path in paths:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(record, dict):
                continue
            job_id = record.get("job_id")
            if record.get("status") in ("queued", "running") and not self.is_live(job_id):
                updated = self.mark_interrupted(job_id)
                if updated is not None:
                    corrected.append(job_id)
        return corrected

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
        return public_record(json.loads(path.read_text(encoding="utf-8")))

    def list(self, task_name=None):
        records = [
            public_record(json.loads(path.read_text(encoding="utf-8")))
            for path in settings.jobs_dir.glob("*.json")
        ]
        if task_name is not None:
            records = [record for record in records if record["task_name"] == task_name]
        return sorted(records, key=lambda record: record["created_at"], reverse=True)

    def rename(self, job_id, label, *, actor=None):
        """Set (or clear, with ``None``) the job's custom label and return the record.

        Labels are a display convenience: the ``job_id`` remains the identity used
        by routes, artifacts and checkpoints. The change is recorded with
        ``aliased_at``/``aliased_by`` so a renamed job keeps its provenance.
        """
        record = self.get(job_id)
        if record is None:
            return None
        cleaned = normalize_label(label)
        record["label"] = cleaned
        record["aliased_at"] = _now()
        record["aliased_by"] = actor
        record["updated_at"] = _now()
        record["display_name"] = display_name(record)
        self._write(record)
        return record

    def submit(self, task_name, kind, fn_factory, *, backend="local", metadata_factory=None, label=None):
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
            # Human handle; the uuid above stays the canonical identity.
            "label": normalize_label(label),
            # What the job is allowed to use. Recorded at submit time because a
            # settings change afterwards must not rewrite the history of a run.
            "compute": compute.limits(),
        }
        if metadata_factory is not None:
            record.update(metadata_factory(job_id))
        # Only jobs that carry a re-submittable request can be paused and
        # resumed later; the rest can still be cancelled.
        record["resumable"] = isinstance(record.get("resume"), dict)
        control_obj = self._register(control.JobControl(job_id))
        record["heartbeat_at"] = None
        self._write(record)
        logs.store.append(
            record,
            "INFO",
            "Job accepted (kind=%s, backend=%s)" % (kind, backend),
        )
        record["log_counts"] = logs.store.counts(job_id)
        self._write(record)

        def _run():
            token = logs.set_current_job(
                {
                    "job_id": job_id,
                    "task_name": task_name,
                    "kind": kind,
                }
            )
            control_token = control.set_current(control_obj)
            control_obj.start()
            heartbeat_stop = threading.Event()
            heartbeat = None
            try:
                record["status"] = "running"
                record["updated_at"] = _now()
                record["heartbeat_at"] = control_obj.heartbeat()
                # Pin BLAS/OpenMP before any numeric work starts in this thread,
                # so the job's own fits and its loky children respect the budget.
                record["compute_applied"] = compute.apply_limits()
                logs.log_event("INFO", "Job started (kind=%s)" % kind)
                record["log_counts"] = logs.store.counts(job_id)
                self._write(record)

                heartbeat = threading.Thread(
                    target=self._heartbeat_loop,
                    args=(control_obj, record, heartbeat_stop),
                    name="sksurrogate-heartbeat-%s" % job_id[:8],
                    daemon=True,
                )
                heartbeat.start()

                # A stop request that arrived while the job was queued is
                # honoured before any work starts.
                if control_obj.should_stop():
                    from SKSurrogate.structsearch import SearchStopped

                    raise SearchStopped("stop requested before the job started")
                fn = fn_factory(job_id)
                if backend == "dask":
                    from SKSurrogate.execution import DaskExecutionBackend

                    cluster = compute.dask_cluster()
                    if cluster is None:
                        raise RuntimeError(
                            "The Dask backend requires the optional 'dask[distributed]' package"
                        )
                    client = cluster.get_client()
                    execution = DaskExecutionBackend(
                        settings.jobs_dir / "execution",
                        task_name=job_id,
                        client=client,
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
                        client.close()
                        cluster.close()
                else:
                    record["result"] = fn()
                record["status"] = "completed"
                logs.log_event("INFO", "Job completed (kind=%s)" % kind)
            except Exception as exc:
                from SKSurrogate.structsearch import SearchStopped

                if isinstance(exc, SearchStopped):
                    # A stop request is not a failure: the job keeps its
                    # checkpoint (pause) or is abandoned (cancel).
                    self._finish_stopped(record, control_obj, kind)
                else:
                    record["status"] = "failed"
                    record["error"] = traceback.format_exc()
                    frames = traceback.extract_tb(exc.__traceback__)
                    location = (
                        " at %s:%s" % (frames[-1].filename.rsplit("/", 1)[-1], frames[-1].lineno)
                        if frames
                        else ""
                    )
                    logs.log_event(
                        "ERROR",
                        "Job failed: %s: %s%s" % (type(exc).__name__, exc, location),
                        detail=record["error"],
                    )
            finally:
                # Cleanup must run even if the bookkeeping above raised: a job
                # left registered as live would block its checkpoint for ever.
                heartbeat_stop.set()
                if heartbeat is not None:
                    heartbeat.join(timeout=1)
                try:
                    record["updated_at"] = _now()
                    record["log_counts"] = logs.store.counts(job_id)
                except Exception:  # pragma: no cover - log store unavailable
                    pass
                self._merge_control_state(record, control_obj)
                try:
                    self._write(record)
                finally:
                    control_obj.finish()
                    self._unregister(job_id)
                    control.reset_current(control_token)
                    logs.reset_current_job(token)

        compute.job_executor().submit(_run)
        return job_id

    def _finish_stopped(self, record, control_obj, kind):
        """Record a stop request as ``paused`` (resumable) or ``cancelled``."""
        mode = control_obj.mode or "cancel"
        stopped_at = _now()
        record["stop"] = {
            "mode": mode,
            "requested_by": control_obj.requested_by,
            "requested_at": control_obj.requested_at,
            "stopped_at": stopped_at,
        }
        if mode == "pause":
            record["status"] = "paused"
            record["paused_at"] = stopped_at
            logs.log_event(
                "INFO",
                "Job paused by request (kind=%s); resume it to continue from its checkpoint" % kind,
            )
        else:
            record["status"] = "cancelled"
            record["cancelled_at"] = stopped_at
            logs.log_event("INFO", "Job cancelled by request (kind=%s)" % kind)
        return record

    @staticmethod
    def _merge_control_state(record, control_obj):
        """Keep pending stop fields on the record this thread writes.

        ``request_stop`` writes its own copy of the record; without this the next
        heartbeat (or the final result write) would drop ``stop_requested_at``
        and the UI would lose its *stopping…* state until the job exits.
        """
        if control_obj.requested_at:
            record["stop_requested_at"] = control_obj.requested_at
            record["stop_mode"] = control_obj.mode
            record["stop_requested_by"] = control_obj.requested_by
        return record

    def _heartbeat_loop(self, control_obj, record, stop_event):
        """Refresh ``heartbeat_at`` for a live job so a hung worker is visible."""
        while not stop_event.wait(control.HEARTBEAT_SECONDS):
            with _write_lock:
                if record.get("status") not in ("queued", "running"):
                    return
                if control_obj.finished_at is not None:
                    return
                record["heartbeat_at"] = control_obj.heartbeat()
                self._merge_control_state(record, control_obj)
                try:
                    self._write(record)
                except OSError:
                    return

    def submit_idempotent(
        self,
        task_name,
        kind,
        idempotency_key,
        request_fingerprint,
        fn_factory,
        *,
        backend="local",
        metadata_factory=None,
        label=None,
    ):
        """Submit once per task/kind/key and reject key reuse for another request.

        The lock coordinates submitters in this API process. The JSON job store
        is not a distributed queue and does not provide cross-process locking.
        """
        with _idempotency_lock:
            for record in self.list(task_name):
                if (
                    record.get("kind") == kind
                    and record.get("idempotency_key") == idempotency_key
                ):
                    if record.get("request_fingerprint") != request_fingerprint:
                        raise ValueError(
                            "Idempotency key was already used for a different request"
                        )
                    return record["job_id"], False

            def idempotent_metadata(job_id):
                metadata = {
                    "idempotency_key": idempotency_key,
                    "request_fingerprint": request_fingerprint,
                }
                if metadata_factory is not None:
                    metadata.update(metadata_factory(job_id))
                return metadata

            job_id = self.submit(
                task_name,
                kind,
                fn_factory,
                backend=backend,
                metadata_factory=idempotent_metadata,
                label=label,
            )
            return job_id, True


job_manager = JobManager()
