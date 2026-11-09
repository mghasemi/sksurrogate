"""Structured, per-job log capture for the SKSurrogate control plane.

The optimization core emits standard-library ``logging`` records on the
``SKSurrogate`` logger namespace (see :doc:`telemetry`). This module turns those
records — plus lifecycle lines written by the job manager itself — into a
queryable per-job event stream:

* an in-memory ring buffer per job for fast reads during a run,
* an append-only JSONL file per job (``<API home>/logs/<job_id>.jsonl``) so the
  history survives an API restart,
* a ``logging.Handler`` that routes records to whichever job is bound to the
  current thread through :func:`set_current_job` (a ``ContextVar``), which keeps
  concurrent jobs from mixing their lines.

Entries are plain JSON-safe dictionaries::

    {"seq": 7, "ts": "2026-11-09T01:02:03.456+00:00", "level": "WARNING",
     "logger": "SKSurrogate.structsearch", "message": "...", "job_id": "...",
     "task_name": "credit-g", "kind": "experiment", "trial": 12, "phase": "explore"}

Severity vocabulary is the standard library's: ``DEBUG`` < ``INFO`` < ``WARNING``
< ``ERROR`` < ``CRITICAL``. Queries take either a minimum severity (``level``) or
an explicit subset (``levels``).
"""

import contextvars
import json
import logging
import threading
import traceback
from collections import deque
from datetime import datetime, timezone

from .config import settings

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_LEVEL_NUMBER = {name: getattr(logging, name) for name in LEVELS}
DEFAULT_MAX_ENTRIES = 2000

_current_job = contextvars.ContextVar("sksurrogate_log_job", default=None)

def _now():
    return datetime.now(timezone.utc).isoformat()


def normalise_level(value, default="DEBUG"):
    """Return a canonical level name; raise ``ValueError`` for unknown input."""
    name = str(value or default).strip().upper()
    if name not in _LEVEL_NUMBER:
        raise ValueError(
            "Unknown log level %r; expected one of %s" % (value, ", ".join(LEVELS))
        )
    return name


def parse_levels(level=None, levels=None):
    """Resolve a ``level`` (minimum severity) or ``levels`` (explicit set).

    ``level`` wins when both are given. Both empty means "everything".
    """
    if level:
        minimum = _LEVEL_NUMBER[normalise_level(level)]
        return [name for name in LEVELS if _LEVEL_NUMBER[name] >= minimum]
    if levels:
        wanted = {
            normalise_level(item)
            for item in str(levels).split(",")
            if str(item).strip()
        }
        return [name for name in LEVELS if name in wanted]
    return list(LEVELS)


def set_current_job(job):
    """Bind ``job`` (mapping with ``job_id``/``task_name``/``kind``) to this context."""
    return _current_job.set(dict(job))


def reset_current_job(token):
    """Undo a previous :func:`set_current_job`."""
    try:
        _current_job.reset(token)
    except ValueError:  # token created in another context
        _current_job.set(None)


class JobLogStore:
    """Ring-buffer + JSONL persistence of per-job log entries."""

    def __init__(self, root=None, max_entries=DEFAULT_MAX_ENTRIES):
        self._root = root
        self.max_entries = int(max_entries)
        self._entries = {}
        self._counts = {}
        self._seq = {}
        self._lock = threading.RLock()

    # -- storage ---------------------------------------------------------
    @property
    def root(self):
        return self._root if self._root is not None else settings.logs_dir

    def path_for(self, job_id):
        return self.root / (str(job_id) + ".jsonl")

    def _append_file(self, job_id, entry):
        try:
            path = self.path_for(job_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            # A read-only or full disk must never take down a running job.
            pass

    def _load_from_file(self, job_id):
        path = self.path_for(job_id)
        if not path.exists():
            return []
        loaded = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if isinstance(entry, dict):
                    loaded.append(entry)
        except OSError:
            return []
        return loaded

    # -- writes ----------------------------------------------------------
    def append(self, job, level, message, logger_name="sksurrogate.api", **fields):
        """Append one entry for ``job``; returns the stored (JSON-safe) entry."""
        job_id = str(job["job_id"])
        level_name = normalise_level(level)
        with self._lock:
            seq = self._seq.get(job_id, 0) + 1
            self._seq[job_id] = seq
            entry = {
                "seq": seq,
                "ts": _now(),
                "level": level_name,
                "logger": str(logger_name),
                "message": str(message),
                "job_id": job_id,
                "task_name": job.get("task_name"),
                "kind": job.get("kind"),
            }
            for key, value in fields.items():
                if value is not None:
                    entry[key] = value
            entries = self._entries.get(job_id)
            if entries is None:
                entries = deque(maxlen=self.max_entries)
                self._entries[job_id] = entries
            entries.append(entry)
            counts = self._counts.setdefault(job_id, {name: 0 for name in LEVELS})
            counts[level_name] += 1
            self._append_file(job_id, entry)
            return dict(entry)

    # -- reads -----------------------------------------------------------
    def entries(self, job_id):
        """All known entries for one job (memory first, then the JSONL file)."""
        job_id = str(job_id)
        with self._lock:
            if job_id in self._entries:
                return [dict(entry) for entry in self._entries[job_id]]
        loaded = self._load_from_file(job_id)
        with self._lock:
            entries = deque(loaded[-self.max_entries:], maxlen=self.max_entries)
            self._entries[job_id] = entries
            counts = {name: 0 for name in LEVELS}
            seq = 0
            for entry in loaded:
                name = entry.get("level")
                if name in counts:
                    counts[name] += 1
                seq = max(seq, int(entry.get("seq") or 0))
            self._counts.setdefault(job_id, counts)
            self._seq[job_id] = max(self._seq.get(job_id, 0), seq)
            return [dict(entry) for entry in entries]

    def counts(self, job_id):
        """Per-level counts for one job (zeros when nothing was captured)."""
        job_id = str(job_id)
        with self._lock:
            if job_id in self._counts:
                return dict(self._counts[job_id])
        self.entries(job_id)
        with self._lock:
            return dict(self._counts.get(job_id, {name: 0 for name in LEVELS}))

    def known_jobs(self):
        with self._lock:
            in_memory = set(self._entries)
        try:
            on_disk = {path.stem for path in self.root.glob("*.jsonl")}
        except OSError:
            on_disk = set()
        return in_memory | on_disk

    def query(
        self,
        *,
        job_id=None,
        level=None,
        levels=None,
        query_text=None,
        kind=None,
        limit=200,
        since_seq=None,
        newest_first=True,
    ):
        """Filter entries by job, severity, kind and free text.

        Returns ``{"entries": [...], "counts": {...}, "total": n}`` where
        ``counts`` covers the returned selection and ``total`` is its size
        before ``limit`` was applied.
        """
        allowed = set(parse_levels(level=level, levels=levels))
        job_ids = [str(job_id)] if job_id else sorted(self.known_jobs())
        needle = (query_text or "").strip().lower()
        selected = []
        for current in job_ids:
            for entry in self.entries(current):
                if entry.get("level") not in allowed:
                    continue
                if kind and entry.get("kind") != kind:
                    continue
                if since_seq is not None and int(entry.get("seq") or 0) <= int(since_seq):
                    continue
                if needle:
                    haystack = "%s %s %s" % (
                        entry.get("message", ""),
                        entry.get("logger", ""),
                        entry.get("task_name", ""),
                    )
                    if needle not in haystack.lower():
                        continue
                selected.append(entry)
        if newest_first:
            selected.sort(key=lambda item: (item.get("ts", ""), item.get("seq", 0)), reverse=True)
        counts = {name: 0 for name in LEVELS}
        for entry in selected:
            name = entry.get("level")
            if name in counts:
                counts[name] += 1
        total = len(selected)
        if limit is not None:
            selected = selected[: int(limit)]
        return {"entries": selected, "counts": counts, "total": total}

    def clear(self, job_id=None):
        """Drop stored entries (memory and file) — test/administration helper."""
        with self._lock:
            targets = [str(job_id)] if job_id else sorted(self.known_jobs())
            for target in targets:
                self._entries.pop(target, None)
                self._counts.pop(target, None)
                self._seq.pop(target, None)
                try:
                    self.path_for(target).unlink()
                except OSError:
                    pass


class _JobLogHandler(logging.Handler):
    """Forward records of a bound job into the :class:`JobLogStore`."""

    def __init__(self, store):
        super().__init__(level=logging.DEBUG)
        self.store = store

    def emit(self, record):
        job = _current_job.get()
        if not job:
            return
        try:
            message = record.getMessage()
        except Exception:  # a malformed format string must not break logging
            message = str(record.msg)
        fields = {}
        detail = record.__dict__.get("detail")
        if record.exc_info:
            detail = "".join(traceback.format_exception(*record.exc_info))
        if detail:
            fields["detail"] = str(detail)
        for name in ("trial", "phase", "status", "task_name_extra"):
            value = record.__dict__.get(name)
            if value is not None:
                fields[name] = value
        try:
            self.store.append(
                job, record.levelname, message, logger_name=record.name, **fields
            )
        except Exception:
            self.handleError(record)


store = JobLogStore()
_handler = None
_install_lock = threading.Lock()
CAPTURED_LOGGERS = ("SKSurrogate", "sksurrogate.api", "sksurrogate.ui")


def install(log_store=None):
    """Attach the capture handler to the package loggers (idempotent)."""
    global _handler
    with _install_lock:
        if _handler is not None:
            return _handler
        _handler = _JobLogHandler(log_store or store)
        for name in CAPTURED_LOGGERS:
            logger = logging.getLogger(name)
            logger.addHandler(_handler)
            if logger.level == logging.NOTSET or logger.level > logging.DEBUG:
                # Capture DEBUG+ regardless of the root configuration; the handler
                # decides what to keep and the stdout fallback still applies to
                # un-handled loggers.
                logger.setLevel(logging.DEBUG)
        return _handler


def uninstall():
    """Detach the capture handler (used by tests)."""
    global _handler
    with _install_lock:
        if _handler is None:
            return
        for name in CAPTURED_LOGGERS:
            logging.getLogger(name).removeHandler(_handler)
        _handler = None


def log_event(level, message, job=None, **fields):
    """Write one entry for ``job`` (defaults to the context-bound job)."""
    target = job if job is not None else _current_job.get()
    if not target:
        return None
    return store.append(target, level, message, **fields)


_FAILURE_WARN_LIMIT = 5
_failure_counts = {}
_failure_lock = threading.Lock()


def trial_callback(event):
    """``callbacks`` hook that turns a core trial event into a log entry.

    Failed or rejected trials are logged at WARNING, but only the first
    ``_FAILURE_WARN_LIMIT`` per job/status pair — a job whose every candidate
    fails would otherwise bury the rest of the log in identical warnings. Later
    occurrences are kept at DEBUG and remain inspectable through the filters.
    """
    trial = event.get("trial")
    phase = event.get("phase")
    status = event.get("status")
    score = event.get("score")
    params = event.get("params")
    level = "INFO"
    if status in ("failed", "invalid"):
        level = "WARNING"
        job = _current_job.get()
        key = (job.get("job_id") if job else None, status)
        with _failure_lock:
            seen = _failure_counts.get(key, 0) + 1
            _failure_counts[key] = seen
        if seen > _FAILURE_WARN_LIMIT:
            level = "DEBUG"
    elif status == "pruned":
        level = "DEBUG"
    message = "trial %s (%s) %s" % (trial, phase or "unknown phase", status)
    if score is not None:
        message += " score=%s" % score
    if params:
        message += " params=%s" % json.dumps(params, default=str, sort_keys=True)
    return log_event(level, message, trial=trial, phase=phase, status=status)
