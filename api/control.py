"""Cooperative job control: pause/cancel requests, liveness and heartbeats.

A Python thread cannot be killed, so stopping a job means asking it to return at
its next safe point: ``SKSurrogate``'s searches accept a ``stop_check`` callable
and raise ``SearchStopped`` between candidate evaluations. This module owns the
per-job request object and exposes it through a ``ContextVar`` that the job
thread binds — exactly the mechanism :mod:`api.logs` uses for log routing — so
deep call sites (``api/training``) can reach the control of the job they serve.

Liveness is process-local by construction: a job is "live" only while its
control object is registered by :class:`api.jobs.JobManager`, which is what lets
the UI distinguish a genuinely running job from a record left behind by a
crashed API process.
"""

import contextvars
import threading
from datetime import datetime, timezone

#: Accepted stop modes. ``pause`` keeps the checkpoint so the run can resume;
#: ``cancel`` abandons it and is final.
STOP_MODES = ("pause", "cancel")

#: Seconds between heartbeat writes for a live job.
HEARTBEAT_SECONDS = 10


def _now():
    return datetime.now(timezone.utc).isoformat()


class JobControl:
    """One job's stop request plus its liveness timestamps."""

    def __init__(self, job_id):
        self.job_id = job_id
        self._stop = threading.Event()
        self.mode = None
        self.requested_by = None
        self.requested_at = None
        self.started_at = None
        self.finished_at = None
        self.heartbeat_at = None

    def request(self, mode, requested_by=None):
        """Record a stop request; the latest request wins."""
        if mode not in STOP_MODES:
            raise ValueError("stop mode must be one of %s" % (", ".join(STOP_MODES),))
        self.mode = mode
        self.requested_by = requested_by
        self.requested_at = _now()
        self._stop.set()
        return self

    def should_stop(self):
        return self._stop.is_set()

    def heartbeat(self):
        self.heartbeat_at = _now()
        return self.heartbeat_at

    def start(self):
        self.started_at = _now()
        return self

    def finish(self):
        self.finished_at = _now()
        self.heartbeat_at = self.finished_at
        return self


_current_control = contextvars.ContextVar("sksurrogate_job_control", default=None)


def set_current(control):
    """Bind ``control`` to this thread's context (the per-job execution context)."""
    return _current_control.set(control)


def reset_current(token):
    try:
        _current_control.reset(token)
    except ValueError:  # token created in another context
        _current_control.set(None)


def current():
    """The control of the job running in this context, or ``None``."""
    return _current_control.get()


def stop_requested():
    """``stop_check`` implementation: True once this job has been asked to stop."""
    control = _current_control.get()
    return bool(control is not None and control.should_stop())
