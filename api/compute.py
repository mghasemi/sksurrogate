"""Compute limits shared by every background job.

The API runs jobs on a thread pool inside one process, and each job's search
fans out over CV folds through joblib (a per-process loky pool). Without an
explicit budget those two layers compose into unbounded oversubscription:
``ThreadPoolExecutor(max_workers=4)`` times ``AML(n_jobs=-1)`` can want
``4 x cpu_count`` workers, plus one BLAS thread pool per job thread.

This module owns the budget. Operators set it from *Settings / Audit* (or
``PUT /api/settings/compute``); every value is capped by the cores this process
may actually use (``sched_getaffinity``, so a ``taskset``/cpuset-limited
container reports its real allotment). The strict invariant is::

    job_workers * blas_threads + job_threads <= cores
    dask_workers * dask_threads <= cores

``job_threads`` is a *total*, not a per-job number: joblib keeps one reusable
loky executor per process, so every job shares that pool. Sizing it as a total
is what keeps the limit honest when several jobs run at once.

Enforcement happens in three places:

* :func:`job_executor` sizes — and resizes — the job thread pool.
* :func:`apply_limits` sets ``OMP_NUM_THREADS`` & friends (children inherit
  them; joblib only fills in its own default when the variable is absent) and
  applies a process-wide threadpoolctl limit for in-process BLAS.
* ``api/routers/*`` pass ``n_jobs=job_threads`` to ``AML`` so the fold pool,
  ``cross_val_score`` and out-of-fold stacking all use the same width.

Settings apply to jobs submitted afterwards; a running job keeps the limits it
started with. The persisted file lives at ``<API home>/settings/compute.json``.
"""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from .config import settings

logger = logging.getLogger("sksurrogate.api")

#: Integer knobs, all positive, all bounded by the available core count.
KEYS = ("job_workers", "job_threads", "blas_threads", "dask_workers", "dask_threads")

#: Environment variables the thread libraries read at start-up. joblib's loky
#: backend only substitutes its own default when the variable is unset, so
#: setting these in the parent is what makes the limit survive into workers.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

_lock = threading.RLock()
_executor = None
_executor_workers = None


def cpu_count():
    """Cores this process may use (honours CPU affinity/cpusets, not quotas)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # pragma: no cover - non-Linux hosts
        return max(1, os.cpu_count() or 1)


def defaults(cores=None):
    """Default budget: four job slots, BLAS pinned to one thread per job.

    ``job_threads`` is the remainder after reserving one BLAS thread per job
    slot, which reproduces today's near-full-CPU fold pool without exceeding the
    core count. Four job slots is the historical ``ThreadPoolExecutor`` width,
    clamped down on small machines.
    """
    cores = int(cores or cpu_count())
    workers = min(4, cores)
    blas = 1
    return {
        "job_workers": workers,
        "job_threads": max(1, cores - workers * blas),
        "blas_threads": blas,
        "dask_workers": min(4, cores),
        "dask_threads": 1,
    }


def _coerce(values):
    """Return a clean int per key, or raise ``ValueError`` naming the field."""
    cleaned = {}
    for key in KEYS:
        if key not in values:
            raise ValueError("Missing setting %r" % key)
        raw = values[key]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError("%s must be an integer" % key)
        if raw < 1:
            raise ValueError("%s must be at least 1" % key)
        cleaned[key] = raw
    return cleaned


def validate(values, cores=None):
    """Validate a candidate budget against the core count; return the cleaned copy."""
    cores = int(cores or cpu_count())
    cleaned = _coerce(values)
    for key in ("job_workers", "job_threads", "blas_threads", "dask_workers", "dask_threads"):
        if cleaned[key] > cores:
            raise ValueError(
                "%s = %d exceeds the %d cores available to this API process"
                % (key, cleaned[key], cores)
            )

    used = cleaned["job_workers"] * cleaned["blas_threads"] + cleaned["job_threads"]
    if used > cores:
        spare = cores - cleaned["job_workers"] * cleaned["blas_threads"]
        raise ValueError(
            "job_workers x blas_threads + job_threads = %d x %d + %d = %d exceeds the "
            "%d available cores; reduce job_threads to at most %d (or job_workers / "
            "blas_threads)"
            % (
                cleaned["job_workers"],
                cleaned["blas_threads"],
                cleaned["job_threads"],
                used,
                cores,
                max(1, spare),
            )
        )

    dask_used = cleaned["dask_workers"] * cleaned["dask_threads"]
    if dask_used > cores:
        raise ValueError(
            "dask_workers x dask_threads = %d x %d = %d exceeds the %d available cores"
            % (cleaned["dask_workers"], cleaned["dask_threads"], dask_used, cores)
        )
    return cleaned


def path():
    """Location of the persisted compute settings."""
    return settings.compute_settings_path


def load(cores=None):
    """Persisted values, falling back to :func:`defaults` when absent or unusable.

    A stale file (e.g. written on a bigger host, or edited by hand) is not
    trusted: it is re-validated and replaced in memory by the defaults when it
    no longer fits, so the API can never start with an impossible budget.
    """
    cores = int(cores or cpu_count())
    target = path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return defaults(cores)
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable compute settings at %s: %s", target, exc)
        return defaults(cores)
    if not isinstance(raw, dict):
        logger.warning("Ignoring malformed compute settings at %s", target)
        return defaults(cores)
    try:
        return validate(raw, cores=cores)
    except ValueError as exc:
        logger.warning(
            "Compute settings at %s do not fit this host (%s); using defaults", target, exc
        )
        return defaults(cores)


def current():
    """The effective budget (persisted values, or defaults)."""
    return load()


def save(values):
    """Validate, persist atomically, and apply the new limits immediately."""
    cores = cpu_count()
    cleaned = validate(values, cores=cores)
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(cleaned, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(target)
    apply_limits(cleaned)
    # A worker pool created under the old limits must not keep serving jobs.
    reset_executor()
    return cleaned


def snapshot():
    """Everything the settings UI needs: cores, effective values, defaults, budget."""
    cores = cpu_count()
    values = current()
    used = values["job_workers"] * values["blas_threads"] + values["job_threads"]
    return {
        "cores": cores,
        "values": values,
        "defaults": defaults(cores),
        "budget": {
            "used": used,
            "available": cores,
            "reserved_blas": values["job_workers"] * values["blas_threads"],
            "fold_workers": values["job_threads"],
            "summary": "%d jobs x %d BLAS thread(s) + %d fold workers = %d of %d cores"
            % (
                values["job_workers"],
                values["blas_threads"],
                values["job_threads"],
                used,
                cores,
            ),
            "dask_summary": "%d Dask worker(s) x %d thread(s) = %d of %d cores"
            % (
                values["dask_workers"],
                values["dask_threads"],
                values["dask_workers"] * values["dask_threads"],
                cores,
            ),
        },
    }


def limits():
    """The per-job subset recorded on each job record for reproducibility."""
    values = current()
    return {
        "job_threads": values["job_threads"],
        "blas_threads": values["blas_threads"],
        "dask_workers": values["dask_workers"],
        "dask_threads": values["dask_threads"],
    }


def apply_limits(values=None):
    """Pin thread libraries to ``blas_threads`` (environment + in-process BLAS).

    The environment half is what reaches loky workers; the threadpoolctl half
    covers BLAS work that runs on the job thread itself (e.g. the final
    ``pipeline.fit``). Both are process-wide and idempotent.
    """
    values = values or current()
    width = int(values["blas_threads"])
    applied = {}
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(width)
        applied[name] = str(width)
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(limits=width)
        applied["threadpoolctl"] = width
    except Exception as exc:  # pragma: no cover - depends on optional BLAS libs
        logger.debug("Could not apply threadpoolctl limit: %s", exc)
    return applied


def job_executor():
    """The job thread pool, sized to ``job_workers`` and rebuilt when it changes."""
    global _executor, _executor_workers
    workers = int(current()["job_workers"])
    with _lock:
        if _executor is None or _executor_workers != workers:
            previous = _executor
            _executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sksurrogate-job")
            _executor_workers = workers
            if previous is not None:
                # Queued work still runs; the pool simply stops accepting more.
                previous.shutdown(wait=False)
        return _executor


def reset_executor():
    """Drop the cached executor (used by :func:`save` and by tests)."""
    global _executor, _executor_workers
    with _lock:
        previous = _executor
        _executor = None
        _executor_workers = None
        if previous is not None:
            previous.shutdown(wait=False)


def dask_cluster():
    """A local Dask cluster sized by the settings, or ``None`` when unavailable.

    Constructing ``Client()`` with no arguments would build a cluster with one
    worker per core and ignore the budget; the cluster is therefore always
    created with explicit ``n_workers``/``threads_per_worker``.
    """
    try:
        from dask.distributed import LocalCluster
    except ImportError:  # optional dependency
        return None
    values = current()
    return LocalCluster(
        n_workers=int(values["dask_workers"]),
        threads_per_worker=int(values["dask_threads"]),
        processes=True,
        # A fixed dashboard port collides when anything else already listens on
        # 8787 (and the dashboard is noise for a job-scoped cluster); ":0" asks
        # the OS for a free port instead.
        dashboard_address=":0",
        silence_logs=logging.ERROR,
    )
