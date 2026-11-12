"""Per-user data reset: erase everything the caller owns.

The control plane keys storage by *task* (datasets, bundles, registry versions,
predictions, monitoring records, checkpoints, jobs and their logs) and by
*project*. There is no per-row ownership inside those trees, so a user-level
reset is a scoped filesystem operation:

* every task the caller **owns** (an explicit ``owner`` grant) is removed with
  all of its derivative artifacts;
* projects the caller owns are removed;
* resources shared with other users are skipped unless ``include_shared`` is set,
  because deleting them would destroy somebody else's work;
* unowned legacy material is never touched by an authenticated reset — an
  administrator can adopt it first (see ``/api/access/adopt-orphans``);
* in a no-auth (single anonymous user) deployment the caller owns everything, so
  the reset wipes the whole local workspace.

The account, sessions and API keys are deliberately kept: this is a data reset,
not an account deletion. One ``account.reset`` audit event records the counts.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from SKSurrogate import ModelRegistry

from . import acl, audit
from .config import settings
from .jobs import job_manager
from .logs import store as log_store


class LiveJobsError(RuntimeError):
    """Raised when a scoped task still has a running job (HTTP 409)."""

    def __init__(self, jobs):
        self.jobs = jobs
        super().__init__(
            "Stop the running jobs before resetting: %s"
            % ", ".join("%s (%s)" % (job["job_id"], job["task_name"]) for job in jobs)
        )


def _task_names() -> set[str]:
    """Every task name known to the storage layout (mirrors ``GET /api/tasks``)."""
    names: set[str] = set()
    for directory in (
        settings.datasets_dir,
        settings.bundles_dir,
        settings.registry_dir,
        settings.monitoring_dir,
        settings.checkpoints_dir,
        settings.predictions_dir,
    ):
        if directory.is_dir():
            names.update(path.name for path in directory.iterdir() if path.is_dir())
    if settings.mltrace_dir.is_dir():
        names.update(path.stem for path in settings.mltrace_dir.glob("*.db"))
    if settings.jobs_dir.is_dir():
        for path in settings.jobs_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            task_name = record.get("task_name") if isinstance(record, dict) else None
            if isinstance(task_name, str) and task_name:
                names.add(task_name)
    return names


def _project_ids() -> set[str]:
    if not settings.projects_dir.is_dir():
        return set()
    return {path.stem for path in settings.projects_dir.glob("*.json") if path.is_file()}


def _scope(kind: str, resource_id: str, principal, include_shared: bool):
    """Return ``(owned, shared)`` for one resource under this caller.

    ``owned`` means "delete it"; ``shared`` means "skip it because other users
    have access" (used to report *why* something was left behind).
    """
    if not settings.auth_enforced:
        return True, False
    if principal.user_id is None:
        return False, False
    role = acl.store().role_of(kind, resource_id, principal.user_id)
    if role != "owner":
        return False, False
    shared = any(
        grant["user_id"] != principal.user_id
        for grant in acl.store().list_grants(kind, resource_id)
    )
    if shared and not include_shared:
        return False, True
    return True, shared


def _remove(path: Path):
    """Delete a file or directory tree; return ``(file_count, byte_count)``."""
    if not path.exists():
        return 0, 0
    if path.is_dir():
        files = 0
        size = 0
        for entry in path.rglob("*"):
            if entry.is_file():
                files += 1
                try:
                    size += entry.stat().st_size
                except OSError:
                    pass
        shutil.rmtree(path, ignore_errors=True)
        return files, size
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    path.unlink(missing_ok=True)
    return 1, size


def _task_paths(task_name: str):
    """Every stored artifact that belongs to one task."""
    mltrace_path = settings.mltrace_db_path(task_name)
    return [
        settings.task_dataset_dir(task_name),
        settings.task_bundles_dir(task_name),
        settings.task_predictions_dir(task_name),
        settings.monitoring_dir / task_name,
        settings.checkpoints_dir / task_name,
        mltrace_path,
        Path(str(mltrace_path) + "-wal"),
        Path(str(mltrace_path) + "-shm"),
    ]


def reset_user_data(principal, *, include_shared: bool = False) -> dict:
    """Delete the caller's tasks/projects and all derivative artifacts.

    Raises :class:`LiveJobsError` before deleting anything when a scoped task
    still has a live job. Returns counts and the resources that were skipped
    because they are shared with other users.
    """
    if settings.auth_enforced and principal.user_id is None:
        raise PermissionError("A per-user reset requires a signed-in account")

    tasks: list[str] = []
    projects: list[str] = []
    skipped = {"tasks": [], "projects": []}

    for task_name in sorted(_task_names()):
        owned, shared = _scope(acl.TASK, task_name, principal, include_shared)
        if owned:
            tasks.append(task_name)
        elif shared:
            skipped["tasks"].append(task_name)

    for project_id in sorted(_project_ids()):
        owned, shared = _scope(acl.PROJECT, project_id, principal, include_shared)
        if owned:
            projects.append(project_id)
        elif shared:
            skipped["projects"].append(project_id)

    live = []
    for task_name in tasks:
        for record in job_manager.list(task_name):
            if job_manager.is_live(record.get("job_id")):
                live.append(
                    {"job_id": record.get("job_id"), "task_name": task_name}
                )
    if live:
        raise LiveJobsError(live)

    files_deleted = 0
    bytes_freed = 0
    jobs_deleted = 0
    log_files = 0

    for task_name in tasks:
        for record in job_manager.list(task_name):
            job_id = record.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                continue
            log_path = settings.logs_dir / (job_id + ".jsonl")
            if log_path.exists():
                log_files += 1
            log_store.clear(job_id)
            log_path.unlink(missing_ok=True)
            job_path = settings.job_record_path(job_id)
            removed, size = _remove(job_path)
            files_deleted += removed
            bytes_freed += size
            jobs_deleted += 1

        acl.store().remove_resource(acl.TASK, task_name)
        ModelRegistry(settings.registry_dir).remove_task(task_name)
        for path in _task_paths(task_name):
            removed, size = _remove(path)
            files_deleted += removed
            bytes_freed += size

    for project_id in projects:
        acl.store().remove_resource(acl.PROJECT, project_id)
        removed, size = _remove(settings.projects_dir / (project_id + ".json"))
        files_deleted += removed
        bytes_freed += size

    result = {
        "tasks": tasks,
        "projects": projects,
        "jobs": jobs_deleted,
        "log_files": log_files,
        "files": files_deleted,
        "bytes": bytes_freed,
        "skipped_shared": skipped,
        "open_mode": not settings.auth_enforced,
    }
    audit.record(
        "account.reset",
        principal=principal,
        tasks=len(tasks),
        projects=len(projects),
        jobs=jobs_deleted,
        files=files_deleted,
        skipped_shared=len(skipped["tasks"]) + len(skipped["projects"]),
    )
    return result


__all__ = ["LiveJobsError", "reset_user_data"]
