"""Partition defaulting: a request that names no partition should use the task's data.

Several request models used to default to the literal partition name ``"train"``.
A task whose data lives in another partition — a single ``main`` partition, or one
named like the task — then answered ``404 No dataset partition 'train' stored`` for
a request that never asked for a partition at all. (The same bug hit synthetic
sampling and temporal declarations.)

The rule here: an **explicit** name always wins and its absence is an error; an
**omitted** name resolves to ``train`` when it exists, otherwise to the task's only
(or first) registered partition. Nothing is guessed silently: the resolved name is
returned to the caller, which echoes it in its response.
"""

from __future__ import annotations

from .config import settings
from .deps import open_tracker


def available_partitions(task_name: str) -> list[str]:
    """Sorted names of the task's stored partitions (``[]`` when unknown).

    Guarded on the tracking database's existence because ``open_tracker`` would
    create an empty task database as a side effect, which then turns "no dataset
    registered" into a confusing "partition not found".
    """
    try:
        if not settings.mltrace_db_path(task_name).exists():
            return []
    except OSError:
        return []
    try:
        with open_tracker(task_name) as tracker:
            return sorted(tracker.dataset_splits() or {})
    except Exception:  # pragma: no cover - a missing task has no partitions
        return []


def resolve_partition(task_name: str, partition: str | None, prefer: str = "train") -> str:
    """The partition to read for a request that may not have named one."""
    if partition:
        return partition
    available = available_partitions(task_name)
    if not available:
        # No registered data: keep the historical name so the caller's 404 message
        # ("No dataset partition 'train' stored") stays recognisable.
        return prefer
    return prefer if prefer in available else available[0]


def resolve_optional_partition(
    task_name: str, partition: str | None, prefer: str = "validation"
) -> str | None:
    """A partition that may legitimately not exist (e.g. ``validation``).

    Returns ``None`` when the caller named nothing and nothing plausible is
    registered; an explicit name is passed through so a typo still errors.
    """
    if partition:
        return partition
    available = available_partitions(task_name)
    if prefer in available:
        return prefer
    return None


__all__ = [
    "available_partitions",
    "resolve_optional_partition",
    "resolve_partition",
]
