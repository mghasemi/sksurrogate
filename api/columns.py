"""Dataset-level column exclusions ("ignored columns").

A user can mark feature columns of a registered dataset as *ignored* from the
Datasets page. The choice is stored in the task's tracking metadata
(``ignored_columns``) instead of rewriting the stored CSV, so it is reversible and
existing data fingerprints stay valid.

Every analysis path that builds a feature matrix from a stored partition must
drop these columns, otherwise an ignored column silently reappears in an
experiment, sensitivity analysis or profile. Read through
:func:`read_ignored_columns` and apply :func:`drop_ignored` at the point where a
partition is loaded; the target column can never be ignored.
"""

from .config import settings
from .deps import open_tracker


def normalize_ignored_columns(columns, known_columns, target):
    """Validate and normalize a caller-supplied ignore list.

    Duplicates are collapsed, unknown columns are rejected (a typo must not
    silently do nothing), and the target column is refused. Raises ``ValueError``
    with a user-facing message; routers map that to HTTP 400.
    """
    if columns is None:
        return []
    if not isinstance(columns, (list, tuple)):
        raise ValueError("ignored_columns must be a list of column names")
    known = {str(column) for column in (known_columns if known_columns is not None else [])}
    normalized = []
    for column in columns:
        if not isinstance(column, str) or not column.strip():
            raise ValueError("ignored_columns must contain non-empty column names")
        name = column.strip()
        if name == target:
            raise ValueError("The target column %r cannot be ignored" % name)
        if name not in known:
            raise ValueError("Unknown column %r in ignored_columns" % name)
        if name not in normalized:
            normalized.append(name)
    return normalized


def read_ignored_columns(task_name):
    """The task's ignored columns, or ``[]`` when nothing is registered/ignored.

    This is called on every partition load, so a missing tracking database or an
    unreadable metadata row degrades to "nothing ignored" rather than failing an
    analysis for a task that has no metadata yet.
    """
    try:
        if not settings.mltrace_db_path(task_name).exists():
            return []
    except OSError:
        return []
    try:
        with open_tracker(task_name) as tracker:
            metadata = tracker.GetMetadata()
    except Exception:  # pragma: no cover - defensive: metadata is auxiliary here
        return []
    values = metadata.get("ignored_columns") if isinstance(metadata, dict) else None
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, str) and value]


def drop_ignored(frame, ignored):
    """Return ``frame`` without the ignored columns that are actually present."""
    present = [column for column in ignored if column in frame.columns]
    if not present:
        return frame
    return frame.drop(columns=present)


__all__ = [
    "drop_ignored",
    "normalize_ignored_columns",
    "read_ignored_columns",
]
