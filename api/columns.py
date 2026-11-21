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


def read_deduced_types(task_name):
    """Per-column deduced types for the task (``{}`` when unavailable).

    Populated at registration from ``DataPreprocess`` plus any user type
    overrides, so it is the durable record of "the user says this column is a
    datetime". The CSV itself cannot carry the dtype.
    """
    try:
        if not settings.mltrace_db_path(task_name).exists():
            return {}
    except OSError:
        return {}
    try:
        with open_tracker(task_name) as tracker:
            metadata = tracker.GetMetadata()
    except Exception:  # pragma: no cover - defensive: metadata is auxiliary here
        return {}
    values = metadata.get("dataset_deduced_types") if isinstance(metadata, dict) else None
    if not isinstance(values, dict):
        return {}
    return {str(column): str(typ) for column, typ in values.items()}


def declared_datetime_columns(task_name):
    """Columns the dataset declares as ``datetime64`` (deduced or overridden)."""
    return [
        column
        for column, typ in read_deduced_types(task_name).items()
        if typ == "datetime64"
    ]


def coerce_declared_datetimes(frame, declared=None, task_name=None, strict=False):
    """Cast columns declared ``datetime64`` back to a datetime dtype.

    A datetime dtype does not survive either storage path — ``to_csv`` writes ISO
    text and SQLite has no date type — so a frame read back from disk is always
    ``object``. Without this cast, a column the user explicitly typed as
    ``datetime64`` would be one-hot encoded as if it were a category, one column per
    distinct timestamp.

    ``declared`` is the list of column names (from
    :func:`declared_datetime_columns`); pass ``task_name`` instead to read them from
    metadata. Timezone-aware values are normalised to UTC-naive, matching what the
    feature transformer and the temporal axis do.

    ``strict=True`` raises ``ValueError`` when a declared column parses to nothing —
    that is a registration-time check, not a load-time one. At load time the default
    is lenient: an unparseable column is left untouched (and reaches the model as a
    category, visibly wrong) rather than failing every analysis.
    """
    import pandas as pd

    if declared is None:
        declared = declared_datetime_columns(task_name) if task_name else []
    if not declared:
        return frame
    coerced = frame
    for column in declared:
        if column not in coerced.columns:
            continue
        series = coerced[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        if parsed.isna().all() and series.notna().any():
            if strict:
                offending = [
                    str(value) for value in series.dropna().head(3).tolist()
                ]
                raise ValueError(
                    "column %r is declared datetime64 but none of its values parse "
                    "as timestamps (first values: %s)"
                    % (column, ", ".join(offending))
                )
            continue
        if coerced is frame:
            coerced = frame.copy()
        coerced[column] = parsed.dt.tz_localize(None)
    return coerced


__all__ = [
    "coerce_declared_datetimes",
    "declared_datetime_columns",
    "drop_ignored",
    "normalize_ignored_columns",
    "read_deduced_types",
    "read_ignored_columns",
]
