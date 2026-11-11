"""Validation for caller-supplied names used as filesystem path segments.

Request data (task names, model versions, partitions, request IDs) is routinely
joined onto a storage root. Untrusted values such as ``../../etc/passwd`` would
escape that root and let a keyed caller read arbitrary ``*.csv`` files or load
arbitrary ``*.bundle`` files through ``joblib``. Every such join goes through
:func:`safe_segment` or :func:`contained_path`; both raise :class:`UnsafeName`,
which the API maps to HTTP 400.
"""

from pathlib import Path

_SEPARATORS = ("/", "\\", "\x00")


class UnsafeName(ValueError):
    """Raised when a caller-supplied name cannot be used as a path segment."""


def safe_segment(value, field="name"):
    """Return ``value`` if it is a single, non-traversing path segment.

    Accepts any character except path separators and NUL, so legitimately odd
    names (spaces, unicode, dots inside a name) keep working while ``.``,
    ``..`` and separator-bearing values are rejected.
    """
    if not isinstance(value, str) or not value.strip():
        raise UnsafeName("%s must be a non-empty string" % field)
    if any(character in value for character in _SEPARATORS):
        raise UnsafeName("%s must not contain path separators" % field)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise UnsafeName("%s must not contain control characters" % field)
    if value in (".", "..") or value.startswith(".."):
        raise UnsafeName("%s must not be a relative path component" % field)
    return value


def contained_path(root, segment, suffix="", field="path segment"):
    """Join ``segment`` under ``root`` and prove the result cannot escape it.

    Defense in depth on top of :func:`safe_segment`: even if a segment slips
    validation, the resolved target must remain inside ``root``. The returned
    path is un-resolved so callers keep their existing path arithmetic and
    error messages.
    """
    segment = safe_segment(segment, field)
    root_path = Path(root)
    candidate = root_path / (segment + suffix)
    root_resolved = root_path.resolve()
    resolved = candidate.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise UnsafeName("path segment escapes its root: %r" % segment)
    return candidate
