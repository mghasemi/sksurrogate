"""Persistent audit trail for authentication and authorization events.

The per-job log store (:mod:`api.logs`) is scoped to background jobs, so a login
attempt or a key revocation has nowhere to live there. This module appends one
JSON line per identity event to ``<API home>/logs/auth.jsonl`` — append-only,
never rewritten, and never containing a password, a session token or a full API
key (only the key prefix, which the API already shows in listings).

Why file-backed rather than a table: the events must survive a restart even if
the auth database is replaced, and an operator can ``tail -f`` or ship the file
without opening SQLite. Reads are bounded (last N lines), so the endpoint stays
cheap as the file grows.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

from .config import settings

LOGGER = logging.getLogger("sksurrogate.api.auth")

#: Hard cap on how many events one read returns.
MAX_READ = 500

_LOCK = threading.Lock()


def _path():
    return settings.root / "logs" / "auth.jsonl"


def record(event: str, *, outcome: str = "ok", principal=None, username: Optional[str] = None, **fields) -> dict:
    """Append one audit event.

    ``fields`` must contain only non-secret metadata (key prefixes, roles,
    resource ids, client addresses). Never pass a password, a session token or
    a full API key — the helper cannot detect them, so it is a call-site rule
    that :func:`record` enforces only by convention and by the tests.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": str(event),
        "outcome": str(outcome),
    }
    if principal is not None:
        user_id = getattr(principal, "user_id", None)
        if user_id is not None:
            entry["user_id"] = user_id
        entry["username"] = str(getattr(principal, "username", ""))
        entry["source"] = str(getattr(principal, "source", ""))
    elif username:
        entry["username"] = str(username)
    for key, value in fields.items():
        if value is not None:
            entry[key] = value

    path = _path()
    try:
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            # "a" is atomic enough for line-sized appends on a local filesystem;
            # the lock serialises threads inside this process.
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, sort_keys=True) + "\n")
    except OSError as exc:  # an unwritable audit log must not break the request
        LOGGER.warning("Could not write the auth audit log: %s", exc)
    return entry


def read(limit: int = 200) -> list[dict]:
    """Return the most recent audit events, newest first (bounded)."""
    limit = max(1, min(int(limit or 200), MAX_READ))
    path = _path()
    if not path.is_file():
        return []
    tail: list[str] = []
    try:
        with path.open("rb") as stream:
            # Read from the end in blocks so a multi-megabyte file stays cheap.
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            block = 4096
            position = size
            buffer = b""
            while position > 0 and buffer.count(b"\n") <= limit:
                step = min(block, position)
                position -= step
                stream.seek(position)
                buffer = stream.read(step) + buffer
            tail = buffer.decode("utf-8", "replace").splitlines()
    except OSError:
        return []
    entries = []
    for line in tail[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue
    entries.reverse()
    return entries


def clear() -> None:
    """Delete the audit file (tests only)."""
    try:
        _path().unlink()
    except FileNotFoundError:
        pass
