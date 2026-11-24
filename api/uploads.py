"""Bounded upload reading for the control plane.

Every upload endpoint used to call ``await file.read()``, which materialises the
whole body in memory before any validation. A client (or a misconfigured
client) could therefore exhaust the process with one large file, and the
resulting failure was an OOM kill rather than a clean response.

:func:`read_upload` streams the spooled file in chunks and stops as soon as the
configured cap is exceeded, raising 413 with the limit and the actual byte count.
The cap is ``Settings.max_upload_mb`` (``SKSURROGATE_MAX_UPLOAD_MB``, default
256 MB) and is read at call time so a test or a deployment can change it without
reimporting the module.
"""

from __future__ import annotations

from fastapi import HTTPException

from .config import settings

#: One chunk is 1 MiB: enough to keep the read loop cheap, small enough that an
#: oversized body is rejected after a bounded amount of work.
CHUNK_SIZE = 1 << 20


def max_upload_bytes() -> int:
    """The configured cap in bytes."""
    return int(settings.max_upload_mb) * 1024 * 1024


async def read_upload(file, *, label: str = "upload") -> bytes:
    """Read an ``UploadFile`` completely, refusing a body above the cap.

    :raises HTTPException: 413 when the accumulated size exceeds
        ``settings.max_upload_mb``. The message names the limit, the label
        (e.g. "dataset CSV") and the environment variable, so the caller can fix
        the deployment rather than guess.
    """
    limit = max_upload_bytes()
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=(
                    "The %s exceeds the %d MB upload limit (read at least %d bytes). "
                    "Raise SKSURROGATE_MAX_UPLOAD_MB on the API process or split the "
                    "file." % (label, settings.max_upload_mb, total)
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)
