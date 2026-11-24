"""Token-bucket rate limiting for the expensive control-plane surfaces.

The login form already has its own failed-attempt throttle (``api.auth``). This
module covers the two other classes of request that can cost real work:

* **job submissions** — every ``POST`` that queues an experiment, ensemble,
  nested CV, backtest, report, retraining run and so on;
* **inference** — ``/api/inference/...`` and ``/api/serving/...`` predictions.

Why a token bucket: it bounds the *rate* while allowing a short burst (a user
submitting three jobs in a row is normal), and it degrades to a clean 429 with
``Retry-After`` instead of queueing work the process cannot finish.

Deployment limits, stated plainly: buckets live in this process's memory. A
multi-worker uvicorn deployment therefore gets per-worker limits, and a restart
forgets them. That is adequate for the single-process control plane this project
ships; a public deployment should put a real limiter (reverse proxy or gateway)
in front. ``SKSURROGATE_RATE_LIMIT=1`` enables the middleware;
``SKSURROGATE_RATE_LIMIT_JOBS_PER_MINUTE`` and
``SKSURROGATE_RATE_LIMIT_INFERENCE_PER_MINUTE`` set the rates (default 60 and
300 per client).
"""

from __future__ import annotations

import json
import math
import re
import threading
import time

from .config import settings

#: Paths that queue background work, keyed by a stable class name.
_JOB_PATTERNS = (
    re.compile(r"^/api/experiments/[^/]+/run/?$"),
    re.compile(r"^/api/ensembles/[^/]+/fit/?$"),
    re.compile(r"^/api/evaluation/[^/]+/(nested-cv|backtest)/?$"),
    re.compile(r"^/api/bundles/[^/]+/train-baseline/?$"),
    re.compile(r"^/api/reports/[^/]+/explain/?$"),
    re.compile(r"^/api/(sensitivity|synthdata|retraining)/.*$"),
)

#: Prediction paths (they load a bundle and score rows synchronously).
_INFERENCE_PATTERNS = (
    re.compile(r"^/api/inference/.*$"),
    # Declaring or stopping a serving pin is cheap; only prediction is limited.
    re.compile(r"^/api/serving/[^/]+/(predict|forecast)/?$"),
)


def classify(method: str, path: str) -> str | None:
    """Which bucket a request belongs to (``"jobs"``, ``"inference"`` or ``None``).

    Only mutating/predicting methods are limited: polling a job's status or
    reading a bundle must never be throttled behind the work it observes.
    """
    method = (method or "").upper()
    path = path or ""
    if method == "POST":
        for pattern in _JOB_PATTERNS:
            if pattern.match(path):
                return "jobs"
    if method in ("POST", "PUT", "PATCH"):
        for pattern in _INFERENCE_PATTERNS:
            if pattern.match(path):
                return "inference"
    return None


def limit_for(bucket: str) -> int:
    """Requests per minute for a bucket; ``0`` disables that bucket."""
    if bucket == "jobs":
        return int(settings.rate_limit_jobs_per_minute)
    if bucket == "inference":
        return int(settings.rate_limit_inference_per_minute)
    return 0


class TokenBucket:
    """A single refilling bucket of ``capacity`` tokens at ``per_minute``.

    ``take`` never blocks: it returns ``(allowed, retry_after_seconds)``.
    ``now`` is injectable so the refill arithmetic is testable without sleeping.
    """

    def __init__(self, per_minute: int, *, burst: int | None = None, now: float | None = None):
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self.rate = float(per_minute) / 60.0
        self.capacity = float(burst if burst is not None else max(1, per_minute))
        self.tokens = self.capacity
        self.updated = time.monotonic() if now is None else float(now)
        self._lock = threading.Lock()

    def take(self, now: float | None = None) -> tuple[bool, float]:
        with self._lock:
            current = time.monotonic() if now is None else float(now)
            elapsed = max(0.0, current - self.updated)
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.updated = current
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True, 0.0
            return False, (1.0 - self.tokens) / self.rate


class RateLimiter:
    """Per-(bucket, client) token buckets for one process."""

    def __init__(self):
        self._buckets: dict[tuple[str, str], TokenBucket] = {}
        self._lock = threading.Lock()

    def check(self, bucket: str, client: str, per_minute: int) -> tuple[bool, float]:
        if per_minute <= 0:
            return True, 0.0
        key = (bucket, client or "unknown")
        with self._lock:
            token_bucket = self._buckets.get(key)
            # Rebuild when the configured rate changes (tests, a live config edit).
            if token_bucket is None or abs(token_bucket.rate - per_minute / 60.0) > 1e-9:
                token_bucket = TokenBucket(per_minute)
                self._buckets[key] = token_bucket
        return token_bucket.take()

    def reset(self) -> None:
        """Forget every bucket (tests and a settings change)."""
        with self._lock:
            self._buckets.clear()


LIMITER = RateLimiter()


class RateLimitMiddleware:
    """Pure ASGI middleware: 429 with ``Retry-After`` when a bucket is empty.

    Runs before routing, so a limited request is refused without touching the
    queue, the filesystem or the model. Reads settings at call time so a test or
    a deployment can change the policy without reimporting the app.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not settings.rate_limit_enabled:
            return await self.app(scope, receive, send)
        bucket = classify(scope.get("method", ""), scope.get("path", ""))
        if bucket is None:
            return await self.app(scope, receive, send)
        client = scope.get("client") or ("unknown", 0)
        allowed, retry_after = LIMITER.check(bucket, client[0], limit_for(bucket))
        if allowed:
            return await self.app(scope, receive, send)
        payload = json.dumps(
            {
                "detail": (
                    "Too many %s requests from this client; retry in %.0f second(s). "
                    "The limits are SKSURROGATE_RATE_LIMIT_JOBS_PER_MINUTE and "
                    "SKSURROGATE_RATE_LIMIT_INFERENCE_PER_MINUTE on the API process."
                    % (bucket, max(1.0, retry_after))
                )
            }
        ).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"retry-after", str(max(1, int(math.ceil(retry_after)))).encode("ascii")),
        ]
        await send({"type": "http.response.start", "status": 429, "headers": headers})
        await send({"type": "http.response.body", "body": payload})
