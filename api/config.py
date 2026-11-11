"""Storage layout configuration for the SKSurrogate API.

Everything is filesystem + SQLite based (see docs/ui-plan.md, section 9):
auth is opt-in via environment variables (documented below), and no external
database or object storage is required for v1.
"""

import os
from pathlib import Path

from .safety import safe_segment


def _flag(value):
    """Parse an on/off environment variable (``1``/``true``/``yes``)."""
    return (value or "").strip().lower() in {"1", "true", "yes"}


def _int_env(name, default):
    """Read a positive integer environment variable, falling back on bad input."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


class Settings:
    """Resolves the on-disk layout under a single configurable root directory.

    Security knobs (all optional — v1 defaults keep the API open for local use):

    - ``SKSURROGATE_API_KEY``: when set, every endpoint except ``/api/health``
      requires this key (sent as an ``X-API-Key`` header, a Bearer token, or —
      on WebSocket routes only — an ``api_key`` query parameter). It resolves
      to an admin-equivalent **break-glass** principal
      (``api.auth.STATIC_KEY``) and needs no user account.
    - ``SKSURROGATE_REQUIRE_AUTH``: ``1``/``true``/``yes`` makes the API refuse
      to start unless at least one credential exists — the shared key, an
      existing account, or bootstrap credentials below. Also switches the
      middleware from "reject a wrong key" to "require an identity": browser
      sessions, per-user API keys and the shared key are all accepted, and
      ``/api/auth/status``/``login`` are the only anonymous routes. Read in
      ``api.main`` at import time (``Settings.require_auth``) and re-read there
      at call time for the fail-fast check.
    - ``SKSURROGATE_ADMIN_USERNAME`` / ``SKSURROGATE_ADMIN_PASSWORD``: create
      that administrator on first start when no such user exists (idempotent;
      ignored once the account exists). Delete the password variable after the
      first successful start.
    - ``SKSURROGATE_API_CORS_ORIGINS``: comma-separated list of allowed CORS
      origins. When unset, only local dev origins (localhost / 127.0.0.1 on any
      port) are allowed instead of the old wildcard.
    - ``SKSURROGATE_SESSION_TTL``: lifetime in seconds of a browser session
      (default ``604800``, i.e. 7 days).
    - ``SKSURROGATE_SESSION_COOKIE_SECURE``: ``1``/``true``/``yes`` adds the
      ``Secure`` flag to the session cookie. Set it whenever the UI is served
      over HTTPS.
    """

    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("SKSURROGATE_API_HOME", "./var/sksurrogate-api")).resolve()
        self.api_key = os.environ.get("SKSURROGATE_API_KEY") or None
        self.require_auth = _flag(os.environ.get("SKSURROGATE_REQUIRE_AUTH"))
        self.session_ttl_seconds = _int_env("SKSURROGATE_SESSION_TTL", 7 * 24 * 3600)
        self.session_cookie_secure = _flag(os.environ.get("SKSURROGATE_SESSION_COOKIE_SECURE"))
        self.cors_origins = [o.strip() for o in os.environ.get("SKSURROGATE_API_CORS_ORIGINS", "").split(",") if o.strip()]

    @property
    def auth_db_path(self):
        """SQLite database holding users, sessions and API keys."""
        return self.root / "auth" / "auth.db"

    @property
    def acl_db_path(self):
        """SQLite database holding ownership and sharing grants (see ``api.acl``)."""
        return self.root / "auth" / "acl.db"

    @property
    def auth_enforced(self):
        """Whether callers must identify themselves (see the class docstring)."""
        return self.require_auth or self.api_key is not None

    @property
    def datasets_dir(self):
        return self.root / "datasets"

    @property
    def mltrace_dir(self):
        return self.root / "mltrace"

    @property
    def bundles_dir(self):
        return self.root / "bundles"

    @property
    def registry_dir(self):
        return self.root / "registry"

    @property
    def predictions_dir(self):
        return self.root / "predictions"

    @property
    def monitoring_dir(self):
        return self.root / "monitoring"

    @property
    def jobs_dir(self):
        return self.root / "jobs"

    @property
    def logs_dir(self):
        return self.root / "logs"

    @property
    def checkpoints_dir(self):
        return self.root / "checkpoints"

    @property
    def projects_dir(self):
        return self.root / "projects"

    def ensure(self):
        for directory in (
            self.datasets_dir,
            self.mltrace_dir,
            self.bundles_dir,
            self.registry_dir,
            self.predictions_dir,
            self.monitoring_dir,
            self.jobs_dir,
            self.logs_dir,
            self.checkpoints_dir,
            self.projects_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def task_dataset_dir(self, task_name):
        return self.datasets_dir / safe_segment(task_name, "task name")

    def mltrace_db_path(self, task_name):
        return self.mltrace_dir / (safe_segment(task_name, "task name") + ".db")

    def task_bundles_dir(self, task_name):
        return self.bundles_dir / safe_segment(task_name, "task name")

    def task_predictions_dir(self, task_name):
        return self.predictions_dir / safe_segment(task_name, "task name")

    def monitor_log_path(self, task_name, model_version):
        return (
            self.monitoring_dir
            / safe_segment(task_name, "task name")
            / (safe_segment(model_version, "model version") + ".json")
        )

    def monitor_alerts_path(self, task_name):
        return self.monitoring_dir / safe_segment(task_name, "task name") / "alerts.jsonl"

    def job_record_path(self, job_id):
        return self.jobs_dir / (safe_segment(job_id, "job id") + ".json")

    def task_checkpoint_dir(self, task_name, job_id):
        return (
            self.checkpoints_dir
            / safe_segment(task_name, "task name")
            / safe_segment(job_id, "job id")
        )


settings = Settings().ensure()
