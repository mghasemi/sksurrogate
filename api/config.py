"""Storage layout configuration for the SKSurrogate API.

Everything is filesystem + SQLite based (see docs/ui-plan.md, section 9):
auth is opt-in via environment variables (documented below), and no external
database or object storage is required for v1.
"""

import os
from pathlib import Path


class Settings:
    """Resolves the on-disk layout under a single configurable root directory.

    Security knobs (all optional — v1 defaults keep the API open for local use):

    - ``SKSURROGATE_API_KEY``: when set, every endpoint except ``/api/health``
      requires this key (sent as an ``X-API-Key`` header, a Bearer token, or —
      on WebSocket routes only — an ``api_key`` query parameter).
    - ``SKSURROGATE_API_CORS_ORIGINS``: comma-separated list of allowed CORS
      origins. When unset, only local dev origins (localhost / 127.0.0.1 on any
      port) are allowed instead of the old wildcard.
    """

    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("SKSURROGATE_API_HOME", "./var/sksurrogate-api")).resolve()
        self.api_key = os.environ.get("SKSURROGATE_API_KEY") or None
        self.cors_origins = [o.strip() for o in os.environ.get("SKSURROGATE_API_CORS_ORIGINS", "").split(",") if o.strip()]

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
    def checkpoints_dir(self):
        return self.root / "checkpoints"

    def ensure(self):
        for directory in (
            self.datasets_dir,
            self.mltrace_dir,
            self.bundles_dir,
            self.registry_dir,
            self.predictions_dir,
            self.monitoring_dir,
            self.jobs_dir,
            self.checkpoints_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def task_dataset_dir(self, task_name):
        return self.datasets_dir / task_name

    def mltrace_db_path(self, task_name):
        return self.mltrace_dir / (task_name + ".db")

    def task_bundles_dir(self, task_name):
        return self.bundles_dir / task_name

    def task_predictions_dir(self, task_name):
        return self.predictions_dir / task_name

    def monitor_log_path(self, task_name, model_version):
        return self.monitoring_dir / task_name / (model_version + ".json")

    def monitor_alerts_path(self, task_name):
        return self.monitoring_dir / task_name / "alerts.jsonl"

    def job_record_path(self, job_id):
        return self.jobs_dir / (job_id + ".json")

    def task_checkpoint_dir(self, task_name, job_id):
        return self.checkpoints_dir / task_name / job_id


settings = Settings().ensure()
