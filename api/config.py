"""Storage layout configuration for the SKSurrogate API.

Everything is filesystem + SQLite based (see docs/ui-plan.md, section 9):
no auth, no external database, no object storage required for v1.
"""

import os
from pathlib import Path


class Settings:
    """Resolves the on-disk layout under a single configurable root directory."""

    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("SKSURROGATE_API_HOME", "./var/sksurrogate-api")).resolve()

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

    def job_record_path(self, job_id):
        return self.jobs_dir / (job_id + ".json")

    def task_checkpoint_dir(self, task_name, job_id):
        return self.checkpoints_dir / task_name / job_id


settings = Settings().ensure()
