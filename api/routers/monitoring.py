"""Monitoring endpoints: drift, fairness, delayed-label performance, request stats.

Wraps ``SKSurrogate.monitoring``. ``InferenceMonitor`` is normally an
in-memory accumulator; here it is rehydrated from the JSON log persisted by
the inference router (see ``deps.append_monitor_record``) so summaries
survive process restarts, staying consistent with the SQLite/file-only
storage decision in docs/ui-plan.md.
"""

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

from SKSurrogate import (
    InferenceMonitor,
    delayed_label_performance,
    drift_report,
    fairness_report,
)

from ..config import settings
from ..deps import load_monitor_records, not_found

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


class DriftRequest(BaseModel):
    reference_partition: str
    current_partition: str
    model_version: str | None = None
    psi_threshold: float = 0.2
    categorical_threshold: float = 0.2
    missingness_threshold: float = 0.1
    range_threshold: float = 0.1


def _load_partition(task_name, partition):
    csv_path = settings.task_dataset_dir(task_name) / (partition + ".csv")
    if not csv_path.exists():
        raise not_found("No dataset partition %r stored for task %r" % (partition, task_name))
    return pd.read_csv(csv_path)


@router.post("/{task_name}/drift")
def check_drift(task_name: str, body: DriftRequest):
    """Compare two stored dataset partitions for schema/statistical drift."""
    reference = _load_partition(task_name, body.reference_partition)
    current = _load_partition(task_name, body.current_partition)
    return drift_report(
        reference,
        current,
        model_version=body.model_version,
        psi_threshold=body.psi_threshold,
        categorical_threshold=body.categorical_threshold,
        missingness_threshold=body.missingness_threshold,
        range_threshold=body.range_threshold,
    )


class FairnessRequest(BaseModel):
    y_true: list
    y_pred: list
    groups: list
    positive_label: int = 1


@router.post("/{task_name}/fairness")
def check_fairness(task_name: str, body: FairnessRequest):
    """Report demographic-parity and equal-opportunity gaps across groups."""
    return fairness_report(body.y_true, body.y_pred, body.groups, positive_label=body.positive_label)


class DelayedLabelRequest(BaseModel):
    predictions: list
    labels: list
    model_version: str | None = None
    task_type: str = "classification"


@router.post("/{task_name}/delayed-label-performance")
def check_delayed_label_performance(task_name: str, body: DelayedLabelRequest):
    """Score recorded predictions once ground-truth labels become available."""
    return delayed_label_performance(
        body.predictions, body.labels, model_version=body.model_version, task_type=body.task_type
    )


@router.get("/{task_name}/{model_version}/summary")
def monitor_summary(task_name: str, model_version: str):
    """Return aggregated latency/throughput/error-rate stats for a model version."""
    monitor = InferenceMonitor(model_version)
    for record in load_monitor_records(task_name, model_version):
        monitor.record(record["latency_ms"], record["rows"], success=record.get("success", True))
    return monitor.summary()
