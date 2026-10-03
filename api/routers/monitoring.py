"""Monitoring endpoints: drift, fairness, delayed-label performance, request stats.

Wraps ``SKSurrogate.monitoring``. ``InferenceMonitor`` is normally an
in-memory accumulator; here it is rehydrated from the JSON log persisted by
the inference router (see ``deps.append_monitor_record``) so summaries
survive process restarts, staying consistent with the SQLite/file-only
storage decision in docs/ui-plan.md.

Phase 2 additions (docs/ui-gap-implementation-plan.md): prediction-distribution
drift, subgroup performance / loss reports, and the sensitive-feature / PII
scan. Prediction sources resolve to stored partitions or persisted inference
artifacts (``predictions/{task}/{version}/{request_id}.csv``, written by
``POST /api/inference/.../predict-batch``); inline arrays are accepted for
ad-hoc checks.
"""

import json
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from SKSurrogate import (
    InferenceMonitor,
    delayed_label_performance,
    drift_report,
    fairness_report,
    prediction_distribution_report,
    sensitive_feature_report,
    subgroup_performance_report,
)

from ..config import settings
from ..deps import append_drift_alerts, bad_request, load_monitor_records, not_found, open_tracker

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


@router.get("/alerts")
def monitoring_alerts(task: str | None = None, limit: int = Query(default=20, ge=1, le=100)):
    """Return recent persisted drift alerts and inference failures."""
    if settings.monitoring_dir.is_dir():
        task_dirs = sorted(
            (path for path in settings.monitoring_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
    else:
        task_dirs = []
    if task is not None:
        task_dirs = [path for path in task_dirs if path.name == task]

    alerts = []
    for task_dir in task_dirs:
        alerts_path = settings.monitor_alerts_path(task_dir.name)
        if alerts_path.exists():
            with alerts_path.open(encoding="utf-8") as stream:
                alerts.extend(json.loads(line) for line in stream if line.strip())

        for monitor_path in task_dir.glob("*.json"):
            version = monitor_path.stem
            for record in load_monitor_records(task_dir.name, version):
                if record.get("success", True):
                    continue
                timestamp = record.get("timestamp")
                if timestamp is None:
                    timestamp = datetime.fromtimestamp(
                        monitor_path.stat().st_mtime, tz=timezone.utc
                    ).isoformat()
                alerts.append(
                    {
                        "timestamp": timestamp,
                        "task": task_dir.name,
                        "version": version,
                        "type": "inference_error",
                        "severity": "error",
                        "detail": "Inference request failed for %s row(s)." % record.get("rows", 0),
                    }
                )

    alerts.sort(key=lambda alert: alert["timestamp"], reverse=True)
    return {"alerts": alerts[:limit]}


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


class PredictionSource(BaseModel):
    """Where a prediction series comes from.

    Exactly one of ``partition`` / ``request_id`` / ``values`` must be set:

    - ``partition`` — a stored dataset partition; its target column is used as
      the reference predictions (the registered labels stand in for them);
    - ``request_id`` — a persisted inference artifact from
      ``POST /api/inference/{task}/predict-batch``;
    - ``values`` — an inline array for ad-hoc checks.

    An optional ``column`` names which column to read: for a partition it
    defaults to the registered target (useful when labels stand in for
    predictions), and for an artifact it defaults to ``prediction``. Set it to
    pull any other stored column, e.g. a demographic attribute used as groups.
    """

    partition: str | None = None
    request_id: str | None = None
    # ``float | str`` so categorical group labels (e.g. ["a", "b"]) survive;
    # numeric sources coerce to float downstream.
    values: list[float | str] | None = None
    column: str | None = None


def _resolve_prediction_series(task_name, model_version, source: PredictionSource, *, numeric=True):
    """Materialize one series as a 1-D pandas Series.

    ``numeric`` coerces the values to float (predictions / labels); pass
    ``False`` for group labels so categorical values survive intact.
    """
    set_fields = [name for name in ("partition", "request_id", "values") if getattr(source, name) is not None]
    if len(set_fields) != 1:
        raise bad_request("Exactly one of partition, request_id or values must be provided per prediction source")

    def coerce(series):
        return pd.to_numeric(series, errors="coerce") if numeric else series

    if source.values is not None:
        return coerce(pd.Series(source.values)).rename("prediction")

    if source.partition is not None:
        frame = _load_partition(task_name, source.partition)
        column = source.column or _task_target(task_name)
        if column is None or column not in frame.columns:
            raise bad_request(
                "Partition %r has no usable column (tried %r)" % (source.partition, column)
            )
        return coerce(frame[column]).rename("prediction")

    csv_path = settings.task_predictions_dir(task_name) / model_version / (source.request_id + ".csv")
    if not csv_path.exists():
        raise not_found(
            "No persisted prediction artifact %r for task %r, version %r"
            % (source.request_id, task_name, model_version)
        )
    frame = pd.read_csv(csv_path)
    column = source.column or "prediction"
    if column not in frame.columns:
        raise bad_request("Prediction artifact %r has no %r column" % (source.request_id, column))
    return coerce(frame[column]).rename("prediction")


def _task_target(task_name):
    """The task's registered target column name, or ``None`` when unknown."""
    db_path = settings.mltrace_db_path(task_name)
    if not db_path.exists():
        return None
    try:
        with open_tracker(task_name) as tracker:
            return tracker.GetMetadata().get("target_name")
    except Exception:
        return None


class PredictionDriftRequest(BaseModel):
    reference_source: PredictionSource
    current_source: PredictionSource
    threshold: float = Field(default=0.2, gt=0.0)
    bins: int = Field(default=10, ge=2, le=50)


@router.post("/{task_name}/{model_version}/prediction-drift")
def check_prediction_drift(task_name: str, model_version: str, body: PredictionDriftRequest):
    """Report distribution drift between two prediction series (PSI / TVD)."""
    reference = _resolve_prediction_series(task_name, model_version, body.reference_source)
    current = _resolve_prediction_series(task_name, model_version, body.current_source)
    if len(reference) < 2 or len(current) < 2:
        raise bad_request("Each prediction series needs at least two rows")
    report = prediction_distribution_report(
        reference.tolist(),
        current.tolist(),
        model_version=model_version,
        threshold=body.threshold,
        bins=body.bins,
    )
    append_drift_alerts(
        task_name,
        model_version,
        report["alerts"],
        thresholds={"psi": body.threshold, "categorical": body.threshold},
    )
    return report


class SubgroupPerformanceRequest(BaseModel):
    y_true_source: PredictionSource
    y_pred_source: PredictionSource
    groups_source: PredictionSource | None = None
    inline_groups: list[str] | list[int] | None = None
    metric: str = "accuracy"
    positive_label: int = 1


@router.post("/{task_name}/{model_version}/subgroup-performance")
def check_subgroup_performance(task_name: str, model_version: str, body: SubgroupPerformanceRequest):
    """Per-group metric (accuracy/precision/recall/f1/loss) plus the fairness gap.

    ``groups`` come from a stored partition column (``groups_source.partition``),
    a persisted artifact column, or an inline array of equal length to the other
    two series.
    """
    if body.metric not in {"accuracy", "precision", "recall", "f1", "loss"}:
        raise bad_request("metric must be one of accuracy, precision, recall, f1, loss")

    y_true = _resolve_prediction_series(task_name, model_version, body.y_true_source)
    y_pred = _resolve_prediction_series(task_name, model_version, body.y_pred_source)
    if len(y_true) != len(y_pred):
        raise bad_request("y_true and y_pred must contain the same number of rows")

    if body.inline_groups is not None:
        groups = pd.Series(body.inline_groups).rename("group")
        if len(groups) != len(y_true):
            raise bad_request("inline_groups must have the same length as y_true/y_pred")
    elif body.groups_source is not None:
        series = _resolve_prediction_series(task_name, model_version, body.groups_source, numeric=False)
        groups = series.rename("group")
        if len(groups) != len(y_true):
            raise bad_request("groups must have the same length as y_true/y_pred")
    else:
        raise bad_request("Provide either inline_groups or a groups_source")

    try:
        return subgroup_performance_report(
            y_true.tolist(),
            y_pred.tolist(),
            groups.tolist(),
            metric=body.metric,
            positive_label=body.positive_label,
        )
    except ValueError as exc:
        raise bad_request(str(exc))


class SensitiveScanRequest(BaseModel):
    partition: str | None = None
    columns: list[str] | None = None
    sensitive_features: list[str] | None = None


@router.post("/{task_name}/sensitive-scan")
def scan_sensitive(task_name: str, body: SensitiveScanRequest):
    """Flag PII-like and explicitly sensitive columns in a stored partition.

    ``columns`` (optional) restricts the scan to a subset of the partition's
    columns; without it every column is inspected. The scan only looks at
    column *names* — no cell values are read or returned.
    """
    if body.partition is None and not body.columns:
        raise bad_request("Provide either a partition name or an explicit column list")

    frame = pd.DataFrame(index=range(0))  # empty frame; only the columns matter
    if body.partition is not None:
        stored = _load_partition(task_name, body.partition)
        if body.columns:
            missing = [column for column in body.columns if column not in stored.columns]
            if missing:
                raise bad_request("Unknown columns in partition %r: %s" % (body.partition, ", ".join(missing)))
            frame = pd.DataFrame(index=range(0), columns=body.columns)
        else:
            frame = pd.DataFrame(index=range(0), columns=list(stored.columns))
    else:
        frame = pd.DataFrame(index=range(0), columns=list(body.columns or []))

    return sensitive_feature_report(frame, sensitive_features=body.sensitive_features)


@router.post("/{task_name}/drift")
def check_drift(task_name: str, body: DriftRequest):
    """Compare two stored dataset partitions for schema/statistical drift."""
    reference = _load_partition(task_name, body.reference_partition)
    current = _load_partition(task_name, body.current_partition)
    report = drift_report(
        reference,
        current,
        model_version=body.model_version,
        psi_threshold=body.psi_threshold,
        categorical_threshold=body.categorical_threshold,
        missingness_threshold=body.missingness_threshold,
        range_threshold=body.range_threshold,
    )
    append_drift_alerts(
        task_name,
        body.model_version,
        report["alerts"],
        thresholds={
            "psi": body.psi_threshold,
            "categorical": body.categorical_threshold,
            "missingness": body.missingness_threshold,
            "range": body.range_threshold,
        },
    )
    return report


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
