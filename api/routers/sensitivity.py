"""Feature-selection/sensitivity-analysis endpoints, wrapping ``SKSurrogate.SensAprx``.

Runs as a background job (see ``api.jobs``) since Sobol/Morris analysis can
be slow on larger feature sets.
"""

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

from SKSurrogate import CorrelationThreshold, SensAprx

from ..config import settings
from ..deps import bad_request, not_found, open_tracker
from ..jobs import job_manager

router = APIRouter(prefix="/api/sensitivity", tags=["sensitivity"])


class SensitivityRequest(BaseModel):
    method: str = "sobol"
    n_features_to_select: int = 5
    train_partition: str = "train"


def _load_train_xy(task_name, train_partition):
    with open_tracker(task_name) as tracker:
        metadata = tracker.GetMetadata()
    target = metadata.get("target_name")
    if target is None:
        raise not_found("No dataset registered for task %r" % task_name)
    csv_path = settings.task_dataset_dir(task_name) / (train_partition + ".csv")
    if not csv_path.exists():
        raise not_found("No dataset partition %r stored for task %r" % (train_partition, task_name))
    frame = pd.read_csv(csv_path)
    feature_columns = [column for column in frame.columns if column != target]
    return feature_columns, frame[feature_columns].values, frame[target].values


@router.post("/{task_name}/run")
def run_sensitivity(task_name: str, body: SensitivityRequest):
    """Submit a sensitivity-analysis job and return its job id immediately (202-style)."""
    if body.method not in {"sobol", "morris", "delta-mmnt"}:
        raise bad_request("method must be one of sobol, morris, delta-mmnt")
    feature_columns, X, y = _load_train_xy(task_name, body.train_partition)

    def _task():
        analyzer = SensAprx(n_features_to_select=body.n_features_to_select, method=body.method)
        analyzer.fit(X, y)
        top_indices = [int(index) for index in analyzer.top_features_[: body.n_features_to_select]]
        return {
            "method": body.method,
            "n_features_to_select": body.n_features_to_select,
            "feature_columns": feature_columns,
            "top_feature_indices": top_indices,
            "top_feature_names": [feature_columns[index] for index in top_indices],
            "weights": [float(value) for value in analyzer.weights_],
        }

    job_id = job_manager.submit(task_name, "sensitivity", lambda job_id: _task)
    return {"job_id": job_id, "status": "queued"}


@router.post("/{task_name}/correlation-threshold")
def run_correlation_threshold(task_name: str, threshold: float = 0.7, train_partition: str = "train"):
    """Fast, synchronous correlation-based feature pruning (no job needed)."""
    feature_columns, X, _ = _load_train_xy(task_name, train_partition)
    selector = CorrelationThreshold(threshold=threshold)
    selector.fit(X)
    dropped_indices = sorted(set(range(len(feature_columns))) - set(selector.indices))
    return {
        "threshold": threshold,
        "feature_columns": feature_columns,
        "kept_feature_names": [feature_columns[index] for index in selector.indices],
        "dropped_feature_names": [feature_columns[index] for index in dropped_indices],
    }
