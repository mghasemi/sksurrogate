"""Feature-selection/sensitivity-analysis endpoints, wrapping ``SKSurrogate.SensAprx``.

Runs as a background job (see ``api.jobs``) since Sobol/Morris analysis can
be slow on larger feature sets. Also exposes the heatmap data behind
``mltrack.heatmap``: the Pearson correlation matrix and the stored feature
weights table, rendered client-side by the web UI.
"""

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

from SKSurrogate import CorrelationThreshold, SensAprx

from ..config import settings
from ..deps import bad_request, finite_or_none as _finite, not_found, open_tracker
from ..jobs import job_manager

router = APIRouter(prefix="/api/sensitivity", tags=["sensitivity"])


class SensitivityRequest(BaseModel):
    method: str = "sobol"
    n_features_to_select: int = 5
    train_partition: str = "train"


def _heatmap_payload(frame: pd.DataFrame, index_col: str | None, max_features: int) -> dict:
    """Shape a square feature-by-feature frame the way ``mltrack.heatmap`` consumes it."""
    if len(frame.columns) > max_features:
        raise bad_request(
            "Dataset has %d features; the heatmap is limited to %d — prune first or lower the limit"
            % (len(frame.columns), max_features)
        )
    labels = list(frame.index) if index_col is None else [str(v) for v in frame[index_col]]
    data_cols = [c for c in frame.columns if c != index_col]
    return {
        "labels": labels,
        "columns": data_cols,
        "values": [[_finite(v) for v in row] for row in frame[data_cols].itertuples(index=False)],
    }


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
        if analyzer.weights_ is None:
            raise RuntimeError("sensitivity analysis produced no weights")
        weights = [_finite(value) for value in analyzer.weights_]
        return {
            "method": body.method,
            "n_features_to_select": body.n_features_to_select,
            "feature_columns": feature_columns,
            "top_feature_indices": top_indices,
            "top_feature_names": [feature_columns[index] for index in top_indices],
            "weights": weights,
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

@router.get("/{task_name}/correlation-matrix")
def get_correlation_matrix(task_name: str, train_partition: str = "train", max_features: int = 60):
    """Pearson correlation matrix of the features — the data behind ``mltrack.heatmap``."""
    feature_columns, X, _ = _load_train_xy(task_name, train_partition)
    frame = pd.DataFrame(X, columns=feature_columns).corr()
    return {
        "task_name": task_name,
        "partition": train_partition,
        "kind": "correlation",
        **_heatmap_payload(frame, None, max_features),
    }


@router.get("/{task_name}/weights-heatmap")
def get_weights_heatmap(task_name: str, train_partition: str = "train", max_features: int = 60):
    """Feature weights table (pearson / variance / sobol / …) as heatmap data.

    Prefers the ``weights`` table of the task's tracking database — the default
    source of ``mltrack.heatmap``, so the UI renders exactly what the library
    plots. ``mltrack.FeatureWeights`` only populates that table from an
    unpartitioned ``data`` table, which API-registered tasks do not have, so we
    fall back to the two cheap weights (Pearson correlation with the target and
    feature variance) computed from the stored train partition. Richer columns
    appear automatically once the weights table has been filled.
    """
    if not settings.mltrace_db_path(task_name).exists():
        raise not_found("No dataset registered for task %r" % task_name)
    with open_tracker(task_name) as tracker:
        weights_df = tracker.RetrieveWeights()
    if len(weights_df) > 0 and "feature" in weights_df.columns:
        payload = _heatmap_payload(weights_df, "feature", max_features)
        return {"task_name": task_name, "kind": "weights", "available": True, "source": "stored", **payload}

    feature_columns, X, y = _load_train_xy(task_name, train_partition)
    frame = pd.DataFrame(X, columns=feature_columns)
    frame["__target__"] = y
    weights_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "pearson": [frame[column].corr(frame["__target__"]) for column in feature_columns],
            "variance": [frame[column].var() for column in feature_columns],
        }
    )
    payload = _heatmap_payload(weights_df, "feature", max_features)
    return {"task_name": task_name, "kind": "weights", "available": True, "source": "computed", **payload}