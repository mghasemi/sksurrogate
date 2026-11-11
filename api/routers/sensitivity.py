"""Feature-selection/sensitivity-analysis endpoints, wrapping ``SKSurrogate.SensAprx``.

Runs as a background job (see ``api.jobs``) since Sobol/Morris analysis can
be slow on larger feature sets. Also exposes the heatmap data behind
``mltrack.heatmap``: the Pearson correlation matrix and the stored feature
weights table, rendered client-side by the web UI.
"""

import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from SKSurrogate import CorrelationThreshold, SensAprx

from ..config import settings
from ..features import NumericFeatureError, numeric_train_xy
from ..safety import safe_segment
from ..deps import bad_request, finite_or_none as _finite, not_found, open_tracker, task_access
from ..jobs import job_manager

router = APIRouter(prefix="/api/sensitivity", tags=["sensitivity"], dependencies=[Depends(task_access)])


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
    """Load a partition and project it onto its numeric feature columns.

    Returns ``(projection, fingerprint)`` where ``projection`` is an
    :class:`api.features.NumericTrainData`. The projection is not optional:
    sensitivity (SALib + surrogate regressor) and Pearson-correlation pruning are
    numeric procedures, and a mixed-type partition previously crashed inside
    sklearn with ``could not convert string to float``. Non-numeric columns are
    excluded and reported, and an empty projection is a 400 — not a job that
    fails later with a traceback.
    """
    with open_tracker(task_name) as tracker:
        metadata = tracker.GetMetadata()
        partition_history = tracker.dataset_splits()
    target = metadata.get("target_name")
    if target is None:
        raise not_found("No dataset registered for task %r" % task_name)
    csv_path = settings.task_dataset_dir(task_name) / (safe_segment(train_partition, "partition") + ".csv")
    if not csv_path.exists():
        raise not_found("No dataset partition %r stored for task %r" % (train_partition, task_name))
    frame = pd.read_csv(csv_path)
    try:
        projection = numeric_train_xy(frame, target)
    except NumericFeatureError as exc:
        raise bad_request(str(exc))
    partition_metadata = partition_history.get(train_partition, {})
    return projection, partition_metadata.get("fingerprint")


@router.post("/{task_name}/run")
def run_sensitivity(task_name: str, body: SensitivityRequest):
    """Submit a sensitivity-analysis job and return its job id immediately (202-style)."""
    if body.method not in {"sobol", "morris", "delta-mmnt"}:
        raise bad_request("method must be one of sobol, morris, delta-mmnt")
    projection, train_fingerprint = _load_train_xy(task_name, body.train_partition)
    feature_columns, X, y = projection.columns, projection.X, projection.y

    def _task():
        analyzer = SensAprx(n_features_to_select=body.n_features_to_select, method=body.method)
        analyzer.fit(X, y)
        top_indices = [int(index) for index in analyzer.top_features_[: body.n_features_to_select]]
        if analyzer.weights_ is None:
            raise RuntimeError("sensitivity analysis produced no weights")
        weights = [_finite(value) for value in analyzer.weights_]

        # Persist this run as a weighting column so RetrieveWeights/TopFeatures
        # (and therefore the "Consensus top features" card) see it. Storing is
        # auxiliary: an unwritable database must not discard a valid analysis.
        column = {"sobol": "sobol", "morris": "morris", "delta-mmnt": "delta_mmnt"}[body.method]
        weights_stored = False
        storage_error = None
        try:
            with open_tracker(task_name) as tracker:
                tracker.PersistWeights({column: weights}, features=feature_columns)
            weights_stored = True
        except Exception as exc:  # pragma: no cover - auxiliary
            storage_error = "%s: %s" % (type(exc).__name__, exc)
        return {
            "method": body.method,
            # For delta-mmnt: which SALib estimator produced these weights
            # ("delta" on SALib<=1.5; "delta_balanced" and friends on >=1.6).
            "estimator": getattr(analyzer, "delta_estimator_", None),
            "n_features_to_select": body.n_features_to_select,
            "feature_columns": feature_columns,
            "top_feature_indices": top_indices,
            "top_feature_names": [feature_columns[index] for index in top_indices],
            "weights": weights,
            "weights_column": column,
            "weights_stored": weights_stored,
            "storage_error": storage_error,
            "evaluation_protocol": {
                "train_partition": body.train_partition,
                "train_fingerprint": train_fingerprint,
                "row_count": len(X),
                "feature_columns": feature_columns,
                "excluded_columns": projection.excluded,
                "rows_dropped": projection.dropped_rows,
            },
            "interpretation": {
                "category": "input_sensitivity",
                "measure": body.method,
                "train_partition": body.train_partition,
                "causal_effect": False,
                "limitations": [
                    "Sensitivity describes the fitted surrogate's response over method-generated independent input ranges based on observed feature bounds plus a margin.",
                    "Only numeric feature columns participate: non-numeric columns are excluded (see evaluation_protocol.excluded_columns) because the sampling procedure and its surrogate regressor are numeric.",
                    "Correlated features, and numeric columns that are identifiers rather than measurements, may produce unrealistic perturbations; these indices are not causal effects.",
                ],
            },
        }

    job_id = job_manager.submit(
        task_name,
        "sensitivity",
        lambda job_id: _task,
        metadata_factory=lambda job_id: {"request": body.model_dump(mode="json")},
    )
    return {"job_id": job_id, "status": "queued"}


@router.post("/{task_name}/correlation-threshold")
def run_correlation_threshold(task_name: str, threshold: float = 0.7, train_partition: str = "train"):
    """Fast, synchronous correlation-based feature pruning (no job needed)."""
    projection, _ = _load_train_xy(task_name, train_partition)
    feature_columns, X = projection.columns, projection.X
    selector = CorrelationThreshold(threshold=threshold)
    selector.fit(X)
    dropped_indices = sorted(set(range(len(feature_columns))) - set(selector.indices))
    return {
        "threshold": threshold,
        "feature_columns": feature_columns,
        "kept_feature_names": [feature_columns[index] for index in selector.indices],
        "dropped_feature_names": [feature_columns[index] for index in dropped_indices],
        "excluded_feature_names": projection.excluded_names,
        "excluded_columns": projection.excluded,
        "rows_dropped": projection.dropped_rows,
        "interpretation": {
            "category": "numeric_only",
            "note": "Pearson correlation is undefined for non-numeric columns; they are excluded from pruning and reported separately.",
        },
    }

@router.get("/{task_name}/correlation-matrix")
def get_correlation_matrix(task_name: str, train_partition: str = "train", max_features: int = 60):
    """Pearson correlation matrix of the features — the data behind ``mltrack.heatmap``."""
    projection, _ = _load_train_xy(task_name, train_partition)
    frame = pd.DataFrame(projection.X, columns=projection.columns).corr()
    return {
        "task_name": task_name,
        "partition": train_partition,
        "kind": "correlation",
        "interpretation": {
            "category": "descriptive_association",
            "measure": "Pearson correlation",
            "causal_effect": False,
            "excluded_columns": projection.excluded,
            "rows_dropped": projection.dropped_rows,
        },
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
        # Report which columns the stored weightings cover, so the UI can state
        # the same numeric-only caveat it shows for a computed table. The
        # projection is auxiliary here: a missing partition must not hide the
        # stored weights.
        try:
            projection, _ = _load_train_xy(task_name, train_partition)
            excluded = projection.excluded
        except Exception:  # pragma: no cover - defensive
            excluded = []
        return {
            "task_name": task_name,
            "kind": "weights",
            "available": True,
            "source": "stored",
            "excluded_columns": excluded,
            **payload,
        }

    projection, _ = _load_train_xy(task_name, train_partition)
    frame = pd.DataFrame(projection.X, columns=projection.columns)
    frame["__target__"] = projection.y
    weights_df = pd.DataFrame(
        {
            "feature": projection.columns,
            "pearson": [frame[column].corr(frame["__target__"]) for column in projection.columns],
            "variance": [frame[column].var() for column in projection.columns],
        }
    )
    payload = _heatmap_payload(weights_df, "feature", max_features)
    return {
        "task_name": task_name,
        "kind": "weights",
        "available": True,
        "source": "computed",
        "excluded_columns": projection.excluded,
        **payload,
    }


@router.post("/{task_name}/base-weights")
def compute_base_weights(task_name: str, train_partition: str = "train"):
    """Compute and store the dataset-level weightings (feature-target Pearson, variance).

    Registration seeds these for new partitions; this endpoint backfills tasks
    registered before that existed (and re-seeds after a partition is replaced).
    It is the explicit, side-effecting counterpart to the read-only
    ``top-features`` endpoint: the consensus card offers it when the task has no
    stored weighting at all.
    """
    projection, _ = _load_train_xy(task_name, train_partition)
    data = pd.DataFrame(projection.X, columns=projection.columns)
    values = pd.Series(projection.y)
    weights = {
        "pearson": [data[column].corr(values) for column in projection.columns],
        "variance": [data[column].var() for column in projection.columns],
    }
    with open_tracker(task_name) as tracker:
        stored = tracker.PersistWeights(weights, features=projection.columns)
    return {
        "task_name": task_name,
        "partition": train_partition,
        "weights_columns": [column for column in stored.columns if column != "feature"],
        "feature_count": len(projection.columns),
        "row_count": int(len(projection.X)),
        "excluded_columns": projection.excluded,
        "rows_dropped": projection.dropped_rows,
    }


@router.get("/{task_name}/top-features")
def get_top_features(task_name: str, num: int = 10):
    """Consensus top features across the stored weightings (``mltrack.TopFeatures``).

    Ranks features by how often they appear in the top ``num`` of every weighting
    column of the task's ``weights`` table. That table is only populated when a
    weights analysis has been run and persisted for the task, so when it is missing
    or empty the endpoint reports ``available: false`` with a hint instead of an
    error — the UI renders a disabled card pointing at the sensitivity job.
    """
    if not settings.mltrace_db_path(task_name).exists():
        raise not_found("No dataset registered for task %r" % task_name)

    unavailable = {
        "task_name": task_name,
        "available": False,
        "hint": "No feature weightings stored yet — run a sensitivity analysis first.",
        "features": [],
    }

    with open_tracker(task_name) as tracker:
        try:
            weights_df = tracker.RetrieveWeights()
        except Exception:
            return unavailable
        if len(weights_df) == 0 or "feature" not in weights_df.columns:
            return unavailable
        weight_columns = [column for column in weights_df.columns if column != "feature"]
        if not weight_columns:
            return unavailable
        ranking = tracker.TopFeatures(num=num)

    return {
        "task_name": task_name,
        "available": True,
        "num": num,
        "weightings": len(weight_columns),
        "weight_columns": weight_columns,
        # OrderedDict → list of [feature, appearances] pairs, most frequent first.
        "features": [[feature, int(count)] for feature, count in ranking.items()],
    }