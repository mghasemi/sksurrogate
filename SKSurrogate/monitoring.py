"""Data-quality and drift monitoring for tabular model inputs."""

from collections import Counter

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error


def _schema(frame):
    return {
        column: {
            "dtype": str(frame[column].dtype),
            "nullable": bool(frame[column].isna().any()),
        }
        for column in frame.columns
    }


def _distribution(values, categories):
    counts = Counter(str(value) for value in values.dropna())
    total = float(sum(counts.values()))
    return {
        category: counts.get(category, 0) / total if total else 0.0
        for category in categories
    }


def _numeric_drift(reference, current, bins):
    reference = reference.dropna().to_numpy(dtype=float)
    current = current.dropna().to_numpy(dtype=float)
    if not len(reference) or not len(current):
        return {"method": "psi", "value": None, "bins": []}
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        value = 0.0 if np.all(current == reference[0]) else 1.0
        return {"method": "psi", "value": value, "bins": [float(edges[0])]}
    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)
    reference_share = np.clip(reference_counts / len(reference), 1.0e-12, None)
    current_share = np.clip(current_counts / len(current), 1.0e-12, None)
    psi = float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))
    return {"method": "psi", "value": psi, "bins": edges.tolist()}


def drift_report(
    reference,
    current,
    *,
    model_version=None,
    reference_dataset_fingerprint=None,
    current_dataset_fingerprint=None,
    psi_threshold=0.2,
    categorical_threshold=0.2,
    missingness_threshold=0.1,
    range_threshold=0.1,
    bins=10,
):
    """Compare reference and current tabular data and return a JSON-safe report.

    Schema and data-quality changes are reported independently from statistical drift.
    Numeric drift uses population stability index (PSI); categorical drift uses total
    variation distance. Threshold crossings are listed in ``alerts``.
    """
    if not isinstance(reference, pd.DataFrame) or not isinstance(current, pd.DataFrame):
        raise TypeError("reference and current must be pandas DataFrames")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    reference_schema = _schema(reference)
    current_schema = _schema(current)
    reference_columns = list(reference.columns)
    current_columns = list(current.columns)
    missing = [column for column in reference_columns if column not in current_columns]
    extra = [column for column in current_columns if column not in reference_columns]
    reordered = not missing and not extra and reference_columns != current_columns
    schema_changes = {
        "missing_columns": missing,
        "extra_columns": extra,
        "reordered": reordered,
        "dtype_changes": {
            column: {"reference": reference_schema[column]["dtype"], "current": current_schema[column]["dtype"]}
            for column in reference_columns
            if column in current_schema
            and reference_schema[column]["dtype"] != current_schema[column]["dtype"]
        },
    }
    quality = {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "missingness": {},
    }
    drift = {"numeric": {}, "categorical": {}, "range": {}}
    alerts = []
    for column in reference_columns:
        if column not in current:
            continue
        reference_missingness = float(reference[column].isna().mean())
        current_missingness = float(current[column].isna().mean())
        missingness_delta = current_missingness - reference_missingness
        quality["missingness"][column] = {
            "reference": reference_missingness,
            "current": current_missingness,
            "delta": missingness_delta,
        }
        if abs(missingness_delta) >= missingness_threshold:
            alerts.append({"type": "missingness", "column": column, "value": missingness_delta})
        if not pd.api.types.is_numeric_dtype(reference[column]):
            categories = sorted(
                set(str(value) for value in reference[column].dropna())
                | set(str(value) for value in current[column].dropna())
            )
            reference_distribution = _distribution(reference[column], categories)
            current_distribution = _distribution(current[column], categories)
            distance = 0.5 * sum(
                abs(reference_distribution[category] - current_distribution[category])
                for category in categories
            )
            drift["categorical"][column] = {
                "metric": "total_variation_distance",
                "value": float(distance),
                "reference_distribution": reference_distribution,
                "current_distribution": current_distribution,
            }
            if distance >= categorical_threshold:
                alerts.append({"type": "categorical_drift", "column": column, "value": float(distance)})
        else:
            numeric = _numeric_drift(reference[column], current[column], bins)
            drift["numeric"][column] = numeric
            if numeric["value"] is not None and numeric["value"] >= psi_threshold:
                alerts.append({"type": "numeric_drift", "column": column, "value": numeric["value"]})
            reference_min = float(reference[column].min()) if not reference[column].dropna().empty else None
            reference_max = float(reference[column].max()) if not reference[column].dropna().empty else None
            if reference_min is not None and reference_max is not None:
                values = current[column].dropna()
                out_of_range = (
                    float(((values < reference_min) | (values > reference_max)).mean())
                    if not values.empty
                    else 0.0
                )
                drift["range"][column] = {
                    "reference_min": reference_min,
                    "reference_max": reference_max,
                    "current_min": float(values.min()) if not values.empty else None,
                    "current_max": float(values.max()) if not values.empty else None,
                    "out_of_range_rate": out_of_range,
                }
                if out_of_range >= range_threshold:
                    alerts.append({"type": "range_drift", "column": column, "value": out_of_range})
    return {
        "model_version": model_version,
        "reference_dataset_fingerprint": reference_dataset_fingerprint,
        "current_dataset_fingerprint": current_dataset_fingerprint,
        "schema_changes": schema_changes,
        "data_quality": quality,
        "drift": drift,
        "thresholds": {
            "psi": psi_threshold,
            "categorical": categorical_threshold,
            "missingness": missingness_threshold,
            "range": range_threshold,
        },
        "alerts": alerts,
    }


def prediction_distribution_report(
    reference_predictions,
    current_predictions,
    *,
    model_version=None,
    threshold=0.2,
    bins=10,
):
    """Report distribution drift in model predictions."""
    reference = pd.DataFrame({"prediction": reference_predictions})
    current = pd.DataFrame({"prediction": current_predictions})
    report = drift_report(
        reference,
        current,
        model_version=model_version,
        psi_threshold=threshold,
        categorical_threshold=threshold,
        missingness_threshold=1.0,
        range_threshold=1.0,
        bins=bins,
    )
    report["prediction_column"] = "prediction"
    report["alerts"] = [
        alert for alert in report["alerts"]
        if alert["type"] in {"numeric_drift", "categorical_drift"}
    ]
    return report


def delayed_label_performance(predictions, labels, *, model_version=None, task_type="classification"):
    """Compute performance after delayed labels arrive for recorded predictions."""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must contain the same number of rows")
    if not len(labels):
        raise ValueError("predictions and labels cannot be empty")
    if task_type == "classification":
        metrics = {"accuracy": float(accuracy_score(labels, predictions))}
    elif task_type == "regression":
        metrics = {
            "mae": float(mean_absolute_error(labels, predictions)),
            "mse": float(mean_squared_error(labels, predictions)),
        }
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")
    return {"model_version": model_version, "rows": int(len(labels)), "metrics": metrics}


def fairness_report(y_true, y_pred, groups, *, positive_label=1):
    """Summarize demographic parity and equal-opportunity gaps across groups."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    if len(y_true) != len(y_pred) or len(y_true) != len(groups):
        raise ValueError("y_true, y_pred, and groups must have the same length")
    if not len(y_true):
        raise ValueError("inputs cannot be empty")
    by_group = {}
    selection_rates = []
    opportunity_rates = []
    for group in sorted(np.unique(groups)):
        mask = groups == group
        predicted_positive = y_pred[mask] == positive_label
        actual_positive = y_true[mask] == positive_label
        selection_rate = float(np.mean(predicted_positive)) if np.any(mask) else 0.0
        true_positive_rate = float(np.mean(predicted_positive[actual_positive])) if np.any(actual_positive) else 0.0
        by_group[str(group)] = {
            "count": int(np.sum(mask)),
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "accuracy": float(np.mean(y_true[mask] == y_pred[mask])) if np.any(mask) else 0.0,
        }
        selection_rates.append(selection_rate)
        opportunity_rates.append(true_positive_rate)
    fairness = {
        "demographic_parity_gap": float(max(selection_rates) - min(selection_rates)) if selection_rates else 0.0,
        "equal_opportunity_gap": float(max(opportunity_rates) - min(opportunity_rates)) if opportunity_rates else 0.0,
    }
    return {"groups": by_group, "fairness": fairness}


def subgroup_performance_report(y_true, y_pred, groups, *, metric="accuracy", positive_label=1):
    """Report group-level metrics and the maximum performance gap between subgroups."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    if len(y_true) != len(y_pred) or len(y_true) != len(groups):
        raise ValueError("y_true, y_pred, and groups must have the same length")
    if metric not in {"accuracy", "precision", "recall", "f1", "loss"}:
        raise ValueError("metric must be one of accuracy, precision, recall, f1, or loss")
    results = {}
    values = []
    for group in sorted(np.unique(groups)):
        mask = groups == group
        group_true = y_true[mask]
        group_pred = y_pred[mask]
        if metric == "accuracy":
            value = float(np.mean(group_true == group_pred)) if len(group_true) else 0.0
        elif metric == "precision":
            positive_pred = group_pred == positive_label
            actual_positive = group_true == positive_label
            value = float(np.mean(positive_pred[actual_positive])) if np.any(actual_positive) else 0.0
        elif metric == "recall":
            actual_positive = group_true == positive_label
            value = float(np.mean(group_pred[actual_positive] == positive_label)) if np.any(actual_positive) else 0.0
        elif metric == "f1":
            actual_positive = group_true == positive_label
            actual_negative = group_true != positive_label
            tp = np.sum((group_pred == positive_label) & actual_positive)
            fp = np.sum((group_pred == positive_label) & actual_negative)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / np.sum(actual_positive) if np.any(actual_positive) else 0.0
            value = 0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall) / (precision + recall)
        else:
            value = 1.0 - float(np.mean(group_true == group_pred)) if len(group_true) else 0.0
        results[str(group)] = {metric: value, "count": int(len(group_true))}
        values.append(value)
    return {"groups": results, "metric": metric, "fairness_gap": float(max(values) - min(values)) if values else 0.0}


def loss_report(y_true, y_pred, groups=None, *, positive_label=1):
    """Convenience wrapper for group-level loss reporting."""
    if groups is None:
        groups = np.zeros_like(np.asarray(y_true), dtype=object)
    return subgroup_performance_report(y_true, y_pred, groups, metric="loss", positive_label=positive_label)


def sensitive_feature_report(frame, *, sensitive_features=None):
    """Flag sensitive and PII-like columns before model training or serving."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    explicit_sensitive = set(sensitive_features or [])
    pii_tokens = (
        "email",
        "phone",
        "ssn",
        "social_security",
        "address",
        "name",
        "passport",
        "tax_id",
        "dob",
        "date_of_birth",
        "ip_address",
    )
    pii_columns = []
    sensitive_columns = []
    for column in frame.columns:
        lowered = str(column).lower()
        if any(token in lowered for token in pii_tokens):
            pii_columns.append(column)
        if column in explicit_sensitive or any(token in lowered for token in ("ssn", "secret", "password", "token", "api_key")):
            sensitive_columns.append(column)
    pii_columns = sorted(set(pii_columns))
    sensitive_columns = sorted(set(sensitive_columns))
    warnings = []
    if pii_columns:
        warnings.append("PII-like columns detected: %s" % ", ".join(pii_columns))
    if sensitive_columns:
        warnings.append("Sensitive columns detected: %s" % ", ".join(sensitive_columns))
    return {
        "pii_columns": pii_columns,
        "sensitive_columns": sensitive_columns,
        "warnings": warnings,
    }


class InferenceMonitor:
    """Accumulate latency, error-rate, and throughput metrics for requests."""

    def __init__(self, model_version=None):
        self.model_version = model_version
        self._requests = []

    def record(self, latency_ms, rows, *, success=True):
        if latency_ms < 0 or rows < 0:
            raise ValueError("latency_ms and rows must be non-negative")
        self._requests.append({"latency_ms": float(latency_ms), "rows": int(rows), "success": bool(success)})

    def summary(self):
        total = len(self._requests)
        rows = sum(item["rows"] for item in self._requests)
        latency = sum(item["latency_ms"] for item in self._requests)
        successes = sum(item["success"] for item in self._requests)
        return {
            "model_version": self.model_version,
            "requests": total,
            "rows": rows,
            "error_rate": (total - successes) / total if total else 0.0,
            "mean_latency_ms": latency / total if total else 0.0,
            "throughput_rows_per_second": rows / (latency / 1000.0) if latency else 0.0,
        }