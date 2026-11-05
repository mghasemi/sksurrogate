"""Data-quality and drift monitoring for tabular model inputs."""

from collections import Counter
import re

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


def _profile_column_kind(series):
    if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(
        series.dtype
    ) and not pd.api.types.is_complex_dtype(series.dtype):
        return "numeric"
    return "categorical"


def _numeric_summary(series):
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    finite = np.isfinite(values)
    non_finite_count = int(np.sum(~finite))
    values = values[finite]
    if not values.size:
        return {
            "count": 0,
            "non_finite_count": non_finite_count,
            "mean": None,
            "std": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }
    quantiles = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "count": int(values.size),
        "non_finite_count": non_finite_count,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "q25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q75": float(quantiles[2]),
        "max": float(np.max(values)),
    }


def _profile_categorical_values(series, max_categories):
    counts = series.dropna().astype(str).value_counts()
    total = int(counts.sum())
    return {
        "count": total,
        "unique": int(counts.size),
        "top": [
            {
                "value": str(value),
                "count": int(count),
                "proportion": float(count / total) if total else 0.0,
            }
            for value, count in counts.head(max_categories).items()
        ],
        "other_count": int(counts.iloc[max_categories:].sum()),
    }


def _profile_categorical_codes(series, max_categories):
    values = series.astype("string").fillna("<missing>")
    counts = values.value_counts()
    retained = set(counts.head(max_categories).index)
    return values.where(values.isin(retained), "<other>").astype(str)


def _cramers_v(left, right, max_categories):
    left_codes = _profile_categorical_codes(left, max_categories)
    right_codes = _profile_categorical_codes(right, max_categories)
    table = pd.crosstab(left_codes, right_codes).to_numpy(dtype=float)
    sample_count = float(table.sum())
    if sample_count < 2 or min(table.shape) < 2:
        return None
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / sample_count
    valid = expected > 0
    chi_square = float(np.sum(((table - expected) ** 2)[valid] / expected[valid]))
    denominator = sample_count * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(chi_square / denominator)) if denominator > 0 else None


def _correlation_ratio(categories, values, max_categories):
    category_codes = _profile_categorical_codes(categories, max_categories)
    numeric_values = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    valid = numeric_values.notna()
    numeric_values = numeric_values[valid].to_numpy(dtype=float)
    category_codes = category_codes[valid.to_numpy()]
    if numeric_values.size < 2 or np.var(numeric_values) == 0:
        return None
    mean = float(np.mean(numeric_values))
    between = 0.0
    for category in np.unique(category_codes):
        group = numeric_values[category_codes == category]
        between += len(group) * (float(np.mean(group)) - mean) ** 2
    return float(np.sqrt(between / (numeric_values.size * np.var(numeric_values))))


def _profile_association(left, right, left_kind, right_kind, max_categories):
    if left_kind == "numeric" and right_kind == "numeric":
        pair = pd.concat(
            [
                pd.to_numeric(left, errors="coerce").replace([np.inf, -np.inf], np.nan).rename("left"),
                pd.to_numeric(right, errors="coerce").replace([np.inf, -np.inf], np.nan).rename("right"),
            ],
            axis=1,
        ).dropna()
        if len(pair) < 2 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
            return "pearson", None
        return "pearson", float(pair["left"].corr(pair["right"]))
    if left_kind == "categorical" and right_kind == "categorical":
        return "cramers_v", _cramers_v(left, right, max_categories)
    if left_kind == "categorical":
        return "correlation_ratio", _correlation_ratio(left, right, max_categories)
    return "correlation_ratio", _correlation_ratio(right, left, max_categories)


def dataset_profile(
    frame,
    *,
    target=None,
    features=None,
    max_rows=5000,
    max_features=50,
    max_categories=10,
    random_state=0,
):
    """Return bounded mixed-type EDA summaries and association diagnostics.

    Numeric columns use descriptive quantiles; other columns use bounded
    frequency summaries. Target associations and pairwise feature associations
    use Pearson correlation, correlation ratio, or Cramer's V as appropriate.
    Rows are sampled without replacement when needed. These statistics are
    exploratory associations, not evidence of causation or scientific
    mechanisms.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("frame must contain at least one row")
    for name, value, maximum in (
        ("max_rows", max_rows, 10000),
        ("max_features", max_features, 100),
        ("max_categories", max_categories, 50),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 1
            or value > maximum
        ):
            raise ValueError("%s must be an integer between 1 and %d" % (name, maximum))
    if random_state is not None and (
        isinstance(random_state, bool)
        or not isinstance(random_state, (int, np.integer))
    ):
        raise ValueError("random_state must be an integer or None")
    if frame.columns.duplicated().any():
        raise ValueError("frame column names must be unique")
    if target is not None and target not in frame.columns:
        raise ValueError("target column %r is not in frame" % target)

    all_features = [column for column in frame.columns if column != target]
    if features is None:
        if len(all_features) > max_features:
            raise ValueError(
                "Dataset has %d features, exceeding max_features=%d; "
                "select features explicitly" % (len(all_features), max_features)
            )
        selected_features = all_features
    else:
        if isinstance(features, str):
            features = [features]
        if not isinstance(features, (list, tuple)) or not features:
            raise ValueError("features must be a non-empty sequence")
        if len(features) > max_features:
            raise ValueError("Selected features exceed max_features=%d" % max_features)
        if len(features) != len(set(features)):
            raise ValueError("features must be unique")
        invalid = [column for column in features if column not in all_features]
        if invalid:
            raise ValueError("Unknown or non-feature columns selected: %s" % invalid)
        selected_features = list(features)
    if not selected_features:
        raise ValueError("At least one feature column is required")

    original_rows = len(frame)
    if original_rows > max_rows:
        sample_seed = None if random_state is None else int(random_state)
        sample = frame.sample(n=max_rows, replace=False, random_state=sample_seed)
    else:
        sample = frame.copy()

    kinds = {column: _profile_column_kind(sample[column]) for column in selected_features}
    profile_columns = []
    for column in selected_features:
        series = sample[column]
        kind = kinds[column]
        profile_columns.append(
            {
                "column": column,
                "dtype": str(frame[column].dtype),
                "kind": kind,
                "missing_count": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()),
                "unique_count": int(series.nunique(dropna=True)),
                "distribution": (
                    _numeric_summary(series)
                    if kind == "numeric"
                    else _profile_categorical_values(series, max_categories)
                ),
            }
        )

    target_profile = None
    target_associations = []
    if target is not None:
        target_series = sample[target]
        target_kind = _profile_column_kind(target_series)
        target_profile = {
            "column": target,
            "dtype": str(frame[target].dtype),
            "kind": target_kind,
            "missing_count": int(target_series.isna().sum()),
            "missing_rate": float(target_series.isna().mean()),
            "unique_count": int(target_series.nunique(dropna=True)),
            "distribution": (
                _numeric_summary(target_series)
                if target_kind == "numeric"
                else _profile_categorical_values(target_series, max_categories)
            ),
        }
        for column in selected_features:
            method, value = _profile_association(
                sample[column],
                target_series,
                kinds[column],
                target_kind,
                max_categories,
            )
            target_associations.append(
                {"feature": column, "method": method, "value": value}
            )
        target_associations.sort(
            key=lambda item: (
                item["value"] is None,
                -item["value"] if item["value"] is not None else 0.0,
                str(item["feature"]),
            )
        )

    feature_associations = []
    for left_index, left in enumerate(selected_features):
        for right in selected_features[left_index + 1 :]:
            method, value = _profile_association(
                sample[left], sample[right], kinds[left], kinds[right], max_categories
            )
            if value is not None:
                feature_associations.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "method": method,
                        "value": value,
                    }
                )
    feature_associations.sort(
        key=lambda item: (-abs(item["value"]), str(item["feature_a"]), str(item["feature_b"]))
    )
    return {
        "rows": int(original_rows),
        "sampled_rows": int(len(sample)),
        "row_limit": int(max_rows),
        "random_state": int(random_state) if random_state is not None else None,
        "row_sampling": {
            "method": "without_replacement" if original_rows > max_rows else "none",
            "truncated": bool(original_rows > max_rows),
        },
        "feature_count": int(len(selected_features)),
        "feature_limit": int(max_features),
        "target": target_profile,
        "columns": profile_columns,
        "target_associations": target_associations,
        "feature_associations": feature_associations,
        "interpretation": (
            "Descriptive diagnostics and associations only; results do not "
            "establish causation, leakage, or scientific mechanisms."
        ),
    }


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


def _grouped_inputs(y_true, y_pred, groups):
    """Convert to arrays and check the three inputs have equal length."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    if len(y_true) != len(y_pred) or len(y_true) != len(groups):
        raise ValueError("y_true, y_pred, and groups must have the same length")
    return y_true, y_pred, groups


def fairness_report(y_true, y_pred, groups, *, positive_label=1):
    """Summarize demographic parity and equal-opportunity gaps across groups."""
    y_true, y_pred, groups = _grouped_inputs(y_true, y_pred, groups)
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
    y_true, y_pred, groups = _grouped_inputs(y_true, y_pred, groups)
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
            value = float(np.mean(group_true[positive_pred] == positive_label)) if np.any(positive_pred) else 0.0
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


def sensitive_feature_report(frame, *, sensitive_features=None, max_scan_rows=1000):
    """Flag name-based sensitive columns and common PII-like value patterns.

    Value scanning is limited to the first ``max_scan_rows`` rows and inspects
    only text and integer cells. The report contains pattern counts and column
    names, never the matched cell contents. Pattern matches are indicators,
    not validation that a value is genuine PII.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if (
        isinstance(max_scan_rows, bool)
        or not isinstance(max_scan_rows, (int, np.integer))
        or max_scan_rows < 1
        or max_scan_rows > 10000
    ):
        raise ValueError("max_scan_rows must be an integer between 1 and 10000")
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
        normalized_name = re.sub(
            r"[^a-z0-9]+",
            "_",
            re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(column)).lower(),
        ).strip("_")
        padded_name = "_" + normalized_name + "_"
        if any("_%s_" % token in padded_name for token in pii_tokens):
            pii_columns.append(column)
        if column in explicit_sensitive or any(
            "_%s_" % token in padded_name
            for token in ("ssn", "secret", "password", "token", "api_key")
        ):
            sensitive_columns.append(column)
    pii_columns = sorted(set(pii_columns))
    sensitive_columns = sorted(set(sensitive_columns))
    warnings = []
    if pii_columns:
        warnings.append("PII-like columns detected: %s" % ", ".join(pii_columns))
    if sensitive_columns:
        warnings.append("Sensitive columns detected: %s" % ", ".join(sensitive_columns))

    patterns = {
        "email": re.compile(
            r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])",
            re.IGNORECASE,
        ),
        "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
        "phone": re.compile(
            r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?!\w)"
        ),
    }
    scan_frame = frame.head(int(max_scan_rows))
    value_findings = []
    for column in scan_frame.columns:
        counts = {name: 0 for name in patterns}
        scanned_values = 0
        for value in scan_frame[column]:
            if isinstance(value, str):
                text = value
            elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
                text = str(value)
            else:
                continue
            scanned_values += 1
            for name, pattern in patterns.items():
                if pattern.search(text):
                    counts[name] += 1
        matched_counts = {name: count for name, count in counts.items() if count}
        if matched_counts:
            value_findings.append(
                {
                    "column": column,
                    "pattern_counts": matched_counts,
                    "scanned_values": int(scanned_values),
                }
            )
    if value_findings:
        warnings.append(
            "PII-like value patterns detected in %d column(s); review findings "
            "without treating pattern matches as verified identities."
            % len(value_findings)
        )
    return {
        "pii_columns": pii_columns,
        "sensitive_columns": sensitive_columns,
        "value_findings": value_findings,
        "value_scan": {
            "rows_scanned": int(len(scan_frame)),
            "row_limit": int(max_scan_rows),
            "truncated": bool(len(frame) > max_scan_rows),
        },
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