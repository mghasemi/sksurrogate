"""Scientific EDA diagnostics for one registered partition — Tier 2.

Pure pandas/NumPy helpers with no control-plane state, consumed by
``api/routers/datasets.py`` and surfaced by the Data Preprocessing page:

* :func:`quality_flags` — identifier and high-cardinality columns, quasi-constant
  and near-duplicate columns, leakage indicators (near-perfect target association,
  features constant inside a class, disjoint class ranges), IQR **and** MAD
  outlier counts, and the duplicate-row summary.
* :func:`missing_mechanism` — missing-versus-observed mean shifts, missingness
  indicator correlations, and Little's MCAR test on the numeric columns.
* :func:`semantic_audit` — numeric-looking text, boolean and date-like text,
  whitespace/case-inconsistent levels, and mixed-type columns.

Every result is descriptive unless it is explicitly a named formal test
(``little_mcar``). Each block carries its sample scope so a reader can tell a
full-partition count from a bounded-sample estimate, and no helper modifies the
frame it receives.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from scipy import stats

#: Columns whose names look like row identities. Separators keep "north" from
#: matching "no"; the name hint only ever *supports* a numeric signal.
_IDENTIFIER_NAME = re.compile(
    r"(^|[_\s])(id|uuid|guid|key|index|row|rownumber|no|number)([_\s]|$)", re.IGNORECASE
)
#: Name hints that matter only when the column is also high-cardinality.
_QUASI_IDENTIFIER_NAME = re.compile(
    r"(name|surname|email|phone|address|title|iban|account)", re.IGNORECASE
)

HIGH_CARDINALITY_RATIO = 0.5
IDENTIFIER_RATIO = 0.95
QUASI_CONSTANT_SHARE = 0.95
NEAR_DUPLICATE_CORRELATION = 0.999
LEAKAGE_ASSOCIATION = 0.98


def _is_numeric(series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series)) and not pd.api.types.is_bool_dtype(series)


def _date_rate(text) -> float:
    """Share of text values that parse as dates, with a cheap sample gate.

    ``pd.to_datetime`` on every value of every text column is expensive, so a
    20-value probe decides whether the full parse is worth running.
    """
    probe = text.head(20)
    if probe.empty:
        return 0.0
    try:
        probe_rate = float(pd.to_datetime(probe, errors="coerce", format="mixed").notna().mean())
    except (TypeError, ValueError):
        return 0.0
    if probe_rate < 0.8:
        return 0.0
    try:
        return float(pd.to_datetime(text, errors="coerce", format="mixed").notna().mean())
    except (TypeError, ValueError):
        return 0.0


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _sample(frame, max_rows, random_state):
    if max_rows and len(frame) > max_rows > 0:
        return frame.sample(max_rows, random_state=random_state), True
    return frame, False


def _outlier_row(column, series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return None
    q25, median, q75 = (float(v) for v in np.quantile(values, [0.25, 0.5, 0.75]))
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    iqr_count = int(np.sum((values < lower) | (values > upper)))
    mad = float(np.median(np.abs(values - median)))
    mad_scale = 1.4826 * mad
    if mad_scale > 0:
        mad_bound = 3.0 * mad_scale
        mad_count = int(np.sum(np.abs(values - median) > mad_bound))
    else:
        mad_bound = 0.0
        mad_count = 0
    return {
        "column": column,
        "count": int(values.size),
        "median": median,
        "iqr_count": iqr_count,
        "iqr_rate": float(iqr_count / values.size),
        "iqr_lower_fence": lower,
        "iqr_upper_fence": upper,
        "mad": mad,
        "mad_bound": float(mad_bound),
        "mad_count": mad_count,
        "mad_rate": float(mad_count / values.size),
    }


def _near_duplicates(sample, features):
    """Pairs that are identical, equal after rounding, or |r| ~ 1.

    Column hashes and one pandas correlation matrix keep this linear in the
    number of rows: the naive pairwise scan on 100 columns × 5,000 rows would
    dominate the endpoint.
    """
    found = []
    reported = set()
    if len(features) < 2:
        return found
    hashes = {
        column: hash_pandas_object(sample[column], index=False).to_numpy()
        for column in features
    }
    for index, column in enumerate(features):
        for earlier in features[:index]:
            if np.array_equal(hashes[column], hashes[earlier]):
                found.append({"feature": column, "duplicate_of": earlier, "method": "identical"})
                reported.add(column)
                break

    numeric = [column for column in features if _is_numeric(sample[column])]
    if len(numeric) < 2:
        return found
    matrix = sample[numeric].apply(pd.to_numeric, errors="coerce")
    correlation = matrix.corr()
    for index, column in enumerate(numeric):
        if column in reported:
            continue
        for earlier in numeric[:index]:
            if earlier in reported:
                continue
            value = correlation.loc[column, earlier]
            if value is None or not np.isfinite(value) or abs(float(value)) < 0.999:
                continue
            left = matrix[column].to_numpy(dtype=float)
            right = matrix[earlier].to_numpy(dtype=float)
            valid = np.isfinite(left) & np.isfinite(right)
            if int(valid.sum()) >= 10 and np.array_equal(
                np.round(left[valid], 6), np.round(right[valid], 6)
            ):
                found.append(
                    {"feature": column, "duplicate_of": earlier, "method": "equal_after_rounding"}
                )
            elif abs(float(value)) >= NEAR_DUPLICATE_CORRELATION:
                found.append(
                    {
                        "feature": column,
                        "duplicate_of": earlier,
                        "method": "near_perfect_correlation",
                        "value": float(value),
                    }
                )
            else:
                continue
            reported.add(column)
            break
    return found


def _leakage(sample, features, target, associations):
    flags = []
    for entry in associations or []:
        value = entry.get("value")
        if value is not None and abs(float(value)) >= LEAKAGE_ASSOCIATION:
            flags.append(
                {
                    "kind": "near_perfect_target_association",
                    "column": entry.get("feature"),
                    "detail": "%s %s = %.3f"
                    % (entry.get("method"), "value", float(value)),
                    "severity": "warn",
                }
            )
    if target is None or target not in sample.columns:
        return flags
    target_series = sample[target]
    if _is_numeric(target_series):
        return flags
    classes = target_series.astype("string").dropna()
    counts = classes.value_counts()
    if counts.size < 2 or counts.size > 10:
        return flags
    retained = [label for label, count in counts.items() if int(count) >= 10]
    codes = target_series.astype("string")
    for column in features:
        series = sample[column]
        if series.nunique(dropna=True) < 2:
            continue
        for label in retained:
            subset = series[codes == label]
            if subset.nunique(dropna=True) == 1 and subset.notna().any():
                flags.append(
                    {
                        "kind": "feature_constant_within_class",
                        "column": column,
                        "detail": "constant %r inside class %r (%d rows)"
                        % (subset.dropna().iloc[0], label, int(counts[label])),
                        "severity": "warn",
                    }
                )
                break
        if _is_numeric(series) and len(retained) >= 2:
            ranges = {}
            for label in retained:
                values = pd.to_numeric(series[codes == label], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    ranges[label] = (float(values.min()), float(values.max()))
            labels = list(ranges)
            for left_index, left in enumerate(labels):
                for right in labels[left_index + 1:]:
                    left_range, right_range = ranges[left], ranges[right]
                    if left_range[1] < right_range[0] or right_range[1] < left_range[0]:
                        flags.append(
                            {
                                "kind": "disjoint_class_ranges",
                                "column": column,
                                "detail": "range %s [%.4g, %.4g] and %s [%.4g, %.4g] do not overlap"
                                % (left, left_range[0], left_range[1], right, right_range[0], right_range[1]),
                                "severity": "info",
                            }
                        )
                        break
                else:
                    continue
                break
    return flags


def quality_flags(
    frame,
    *,
    target=None,
    ignored=None,
    target_associations=None,
    max_rows=5000,
    random_state=0,
    max_columns=100,
):
    """Return Tier-2 data-quality diagnostics for one partition.

    ``target_associations`` is the profile's association list; when supplied,
    near-perfect associations are reported as leakage hints. All thresholds are
    advisory: the report never proposes an automatic exclusion.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("frame must contain at least one row")
    ignored = set(ignored or [])
    features = [column for column in frame.columns if column != target and column not in ignored][:max_columns]
    sample, sampled = _sample(frame, max_rows, random_state)
    rows = int(len(sample))

    identifier_columns = []
    high_cardinality = []
    quasi_constant = []
    date_like_columns = []
    numeric_text_columns = []
    outliers = []
    for column in features:
        series = sample[column]
        unique = int(series.nunique(dropna=True))
        ratio = float(unique / rows) if rows else 0.0
        date_rate = 0.0
        numeric_rate = 0.0
        if not _is_numeric(series) and not series.dropna().empty:
            text = series.dropna().astype("string")
            date_rate = _date_rate(text)
            numeric_rate = float(pd.to_numeric(text, errors="coerce").notna().mean())
            if date_rate >= 0.9:
                date_like_columns.append({"column": column, "unique_ratio": ratio})
                continue
            if numeric_rate >= 0.95:
                numeric_text_columns.append({"column": column, "unique_ratio": ratio})
                continue
        name_identifier = bool(_IDENTIFIER_NAME.search(str(column)))
        name_hint = bool(_QUASI_IDENTIFIER_NAME.search(str(column)))
        reasons = []
        # A continuous numeric measurement with many distinct values is not an
        # identifier; for numeric columns only a name hint or row-like
        # progression qualifies.
        if not _is_numeric(series) and ratio >= IDENTIFIER_RATIO:
            reasons.append("unique_ratio %.3f" % ratio)
        if name_identifier:
            reasons.append("name looks like an identifier")
        elif _is_numeric(series) and ratio >= IDENTIFIER_RATIO:
            values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
            if values.size >= 3 and np.all(np.diff(values) > 0) and np.allclose(np.diff(values), 1.0):
                reasons.append("consecutive integer row index")
        if name_hint and ratio >= HIGH_CARDINALITY_RATIO:
            reasons.append("name looks like a free-text identifier")
        if reasons:
            identifier_columns.append(
                {"column": column, "unique_ratio": ratio, "distinct": unique, "reason": "; ".join(reasons)}
            )
        elif not _is_numeric(series) and ratio >= HIGH_CARDINALITY_RATIO:
            high_cardinality.append({"column": column, "unique_ratio": ratio, "levels": unique})

        if unique > 1:
            top = series.astype("string").value_counts(dropna=True)
            if not top.empty:
                share = float(top.iloc[0] / max(int(series.notna().sum()), 1))
                if share >= QUASI_CONSTANT_SHARE:
                    quasi_constant.append(
                        {"column": column, "top_value": str(top.index[0]), "top_share": share}
                    )
        if _is_numeric(series):
            row = _outlier_row(column, series)
            if row is not None and (row["iqr_count"] or row["mad_count"]):
                outliers.append(row)

    near_duplicates = _near_duplicates(sample, features)
    leakage = _leakage(sample, features, target, target_associations)

    duplicates = {"exact_rows": 0, "feature_duplicates": 0, "conflicting_label_groups": 0,
                  "conflicting_label_rows": 0}
    if features:
        feature_frame = sample[features]
        duplicates["feature_duplicates"] = int(feature_frame.duplicated(keep="first").sum())
        duplicates["exact_rows"] = int(sample.duplicated(keep="first").sum())
        if target is not None and target in sample.columns:
            duplicated_mask = feature_frame.duplicated(keep=False)
            if duplicated_mask.any():
                grouped = sample.loc[duplicated_mask].groupby(features, dropna=False)[target].nunique(dropna=True)
                conflicts = grouped[grouped > 1]
                duplicates["conflicting_label_groups"] = int(conflicts.size)
                if conflicts.size:
                    keys = set(conflicts.index)
                    key_rows = sample.loc[duplicated_mask, features].itertuples(index=False, name=None)
                    duplicates["conflicting_label_rows"] = int(sum(1 for key in key_rows if key in keys))

    flags = []
    for entry in identifier_columns:
        flags.append(
            {
                "kind": "identifier_like",
                "severity": "warn",
                "column": entry["column"],
                "detail": entry["reason"],
            }
        )
    for entry in high_cardinality:
        flags.append(
            {
                "kind": "high_cardinality",
                "severity": "warn",
                "column": entry["column"],
                "detail": "%d levels (%.1f%% of rows) — one-hot encoding will be wide"
                % (entry["levels"], entry["unique_ratio"] * 100),
            }
        )
    for entry in quasi_constant:
        flags.append(
            {
                "kind": "quasi_constant",
                "severity": "info",
                "column": entry["column"],
                "detail": "%r covers %.1f%% of non-missing rows"
                % (entry["top_value"], entry["top_share"] * 100),
            }
        )
    for entry in near_duplicates:
        flags.append(
            {
                "kind": "near_duplicate_feature",
                "severity": "warn",
                "column": entry["feature"],
                "detail": "%s of %r" % (entry["method"].replace("_", " "), entry["duplicate_of"]),
            }
        )
    for entry in numeric_text_columns:
        flags.append(
            {
                "kind": "numeric_text",
                "severity": "warn",
                "column": entry["column"],
                "detail": "values parse as numbers but the column is stored as text",
            }
        )
    for entry in date_like_columns:
        flags.append(
            {
                "kind": "date_like",
                "severity": "info",
                "column": entry["column"],
                "detail": "values parse as dates — consider a datetime dtype instead of free text",
            }
        )
    flags.extend(leakage)
    if duplicates["conflicting_label_groups"]:
        flags.append(
            {
                "kind": "conflicting_labels",
                "severity": "err",
                "column": None,
                "detail": "%d duplicate feature group(s) carry more than one label"
                % duplicates["conflicting_label_groups"],
            }
        )

    return {
        "rows": rows,
        "sampled": bool(sampled),
        "scope": "sampled rows and selected features; flags are advisory",
        "feature_count": len(features),
        "identifier_columns": identifier_columns,
        "high_cardinality": high_cardinality,
        "quasi_constant": quasi_constant,
        "date_like_columns": date_like_columns,
        "numeric_text_columns": numeric_text_columns,
        "near_duplicate_columns": near_duplicates,
        "leakage": leakage,
        "outliers": outliers,
        "duplicates": duplicates,
        "flags": flags,
        "thresholds": {
            "identifier_unique_ratio": IDENTIFIER_RATIO,
            "high_cardinality_ratio": HIGH_CARDINALITY_RATIO,
            "quasi_constant_share": QUASI_CONSTANT_SHARE,
            "near_duplicate_correlation": NEAR_DUPLICATE_CORRELATION,
            "leakage_association": LEAKAGE_ASSOCIATION,
        },
        "interpretation": (
            "Flags name review candidates (identifier columns, near-duplicates, leakage "
            "hints); none of them is an automatic exclusion and none is a causal claim."
        ),
    }


def _indicator_phi(left: pd.Series, right: pd.Series):
    """Phi coefficient between two missingness indicators; None when undefined."""
    left_values = left.to_numpy(dtype=bool).astype(float)
    right_values = right.to_numpy(dtype=bool).astype(float)
    if left_values.std() == 0 or right_values.std() == 0:
        return None
    return float(np.corrcoef(left_values, right_values)[0, 1])


def little_mcar(matrix, mask, max_iter=200, tol=1e-8):
    """Little's (1988) MCAR test for multivariate-normal data with missingness.

    ``matrix`` is ``n x p`` with NaN where unobserved; ``mask`` is the matching
    observed-indicator matrix. Parameters are estimated by EM, then the statistic

    .. math::

        d^2 = \\sum_j n_j (\\bar y_j - \\hat\\mu_j)^\\top \\hat\\Sigma_{jj}^{-1}
              (\\bar y_j - \\hat\\mu_j)

    is compared with a chi-square on ``sum_j p_j - p`` degrees of freedom, where
    ``j`` indexes the distinct missingness patterns. Returns ``None`` when the
    test is not applicable (too few rows per pattern, singular covariance,
    non-convergence).
    """
    n, p = matrix.shape
    observed = mask
    if p < 2 or n < p + 2:
        return {"statistic": None, "df": None, "p_value": None, "reason": "too_few_rows"}
    patterns, pattern_codes = np.unique(observed, axis=0, return_inverse=True)

    mu = np.array([np.nanmean(matrix[:, column]) if observed[:, column].any() else 0.0
                   for column in range(p)])
    filled = np.where(observed, matrix, mu)
    sigma = np.cov(filled, rowvar=False, bias=True)
    if sigma.ndim == 0:
        return {"statistic": None, "df": None, "p_value": None, "reason": "degenerate_covariance"}
    sigma = np.atleast_2d(sigma)
    scale = float(np.trace(sigma)) / p if np.trace(sigma) > 0 else 1.0
    sigma = sigma + np.eye(p) * (1e-8 * scale)

    previous = -np.inf
    converged = False
    for _ in range(max_iter):
        mu_new = np.zeros(p)
        second = np.zeros((p, p))
        log_likelihood = 0.0
        for pattern, code in zip(patterns, range(len(patterns))):
            rows = pattern_codes == code
            count = int(rows.sum())
            if count == 0:
                continue
            missing = ~pattern
            present = pattern
            values = matrix[np.ix_(rows, present)]
            if present.any():
                sigma_oo = sigma[np.ix_(present, present)]
                try:
                    sigma_oo_inv = np.linalg.inv(sigma_oo)
                    _, log_det = np.linalg.slogdet(sigma_oo)
                except np.linalg.LinAlgError:
                    return {"statistic": None, "df": None, "p_value": None, "reason": "singular_pattern_covariance"}
                deviation = values - mu[present]
                log_likelihood += -0.5 * (count * log_det + float(np.einsum("ij,jk,ik->", deviation, sigma_oo_inv, deviation)))
                if missing.any():
                    beta = sigma[np.ix_(missing, present)] @ sigma_oo_inv
                    conditional_mean = mu[missing] + deviation @ beta.T
                    conditional_cov = sigma[np.ix_(missing, missing)] - beta @ sigma[np.ix_(present, missing)]
                    conditional_cov = (conditional_cov + conditional_cov.T) / 2.0
                    filled_block = np.empty((count, p))
                    filled_block[:, present] = values
                    filled_block[:, missing] = conditional_mean
                    conditional_cov = conditional_cov + np.eye(int(missing.sum())) * (1e-10 * scale)
                    second_missing = conditional_cov * count
                else:
                    filled_block = values
                    second_missing = np.zeros((int(missing.sum()), int(missing.sum())))
            else:  # pragma: no cover - a fully-missing pattern carries no information
                filled_block = np.tile(mu, (count, 1))
                second_missing = np.zeros((p, p))
            mu_new += filled_block.sum(axis=0)
            second += filled_block.T @ filled_block
            if missing.any():
                second[np.ix_(missing, missing)] += second_missing
        mu_new /= n
        sigma_new = second / n - np.outer(mu_new, mu_new)
        sigma_new = (sigma_new + sigma_new.T) / 2.0
        sigma_new = sigma_new + np.eye(p) * (1e-10 * scale)
        delta = float(np.max(np.abs(mu_new - mu)))
        mu, sigma = mu_new, sigma_new
        if abs(log_likelihood - previous) < tol and delta < tol:
            converged = True
            break
        previous = log_likelihood
    if not converged:
        return {"statistic": None, "df": None, "p_value": None, "reason": "em_did_not_converge"}

    statistic = 0.0
    degrees = 0
    used_patterns = 0
    for pattern, code in zip(patterns, range(len(patterns))):
        rows = pattern_codes == code
        count = int(rows.sum())
        if count < 2 or not pattern.any():
            continue
        observed_mean = np.nanmean(matrix[np.ix_(rows, pattern)], axis=0)
        sigma_oo = sigma[np.ix_(pattern, pattern)]
        try:
            sigma_oo_inv = np.linalg.inv(sigma_oo)
        except np.linalg.LinAlgError:
            return {"statistic": None, "df": None, "p_value": None, "reason": "singular_pattern_covariance"}
        deviation = observed_mean - mu[pattern]
        statistic += count * float(deviation @ sigma_oo_inv @ deviation)
        degrees += int(pattern.sum())
        used_patterns += 1
    degrees -= p
    if degrees <= 0 or used_patterns < 2:
        return {"statistic": None, "df": None, "p_value": None, "reason": "insufficient_patterns"}
    statistic = max(statistic, 0.0)
    p_value = float(stats.chi2.sf(statistic, degrees))
    return {
        "statistic": float(statistic),
        "df": int(degrees),
        "p_value": p_value,
        "patterns": used_patterns,
        "rows": int(n),
        "columns": int(p),
        "method": "little_mcar",
        "reason": None,
    }


def missing_mechanism(
    frame,
    *,
    target=None,
    ignored=None,
    max_rows=2000,
    max_columns=12,
    random_state=0,
    max_shifts=30,
    max_pairs=10,
):
    """Descriptive missingness structure plus a formal Little's MCAR test.

    The descriptive part compares each numeric column's mean when a *different*
    column is missing versus observed (a MAR-style signal) and correlates the
    missingness indicators. The formal part runs :func:`little_mcar` on at most
    ``max_columns`` numeric columns with a complete-value sample.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("frame must contain at least one row")
    ignored = set(ignored or [])
    features = [column for column in frame.columns if column != target and column not in ignored]
    sample, sampled = _sample(frame, max_rows, random_state)
    numeric = [column for column in features if _is_numeric(sample[column])][:max_columns]
    with_missing = [column for column in numeric if sample[column].isna().any()]
    usable = with_missing
    matrix = sample[usable].to_numpy(dtype=float) if usable else np.zeros((0, 0))
    mask = np.isfinite(matrix)

    shifts = []
    for missing_column in features:
        missing_mask = sample[missing_column].isna()
        if not missing_mask.any() or missing_mask.all():
            continue
        for other in numeric:
            values = pd.to_numeric(sample[other], errors="coerce")
            observed_values = values[~missing_mask].to_numpy(dtype=float)
            missing_values = values[missing_mask].to_numpy(dtype=float)
            observed_values = observed_values[np.isfinite(observed_values)]
            missing_values = missing_values[np.isfinite(missing_values)]
            if observed_values.size < 5 or missing_values.size < 5:
                continue
            pooled = float(np.sqrt((np.var(observed_values) + np.var(missing_values)) / 2.0))
            if pooled <= 0:
                continue
            difference = float(np.mean(missing_values) - np.mean(observed_values))
            shifts.append(
                {
                    "missing_column": missing_column,
                    "numeric_column": other,
                    "mean_observed": float(np.mean(observed_values)),
                    "mean_missing": float(np.mean(missing_values)),
                    "standardized_difference": difference / pooled,
                    "rows_missing": int(missing_values.size),
                }
            )
    shifts.sort(key=lambda item: -abs(item["standardized_difference"]))

    pairs_dropped = 0
    pairs = []
    if len(usable) >= 2:
        positions = {column: index for index, column in enumerate(usable)}
        for left_index, left in enumerate(usable):
            for right in usable[left_index + 1:]:
                left_missing = pd.Series(~mask[:, left_index])
                right_missing = pd.Series(~mask[:, positions[right]])
                if not left_missing.any() and not right_missing.any():
                    continue
                value = _indicator_phi(left_missing, right_missing)
                if value is not None:
                    pairs.append({"feature_a": left, "feature_b": right, "phi": value})
        pairs.sort(key=lambda item: -abs(item["phi"]))
        pairs_dropped = max(len(pairs) - max_pairs, 0)
        pairs = pairs[:max_pairs]

    mcar = None
    if len(usable) >= 2:
        mcar = little_mcar(matrix, mask)
    elif not with_missing:
        mcar = {"statistic": None, "df": None, "p_value": None, "reason": "no_missing_values"}
    else:
        mcar = {
            "statistic": None,
            "df": None,
            "p_value": None,
            "reason": "fewer_than_two_columns_with_missing_values",
        }

    complete = int(len(sample)) if not len(usable) else int(mask.all(axis=1).sum())
    return {
        "rows": int(len(sample)),
        "sampled": bool(sampled),
        "columns": usable,
        "complete_rows": complete,
        "scope": "sampled rows; numeric columns only for the formal test",
        "shifts": shifts[:max_shifts],
        "shifts_truncated": max(len(shifts) - max_shifts, 0),
        "pattern_correlations": pairs,
        "pattern_correlations_truncated": pairs_dropped,
        "mcar": mcar,
        "interpretation": (
            "Mean shifts and indicator correlations describe the missingness structure; "
            "Little's test assumes multivariate normality and, with large n, rejects on "
            "trivial departures. A rejection does not identify a mechanism."
        ),
    }


def semantic_audit(
    frame,
    *,
    target=None,
    ignored=None,
    max_rows=2000,
    random_state=0,
    max_columns=100,
    max_examples=3,
):
    """Flag columns whose stored dtype and values disagree.

    Detects numeric-looking text, boolean-like text, date-like text,
    whitespace/case-inconsistent categorical levels, and mixed-type columns.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("frame must contain at least one row")
    ignored = set(ignored or [])
    features = [column for column in frame.columns if column != target and column not in ignored][:max_columns]
    sample, sampled = _sample(frame, max_rows, random_state)
    issues = []
    for column in features:
        series = sample[column]
        if series.dropna().empty:
            continue
        if _is_numeric(series):
            continue
        text = series.dropna().astype("string")
        if text.empty:
            continue
        numeric_parse = pd.to_numeric(text, errors="coerce")
        numeric_rate = float(numeric_parse.notna().mean())
        lowered = text.str.strip().str.lower()
        boolean_rate = float(lowered.isin({"true", "false", "yes", "no", "y", "n", "t", "f"}).mean())
        date_rate = _date_rate(text)
        stripped = text.str.strip()
        changed = int((stripped != text).sum())
        normalised = lowered
        case_variants = int(text.nunique()) - int(normalised.nunique())
        if numeric_rate >= 0.95:
            issues.append(
                {
                    "column": column,
                    "issue": "numeric_text",
                    "detail": "%.0f%% of non-missing values parse as numbers but the column is text" % (numeric_rate * 100),
                    "examples": [str(value) for value in text.head(max_examples)],
                    "severity": "warn",
                }
            )
        elif 0.05 <= numeric_rate < 0.95:
            non_numeric_text = text[numeric_parse.isna()]
            issues.append(
                {
                    "column": column,
                    "issue": "mixed_type",
                    "detail": "%.0f%% of values parse as numbers — the rest do not" % (numeric_rate * 100),
                    "examples": list(non_numeric_text.head(max_examples)),
                    "severity": "warn",
                }
            )
        if boolean_rate >= 0.95:
            issues.append(
                {
                    "column": column,
                    "issue": "boolean_text",
                    "detail": "values look boolean (%s)" % ", ".join(sorted(set(lowered.unique()))[:4]),
                    "examples": [str(value) for value in text.head(max_examples)],
                    "severity": "info",
                }
            )
        if date_rate >= 0.90 and numeric_rate < 0.95:
            issues.append(
                {
                    "column": column,
                    "issue": "date_like_text",
                    "detail": "%.0f%% parse as dates — consider a datetime dtype" % (date_rate * 100),
                    "examples": [str(value) for value in text.head(max_examples)],
                    "severity": "info",
                }
            )
        if changed:
            issues.append(
                {
                    "column": column,
                    "issue": "whitespace",
                    "detail": "%d value(s) carry leading/trailing whitespace" % changed,
                    "examples": list(text[stripped != text].head(max_examples)),
                    "severity": "info",
                }
            )
        if case_variants > 0:
            issues.append(
                {
                    "column": column,
                    "issue": "case_inconsistent",
                    "detail": "%d extra level(s) exist only because of letter case" % case_variants,
                    "examples": [str(value) for value in text.head(max_examples)],
                    "severity": "info",
                }
            )
    return {
        "rows": int(len(sample)),
        "sampled": bool(sampled),
        "scope": "sampled rows and selected text columns",
        "issues": issues[: max_columns * 2],
        "interpretation": (
            "These are dtype/value mismatches: parsing text as numbers or dates changes "
            "what a model sees, so review each column before encoding."
        ),
    }


def _cohens_d(left: np.ndarray, right: np.ndarray):
    n_left, n_right = len(left), len(right)
    if n_left < 2 or n_right < 2:
        return None
    pooled = np.sqrt(
        ((n_left - 1) * np.var(left, ddof=1) + (n_right - 1) * np.var(right, ddof=1))
        / (n_left + n_right - 2)
    )
    if not np.isfinite(pooled) or pooled <= 0:
        return None
    return float((np.mean(left) - np.mean(right)) / pooled)


def _cliffs_delta(left: np.ndarray, right: np.ndarray):
    """Cliff's delta from the Mann-Whitney statistic: ``2U/(n1 n2) - 1``.

    Rank-based, so the point estimate and its bootstrap replicates are
    ``O(n log n)`` instead of the naive ``O(n^2)`` pairwise count.
    """
    n_left, n_right = len(left), len(right)
    if n_left < 2 or n_right < 2:
        return None
    ranks = stats.rankdata(np.concatenate([left, right]))
    u_left = float(np.sum(ranks[:n_left]) - n_left * (n_left + 1) / 2.0)
    return 2.0 * u_left / (n_left * n_right) - 1.0


def _effect_interval(left, right, rng, reps, subsample):
    """Bootstrap CIs for Cohen's d and Cliff's delta (percentile method)."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if subsample and (len(left) > subsample or len(right) > subsample):
        if len(left) > subsample:
            left = rng.choice(left, subsample, replace=False)
        if len(right) > subsample:
            right = rng.choice(right, subsample, replace=False)
    d_values = []
    delta_values = []
    for _ in range(reps):
        sample_left = left[rng.integers(0, len(left), len(left))]
        sample_right = right[rng.integers(0, len(right), len(right))]
        d = _cohens_d(sample_left, sample_right)
        delta = _cliffs_delta(sample_left, sample_right)
        if d is not None:
            d_values.append(d)
        if delta is not None:
            delta_values.append(delta)
    def interval(values):
        if len(values) < max(10, reps // 4):
            return None
        low, high = np.quantile(values, [0.025, 0.975])
        return [float(low), float(high)]
    return interval(d_values), interval(delta_values)


def effect_sizes(
    frame,
    *,
    target=None,
    group_by=None,
    ignored=None,
    max_groups=3,
    max_features=8,
    max_rows=2000,
    bootstrap=300,
    subsample=400,
    min_group_rows=10,
    random_state=0,
):
    """Standardized effect sizes between target classes (or a chosen grouping).

    Reports Cohen's d and Cliff's delta per numeric feature and class pair, each
    with a percentile bootstrap interval. Groups beyond ``max_groups`` (by size),
    features beyond ``max_features``, and bootstrap replicates are capped so the
    endpoint stays bounded. No multiplicity correction is applied; intervals are
    per-comparison and a "significant" interval among many comparisons is weak
    evidence.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("frame must contain at least one row")
    ignored = set(ignored or [])
    grouping = group_by or target
    if grouping is None or grouping not in frame.columns:
        return {
            "grouping": None,
            "comparisons": [],
            "reason": "no_grouping_column",
            "interpretation": "Effect sizes need a categorical target or an explicit group_by column.",
        }
    if _is_numeric(frame[grouping]):
        return {
            "grouping": grouping,
            "comparisons": [],
            "reason": "grouping_column_is_numeric",
            "interpretation": (
                "Effect sizes compare groups; a numeric target has no classes. "
                "Pick a categorical group_by column or bin the target first."
            ),
        }
    sample, sampled = _sample(frame, max_rows, random_state)
    labels = sample[grouping].astype("string")
    counts = labels.dropna().value_counts()
    retained = [
        (label, int(count))
        for label, count in counts.items()
        if int(count) >= min_group_rows
    ][:max_groups]
    if len(retained) < 2:
        return {
            "grouping": grouping,
            "comparisons": [],
            "reason": "fewer_than_two_eligible_groups",
            "groups": [{"label": label, "rows": count} for label, count in retained],
            "interpretation": (
                "At least two groups with %d+ rows are required." % min_group_rows
            ),
        }
    features = [
        column
        for column in sample.columns
        if column != grouping and column != target and column not in ignored
        and _is_numeric(sample[column])
        and np.nanstd(pd.to_numeric(sample[column], errors="coerce").to_numpy(dtype=float)) > 0
    ][:max_features]
    rng = np.random.default_rng(random_state)
    comparisons = []
    for column in features:
        values = pd.to_numeric(sample[column], errors="coerce").to_numpy(dtype=float)
        for left_index, (left_label, left_rows) in enumerate(retained):
            for right_label, right_rows in retained[left_index + 1:]:
                left_values = values[(labels == left_label).to_numpy()]
                right_values = values[(labels == right_label).to_numpy()]
                left_values = left_values[np.isfinite(left_values)]
                right_values = right_values[np.isfinite(right_values)]
                if len(left_values) < min_group_rows or len(right_values) < min_group_rows:
                    continue
                d = _cohens_d(left_values, right_values)
                delta = _cliffs_delta(left_values, right_values)
                if d is None and delta is None:
                    continue
                d_ci, delta_ci = _effect_interval(left_values, right_values, rng, bootstrap, subsample)
                comparisons.append(
                    {
                        "feature": column,
                        "group_a": left_label,
                        "group_b": right_label,
                        "n_a": int(len(left_values)),
                        "n_b": int(len(right_values)),
                        "cohens_d": d,
                        "cohens_d_ci": d_ci,
                        "cliffs_delta": delta,
                        "cliffs_delta_ci": delta_ci,
                        "median_difference": float(np.median(left_values) - np.median(right_values)),
                        "direction": (
                            "higher_in_a" if d is not None and d > 0
                            else "higher_in_b" if d is not None else None
                        ),
                    }
                )
    comparisons.sort(
        key=lambda entry: (
            -abs(entry["cliffs_delta"]) if entry["cliffs_delta"] is not None else 0.0
        )
    )
    return {
        "grouping": grouping,
        "rows": int(len(sample)),
        "sampled": bool(sampled),
        "groups": [{"label": label, "rows": count} for label, count in retained],
        "bootstrap": int(bootstrap),
        "subsample": int(subsample),
        "comparisons": comparisons,
        "scope": "sampled rows; numeric features; per-comparison intervals",
        "interpretation": (
            "Cohen's d is a standardized mean difference (0.2 small, 0.5 medium, "
            "0.8 large, in the conventional reading); Cliff's delta is the "
            "probability of a higher value minus the reverse. Intervals are "
            "bootstrap percentiles and are not adjusted for multiple comparisons."
        ),
    }


def formal_tests(frame, *, target=None, ignored=None, max_rows=2000, max_columns=20, alpha=0.05):
    """Normality tests per numeric column (Anderson-Darling).

    Returns the statistic, the critical value at the requested significance level
    (nearest available), the rejection verdict and a fixed caveat. This is the only
    block in the EDA family that is a hypothesis test; it never enters
    ``dataset_profile``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("frame must contain at least one row")
    ignored = set(ignored or [])
    sample, sampled = _sample(frame, max_rows, 0)
    features = [
        column for column in sample.columns
        if column != target and column not in ignored and _is_numeric(sample[column])
    ][:max_columns]
    results = []
    for column in features:
        values = pd.to_numeric(sample[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 8:
            results.append(
                {"column": column, "n": int(values.size), "statistic": None,
                 "critical_value": None, "rejects_normality": None,
                 "reason": "fewer_than_8_finite_values"}
            )
            continue
        if np.unique(values).size < 3:
            results.append(
                {"column": column, "n": int(values.size), "statistic": None,
                 "critical_value": None, "rejects_normality": None,
                 "reason": "fewer_than_3_distinct_values"}
            )
            continue
        try:
            try:
                # Deliberately read the tabulated critical values (SciPy < 1.19).
                # The ``method="interpolate"`` path caps its p-value at the
                # smallest tabulated level (0.15), which reads like a real
                # p-value and is not one; the FutureWarning is suppressed only
                # because we handle the 1.19 signature below.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    result = stats.anderson(values, dist="norm")
            except TypeError:  # SciPy >= 1.19 removed the table attributes
                result = stats.anderson(values, dist="norm", method="interpolate")
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
            results.append(
                {"column": column, "n": int(values.size), "statistic": None,
                 "critical_value": None, "p_value": None, "rejects_normality": None,
                 "reason": str(exc)}
            )
            continue
        critical_values = getattr(result, "critical_values", None)
        if critical_values is None:
            p_value = float(getattr(result, "pvalue", None))
            results.append(
                {
                    "column": column,
                    "n": int(values.size),
                    "statistic": float(result.statistic),
                    "p_value": p_value,
                    "p_value_censored": bool(p_value >= 0.149),
                    "critical_value": None,
                    "significance_level": float(alpha),
                    "rejects_normality": bool(p_value < alpha),
                    "reason": None,
                }
            )
            continue
        levels = np.asarray(result.significance_level, dtype=float)
        index = int(np.argmin(np.abs(levels - alpha * 100.0)))
        critical = float(critical_values[index])
        results.append(
            {
                "column": column,
                "n": int(values.size),
                "statistic": float(result.statistic),
                "critical_value": critical,
                "p_value": None,
                "p_value_censored": None,
                "significance_level": float(levels[index] / 100.0),
                "rejects_normality": bool(result.statistic > critical),
                "reason": None,
            }
        )
    return {
        "rows": int(len(sample)),
        "sampled": bool(sampled),
        "alpha": float(alpha),
        "columns": results,
        "method": "anderson_darling",
        "scope": "sampled rows; numeric feature columns",
        "caveat": (
            "Anderson-Darling tests normality, not whether a transformation is "
            "needed; with large n it rejects on trivial departures, and a "
            "non-rejection never proves normality."
        ),
    }


def _psi(reference, current, bins=10):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size < 2 or current.size < 2:
        return None
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return 0.0 if np.allclose(current, reference[0]) else 1.0
    reference_share = np.clip(np.histogram(reference, bins=edges)[0] / reference.size, 1e-12, None)
    current_share = np.clip(np.histogram(current, bins=edges)[0] / current.size, 1e-12, None)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def partition_drift(reference, current, *, features=None, target=None, ignored=None,
                    max_columns=50, max_rows=5000, random_state=0):
    """Distribution shift between two partitions of the same task.

    Numeric features report a two-sample Kolmogorov-Smirnov test and the
    population stability index (decile bins taken from the reference); categorical
    features report a chi-square test of homogeneity, Cramer's V and the total
    variation distance. All tests are per feature and uncorrected.
    """
    for name, frame in (("reference", reference), ("current", current)):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("%s must be a pandas DataFrame" % name)
        if frame.empty:
            raise ValueError("%s frame must contain at least one row" % name)
    ignored = set(ignored or [])
    reference_sample, reference_sampled = _sample(reference, max_rows, random_state)
    current_sample, current_sampled = _sample(current, max_rows, random_state)
    if features is None:
        features = [
            column for column in reference.columns
            if column in current_sample.columns and column != target and column not in ignored
        ][:max_columns]
    numeric_rows = []
    categorical_rows = []
    for column in features:
        if column not in reference_sample.columns or column not in current_sample.columns:
            continue
        left_series = reference_sample[column]
        right_series = current_sample[column]
        if _is_numeric(left_series) and _is_numeric(right_series):
            left = pd.to_numeric(left_series, errors="coerce").to_numpy(dtype=float)
            right = pd.to_numeric(right_series, errors="coerce").to_numpy(dtype=float)
            left = left[np.isfinite(left)]
            right = right[np.isfinite(right)]
            if left.size < 8 or right.size < 8:
                numeric_rows.append({"column": column, "reason": "too_few_finite_values"})
                continue
            ks = stats.ks_2samp(left, right)
            numeric_rows.append(
                {
                    "column": column,
                    "n_reference": int(left.size),
                    "n_current": int(right.size),
                    "ks_statistic": float(ks.statistic),
                    "ks_p_value": float(ks.pvalue),
                    "psi": _psi(left, right),
                    "mean_reference": float(np.mean(left)),
                    "mean_current": float(np.mean(right)),
                    "shift_in_pooled_sd": float(
                        (np.mean(right) - np.mean(left))
                        / (np.sqrt((np.var(left) + np.var(right)) / 2.0) or np.nan)
                    ),
                    "reason": None,
                }
            )
        else:
            left = left_series.astype("string").fillna("<missing>")
            right = right_series.astype("string").fillna("<missing>")
            left_counts = left.value_counts()
            right_counts = right.value_counts()
            levels = sorted(set(left_counts.index) | set(right_counts.index), key=str)
            if len(levels) < 2:
                categorical_rows.append({"column": column, "reason": "insufficient_levels"})
                continue
            if len(levels) > 50:
                # A chi-square on a near-unique text column is uninformative
                # (expected counts collapse, df explodes); report the shift
                # descriptively instead of a meaningless p-value.
                share_left = left_counts / max(int(left_counts.sum()), 1)
                share_right = right_counts / max(int(right_counts.sum()), 1)
                categorical_rows.append(
                    {
                        "column": column,
                        "levels": int(len(levels)),
                        "total_variation": float(
                            0.5 * np.abs(
                                share_left.reindex(levels, fill_value=0.0).to_numpy(dtype=float)
                                - share_right.reindex(levels, fill_value=0.0).to_numpy(dtype=float)
                            ).sum()
                        ),
                        "reason": "too_many_levels",
                    }
                )
                continue
            # Compare the two marginal distributions directly. A crosstab of the
            # two samples would pair independent rows and test the wrong thing.
            table = np.array(
                [
                    [int(left_counts.get(level, 0)) for level in levels],
                    [int(right_counts.get(level, 0)) for level in levels],
                ],
                dtype=float,
            )
            try:
                chi2, p_value, _, _ = stats.chi2_contingency(table)
            except ValueError as exc:  # pragma: no cover - defensive
                categorical_rows.append({"column": column, "reason": str(exc)})
                continue
            share_left = left_counts / max(int(left_counts.sum()), 1)
            share_right = right_counts / max(int(right_counts.sum()), 1)
            total_variation = float(
                0.5 * np.abs(
                    share_left.reindex(levels, fill_value=0.0).to_numpy(dtype=float)
                    - share_right.reindex(levels, fill_value=0.0).to_numpy(dtype=float)
                ).sum()
            )
            total = float(table.sum())
            categorical_rows.append(
                {
                    "column": column,
                    "chi2": float(chi2),
                    "p_value": float(p_value),
                    "cramers_v": float(np.sqrt(chi2 / total)) if total else None,
                    "total_variation": total_variation,
                    "levels": int(len(levels)),
                    "reason": None,
                }
            )
    return {
        "features": list(features),
        "reference_rows": int(len(reference_sample)),
        "current_rows": int(len(current_sample)),
        "sampled": bool(reference_sampled or current_sampled),
        "numeric": numeric_rows,
        "categorical": categorical_rows,
        "scope": "sampled rows; per-feature tests, uncorrected",
        "caveats": [
            "KS and chi-square p-values are per comparison and are not adjusted for the number of features tested.",
            "PSI above 0.2 conventionally signals a meaningful shift; it is descriptive, not a test.",
            "A shift between partitions can be legitimate (a temporal split) or a leak (a random split with duplicated rows).",
        ],
    }


def stability(frame, *, target=None, ignored=None, max_rows=2000, max_columns=20, random_state=0):
    """How far do the summary statistics move between two random halves?

    Reports per-column relative differences for numeric quantiles/mean and the
    largest share difference for categorical levels, plus the effective sample
    size (complete cases across the tested numeric columns). Large values mean the
    profile's numbers depend on which rows happened to be sampled.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("frame must contain at least one row")
    ignored = set(ignored or [])
    sample, sampled = _sample(frame, max_rows, random_state)
    if len(sample) < 20:
        return {
            "rows": int(len(sample)),
            "columns": [],
            "reason": "fewer_than_20_rows",
            "max_relative_difference": None,
            "interpretation": "Too few rows to compare halves.",
        }
    shuffled = sample.sample(frac=1.0, random_state=random_state)
    half = len(shuffled) // 2
    left, right = shuffled.iloc[:half], shuffled.iloc[half:]
    features = [
        column for column in sample.columns
        if column != target and column not in ignored
    ][:max_columns]
    columns = []
    numeric_columns = []
    for column in features:
        if _is_numeric(sample[column]):
            left_values = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
            right_values = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
            left_values = left_values[np.isfinite(left_values)]
            right_values = right_values[np.isfinite(right_values)]
            if left_values.size < 5 or right_values.size < 5:
                continue
            numeric_columns.append(column)
            pooled = np.concatenate([left_values, right_values])
            q25, q75 = np.quantile(pooled, [0.25, 0.75])
            scale = float(q75 - q25)
            if not np.isfinite(scale) or scale <= 0:
                scale = float(np.std(pooled, ddof=1))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            comparisons = {}
            for name, function in (
                ("mean", np.mean),
                ("median", np.median),
                ("std", lambda values: np.std(values, ddof=1)),
                ("q25", lambda values: np.quantile(values, 0.25)),
                ("q75", lambda values: np.quantile(values, 0.75)),
            ):
                first = float(function(left_values))
                second = float(function(right_values))
                # Scale by the column's IQR so a zero-centred column does not
                # explode the ratio when its mean happens to be near zero.
                comparisons[name] = abs(first - second) / scale
            columns.append(
                {
                    "column": column,
                    "kind": "numeric",
                    "scale": scale,
                    "relative_differences": comparisons,
                    "max_relative_difference": max(comparisons.values()),
                }
            )
        else:
            left_share = left[column].astype("string").value_counts(normalize=True)
            right_share = right[column].astype("string").value_counts(normalize=True)
            levels = left_share.index.union(right_share.index)
            if not len(levels):
                continue
            difference = float(
                np.max(
                    np.abs(
                        left_share.reindex(levels, fill_value=0.0).to_numpy(dtype=float)
                        - right_share.reindex(levels, fill_value=0.0).to_numpy(dtype=float)
                    )
                )
            )
            columns.append(
                {
                    "column": column,
                    "kind": "categorical",
                    "max_share_difference": difference,
                    "max_relative_difference": difference,
                }
            )
    effective_rows = int(len(sample))
    if numeric_columns:
        matrix = sample[numeric_columns].apply(pd.to_numeric, errors="coerce")
        effective_rows = int(matrix.notna().all(axis=1).sum())
    overall = max((entry["max_relative_difference"] for entry in columns), default=None)
    return {
        "rows": int(len(sample)),
        "sampled": bool(sampled),
        "half_rows": int(half),
        "effective_rows": effective_rows,
        "columns": columns,
        "max_relative_difference": overall,
        "scope": "two disjoint random halves of the sampled rows",
        "interpretation": (
            "Values near 1 mean the summary is unstable at this sample size; the "
            "profile should be read as approximate. Categorical entries are absolute "
            "share differences, numeric ones are relative differences."
        ),
    }


def _markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def _format_number(value, digits=4):
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "—"
    if number != 0 and abs(number) < 1e-3:
        return "%.3e" % number
    return ("%." + str(digits) + "f") % number


def markdown_report(payload, *, generated_at=None, max_rows=25):
    """Render the EDA payload as a self-contained Markdown report.

    Every section repeats its sample scope and the run limits, so a reader can
    reproduce or discount the numbers without opening the UI.
    """
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    missing = payload.get("missing") or {}
    composition = payload.get("target_composition") or {}
    formal = payload.get("formal_tests") or {}
    effects = payload.get("effect_sizes") or {}
    stability_report = payload.get("stability") or {}
    flags = payload.get("quality_flags") or {}
    mechanism = payload.get("missing_mechanism") or {}
    audit = payload.get("type_audit") or {}
    sampling = payload.get("row_sampling") or {}

    lines = []
    lines.append("# EDA report — `%s` / `%s`" % (payload.get("task_name"), payload.get("partition")))
    lines.append("")
    lines.append("- Generated: %s (UTC)" % generated_at)
    lines.append("- Target: `%s`; rows: %s; columns: %s"
                 % (payload.get("target"), payload.get("rows"), len(payload.get("columns") or [])))
    lines.append("- Ignored columns: %s" % (", ".join(payload.get("ignored_columns") or []) or "none"))
    lines.append("- Row sampling: %s (limit %s)"
                 % (sampling.get("method", "none"), payload.get("feature_limit")))
    lines.append("- All statistics are descriptive unless a section is labelled as a test;"
                 " no causal claim is made anywhere in this report.")
    lines.append("")

    lines.append("## Missingness")
    lines.append("")
    lines.append("- Missing cells: %s of %s (%s%%)"
                 % (missing.get("missing_cells"), missing.get("total_cells"),
                    _format_number((missing.get("missing_pct") or 0) * 100, 2)))
    lines.append("- Complete rows: %s (%s%%)"
                 % (missing.get("complete_rows"), _format_number((missing.get("complete_pct") or 0) * 100, 2)))
    lines.append("- Target missing: %s" % missing.get("target_missing"))
    column_rows = []
    for entry in (missing.get("columns") or [])[:max_rows]:
        column_rows.append(
            [
                "`%s`" % entry.get("column"),
                entry.get("deduced_type") or entry.get("dtype"),
                "%s (%s%%)" % (entry.get("missing"), _format_number((entry.get("missing_pct") or 0) * 100, 1)),
                entry.get("unique"),
                "yes" if entry.get("constant") else "",
                "yes" if entry.get("ignored") else "",
            ]
        )
    if column_rows:
        lines.append("")
        lines.append(_markdown_table(["Column", "Type", "Missing", "Distinct", "Constant", "Ignored"], column_rows))
    lines.append("")

    if composition:
        lines.append("## Target composition")
        lines.append("")
        if composition.get("kind") == "categorical":
            lines.append("- Classes: %s; minority share: %s; imbalance ratio: %s"
                         % (composition.get("distinct_classes"),
                            _format_number((composition.get("minority_share") or 0) * 100, 1) + "%",
                            _format_number(composition.get("imbalance_ratio"), 2)))
            lines.append("- Recommended stratified folds: %s (feasible: %s)"
                         % (composition.get("recommended_folds"), composition.get("stratification_feasible")))
            lines.append("")
            lines.append(_markdown_table(
                ["Class", "Rows", "Share"],
                [["`%s`" % entry.get("label"), entry.get("count"),
                  _format_number((entry.get("share") or 0) * 100, 1) + "%"]
                 for entry in (composition.get("classes") or [])[:max_rows]],
            ))
        else:
            distribution = composition.get("distribution") or {}
            lines.append("- Numeric target with %s distinct values%s"
                         % (composition.get("distinct_values"),
                            " (binary-like)" if composition.get("binary_like") else ""))
            lines.append("- Skewness: %s; excess kurtosis: %s; IQR outliers: %s"
                         % (_format_number(distribution.get("skewness"), 3),
                            _format_number(distribution.get("excess_kurtosis"), 3),
                            distribution.get("iqr_outlier_count")))
        lines.append("")

    lines.append("## Distributions")
    lines.append("")
    distribution_rows = []
    for entry in (payload.get("column_profiles") or [])[:max_rows]:
        distribution = entry.get("distribution") or {}
        if entry.get("kind") == "numeric":
            summary = "min %s · q25 %s · median %s · q75 %s · max %s" % (
                _format_number(distribution.get("min")),
                _format_number(distribution.get("q25")),
                _format_number(distribution.get("median")),
                _format_number(distribution.get("q75")),
                _format_number(distribution.get("max")),
            )
        else:
            summary = "; ".join(
                "%s (%s%%)" % (level.get("value"), _format_number((level.get("proportion") or 0) * 100, 1))
                for level in (distribution.get("top") or [])[:4]
            )
        distribution_rows.append(["`%s`" % entry.get("column"), entry.get("kind"), summary])
    if distribution_rows:
        lines.append(_markdown_table(["Column", "Kind", "Summary"], distribution_rows))
    lines.append("")

    lines.append("## Associations")
    lines.append("")
    lines.append("Target associations (each row names its statistic):")
    lines.append("")
    association_rows = [
        ["`%s`" % entry.get("feature"), entry.get("method"), _format_number(entry.get("value"), 3)]
        for entry in (payload.get("target_associations") or [])[:max_rows]
    ]
    lines.append(_markdown_table(["Feature", "Statistic", "Value"], association_rows) if association_rows
                 else "No association could be computed.")
    lines.append("")
    lines.append("These are exploratory associations, not causal effects or hypothesis tests.")
    lines.append("")

    multicollinearity = payload.get("multicollinearity")
    if multicollinearity:
        lines.append("## Collinearity")
        lines.append("")
        lines.append("- Condition number: %s (%s)"
                     % (_format_number(multicollinearity.get("condition_number")),
                        multicollinearity.get("condition_number_reason") or "standardized design matrix"))
        lines.append("")
        lines.append(_markdown_table(
            ["Feature", "VIF", "R²"],
            [["`%s`" % entry.get("feature"),
              _format_number(entry.get("value"), 2) if entry.get("value") is not None
              else (entry.get("reason") or "undefined"),
              _format_number(entry.get("r_squared"), 3)]
             for entry in (multicollinearity.get("vif") or [])],
        ))
        lines.append("")

    if flags:
        lines.append("## Data-quality flags")
        lines.append("")
        if flags.get("flags"):
            lines.append(_markdown_table(
                ["Severity", "Flag", "Column", "Detail"],
                [[flag.get("severity"), flag.get("kind"), flag.get("column") or "—", flag.get("detail")]
                 for flag in flags.get("flags")[:max_rows]],
            ))
        else:
            lines.append("No flag raised.")
        lines.append("")
        duplicates = flags.get("duplicates") or {}
        lines.append("- Feature-duplicate rows: %s; exact duplicates: %s; conflicting-label groups: %s (%s rows)"
                     % (duplicates.get("feature_duplicates"), duplicates.get("exact_rows"),
                        duplicates.get("conflicting_label_groups"), duplicates.get("conflicting_label_rows")))
        lines.append("- Scope: %s" % flags.get("scope"))
        lines.append("")

    if mechanism:
        lines.append("## Missing-data mechanism")
        lines.append("")
        mcar = mechanism.get("mcar") or {}
        if mcar.get("statistic") is not None:
            lines.append("- Little's MCAR: χ² = %s, df = %s, p = %s (n = %s rows, %s columns)"
                         % (_format_number(mcar.get("statistic"), 3), mcar.get("df"),
                            _format_number(mcar.get("p_value")), mcar.get("rows"), mcar.get("columns")))
        else:
            lines.append("- Little's MCAR: not available (%s)" % (mcar.get("reason") or "unknown"))
        lines.append("- Columns tested: %s" % (", ".join(mechanism.get("columns") or []) or "none"))
        if mechanism.get("shifts"):
            lines.append("")
            lines.append(_markdown_table(
                ["Missing column", "Numeric column", "Mean observed", "Mean missing", "Std. diff"],
                [[entry.get("missing_column"), entry.get("numeric_column"),
                  _format_number(entry.get("mean_observed")), _format_number(entry.get("mean_missing")),
                  _format_number(entry.get("standardized_difference"), 3)]
                 for entry in mechanism.get("shifts")[:12]],
            ))
        lines.append("")
        lines.append(mechanism.get("interpretation") or "")
        lines.append("")

    if audit and audit.get("issues"):
        lines.append("## Type audit")
        lines.append("")
        lines.append(_markdown_table(
            ["Severity", "Column", "Issue", "Detail"],
            [[issue.get("severity"), issue.get("column"), issue.get("issue"), issue.get("detail")]
             for issue in audit.get("issues")[:max_rows]],
        ))
        lines.append("")

    if formal:
        lines.append("## Formal tests — normality (Anderson-Darling)")
        lines.append("")
        lines.append(_markdown_table(
            ["Column", "n", "Statistic", "p", "Critical (5%)", "Rejects normality", "Reason"],
            [["`%s`" % entry.get("column"), entry.get("n"),
              _format_number(entry.get("statistic"), 3),
              _format_number(entry.get("p_value")),
              _format_number(entry.get("critical_value"), 3),
              entry.get("rejects_normality"),
              entry.get("reason") or ""]
             for entry in (formal.get("columns") or [])[:max_rows]],
        ))
        lines.append("")
        lines.append(formal.get("caveat") or "")
        lines.append("")

    if effects and effects.get("comparisons"):
        lines.append("## Effect sizes (bootstrap intervals)")
        lines.append("")
        lines.append("- Grouping: `%s`; groups: %s; bootstrap replicates: %s"
                     % (effects.get("grouping"),
                        ", ".join("%s (%s)" % (group.get("label"), group.get("rows"))
                                  for group in effects.get("groups") or []),
                        effects.get("bootstrap")))
        lines.append("")
        lines.append(_markdown_table(
            ["Feature", "Group A", "Group B", "Cohen's d [95% CI]", "Cliff's δ [95% CI]"],
            [["`%s`" % entry.get("feature"), entry.get("group_a"), entry.get("group_b"),
              "%s %s" % (_format_number(entry.get("cohens_d"), 2),
                         "[%s, %s]" % (_format_number((entry.get("cohens_d_ci") or [None, None])[0], 2),
                                       _format_number((entry.get("cohens_d_ci") or [None, None])[1], 2))
                         if entry.get("cohens_d_ci") else ""),
              "%s %s" % (_format_number(entry.get("cliffs_delta"), 2),
                         "[%s, %s]" % (_format_number((entry.get("cliffs_delta_ci") or [None, None])[0], 2),
                                       _format_number((entry.get("cliffs_delta_ci") or [None, None])[1], 2))
                         if entry.get("cliffs_delta_ci") else "")]
             for entry in (effects.get("comparisons") or [])[:max_rows]],
        ))
        lines.append("")
        lines.append(effects.get("interpretation") or "")
        lines.append("")

    if stability_report:
        lines.append("## Summary stability")
        lines.append("")
        lines.append("- Two halves of the sampled rows; effective rows: %s of %s"
                     % (stability_report.get("effective_rows"), stability_report.get("rows")))
        lines.append("- Largest relative difference: %s"
                     % _format_number(stability_report.get("max_relative_difference"), 3))
        lines.append("")
        lines.append(_markdown_table(
            ["Column", "Kind", "Max relative difference"],
            [["`%s`" % entry.get("column"), entry.get("kind"),
              _format_number(entry.get("max_relative_difference"), 3)]
             for entry in (stability_report.get("columns") or [])[:max_rows]],
        ))
        lines.append("")
        lines.append(stability_report.get("interpretation") or "")
        lines.append("")

    lines.append("## Methods and limits")
    lines.append("")
    lines.append("- Numeric summaries and associations come from the bounded `dataset_profile`"
                 " contract (≤5,000 sampled rows, ≤100 features, ≤10 categories).")
    lines.append("- Formal tests are Anderson-Darling (normality) and Little's MCAR; effect"
                 " intervals are percentile bootstrap; nothing is corrected for multiplicity.")
    lines.append("- Counts labelled *exact* (target composition, duplicate rows) cover the full"
                 " partition; everything else is bounded to the sampled rows named above.")
    lines.append("- This report never modifies the partition and never removes a column.")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "effect_sizes",
    "formal_tests",
    "little_mcar",
    "markdown_report",
    "missing_mechanism",
    "partition_drift",
    "quality_flags",
    "semantic_audit",
    "stability",
]
