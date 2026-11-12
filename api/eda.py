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


__all__ = ["little_mcar", "missing_mechanism", "quality_flags", "semantic_audit"]
