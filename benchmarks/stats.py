"""Paired statistics for benchmark comparisons.

Statistical contract
--------------------
Every comparison in this module is **paired**: the treatment and control arrays
must be aligned observation by observation (same dataset, same split, same
seed). The unit of inference is the per-pair difference, never the marginal
distributions. Feeding unpaired arrays here silently produces a meaningless
"paired" delta, so callers are responsible for alignment; only length equality
can be checked mechanically.

The reported intervals are **percentile bootstrap** intervals over the paired
deltas. A percentile bootstrap is a frequentist resampling summary: it is *not*
a Bayesian posterior interval and says nothing about the probability that the
true effect lies in the interval. With few pairs (say n < 10) the interval is
correspondingly crude, and with constant deltas it collapses to a point, which
is a statement about the sample, not about the population.

Multiple method-versus-control comparisons share one control, so their p-values
are not independent. :func:`holm_correction` / :func:`wilcoxon_holm` apply the
Holm step-down procedure, which controls the family-wise error rate under
arbitrary dependence; raw Wilcoxon p-values must not be read individually when
more than one method is compared.

Cliff's delta is an **ordinal** effect size: it is the probability that a
random treatment observation exceeds a random control observation, minus the
reverse. It is not a mean difference and is unaffected by monotone rescaling,
so it answers "how often does A beat B", not "by how much".

NaN handling: pairwise NaNs are dropped by :func:`paired_deltas` (a pair is
usable only when both sides are present). The unpaired entry points drop NaNs
from the input vector and report the retained count. Empty input after
dropping raises ``ValueError`` rather than returning a degenerate interval.

No scikit-learn import: this module depends only on numpy and scipy.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

__all__ = [
    "bootstrap_ci",
    "bootstrap_paired_ci",
    "cliffs_delta",
    "holm_correction",
    "paired_deltas",
    "paired_report_row",
    "wilcoxon_holm",
    "wilcoxon_pvalues",
]

# Cliff's delta magnitude bands (Romano et al. 2006, ordinal thresholds).
_CLIFF_NEGLIGIBLE = 0.147
_CLIFF_SMALL = 0.33
_CLIFF_MEDIUM = 0.474


def _as_clean_1d(values, *, name: str) -> np.ndarray:
    """Return ``values`` as a 1-D float array with NaNs removed.

    Raises ``ValueError`` when nothing remains: an empty sample has no
    estimable statistic and returning NaN would hide the caller's bug.
    """
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one non-NaN value")
    return arr


def _percentile_bounds(confidence: float) -> tuple[float, float]:
    """Two-sided equal-tailed percentile bounds for ``confidence``."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    tail = (1.0 - confidence) / 2.0
    return 100.0 * tail, 100.0 * (1.0 - tail)


def bootstrap_ci(
    values,
    *,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    statistic=np.mean,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI for ``statistic`` over ``values``.

    ``seed`` fixes ``np.random.default_rng``, so repeated calls reproduce the
    same endpoints exactly. Endpoints are the equal-tailed percentiles of the
    resampled statistic distribution; they are *not* a Bayesian credible
    interval. NaNs are dropped before resampling and ``n`` reports the count
    actually used.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    arr = _as_clean_1d(values, name="values")
    low_pct, high_pct = _percentile_bounds(confidence)

    rng = np.random.default_rng(seed)
    # Draw all resample indices at once: (n_resamples, n) is cheap at the
    # benchmark sizes here and keeps the RNG stream reproducible per call.
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    samples = np.asarray([statistic(arr[row]) for row in idx], dtype=float)

    estimate = float(statistic(arr))
    low, high = np.percentile(samples, [low_pct, high_pct])
    return {
        "estimate": estimate,
        "low": float(low),
        "high": float(high),
        "n": int(arr.size),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
    }


def paired_deltas(treatment, control) -> np.ndarray:
    """Element-wise ``treatment - control`` for aligned pairs.

    Equal lengths are required (otherwise alignment is guessed, and guessing
    would fabricate a pairing). Pairs where *either* side is NaN are dropped
    and not replaced; the returned array may therefore be shorter than the
    inputs. Deltas of exactly zero are kept: they are informative ties.
    """
    t = np.asarray(treatment, dtype=float).ravel()
    c = np.asarray(control, dtype=float).ravel()
    if t.size != c.size:
        raise ValueError(
            f"paired arrays must have equal length, got {t.size} and {c.size}"
        )
    keep = ~(np.isnan(t) | np.isnan(c))
    return t[keep] - c[keep]


def bootstrap_paired_ci(
    treatment,
    control,
    *,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI for the mean paired difference.

    Adds win/tie/loss counts (delta > 0, == 0, < 0) and ``n_pairs`` to the
    :func:`bootstrap_ci` keys. ``n_pairs`` is the number of usable pairs after
    NaN dropping, so it equals ``n`` here; the separate key makes report rows
    self-describing. As above: percentile, not Bayesian.
    """
    deltas = paired_deltas(treatment, control)
    result = bootstrap_ci(
        deltas,
        n_resamples=n_resamples,
        confidence=confidence,
        statistic=np.mean,
        seed=seed,
    )
    result.update(
        {
            "mean_delta": result["estimate"],
            "wins": int(np.sum(deltas > 0)),
            "ties": int(np.sum(deltas == 0)),
            "losses": int(np.sum(deltas < 0)),
            "n_pairs": int(deltas.size),
        }
    )
    return result


def wilcoxon_pvalues(treatment, control) -> dict:
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Uses ``zero_method="wilcox"`` (zero differences are discarded and the
    sample size reduced to the non-zero count). When every pair ties there is
    no rank information; instead of letting scipy raise, this returns the
    documented sentinel ``p_value = 1.0`` with ``statistic = 0.0`` and
    ``zero_differences = n_pairs``. Treat that as "no evidence", not as a
    computed p-value. Pairwise NaNs are dropped by :func:`paired_deltas`.
    """
    deltas = paired_deltas(treatment, control)
    n_pairs = int(deltas.size)
    zero_differences = int(np.sum(deltas == 0))
    nonzero = deltas[deltas != 0]

    if nonzero.size == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n_pairs": n_pairs,
            "zero_differences": zero_differences,
        }

    result = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_pairs": n_pairs,
        "zero_differences": zero_differences,
    }


def holm_correction(pvalues, *, alpha: float = 0.05) -> dict:
    """Holm step-down adjustment, preserving input order.

    With ``m`` hypotheses sorted by ascending p, the i-th adjusted value is
    ``max_{j <= i} (m - j) * p_(j)`` (0-indexed), clipped at 1.0. The running
    maximum enforces monotonicity, so the reject set is a prefix of the sorted
    order. ``reject`` compares the *adjusted* value to ``alpha``. Holm controls
    the family-wise error rate without assuming independence; raw p-values do
    not.
    """
    raw = [float(p) for p in np.asarray(pvalues, dtype=float).ravel()]
    if not raw:
        raise ValueError("pvalues must not be empty")
    if any(not np.isfinite(p) or p < 0.0 or p > 1.0 for p in raw):
        raise ValueError("pvalues must be finite and lie in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")

    m = len(raw)
    order = np.argsort(np.asarray(raw), kind="stable")
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (m - rank) * raw[index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)

    adjusted = [0.0] * m
    for rank, index in enumerate(order):
        adjusted[index] = float(adjusted_sorted[rank])

    return {
        "adjusted": adjusted,
        "reject": [value <= alpha for value in adjusted],
        "alpha": float(alpha),
    }


def wilcoxon_holm(
    treatment_by_method: dict,
    control,
    *,
    alpha: float = 0.05,
) -> dict:
    """Per-method Wilcoxon tests against one control, Holm-adjusted.

    All methods share the control, so the raw p-values are dependent; the Holm
    step-down procedure (not Bonferroni-by-hand) is what keeps the family-wise
    error rate at ``alpha``. Each entry carries the raw Wilcoxon fields plus
    ``p_adjusted`` and ``reject``. A method whose pairs all tie contributes the
    p=1.0 sentinel and can never be rejected.
    """
    control_arr = np.asarray(control, dtype=float).ravel()
    names = list(treatment_by_method)
    raw: dict[str, dict] = {}
    for name in names:
        raw[name] = wilcoxon_pvalues(treatment_by_method[name], control_arr)

    corrected = holm_correction([raw[name]["p_value"] for name in names], alpha=alpha)

    report: dict[str, dict] = {}
    for position, name in enumerate(names):
        entry = dict(raw[name])
        entry["p_adjusted"] = float(corrected["adjusted"][position])
        entry["reject"] = bool(corrected["reject"][position])
        entry["mean_delta"] = float(
            np.mean(paired_deltas(treatment_by_method[name], control_arr))
        )
        report[name] = entry
    return report


def cliffs_delta(treatment, control) -> dict:
    """Cliff's delta: ordinal effect size for two independent samples.

    ``delta = (#{t > c} - #{t < c}) / (n_t * n_c)`` lies in [-1, 1]; +1 means
    every treatment observation exceeds every control observation. Magnitude
    bands use |delta| < 0.147 negligible, < 0.33 small, < 0.474 medium, else
    large. This is a probability-of-superiority summary, **not** a mean
    difference. NaNs are ignored; empty samples raise ``ValueError``.
    """
    t = _as_clean_1d(treatment, name="treatment")
    c = _as_clean_1d(control, name="control")

    greater = int(np.sum(t[:, None] > c[None, :]))
    less = int(np.sum(t[:, None] < c[None, :]))
    delta = (greater - less) / (t.size * c.size)

    magnitude_abs = abs(delta)
    if magnitude_abs < _CLIFF_NEGLIGIBLE:
        magnitude = "negligible"
    elif magnitude_abs < _CLIFF_SMALL:
        magnitude = "small"
    elif magnitude_abs < _CLIFF_MEDIUM:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {
        "delta": float(delta),
        "magnitude": magnitude,
        "n_treatment": int(t.size),
        "n_control": int(c.size),
    }


def paired_report_row(name: str, treatment, control, *, seed: int = 0) -> dict:
    """One report row summarising a paired treatment-versus-control comparison.

    ``distinguishable`` is a two-part *descriptive* rule: the bootstrap CI for
    the mean paired delta excludes zero AND the uncorrected Wilcoxon p < 0.05.
    It is not a family-wise claim; when several methods are compared, use
    :func:`wilcoxon_holm` and report the adjusted p-values instead. The CI is a
    percentile bootstrap interval (frequentist, not Bayesian), and the Cliff's
    delta is ordinal, so it must not be read as a mean effect.
    """
    ci = bootstrap_paired_ci(treatment, control, seed=seed)
    test = wilcoxon_pvalues(treatment, control)
    effect = cliffs_delta(treatment, control)

    excludes_zero = ci["low"] > 0.0 or ci["high"] < 0.0
    return {
        "name": name,
        "mean_delta": ci["mean_delta"],
        "ci_low": ci["low"],
        "ci_high": ci["high"],
        "confidence": ci["confidence"],
        "wins": ci["wins"],
        "ties": ci["ties"],
        "losses": ci["losses"],
        "n_pairs": ci["n_pairs"],
        "wilcoxon_statistic": test["statistic"],
        "wilcoxon_p": test["p_value"],
        "zero_differences": test["zero_differences"],
        "cliffs_delta": effect["delta"],
        "cliffs_magnitude": effect["magnitude"],
        "distinguishable": bool(excludes_zero and test["p_value"] < 0.05),
    }
