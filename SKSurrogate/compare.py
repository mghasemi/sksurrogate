"""Paired model comparison: statistics, effect sizes and cost/score trade-offs.

This module is the single implementation of the paired statistics used by both
the control plane (``api.routers.comparison``) and the benchmark study code
(``benchmarks.stats`` re-exports these names). It depends only on NumPy and
SciPy, so the core is importable without scikit-learn and without the platform.

Statistical contract
--------------------
Every comparison here is **paired**: the treatment and control arrays must be
aligned observation by observation (same dataset, same split, same seed). The
unit of inference is the per-pair difference, never the marginal distributions.
Feeding unpaired arrays produces a meaningless "paired" delta, so callers are
responsible for alignment; only length equality can be checked mechanically.

Reported intervals are **percentile bootstrap** intervals over the paired
deltas. A percentile bootstrap is a frequentist resampling summary: it is not a
Bayesian posterior interval and says nothing about the probability that the true
effect lies in the interval. With few pairs (say ``n < 10``) the interval is
correspondingly crude; with constant deltas it collapses to a point, which is a
statement about the sample, not about the population.

Multiple model-versus-control comparisons share one control, so their p-values
are dependent. :func:`holm_correction` / :func:`compare_scores` apply the Holm
step-down procedure, which controls the family-wise error rate under arbitrary
dependence; raw Wilcoxon p-values must not be read individually when more than
one model is compared.

Cliff's delta is an **ordinal** effect size: the probability that a random
treatment observation exceeds a random control observation, minus the reverse.
It is not a mean difference and is unaffected by monotone rescaling, so it
answers "how often does A beat B", not "by how much".

NaN handling: pairwise NaNs are dropped (a pair is usable only when both sides
are present). Empty input after dropping raises ``ValueError`` rather than
returning a degenerate interval.

Selection caveat
---------------
Scores carried by stored model bundles were produced under a hyper-parameter
search that *selected* the best configuration on the same folds. Comparing such
scores across bundles is an **exploratory** summary of already-selected models,
not an unbiased estimate of generalization; :func:`compare_scores` reports the
numbers, and the API layer attaches the provenance/warning that keeps this
distinction visible. An unbiased comparison needs a fresh or nested split.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

__all__ = [
    "bootstrap_ci",
    "bootstrap_paired_ci",
    "brier_score",
    "cliffs_delta",
    "compare_scores",
    "expected_calibration_error",
    "fitted_parameter_count",
    "holm_correction",
    "paired_deltas",
    "paired_report_row",
    "pareto_frontier",
    "rank_stability",
    "simpler_within_tolerance",
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
    would fabricate a pairing). Pairs where *either* side is NaN are dropped and
    not replaced; the returned array may therefore be shorter than the inputs.
    Deltas of exactly zero are kept: they are informative ties.
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
    NaN dropping, so it equals ``n`` here; the separate key keeps report rows
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
    sample size reduced to the non-zero count). When every pair ties there is no
    rank information; instead of letting SciPy raise, this returns the
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
    step-down procedure (not Bonferroni-by-hand) keeps the family-wise error
    rate at ``alpha``. Each entry carries the raw Wilcoxon fields plus
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
    :func:`compare_scores` (Holm-adjusted) and report the adjusted p-values
    instead. The CI is a percentile bootstrap interval (frequentist, not
    Bayesian), and Cliff's delta is ordinal, so it must not be read as a mean
    effect.
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


def _paired_cliffs_delta(deltas) -> dict:
    """Cliff's delta for paired differences against zero.

    With paired data the ordinal comparison collapses to counting the signs:
    ``(wins - losses) / n``. Using the two-sample formula here would divide by
    ``n**2`` and understate the effect, so this helper is the paired form only.
    """
    values = _as_clean_1d(deltas, name="deltas")
    wins = int(np.sum(values > 0))
    losses = int(np.sum(values < 0))
    delta = (wins - losses) / values.size
    magnitude_abs = abs(delta)
    if magnitude_abs < _CLIFF_NEGLIGIBLE:
        magnitude = "negligible"
    elif magnitude_abs < _CLIFF_SMALL:
        magnitude = "small"
    elif magnitude_abs < _CLIFF_MEDIUM:
        magnitude = "medium"
    else:
        magnitude = "large"
    return {"delta": float(delta), "magnitude": magnitude, "n": int(values.size)}


def compare_scores(
    scores_by_model: dict,
    *,
    control: str | None = None,
    higher_is_better: bool = True,
    alpha: float = 0.05,
    n_resamples: int = 10000,
    seed: int = 0,
) -> dict:
    """Compare aligned per-fold scores of several models against one control.

    ``scores_by_model`` maps a model name to its per-fold scores, all lists
    aligned to the same folds (index ``i`` is the same split for every model).
    Deltas are oriented so that **positive always means the treatment is
    better**: ``treatment - control`` when ``higher_is_better`` and the reverse
    otherwise. The control defaults to the first key.

    Every non-control model is tested against the control with the Wilcoxon
    signed-rank test; p-values are Holm-adjusted across those comparisons. A
    model is ``distinguishable`` when the percentile bootstrap interval of its
    mean oriented delta excludes zero *and* the adjusted p-value is below
    ``alpha``. Those are frequentist, dependence-aware summaries at the fold
    level; they say nothing about a population of datasets.

    Raises ``ValueError`` for fewer than two models, an unknown control, or
    unaligned score lengths.
    """
    names = list(scores_by_model)
    if len(names) < 2:
        raise ValueError("at least two models are required for a comparison")
    control = names[0] if control is None else control
    if control not in scores_by_model:
        raise ValueError(f"control {control!r} is not one of {names!r}")

    aligned = {
        name: np.asarray(scores_by_model[name], dtype=float).ravel() for name in names
    }
    lengths = {name: values.size for name, values in aligned.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "paired scores must be aligned to the same folds; got lengths %r" % lengths
        )

    def oriented(treatment_values, control_values):
        deltas = paired_deltas(treatment_values, control_values)
        return deltas if higher_is_better else -deltas

    control_values = aligned[control]
    treatment_names = [name for name in names if name != control]
    treatments = {
        name: oriented(aligned[name], control_values) for name in treatment_names
    }
    # A zero vector as the "control" turns wilcoxon_pvalues into a one-sample
    # signed-rank test on the oriented deltas, which is exactly the paired test.
    raw = {
        name: wilcoxon_pvalues(treatments[name], np.zeros_like(treatments[name]))
        for name in treatment_names
    }
    corrected = holm_correction(
        [raw[name]["p_value"] for name in treatment_names], alpha=alpha
    )

    comparisons = []
    for position, name in enumerate(treatment_names):
        deltas = treatments[name]
        base = bootstrap_ci(
            deltas,
            n_resamples=n_resamples,
            statistic=np.mean,
            seed=seed,
        )
        effect = _paired_cliffs_delta(deltas)
        excludes_zero = base["low"] > 0.0 or base["high"] < 0.0
        comparisons.append(
            {
                "treatment": name,
                "control": control,
                "mean_delta": float(deltas.mean()),
                "ci_low": base["low"],
                "ci_high": base["high"],
                "confidence": base["confidence"],
                "wins": int(np.sum(deltas > 0)),
                "ties": int(np.sum(deltas == 0)),
                "losses": int(np.sum(deltas < 0)),
                "n_pairs": int(deltas.size),
                "wilcoxon_statistic": raw[name]["statistic"],
                "wilcoxon_p": raw[name]["p_value"],
                "p_adjusted": float(corrected["adjusted"][position]),
                "cliffs_delta": effect["delta"],
                "cliffs_magnitude": effect["magnitude"],
                "distinguishable": bool(
                    excludes_zero and corrected["adjusted"][position] < alpha
                ),
            }
        )

    models = {}
    for name in names:
        values = aligned[name]
        finite = values[~np.isnan(values)]
        models[name] = {
            "mean": float(finite.mean()) if finite.size else None,
            "std": float(finite.std(ddof=1)) if finite.size > 1 else None,
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "n": int(finite.size),
        }

    return {
        "metric_direction": "higher_is_better" if higher_is_better else "lower_is_better",
        "control": control,
        "alpha": float(alpha),
        "n_resamples": int(n_resamples),
        "models": models,
        "comparisons": comparisons,
    }


def fitted_parameter_count(estimator, *, max_depth: int = 6) -> int:
    """Count the numeric array elements carried by a fitted estimator.

    A complexity proxy that needs no metadata: every numeric ``numpy`` array in
    the object's state counts one per element, and nested estimators (pipeline
    steps, a list of fitted models) are visited recursively. It is a lower bound
    — it ignores scalars, dictionaries of arrays behind unusual attribute names,
    and any state a custom estimator keeps elsewhere — so it ranks models
    sensibly but is not a model-size guarantee. Use bytes on disk when an exact
    size is needed.
    """
    seen: set[int] = set()

    def visit(value, depth: int) -> int:
        if depth > max_depth or id(value) in seen:
            return 0
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.number):
                seen.add(id(value))
                return int(value.size)
            return 0
        if isinstance(value, (str, bytes, int, float, bool, type(None))):
            return 0
        if isinstance(value, dict):
            seen.add(id(value))
            return sum(visit(item, depth + 1) for item in value.values())
        if isinstance(value, (list, tuple, set)):
            seen.add(id(value))
            return sum(visit(item, depth + 1) for item in value)
        state = getattr(value, "__dict__", None)
        if isinstance(state, dict):
            seen.add(id(value))
            return sum(visit(item, depth + 1) for item in state.values())
        return 0

    return visit(estimator, 0)


def simpler_within_tolerance(
    entries: dict,
    *,
    score_key: str = "score",
    complexity_key: str = "complexity",
    higher_is_better: bool = True,
    tolerance: float = 0.01,
) -> dict:
    """The simplest model whose score is within ``tolerance`` of the best.

    ``tolerance`` is in the metric's own units: an absolute margin, not a
    percentage, because a percentage-of-best rule is undefined around zero and
    rewards a worse run on a negative metric. Models missing either objective
    are ignored. Returns ``{"best", "simpler", "tolerance", "best_score"}`` where
    ``simpler`` is ``None`` when the best model is already the simplest eligible
    one.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    ranked = []
    for name, entry in entries.items():
        score = entry.get(score_key)
        complexity = entry.get(complexity_key)
        if score is None or complexity is None:
            continue
        score, complexity = float(score), float(complexity)
        if not (np.isfinite(score) and np.isfinite(complexity)):
            continue
        ranked.append((name, score, complexity))
    if not ranked:
        return {
            "best": None,
            "simpler": None,
            "tolerance": float(tolerance),
            "best_score": None,
        }
    best = max(ranked, key=lambda item: item[1]) if higher_is_better else min(
        ranked, key=lambda item: item[1]
    )
    threshold = best[1] - tolerance if higher_is_better else best[1] + tolerance
    eligible = [
        item
        for item in ranked
        if (item[1] >= threshold if higher_is_better else item[1] <= threshold)
    ]
    simplest = min(eligible, key=lambda item: item[2])
    return {
        "best": best[0],
        "best_score": float(best[1]),
        "simpler": simplest[0] if simplest[0] != best[0] else None,
        "simpler_score": float(simplest[1]),
        "simpler_complexity": float(simplest[2]),
        "tolerance": float(tolerance),
    }


def pareto_frontier(
    entries: dict,
    *,
    score_key: str = "score",
    complexity_key: str = "complexity",
    cost_key: str = "cost",
    higher_is_better: bool = True,
) -> dict:
    """Pareto front over (score, complexity, cost), all minimised internally.

    ``entries`` maps a name to a mapping carrying the three numeric keys. A
    lower score is better, so a higher-is-better metric is negated first.
    Complexity and cost are treated as "smaller is better" (for example model
    parameters or bytes on disk, and mean fit-plus-predict milliseconds).

    Returns ``{"objectives", "frontier": [names], "dominated": {name: dominator}}``
    preserving the input order. Entries with a missing or non-finite objective
    are left out of the front and named in ``incomplete`` — a model cannot be
    ranked on a dimension it does not report.
    """
    objectives = [score_key, complexity_key, cost_key]
    vectors = {}
    incomplete = []
    for name, entry in entries.items():
        values = []
        for key in objectives:
            value = entry.get(key)
            if value is None:
                values.append(None)
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = float("nan")
            values.append(number if np.isfinite(number) else None)
        if any(value is None for value in values):
            incomplete.append(name)
            continue
        vector = np.asarray(values, dtype=float)
        if higher_is_better:
            vector[0] = -vector[0]
        vectors[name] = vector

    names = [name for name in entries if name in vectors]
    frontier, dominated = [], {}
    for name in names:
        point = vectors[name]
        dominator = None
        for other in names:
            if other == name:
                continue
            candidate = vectors[other]
            if np.all(candidate <= point) and np.any(candidate < point):
                dominator = other
                break
        if dominator is None:
            frontier.append(name)
        else:
            dominated[name] = dominator
    return {
        "objectives": {
            "score": "higher_is_better" if higher_is_better else "lower_is_better",
            "complexity": "lower_is_better",
            "cost": "lower_is_better",
        },
        "frontier": frontier,
        "dominated": dominated,
        "incomplete": incomplete,
    }


def rank_stability(scores_by_model: dict, *, higher_is_better: bool = True) -> dict:
    """Per-fold ranks across models and how stable each model's rank is.

    ``scores_by_model`` must be fold-aligned (the same contract as
    :func:`compare_scores`). Ranks are 1-based with 1 = best and ties resolved
    by average rank. ``rank_consistency`` is the share of folds on which the
    model holds its modal rank; it is descriptive — with five folds a
    consistency of 0.6 means two folds disagreed, not that the ranking is
    wrong, and it is never a confidence statement.
    """
    names = list(scores_by_model)
    aligned = {
        name: np.asarray(scores_by_model[name], dtype=float).ravel() for name in names
    }
    lengths = {name: values.size for name, values in aligned.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "paired scores must be aligned to the same folds; got lengths %r" % lengths
        )
    matrix = np.vstack([aligned[name] for name in names])
    oriented = -matrix if higher_is_better else matrix
    # Average ranks for ties, one column per fold.
    ranks = np.empty_like(oriented)
    for fold in range(oriented.shape[1]):
        order = np.argsort(oriented[:, fold], kind="stable")
        sorted_values = oriented[order, fold]
        positions = np.empty(len(names), dtype=float)
        start = 0
        while start < len(sorted_values):
            stop = start + 1
            while stop < len(sorted_values) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            positions[order[start:stop]] = (start + 1 + stop) / 2.0
            start = stop
        ranks[:, fold] = positions

    summary = {}
    for index, name in enumerate(names):
        model_ranks = ranks[index]
        values, counts = np.unique(model_ranks, return_counts=True)
        modal = float(values[np.argmax(counts)])
        summary[name] = {
            "mean_rank": float(model_ranks.mean()),
            "rank_std": float(model_ranks.std(ddof=1)) if model_ranks.size > 1 else 0.0,
            "best_folds": int(np.sum(model_ranks == 1)),
            "modal_rank": modal,
            "rank_consistency": float(np.sum(model_ranks == modal) / model_ranks.size),
            "unstable": bool(np.sum(model_ranks == modal) / model_ranks.size < 0.5),
        }
    return {"n_folds": int(ranks.shape[1]), "models": summary}


def brier_score(y_true, probabilities, *, labels=None) -> float:
    """Brier score for binary or multiclass probabilities.

    ``probabilities`` is ``(n, k)`` for ``k`` classes; a 1-D binary vector is
    accepted as the positive-class probability and expanded to two columns.

    The reported number is the **multiclass** Brier score: the mean over rows of
    the sum of squared errors across classes, in ``[0, 2]``. For binary data this
    is twice the ``sklearn.metrics.brier_score_loss`` convention (a 0.5 forecast
    scores 0.5), so values are not comparable across the two conventions.
    """
    truth = np.asarray(y_true).ravel()
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim == 1:
        # A positive-class probability vector always implies the two labels 0/1,
        # even when this sample happens to contain only one of them.
        if labels is None:
            labels = [0, 1]
        probs = np.column_stack([1.0 - probs, probs])
    if probs.ndim != 2:
        raise ValueError("probabilities must be 2-D (rows x classes)")
    if probs.shape[0] != truth.size:
        raise ValueError(
            "probabilities and y_true must have the same number of rows, got %d and %d"
            % (probs.shape[0], truth.size)
        )
    if labels is None:
        labels = np.unique(truth)
    labels = list(labels)
    if probs.shape[1] != len(labels):
        raise ValueError(
            "probabilities has %d columns but %d class labels were given"
            % (probs.shape[1], len(labels))
        )
    lookup = {label: index for index, label in enumerate(labels)}
    try:
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(truth.size), [lookup[value] for value in truth]] = 1.0
    except KeyError as exc:
        raise ValueError("y_true contains a label absent from labels: %r" % exc) from None
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true,
    probabilities,
    *,
    n_bins: int = 10,
    labels=None,
) -> dict:
    """Top-label expected calibration error (ECE) and its bin table.

    Confidence is the maximum predicted probability and correctness is whether
    the arg-max prediction equals the truth. ``ECE = sum_b (n_b / n) *
    |accuracy_b - confidence_b|``. It is a descriptive summary of the fitted
    model's calibration on this sample: it depends on the bin count and, with
    few rows per bin, is noisy — it is not a test of calibration.
    """
    truth = np.asarray(y_true).ravel()
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probabilities must be 2-D (rows x classes) for ECE")
    if probs.shape[0] != truth.size:
        raise ValueError(
            "probabilities and y_true must have the same number of rows, got %d and %d"
            % (probs.shape[0], truth.size)
        )
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if labels is None:
        labels = np.unique(truth)
    labels = list(labels)
    predicted_index = np.argmax(probs, axis=1)
    predicted = np.asarray(labels, dtype=object)[predicted_index]
    confidence = probs[np.arange(probs.shape[0]), predicted_index]
    correct = predicted == truth

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    # Right-closed bins except the first, so confidence 1.0 lands in the last bin.
    assignment = np.clip(np.searchsorted(edges, confidence, side="left") - 1, 0, n_bins - 1)
    bins = []
    total = 0.0
    for index in range(n_bins):
        mask = assignment == index
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin": index,
                    "range": [float(edges[index]), float(edges[index + 1])],
                    "count": 0,
                    "accuracy": None,
                    "confidence": None,
                }
            )
            continue
        accuracy = float(correct[mask].mean())
        mean_confidence = float(confidence[mask].mean())
        total += count * abs(accuracy - mean_confidence)
        bins.append(
            {
                "bin": index,
                "range": [float(edges[index]), float(edges[index + 1])],
                "count": count,
                "accuracy": accuracy,
                "confidence": mean_confidence,
            }
        )
    return {
        "ece": float(total / max(1, truth.size)),
        "n_bins": int(n_bins),
        "n": int(truth.size),
        "bins": bins,
    }
