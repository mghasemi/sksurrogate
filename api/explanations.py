"""Deterministic, bounded explanation primitives for one stored bundle.

These helpers are deliberately independent of the ``SKSurrogate`` core: the
caller supplies a fitted estimator (or a scalar model-response callable) and a
reference matrix, and nothing here imports plotting code, refits a model, or
adds a dependency beyond NumPy. They back two consumers:

* the explanation router (:mod:`api.routers.explain`) for the control-plane
  endpoints, and
* the controlled-DGP validation harness (:mod:`benchmarks.dgps`), which checks
  the accumulated-local-effects curve against a known linear coefficient.

Interpretation contract
-----------------------
Every helper describes the *fitted model response* over a reference sample. ALE
reduces partial dependence's reliance on off-support predictions, but neither
ALE nor PDP isolates a causal effect, and neither solves sparse support or
conditional-explanation problems automatically.

The full contract (numeric vs categorical features, empty/sparse bins,
classifier output scale, stability under the bounded reference-set policy) is
documented in ``docs/mltrace.rst`` under "Accumulated local effects". The
implementation only supports what that contract states:

* **Numeric features only.** Accumulated local effects require an ordering of
  the feature's values. Nominal categorical features have no canonical order,
  so an ALE curve for them would depend on an arbitrary level ordering; the
  router reports those features as skipped instead of inventing one.
* **Quantile bins with duplicate-edge collapse.** Bin edges are equal-frequency
  quantiles of the reference rows; repeated quantiles (heavy ties) collapse to
  a single edge. If fewer than two distinct edges remain, the feature is
  reported as skipped rather than drawn as a flat curve.
* **Sparse bins merged left, deterministically.** Bins holding fewer than
  ``min_bin_fraction`` of the reference rows are merged with the smaller
  adjacent bin (ties merge left) until every bin clears the floor or only two
  bins remain. With quantile edges no *empty* bin can occur after duplicate-edge
  collapse; rows with a non-finite feature value are excluded from that
  feature's curve and counted in ``excluded_rows``.
* **One scalar response.** Regression uses ``predict``. Binary classification
  uses the positive-class probability (``predict_proba[:, 1]``), falling back
  to ``decision_function`` when probabilities are unavailable; the response is
  labelled in the endpoint payload. Multiclass classification is rejected
  because a single curve would have to pick one class arbitrarily.
* **Descriptive stability, not intervals.** When several reference subsets are
  requested, every subset is accumulated on the *same* bin edges (and every PDP
  curve on the same grid), and only descriptive cross-sample summaries (mean,
  sample SD, min, max) are reported. They are not confidence intervals, not
  significance tests, and not guarantees of generalization.

Additional guarantees (Stage D4)
--------------------------------
* **Grouped importance for correlated features.** Permutation importance splits
  credit between near-duplicate columns in an arbitrary order. Columns whose
  absolute pairwise correlation reaches 0.9 are clustered and permuted
  *together*, so the reported drop is the joint signal rather than a coin flip
  between duplicates. Non-numeric columns are never grouped (their encoding is
  the pipeline's, not the frame's) and the block is skipped with a reason.
* **One labelled response per explanation.** Every scalar-response explanation
  carries ``explained_quantity``: ``prediction``, ``probability`` or
  ``decision_score``. A row-level explanation is never returned without it, so
  a probability-scale attribution cannot be mistaken for a logit.
* **Reconstruction is checked, not assumed.** The ordered-occlusion chain
  telescopes exactly for a deterministic estimator. The residual between the
  sum of contributions and the observed response change is compared with a
  tolerance; a residual above it marks the attribution ``unreliable`` instead of
  reporting a number that does not add up.
* **Out-of-distribution flags are relative.** A row's robust Mahalanobis
  distance (median/IQR scaling, shrunk covariance) is compared with the
  reference sample's own distance distribution at the 99th percentile. The flag
  says "unlike these reference rows", never "wrong prediction".
* **Sign stability across row slices.** Recomputed permutation importance on
  disjoint slices of the stored partition (with the single stored estimator, no
  refit) reports the share of slices agreeing with the majority sign. A feature
  below 0.75 is marked unstable — its direction depends on which rows are
  scored.
"""

from __future__ import annotations

import numpy as np

#: Default number of equal-frequency bins for an ALE curve.
DEFAULT_N_BINS = 20

#: A bin holding less than this fraction of the reference rows is merged.
DEFAULT_MIN_BIN_FRACTION = 0.01


def response_callable(model, family):
    """Return ``(response, meta)`` for a fitted estimator.

    ``response`` maps a 2-D array to a 1-D float array (one scalar model output
    per row). ``meta`` is JSON-ready and records which scalar was reported, so
    an ALE curve on a classifier is never mistaken for a probability-scale
    statement about logits or vice versa.

    Raises :class:`ValueError` with a user-facing message when the estimator
    cannot supply one scalar response (multiclass classification, or a
    classifier exposing neither probabilities nor a decision function).
    """
    if family == "Regression":
        return (lambda values: np.ravel(model.predict(values))), {
            "family": "Regression",
            "output": "predict",
            "causal_effect": False,
        }
    if family != "Classification":
        raise ValueError("Explanations need a scikit-learn estimator")

    classes = getattr(model, "classes_", None)
    classes = None if classes is None else np.asarray(classes)
    if classes is None or classes.size < 2:
        raise ValueError("Classifier does not expose fitted class labels")
    if classes.size > 2:
        raise ValueError(
            "Accumulated local effects need one scalar model response; this "
            "bundle is a %d-class classifier. Refit a binary target or inspect "
            "class-level diagnostics instead." % classes.size
        )
    positive = classes[-1]
    if hasattr(model, "predict_proba"):
        return (
            lambda values: np.ravel(np.asarray(model.predict_proba(values))[:, 1]),
            {
                "family": "Classification",
                "output": "predict_proba[class=%r]" % positive.item(),
                "positive_class": positive.item(),
                "causal_effect": False,
            },
        )
    if hasattr(model, "decision_function"):
        return (
            lambda values: np.ravel(model.decision_function(values)),
            {
                "family": "Classification",
                "output": "decision_function[class=%r]" % positive.item(),
                "positive_class": positive.item(),
                "causal_effect": False,
            },
        )
    raise ValueError(
        "Classifier exposes neither predict_proba nor decision_function, so no "
        "single scalar response is available for accumulated local effects"
    )


def explained_quantity(response_meta):
    """The labelled scalar an explanation describes.

    ``prediction`` (regression), ``probability`` (classifier probabilities) or
    ``decision_score`` (a classifier's signed margin). Kept out of the
    ``response`` mapping itself so that documented payload shape stays additive
    only at the top level.
    """
    output = str((response_meta or {}).get("output") or "")
    if output.startswith("predict_proba"):
        return "probability"
    if output.startswith("decision_function"):
        return "decision_score"
    if output.startswith("predict"):
        return "prediction"
    return None


def reference_subsets(rng, n_rows, size, count, excluded=()):
    """Draw ``count`` distinct index arrays of length ``size`` without replacement.

    This mirrors the bounded reference-sample policy already used for
    permutation-importance stability: the same ``RandomState`` drives every
    draw, and a draw that repeats an earlier subset is rejected and redrawn.
    ``excluded`` holds already-drawn subsets (tuples of sorted indices) so
    callers can extend a sampling sequence without duplicating it.
    """
    if size > n_rows:
        raise ValueError("Reference subset size exceeds the number of available rows")
    seen = set(excluded)
    subsets = []
    while len(subsets) < count:
        keep = rng.choice(n_rows, size, replace=False)
        key = tuple(sorted(int(i) for i in keep))
        if key in seen:
            continue
        seen.add(key)
        subsets.append(keep)
    return subsets


def numeric_bin_edges(column, n_bins=DEFAULT_N_BINS, min_bin_fraction=DEFAULT_MIN_BIN_FRACTION):
    """Equal-frequency bin edges for one numeric feature column.

    Returns ``(edges, counts)``. Edges are quantiles of the finite reference
    values with duplicate quantiles collapsed; sparse bins are merged (left
    neighbour on ties) until each holds at least
    ``ceil(min_bin_fraction * finite_rows)`` rows or only two bins remain.

    Raises :class:`ValueError` when the feature has fewer than two finite
    reference rows or fewer than two distinct quantile edges — both cases where
    an accumulated local effect is undefined.
    """
    values = np.asarray(column, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        raise ValueError("feature has fewer than two finite reference rows")
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.unique(np.quantile(finite, quantiles))
    if edges.size < 3:
        raise ValueError(
            "feature has fewer than two distinct quantile bins; accumulated "
            "local effects are undefined for a (near-)constant feature"
        )
    min_count = max(1, int(np.ceil(float(min_bin_fraction) * finite.size)))
    while edges.size > 3:
        indices = _bin_indices(finite, edges)
        counts = np.bincount(indices, minlength=edges.size - 1)
        sparse = np.flatnonzero(counts < min_count)
        if sparse.size == 0:
            break
        k = int(sparse[0])
        has_left, has_right = k > 0, k < counts.size - 1
        if not has_left:
            drop = k + 1
        elif not has_right:
            drop = k
        else:
            drop = k if counts[k - 1] <= counts[k + 1] else k + 1
        edges = np.delete(edges, drop)
    indices = _bin_indices(finite, edges)
    counts = np.bincount(indices, minlength=edges.size - 1)
    return edges, counts


def _bin_indices(values, edges):
    """Right-open bin membership: value ``e`` belongs to the bin starting at ``e``."""
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def accumulated_local_effects(response, values, feature_index, edges):
    """Accumulate local model-response differences over the given bin edges.

    ``values`` is the reference matrix (rows x features) and ``edges`` the bin
    boundaries from :func:`numeric_bin_edges`. Rows whose feature value is
    non-finite are excluded and reported. The curve is mean-centered using the
    reference-row bin weights, so its weighted mean is zero by construction.

    Returns ``(curve, counts, excluded_rows)`` where ``curve`` has one value per
    bin boundary (the first entry is zero by convention and the centered curve
    is reported for the remaining boundaries).
    """
    matrix = np.asarray(values)
    column = matrix[:, feature_index].astype(float)
    finite = np.isfinite(column)
    excluded = int((~finite).sum())
    indices = _bin_indices(column, edges)
    counts = np.bincount(indices[finite], minlength=len(edges) - 1)

    deltas = np.zeros(len(edges) - 1, dtype=float)
    for k in range(len(edges) - 1):
        rows = np.flatnonzero(finite & (indices == k))
        if rows.size == 0:
            continue
        low = matrix[rows].copy()
        low[:, feature_index] = edges[k]
        high = matrix[rows].copy()
        high[:, feature_index] = edges[k + 1]
        deltas[k] = float(np.mean(response(high) - response(low)))

    accumulated = np.concatenate([[0.0], np.cumsum(deltas)])
    weights = counts / counts.sum() if counts.sum() else np.full(counts.size, 1.0 / counts.size)
    center = float(weights @ accumulated[1:]) if accumulated.size > 1 else 0.0
    return accumulated - center, counts, excluded


def average_pdp_on_grid(response, values, feature_index, grid):
    """Brute-force average partial dependence on an explicit grid.

    Used for cross-sample stability summaries: every subset is evaluated on the
    same grid, so the resulting curves are directly comparable. The primary PDP
    curve in the endpoint may use scikit-learn's recursion shortcut; this
    helper deliberately always uses direct model calls, and the endpoint
    documents the (usually marginal) numerical difference.
    """
    matrix = np.asarray(values)
    column = matrix[:, feature_index].astype(float)
    finite = np.isfinite(column)
    if not finite.any():
        raise ValueError("feature has no finite reference rows")
    reference = matrix[finite]
    curve = np.empty(len(grid), dtype=float)
    for position, point in enumerate(grid):
        perturbed = reference.copy()
        perturbed[:, feature_index] = point
        curve[position] = float(np.mean(response(perturbed)))
    return curve


def descriptive_summary(samples):
    """Per-position mean, sample SD, min, and max of a stack of curves.

    ``samples`` is a ``(n_samples, n_points)`` array. Sample SD uses ``ddof=1``
    and is ``None`` for a single sample; these are descriptive cross-sample
    summaries, not confidence intervals.
    """
    stack = np.asarray(samples, dtype=float)
    if stack.ndim == 1:
        stack = stack[None, :]
    return {
        "mean": stack.mean(axis=0),
        "sample_std": stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros(stack.shape[1]),
        "min": stack.min(axis=0),
        "max": stack.max(axis=0),
        "sample_count": int(stack.shape[0]),
    }


# ---------------------------------------------------------------------------
# Row-level explanations: ordered-occlusion chain decomposition (Stage D3)
# ---------------------------------------------------------------------------

def distinct_permutations(rng, n_items, count):
    """Draw ``count`` distinct random permutations of ``range(n_items)``.

    Used for the symmetric (multi-order) row explanation: every order produces an
    exact telescoping decomposition, and the spread across orders is the
    disclosed order-dependence. Raises :class:`ValueError` when more orders are
    requested than there are distinct permutations.
    """
    if n_items < 1:
        raise ValueError("at least one feature is required for a permutation")
    # n! grows fast; only the small-n regime can bind, but avoid huge factorials.
    limit = 1
    for value in range(2, n_items + 1):
        limit *= value
        if limit >= count:
            break
    if count > limit:
        raise ValueError(
            "at most %d distinct orders exist for %d features" % (limit, n_items)
        )
    seen = set()
    permutations = []
    while len(permutations) < count:
        order = tuple(int(index) for index in rng.permutation(n_items))
        if order in seen:
            continue
        seen.add(order)
        permutations.append(np.asarray(order, dtype=int))
    return permutations


def chain_contributions(response, row_values, baseline_values, order):
    """Ordered-occlusion contributions for one row along one feature order.

    Starting from the baseline vector, each feature named by ``order`` is
    replaced by its observed row value in turn, and the model-response change at
    that step is credited to the feature. The chain telescopes exactly, so
    ``sum(contributions) == f(hybrid_row) - f(baseline)`` for any estimator
    (up to floating-point error). Features not named in ``order`` stay at their
    baseline values, which is what makes a subset request a hybrid-row
    explanation rather than a partial attribution of the full row.

    Returns ``(contributions, hybrid_prediction)`` with ``contributions`` in the
    order the features were visited. This is a model attribution: correlated
    features share credit according to the chosen order, and the decomposition
    is not an intervention or a causal effect.
    """
    row = np.asarray(row_values)
    current = np.asarray(baseline_values).copy()
    previous = float(np.ravel(response(current[None, :]))[0])
    contributions = np.empty(len(order), dtype=float)
    for position, index in enumerate(order):
        following = current.copy()
        following[index] = row[index]
        prediction = float(np.ravel(response(following[None, :]))[0])
        contributions[position] = prediction - previous
        previous = prediction
        current = following
    return contributions, previous


def row_contribution_summary(response, row_values, baseline_values, orders, feature_count):
    """Aggregate ordered-occlusion contributions across multiple feature orders.

    Every order yields an exact decomposition of the same hybrid-row difference;
    their disagreement is the order-dependence, reported as a descriptive sample
    SD / min / max per feature. A single order makes the decomposition exact but
    arbitrary: it does not estimate an order-independent attribution, and this
    helper never claims to.
    """
    matrix = np.zeros((len(orders), feature_count), dtype=float)
    order_totals = np.empty(len(orders), dtype=float)
    for position, order in enumerate(orders):
        contributions, _ = chain_contributions(response, row_values, baseline_values, order)
        matrix[position, order] = contributions
        order_totals[position] = float(contributions.sum())
    summary = descriptive_summary(matrix)
    return {
        "contributions": summary,
        "order_totals": order_totals,
        "matrix": matrix,
    }


# ---------------------------------------------------------------------------
# Explanation guarantees (Stage D4): grouped importance, OOD flags, stability
# ---------------------------------------------------------------------------

def correlated_groups(values, names, *, threshold=0.9, min_finite_rows=20):
    """Cluster numeric features whose absolute pairwise correlation is high.

    Permutation importance splits credit between correlated columns in an
    order-dependent way, so a single column's score drop understates a signal
    that two near-duplicates carry jointly. Grouping them and permuting the
    group together gives an attribution that does not depend on which duplicate
    the estimator happened to use.

    ``values`` is the reference matrix and ``names`` its columns. Only columns
    with at least ``min_finite_rows`` finite, non-constant entries can be
    grouped; every other column forms its own singleton group (it is still
    reported, so the caller never has to merge two group lists). Returns a list
    of ``{"features": [...], "max_abs_correlation": float | None,
    "grouped": bool}`` in input order, preserving the first appearance of each
    group.
    """
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("values must be a 2-D reference matrix")
    if len(names) != matrix.shape[1]:
        raise ValueError("names must have one entry per matrix column")

    usable = []
    for index in range(matrix.shape[1]):
        column = matrix[:, index]
        finite = column[np.isfinite(column)]
        if finite.size >= int(min_finite_rows) and np.ptp(finite) > 0:
            usable.append(index)
    if len(usable) >= 2:
        correlation = np.corrcoef(matrix[:, usable], rowvar=False)
        correlation = np.nan_to_num(np.abs(correlation), nan=0.0)
    else:
        correlation = np.zeros((len(usable), len(usable)))

    position = {column: offset for offset, column in enumerate(usable)}
    parent = list(range(matrix.shape[1]))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    strongest = {}
    for left_offset, left in enumerate(usable):
        for right_offset, right in enumerate(usable):
            if right <= left:
                continue
            value = float(correlation[left_offset, right_offset])
            if value >= float(threshold):
                root_left, root_right = find(left), find(right)
                if root_left != root_right:
                    parent[root_right] = root_left
                strongest[(left, right)] = value

    members = {}
    for index in range(matrix.shape[1]):
        members.setdefault(find(index), []).append(index)

    groups = []
    for root in sorted(members, key=members.get):
        indices = members[root]
        peak = None
        if len(indices) > 1:
            peaks = [
                strongest.get((min(a, b), max(a, b)))
                for a in indices
                for b in indices
                if a < b and (min(a, b), max(a, b)) in strongest
            ]
            peaks = [value for value in peaks if value is not None]
            peak = max(peaks) if peaks else None
        groups.append(
            {
                "features": [str(names[index]) for index in indices],
                "max_abs_correlation": None if peak is None else float(peak),
                "grouped": len(indices) > 1,
            }
        )
    return groups


def grouped_permutation_importance(
    estimator, values, y, groups, *, n_repeats=5, random_state=0
):
    """Permutation score drop for whole feature groups.

    Each group's columns are shuffled together with one shared row permutation,
    so the joint signal is destroyed while the marginals of every column are
    preserved. Returns one entry per group in the order supplied. The score is
    ``estimator.score`` on the supplied rows; like every permutation importance
    this is a model attribution on these rows, not a causal effect.
    """
    matrix = np.asarray(values, dtype=float)
    target = np.asarray(y)
    if matrix.ndim != 2:
        raise ValueError("values must be a 2-D matrix")
    if len(target) != matrix.shape[0]:
        raise ValueError("y must have one entry per row of values")
    if int(n_repeats) < 1:
        raise ValueError("n_repeats must be positive")

    rng = np.random.RandomState(random_state)
    baseline = float(estimator.score(matrix, target))
    results = []
    for group in groups:
        features = list(group.get("features") or [])
        if not features:
            continue
        # ``groups`` carries names; the caller aligns them to matrix positions
        # through ``group["indices"]`` when the names are not column labels.
        indices = group.get("indices")
        if indices is None:
            raise ValueError("each group must carry its column indices")
        indices = list(indices)
        drops = np.empty(int(n_repeats), dtype=float)
        for repeat in range(int(n_repeats)):
            shuffled = matrix.copy()
            order = rng.permutation(matrix.shape[0])
            shuffled[:, indices] = matrix[order][:, indices]
            drops[repeat] = baseline - float(estimator.score(shuffled, target))
        results.append(
            {
                "features": features,
                "grouped": bool(group.get("grouped", len(features) > 1)),
                "max_abs_correlation": group.get("max_abs_correlation"),
                "mean": float(drops.mean()),
                "std": float(drops.std(ddof=1)) if drops.size > 1 else 0.0,
                "n_repeats": int(n_repeats),
                "baseline_score": baseline,
            }
        )
    return results


def out_of_distribution(reference, query, *, percentile=99.0, max_features=200):
    """Flag rows whose feature vector lies outside the reference distribution.

    Robust Mahalanobis distance: median centring, IQR-based scaling (falling
    back to the standard deviation and then to 1 for constant columns), and a
    covariance shrunk towards its diagonal by ``1e-4`` of the mean variance so a
    rank-deficient reference does not produce infinite distances. The threshold
    is the ``percentile`` of the reference rows' own distances, i.e. the flag is
    relative to this reference sample.

    Non-finite cells are imputed with the reference median before the distance
    is computed and counted in ``imputed_cells``. Fewer than two usable numeric
    columns degrade to the maximum absolute robust z-score, which is weaker but
    honest; ``method`` names which statistic was used.

    This is a distributional flag for one fitted model's input space, not a
    guarantee that a prediction is wrong.
    """
    ref = np.asarray(reference, dtype=float)
    qry = np.asarray(query, dtype=float)
    if ref.ndim != 2 or qry.ndim != 2:
        raise ValueError("reference and query must be 2-D matrices")
    if ref.shape[1] != qry.shape[1]:
        raise ValueError("reference and query must have the same number of columns")
    if ref.shape[0] < 10:
        raise ValueError("at least 10 reference rows are needed for an OOD flag")

    width = min(ref.shape[1], int(max_features))
    usable = []
    for index in range(width):
        column = ref[:, index]
        finite = column[np.isfinite(column)]
        if finite.size >= 10 and np.ptp(finite) > 0:
            usable.append(index)
    if not usable:
        raise ValueError("no numeric, non-constant reference column is usable")

    reference = ref[:, usable]
    query_matrix = qry[:, usable]
    center = np.nanmedian(reference, axis=0)
    q75, q25 = np.nanpercentile(reference, [75, 25], axis=0)
    iqr_scale = (q75 - q25) / 1.349
    std_scale = np.nanstd(reference, axis=0, ddof=1)
    scale = np.where(iqr_scale > 0, iqr_scale, np.where(std_scale > 0, std_scale, 1.0))

    imputed = int(np.sum(~np.isfinite(reference))) + int(np.sum(~np.isfinite(query_matrix)))
    reference = np.where(np.isfinite(reference), reference, center)
    query_matrix = np.where(np.isfinite(query_matrix), query_matrix, center)
    z_reference = (reference - center) / scale
    z_query = (query_matrix - center) / scale

    if z_reference.shape[1] >= 2:
        covariance = np.cov(z_reference, rowvar=False)
        shrinkage = 1e-4 * float(np.mean(np.diag(covariance))) + 1e-12
        covariance = covariance + shrinkage * np.eye(covariance.shape[0])
        inverse = np.linalg.pinv(covariance)
        ref_distance = np.sqrt(np.einsum("ij,jk,ik->i", z_reference, inverse, z_reference))
        query_distance = np.sqrt(np.einsum("ij,jk,ik->i", z_query, inverse, z_query))
        threshold = float(np.percentile(ref_distance, float(percentile)))
        method = "robust_mahalanobis"
    else:
        ref_distance = np.abs(z_reference[:, 0])
        query_distance = np.abs(z_query[:, 0])
        threshold = float(np.percentile(ref_distance, float(percentile)))
        method = "robust_max_abs_z"

    return {
        "method": method,
        "threshold": threshold,
        "percentile": float(percentile),
        "reference_rows": int(reference.shape[0]),
        "features_used": len(usable),
        "imputed_cells": imputed,
        "distances": [float(value) for value in query_distance],
        "flagged": [bool(value > threshold) for value in query_distance],
        "threshold_note": (
            "The threshold is the %gth percentile of the reference rows' own "
            "distance, so it is relative to this reference sample and moves with it."
            % float(percentile)
        ),
    }


def sign_consistency(samples, names, *, stable_fraction=0.75):
    """Sign stability of per-fit importances for each feature.

    ``samples`` is a ``(n_fits, n_features)`` array of mean importances from
    independent fits (CV folds or seeds). Permutation importance can be negative
    (shuffling a column improves the score), so its *sign* is meaningful: a
    feature that flips between positive and negative across fits has no stable
    direction on this data. ``sign_consistency`` is the share of fits agreeing
    with the majority sign; ``unstable`` is true below ``stable_fraction``.
    Ties at exactly zero count against neither sign and are reported separately.
    """
    matrix = np.asarray(samples, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("samples must be a 2-D (n_fits, n_features) array")
    if matrix.shape[1] != len(names):
        raise ValueError("names must have one entry per column of samples")
    report = []
    for index, name in enumerate(names):
        column = matrix[:, index]
        positive = int(np.sum(column > 0))
        negative = int(np.sum(column < 0))
        zero = int(np.sum(column == 0))
        decided = positive + negative
        if decided == 0:
            consistency = None
        else:
            consistency = max(positive, negative) / decided
        report.append(
            {
                "feature": str(name),
                "mean": float(np.nanmean(column)) if np.isfinite(column).any() else None,
                "positive_fits": positive,
                "negative_fits": negative,
                "zero_fits": zero,
                "n_fits": int(column.size),
                "sign_consistency": consistency,
                "unstable": bool(consistency is None or consistency < stable_fraction),
            }
        )
    return report
