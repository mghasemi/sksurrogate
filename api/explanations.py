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
