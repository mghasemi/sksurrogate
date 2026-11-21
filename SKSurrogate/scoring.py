"""Scoring-direction helpers shared by the search and ensemble weight optimization.

Both call sites convert a metric into a quantity they *maximize* or *minimize*:
``SurrogateRandomCV`` negates a trial score into a loss and minimizes it, while
``optimize_weights`` compares candidate weight vectors and keeps the largest value.
Scikit-learn's scorers are all higher-is-better (``neg_*`` metrics are already
negated, and ``make_scorer``/``get_scorer`` objects carry a ``_sign``), so the
direction is implicit — except for a *bare* callable, which ``check_scoring`` and
``get_scorer`` both return unchanged. Such a callable silently inverts the search.
"""

from __future__ import annotations


def check_scoring_direction(scoring):
    """Reject a scorer that carries no direction.

    The search normalizes every trial score to a *loss* by negating it and then
    minimizes that loss with ``scipy.optimize.minimize``. That is only sound when
    the value is higher-is-better, which scikit-learn guarantees for scorer names
    (``neg_*`` metrics are already negated) and for ``make_scorer``/``get_scorer``
    objects (they carry a ``_sign``). ``sklearn.check_scoring`` returns a *bare*
    callable unchanged, so a plain lower-is-better function such as

    .. code-block:: python

        def mae(estimator, X, y): return float(np.mean(np.abs(y - estimator.predict(X))))

    would be negated into ``-MAE`` and the search would then *maximize* the error —
    it selects the worst model while reporting a finite, plausible score. The same
    inversion hits the ensemble weight search, which keeps the largest score.

    ``None`` (the estimator's own ``score``, higher-is-better), strings and a
    scikit-learn scorer object are accepted; anything else is refused.
    """
    if scoring is None or isinstance(scoring, str):
        return scoring
    if hasattr(scoring, "_sign") and hasattr(scoring, "_score_func"):
        return scoring
    raise TypeError(
        "scoring must be a scorer name (e.g. 'accuracy', 'neg_root_mean_squared_error') "
        "or a make_scorer(...)/get_scorer(...) object, not a bare callable: a bare "
        "callable carries no direction, and the search minimizes its negation — it "
        "would maximize a loss. Wrap it explicitly: "
        "make_scorer(fn, greater_is_better=False) for an error metric, or "
        "make_scorer(fn, greater_is_better=True) for a score."
    )
