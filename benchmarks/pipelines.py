"""Benchmark-side adapters that make the C-space searchable without patching the library.

Three pieces, each working around a verified property of ``SurrogateRandomCV``:

``PipelineFactory``
    ``_validate_search_space`` (structsearch.py:908-915) requires every search
    parameter name to appear in ``estimator.get_params(deep=True)``. No single
    sklearn pipeline exposes the union of six estimator families' knobs, so the
    factory advertises the flat union and builds the real pipeline at ``fit``
    time from whatever branch is active.

``MixedCategoricalSurface``
    SKSurrogate encodes categorical coordinates ordinally. This regressor maps
    each categorical coordinate to a smooth geometric embedding before fitting a
    Gaussian process, so the surface no longer treats "rf -> hgb -> svc" as a
    metric. Injected through the public ``regressor=`` kwarg.

``pin_threads``
    Wall-clock comparability: SVC/LogReg benefit far more from BLAS threads than
    random forests do, so every arm runs single-threaded for the duration of a
    fit.

Embedding choice
----------------
``"circular"`` (default) maps category ``j`` of ``k`` to ``(cos θ, sin θ)`` with
``θ = 2πj/k``: categories are equidistant, the map is differentiable, and the
surrogate's local optimiser can therefore *use* the surface. ``"onehot"`` is
available for diagnostics, but it makes the surface piecewise constant in the
ordinal coordinate, so SLSQP/COBYLA see a zero gradient and burn iterations on
failed local optimisations (the v1.7 no-op defect). The H2 arm uses the
differentiable embedding so the encoding test is not confounded by a mechanical
optimiser failure.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone

from benchmarks.cspace import CLF_GRID, to_sklearn

__all__ = ["PipelineFactory", "MixedCategoricalSurface", "pin_threads"]


class PipelineFactory(ClassifierMixin, BaseEstimator):
    """A classifier whose parameters are a *union* over six estimator families.

    ``get_params`` exposes every knob in the C-space (plus ``prep``/``reduce``/
    ``clf``); ``fit`` builds the pipeline for the active branch via
    :func:`benchmarks.cspace.to_sklearn` and delegates. Knobs belonging to other
    branches are stored and ignored, which is what lets
    ``SurrogateRandomCV`` validate and ``set_params`` a heterogeneous space.

    :param prep: ``identity`` | ``standard`` | ``robust`` | ``quantile``
    :param reduce: ``identity`` | ``pca4`` | ``pca8`` | ``pca16`` | ``kbest16``
    :param clf: ``logreg`` | ``rf`` | ``et`` | ``hgb`` | ``svc`` | ``knn``
    :param C: regularisation strength (``logreg``, ``svc``)
    :param max_depth: tree depth, ``0`` means ``None`` (``rf``, ``et``)
    :param min_samples_leaf: leaf size (``rf``, ``et``)
    :param learning_rate: boosting shrink (``hgb``)
    :param max_leaf_nodes: leaf budget (``hgb``)
    :param gamma: kernel width (``svc``)
    :param n_neighbors: neighbourhood size (``knn``)
    :param weights: neighbour weighting (``knn``)
    :param random_state: seed forwarded to every sub-estimator
    """

    def __init__(
        self,
        prep: str = "identity",
        reduce: str = "identity",
        clf: str = "logreg",
        C: float = 0.01,
        max_depth: int = 4,
        min_samples_leaf: int = 1,
        learning_rate: float = 0.03,
        max_leaf_nodes: int = 15,
        gamma: Any = "scale",
        n_neighbors: int = 1,
        weights: str = "uniform",
        random_state: int = 0,
    ):
        self.prep = prep
        self.reduce = reduce
        self.clf = clf
        self.C = C
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.random_state = random_state

    # -- C-space bridge ----------------------------------------------------- #

    def config(self) -> dict[str, Any]:
        """The flat config for the active branch (inactive knobs omitted)."""
        if self.clf not in CLF_GRID:
            raise ValueError(f"Unknown classifier family: {self.clf!r}")
        cfg: dict[str, Any] = {"prep": self.prep, "reduce": self.reduce, "clf": self.clf}
        for name in CLF_GRID[self.clf]:
            cfg[name] = getattr(self, name)
        return cfg

    # -- estimator API ------------------------------------------------------ #

    def fit(self, X, y=None):
        self.pipeline_ = to_sklearn(self.config(), random_state=self.random_state)
        self.pipeline_.fit(X, y)
        self.classes_ = self.pipeline_.classes_
        self.n_features_in_ = getattr(self.pipeline_, "n_features_in_", None)
        return self

    def predict(self, X):
        return self.pipeline_.predict(X)

    def predict_proba(self, X):
        return self.pipeline_.predict_proba(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags.multi_class = True
        return tags


class MixedCategoricalSurface(RegressorMixin, BaseEstimator):
    """Response surface for ordinal-encoded categorical spaces.

    :param categorical: ``[(column_index, cardinality), ...]`` — which columns of
        the input vector are categorical and how many levels each has.
    :param embedding: ``"circular"`` (default, differentiable) or ``"onehot"``.
    :param regressor: inner scikit-learn regressor; default
        ``GaussianProcessRegressor()`` with stock RBF kernel.
    """

    def __init__(self, categorical: Sequence[tuple[int, int]] = (), embedding: str = "circular", regressor=None):
        self.categorical = categorical
        self.embedding = embedding
        self.regressor = regressor

    def _columns(self) -> list[tuple[int, int]]:
        return [(int(index), int(cardinality)) for index, cardinality in self.categorical]

    def _expand(self, X) -> np.ndarray:
        values = np.atleast_2d(np.asarray(X, dtype=float))
        columns = self._columns()
        categorical_indices = {index for index, _ in columns}
        # Keep only the genuinely numeric columns: a categorical column must be
        # *replaced* by its embedding, never kept alongside it, or the ordinal
        # signal leaks back in and the embedding changes nothing.
        numeric_indices = [i for i in range(values.shape[1]) if i not in categorical_indices]
        blocks = [values[:, numeric_indices]] if numeric_indices else []
        for index, cardinality in columns:
            if cardinality < 2:
                continue
            if self.embedding == "circular":
                # NOTE: no rounding here. Rounding first would make the surface
                # piecewise constant and defeat the purpose — the local optimiser
                # proposes continuous coordinates and needs a slope. Level j sits
                # at angle 2*pi*j/k; the map is continuous in the raw coordinate,
                # so the surface interpolates between categories.
                level = np.clip(values[:, index], 0.0, float(cardinality))
                angle = 2.0 * np.pi * level / float(cardinality)
                blocks.append(np.column_stack([np.cos(angle), np.sin(angle)]))
            elif self.embedding == "onehot":
                # One-hot has to round: it is the piecewise-constant reference.
                levels = np.clip(np.rint(values[:, index]).astype(int), 0, cardinality - 1)
                onehot = np.zeros((values.shape[0], cardinality))
                onehot[np.arange(values.shape[0]), levels] = 1.0
                blocks.append(onehot)
            else:
                raise ValueError(f"Unknown embedding: {self.embedding!r}")
        if not blocks:
            raise ValueError("MixedCategoricalSurface received no usable columns")
        return np.hstack(blocks)

    def fit(self, X, y):
        if self.regressor is None:
            from sklearn.gaussian_process import GaussianProcessRegressor

            self.regressor_ = GaussianProcessRegressor()
        else:
            self.regressor_ = clone(self.regressor)
        self.regressor_.fit(self._expand(X), np.asarray(y, dtype=float))
        return self

    def predict(self, X) -> np.ndarray:
        return self.regressor_.predict(self._expand(X))


@contextlib.contextmanager
def pin_threads(threads: int = 1):
    """Force single-threaded BLAS/OpenMP for the duration of a fit.

    Wall-clock is a primary axis in Protocol A; SVC and LogisticRegression scale
    with BLAS threads while tree ensembles largely do not, so an unpinned run
    silently rewards the linear/SVM branches. Environment variables must be set
    before the numerical libraries initialise their pools, hence the belt-and-
    braces ``threadpoolctl`` usage in the runners.
    """
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = str(threads)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
