"""Tests for the benchmark-side adapters (benchmarks/pipelines.py).

Each test pins one workaround for a verified property of ``SurrogateRandomCV``:
the union-parameter contract, the zero-feature probe's treatment of sklearn
sentinels, and the ordinal-encoding hypothesis that motivates the H2 arm.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split

from benchmarks.cspace import CLF_GRID, enumerate_space, is_valid, srs_categorical_columns
from benchmarks.pipelines import MixedCategoricalSurface, PipelineFactory

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


def test_factory_exposes_the_union_and_survives_clone():
    factory = PipelineFactory()
    names = set(factory.get_params(deep=True))
    # Every flat coordinate of the SRS space must be a valid ``set_params`` key,
    # or SurrogateRandomCV._validate_search_space rejects the search outright.
    assert {"prep", "reduce", "clf", "C", "max_depth", "min_samples_leaf",
            "learning_rate", "max_leaf_nodes", "gamma", "n_neighbors", "weights"} <= names
    assert clone(factory).get_params()["clf"] == factory.get_params()["clf"]


def test_factory_config_round_trips_through_set_params():
    for cfg in enumerate_space()[::89]:
        factory = PipelineFactory().set_params(**cfg)
        assert factory.config() == cfg
        assert is_valid(factory.config())


def test_factory_builds_and_fits_a_spread_of_enumerated_pipelines():
    for cfg in enumerate_space()[::97]:
        model = PipelineFactory().set_params(**cfg).fit(Xtr, ytr)
        assert model.predict(Xte).shape == (Xte.shape[0],)
        assert set(np.unique(model.classes_)) == set(np.unique(ytr))


def test_factory_ignores_knobs_of_inactive_branches():
    # An SRS draw can carry knobs of a branch that is not active; the factory must
    # build the active branch and drop the rest (conditional masking happens
    # upstream, but a direct set_params must not explode).
    factory = PipelineFactory().set_params(
        clf="knn", n_neighbors=5, weights="distance", C=100.0, gamma=0.1, max_depth=8
    )
    factory.fit(Xtr, ytr)
    assert factory.config() == {
        "prep": "identity", "reduce": "identity",
        "clf": "knn", "n_neighbors": 5, "weights": "distance",
    }


def test_identity_transform_is_not_rejected_as_zero_feature(tmp_path, monkeypatch):
    """A 5-evaluation SRS run must never reject identity-transformed pipelines.

    ``_zero_feature_reason`` probes intermediate steps with ``.fit_transform`` /
    ``.fit``; sklearn's ``'passthrough'`` sentinel would fail that probe, which is
    why the space uses ``FunctionTransformer()`` instead.
    """
    from SKSurrogate import SurrogateRandomCV
    from benchmarks.cspace import srs_conditional, to_srs_space

    monkeypatch.chdir(tmp_path)  # SRS checkpoints <task_name>.pkl into the CWD
    search = SurrogateRandomCV(
        PipelineFactory(),
        to_srs_space(),
        max_iter=8,
        min_evals=4,
        max_evals=5,
        cv=3,
        n_jobs=1,
        refit=False,
        random_state=0,
        regressor=GaussianProcessRegressor(),
        conditional=srs_conditional(),
        warm_start=False,
        Continue=False,
        task_name="test_cspace_identity",
    )
    search.fit(Xtr, ytr)

    errors = [entry.get("error") or "" for entry in search.evaluation_history_]
    assert not any("pipeline preprocessing failed" in message for message in errors), errors
    assert search.evaluation_history_, "no evaluations recorded"
    assert all(entry["status"] == "complete" for entry in search.evaluation_history_)


def _ordinal_gp_and_surface():
    # Column 0 is categorical with four levels; only level 3 is "good".
    Xd = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1]], dtype=float)
    yd = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    plain = GaussianProcessRegressor().fit(Xd, yd)
    surface = MixedCategoricalSurface(
        categorical=[(0, 4)], embedding="circular", regressor=GaussianProcessRegressor()
    ).fit(Xd, yd)
    return plain, surface


def test_mixed_surface_does_not_extrapolate_along_the_ordinal_axis():
    """Level 3 and level 0 are one step apart on the circle but 3 apart on the line."""
    plain, surface = _ordinal_gp_and_surface()
    probe = np.array([[3.5, 0.0]])             # between the last and first category
    ordinal = float(plain.predict(probe)[0])   # ordinal GP keeps climbing past level 3 (>1.0)
    cyclic = float(surface.predict(probe)[0])  # interpolates between level 3 and level 0 (~0.59)
    assert ordinal > 1.05                      # overshoots the observed maximum y = 1.0
    assert 0.3 < cyclic < 0.8                  # sits between the two categories
    assert ordinal - cyclic > 0.3


def test_mixed_surface_distance_depends_on_category_not_on_index_gap():
    """Equidistant categories: |1-0| and |2-1| are the same distance in the embedding."""
    _, surface = _ordinal_gp_and_surface()
    emb = surface._expand(np.array([[0.0, 0], [1.0, 0], [2.0, 0], [3.0, 0]]))
    circle = emb[:, -2:]  # one categorical column -> the embedding is the last two columns
    d01 = np.linalg.norm(circle[0] - circle[1])
    d12 = np.linalg.norm(circle[1] - circle[2])
    assert d01 == pytest.approx(d12)
    # ... while the ordinal distance between 0 and 2 is twice that, and the wrap
    # (0 -> 3) is one step: on the circle the last category is a neighbour of the first.
    d02 = np.linalg.norm(circle[0] - circle[2])
    d03 = np.linalg.norm(circle[0] - circle[3])
    assert d02 == pytest.approx(2.0)
    assert d03 == pytest.approx(d01)


def test_circular_embedding_is_smooth_and_onehot_is_not():
    """The H2 arm needs a differentiable surface, else SLSQP sees zero gradient."""
    _, circular = _ordinal_gp_and_surface()
    onehot = MixedCategoricalSurface([(0, 4)], "onehot").fit(
        np.array([[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1]], float),
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
    )

    eps = 1e-3

    def slope(surface, x):
        return abs(float(surface.predict([[x, 0]])[0]) - float(surface.predict([[x + eps, 0]])[0]))

    # Away from a rounding boundary (0.6 is not a half-integer).
    assert slope(circular, 0.6) > 0                       # a finite-difference slope exists
    assert slope(onehot, 0.6) == pytest.approx(0.0, abs=1e-12)  # piecewise constant


def test_srs_categorical_columns_match_the_flat_space_order():
    columns = srs_categorical_columns()
    assert [index for index, _ in columns] == list(range(len(columns)))
    cardinalities = dict(
        zip(
            ["prep", "reduce", "clf", "C", "max_depth", "min_samples_leaf",
             "learning_rate", "max_leaf_nodes", "gamma", "n_neighbors", "weights"],
            [cardinality for _, cardinality in columns],
        )
    )
    assert cardinalities["prep"] == 4
    assert cardinalities["reduce"] == 5
    assert cardinalities["clf"] == len(CLF_GRID)
    assert cardinalities["weights"] == 2
