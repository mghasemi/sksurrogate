"""Frozen pipeline C-space for the pipeline-space benchmark (DESIGN-cspace.md).

One source of truth for six heterogeneous search arms.  The space is
**enumerable** — 4 preprocessors x 5 reducers x 61 classifier configurations =
1220 distinct pipelines — which is what makes an exact oracle possible and
keeps "space search" honest (no method is credited for exploring pipelines the
space does not contain).

Adapters
--------
``to_sklearn(cfg)``      build the concrete pipeline.
``to_srs_space()``       flat 11-coordinate space for :class:`SurrogateRandomCV`
                         (SKSurrogate encodes categoricals *ordinally*, so the
                         estimator identity becomes a coordinate — the object
                         under test, see DESIGN-cspace.md).
``srs_conditional()``    branch gating.  Note the library keeps a coordinate for
                         a masked parameter and only drops it from
                         ``set_params`` (structsearch.py ``_apply_conditional_parameters``),
                         so SRS still pays to explore it; that cost is measured
                         as ``invalid_rate``.
``sample_uniform(rng)``  uniform draw of a *valid* config (random-search arm).
``optuna_suggest(t)``    native conditional branching (TPE arm).

Design notes
------------
* "No transform" is ``FunctionTransformer()`` (identity), never sklearn's
  ``'passthrough'`` sentinel: ``SurrogateRandomCV._zero_feature_reason`` probes
  each intermediate step with ``.fit_transform`` and treats a sentinel string
  as a preprocessing failure.
* ``max_depth = 0`` means ``None`` (unbounded), because the MRS flat space needs
  a numeric coordinate.
* PCA components stop at 16: the smallest dataset has 20 features, so a larger
  component count would be a guaranteed fit failure rather than a search
  decision.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Iterable

import numpy as np

__all__ = [
    "PREP_OPTIONS",
    "REDUCE_OPTIONS",
    "CLF_GRID",
    "PIPELINE_OPTIONS",
    "CARDINALITY",
    "enumerate_space",
    "is_valid",
    "normalise",
    "to_sklearn",
    "to_srs_space",
    "srs_conditional",
    "sample_uniform",
    "optuna_suggest",
    "srs_categorical_columns",
    "canonical_key",
]

PREP_OPTIONS: list[str] = ["identity", "standard", "robust", "quantile"]
REDUCE_OPTIONS: list[str] = ["identity", "pca4", "pca8", "pca16", "kbest16"]

#: Per-family knob grids. Cardinalities: 5 + 12 + 12 + 9 + 15 + 8 = 61.
#:
#: ``C`` is shared by ``logreg`` and ``svc``. Both branches must declare the *same*
#: value set: the flat SRS space has one ``C`` coordinate with the union grid, so a
#: branch that excludes a value would turn "the sampler drew 100.0 for logreg" into an
#: invalid candidate. That mismatch would be an artifact of the encoding, not the
#: effect under test — validity depends on *which* knob is active, never on which
#: value it holds.
CLF_GRID: dict[str, dict[str, list[Any]]] = {
    "logreg": {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "rf": {"max_depth": [4, 8, 16, 0], "min_samples_leaf": [1, 5, 20]},
    "et": {"max_depth": [4, 8, 16, 0], "min_samples_leaf": [1, 5, 20]},
    "hgb": {"learning_rate": [0.03, 0.1, 0.3], "max_leaf_nodes": [15, 31, 63]},
    "svc": {"C": [0.01, 0.1, 1.0, 10.0, 100.0], "gamma": ["scale", 0.01, 0.1]},
    "knn": {"n_neighbors": [1, 5, 15, 35], "weights": ["uniform", "distance"]},
}

PIPELINE_OPTIONS: dict[str, Any] = {
    "prep": PREP_OPTIONS,
    "reduce": REDUCE_OPTIONS,
    "clf": list(CLF_GRID),
}

_N_CLF_CONFIGS = sum(
    int(np.prod([len(v) for v in knobs.values()])) for knobs in CLF_GRID.values()
)

#: 4 x 5 x 61
CARDINALITY: int = len(PREP_OPTIONS) * len(REDUCE_OPTIONS) * _N_CLF_CONFIGS

#: Keys that are not classifier knobs.
_STRUCTURAL = ("prep", "reduce", "clf")

#: Union of every knob name, with the grid each one may take.  A flat space needs
#: one coordinate per knob across all families; only the active branch's knobs
#: reach ``set_params``.
_KNOB_GRID: dict[str, list[Any]] = {}
for _knobs in CLF_GRID.values():
    for _name, _values in _knobs.items():
        merged = _KNOB_GRID.setdefault(_name, [])
        for _value in _values:
            if _value not in merged:
                merged.append(_value)


def _clf_configs() -> Iterable[dict[str, Any]]:
    """Every valid (clf, knob...) combination in a deterministic order."""
    for family, knobs in CLF_GRID.items():
        names = list(knobs)
        for combination in itertools.product(*(knobs[name] for name in names)):
            yield {"clf": family, **dict(zip(names, combination))}


def enumerate_space() -> list[dict[str, Any]]:
    """All 1140 configurations, deterministic order (prep -> reduce -> clf)."""
    space: list[dict[str, Any]] = []
    for prep in PREP_OPTIONS:
        for reduce in REDUCE_OPTIONS:
            for clf_cfg in _clf_configs():
                space.append({"prep": prep, "reduce": reduce, **clf_cfg})
    return space


def is_valid(cfg: dict[str, Any]) -> bool:
    """True iff ``cfg`` is one of the enumerated pipelines.

    Type-strict on purpose: ``True`` must not pass for ``1`` (a bare bool in a
    categorical grid would silently collide with an integer neighbour).
    """
    if not isinstance(cfg, dict):
        return False
    if set(cfg) - set(_STRUCTURAL) - set(_KNOB_GRID):
        return False
    if cfg.get("prep") not in PREP_OPTIONS or cfg.get("reduce") not in REDUCE_OPTIONS:
        return False
    family = cfg.get("clf")
    if family not in CLF_GRID:
        return False
    knobs = CLF_GRID[family]
    present = {name for name in cfg if name not in _STRUCTURAL}
    if present != set(knobs):
        return False
    for name, allowed in knobs.items():
        value = cfg[name]
        # Reject bool masquerading as an int (True == 1 in Python).
        if isinstance(value, bool):
            return False
        if value not in allowed:
            return False
        # ``in`` uses ==, so a float 1.0 would match the int 1; require same type
        # for numeric grids to keep the flat coordinate decode unambiguous.
        match = [v for v in allowed if v == value and type(v) is type(value)]
        if not match:
            return False
    return True


def normalise(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return ``cfg`` limited to its branch knobs (raises if invalid)."""
    if not is_valid(cfg):
        raise ValueError(f"Not a member of the frozen C-space: {cfg!r}")
    return dict(cfg)


def canonical_key(cfg: dict[str, Any]) -> tuple:
    """Hashable identity of a config (knob order independent)."""
    return tuple(sorted(cfg.items()))


# --------------------------------------------------------------------------- #
# sklearn construction
# --------------------------------------------------------------------------- #

def _prep(name: str):
    from sklearn.preprocessing import (
        FunctionTransformer,
        QuantileTransformer,
        RobustScaler,
        StandardScaler,
    )

    return {
        "identity": lambda: FunctionTransformer(),
        "standard": StandardScaler,
        "robust": RobustScaler,
        "quantile": lambda: QuantileTransformer(
            n_quantiles=1000, output_distribution="normal", random_state=0
        ),
    }[name]()


def _reduce(name: str):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import FunctionTransformer

    return {
        "identity": lambda: FunctionTransformer(),
        "pca4": lambda: PCA(n_components=4, random_state=0),
        "pca8": lambda: PCA(n_components=8, random_state=0),
        "pca16": lambda: PCA(n_components=16, random_state=0),
        "kbest16": lambda: SelectKBest(score_func=f_classif, k=16),
    }[name]()


def _clf(cfg: dict[str, Any], random_state: int):
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    family = cfg["clf"]
    if family == "logreg":
        return LogisticRegression(C=float(cfg["C"]), max_iter=2000, random_state=random_state)
    if family in ("rf", "et"):
        cls = RandomForestClassifier if family == "rf" else ExtraTreesClassifier
        depth = cfg["max_depth"]
        return cls(
            n_estimators=50,
            max_depth=None if int(depth) == 0 else int(depth),
            min_samples_leaf=int(cfg["min_samples_leaf"]),
            random_state=random_state,
            n_jobs=1,
        )
    if family == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=float(cfg["learning_rate"]),
            max_leaf_nodes=int(cfg["max_leaf_nodes"]),
            random_state=random_state,
        )
    if family == "svc":
        return SVC(
            C=float(cfg["C"]),
            gamma=cfg["gamma"],
            random_state=random_state,
        )
    if family == "knn":
        return KNeighborsClassifier(
            n_neighbors=int(cfg["n_neighbors"]),
            weights=cfg["weights"],
            n_jobs=1,
        )
    raise ValueError(f"Unknown classifier family: {family!r}")


def to_sklearn(cfg: dict[str, Any], random_state: int = 0):
    """Build the concrete pipeline for ``cfg``.

    Steps: ``impute -> prep -> reduce -> clf``.

    The leading ``SimpleImputer`` is **fixed**, not a search axis: arrhythmia (the
    hard 13-class dataset) carries 408 missing cells in 5 columns, one of them
    missing in 376/452 rows. Without a uniform imputer, NaN-intolerant estimators
    (LogReg, SVC, KNN, and anything downstream of PCA/SelectKBest) fail there and
    the arms' failure handling — not their search quality — would decide the
    comparison. ``add_indicator=True`` keeps the missingness pattern as signal,
    which is standard for this clinical dataset. Fitting inside the pipeline keeps
    it per-fold, so no leakage.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    cfg = normalise(cfg)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("prep", _prep(cfg["prep"])),
            ("reduce", _reduce(cfg["reduce"])),
            ("clf", _clf(cfg, random_state)),
        ]
    )


# --------------------------------------------------------------------------- #
# Adapters for the search arms
# --------------------------------------------------------------------------- #

def to_srs_space() -> dict[str, Any]:
    """Flat 11-coordinate space for :class:`SurrogateRandomCV`.

    Uses the library's :class:`Categorical` so the arm exercises exactly the
    documented API (list spaces are coerced to the same object).
    """
    from SKSurrogate import Categorical

    space = {
        "prep": Categorical(list(PREP_OPTIONS)),
        "reduce": Categorical(list(REDUCE_OPTIONS)),
        "clf": Categorical(list(CLF_GRID)),
    }
    for name, values in _KNOB_GRID.items():
        space[name] = Categorical(list(values))
    return space


def srs_categorical_columns() -> list[tuple[int, int]]:
    """``(column_index, cardinality)`` for every coordinate of the flat SRS space.

    Order matches :func:`to_srs_space` insertion order, which is also the order
    ``SurrogateRandomCV.fit`` walks ``params_list`` in — so column ``i`` of the
    surrogate's input vector is exactly ``items[i]`` of this list.
    """
    return [
        (index, len(params.items))
        for index, params in enumerate(to_srs_space().values())
    ]


def srs_conditional() -> dict[str, Callable[[dict[str, Any]], bool]]:
    """Branch rules for the flat SRS space: knob -> predicate on the config.

    A knob is active only for the families that declare it. ``C`` is shared by
    ``logreg`` and ``svc``; the rest map to a single family.
    """
    families: dict[str, set[str]] = {}
    for family, knobs in CLF_GRID.items():
        for name in knobs:
            families.setdefault(name, set()).add(family)
    return {
        name: (lambda params, allowed=frozenset(fams): params.get("clf") in allowed)
        for name, fams in families.items()
    }


def sample_uniform(rng: np.random.Generator) -> dict[str, Any]:
    """Draw one uniformly random *valid* config (random-search arm)."""
    family = list(CLF_GRID)[int(rng.integers(len(CLF_GRID)))]
    cfg: dict[str, Any] = {
        "prep": PREP_OPTIONS[int(rng.integers(len(PREP_OPTIONS)))],
        "reduce": REDUCE_OPTIONS[int(rng.integers(len(REDUCE_OPTIONS)))],
        "clf": family,
    }
    for name, values in CLF_GRID[family].items():
        cfg[name] = values[int(rng.integers(len(values)))]
    return cfg


def optuna_suggest(trial) -> dict[str, Any]:
    """Native conditional branching: only the active branch is suggested."""
    cfg: dict[str, Any] = {
        "prep": trial.suggest_categorical("prep", PREP_OPTIONS),
        "reduce": trial.suggest_categorical("reduce", REDUCE_OPTIONS),
        "clf": trial.suggest_categorical("clf", list(CLF_GRID)),
    }
    for name, values in CLF_GRID[cfg["clf"]].items():
        cfg[name] = trial.suggest_categorical(name, values)
    return cfg
