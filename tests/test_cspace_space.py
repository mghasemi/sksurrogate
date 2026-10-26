"""Frozen-spec tests for the pipeline C-space (benchmarks/DESIGN-cspace.md).

The space is the experimental contract: every arm must enumerate the *same*
1140 pipelines, and each enumerated config must carry exactly the knobs of its
active classifier branch. These tests are the executable form of that contract.
"""

from benchmarks.cspace import (
    CARDINALITY,
    CLF_GRID,
    PIPELINE_OPTIONS,
    enumerate_space,
    is_valid,
)


def test_cardinality_is_frozen():
    assert CARDINALITY == 4 * 5 * (4 + 12 + 12 + 9 + 12 + 8)
    assert CARDINALITY == 1140


def test_branch_cardinalities_match_the_spec():
    assert {name: _count(knobs) for name, knobs in CLF_GRID.items()} == {
        "logreg": 4,
        "rf": 12,
        "et": 12,
        "hgb": 9,
        "svc": 12,
        "knn": 8,
    }


def test_enumeration_is_complete_and_unique():
    space = enumerate_space()
    assert len(space) == CARDINALITY
    keys = {tuple(sorted(cfg.items())) for cfg in space}
    assert len(keys) == CARDINALITY


def test_every_config_is_valid_and_carries_only_its_branch_knobs():
    for cfg in enumerate_space():
        assert is_valid(cfg), cfg
        branch = cfg["clf"]
        knob_names = {name for name in cfg if name not in ("prep", "reduce", "clf")}
        assert knob_names == set(CLF_GRID[branch]), cfg


def test_to_sklearn_rejects_an_inactive_knob():
    import pytest

    from benchmarks.cspace import to_sklearn

    cfg = dict(enumerate_space()[0])
    assert not is_valid({**cfg, "gamma": "scale"})  # knob belongs to another branch
    with pytest.raises(ValueError):
        to_sklearn({**cfg, "gamma": "scale"})


def test_pipeline_options_cover_the_space():
    assert PIPELINE_OPTIONS["prep"] == ["identity", "standard", "robust", "quantile"]
    assert PIPELINE_OPTIONS["reduce"] == ["identity", "pca4", "pca8", "pca16", "kbest16"]
    assert list(PIPELINE_OPTIONS["clf"]) == list(CLF_GRID)


def _count(knobs):
    total = 1
    for values in knobs.values():
        total *= len(values)
    return total
