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

import numpy as np


def test_cardinality_is_frozen():
    assert CARDINALITY == 4 * 5 * (5 + 12 + 12 + 9 + 15 + 8)
    assert CARDINALITY == 1220


def test_branch_cardinalities_match_the_spec():
    assert {name: _count(knobs) for name, knobs in CLF_GRID.items()} == {
        "logreg": 5,
        "rf": 12,
        "et": 12,
        "hgb": 9,
        "svc": 15,
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


def test_srs_space_and_conditionals_generate_only_valid_configs():
    """G1: any uniform draw is a member of the enumerated space."""
    from benchmarks.cspace import sample_uniform

    rng = np.random.default_rng(0)
    seen_families = set()
    for _ in range(1000):
        cfg = sample_uniform(rng)
        assert is_valid(cfg)
        seen_families.add(cfg["clf"])
    assert seen_families == set(CLF_GRID), "uniform sampling does not reach every family"


def test_srs_masked_params_build_a_valid_config():
    """G1: every flat-space draw, after conditional masking, is a valid config.

    Replicates what ``SurrogateRandomCV`` does — decode the ordinal coordinate to
    an item, then drop knobs whose rule does not match — over the whole flat space.
    """
    from benchmarks.cspace import srs_conditional, to_srs_space

    space = to_srs_space()
    rules = srs_conditional()
    rng = np.random.default_rng(1)
    for _ in range(500):
        drawn = {name: params.items[int(rng.integers(len(params.items)))]
                 for name, params in space.items()}
        masked = {name: value for name, value in drawn.items()
                  if name not in rules or rules[name](drawn)}
        assert is_valid(masked), masked


def test_every_branch_knob_has_a_conditional_rule():
    from benchmarks.cspace import srs_conditional, to_srs_space

    rules = srs_conditional()
    knobs = {name for name in to_srs_space() if name not in ("prep", "reduce", "clf")}
    assert set(rules) == knobs


def test_conditionals_activate_exactly_the_declared_branch_knobs():
    from benchmarks.cspace import srs_conditional

    rules = srs_conditional()
    for family, knobs in CLF_GRID.items():
        active = {name for name, rule in rules.items() if rule({"clf": family})}
        assert active == set(knobs), family
