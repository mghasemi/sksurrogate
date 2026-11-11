"""Regression tests for EOA checkpoint resume validation.

A checkpoint may have been written by a run with a different gene set (the
default AML configuration contains far more estimators than a user config) or
on another machine whose absolute checkpoint directory does not exist here.
Replaying it blindly restored incompatible parents and crashed on the first
evaluation, or adopted a dead path and crashed on the next save.
"""

import logging
import pickle
from pathlib import Path

from SKSurrogate.eoa import EOA


def _load(optimizer):
    """Invoke the private restore hook (name-mangled) for a direct unit test."""
    return getattr(optimizer, "_EOA__load")()


def _save(optimizer):
    return getattr(optimizer, "_EOA__save")()


def _optimizer(directory, genes=("a", "b")):
    return EOA(
        population=[(gene,) for gene in genes],
        fitness=lambda population: {gene: 0.0 for gene in population},
        check_point=str(directory) + "/",
        random_state=0,
    )


def _write_checkpoint(directory, *, genes, parents=None, children=None, check_point=None):
    info = dict(
        checkpoint_version=EOA.CHECKPOINT_VERSION,
        population_size=2,
        parents_porp=0.1,
        num_parents=2,
        elits_porp=0.2,
        num_elites=0,
        mutation_prob=0.05,
        max_generations=5,
        generation_num=2,
        genes=list(genes),
        init_genes=[],
        term_genes=list(genes),
        task_name="EOA",
        check_point=check_point or (str(directory) + "/"),
        random_state=0,
        evals={},
        parents=parents or {},
        children=children or {},
    )
    path = Path(directory) / "EOA.eoa"
    path.write_bytes(pickle.dumps(info))
    return path


def test_checkpoint_from_different_gene_set_is_ignored(tmp_path, caplog):
    _write_checkpoint(tmp_path, genes=["a", "c"], parents={("c",): 1.0})
    eoa = _optimizer(tmp_path)
    with caplog.at_level(logging.INFO, logger="SKSurrogate.eoa"):
        _load(eoa)
    assert eoa.genes == ["a", "b"]
    assert eoa.parents == {}
    assert any("different gene set" in record.getMessage() for record in caplog.records)


def test_missing_checkpoint_directory_keeps_current_check_point(tmp_path):
    _write_checkpoint(tmp_path, genes=["a", "b"], check_point="/nonexistent/elsewhere/")
    eoa = _optimizer(tmp_path)
    _load(eoa)
    assert eoa.check_point == str(tmp_path) + "/"
    # The next save must land in the caller's directory, not the dead path.
    _save(eoa)
    assert (tmp_path / "EOA.eoa").exists()


def test_valid_checkpoint_restores_state(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    _write_checkpoint(
        tmp_path,
        genes=["a", "b"],
        parents={("a",): 1.0},
        check_point=str(other) + "/",
    )
    eoa = _optimizer(tmp_path)
    _load(eoa)
    assert eoa.check_point == str(other) + "/"
    assert eoa.generation_num == 2
    assert eoa.parents == {("a",): 1.0}
