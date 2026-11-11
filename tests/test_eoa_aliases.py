from collections import OrderedDict

import numpy as np
import pytest

from SKSurrogate.aml import AML
from SKSurrogate.eoa import EOA


def _population():
    return [(str(index),) for index in range(10)]


def test_eoa_accepts_corrected_parameter_names(tmp_path):
    optimizer = EOA(
        population=_population(),
        fitness=lambda individuals: individuals,
        parents_prop=0.4,
        elites_prop=0.5,
        max_generations=7,
        check_point=f"{tmp_path}/",
    )

    assert optimizer.parents_porp == 0.4
    assert optimizer.num_parents == 4
    assert optimizer.elits_porp == 0.5
    assert optimizer.num_elites == 2
    assert optimizer.max_generations == 7


def test_eoa_keeps_legacy_parameter_names(tmp_path):
    with pytest.warns(DeprecationWarning) as captured:
        optimizer = EOA(
            population=_population(),
            fitness=lambda individuals: individuals,
            parents_porp=0.4,
            elits_porp=0.5,
            max_generation=7,
            num_parents=4,
            check_point=f"{tmp_path}/",
        )

    assert optimizer.parents_porp == 0.4
    assert optimizer.elits_porp == 0.5
    assert optimizer.max_generations == 7
    messages = [str(warning.message) for warning in captured]
    assert len(messages) == 3
    assert any("'parents_porp' is deprecated; use 'parents_prop'" in message for message in messages)
    assert any("'elits_porp' is deprecated; use 'elites_prop'" in message for message in messages)
    assert any("'max_generation' is deprecated; use 'max_generations'" in message for message in messages)


def test_eoa_rejects_conflicting_aliases(tmp_path):
    with pytest.raises(TypeError, match="parents_prop"):
        EOA(
            population=_population(),
            fitness=lambda individuals: individuals,
            parents_prop=0.4,
            parents_porp=0.3,
            check_point=f"{tmp_path}/",
        )


def test_aml_parent_proportion_alias_is_not_overridden_by_default(monkeypatch, tmp_path):
    captured = {}

    class CaptureEOA:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __call__(self):
            return self

    monkeypatch.setattr("SKSurrogate.eoa.EOA", CaptureEOA)
    automl = AML(
        config={"sklearn.dummy.DummyClassifier": {}},
        length=1,
        check_point=f"{tmp_path}/",
        verbose=0,
    )
    monkeypatch.setattr(
        automl,
        "get_top",
        lambda num=5: OrderedDict(
            [(("sklearn.dummy.DummyClassifier",), (object(), 1.0))]
        ),
    )

    automl.eoa_fit(
        np.array([[0.0], [1.0]]),
        np.array([0, 1]),
        parents_prop=0.4,
        max_generations=7,
    )

    assert "num_parents" not in captured
    assert captured["parents_prop"] == 0.4
    assert captured["max_generations"] == 7
