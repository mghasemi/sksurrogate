"""Logging and per-trial telemetry contracts for the MIT core.

Covers the stdlib-logging fallback (:mod:`SKSurrogate._logutil`), the surrogate
fit score, sampling-source tags, the ``callbacks`` hook on
:class:`~SKSurrogate.structsearch.SurrogateRandomCV`, and ``to_audit_dict()``.
"""

import io
import json
import logging

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge

from SKSurrogate.structsearch import SurrogateRandomCV

SAMPLING_VOCABULARY = {"init", "explore", "surrogate_exploit"}


def _data():
    rng = np.random.RandomState(0)
    X = rng.rand(40, 3)
    return X, X.sum(axis=1)


def _search(tmp_path, *, verbose=0, callbacks=None, regressor=None):
    return SurrogateRandomCV(
        Ridge(),
        params={"alpha": (0.1, 10.0), "solver": ["svd", "cholesky"]},
        max_iter=4,
        min_evals=2,
        n_jobs=1,
        verbose=verbose,
        random_state=0,
        regressor=LinearRegression() if regressor is None else regressor,
        callbacks=callbacks,
        task_name=str(tmp_path / "log-telemetry"),
    )


def _isolate_module_logger():
    """Detach the test session's root handlers from the module logger chain.

    pytest installs capture handlers on the root logger, which would otherwise
    count as "logging configured" and suppress the stdout fallback under test.
    """
    module_logger = logging.getLogger("SKSurrogate.structsearch")
    previous = module_logger.propagate
    module_logger.propagate = False
    return module_logger, previous


def test_console_output_is_preserved_when_logging_is_unconfigured(capsys, tmp_path):
    X, y = _data()
    module_logger, previous = _isolate_module_logger()
    try:
        _search(tmp_path, verbose=1).fit(X, y)
    finally:
        module_logger.propagate = previous

    out = capsys.readouterr().out
    lines = out.splitlines()
    assert "Iteration 0 / 4" in lines
    assert lines.count("End of iteration.") == 5
    # Byte-identical fallback: no logging prefix/level is added to the message.
    assert all(not line.startswith("INFO:") for line in lines)


def test_no_double_output_once_a_handler_is_attached(capsys, tmp_path):
    X, y = _data()
    module_logger = logging.getLogger("SKSurrogate.structsearch")
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    previous_level = module_logger.level
    module_logger.addHandler(handler)
    module_logger.setLevel(logging.INFO)
    try:
        _search(tmp_path, verbose=1).fit(X, y)
    finally:
        module_logger.removeHandler(handler)
        module_logger.setLevel(previous_level)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "End of iteration." in buffer.getvalue()


def test_fit_history_records_surrogate_fit_score(tmp_path):
    X, y = _data()
    search = _search(tmp_path).fit(X, y)

    history = search.surrogate_info_["fit_history"]
    assert history
    for record in history:
        assert set(record) >= {
            "iteration",
            "training_points",
            "fit_status",
            "optimization_status",
        }
        assert "fit_score" in record
        assert record["fit_score"] is None or isinstance(record["fit_score"], float)
    assert [record["fit_status"] for record in history][0] == "complete"


def test_sampling_sources_track_every_evaluation(tmp_path):
    X, y = _data()
    search = _search(tmp_path).fit(X, y)

    sources = search.OPTIM.sampling_sources_
    assert len(sources) == len(search.OPTIM.evaluated)
    assert set(sources) <= SAMPLING_VOCABULARY
    assert sources[0] == "init"


def test_callbacks_receive_one_event_per_trial_and_survive_failures(tmp_path):
    X, y = _data()
    events = []
    delivered_after_failure = []

    def collector(event):
        events.append(event)

    def broken(event):
        raise RuntimeError("intentional callback failure")

    def late_collector(event):
        delivered_after_failure.append(event)

    search = _search(tmp_path, callbacks=[collector, broken, late_collector]).fit(X, y)

    assert len(events) == len(search.evaluation_history_)
    assert [event["trial"] for event in events] == list(range(len(events)))
    first_record = search.evaluation_history_[0]
    assert set(events[0]) >= set(first_record) | {"trial", "phase"}
    assert all(
        event["phase"] in SAMPLING_VOCABULARY or event["phase"] is None for event in events
    )
    # The broken callback must not abort the search or starve later callbacks.
    assert len(delivered_after_failure) == len(events)
    assert search.best_params_

    # The callback hook is a public init parameter: get_params/clone stay valid.
    cloned = clone(search)
    assert isinstance(cloned, SurrogateRandomCV)
    assert cloned.get_params(deep=False)["callbacks"] is not None


def test_to_audit_dict_is_json_serialisable_and_complete(tmp_path):
    X, y = _data()
    search = _search(tmp_path).fit(X, y)

    audit = search.to_audit_dict()
    blob = json.dumps(audit)  # must succeed without a custom encoder
    assert json.loads(blob) == audit

    assert set(audit) >= {
        "schema_version",
        "estimator",
        "search_space",
        "scoring",
        "cv",
        "budget",
        "termination",
        "evaluations",
        "surrogate",
        "sampling_sources",
    }
    assert audit["schema_version"] == 1
    assert audit["estimator"]["class"].endswith("Ridge")
    assert set(audit["search_space"]) == {"alpha", "solver"}
    assert len(audit["evaluations"]) == len(search.evaluation_history_)
    assert [item["trial"] for item in audit["evaluations"]] == list(
        range(len(audit["evaluations"]))
    )
    assert len(audit["sampling_sources"]) == len(search.OPTIM.evaluated)


def test_to_audit_dict_works_before_fit(tmp_path):
    search = _search(tmp_path)

    audit = search.to_audit_dict()
    assert audit["evaluations"] == []
    assert audit["sampling_sources"] == []
    assert json.dumps(audit)
