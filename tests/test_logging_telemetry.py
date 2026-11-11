"""Logging and per-trial telemetry contracts for the MIT core.

Covers the stdlib-logging fallback (:mod:`SKSurrogate._logutil`), the surrogate
fit score, sampling-source tags, the ``callbacks`` hook on
:class:`~SKSurrogate.structsearch.SurrogateRandomCV` (including forwarding
through :class:`~SKSurrogate.aml.AML`), and the JSON-safe audit exports:
schema-version-1 search bundles from ``to_audit_dict()`` and the
schema-version-2 bundle from ``AML.to_audit_dict()``.
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


def _aml(tmp_path, *, verbose=0, callbacks=None):
    from SKSurrogate.aml import AML
    from SKSurrogate.structsearch import BoxSample

    # Minimal legacy configuration: one candidate pipeline sequence, one cheap
    # surrogate stage with a small evaluation budget — fast enough to fit in a
    # unit test while exercising the real optimize_pipeline/_cast path.
    return AML(
        config={"sklearn.linear_model.LogisticRegression": {"C": (0.1, 5.0)}},
        length=1,
        min_random_evals=3,
        max_evals=4,
        cv=2,
        n_jobs=1,
        random_state=0,
        surrogates=[(LinearRegression(), 6, BoxSample, "L-BFGS-B")],
        check_point=str(tmp_path / "aml-audit"),
        verbose=verbose,
        callbacks=callbacks,
    )


def test_aml_to_audit_dict_is_json_serialisable_and_complete(tmp_path):
    X = np.random.RandomState(0).rand(30, 4)
    y = (X.sum(axis=1) > 2.0).astype(int)
    events = []
    aml = _aml(tmp_path, callbacks=[events.append]).fit(X, y)

    audit = aml.to_audit_dict()
    blob = json.dumps(audit)  # must succeed without a custom encoder
    assert json.loads(blob) == audit

    assert set(audit) >= {
        "schema_version",
        "object",
        "estimator",
        "preset",
        "scoring",
        "cv",
        "budget",
        "termination",
        "candidates",
        "best_candidate",
        "sampling_sources",
    }
    assert audit["schema_version"] == 2
    assert audit["object"] == "aml"
    assert audit["estimator"]["class"].endswith(".AML")
    assert audit["preset"] is None
    assert audit["termination"]["status"] == "completed"

    # One candidate per evaluated pipeline sequence, in history order.
    candidates = audit["candidates"]
    assert len(candidates) == len(aml.evaluation_history_)
    assert [item["candidate"] for item in candidates] == list(range(len(candidates)))
    for record in candidates:
        assert set(record) >= {"pipeline", "score", "rule_rejected", "errors", "stages"}
        # Each surrogate stage embeds its own schema-version-1 search bundle.
        for stage in record["stages"]:
            assert stage["schema_version"] == 1
            assert len(stage["sampling_sources"]) == len(stage["evaluations"])

    # The AML-level tag list aggregates every stage's per-evaluation tags, and
    # the budget totals aggregate across all stages of all candidates.
    aggregated = {}
    total_evals = 0
    for record in candidates:
        for stage in record["stages"]:
            total_evals += len(stage["evaluations"])
            for tag in stage["sampling_sources"]:
                aggregated[tag] = aggregated.get(tag, 0) + 1
    assert audit["sampling_sources"] == aggregated
    assert set(audit["sampling_sources"]) <= SAMPLING_VOCABULARY
    assert audit["budget"]["candidates"] == len(candidates)
    assert audit["budget"]["evaluations"] == total_evals

    # Callback forwarding: AML passes its callbacks through to every stage, so
    # one trial event is delivered per evaluation of every surrogate search.
    assert len(events) == total_evals
    for event in events:
        assert {"trial", "phase"} <= set(event)
        assert event["phase"] in SAMPLING_VOCABULARY or event["phase"] is None

    # The best candidate matches the fitted result.
    assert aml._best_sequence_ is not None
    assert audit["best_candidate"]["score"] == aml.best_estimator_score
    assert audit["best_candidate"]["pipeline"] == list(aml._best_sequence_)


def test_aml_to_audit_dict_works_before_fit(tmp_path):
    aml = _aml(tmp_path)

    audit = aml.to_audit_dict()
    assert audit["candidates"] == []
    assert audit["sampling_sources"] == {}
    assert audit["best_candidate"] is None
    assert audit["termination"]["status"] is None
    blob = json.dumps(audit)
    assert json.loads(blob) == audit


def test_warning_level_is_not_duplicated_on_stdout(capsys):
    from SKSurrogate import _logutil

    logger = logging.getLogger(__name__)
    previous = logger.propagate
    logger.propagate = False
    try:
        _logutil.emit(logging.WARNING, "warning probe")
    finally:
        logger.propagate = previous

    captured = capsys.readouterr()
    # lastResort handles stderr; the stdout fallback must not repeat the line.
    assert "warning probe" in captured.err
    assert captured.out == ""


class _BrokenRegressor:
    def fit(self, X, y):
        raise RuntimeError("intentional surrogate fit failure")

    def predict(self, X):
        return np.zeros(len(X))


def test_surrogate_fit_failure_is_logged_as_warning_before_raising(tmp_path, caplog):
    X, y = _data()
    search = SurrogateRandomCV(
        Ridge(),
        params={"alpha": (0.1, 10.0)},
        max_iter=3,
        min_evals=1,
        n_jobs=1,
        random_state=0,
        regressor=_BrokenRegressor(),
        task_name=str(tmp_path / "warning-fit"),
    )

    with caplog.at_level(logging.WARNING, logger="SKSurrogate.structsearch"):
        with pytest.raises(RuntimeError, match="intentional surrogate fit failure"):
            search.fit(X, y)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Surrogate fit failed" in record.getMessage() for record in warnings)


def test_no_progress_termination_is_logged_as_warning(tmp_path, caplog):
    from SKSurrogate.structsearch import SurrogateSearch

    search = SurrogateSearch(
        lambda point: point[0] ** 2,
        x0=[0.5],
        bounds=[(-1.0, 1.0)],
        max_iter=6,
        min_evals=100,  # never fit a surrogate -> every iteration is a direct sample
        max_itr_no_prog=0,
        task_name=str(tmp_path / "warning-no-progress"),
        random_state=0,
    )

    with caplog.at_level(logging.WARNING, logger="SKSurrogate.structsearch"):
        search()

    assert search.termination_reason == "max_iter_no_progress"
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("without improvement" in message for message in messages)
