===================================
Logging and Telemetry
===================================

The optimization core reports progress through the Python standard library
``logging`` module and exposes per-trial data through plain dictionaries.
Neither facility adds a dependency, configures global logging state, or
changes any existing result key of a search object.

Logging
=======

Every core module emits on a logger in the ``SKSurrogate`` namespace:

.. list-table::
   :header-rows: 1

   * - Logger
     - Source
   * - ``SKSurrogate.structsearch``
     - ``SurrogateSearch`` / ``SurrogateRandomCV``
   * - ``SKSurrogate.eoa``
     - ``EOA`` evolutionary operators
   * - ``SKSurrogate.aml``
     - ``AML`` pipeline search

Record levels follow the existing ``verbose`` settings of each class, which keep
their meaning:

* ``verbose > 0`` messages are emitted at :data:`logging.INFO`
  (iteration start, ``"End of iteration."``, new best candidate, chosen
  pipelines, ``AML`` scores);
* ``verbose > 1`` messages are emitted at :data:`logging.DEBUG`
  (surrogate fit diagnostics, sampled points, local-optimization status,
  separators);
* ``EOA`` prints its per-generation line unconditionally and therefore emits it
  at :data:`logging.INFO` without a ``verbose`` gate.

Severity beyond progress output is used deliberately, so a configured handler or
the control-plane log view can separate routine noise from problems:

.. list-table::
   :header-rows: 1

   * - Level
     - Emitted for
   * - ``INFO``
     - iteration progress, new best candidate, chosen pipelines, job lifecycle
   * - ``DEBUG``
     - surrogate fit diagnostics, sampled points, local-optimization status,
       pruned or repeatedly failing trials
   * - ``WARNING``
     - a surrogate fit that failed (the exception is logged before it is
       re-raised), termination after too many iterations without improvement,
       and one aggregated line listing surrogate failure counts at the end of a
       search
   * - ``ERROR``
     - a fitted surrogate producing a non-finite prediction, an ``AML`` search
       that could not fit any candidate, a failed user callback, and a failed
       background job (with the traceback)
   * - ``CRITICAL``
     - reserved; the library does not declare catastrophic state on its own

Unconfigured logging keeps the old console behaviour. When no logger in the
emitting logger's propagation chain has a handler, the message is also written
to ``sys.stdout`` — byte-identical to the former ``print()`` output. Records at
WARNING or above are the exception: the standard library's ``logging.lastResort``
already writes them to ``stderr``, so ``emit`` skips the stdout copy for them and
nothing is duplicated. As soon as a handler is configured anywhere in that
chain, the stdout copy stops entirely, so a configured pipeline sees exactly one
copy of each message::

    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("SKSurrogate").setLevel(logging.INFO)

    search = SurrogateRandomCV(Ridge(), {"alpha": (1e-3, 10.0)}, verbose=1)
    search.fit(X, y)   # records go to the configured handler, nothing to stdout

Attaching a handler to ``SKSurrogate`` (as above) or to an individual module
logger both work: messages propagate upward from ``SKSurrogate.structsearch`` to
``SKSurrogate`` and then to the root logger.

.. note::
    Records arrive pre-formatted as text. Structured consumers should use the
    per-trial callback described below rather than parsing log lines.

.. note::
    The control plane captures these records per background job — including the
    WARNING and ERROR lines above — and exposes them with severity filters in the
    web UI's **Logs** page. See :doc:`controlplane` for the endpoint contract and
    :doc:`ui-manual` for the page.

Per-trial telemetry
===================

Surrogate fit score
-------------------

Each completed entry of ``SurrogateSearch.surrogate_fit_history_`` (also
reachable as ``surrogate_info_["fit_history"]``) carries an additional
``fit_score`` key: the fitted regressor's own in-sample ``score`` on the data it
was fitted to. It measures how well the response surface explains the observed
scores so far, and is ``None`` when the regressor exposes no callable ``score``
or returns a non-finite value. All previously existing keys
(``iteration``, ``training_points``, ``fit_status``, ``optimization_status``,
and the failure keys) are unchanged.

Sampling-source tags
--------------------

``search.OPTIM.sampling_sources_`` is a list parallel to
``search.OPTIM.evaluated`` that records where each evaluation came from:

.. list-table::
   :header-rows: 1

   * - Tag
     - Origin
   * - ``init``
     - initial point(s) evaluated before the search loop
   * - ``explore``
     - direct random sample, taken while too few points exist to fit a usable
       surrogate
   * - ``surrogate_exploit``
     - candidate produced by optimizing the fitted surrogate

The tag list is appended in lockstep with ``evaluated``, so
``len(sampling_sources_) == len(evaluated)`` holds for any search built in this
release. It is stored additively in the ``continue_`` checkpoint; a checkpoint
written by an older release loads normally and yields an empty tag list rather
than invented provenance. The two-element shape of ``evaluated`` entries is
unchanged.

Callbacks
---------

``SurrogateRandomCV`` accepts ``callbacks=None``: a single callable, or an
iterable of callables. Each callback is invoked once per evaluated candidate
with one event dictionary::

    def on_trial(event):
        print(event["trial"], event["phase"], event["status"], event["score"])

    search = SurrogateRandomCV(
        RandomForestRegressor(), {"max_depth": (1, 30)}, callbacks=[on_trial]
    )
    search.fit(X, y)

An event is a copy of the trial record appended to ``evaluation_history_``
(``params``, ``score``, ``status``, ``duration``, ``fold_duration``, ``error``,
``partial_folds``, ``failure_records``) plus two keys:

* ``trial`` — the zero-based index of that record in ``evaluation_history_``;
* ``phase`` — the sampling tag of the evaluation that produced it
  (``"init"``, ``"explore"``, ``"surrogate_exploit"``), or ``None`` when the tag
  is not available.

Events are delivered at every record site, so a callback sees rejected
(``invalid``), pruned, and completed or failed trials, not only successful ones.

.. note::
    The very first trial of a search carries ``phase = None``: ``SurrogateSearch``
    evaluates its initial point while the estimator is still constructing the
    optimizer, before the phase attribute is published. That trial is still
    tagged ``"init"`` in ``sampling_sources_``.

Each callback invocation is isolated: if a callback raises, the exception is
logged on ``SKSurrogate.structsearch`` and the remaining callbacks and the
search itself continue. A broken callback can therefore never abort a run.
``callbacks`` is a normal init parameter, so ``get_params`` and ``clone`` keep
working.

Audit export
------------

``SurrogateRandomCV.to_audit_dict()`` packages the search into a JSON-safe
dictionary — the contract between this package and any control plane, notebook,
or archive::

    import json

    audit = search.to_audit_dict()
    payload = json.dumps(audit)   # no custom encoder required

The bundle contains:

.. list-table::
   :header-rows: 1

   * - Key
     - Content
   * - ``schema_version``
     - ``1``, the bundle schema version
   * - ``estimator``
     - ``{"class": module.qualname, "params": {...}}``
   * - ``search_space``
     - Per-parameter encoding: ``type``, ``bounds`` (already encoded for
       ``scale="log"``), ``scale``, or ``items`` for categorical parameters
   * - ``scoring``, ``cv``
     - Scoring/CV summary; objects are reduced to ``str``
   * - ``n_iter``, ``max_iter``, ``random_state``
     - Search-size and seed settings
   * - ``budget``
     - ``summary_["budget"]``: time/CPU/memory limits, elapsed seconds,
       iterations, evaluations
   * - ``termination``
     - ``termination_reason`` and ``status`` from ``summary_``
   * - ``evaluations``
     - Trial records, each augmented with its zero-based ``trial`` index
   * - ``surrogate``
     - ``surrogate_info_`` including ``fit_history``
   * - ``sampling_sources``
     - The optimizer's sampling tags, or ``[]``

Values that are not JSON-native (estimator objects, splitters, NumPy scalars,
non-finite floats) are converted to their string form or ``None``, so
``json.dumps`` succeeds without arguments. The method is also usable before
``fit``; the evaluation and sampling lists are then empty.

What stays out of the core
==========================

Persistent stores (JSONL/SQLite), MLOps connectors (MLflow/W&B), live
progress streaming into the web API, and ``AML``-level callbacks are not part of
this package. They are built on top of the contracts above — a callback that
writes events to disk, or a consumer that reads ``to_audit_dict()`` — so the core
keeps its standard-library-only footprint.
