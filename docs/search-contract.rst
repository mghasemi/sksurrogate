Search contract
===============

.. contents:: On this page
   :local:
   :depth: 2

Purpose
-------

This page is the *definition of done* for the optimizer. It states what a search
guarantees — how its budget is counted, which counter means what, when it stops, and
what ``random_state`` does and does not promise. It is written to be checkable: every
statement below is pinned by a test in ``tests/test_search_invariants.py`` or
``tests/test_search_determinism.py``, and the "Status" column records the current state
of the code against the contract.

Audience: users who need an auditable experiment, the control plane (which counts
evaluations for its own budget reporting), and the benchmark harness (which compares
search strategies under an equal evaluation budget).

The four counters
-----------------

A search consumes work in four distinct ways. They are counted separately because
conflating them is how an "equal-budget" comparison silently becomes unequal.

=========================.. list-table:: Search invariants
   :header-rows: 1
   :widths: 8 68 24

   * - Id
     - Invariant
     - Status
   * - I1
     - ``n_evaluations == len(evaluation_history_) == len(sampling_sources_)``
     - holds
   * - I2
     - Every recorded evaluation corresponds to exactly one objective call
     - holds
   * - I3
     - ``n_evaluations <= max_evals`` whenever ``max_evals`` is set
     - holds
   * - I4
     - With ``max_evals=None``, ``n_evaluations == n_iter`` exactly
     - fixed
   * - I5
     - A round that consumes no evaluation increments the no-progress counter
       (so it can never outlive its budget silently)
     - fixed
   * - I6
     - ``termination_reason`` names the constraint that actually bound the run,
       drawn from the documented vocabulary below
     - fixed
   * - I7
     - Identical ``random_state`` ⇒ identical ``(params, score)`` sequence, in
       one process and across processes
     - fixed
   * - I8
     - Fitting never consumes or reseeds the global ``random``/``numpy.random``
       state
     - holds
   * - I9
     - ``clone(search)`` before and after ``fit`` returns equal
       ``get_params(deep=True)``
     - fixed
   * - I10
     - A surrogate fit or prediction failure degrades that round to exploration;
       it never aborts the search
     - fixed

The failures above were the release-blocking defect class of v0.6.0: an off-by-one
between rounds and evaluations (``n_iter=3`` spent five and ``n_iter=5`` spent seven,
because the initial candidate was evaluated in the constructor *and* the loop ran
``MaxIter + 1`` rounds), an initial candidate drawn from Python's global ``random``
so a fixed ``random_state`` did not reproduce a run, failed rounds that consumed
iterations without evaluating or charging the no-progress counter, and a fit failure
that aborted a whole search. A library user can restore the old abort behaviour with
``surrogate_fit_error="raise"``.

Budget semantics
----------------

=============================  ======================================================
Parameter                      Meaning
=============================  ======================================================
``n_iter`` / ``max_iter``      Number of **candidate evaluations** to spend. The two
                               names are aliases (``n_iter`` is scikit-learn's
                               spelling); giving both with different values is an
                               error. The initial candidate counts toward it.
``max_evals``                  Hard cap on candidate evaluations. The effective budget
                               is ``min(n_iter, max_evals)``: a cap below ``n_iter``
                               binds first, a cap above it does not silently extend the
                               run.
``min_evals``                  Evaluations to collect before the response surface is
                               used. The default is the number of monomials of degree
                               ``deg`` in the search dimension, ``C(d + deg, d)``; an
                               explicit value is honoured as given and reported in
                               ``surrogate_info_["min_evals"]`` together with
                               ``min_evals_source`` (``default`` or ``user``).
``time_limit``                 Wall-clock seconds; checked between evaluations.
``max_itr_no_prog``            Consecutive rounds without a new best candidate before
                               stopping.
=============================  ======================================================

Precedence when several constraints fire at the same boundary:
``time_limit`` → ``max_evals`` → ``n_iter`` → ``max_itr_no_prog``.

Termination vocabulary
----------------------

==========================  ===========================================================
``termination_reason``      Condition
==========================  ===========================================================
``time_limit``              Wall-clock budget exhausted between evaluations.
``max_evals``               ``n_evaluations`` reached ``max_evals``.
``n_iter``                  ``n_evaluations`` reached ``n_iter``: normal completion.
                            ``completed`` is accepted as a legacy alias meaning the
                            same thing.
``max_iter_no_progress``    ``max_itr_no_prog`` consecutive rounds without a strictly
                            better best value.
``stopped``                 A caller-requested stop (``stop_check``, or a control-plane
                            pause/cancel) landed on an evaluation boundary. The search
                            stops cleanly; the exception is raised only after the
                            counters and reason are recorded.
==========================  ===========================================================

``termination_reason`` is reported per stage and per run: ``SurrogateRandomCV`` in
``termination_reason`` and ``summary_``, ``AML`` in ``termination_reason`` and
``to_audit_dict()``.

Randomness
----------

``random_state`` accepts ``int``, ``None``, ``numpy.random.Generator`` or
``numpy.random.RandomState``. It seeds two independent streams derived from one
:class:`numpy.random.SeedSequence`:

* ``init_candidate`` — the starting point inside the encoded search box;
* ``sampling`` — every subsequent sample (``CompactSample`` / ``BoxSample`` /
  ``SphereSample``) and the contraction jitter.

Guarantee: with the same ``random_state``, the same data, ``cv``, ``n_iter``,
``min_evals``, ``scipy_solver`` and the same library versions, two fits produce the
same candidate sequence and the same scores — in one process and across processes.
Fitting never reads or writes the global ``random`` / ``numpy.random`` state.

Not guaranteed, by design:

* the **estimator's** own randomness — pass ``estimator__random_state`` (or set it on
  the estimator) if the estimator is stochastic; the search does not seed it;
* ordering of per-fold work under ``n_jobs != 1`` (scores are order-independent, only
  timing and log interleaving differ);
* reproducibility across scikit-learn / NumPy versions.

Post-conditions on the fitted search
------------------------------------

``SurrogateRandomCV.fit`` returns ``self`` with, at minimum:

``best_params_``, ``best_estimator_`` (when ``refit`` is not ``False``),
``best_score_``, ``best_index_``, ``cv_results_``, ``evaluation_history_``,
``surrogate_info_``, ``surrogate_fit_history_``, ``summary_``, ``termination_reason``,
``n_evaluations_``, ``n_rounds_``, ``failure_summary_``, plus the scikit-learn search
attributes ``scorer_`` and ``n_splits_``.

The effective configuration is never implicit: ``surrogate_info_`` records the
response-surface class and its source (``default`` / ``user``), the effective
``min_evals`` and its source, and the fit/prediction/optimization failure counts, so
``to_audit_dict()`` is a complete account of what the search actually did.
