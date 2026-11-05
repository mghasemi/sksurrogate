===================
Experiment design
===================

SKSurrogate provides a bounded starting point for continuous-domain design
and sequential candidate recommendation. It proposes points only; it does not
evaluate an objective, contact instruments, or execute experiments.

Latin-hypercube initialization
==============================

``latin_hypercube`` creates a seeded design with one point in every equal-width
stratum of each numeric dimension::

    from SKSurrogate import latin_hypercube

    bounds = [(0.0, 10.0), (1.0, 5.0)]
    initial = latin_hypercube(bounds, n_samples=12, random_state=7)

The result is a NumPy array with shape ``(n_samples, n_dimensions)``. Bounds
must be finite, strictly increasing numeric pairs. The current implementation
supports at most 100 dimensions and 10,000 rows.

Sequential recommendation
=========================

After observing a scalar objective to minimize, ``suggest_next_point`` fits a
Gaussian-process surrogate and ranks a bounded seeded Latin-hypercube candidate
pool by the lower confidence bound (LCB): predicted mean minus ``beta`` times
predicted standard deviation. The GP uses a Matern 5/2 kernel with fixed
unit length scales after inputs are normalized to their bounds; it does not
optimize kernel hyperparameters::

    from SKSurrogate import suggest_next_point

    proposal = suggest_next_point(
        observed_parameters,
        observed_objectives,
        bounds,
        n_candidates=1000,
        beta=2.0,
        random_state=7,
    )
    next_parameters = proposal["candidate"]

The returned dictionary contains the proposed point, surrogate mean and
standard deviation, acquisition value, observation and candidate-pool counts,
kernel name and length scales, seed, direction, and a limitation statement. At
most 500 observations, 100 dimensions, and 10,000 candidates are accepted.
Candidate selection is not an objective evaluation; the caller decides whether
and how to measure a proposed point.

Scope and limitations
=====================

This prototype supports finite continuous boxes and a single scalar minimization
objective. It does not yet support categorical or discrete factors, constraints,
noise models, multi-objective acquisition, adaptive hyperparameter tuning, or
Fisher-information/response-surface designs. GP-LCB is a heuristic; predicted
uncertainty is model-dependent, and recommendations can be poor under
misspecification or outside observed regimes. The fixed length scale avoids
unstable hyperparameter optimization but is a modeling assumption, not a
universally appropriate setting. Users must independently enforce physical,
ethical, and operational constraints and obtain appropriate expert review
before any real-world experiment.

The synthetic comparison against uniform random sampling is reproducible with::

    python -m benchmarks.doe_active_learning --smoke
    python -m benchmarks.doe_active_learning

It compares initial GP prediction error and equal-objective-budget best-so-far
regret on three two-dimensional synthetic functions, as well as a real-data
model-tuning tasks. Smoke mode reduces the seed counts and evaluation budgets.
The fixed GP length scale was selected after exploratory warning and
performance probes on the synthetic and diabetes tasks below, so those results
are development evidence, not independent validation. The breast-cancer task
is a separate validation workload. None of these results establish that the
policy improves physical or domain-specific scientific experiments. Do not
select or run it for a real system without representative domain-specific
evaluation.

Reference run
=============

With 10 seeds per function, an initial design of 6, a total budget of 20
evaluations, and a 256-point candidate pool, mean final regret (relative to a
shared 4096-point test grid) and active-policy wins were. This run uses the
fixed unit length scale described above:

.. list-table::
   :header-rows: 1

   * - Function
     - Mean active regret
     - Mean random regret
     - Active final-regret wins
   * - Trigonometric
     - 0.0054
     - 0.1131
     - 10 / 10
   * - Mixed smooth
     - 0.0025
     - 0.0979
     - 10 / 10
   * - Rational
     - 0.0064
     - 0.0592
     - 10 / 10

The fixed-scale LCB policy had lower final regret on all runs in this small,
fixed benchmark. The unit scale was selected after earlier exploratory
comparisons on these function families, so this is not an independent estimate
of generalization. Treat the results as reproducible development evidence, not
a general performance claim.

Real-data model-tuning task
=============================

The second track uses scikit-learn's diabetes regression dataset and a fixed
five-fold shuffled-CV mean-squared-error objective for standardized Ridge
regression. It tunes the continuous value ``log10(alpha)`` over ``[-4, 4]``,
with 4 initial points and 12 total objective evaluations. The reference is the
minimum from a shared 301-point grid. Across five seeds, active LCB had lower
final regret in 5/5 runs; mean regrets were ``2.17e-7`` for active search and
``9.11e-6`` for uniform random search. Both are tiny relative to the roughly
``2970`` grid minimum, so this is a flat-objective smoke test, not a meaningful
general performance claim. It is one real-data model-selection task, not a
physical experiment.

Breast-cancer classification task
=================================

The separate validation track uses scikit-learn's breast-cancer dataset and
fixed five-fold stratified-CV log loss for standardized logistic regression.
It tunes ``log10(C)`` over ``[-3, 3]`` with the same 4-point initialization and
12-evaluation budget. The reference is the minimum over a shared 121-point
grid; a candidate can beat this grid, so its reported grid delta may be
negative. Across five seeds, active LCB had lower grid delta in 5/5 runs; mean
deltas were ``-2.63e-5`` for active search and ``0.00396`` for random search,
against a grid minimum of about ``0.07378``. This is one fixed-fold
classification tuning task, not physical-experiment validation or evidence of
general superiority.

With the fixed kernel configuration, the full benchmark emitted no
Gaussian-process convergence warnings. This addresses the warning source for
this proposal path; the fixed length scale remains a modeling assumption and
should be reconsidered for workloads with substantially different response
smoothness.
