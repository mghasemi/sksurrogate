=================
Combining Models
=================

A search produces several models that are good at different things, and the next
question is usually whether they are better *together*. ``SKSurrogate`` answers it with
the combination methods scikit-learn already ships: voting, averaging, stacking, bagging
and boosting over a selection of your trained models.

The result is an ordinary :class:`~SKSurrogate.ModelBundle` whose estimator happens to be
a meta-estimator, so everything that consumes a bundle applies to it unchanged —
registration, promotion, the serving pin, batch and online prediction, learning curves,
diagnostic curves, nested cross-validation and the run report. Nothing in those paths
knows an ensemble exists.

Combination methods
===================

Every method below combines the *selected models*. Each member keeps its own
preprocessing pipeline, and every member is refitted when the combination is fitted, so
no member's fitted state is carried across a split.

.. list-table::
   :header-rows: 1
   :widths: 14 34 38 14

   * - Method
     - Classification
     - Regression
     - Uses every member
   * - ``voting``
     - ``VotingClassifier``; majority label vote unless soft voting is requested
     - ``VotingRegressor``; unweighted mean of predictions
     - yes
   * - ``averaging``
     - ``VotingClassifier(voting="soft")``; mean of class probabilities
     - ``VotingRegressor``; unweighted mean of predictions
     - yes
   * - ``stacking``
     - ``StackingClassifier`` with a final estimator fitted on out-of-fold member
       predictions
     - ``StackingRegressor`` with a final estimator fitted on out-of-fold member
       predictions
     - yes
   * - ``bagging``
     - one ``BaggingClassifier`` **per member**, then soft voting over the bagged members
     - one ``BaggingRegressor`` **per member**, then averaging
     - yes
   * - ``boosting``
     - one ``AdaBoostClassifier`` **per member**, then soft voting over the boosted members
     - one ``AdaBoostRegressor`` **per member**, then averaging
     - yes

For regression, ``voting`` and ``averaging`` are the same operator: scikit-learn has no
label-voting regressor, and a regressor's natural vote is already an average. The
identity is documented rather than rejected.

Bagging and boosting need a word, because scikit-learn's own classes replicate **one**
base estimator rather than combining a set. To combine a set, each selected member gets
its own bagging (or boosting) wrapper and the wrappers are then aggregated — so every
member genuinely contributes, and the resampling or reweighting happens per member. A
single member is therefore a valid ``bagging``/``boosting`` request (that is ordinary
bagging), while the aggregating methods need at least two.

Three measured constraints
==========================

The composition is shaped by behaviour that is not visible from the class names. Each of
these silently produces a *wrong* model rather than an error, so each is pinned by a test
in ``tests/test_ensembling.py``:

**Members must be array-safe.** ``VotingClassifier``, ``VotingRegressor`` and the
stacking classes consume a heterogeneous member set only when they are handed a
``DataFrame``. Bagging goes further and indexes ``X`` *positionally* before fitting each
replica (``X[indices]``), so a member whose encoder selects columns by name fails with
``Specifying the columns using strings is only supported for dataframes``. Every member is
therefore passed through :func:`~SKSurrogate.DataProcess.frame_tolerant`, and
:class:`~SKSurrogate.DataProcess.ArrayAsFrame` restores the frame from a positional array.

**Positional arrays must keep their dtypes.** A mixed-type matrix is an *object* array, so
a rebuilt frame would carry ``object`` for every column. A DataFrame-first preprocessor
decides per column whether it is numeric or categorical from the dtype it sees, so an
encoder fitted on such an array classified every numeric column as categorical — measured
on a three-column, 300-row frame, the encoder expanded to 423 one-hot columns and a bagged
member scored ROC AUC 0.55 where the same member fitted on the frame scored 0.95.
``ArrayAsFrame`` now infers dtypes on the rebuild, which is what makes "fit on the frame"
and "fit on ``frame.to_numpy()``" the same model.

**Boosting needs ``sample_weight`` in the base's ``fit`` signature.**
``AdaBoostClassifier`` gates its base estimator with ``has_fit_parameter(estimator_,
"sample_weight")`` and then forwards the round weights. ``Pipeline.fit`` has no such named
parameter, and an adapter that swallows weights through ``**fit_params`` does not satisfy
the check either, so :class:`~SKSurrogate.ensembling.SampleWeightRouter` names the
parameter explicitly and routes the weights to the pipeline's last step
(``<step>__sample_weight``). A classification member whose final estimator cannot take
weights is refused rather than boosted into identical rounds.
``AdaBoostRegressor`` does neither: it reweights by *resampling* and calls
``estimator.fit(X_, y_)`` with no weights at all, so regression members need no adapter and
are not refused for lacking sample-weight support.

Soft voting is the one requirement that is checked rather than worked around: it needs
every member to expose ``predict_proba``, so a member without one is refused with the
offending names (hard voting has no such requirement).

Combining models in Python
==========================

.. code-block:: python

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from SKSurrogate import build_ensemble, load_bundle, save_bundle, ModelBundle

    members = [
        ("logistic", load_bundle("bundles/churn/logistic.bundle").model),
        ("tree", load_bundle("bundles/churn/tree.bundle").model),
    ]
    ensemble = build_ensemble(
        "stacking",
        members,
        params={"final_estimator": "logistic_regression", "cv": 5},
        columns=list(features.columns),
        random_state=0,
    )
    ensemble.fit(features, target)

    bundle = ModelBundle(
        ensemble,
        task_name="churn",
        schema=feature_schema,
        metadata={"ensemble": {"method": "stacking", "members": [name for name, _ in members]}},
    )
    save_bundle(bundle, "artifacts/churn-stacking.bundle")

``method_catalogue(family)`` returns the methods and their parameter schema for a problem
family; ``validate_params`` checks a parameter mapping against that schema (an unknown key
is refused rather than ignored, so a typo never silently produces a different
combination); ``resolve_family`` refuses a member set that mixes classifiers with
regressors; and ``member_scores`` / ``cross_validated_scores`` produce the member-versus-
ensemble comparison described next.

How the scores are meant to be read
===================================

A combination is fitted on the task's training partition, and its score is reported
two-sided:

* ``cv_score`` (with ``cv_score_std`` and the per-fold values) is **out-of-fold** under
  the splitter stored for the task — or five fixed folds when no splitter has been chosen.
  The member rows of the comparison are computed on the *same* folds, so the numbers are
  comparable row by row.
* ``train_score`` is the score of the refitted ensemble on all training rows. It is
  optimistic by construction and is reported alongside ``cv_score`` for that reason, never
  instead of it.

Both are stored with the bundle (``bundle.metrics`` and ``bundle.metadata["ensemble"]``)
and the model itself carries the caveat that qualifies them: the members were fitted, and
where searched also tuned, on this same data, so a member that was selected because it
scored well here is then scored on the same rows again. The member and ensemble
cross-validated scores are therefore **jointly optimistic**, and the honest estimate is the
nested cross-validation on the Evaluation page, which re-selects and re-fits the estimator
inside every outer fold. Combining models cannot by itself justify a claim about unseen
data.

Cost grows quickly: each member is fitted once per comparison fold, each bagging member
``n_estimators`` times per fold, and stacking additionally fits its meta-learner on
out-of-fold predictions. A combination of ``n`` members with ``k`` folds costs on the order
of ``n·k`` member fits, so the control plane runs it as a background job and caps a
selection at twelve members.

Combining models from the control plane
=======================================

The web UI's **Ensembles** page (see :doc:`ui-manual`) drives the same code through five
endpoints:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Endpoint
     - Contract
   * - ``GET /api/ensembles/{task}/methods``
     - The combiners that apply to the task's problem family (detected from its stored
       models), each with its ``minimum_members``, its aggregation in plain words,
       ``accepts_weights``, and its parameter schema (type, default, bounds, and the
       accepted names for a stacking final estimator). Also returns ``max_members``, the
       ``calibration`` and ``weight_optimization`` blocks described under
       `Refining a combination`_, and the caveat list shown with the scores.
   * - ``GET /api/ensembles/{task}/candidates``
     - The task's models that can be selected as members, newest first, each with family,
       metrics, source (unregistered bundle or registry) and whether it is itself an
       ensemble. The fitted object is never part of the payload.
   * - ``POST /api/ensembles/{task}/fit``
     - Submits the combination as a background job (``kind: "ensemble"``). Body:
       ``members`` (ordered — weights are positional), ``method``, ``params``,
       ``train_partition``, optional ``scoring`` (defaults to the family's metric),
       ``random_state``, ``label``, ``register`` to also register the result as a
       candidate version, and the three optional refinement blocks ``calibration``,
       ``weight_optimization`` and ``honest_cv``. Answers ``400``/``404`` at submit time
       for a rejected request.
   * - ``GET /api/ensembles/{task}``
     - The task's ensembles with method, aggregation, member count, stored scores, and
       each row's ``calibrated`` / ``weighted`` flags and ``nested_score``.
   * - ``GET /api/ensembles/{task}/{version}``
     - One ensemble's detail: the member list with each member's same-fold score, weight
       and calibration flag; the parameters; the per-member calibration report; the
       optimized weight vector; the out-of-fold folds; the nested estimate when it was
       requested; and the caveats. Answers ``400`` for a bundle that is not an ensemble.

Refusals are explicit rather than approximate: an unknown member version (``404``), a
member selected twice, a mixed family, an unknown method or parameter, soft voting without
``predict_proba``, classification boosting without sample-weight support, or more than
twelve members (``400``). The refinement blocks are validated by the same functions the
library uses, so an unknown weight strategy, a fold count below two, calibration against
regression members, weight optimization together with an explicit ``weights`` parameter, or
weight optimization against stacking are all ``400`` at submit time rather than a failed
job. A rejected request never becomes a failed background job.


The job result carries the same comparison the UI renders, so a script can read it without
loading the bundle:

.. code-block:: json

    {
      "model_version": "c7e267048b8343d399be7a2524710523",
      "method": "stacking",
      "family": "Classification",
      "members": [
        {"model_version": "b86fbd7...", "cv_mean": 0.9267, "cv_std": 0.0200},
        {"model_version": "179bb3d...", "cv_mean": 0.9367, "cv_std": 0.0245}
      ],
      "metrics": {"train_score": 0.9533, "cv_score": 0.9333, "cv_score_std": 0.0149},
      "calibration": null,
      "weight_optimization": null,
      "nested": null,
      "registered": false
    }

Refining a combination
======================

Three optional blocks refine a combination beyond the combiner's own parameters. They are
orthogonal to ``method`` — they post-process every member, or the merge itself — and each is
recorded with the model together with its own evidence and caveat.

Per-member calibration (``calibration``)
----------------------------------------

``{"method": "sigmoid" | "isotonic", "cv": 5, "ensemble": "auto" | "per_fold" | "single"}``

Each member's outermost form is wrapped in scikit-learn's
``CalibratedClassifierCV(estimator=..., cv="prefit")``, so the probabilities that enter the
merge are the calibrated ones. This matters for members whose raw output is not a
probability: a fully grown tree reports 0 and 1, and a soft vote, a stacking meta-learner and
any log-loss reading all treat that as certainty.

Two nesting decisions are load-bearing:

* The calibrator must sit **outside** the resampling wrapper. Inside, ``BaggingClassifier``
  forwards the bootstrap ``sample_weight`` to the calibrator and scikit-learn warns that the
  calibration is "likely to be incorrect"; outside, the calibrator sees the fold's own rows.
  The structure is pinned by a test and the absence of the warning is pinned end to end.
* ``ensemble=True`` (one calibrator per fold) is the wrong default here: the calibration
  would be refitted per bagging batch. ``ensemble="auto"`` means scikit-learn's own default
  *for the installed build* and the other two tokens map to ``per_fold`` / ``single``
  explicitly. ``auto`` is resolved from the installed scikit-learn rather than hard-coded:
  before 1.6 that argument accepts only a boolean (default ``True``), from 1.6 on it also
  accepts the literal ``"auto"``, so a request stays portable across the versions this
  package supports.

Calibration is classification-only and is refused for regression members. What is recorded
is a **before/after pair on identical folds** — the Brier score and the log loss of the
member's probabilities without calibration and with it — so "this member is now calibrated"
is a measurement rather than an assertion: an overconfident tree improves sharply, a
well-calibrated logistic member barely moves, and both are honest outcomes. Calibration
changes probabilities, not decisions, so an accuracy-style metric may barely move while log
loss falls.

Two cases are reported rather than silently approximated. A member exposing only
``decision_function`` (no ``predict_proba``) is perfectly calibratable, but Brier score and
log loss are probability metrics, so that member is recorded as calibrated *without* a
before/after pair instead of substituting a decision-function scale under the same column
name. A member exposing neither response is refused at submit time by name.

Optimized member weights (``weight_optimization``)
--------------------------------------------------

``{"strategy": "least_squares" | "dirichlet" | "greedy", "samples": 200, "rounds": 20}``

The member weights are fitted on the members' **out-of-fold** output, computed once:

* ``least_squares`` solves for the best weights for the target — a Brier score for
  classification, a mean squared error for regression — over the unit simplex (weights
  non-negative and summing to one). It needs no metric, and it is deterministic.
* ``dirichlet`` samples candidate vectors from the simplex and keeps the best value of the
  task's scoring metric: the escape hatch for a metric that is not a squared error.
* ``greedy`` is Caruana's forward selection with replacement, accumulating one member at a
  time on the scoring metric. The result is integer weights, so a member that never improves
  the score stays at zero — visible in the weight column, and named in ``dropped``.

Weights apply wherever members are merged by a weighted vote or average: voting, averaging,
bagging and boosting. Stacking is refused, because its meta-learner already learns the
combination and a weight vector would be silently ignored. An explicit ``weights`` parameter
and ``weight_optimization`` are mutually exclusive for the same reason — the optimized vector
would replace the typed one without saying so.

What the fit records is the vector, the strategy, the folds the vector was fitted on, the
budget the search spent (``samples`` or ``rounds``; ``null`` for the deterministic
least-squares fit), the members it dropped, and the objective it reached. The objective is
``null`` when the search could not evaluate any candidate at all — a scorer that returns a
non-finite value, which is what scikit-learn's own scorers do on a degenerate fold. That case
leaves the uniform vector in place and reports no objective rather than a number that is not
valid JSON.

The honest estimate (``honest_cv``)
-----------------------------------

``{"folds": 5, "inner_folds": 3}``

This is the number that prices the two procedures above. ``honest_cv`` runs an outer
cross-validation in which *everything* is refitted inside each training fold: the members,
their calibration, the weight search, and the combiner. The reported mean is the score of
the **procedure** rather than of the one model that happens to be stored, and it is the only
number here that was not fitted against the data it is scored on.

It is off by default because it costs on the order of ``outer_folds × (members + the weight
search)`` fits. When it is on, the gap between it and the in-fold ``cv_score`` is the measured
optimism of the in-fold number — which is exactly why both are reported side by side instead
of the optimistic one alone.

How large that gap is depends on whether the search has anything to fit, so no single number
describes it. Measured across ten configurations (200–560 rows, two to three members,
least-squares and greedy searches, three to five scoring folds) the gap ran from **0.0000 to
0.0233**. The zero end is not a bug: when the search collapses onto one member — the usual
outcome when one member dominates — the weighted ensemble *is* that member, so both protocols
score the same estimator and there is nothing to pay for. The large end is the case the
estimate exists for: three members including a 1-NN memorizer, 300 rows, greedy search over
three scoring folds, 0.8100 in-sample against 0.7867 ± 0.0340 with everything refit per fold.
A wide ``std`` next to the nested mean is the honest signal that the procedure's own score is
unstable, which is information the in-sample number hides.

Evaluating a combination
========================

Because the result is a bundle, evaluation needs no ensemble-specific route:

* **Learning curves** and **diagnostic curves** (ROC, calibration, gain, lift) take the
  ensemble's ``model_version`` like any other version — ``GET
  /api/evaluation/{task}/learning-curves?model_versions=...`` and ``GET
  /api/evaluation/{task}/{version}/curves``.
* **Nested CV** (``POST /api/evaluation/{task}/nested-cv``) is the estimate that accounts
  for member selection, since each outer fold re-selects nothing but does re-fit the
  combination from scratch.
* The **run report** (``GET /api/bundles/{task}/{version}/report.md``) names the members
  and the combination in its methodology section.

The intermediate representation matters here: nested cross-validation wraps a candidate in
its own encoder when the partition is mixed-type and the candidate does not look
frame-preprocessing. An ensemble has no ``steps`` of its own, so
:func:`~SKSurrogate.DataProcess.preprocesses_frames` now sees through meta-estimators
(voting, stacking, bagging, boosting and single-estimator wrappers) and answers for the
whole combination. Without that, every member would have received an already-encoded
matrix.

Temporal tasks (forecast bundles)
=================================

Combining a temporal task's bundles works on the **generated one-step forecast
design**, not on the raw partition. A temporal partition has no feature column of
its own — the time and series columns are excluded by the temporal contract and the
target is the label — so the combination is fitted on exactly the matrix the
experiment search, the rolling-origin backtest and the explanations consume.

Three consequences are recorded with the model:

* Members enter the merge through their **bare estimators**. A temporal member
  bundle is ``Pipeline(ForecastFeatureBuilder, estimator)`` and expects raw
  history; the builder is removed before combining, because a builder handed a
  design matrix fails with ``missing required column(s): <time>``.
* The scoring folds are order-respecting. With no stored splitter the fallback is
  :class:`sklearn.model_selection.TimeSeriesSplit`, never an integer fold count —
  an integer means a shuffled ``KFold`` in scikit-learn, which would train on the
  future and score the past. The same rule is applied to the ``honest_cv``
  nestings. Because a walk-forward splitter's test blocks do not partition the
  rows (the initial training window is never predicted), the out-of-fold member
  output a weight search needs is replayed fold by fold instead of through
  ``cross_val_predict``, which refuses such a splitter with
  ``only works for partitions``. The rows no fold could predict are reported as
  ``rows_unscored`` next to the weights, not scored with a placeholder.
* The result is still an ordinary forecast bundle:
  ``Pipeline(ForecastFeatureBuilder, ensemble)`` plus the serving geometry
  (``strategy: recursive``), its schema describing the raw history columns. It
  answers ``POST /api/inference/{task}/forecast`` like any temporal bundle, and
  the record carries ``feature_source: temporal_design`` with the builder and spec.

``cv_score`` on such a bundle is an out-of-fold score on the **design**, i.e. a
one-step forecast score, and it is presented as such. Multi-step accuracy is the
rolling-origin backtest's measurement; the ensemble's caveat list says so.

Limitations
===========

* Members are combined as they are: an ensemble does not tune its members' hyperparameters,
  and it does not select members for you. Selecting members *is* the statistical decision,
  and the scores reported for it are optimistic (see above).
* Stacking's meta-learner is fitted on out-of-fold member predictions on the training
  partition; with a small partition and many members the meta-learner has little to work
  with, and its score can fall below the best member's.
* A member set must be one problem family. Mixing a classifier with a regressor is
  refused rather than coerced.
* Regression members are unification-compatible only with the methods listed above;
  ``GradientBoosting*`` and ``HistGradientBoosting*`` cannot consume a member pipeline at
  all (they build their own trees), which is why boosting here means ``AdaBoost*``.
* Calibration needs a member that can score: a classifier exposing neither
  ``predict_proba`` nor ``decision_function`` is refused by name, since there is nothing to
  calibrate. It also needs enough rows per fold — ``isotonic`` in particular is a step
  function, so a small partition makes it overfit the fold it was fitted on.
* An optimized weight vector is fitted on the members' out-of-fold output, which is the same
  output the ensemble's ``cv_score`` is then computed on. That number therefore does not pay
  for the weight search, and it is reported next to the honest nested estimate rather than
  instead of it. A weight search can also be close to a no-op: on members that are already
  similar, or on a small partition, the fitted vector stays near uniform, and the honest
  reading is that the members carry the same information rather than that the search failed.
* The weight search sees only the members' **aggregate** output: it optimizes how to merge
  what the members produce, and cannot repair a member that is wrong on a whole region.
  Calibration is the finer instrument for that (it repairs the probabilities), and neither
  replaces selecting better members.
