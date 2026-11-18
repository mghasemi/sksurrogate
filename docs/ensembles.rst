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
       models), each with its ``minimum_members``, its aggregation in plain words, and its
       parameter schema (type, default, bounds, and the accepted names for a stacking
       final estimator). Also returns ``max_members`` and the caveat list shown with the
       scores.
   * - ``GET /api/ensembles/{task}/candidates``
     - The task's models that can be selected as members, newest first, each with family,
       metrics, source (unregistered bundle or registry) and whether it is itself an
       ensemble. The fitted object is never part of the payload.
   * - ``POST /api/ensembles/{task}/fit``
     - Submits the combination as a background job (``kind: "ensemble"``). Body:
       ``members`` (ordered — weights are positional), ``method``, ``params``,
       ``train_partition``, optional ``scoring`` (defaults to the family's metric),
       ``random_state``, ``label``, and ``register`` to also register the result as a
       candidate version. Answers ``400``/``404`` at submit time for a rejected request.
   * - ``GET /api/ensembles/{task}``
     - The task's ensembles with method, aggregation, member count and stored scores.
   * - ``GET /api/ensembles/{task}/{version}``
     - One ensemble's detail: the member list with each member's same-fold score, the
       parameters, the out-of-fold folds, and the caveats. Answers ``400`` for a bundle
       that is not an ensemble.

Refusals are explicit rather than approximate: an unknown member version (``404``), a
member selected twice, a mixed family, an unknown method or parameter, soft voting without
``predict_proba``, classification boosting without sample-weight support, or more than
twelve members (``400``). A rejected request never becomes a failed background job.

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
      "registered": false
    }

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
