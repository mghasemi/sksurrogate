.. _forecasting:

Time series and forecasting
===========================

SKSurrogate is a tabular surrogate-modelling and experiment-tracking system. It does
not replace a forecasting library, but it can *hold* a forecasting problem: declare the
temporal structure of a dataset, and the tracker validates it, keeps the time index out
of the feature matrix, and refuses cross-validation schemes that leak.

Temporal support is **opt-in**. A task with no temporal declaration behaves exactly as
before: no timestamp inference, no re-ordering, no splitter restrictions.

.. note::

   Current scope (this page documents what ships today): the temporal **data
   contract** — declaration, validation, metadata, ignore-list wiring and
   splitter enforcement — and the **leakage-safe feature builder**
   (``ForecastFeatureBuilder`` / ``make_supervised``), **rolling-origin backtesting**
   (``RollingOriginSplit`` / ``mltrack.backtest``) and **forecast metrics with naive
   baselines** (MASE, sMAPE, pinball/Winkler, naive/seasonal-naive/drift/mean).
   Horizon-aware prediction intervals, multi-step inference and the forecast UI are
   later phases.

Declaring a temporal spec
-------------------------

.. code-block:: python

    from SKSurrogate import TemporalSpec, mltrack

    spec = TemporalSpec(
        time_column="date",       # required
        series_id="store",        # optional: one identifier per series (panel data)
        freq="D",                 # optional: expected pandas offset alias
        horizon=7,                # forecast horizon in steps, >= 1
        order="increasing",       # or "decreasing"
        gap_tolerance=0,          # missing periods tolerated when freq is declared
    )

    tracker = mltrack("weekly-sales", db_name="tracking.db")
    tracker.RegisterData(frame, "sales", partition="train", temporal=spec)
    tracker.SetTemporalSpec(spec)         # re-validate every registered partition
    tracker.GetTemporalSpec()             # -> spec dict, or None
    tracker.ClearTemporalSpec()           # remove the declaration

``RegisterData`` validates the frame **before** writing it, so a rejected upload leaves
no partial state behind. ``SetTemporalSpec`` validates every partition already in the
database (``train``, ``validation``, ``test``, ...) unless you pass ``partitions=[...]``.

What validation checks
----------------------

Structural violations raise ``TemporalValidationError`` (a ``ValueError`` subclass) with
a ``report`` attribute; the report is also stored in the task metadata and returned by
the control plane.

===================  ==============================================================
Check                Failure mode
===================  ==============================================================
Column presence      ``time_column`` / ``series_id`` missing from the frame.
Time parsing         Values that cannot be read as timestamps (or non-finite values
                     on a numeric axis).
Row order            Rows not ``increasing`` (or ``decreasing``) within each series.
Duplicate times      The same timestamp twice inside one series — order-independent.
Missing identifiers  ``NaN`` in the ``series_id`` column.
Declared frequency   More than ``gap_tolerance`` missing periods, or timestamps off
                     the declared grid.
===================  ==============================================================

Advisory findings stay in the report as ``warnings``: irregular spacing when no
``freq`` is declared, gaps measured against an *inferred* frequency, and series shorter
than the horizon (a forecast cannot be backtested without more history).

Two time-axis kinds are accepted. Datetime-like columns are parsed with
``pandas.to_datetime`` and compared as integer nanoseconds — never as floats, because
contemporary timestamps exceed 2\\ :sup:`53` ns and a float axis would silently lose
precision. Integer or float columns are treated as an **index axis** (``0, 1, 2, ...``),
not as 1970 timestamps.

The report looks like this (returned by ``GET /api/datasets/{task}/temporal``)::

    {"ok": true, "time_kind": "datetime", "rows": 120, "series_count": 2,
     "declared_freq": "D", "inferred_freq": "D", "missing_periods": 0,
     "time_range": {"start": "2026-01-01 00:00:00", "end": "2026-04-30 00:00:00"},
     "series_lengths": {"A": 60, "B": 60},
     "series": [{"series": "A", "rows": 60, "regular": true, ...}],
     "warnings": [], "errors": []}

Temporal columns never become features
--------------------------------------

The time column and the series identifier describe *when* and *which series*, not
*what*. They are excluded from the feature matrix by default:

* ``mltrack.RegisterData`` drops them from ``self.X`` and ``get_data``;
* the control plane adds them to the dataset's **ignored columns**, so every training,
  profiling, EDA, sensitivity and monitoring path that honours the ignore list drops
  them too.

Time reaches a model through generated features only. Letting a datetime column through
as ``DataProcess``'s scaled epoch-seconds feature is the trap this rule closes.

.. _forecasting-splitters:

Splitters: order-respecting only
--------------------------------

A temporal task may only use splitters from ``TEMPORAL_CV_SPLITTERS`` (currently
``TimeSeriesSplit``). ``SetCV`` raises a ``ValueError`` for ``KFold``, ``ShuffleSplit``,
stratified splitters, a bare integer fold count and the library default, because each of
them trains on future observations and scores on past ones — the score is then an
artifact of leakage rather than evidence::

    tracker.SetCV({"type": "TimeSeriesSplit", "n_splits": 4, "gap": 7})

Choose ``gap`` at least as large as the forecast horizon so the training window never
contains observations that overlap the test window's information set.

.. _forecasting-manual-features:

Leakage-safe feature construction
---------------------------------

.. code-block:: python

    from SKSurrogate import TemporalSpec, make_supervised

    spec = TemporalSpec(time_column="date", series_id="store", freq="D", horizon=7)
    design = make_supervised(
        frame,                      # must be ordered; still contains the target
        spec,
        target="sales",
        lags=[7, 14, 28],           # each lag must be >= horizon
        seasonal_lags=[7, 365],     # readable alias: lag at the season period
        rolling=[{"window": 28, "stats": ["mean", "std"]}, 7],
        diffs=[1],
        pct_changes=[1],
        calendar=["dow", "doy", "is_weekend"],
        fourier={"periods": [7, 365.25], "terms": 2},
        exogenous=["promo_flag"],   # must be known at the forecast origin
    )

    design.X            # feature matrix
    design.y            # label at t + horizon (direct) or t + 1 (recursive)
    design.meta         # series, origin_time, target_time, horizon, target_step
    design.feature_names
    design.describe()   # JSON-safe provenance for task metadata

The rules the builder enforces, not merely documents:

* **Shift $\ge$ horizon.** Every target-derived feature reads at most up to
  $t - h$: ``lag_k`` is $y_{t-k}$ ($k \ge h$), and a rolling window of width $w$
  covers $y_{t-h-w+1},\dots,y_{t-h}$. A lag shorter than the horizon is rejected
  at ``fit`` time — it would read an observation the forecast is meant to predict.
  ``seasonal_lags=[p]`` is the same rule under a readable name (column
  ``sales__seasonal_lag_7``, ``op="seasonal_lag"`` in the provenance).
* **No centring.** ``center=True`` does not exist in this API.
* **Per-series isolation.** On a panel, every operation is computed inside its own
  series, so one store's history never enters another store's lags.
* **No partial windows by default.** ``min_periods`` defaults to the window width.
* **NaN, never ``inf``.** A zero denominator in ``pct_change`` yields ``NaN``.
* **Immutability of the past.** Calendar and Fourier terms come from the timestamp
  itself, which is known at forecast time; the exogenous columns are passed
  through unchanged and documented as known-at-origin.

Feature names encode their provenance (``sales__lag_7``, ``sales__roll_mean_28``,
``sales__diff_1``, ``cal__dow``, ``fourier__sin_7_1``), and
``design.feature_specs`` carries the same information structurally, including the
minimum ``shift`` of each column.

Direct vs recursive
~~~~~~~~~~~~~~~~~~~

``strategy="direct"`` (default) builds one label per horizon, $y_{t+h}$, so a model
learns the multi-step mapping directly. ``strategy="recursive"`` labels only the
next step ($y_{t+1}$) while the declared horizon still constrains the features;
at serve time the model feeds its own prediction back. ``design.horizon`` is the
declared horizon and ``design.target_step`` is what the label actually is.

Verifying the invariant
~~~~~~~~~~~~~~~~~~~~~~~

``assert_no_leakage`` turns the claim into a test: it masks every non-temporal
column strictly after a cut time and rebuilt features for origins at or before
that cut must be unchanged. A feature that reads the future either changes value
or disappears (it now reads a missing observation), and the guard raises
:class:`~SKSurrogate.temporal.LeakageError` naming the columns:

.. code-block:: python

    from SKSurrogate import assert_no_leakage

    assert_no_leakage(builder, frame, spec)     # {'ok': True, 'cut_points_checked': 3, ...}

This is an empirical guard over structural properties, not a proof; run it once per
feature configuration in your test suite.

Scikit-learn interoperability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ForecastFeatureBuilder`` is an estimator with ``get_params``/``set_params`` and
``fit``/``transform``, so it clones inside pipelines. One wrinkle is deliberate:
``transform`` needs the **target column in the frame** because lag features are
built from it. The control plane's feature matrix drops the target, so materialize
the design matrix before registration, or keep the target in the frame you hand to
the builder.

If you cannot use the builder (exotic features, existing pipeline), the manual
fallback is still: keep rows chronological, shift every feature by at least the
horizon per series, never use ``center=True``, and never aggregate a window that
includes the target step.


Rolling-origin backtesting
--------------------------

``TimeSeriesSplit`` is a fixed-window approximation; ``RollingOriginSplit`` states
the forecast horizon directly, so each test window is exactly ``horizon`` steps
long and origins advance by ``step``::

    from SKSurrogate import RollingOriginSplit, mltrack

    tracker = mltrack("weekly-sales", db_name="tracking.db")
    tracker.SetCV(RollingOriginSplit(n_splits=6, horizon=7, step=7))

The tracker-level entry point builds the design matrix itself and walks origins
forward::

    result = tracker.backtest(
        LinearRegression(),
        n_splits=6,
        horizon=7,
        step=7,
        refit="every_fold",          # or "first", or "every:2"
        scoring=get_scorer("neg_mean_absolute_error"),
        scoring_label="neg_mean_absolute_error",
        feature_kwargs={"lags": [7, 14], "rolling": [{"window": 28, "stats": ["mean"]}]},
    )

    result["pooled"]        # mae, rmse, bias, smape, mase for the model
    result["baselines"]     # the same metrics for each naive baseline
    result["ranking"]       # MASE ordering with the "worse than baseline" flags
    result["folds"]         # per-fold score/metrics, train/test sizes, whether it refitted
    result["origins"]       # per-origin series, origin_time, target_time, y_true, y_pred
    result["per_series"]    # the same metrics per series on a panel
    result["design"]        # features actually used, with their provenance

What the result guarantees:

* **No overlap.** Training rows stop one step before the origin and the test window
  starts there; the geometry comes from :func:`rolling_origin_indices`, which is
  validated against the available history and raises instead of silently shrinking.
* **Honest labels.** Each record pairs a prediction with the actual value at
  ``origin_time + horizon`` (``target_step`` in ``meta``).
* **Counted fits.** The refit policy is enforced, not just recorded: ``"every_fold"``
  fits once per origin, ``"first"`` fits once in total, ``"every:k"`` every k-th
  fold. ``result["folds"][i]["refitted"]`` says which folds refitted.
* **A cost ceiling.** ``cost_limit`` (default 64 fits) turns an accidental
  hundreds-of-origins run into a clear ``ValueError``.
* **Panel semantics.** With a ``series_id``, every series gets its own origins and
  the model is fitted on the union of all series' training windows; series too short
  for the geometry are skipped and listed in ``skipped_series``.
* **Persistence.** The full result is stored in task metadata under
  ``backtest:<config hash>`` unless ``store=False``; per-origin records are capped
  by ``max_records`` with an ``origins_truncated`` count.

Only the **direct** strategy is evaluated today. A recursive forecast predicts one
step at a time by feeding its own output back, which needs the serving state that
arrives with multi-step inference; passing ``strategy="recursive"`` raises rather
than silently returning one-step numbers labelled as multi-step.

**Feature configuration must match how the model was trained.** ``backtest`` refits
the estimator on the design it builds, so a bundle trained on a different feature
set will fail on width rather than produce a misleading score. Keep the
``feature_kwargs`` in the request/bundle metadata alongside the model.


Forecast metrics and baselines
------------------------------

Point metrics (``SKSurrogate.forecast_metrics``):

* **MAE** $\frac1n\sum\lvert y_t-\hat y_t\rvert$ and **RMSE**
  $\sqrt{\frac1n\sum(y_t-\hat y_t)^2}$ — in the target's units.
* **bias** $\frac1n\sum(\hat y_t-y_t)$ — positive means over-forecasting.
* **sMAPE**, reported in percent:
  $\frac{100}{n}\sum \frac{2\lvert\hat y_t-y_t\rvert}
  {\lvert\hat y_t\rvert+\lvert y_t\rvert}$.
  A term where both values are zero contributes 0, never ``inf``.
* **MASE** $\frac{\frac1n\sum\lvert y_t-\hat y_t\rvert}
  {\frac{1}{n-m}\sum_t\lvert y_t-y_{t-m}\rvert}$,
  where the denominator is the mean absolute $m$-step difference of the **in-sample
  history**. Because that scale is fixed by the history, a seasonal-naive forecast
  scores exactly 1.0 — which is what makes "worse than naive" a precise statement.
* **RMSE scaled** — the same scaling applied to RMSE.

Interval and quantile metrics:

* **pinball_loss(y, ŷ, q)** — the quantile loss; at $q=0.5$ it is exactly
  $\tfrac12\,\mathrm{MAE}$.
* **winkler_score(y, lower, upper, α)** — interval width plus
  $\tfrac{2}{α}$ times each miss, for central intervals at level $1-α$.
* **interval_coverage** and **interval_width** — empirical coverage and mean width.

``summarize(...)`` returns one block with all of these, plus optional
``per_series`` / ``per_horizon`` groupings. Its ``aggregation`` argument chooses
what ``primary`` reports: ``"pooled"`` (every prediction weighted equally — honest for
one series, misleading across series of different magnitude) or ``"macro"`` (each
series counts once). The per-series block is always returned alongside it.

Baselines
~~~~~~~~~

Every backtest scores four naive baselines on **the same origins** as the model:

* ``naive`` — the last observation, $\hat y_{T+h} = y_T$;
* ``seasonal_naive`` — the same phase one season earlier,
  $\hat y_{T+h} = y_{T+h-m\lceil h/m\rceil}$ (for $h=m$ this is $y_T$);
* ``drift`` — last value plus the average change per step, times $h$;
* ``mean`` — the arithmetic mean of the history.

``baseline_prediction(history, method, horizon, seasonality)`` computes any of them
from an ordered history array; ``baselines=(...)`` on ``backtest`` selects which are
scored (default: all four). A model that cannot beat ``seasonal_naive`` on MASE has
not earned the word "forecast", and the result says so:

.. code-block:: python

    result["ranking"]
    # {"metric": "mase",
    #  "best": {"label": "baseline:seasonal_naive", "value": 1.0,
    #           "beats_baseline": False, "worse_than_baseline": False},
    #  "ranked": [...], "unranked": [], "winner_beats_baseline": False}

The guard distinguishes parity from failure: MASE **equal** to 1 is exact parity with
the benchmark (``beats_baseline: false``, ``worse_than_baseline: false``), while
``> 1`` is worse. Candidates whose metric is missing or non-finite are listed under
``unranked`` instead of being sorted to the bottom, so a broken run cannot look like
a bad one.

MASE is not a scikit-learn scorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MASE needs the in-sample history of each training window, and a scikit-learn scorer
only ever sees ``(y_true, y_pred)``. Rather than invent a denominator, MASE is
computed by the rolling-origin backtest, where the training window of every origin is
known. For plain CV and learning curves, **sMAPE** is available as
``neg_smape`` (sign-flipped, as scikit-learn scorers are), and pinball loss via
``neg_mean_pinball_loss`` with an explicit ``pinball_alpha``.

Control-plane endpoints
-----------------------

========================================================  ============================================
Endpoint                                                  Purpose
========================================================  ============================================
``POST /api/datasets/{task}/register``                    Accepts an optional ``temporal`` form field
                                                          (JSON) and validates the upload before
                                                          storing it; returns the report.
``GET  /api/datasets/{task}/temporal``                    Current spec, per-partition validation
                                                          reports, temporal columns, ignore list and
                                                          whether the stored CV splitter is
                                                          temporal-safe.
``PUT  /api/datasets/{task}/temporal``                    Declare or replace the spec; validates every
                                                          registered partition; merges the temporal
                                                          columns into the ignore list. Answers 422
                                                          with the report when a partition violates
                                                          it.
``DELETE /api/datasets/{task}/temporal``                  Remove the declaration; drops only the
                                                          ignore entries this feature added.
``POST /api/evaluation/{task}/backtest``                  Queue a rolling-origin backtest for a bundle
                                                          (``model_version`` or ``alias``, horizon,
                                                          n_splits, refit, features); returns a job id
                                                          whose result holds per-origin records and
                                                          metrics.
========================================================  ============================================

Non-goals
---------

* No timestamp inference, re-sorting or frequency guessing at registration: a frame that
  is not already ordered in the declared direction is rejected, not repaired.
* No automatic model selection among forecasting families, no reconciliation across
  hierarchies, no probabilistic multivariate forecasts.
* No bundled forecasting backends (StatsForecast, sktime, Darts). A model is still any
  scikit-learn-compatible estimator you pass in.
