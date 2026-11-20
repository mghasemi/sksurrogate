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

   Current scope (this page documents what ships today): the temporal **data contract**
   — declaration, validation, metadata, ignore-list wiring and splitter enforcement.
   Leakage-safe lag/rolling feature generation, rolling-origin backtesting and
   forecast-specific metrics are planned phases and are *not* available yet. Until they
   land, build lag and rolling features yourself, with the shift discipline described in
   :ref:`forecasting-manual-features`.

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

Building features by hand (until feature generation lands)
----------------------------------------------------------

Keep rows in chronological order and construct every lag/rolling column with an
explicit shift of at least the horizon, per series::

    frame = frame.sort_values(["store", "date"])
    grouped = frame.groupby("store", sort=False)["sales"]
    frame["lag_7"] = grouped.shift(7)
    frame["roll_mean_28"] = grouped.transform(
        lambda series: series.shift(7).rolling(28, min_periods=28).mean()
    )

Never use ``center=True``, and never aggregate a window that includes the target step.
Register the resulting frame as the ``train`` partition and the strictly later
observations as ``validation``/``test``.

Control-plane endpoints
-----------------------

======================================================  =============================================
Endpoint                                                Purpose
======================================================  =============================================
``POST /api/datasets/{task}/register``                   Accepts an optional ``temporal`` form field
                                                        (JSON object) and validates the upload before
                                                        storing it; returns the report.
``GET  /api/datasets/{task}/temporal``                   Current spec, per-partition validation
                                                        reports, temporal columns, ignore list and
                                                        whether the stored CV splitter is temporal-safe.
``PUT  /api/datasets/{task}/temporal``                   Declare/replace the spec; validates every
                                                        registered partition; adds the temporal columns
                                                        to the ignore list. Answers **422**
                                                        (``X-Reason: temporal-validation``) with the
                                                        report when a partition violates it.
``DELETE /api/datasets/{task}/temporal``                 Remove the declaration; drops only the ignore
                                                        entries this feature added.
======================================================  =============================================

Example::

    curl -X PUT http://127.0.0.1:8000/api/datasets/weekly-sales/temporal \
      -H 'Content-Type: application/json' \
      -d '{"time_column": "date", "series_id": "store", "freq": "D", "horizon": 7}'

Non-goals
---------

* No timestamp inference, re-sorting or frequency guessing at registration: a frame that
  is not already ordered in the declared direction is rejected, not repaired.
* No automatic model selection among forecasting families, no reconciliation across
  hierarchies, no probabilistic multivariate forecasts.
* No bundled forecasting backends (StatsForecast, sktime, Darts). A model is still any
  scikit-learn-compatible estimator you pass in.
