Changelog
=========

[Unreleased]
------------

Partition defaults, temporal experiments and the Advanced scorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Fixed:** the web UI's train-partition field started as the literal ``train``
  and always sent it, so a task whose data lives in another partition (a single
  ``main``, or one named after the task) failed with ``404: No dataset partition
  'train' stored``. Experiments, Bundles, Evaluation, Feature Analysis and
  Retraining now derive the default from the dataset metadata (``train`` when it
  exists, else the first registered partition) and omit the field while it is
  unknown. Every API request model that defaulted to ``"train"`` now defaults to
  ``None`` and resolves through ``api/partitions.py``, so a stale frontend cannot
  reintroduce the 404.
* **Fixed:** the explainability report was the last caller that hard-coded the
  literal partition ``train``. ``POST /api/reports/{task}/explain`` called
  ``explain._features_and_target(task, "train")`` before any diagnostic ran, so a
  task stored as ``main`` failed the whole job with ``404: No dataset partition
  'train' stored``. The report now resolves the task's partition once and uses it
  for the row-order clamp, the learning-curve fits, the recorded train fingerprint
  and the PDF title; the row-order clamp also counts the *explainable* feature
  space (``explain._explain_frame``: raw feature columns, or a temporal task's
  generated forecast design) instead of the raw partition columns, so a temporal
  report no longer clamps from a zero-feature frame.
* **Fixed:** the Evaluation page reported every bundle of a temporal task as
  ``Skipped N bundle(s): … expects feature columns absent from partition 'main':
  Month``. ``compute_learning_curve`` sliced the raw partition by the bundle's
  recorded schema, but a temporal bundle's schema names the *raw history* frame
  (time/series columns included) while ``load_partition`` drops exactly those
  columns; and even with the columns present it refitted the whole
  ``Pipeline(ForecastFeatureBuilder, …)`` on a feature matrix. Temporal curves now
  run on the generated one-step forecast design through the bare estimator
  (``feature_source: temporal_design``) under the task's walk-forward splitter,
  never a shuffled fold count. Each reported size is a *training length* up to the
  longest window the splitter offers, computed by replaying the folds:
  ``sklearn.model_selection.learning_curve`` anchors its sizes on the smallest
  fold's training window, which is the first origin — on a 33-row design with
  ``TimeSeriesSplit(5)`` that curve stopped at 8 training rows and hid what more
  history buys (it now reaches 24). The explainability report's learning curves
  benefit from the same path, and each curve echoes the ``cv`` it used.
* **Fixed:** serving a forecast for a task registered with ``period_start`` rejected
  the dataset's own history. ``TemporalServeState.validate_history`` parsed the time
  column with ``pd.to_datetime`` even when the spec declares an anchor, so a period
  label like ``1-Jan`` (*year 1, January*) produced an axis that was either outside
  pandas' nanosecond range or — for these labels — running backwards
  (``3-Oct`` → 2026-10-03, then ``2-Jan`` → 2026-01-02), and the request failed with
  ``history time values must be strictly increasing``. The axis is now the registered
  grid (anchor stepped by ``freq``, row order as time order — the rule
  ``_time_axis`` already applies at registration), so labels are accepted unparsed
  and forecast times continue the grid. A declared ``time_format`` is applied to
  timestamp text, and the unparseable-column error names the ``period_start`` remedy.
  The one-series check also moved before the axis check, so a panel frame now answers
  ``history must contain exactly one <series> value`` instead of a misleading
  "sort the frame" message.
* **Fixed:** combining models on a **temporal** task crashed the background job with
  ``ValueError: at least one array or dtype is required``. The ensemble routes built
  their feature matrix from the raw partition, which for a temporal task has no
  feature column (time/series excluded, target is the label), so the aggregator's
  member reached scikit-learn's ``check_X_y`` with an ``(n, 0)`` matrix. Members are
  now refit on the generated one-step forecast design through their bare
  estimators, scored under an order-respecting splitter (the task's stored one,
  else ``TimeSeriesSplit``; never a shuffled integer fold count), and the combined
  bundle is composed as ``Pipeline(ForecastFeatureBuilder, ensemble)`` with the
  serving geometry — so it forecasts recursively through the normal inference path.
  The ensemble record carries ``feature_source: temporal_design``, the builder/spec
  and a caveat that ``cv_score`` is a one-step design score.
* **Fixed:** combining temporal bundles with ``weight_optimization`` failed with
  ``ValueError: cross_val_predict only works for partitions``. The weight search
  computes out-of-fold member output with ``cross_val_predict``, which refuses a
  walk-forward splitter because its test blocks leave the initial training window
  (and any gap) uncovered. ``SKSurrogate.ensembling.out_of_fold_outputs`` now
  replays such folds fold by fold — clone, fit on the fold's training rows, predict
  its test rows — and drops the never-predictable rows from every member through
  one shared mask, so the weights are still fitted on identical, never-trained-on
  rows. The fitted vector reports ``rows_scored``/``rows_unscored`` beside the
  weights; on a 90-row temporal fixture five walk-forward folds score 70 design
  rows and leave the 17-row initial window unscored.
* **Fixed:** the Experiments “Advanced” tab sent the regression search space with
  a hard-coded ``accuracy`` scorer for a regression-declared task, and AML then
  rejected every candidate with ``ValueError: continuous is not supported``. The
  scorer now follows the effective problem type until the user picks one.
* **Added:** an experiment on a **temporal** task now searches the generated
  one-step forecast design (``ForecastFeatureBuilder`` with the declared horizon
  as the minimum lag) under the task's order-respecting splitter, and the bundle
  it produces carries that builder plus the serving geometry, so it forecasts
  recursively through the normal inference/serving path. Before this, a temporal
  partition's only non-target column is its excluded time column, so every
  candidate died with ``ValueError: at least one array or dtype is required``.
* **Fixed:** a partition with no feature columns left (target plus ignored
  columns only) now answers an actionable **400** naming the columns instead of
  the same opaque sklearn message, and pipeline optimization points a temporal
  task at the Experiments/baseline paths.
* **Fixed:** the rolling-origin backtest of a temporal experiment bundle fitted
  the bundle's whole pipeline on its own design matrix and failed with
  ``ForecastFeatureBuilder is missing required column(s): Month, Sales …``. The
  route now finds the builder by type (not by the ``features`` step name, which a
  ``make_pipeline`` bundle does not use) and refits the bare estimator, and an
  omitted ``strategy`` defaults to the bundle's own (``recursive`` for a one-step
  experiment model) instead of always scoring it as ``direct``. Temporal
  experiment bundles are now also composed as ``features``/``model`` explicitly,
  matching the baseline bundle contract.
* **Fixed:** explanations, ALE, row attribution and prediction intervals on a
  temporal bundle read the raw partition and failed with
  ``Found array with 0 feature(s)`` (the time column is excluded and the target is
  the label). They now run on the generated forecast design — the same one-step
  matrix the search and the backtest see — with the bare estimator behind the
  feature builder (frame-tolerant for preset pipelines), and echo
  ``feature_source: "temporal_design"``.
* **Fixed:** nested cross-validation of a temporal bundle handed the bundle's whole
  pipeline to the tracker, which built folds from the partition's raw columns, and
  failed with ``ForecastFeatureBuilder works on pandas DataFrames``. It now scores
  the generated one-step design via ``mltrack.nested_cv_evaluation(..., design=...)``
  (new parameter: prebuilt ``(X_train, y_train, X_valid, y_valid)``) with the bare
  estimator, turns an integer fold count into ``TimeSeriesSplit`` and refuses a
  shuffled splitter on a temporal task. ``StackingEstimator.fit_transform`` also
  clamps its internal fold count to the rows it receives, so a small
  outer/inner block no longer fails with "Cannot have number of splits ...".
* **Fixed:** the explainability report counted every `unavailable` diagnostic as a
  failure, so a healthy regression task reported "15 unavailable diagnostics"
  made up of "classification curves require a classifier" and "no completed
  nested-CV job". Those are now marked `by_design` and reported separately as
  `diagnostics_not_applicable`; `diagnostic_failures` counts only diagnostics that
  ran and failed.

Experiments search space follows the task's problem type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Fixed:** the Experiments page's Advanced “Search space” editor opened with the
  classification default even for regression tasks — a regression search was
  pre-filled with ``LogisticRegression`` and classifier estimators. It now
  pre-fills from the task's effective problem type (declared on the Datasets page,
  or picked in Quick start): regression opens on the library preset space
  (RandomForest/ExtraTrees regressors); classification keeps its classic editor
  default. An edited space is never clobbered — the pre-fill re-applies on a type
  switch only until you edit it, and “Reset to <type> defaults” re-applies it
  explicitly any time.
* **Added:** ``GET /api/experiments/presets`` returns a ``default_config`` per
  preset, derived from the same table that
  :class:`~SKSurrogate.aml.AML`'s ``preset=...`` uses (``PRESET_PARAM_SPECS`` in
  ``SKSurrogate.aml``), so the editor pre-fill and the library presets cannot
  drift apart.

Temporal declaration no longer 500s on timestamps outside the nanosecond range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Fixed:** ``PUT /api/datasets/{task}/temporal`` answered ``500 Internal Server
  Error`` (``OverflowError: Overflow when converting between datetime64 units``) for
  any time column whose parsed values fall outside pandas' int64-nanosecond range
  (1677-09-21 .. 2262-04-11). The reported case is a ``Month`` column of
  ``1-Jan``/``1-Feb``/... labels: pandas 3 parses those to **year 1** (and keeps
  ``datetime64[us]`` resolution), so the axis cast into nanoseconds overflowed deep
  inside pandas. The declaration now answers **422** with the per-partition report and
  an actionable message naming the parsed range and the fix ("values without a year
  parse to year 1 — supply full dates or a numeric step index").
* **Fixed:** timezone-aware timestamps are normalised to UTC before building the
  int64 axis, so ``2026-01-01T00:00:00+02:00`` validates instead of failing on the
  cast (the axis is UTC-based by definition).
* **Fixed:** the temporal route also converts a residual ``OverflowError`` into a 400
  rather than letting it escape as a server fault.
* **Fixed:** registering an **empty CSV** crashed type deduction with
  ``ZeroDivisionError`` (``DataPreprocess.is_categorical/is_bin`` divided by a
  population of zero); a header-only upload now answers 400, and the library guards
  the division so an all-missing *feature* column is typed instead of crashing.
* **Fixed:** a **target column with no observed values** was accepted at
  registration, leaving a task nothing can be modelled on; it now answers 400.

Datetime columns: the declared type is real, and the feature is centred
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Fixed:** a column typed ``datetime64`` was only *labelled* datetime — the stored
  CSV has no dtype, `load_partition` returned ``object``, and
  ``FoldLocalDataPreprocessor`` fell through to one-hot encoding (one column per
  distinct timestamp). The declaration is now stored with the dataset
  (``dataset_deduced_types``) and the column is re-cast on load in every consumer
  (training/baselines, preprocessing profiles and the encoding preview, sensitivity,
  synthetic data, inference). A datetime64 override whose values do not parse is
  refused at registration with the offending values.
* **Changed:** datetime features are converted to **seconds relative to a fitted
  anchor** (``DatetimeCentering``) instead of raw seconds since 1970. Raw epoch
  seconds made an intercept-plus-slope design ill-conditioned
  ($\operatorname{cond}\approx 6\times10^{11}$ versus $\approx 1$ centred), which
  inflates coefficient variance and neutralises L2 penalties. The anchor is a
  per-column ``min`` recorded in the fitted transformer (so it travels in the
  bundle); subtraction happens in the column's own resolution, so year-1/year-1000
  timestamps give a large finite value instead of an ``OverflowError``.
* **Changed:** the AML search path now uses
  ``FoldLocalDataPreprocessor(scale_numeric=True)`` like the baseline path, so
  scale-sensitive candidates (ridge/SVR/kernel) no longer receive raw magnitudes.
  Tree presets are invariant to it.
* **Fixed:** ``DataPreprocess.encode()`` (and ``mltrack(encode=True)``) raised
  ``UFuncTypeError`` for every datetime column on numpy >= 2 — ``DateTime2Num``
  divided float seconds by a ``timedelta64``. ``DateTime2Num`` now accepts datetimes
  *or* numeric seconds, uses the column's unit, and maps a constant column to the
  lower bound instead of dividing by a zero span.

Dead code removed from the AML search path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Removed:** `AML._uses_preset_surrogates` — assigned in three places and read
  nowhere (verified across the repo, including dynamic ``getattr`` lookups). The
  behaviour it was meant to record is already available from
  `_uses_implicit_surrogates` and `preset`.
* **Removed:** an unused `from .structsearch import BoxSample, CompactSample` inside
  `AML.eoa_fit`; sampling is resolved inside ``structsearch``, and neither name was
  used in that method body (both remain public API).

No behaviour change; the removal was verified against the AML, experiment and
surrogate-diagnostic suites.

Period labels, ambiguous formats, and partition defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Fixed:** the error raised for timestamp values that parse outside the
  nanosecond range did not mention ``period_start`` — the option that actually fixes
  period labels — so the message told the caller to rewrite data that the library can
  read fine once anchored. Both the overflow and the unparseable-value branches now
  name the anchor (``period_start`` + ``freq``) first.
* **Fixed:** the **Time column** and **Series id** pickers offered the task's
  *target* column. The server refuses it (a target is what you forecast, not when),
  but the dropdown made the mistake one click away and the resulting
  ``column 'X' is both the target and a temporal column`` read like a form bug. The
  pickers now exclude the target, and the refusal explains the distinction instead of
  repeating the name: ``time_column`` must name *when* the observation happened
  and ``series_id`` *which series*, while the target is the value being predicted.
* **Clarified:** the declared temporal summary shows ``anchor=<period_start>`` and
  ``format=<time_format>`` when they are set, so a declaration built from a period
  label is visible in the card rather than implied.
* **Added:** ``TemporalSpec.period_start`` — the timestamp of each series' first row,
  for a time column that is a *period label* (``1-Jan``/``2-Jan``/``3-Dec``, ``Q1``,
  ``period 1``) rather than a date. It requires ``freq``; the axis is built from the
  anchor stepped by ``freq`` per row within each series, so ``1-Jan`` no longer has
  to be read as year 1 (which is outside the int64-nanosecond range and made such
  datasets impossible to declare). The validation report records
  ``time_source: "period_start"`` so nobody mistakes labels for validated
  timestamps. The feature builder now takes its calendar/Fourier axis from the same
  derivation, so ``cal__month`` works on such data.
* **Added:** ``TemporalSpec.time_format`` — an explicit ``strptime`` format for text
  that parses ambiguously (``03/01/2026`` as ``"%d/%m/%Y"``). A bad directive string
  is refused when the spec is constructed, not at validation time.
* **Fixed:** ``nested_cv_evaluation(validation_partition=None)`` read the
  unpartitioned ``data`` stub instead of treating ``None`` as "no validation
  partition", then died with a bare ``KeyError: <target>``. ``None`` now skips
  validation scoring, which is what the control plane means when a task has
  registered only a training partition.
* **Fixed:** the request models for ``train-baseline``, ``nested-cv``, ``backtest``,
  ``learning-curves`` and experiment submission defaulted ``train_partition`` to the
  literal ``"train"``, so a task registered with a single ``main`` partition answered
  ``404 No dataset partition 'train' stored`` for a request that never named one. An
  omitted partition now resolves to ``train`` when it exists, otherwise to the task's
  only/first registered partition (``api.partitions``); an explicit name is passed
  through, and a wrong explicit name still 404s. The resolved name is echoed in the
  response.

"Time series" is not a learning-problem option — and the card now says so
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Clarified:** the **Task definition** card labels the field **Learning problem**
  (it previously read “Problem type”, which invited the reading that time series
  should be a third option). The card now states the rule at the point of
  confusion: time structure is a separate axis; a temporal task shows its
  ``regression`` badge with a ``temporal`` chip and a note that it is scored by a
  rolling-origin backtest with MAE/RMSE/MASE; a tabular task explains that there is
  no “time series” choice and offers a **Declare time structure** button that jumps
  to the temporal card. No behaviour change.

Synthetic sampling no longer assumes a partition called ``train``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Fixed:** ``POST /api/datasets/{task}/synthetic`` defaulted the source
  partition to the literal ``"train"`` and answered
  ``404 No dataset partition 'train' stored`` for any task whose data is
  registered under another name (e.g. a task whose single partition is called like
  the task). An omitted ``partition`` now resolves to ``train`` when it exists,
  otherwise to the first registered partition. An explicit unknown name still
  404s, but the message lists the partitions that do exist.
* **Fixed:** the Datasets **Synthetic data** card initialised its partition select
  to ``"train"``; when that name was not registered the browser displayed the first
  real option while still sending ``"train"``. The control now binds to the
  resolved partition (and disables generation when the task has none).
* **Fixed:** resolving the partition could create an empty task database as a side
  effect, which masked ``No dataset registered for task`` behind a
  partition-not-found message.

Declared problem type: one decision instead of five inferences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Added:** a task now stores its **problem type** (``regression`` or
  ``classification``) at registration, inferred conservatively from the target
  (boolean, non-numeric, or an integer 0/1 column → classification; other numeric
  columns → regression) and overridable. Metadata carries ``problem_type``,
  ``problem_type_source`` (``inferred``/``user``), ``problem_type_reason`` and the
  raw ``problem_type_inferred``. New API: ``PUT``/``DELETE
  /api/datasets/{task}/problem-type``; the value is echoed by ``GET
  /api/datasets/{task}``, the temporal payload and the preprocessing payload.
* **Fixed:** Quick start defaulted to the *classification* preset regardless of the
  task, so the first experiment on a regression dataset could be a classification
  search. The preset now defaults to the declared problem type (server-side too, when
  the request carries neither a config nor a preset), the Experiments page follows it,
  and a mismatch between the declared type and the chosen preset is shown as a hint.
* **Fixed:** ``POST /api/bundles/{task}/train-baseline`` always defaulted to
  ``linear_regression``; it now picks ``logistic_regression`` for a classification
  task and refuses an estimator that contradicts the declaration.
* **Fixed:** delayed-label monitoring defaulted to ``task_type="classification"``; it
  now follows the declared problem type, so a regression task is scored with MAE/RMSE
  rather than accuracy.
* **Added:** a temporal task is pinned to regression — the forecast feature builder
  and backtest are regression-shaped. Declaring a temporal spec on a task whose stored
  type is classification (or requesting ``problem_type="classification"`` alongside a
  temporal spec) is refused with an explicit 400 instead of silently scoring labels as
  forecasts.
* **UI:** the Datasets metadata card is now **Task definition** — target, problem
  type (with its source and reason, ``Change`` / ``Reset to inferred``) and time
  structure in one place. The Data Preprocessing page opens with a task-context
  banner stating the problem type and, for temporal tasks, that the stored partition
  is only the time axis plus the label while the model's inputs are generated.
  ``docs/forecasting.rst`` and ``docs/ui-manual.rst`` document the three axes.

Numeric analyses and data-quality views on temporal tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Fixed:** ``POST /api/sensitivity/{task}/run`` (and the Pearson
  ``/correlation-threshold`` / ``/correlation-matrix`` endpoints) answered
  ``400 Sensitivity and correlation analysis need numeric feature columns ... (non-numeric:
  all columns). Encode or drop those columns`` for any time-series task. A temporal
  partition has no raw feature column to encode: ``time``/``series_id`` are excluded by the
  temporal contract and the target is the label. These analyses now project onto the
  **generated design** (``make_supervised``: lags, rollings, diffs, pct-changes, calendar
  fields), report ``feature_source: "temporal_design"`` with per-column provenance, and
  accept a ``features`` configuration (default: one lag at the declared horizon).
* **Fixed:** an empty projection caused by the ignore list now says the partition has no
  feature column left instead of blaming non-numeric columns that do not exist.
* **Fixed:** ``ForecastFeatureBuilder.fit`` rejected calendar features whenever the time
  column's *stored* dtype was not datetime, so an ISO date read from CSV (``object``
  dtype) failed even though ``transform`` and ``validate_temporal_frame`` both parse it.
  The gate now uses the contract's parser.
* **Fixed:** ``CorrelationThreshold.fit`` raised ``IndexError`` on a single feature
  (``numpy.corrcoef`` returns a scalar); it now keeps every feature — nothing to prune.
* **UI:** the Feature Analysis page labels a generated-design run and states that the
  ranking explains history windows, not measured inputs.
* **Fixed:** ``GET /datasets/{task}/{partition}/duplicates`` answered "No feature columns
  available to define duplicate rows" for a temporal task (nothing is left after the
  time/series exclusions and the target). Duplicates are now judged on the row identity
  (``series + time`` or ``time``), reported as ``key_kind: "temporal_identity"`` with a
  warning that registration enforces that key; extra raw feature columns still define the
  key, and ``columns=`` overrides it.
* **Fixed:** ``GET /datasets/{task}/{partition}/encoding-preview`` showed the generated
  forecast design instead of failing with "No active feature columns to encode": every
  generated feature is numeric (``kind: "generated"`` with its op/shift), so the preview
  no longer invents a categorical encoding plan for a temporal partition.
* **Encoding preview detail:** each preview row now carries its origin timestamp
  (``preview_label_name``/``preview_labels``) — the time axis is consumed by the
  feature builder, not hidden — and the endpoint accepts the same ``features`` JSON as
  the backtest, so a preview can reproduce the model's actual design instead of always
  showing the one-lag default. A tabular task is refused with a clear message if
  ``features`` is supplied.
* **UI:** the Preprocessing page labels the duplicate key, shows the origin column in
  the encoded rows, and exposes a "Design features (JSON)" control with an
  apply/reset action.

Temporal data contract (time-series support, phase 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Opt-in temporal declaration for datasets. Nothing changes for tabular tasks: a task with
no ``TemporalSpec`` keeps its current behaviour exactly.

* **`SKSurrogate/temporal.py`** — ``TemporalSpec`` (time column, optional series id,
  frequency, horizon, row order, gap tolerance), ``infer_freq``, and
  ``validate_temporal_frame``, which returns a structured report and raises
  ``TemporalValidationError`` for unordered rows, duplicate timestamps, unparseable or
  missing times, missing series identifiers, and gaps beyond the declared tolerance.
  Datetime axes are compared as integer nanoseconds (never floats); numeric columns are
  treated as an index axis rather than 1970 timestamps.
* **`mltrack`** — ``RegisterData(..., temporal=...)`` validates before writing and
  persists the spec plus a per-partition report; ``GetTemporalSpec`` /
  ``SetTemporalSpec`` / ``ClearTemporalSpec`` and ``available_partitions`` are new. The
  time and series columns are excluded from ``X``/``get_data``. A temporal task may only
  use ``TEMPORAL_CV_SPLITTERS`` (``TimeSeriesSplit``): ``SetCV`` rejects ``KFold``,
  ``ShuffleSplit``, integer fold counts and the default splitter. New exports:
  ``TemporalSpec``, ``TemporalValidationError``, ``validate_temporal_frame``,
  ``infer_freq``, ``temporal_columns``, ``TEMPORAL_CV_SPLITTERS``,
  ``cv_is_temporal_safe``.
* **API** — ``POST /api/datasets/{task}/register`` accepts a ``temporal`` JSON form field;
  new ``GET``/``PUT``/``DELETE /api/datasets/{task}/temporal``. ``PUT`` validates every
  registered partition and answers 422 (``X-Reason: temporal-validation``) with the
  report; declaring a spec adds the time/series columns to the dataset's ignored columns,
  and ``DELETE`` removes only those entries.
* **Fix** — a brand-new task now sets ``mltrack.task`` in ``__init__`` (it was only set
  when reopening an existing task, which broke error messages and payloads that read it).
* **Docs** — new :doc:`forecasting` chapter; ``docs/mltrace.rst`` points to it at the
  time-series paragraph.
* **Tests** — ``tests/test_temporal_contract.py`` (32 tests).

Leakage-safe forecast feature builder (time-series phase 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ForecastFeatureBuilder`` / ``make_supervised`` construct lag, rolling, difference,
calendar and Fourier features for a temporal task, with the shift rule enforced in
code rather than documented and hoped for.

* **Shift $\ge$ horizon, per series** — ``lag_k`` reads $y_{t-k}$ for $k \ge h$; a
  rolling window of width $w$ covers $y_{t-h-w+1..t-h}$; a lag shorter than the
  horizon is rejected at ``fit``. On a panel every operation is computed inside its
  own series, so one series' history never enters another's features.
* **No silent degradation** — no ``center=True`` path exists, ``min_periods``
  defaults to the window width (no partial windows), and a zero denominator in
  ``pct_change`` yields ``NaN`` rather than ``inf``.
* **Supervised designs** — ``make_supervised(..., strategy="direct")`` labels
  $y_{t+h}$; ``"recursive"`` labels $y_{t+1}$ while the declared horizon still
  constrains the features. ``ForecastDesign`` carries ``X``, ``y``, per-row
  ``meta`` (series, origin/target time, horizon, target step), feature provenance
  with each column's minimum shift, and a JSON-safe ``describe()``.
* **Verifiable invariant** — ``assert_no_leakage`` masks every post-cut observation
  and requires pre-cut features to be unchanged, catching future-reading and centred
  windows with ``LeakageError``.
* **Exports** — ``ForecastFeatureBuilder``, ``ForecastDesign``, ``FeatureSpec``,
  ``make_supervised``, ``assert_no_leakage``, ``LeakageError``, ``ROLL_STATS``,
  ``CALENDAR_FEATURES``.
* **Docs** — ``docs/forecasting.rst`` gains the builder, direct/recursive and
  leakage-verification sections.
* **Tests** — ``tests/test_forecast_features.py`` (22 tests).

Rolling-origin backtesting (time-series phase 3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forecast-quality evaluation is now a rolling-origin walk-forward, not a fixed-window
CV approximation.

* **`RollingOriginSplit`** (``SKSurrogate/temporal.py``) states the horizon directly:
  each test window is exactly ``horizon`` steps, origins advance by ``step``, and the
  training window either expands or slides a fixed ``min_train``. Registered in
  ``TEMPORAL_CV_SPLITTERS``, serializes through ``cv_to_spec``/``build_cv`` and renders
  in ``cv_param_defs`` (including the optional ``min_train``).
* **`rolling_origin_indices`** is the single geometry source; short series raise a
  message naming the geometry instead of silently shrinking.
* **`mltrack.backtest`** builds the design matrix with the feature builder, walks
  origins, and returns per-origin records (series, origin/target time, y_true, y_pred),
  per-fold scores and MAE/RMSE/bias, pooled and per-series/per-horizon aggregates, and
  the features used. Refit policies are enforced by counting fits: ``"every_fold"``,
  ``"first"``, ``"every:k"``. ``cost_limit`` turns an oversized run into a clear error,
  and the result is persisted under ``backtest:<config hash>``.
* **Panel semantics** — with a ``series_id`` each series gets its own origins, the model
  is fitted on the union of the series' training windows, and series too short for the
  geometry are listed in ``skipped_series``.
* **API** — ``POST /api/evaluation/{task}/backtest`` queues the backtest for a bundle
  (``model_version`` or ``alias``, horizon, n_splits, step, refit, features) and appends
  a ``rolling_origin_backtest`` entry to the bundle's ``evaluations`` metadata.
* **`seasonal_lags`** — readability alias for a lag at the season period: column
  ``<target>__seasonal_lag_<p>``, ``op="seasonal_lag"`` in the provenance, same horizon
  validation as any other lag.
* **Known limit** — only the ``direct`` strategy is evaluated; ``strategy="recursive"``
  raises until multi-step inference lands (phase 6), rather than reporting one-step
  numbers as multi-step.
* **Docs** — ``docs/forecasting.rst`` gains the backtesting section and the backtest
  endpoint row.
* **Tests** — ``tests/test_backtest.py`` (14 tests) plus 2 seasonal-lag tests in
  ``tests/test_forecast_features.py``.

Forecast metrics and naive baselines (time-series phase 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SKSurrogate/forecast_metrics.py`` adds the metrics that make a forecast score
interpretable, and scores the naive baselines on the same origins as the model.

* **Point metrics** — ``mae``, ``rmse``, ``bias``, ``smape`` (percent, zero-safe),
  ``mase`` and ``rmse_scaled``. MASE is scaled by the in-sample mean absolute
  ``seasonality``-step difference, so a seasonal-naive forecast scores exactly 1.0;
  a constant series has no scale and yields ``nan`` rather than a fabricated number.
* **Quantile/interval metrics** — ``pinball_loss`` (0.5 quantile equals half the MAE),
  ``winkler_score`` (width plus ``2/alpha`` per miss), ``interval_coverage`` and
  ``interval_width``; ``summarize`` bundles them with optional ``per_series`` and
  ``per_horizon`` blocks and a ``pooled`` / ``macro`` aggregation choice.
* **Baselines** — ``baseline_prediction`` implements ``naive``, ``seasonal_naive``
  (phase-correct for any horizon, not just ``h == m``), ``drift`` and ``mean``.
  ``mltrack.backtest`` scores them on the identical origins and returns
  ``result["baselines"]`` with the same MAE/RMSE/bias/sMAPE/MASE block as the model.
* **Ranking guard** — ``forecast_ranking`` orders candidates by MASE and distinguishes
  parity (MASE == 1, ``beats_baseline: false``, ``worse_than_baseline: false``) from
  failure (> 1), and lists non-finite entries as ``unranked`` instead of sorting them
  to the bottom. Every backtest result carries ``result["ranking"]``.
* **Per-origin provenance** — each origin record now carries ``mase_scale`` (its
  in-sample denominator) and ``baselines`` (the baseline predictions), so a reported
  MASE can be audited from the record itself.
* **Scoring options** — ``neg_smape`` is available as a learning-curve and nested-CV
  scorer and appears in a new "Forecast" group of ``standard_scoring_options``.
  MASE is deliberately **not** a scorer: it needs the in-sample history, which a
  ``(y_true, y_pred)`` scorer never sees — it is a backtest metric.
* **API** — ``POST /api/evaluation/{task}/backtest`` accepts ``seasonality`` and
  ``baselines``.
* **Docs** — new "Forecast metrics and baselines" section in ``docs/forecasting.rst``,
  including why MASE is not a scikit-learn scorer.
* **Tests** — ``tests/test_forecast_metrics.py`` (31 tests).

Horizon-aware prediction intervals (time-series phase 5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SKSurrogate/forecast_intervals.py`` calibrates prediction intervals from the
held-out residuals of a rolling-origin backtest, so a served model can put a width on
its point forecast without recomputing anything.

* **Split-conformal calibration** — ``calibrate_from_residuals`` / ``calibrate_backtest``
  produce ``lower_offset``/``upper_offset`` from the conformal order statistics.
  ``method="absolute"`` uses the ``ceil((n+1)(1-alpha))/n`` quantile of ``|residual|``;
  ``method="asymmetric"`` uses the empirical ``alpha/2`` and ``1-alpha/2`` order
  statistics for skewed errors.
* **Refuses rather than fudges** — below ``min_residuals`` (default 10) the backtest
  result reports ``intervals.available = false`` with the reason instead of computing a
  quantile from three points.
* **Diagnostics, not assertions** — ``interval_diagnostics`` reports empirical coverage
  with a Wilson interval, mean/median width and the Winkler score, and passes the
  horizon step as the grouping key so a horizon-1 and a horizon-12 interval are
  reported separately. The docs state that walk-forward residuals are ordered, so the
  exchangeability assumption behind nominal coverage is approximate.
* **Bundles carry the calibration** — ``ModelBundle(interval_calibration=...)``
  persists it through save/load (a legacy record without the key still loads), and
  ``bundle.interval_bounds(predictions)`` applies it. An uncalibrated bundle raises
  instead of returning a zero-width interval.
* **Backtest wiring** — ``mltrack.backtest(alpha=, interval_method=, min_residuals=)``
  adds ``result["intervals"]``; the backtest route persists a usable calibration onto
  the bundle and records it in the bundle's ``evaluations`` entry.
* **Fix** — the per-fold ``score`` is now ``None`` instead of calling ``R^2`` on a
  one-row test window (which is undefined and emitted an ``UndefinedMetricWarning``),
  and ``mean_fold_score`` averages the folds that have a score.
* **Docs** — new "Prediction intervals" section in ``docs/forecasting.rst``.
* **Tests** — ``tests/test_forecast_intervals.py`` (19 tests), including a Monte-Carlo
  coverage check and width monotonicity across horizons 1–4.

Bundled multi-step inference (time-series phase 6)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A trained forecasting bundle can now be *served*: one call with trailing history returns the
full-horizon forecast — point predictions and, when the bundle carries an interval calibration,
calibrated bounds — without re-fitting or rebuilding features. Tabular bundles are unaffected;
v1 records load as before.

* **Bundle format v2** — ``BUNDLE_FORMAT_VERSION`` is now 2; the record carries an optional
  ``temporal`` block (spec, target column, feature configuration, strategy) written by
  :class:`TemporalServeState`. The state persists geometry only: a fitted builder is defined
  behaviourally by its constructor arguments, so no fit artifacts are stored. A v1 bundle loads
  as a plain tabular bundle; corrupted temporal blocks fail at load time when they are rebuilt
  and re-validated against the spec.
* **predict_forecast** — ``bundle.predict_forecast(history, horizon=None, future_exogenous=...)``
  returns one row per forecast step (times, means, bounds only if calibrated). Direct bundles
  answer exactly their design horizon; recursive bundles accept any positive horizon and feed
  each prediction back in as the next observation. Results match the offline rolling-origin
  reference to $10^{-9}$ on identical origins — including a state-feedback check confirming that
  from step $H$ onward the lag window contains earlier *predictions*, not true values.
* **Serving API** — top-level ``serve_forecast(...)`` / ``predict_batch_forecast(...)``, with
  ``predict_batch`` dispatching temporal bundles to the forecast path; plus control-plane routes
  ``POST /api/inference/{task}/forecast`` and the serving pin's
  ``POST /api/serving/{task}/forecast``, which validate payloads against the recorded spec
  before touching the model. Insufficient or invalid history is rejected with an actionable
  message (400), tabular bundles get "no temporal design" (400) rather than a fabricated
  forecast, and missing payload fields surface as pydantic's 422.
* **Docs** — new "Bundled multi-step inference" section in ``docs/forecasting.rst`` plus an
  endpoint-table row; format-v2 fields in ``docs/modelbundles.rst``; the batch forecast contract
  in ``docs/inference.rst``. The minimum-history contract (per-term bounds, direct vs recursive)
  is now stated consistently across docs and docstrings.
* **Tests** — ``tests/test_forecast_inference.py``: 21 tests covering bundle round-trips, the
  serve-state contract, batch-frame shape/metrics/request-id, interval wiring, a two-series panel
  serving per-series 12-step forecasts with calibrated intervals, and HTTP round-trips through
  the FastAPI test client.

Forecasting panels in the web UI (time-series phase 7)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The control plane now exposes the whole forecasting workflow; every surface is opt-in and a
tabular task's pages are unchanged.

* **Datasets — Temporal structure card** — declares the time column, optional series id,
  frequency, horizon, order and gap tolerance; renders the server's per-partition validation
  report (rows, series count, inferred frequency, missing periods, status), a regular/irregular
  badge, a temporal-safety badge for the stored CV splitter, and a "Use inferred" shortcut when
  the data's inferred frequency differs from the declared one. Clearing the declaration restores
  the columns as features.
* **Splitters, decision D2** — once a declaration exists, the splitter picker offers only
  order-respecting splitters and the server rejects the rest. ``GET /api/datasets/{task}/cv``
  gained ``temporal_options`` and now returns ``param_defs`` for temporal-only splitters too
  (``RollingOriginSplit`` was previously unreachable from the UI: it is not in
  ``STANDARD_CV_SPLITTERS``).
* **Evaluation — Rolling-origin backtest card** — submits the backtest as a job and renders
  pooled MAE/RMSE/MASE with interval provenance, a per-horizon error table, a candidate-vs-naive
  leaderboard scored on the identical origins, and a forecast-vs-actual fan chart per panel
  series using the stored conformal calibration. The feature configuration is an editable JSON
  field because the backtest builds its own design; ``POST /api/evaluation/{task}/backtest`` now
  unwraps a temporal bundle's ``ForecastFeatureBuilder`` and refits the bare estimator (fitting
  the whole pipeline failed on the already-materialized design).
* **Experiments — run-settings summary** — a temporal task states its horizon, frequency and
  panel column, and points multi-step scoring at the backtest instead of implying the search
  scores it.
* **Inference — Forecast console** — paste a history series, pick a bundle or the serving pin,
  and read/download the multi-step forecast (``time,prediction,lower,upper``). Refusals are the
  server's actionable messages (missing series column, history shorter than the design's
  required tail).
* **Docs** — "In the web UI" section with two figures in ``docs/forecasting.rst``; a
  time-series paragraph in ``docs/ui-manual.rst``.
* **Tests** — ``tests/test_temporal_contract.py`` (+1: temporal splitters and their parameter
  defs are exposed) and ``tests/test_backtest.py`` (+1: the route unwraps a temporal bundle).

Monitoring windows, benchmark DGPs and release hygiene (time-series phase 8)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Windowed drift** — ``drift_report`` gains ``reference_window`` / ``current_window``:
  each frame is trimmed to its trailing window (ordered by ``time_column`` when given) and
  ``drift_report(None, history, ...)`` compares the last :math:`m` observations against the
  :math:`n` before them. The report records ``windows`` (mode and sizes) and the row counts
  are post-trim; the ordering column is excluded from the statistics because a timeline
  marker necessarily differs between adjacent windows. ``POST /api/monitoring/{task}/drift``
  accepts the three fields and treats a missing ``reference_partition`` as the single-frame
  form; impossible windows are 400s with the count that failed.
* **Label maturity** — ``delayed_label_performance`` accepts per-row ``horizons`` (scoring
  per step in ``by_horizon``) and a ``maturity`` cap: rows whose horizon exceeds the cap are
  counted in ``pending_rows`` instead of being scored. ``maturity`` without ``horizons`` is
  refused, because the step of each row is the quantity under check. Wired through
  ``POST /api/monitoring/{task}/delayed-label-performance``.
* **Forecast DGPs** — ``benchmarks/forecast_dgps.py`` adds five seeded processes (random
  walk with drift, weekly and daily seasonality, a five-series panel, and a driftless walk
  whose role is the *must-lose* check: naive is Bayes-optimal there) and writes a versioned
  ``forecast_dgp_validation.json`` with the full MAE ranking per DGP. Measured defaults: 40
  rolling origins (four origins leave naive MASE at ~2.2 from sampling noise alone), naive
  wins on ``easy_naive`` with MASE 1.25, and — honestly recorded rather than tuned away —
  naive also beats a levels-regression on the drifted walk at horizon 1.
* **Docs** — ``docs/forecasting.rst`` gains "Monitoring a forecast", an opt-in-vs-default
  table and a benchmark section; the MASE/sMAPE/Winkler docstrings now carry their
  formulas.
* **Packaging** — a ``forecasting`` extra exists in ``setup.py`` and is deliberately empty:
  the feature needs only the core numpy/pandas/scikit-learn dependencies, so the extra
  documents the supported install and reserves a home for a future adapter. ``import
  SKSurrogate`` is identical with and without it.
* **Tests** — ``tests/test_temporal_monitoring.py`` (10) and
  ``tests/test_forecast_dgps.py`` (6).

Forecasting review pass (P1--P8 gap closure)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Recursive backtesting** — ``mltrack.backtest(strategy="recursive")`` now evaluates a
  recursive design: it trains on :math:`y_{t+1}`, replays the horizon from each cutoff by
  feeding predictions back, and records one origin plus ``horizon`` step records. The
  replay shares ``temporal.recursive_steps`` with serving, so offline and served forecasts
  agree by construction. ``POST /api/evaluation/{task}/backtest`` accepts ``strategy`` and
  the Evaluation card exposes it.
* **Defect fixed by that refactor** — recursive serving predicted from an *appended empty
  row*, so its first step was really :math:`y_{T+2}` (and it over-extended by one step).
  Serving now reads features at the current last origin before appending; the oracle in
  ``tests/test_forecast_inference.py`` encoded the old error and is corrected.
* **Temporal baseline bundles (P4.2)** — ``fit_baseline_bundle`` gains the branch it was
  missing: on a task with a declared temporal spec it builds a
  ``ForecastFeatureBuilder`` pipeline, stores the serving geometry on the bundle, and
  records ``baseline_scores`` from a rolling-origin backtest (``refit="first"``, seasonal
  naive always scored). Previously the route produced a tabular bundle that could not
  answer ``predict_forecast`` at all. Tests:
  ``tests/test_temporal_baseline_bundle.py``.
* **Baselines** — ``historical_mean_by_series`` joins ``BASELINE_METHODS`` as the explicit
  per-series mean (identical to ``mean`` because every call conditions on one series'
  history; documented and tested rather than silently duplicated).
* **Timings (P3.2)** — fold and origin records carry ``fit_time`` (0 when a fit was
  reused) and ``predict_time``.
* **Baselines on the bundle record** — the backtest route persists ``baselines`` in the
  bundle's evaluation entry and ``baseline_scores`` at bundle level.
* **Docs accuracy** — ``docs/mlops-roadmap.md`` no longer claims time-series split support
  is pending; ``docs/forecasting.rst`` documents recursive evaluation, the baseline-bundle
  route, the five baselines, and why ``rmse_scaled``/``winkler_score``/``mase`` are not
  scikit-learn scorers. The Evaluation leaderboard pins the naive rows above the candidate.
* **Packaging** — ``TIME_SERIES_PLAN.md`` is now listed in ``.gitignore``; the plan doc was
  untracked but not ignored, so it was one ``git add`` away from publication.

Pipeline-space benchmark track (C-space)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A second benchmark track alongside the fixed-pipeline suite (`benchmarks/DESIGN.md`,
frozen at v1.9). It searches a **heterogeneous** space — preprocessing transformers ×
estimator families — which is the AutoML question the fixed-pipeline suite does not answer.
Its contract is `benchmarks/DESIGN-cspace.md` (v0.3), including pre-registered hypotheses
with explicit falsifiers; results are in `docs/benchmarks-cspace.md` (630 cells, 0 errors).

**Headline:** no arm separates on held-out accuracy in either currency. Surrogate guidance
is evaluation-efficient (median 6 evaluations to within 1 % of best, versus 11--17 for the
other arms) but **not** wall-clock efficient: 49 % of the SRS time budget goes to surface
fitting and the acquisition optimiser, so it buys 281 candidates where random buys 553.
The ordinal-encoding defect is load-bearing — a mixed categorical surface gains $\geq 0.3$
pt best-CV in 4/6 datasets. Default TPE re-proposes seen configurations for 28 % (40 evals)
to 65 % (350+ evals) of its budget.

* **Space** (`benchmarks/cspace.py`): 4 preprocessors × 5 reducers × 61 classifier
  configurations = **1,220 enumerable pipelines** across LogReg, RandomForest, ExtraTrees,
  HistGradientBoosting, SVC and KNN, with per-family knobs gated by conditional rules.
  Every pipeline starts with a fixed `SimpleImputer(median, add_indicator=True)` — arrhythmia
  carries 408 missing cells, and without it three families fail there for reasons unrelated
  to search quality.
* **Arms** (`benchmarks/cspace_methods.py`): uniform random, Optuna-TPE, Optuna-Random
  (sampler control), SRS with the default GP surface, SRS with a mixed categorical surface,
  and the no-surrogate ablation; FLAML and AutoGluon at the system level only.
* **Currencies**: equal wall-clock (primary; one evaluation ranges 5 ms to ~2 s across
  families) and equal 40 candidate evaluations (secondary, comparable to `DESIGN.md`).
* **Runner/Analysis**: `python -m benchmarks.run_cspace --protocol {A,B,C}` writes a
  provenance manifest (git SHA + dirty, package versions, thread policy), a crash-resilient
  `results.jsonl`, per-cell JSON and `summary.csv`; `benchmarks.analyze_cspace` reports exact
  sign tests, Cliff's delta, bootstrap CIs and Holm-Bonferroni within family.
* **Gates**: `python -m benchmarks.gate_cspace_fit` fits every configuration on every pinned
  dataset before a full run may start (this gate is what caught the arrhythmia NaNs).

Three library defects surfaced while building the track and are documented, with verified
reproducers, in `benchmarks/upstream_issues.md`: the combinatorial `MinEvals` floor that
silently disables the surrogate above ~5 parameters, `'passthrough'`/`None` steps being
rejected by the zero-feature probe, and ordinal encoding of `Categorical` parameters in
heterogeneous spaces. The benchmark measures all three; the library is unpatched in this
release per the `DESIGN.md` v1.4 principle.
Combining trained models
~~~~~~~~~~~~~~~~~~~~~~~~

* New ``SKSurrogate.ensembling``: combine a selection of trained bundles into one
  scikit-learn meta-estimator — ``voting`` (hard/soft + positional weights),
  ``averaging``, ``stacking`` (configurable final estimator and folds), ``bagging``
  and ``boosting``. Bagging and boosting give every selected member its own
  resampling/reweighting wrapper and aggregate the wrappers, so each member
  contributes; the result is an ordinary ``ModelBundle``, so registration, serving,
  learning curves, diagnostic curves, nested CV and the run report consume it
  unchanged.
* New **Ensembles** page and ``/api/ensembles`` routes: ``methods`` (the combiners
  that apply to the task's problem family, with their parameter schema),
  ``candidates`` (selectable models), ``fit`` (background job, ``kind: ensemble``),
  list and per-version detail. Requests are validated at submit time — unknown
  member (404), duplicate or mixed-family members, unknown method/parameter, soft
  voting without ``predict_proba``, classification boosting without sample-weight
  support, more than twelve members (400) — so a rejection is never a failed job.
* Scores are reported two-sided: ``cv_score`` is out-of-fold on the task's stored
  splitter (member rows use the same folds), while ``train_score`` is the refit
  score and is flagged optimistic. Every response and every stored bundle carries
  the caveat that member selection was made on this same data, so the comparison is
  jointly optimistic and nested CV remains the honest estimate.
* Two core fixes the combination depends on, both pinned by tests: ``ArrayAsFrame``
  now infers dtypes when it rebuilds a frame from a positional array (a mixed-type
  matrix is an object array, so a frame-first encoder fitted on one classified every
  numeric column as categorical — measured 3 columns expanding to 423, and a bagged
  member scoring AUC 0.55 against 0.95 for the same member fitted on the frame), and
  ``preprocesses_frames`` now sees through aggregating meta-estimators, so nested CV
  no longer stacks a second encoder in front of an ensemble's members.
* Documented in ``docs/ensembles.rst`` (new chapter), the UI manual's *Combine
  models into an ensemble* section, and the model-bundle chapter's cross-link.

Calibrating members, optimizing weights, pricing the search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Per-member calibration** (``calibration``): each member's outermost form is
  wrapped in scikit-learn's ``CalibratedClassifierCV`` (sigmoid or isotonic, with
  the calibrator count selectable), so the probabilities entering the merge are
  calibrated. The calibrator sits *outside* any bagging/boosting wrapper — inside,
  the bootstrap weights reach it and scikit-learn warns the calibration is
  "likely to be incorrect". Recorded as a measured before/after pair per member
  (Brier score and log loss on identical folds), so "calibrated" is evidence
  rather than a claim; classification-only, refused for regression members.
* **Weight optimization** (``weight_optimization``): member weights fitted on the
  members' out-of-fold output by a convex simplex least-squares (Brier/MSE, no
  metric needed), a Dirichlet search over the task's scorer, or Caruana greedy
  selection (integer weights, so a member that never helps is visibly dropped and
  named). Applies wherever members are merged by a weighted vote/average —
  voting, averaging, bagging, boosting — and is refused for stacking, which
  learns its own combination, and alongside an explicit ``weights`` vector.
* **The honest estimate** (``honest_cv``): an outer cross-validation that refits
  the members, their calibration *and* the weight search inside every training
  fold, reporting the score of the procedure rather than of the one stored model,
  with the per-fold weight vectors recorded so the stability of the search is
  visible. Off by default (it costs outer-folds × (members + search) fits).
  Measured across ten configurations, the gap between the in-sample and the
  nested number ran 0.0000 (the search collapsed onto one member, so the ensemble
  *is* that member) to 0.0233 — which is why both are reported side by side and
  the caveat states the range instead of a single figure.
* All three blocks are validated by the same core functions the library uses, so
  a bad strategy, a fold count the data cannot fill, calibration on regression
  members, or weights together with optimization are ``400`` at submit time. A
  failed nested estimate is recorded as an error next to a fitted, scored,
  usable ensemble rather than taking the model down.
* The **Refinements** card on the Ensembles page renders those controls from the
  server's schema and disables what the request would refuse; the detail view
  adds the weight column, the calibration before/after table and the
  nested-versus-optimistic gap.
* Three robustness fixes found while auditing the above. ``ensemble="auto"`` for
  the calibrator is resolved against the *installed* scikit-learn instead of being
  hard-coded: that argument accepts only a boolean before 1.6 (so a literal
  ``"auto"`` made every calibration call fail at the declared ``scikit-learn>=1.5``
  floor) and names its default from 1.6 on. A member exposing only
  ``decision_function`` is reported as calibrated *without* a Brier/log-loss pair
  rather than surfacing a ``predict_proba`` AttributeError for a member whose
  calibration worked. And a weight search that could not score any candidate now
  reports ``objective: null`` instead of ``-Infinity``/``NaN``, which was not valid
  JSON in the job result.

Experiments: search strategy × surrogate family
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The Experiments page's advanced "Run settings" no longer conflates the
  surrogate with the structural search. **Search strategy** is now
  **Evolutionary (EOA)** — a genetic search over the space of all possible
  pipelines instead of walking the whole space — or **Exhaustive search**
  (evaluate every admissible pipeline). EOA honours **Max generation** and
  **Num parents**; exhaustive ignores them. The old ``Search engine`` dropdown
  (``strategy="surrogate"``, "Surrogate only (no evolution)") is retired.
* **Surrogate optimization runs on every pipeline the search evaluates.**
  Exactly one surrogate estimator is chosen per experiment — a dropdown of GPR,
  KRR or SVR (``RandomizedSearchCV``-tuned over the estimator's own
  hyperparameter grid, default ``gpr``) — plus parameters ``itrs`` (default
  30), ``sampling`` (default ``BoxSample``; CompactSample/SphereSample also
  available) and ``min_random_evals`` (default 10, the number of random initial
  hyperparameter samples before the surrogate is exploited). The API fields
  ``surrogate_family``/``surrogate_mode`` are replaced by ``surrogate_estimator``,
  ``surrogate_itrs``, ``surrogate_sampling`` and ``surrogate_min_random_evals``;
  they are recorded in the bundle audit/reproducibility record under a
  ``surrogate`` object. Unknown estimators or samplers are rejected with a
  clear 400.
* **The library now exposes the same model.** ``SKSurrogate`` adds
  ``build_surrogate(estimator, itrs=30, sampling="BoxSample", solver="L-BFGS-B")``
  plus the ``SURROGATE_ESTIMATORS`` (``gpr``/``krr``/``svr``) and
  ``SURROGATE_SAMPLERS`` (BoxSample/CompactSample/SphereSample) lists, and
  re-exports ``CompactSample``; the API's experiment surrogate is a thin wrapper
  over this builder, so the estimator grids live in one place. ``AML.fit``'s
  pipeline-search strategy is renamed ``"surrogate"`` -> ``"exhaustive"``
  (the old spelling is rejected), and ``AML``'s implicit default surrogate — a
  hardcoded KRR+GPR pair — is now a single Gaussian Process via the shared
  builder (``gpr``/``itrs=30``/``BoxSample``), so the library default matches
  the Experiments default. Surrogate optimization is applied to every candidate
  pipeline the chosen strategy evaluates.

Explain page
~~~~~~~~~~~~

* Prediction intervals on classifier bundles now return guidance instead of a terse error: ``POST /api/evaluation/{task}/{model}/intervals`` answers 422 with an explanation that the selected bundle is a classification model, that split-conformal prediction intervals are only defined for numeric regression targets, and to select or train a regression bundle (e.g. via Baselines) instead. The refusal is also tagged with the stable machine-readable code ``intervals_require_regression_bundle`` in the ``X-Reason`` response header so clients can branch on it without parsing the human-facing detail text; the web UI client surfaces that code on its error objects. The Explain page recognizes this refusal (by code, or by wording against older servers) and renders it as an informational note rather than an error banner; other interval failures still surface as errors.

Baselines train on mixed-type partitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``POST /api/bundles/{task}/train-baseline`` answered 500 on any partition
  with a non-numeric feature column (object/category/string/datetime/bool):
  the baseline pipeline hardwired a bare ``StandardScaler`` in front of the
  estimator, which cannot fit such frames — the crash died inside
  ``check_array`` and surfaced as an opaque 500 from "Train baseline" on the
  Model Bundles page. Mixed-type partitions now get the same fold-local
  encoder the experiment path installs (``FoldLocalDataPreprocessor`` with
  median imputation + scaling for numerics, one-hot encoding with
  ``handle_unknown="ignore"`` for categoricals, epoch seconds for datetimes);
  purely numeric partitions keep their historical scaler-first pipeline so
  existing bundles stay shape-compatible. The reproducibility record lists the
  transformation that was actually fitted.
* Baseline validation scores are computed through ``load_partition`` (previously
  a raw CSV read), so columns ignored on the Datasets page are excluded from
  the score as well as the fit. Regression tests in
  ``tests/test_baseline_mixed_types.py`` cover mixed-type and numeric-only
  partitions, bundle round-trips and inference on the stored frame.

Serving a pinned model
~~~~~~~~~~~~~~~~~~~~~~

* New **serving pin**: declare once which trained/registered model a task
  serves (``POST /api/serving/{task}``), score through it without naming a
  model (``POST /api/serving/{task}/predict``, 409 while nothing is served),
  read it back with the resolved version, resolve mode and append-only pin
  history (``GET /api/serving/{task}``), list active pins
  (``GET /api/serving``) and stop serving (``DELETE /api/serving/{task}``).
  A ``version`` pin stays frozen across registry promotions; an ``alias`` pin
  follows the pointer, so a promotion or rollback changes what is served.
* The web UI **declares** the state instead of leaving it implicit: a
  ``Serving <version> · <task>`` pill in the top bar on every page while a
  model is served, and a **Serving** card on the Inference page to pin,
  replace, stop and exercise the served endpoint.
* The served path shares its prediction core with
  ``POST /api/inference/{task}/predict`` (input projection, pre-flight
  validation, monitoring record keyed to the answering model), so the two
  surfaces cannot drift; both are documented in ``docs/inference.rst``.

Data Preprocessing page
~~~~~~~~~~~~~~~~~~~~~~~

* New **Data Preprocessing** page under “1. Data”: missing-data overview
  (counts/percentages, complete rows, per-column detected and implied types,
  numeric quantiles or top categories), a dataset preview with missing cells
  marked, data-quality flags (duplicate rows/features, constant and all-missing
  columns) and observed missingness patterns.
* Materialized imputation: drop rows, mean, median, most frequent, constant,
  KNN and iterative/MICE. A preview reports before/after missing counts, cells
  imputed and distribution shift without writing. Saving creates a **new**
  partition (source untouched), stores an ``imputation_history`` recipe and
  registers the CSV for lineage; the target is never imputed, ignored columns
  are refused, numeric-only methods validate dtypes, and an existing output name
  answers 409.
* Endpoints: ``GET /api/datasets/{task}/{partition}/preprocessing`` and
  ``POST /api/datasets/{task}/impute``.
* **Scientific EDA cards** on the same page: per-column distribution rails and
  categorical bars with IQR outlier counts, exact target composition (class
  counts, minority share, imbalance ratio, feasible stratified fold count) or
  regression shape statistics, ranked target and feature associations with a
  Pearson-only numeric heatmap plus a Spearman/Kendall table, and VIF /
  condition-number collinearity diagnostics. The overview endpoint now forwards
  ``column_profiles``, ``target_associations``, ``feature_associations``
  (capped at ``association_limit``), ``rank_associations``,
  ``multicollinearity`` and ``target_composition``; profile blocks are
  descriptive and each card states its sampling scope.
* **Tier-2 diagnostics** on the same page: data-quality flags (identifier and
  high-cardinality columns, quasi-constant and near-duplicate features, leakage
  hints, conflicting duplicate labels, IQR **and** MAD outlier counts),
  missingness structure with Little's MCAR test (EM-estimated multivariate
  normal, chi-square on missingness patterns, explicit refusal reasons), and a
  type audit (numeric-looking text, mixed types, boolean/date-like text,
  whitespace and case variants). New response blocks: ``quality_flags``,
  ``missing_mechanism``, ``type_audit``; helpers live in ``api/eda.py``.
* **Tier-3 statistics** on the same page: Anderson–Darling normality tests per
  numeric column (tabulated critical values, verdict, caveat), Cohen's *d* and
  Cliff's δ with percentile bootstrap intervals between the largest target
  classes, IQR-scaled half-sample stability with the effective sample size, a new
  ``GET .../partition-drift`` route (two-sample KS + PSI per numeric feature,
  chi-square homogeneity + Cramér's V + total variation per categorical feature,
  >50-level columns skipped), and ``GET .../eda-report`` returning a
  self-contained Markdown report that the UI downloads as
  ``eda-<task>-<partition>.md``. Formal tests stay out of ``dataset_profile``.
* **Pre-flight checks before a run**: ``GET /api/datasets/{task}/preflight``
  returns advisory warnings and hints (identifier columns still modeled,
  conflicting duplicate labels, class imbalance, heavily missing columns, strong
  skew, small complete-case sample, constants) with stable codes, named columns
  and the thresholds used. The Experiments page renders them above the tabs and
  **never blocks submission** (``blocking: false``).
* **Notebook parity**: ``examples/notebooks/scientific_workflow.ipynb`` now
  exports ``eda-formal-tests.csv``, ``eda-effect-sizes.csv``,
  ``eda-stability.csv``, ``eda-quality-flags.csv`` and ``eda-report.md`` from
  the same helpers the Data page uses, so the offline report and the UI cannot
  drift apart.
* **Tabbed Data Preprocessing page**: the twelve cards are grouped into
  **Overview**, **Distributions**, **Relationships**, **Quality & missingness**,
  **Stability & drift** and **Transform**. The active tab lives in the URL
  (``?tab=quality``), so a view survives a reload and can be linked; only the
  active tab's cards mount, so the drift query runs when its tab is opened.

Categorical encoding, outlier clipping and duplicate removal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Categorical encoding preview** (`GET .../encoding-preview`): fits the
  fold-local model preprocessor on a bounded sample and reports the exact
  encoded feature names, per-column kind/cardinality and the first encoded rows,
  with explicit notes that vocabularies are learned inside each training fold.
* **Outlier clipping / winsorizing** (`POST /api/datasets/{task}/clip`): IQR
  fences, quantile winsorizing and z-score clipping with preview-before-save,
  per-column bounds/tail counts/share and before/after moments; saves a new
  partition with ``clipping_history`` lineage.
* **Duplicate rows** (`GET .../duplicates`, `POST /api/datasets/{task}/deduplicate`):
  exact feature-duplicate counts, conflicting-label groups, and warnings about
  sample change, target-balance shift, numeric mean shifts and cross-validation
  leakage before removal; preview or save a deduplicated partition with
  ``deduplication_history`` lineage.
* Fixed a tracker lifecycle bug where garbage-collected trackers closed the
  process-wide Peewee database mid-request (``Cannot operate on a closed database``).
* **Experiment Design** now explains its purpose and workflow in-page and can
  seed design variables from the selected task's numeric feature ranges
  (target/ignored/constant columns excluded, observed min/max imported, editable
  bounds). Docs contrast DoE with model search and state the observed-range
  caveat.
* **Diagnostic curves** now slice the preferred partition by the **bundle's own
  schema** and split it deterministically, instead of feeding the tracker's
  current feature list to ``split_train``. Fixes a 500
  (``ArrayAsFrame expected 13 feature column(s); got shape (1, 9)``) after
  dataset columns were ignored post-training.

Per-user data reset
~~~~~~~~~~~~~~~~~~~

* ``POST /api/settings/reset`` erases every task and research project the caller
  owns, with all derivative artifacts: datasets, mltrace databases, bundles,
  registry versions and index entries, predictions, monitoring records,
  checkpoints, job records and per-job logs. Accounts, sessions and API keys are
  kept.
* Ownership is resolved from ``auth/acl.db``: only explicit ``owner`` grants are
  removed, unowned legacy material is untouched, and resources shared with other
  users are skipped unless ``include_shared`` is set. A live job on a scoped task
  answers 409. In no-auth mode the single anonymous user owns everything, so the
  reset wipes the local workspace.
* The request requires a typed confirmation (``reset <username>`` or
  ``reset everything``), records one ``account.reset`` audit event with counts,
  and is exposed as a **Danger zone** card on Settings with a result summary.

Synthetic generation handles text-binary columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* A two-value **string** column (``Gender`` = Female/Male) was detected as
  ``binary`` and defaulted to SynthData's numeric ``bin`` type, which failed with
  ``'<' not supported between instances of 'str' and 'float'``. String binary
  columns now default to ``cat`` (backend and UI inference alike), the UI keeps
  no stale per-column types across tasks, and an explicit ``bin``/``int``/``real``
  override on a text column answers 400 with an actionable message instead of
  crashing generation. Overrides for columns excluded by the dataset are ignored.

Ignored dataset columns
~~~~~~~~~~~~~~~~~~~~~~~

* The Datasets page type selectors (**Detected / selected type** during upload and
  **Column generation types** in the Synthetic data card) gained an **ignore**
  option. Ignoring is a dataset-level, reversible choice stored in the task
  metadata: the column stays in the stored CSV, but every modeling path drops it.
* ``PUT /api/datasets/{task}/ignored-columns`` replaces the list; registration
  accepts an ``ignored_columns`` form field, and dataset metadata returns
  ``ignored_columns`` plus ``effective_feature_count``. Unknown columns and the
  target answer 400.
* Enforcement is centralised in ``api/columns.py`` and applied by training
  (`load_partition`), sensitivity/correlation analysis, dataset profiles, nested
  CV, diagnostic curves, monitoring drift, synthetic generation and synthesis
  facts. ``mltrack.nested_cv_evaluation`` accepts ``exclude_columns`` for the
  nested-CV path. Stored weights for ignored features are filtered from the
  heatmap and consensus ranking.

Editable target column
~~~~~~~~~~~~~~~~~~~~~~

* The target is no longer locked at registration. ``PUT
  /api/datasets/{task}/target`` re-points which registered column downstream
  analysis predicts (``{target: "<column>"}``, 400 for an unknown or currently
  ignored column, 404 without a registered dataset). It updates both the tracker
  ``Task.target`` row and the metadata ``target_name``/``dataset_feature_count``,
  and reseeds the Pearson/variance weightings against the new target so the
  consensus ranking stays consistent. The stored partitions, schema and
  fingerprint are untouched — metadata-only and reversible.
* The Datasets page clarifies the initial choice — **Target column (what to
  predict)** with a hint naming what it feeds — and the **Registered metadata**
  card now has a **Change** control to edit the target after upload instead of
  re-uploading the CSV.

Regularized Greedy Forest (rgf-python)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``rgf.sklearn.RGFClassifier`` (and the rest of the ``RGF`` family) is now a
  supported, installed estimator: ``rgf-python>=3.12.0`` was added to
  ``requirements.txt`` and ``setup.py`` ``install_requires`` so search-space
  configs naming ``rgf.sklearn.*`` resolve in ``aml.types()`` instead of
  failing the whole experiment job. ``FastRGF`` variants need the separate
  ``FastRGF`` executables (not shipped here); the standard ``RGFClassifier`` /
  ``RGFRegressor`` estimators work out of the box.

Mid-pipeline samplers accepted in search pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* A search space containing ``imblearn.over_sampling.SMOTE`` could abort the
  whole experiment job with ``Pipeline intermediate step 'stp_0' must implement
  fit and transform``. The structure validator enforced the sklearn rule that
  every intermediate step has ``transform``, but a legitimate ``[SMOTE,
  estimator]`` pipeline uses ``imblearn.pipeline.Pipeline``, which routes the
  sampler through ``fit_resample`` (SMOTE has no ``transform``). The validator
  now accepts an intermediate step that implements either ``transform`` or
  ``fit_resample``, so sampler-first pipelines validate and fit instead of
  crashing the search. Regression tests in ``tests/test_mid_pipeline_sampler.py``.

Nested CV accepts cat_cols bundles without a second encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Nested cross-validation on a bundle trained through the explicit
  ``cat_cols`` path crashed with ``X does not contain the columns listed in
  cols: ['spr_risk_level']``. ``AML``'s ``cat_cols`` path builds the pipeline
  with ``category_encoders.one_hot.OneHotEncoder(cols=[...])`` first, which
  needs a DataFrame (named columns) at fit. ``nested_cv_evaluation`` did not
  recognise that as "already preprocessing frames", so it stacked a
  ``FoldLocalDataPreprocessor`` in front; that step emits a numpy matrix and
  the inner encoder could not recover the named columns. ``preprocesses_frames``
  now also returns ``True`` for pipelines whose first step is a
  ``category_encoders`` encoder, so the candidate gets the raw frame instead of
  a second, double-encoding wrapper (regression test in
  ``tests/test_mixed_type_experiments.py``).

Learning curves respect cat_cols bundles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The Evaluation page's metric tabs failed every bundle trained through the
  explicit ``cat_cols`` path: scikit-learn's ``learning_curve`` retrains the
  pipeline on each fold, and with a numpy array as input its internal
  cross-validation strips column names, so
  ``category_encoders.one_hot.OneHotEncoder(cols=[...])`` raised ``X does not
  contain the columns listed in cols: ['<feature>']`` and all folds failed.
  The learning-curve endpoint now feeds a DataFrame sliced to the bundle's own
  recorded schema (falling back to every non-target column for legacy bundles)
  instead of positional values, so named columns reach the encoder on every
  fold. A partition missing schema features answers HTTP 400 naming them.
  Regression tests in ``tests/test_phase1_endpoints.py`` pin both a
  classification and a regression cat_cols bundle through the endpoint.

Human names for jobs and model versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Every job record now carries a ``display_name``: the user's ``label`` when one
  was given, otherwise ``<kind> · <task> · <local date time>``. Experiment and
  pipeline-optimization submissions accept an optional ``label``; the same name
  is stored in the produced bundle's metadata and returned as
  ``display_name`` in the submit response. ``PATCH /api/jobs/{id}`` renames an
  existing job (or clears the label with ``null``) and records
  ``aliased_at``/``aliased_by``; ``job_id`` stays the canonical identity.
* ``ModelRegistry`` gained user-chosen named aliases, separate from the
  lifecycle pointers: ``set_named_alias`` / ``named_aliases`` /
  ``resolve_version`` / ``delete_named_alias``, backed by
  ``GET|POST|DELETE /api/registry/{task}/aliases``. Aliases resolve in
  ``load``/``load_version``, registry promotion and the inference endpoints
  (``model_version`` falls back to the registry when no stored bundle file
  matches). Names are validated lowercase slugs that cannot collide with
  lifecycle states, and every set/delete is audited.
* The web UI renders these names instead of truncated ids in the run banner,
  the job tracker, the Jobs table, the Logs selectors, the Registry aliases
  card, and the bundle detail heading.

Pause, stop and resume jobs from the UI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Nothing could stop a run once submitted: the job pool had no control path and
  the searches had no cancellation hook. ``AML`` and ``SurrogateRandomCV`` now
  take a ``stop_check`` callable and raise the new ``SearchStopped`` between
  candidate evaluations; it is deliberately not a ``ValueError``, which the
  optimizer already treats as a failed candidate.
* ``POST /api/jobs/{id}/stop`` with ``pause`` or ``cancel``. Pausing keeps the
  checkpoint and marks the job ``paused``; cancelling marks it ``cancelled``.
  Requests are recorded on the record, are honoured for jobs still waiting in
  the queue, and are rejected (409) for jobs this process does not own.
* ``POST /api/jobs/{id}/resume`` now accepts ``paused`` and ``interrupted`` in
  addition to ``failed``, and covers ``optimize_pipeline`` as well as
  ``experiment``: pipeline jobs now save a re-submittable request, so both kinds
  resume from the same checkpoint. The new job id is returned; the original
  record stays as history.
* The UI gained **Pause** / **Stop** while a job runs and **Resume** once it can
  be resumed, on the job card, the Jobs table and the Experiments tracker.

Job liveness is no longer inferred from the JSON record
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* A crashed API process left jobs displayed as ``running`` forever. The job
  manager now keeps an in-memory registry of the jobs this process actually
  owns: ``/api/jobs/active`` returns only live ids, ``GET /api/jobs/{id}`` and
  the WebSocket stream rewrite a stale ``queued``/``running`` record to
  ``interrupted``, and start-up reconciliation marks every leftover record
  ``interrupted`` with an explanatory error.
* Each live job refreshes ``heartbeat_at`` every 10 s; the tracker warns when a
  job is still registered but has not reported for 30 s, which covers a hung
  worker rather than a dead process.
* New statuses ``paused`` / ``cancelled`` / ``interrupted`` are understood by the
  Jobs filters, counts and badges.
* The top-bar badge now polls in background tabs
  (``refetchIntervalInBackground``); React Query otherwise pauses the interval
  while the tab is hidden and a finished job keeps its "running" pill.

Compute budget for background jobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The job pool and the search's fold pool multiplied into unbounded
  parallelism: a hard-coded `ThreadPoolExecutor(max_workers=4)` combined with
  `AML(n_jobs=-1)` could ask for `4 x cpu_count` workers, plus one BLAS thread
  pool per job thread. New `api/compute.py` owns an explicit budget —
  `job_workers`, `job_threads` (the *total* joblib pool, shared by all jobs),
  `blas_threads`, `dask_workers`, `dask_threads` — with two enforced invariants:
  `job_workers*blas_threads + job_threads <= cores` and
  `dask_workers*dask_threads <= cores`. `cores` is
  `sched_getaffinity`-derived, so cpusets are respected.
* The budget is server state: `GET /api/settings/compute` reports it (with the
  arithmetic), `PUT /api/settings/compute` persists it to
  `<API home>/settings/compute.json` and rejects anything that does not fit with
  422. A *Settings → Compute budget* card edits it; the card is admin-only once
  authentication is enforced.
* Enforcement is not advisory: the executor is built/resized from
  `job_workers`; every job start exports `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/…
  at `blas_threads` (joblib honours a parent-set value, so loky workers inherit
  it) and applies a process-wide `threadpoolctl` limit; `fit_experiment_bundle`
  and `fit_pipeline_bundle` pass `n_jobs=job_threads` to `AML`, covering the
  fold pool, `cross_val_score` and stacking. Verified against a live run:
  `job_threads=2` produced at most two loky workers, each with
  `OMP_NUM_THREADS=1`.
* Each job record stores the budget it ran under (`compute`) and the values it
  applied (`compute_applied`); bundles record `n_jobs` in their reproducibility
  block. Settings apply to subsequently submitted jobs.
* `dask[distributed]` is now installed by `requirements.txt` (and exposed as the
  `dask` extra in `setup.py`), so the Dask backend works in the shipped image.
  A Dask job now builds its `LocalCluster` with `dask_workers` /
  `dask_threads` — previously `Client()` created one worker per core — and
  closes it on completion.
* Defaults preserve the previous capacity without oversubscribing:
  `cores - job_workers` fold workers (28 on a 32-core host), one BLAS thread per
  job slot, four job slots (clamped on small hosts).

Live job activity in the UI
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* A running experiment looked inert: the page showed no spinner, the tracker
  card sat below the fold, and the only cue was a static ``running`` badge.
  ``JobTracker`` now pulses, counts elapsed time, tails the job's captured
  entries and lists the trials being evaluated (index, phase, score and
  hyperparameters). Progress is an indeterminate stripe — the search does not
  know its trial count — with a determinate bar available when a caller passes
  ``expectedTrials``.
* The *Experiments* page shows a banner above the fold while its job is queued
  or running (state, elapsed time, links to the tracker and logs), scrolls the
  tracker into view on submission, and disables both run buttons until the job
  finishes so a double click cannot start a second search.
* The top bar carries an always-on **N jobs active** badge, on every page, that
  links to *Jobs*; the browser tab title is prefixed while work is in flight.
* New ``GET /api/jobs/active`` returns summaries only (no ``result``, no
  ``resume`` payload) for queued/running jobs, so the badge's four-second poll
  stays small even when finished experiments carry large evaluation histories.
* Per-trial log entries now expose structured ``score`` and ``params`` fields
  (the message text is unchanged), which is what the activity feed renders.
* The *Advanced* run settings labelled ``length`` as "Length (trials)"; in
  ``AML`` it caps the number of components in a generated pipeline. The control
  is now "Max pipeline length" with a hint, and docs match.

Experiments on mixed-type partitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Starting an experiment on a dataset with a free-text column (a surname, an
  address) failed with ``All candidate pipelines failed to produce a finite CV
  score: ValueError: could not convert string to float: 'MacDonald'``. The API
  handed ``AML`` an object array (``X.values``), and AML installs its
  fold-local preprocessor only for a ``DataFrame``, so every candidate met raw
  strings. ``fit_experiment_bundle`` and ``optimize_pipeline`` now pass the
  frame, keeping the encoder inside each CV fold.
* The *Advanced* tab (a caller-supplied search space) failed the same way: AML
  installs an encoder only when ``cat_cols`` names the categorical columns.
  ``fit_experiment_bundle`` and ``optimize_pipeline`` now detect them (still a
  pipeline step, so fold-local) and record the choice as
  ``configuration.cat_cols``.
* Nested CV failed on any bundle trained from a preset
  (``ArrayAsFrame expected 13 feature column(s); got shape ()``): the routine
  wrapped its own fold-local preprocessor around an estimator that already
  carried one, so the candidate was fitted on an encoded matrix. It now checks
  ``DataProcess.preprocesses_frames`` and leaves such estimators alone — the
  bundle's own preprocessor is still fitted per fold, so nothing leaks.
* Reproducibility records named every pipeline step ``builtins.tuple``, because
  ``Pipeline.steps`` yields ``(name, estimator)`` pairs and the recorder
  stringified the pair. Experiments now record the step classes they actually
  fitted.
* New ``SKSurrogate.DataProcess.ArrayAsFrame`` and ``frame_tolerant``: a
  DataFrame-first pipeline is prefixed with an adapter so partial dependence,
  accumulated local effects, learning curves and legacy trackers can still
  predict from positional arrays. ``_new_bundle`` applies it, so bundles accept
  frames and arrays alike (both verified against the served model).

Experiment design: suggest-next validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* "Suggest the next experiment" answered HTTP 422 when the pasted results held a
  single run (``samples``/``values`` need two observations) and could also send
  ``null`` bounds when a numeric field was mid-edit, whose error surfaced as a
  raw pydantic detail array. The page now validates the population of runs,
  variable names and bounds itself and names the offending variable; the API
  contract (422 below two observations, 400 for count/arity/bound problems) is
  pinned in ``tests/test_design_endpoints.py``.
* The web client renders FastAPI validation errors as ``field: message`` pairs
  instead of a JSON dump, which applies to every endpoint, not just this one.

Stored weightings and the consensus ranking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* A completed sensitivity run now merges its weights into the task's tracking
  database under a method column (``sobol`` / ``morris`` / ``delta_mmnt``), so
  ``mltrack.RetrieveWeights``, ``mltrack.heatmap`` and ``mltrack.TopFeatures``
  see it and the **Consensus top features** card actually updates. The job
  result reports ``weights_column`` and ``weights_stored``; storing is auxiliary
  and never invalidates a completed analysis.
* New ``mltrack.PersistWeights`` (alias ``persist_weights``) merges externally
  computed weights into the ``weights`` table — the API path needs it because
  ``FeatureWeights`` reads an unpartitioned ``data`` table that registered tasks
  do not have.
* Registration seeds the table with feature-target Pearson and feature variance
  for the numeric columns (reported as ``weights_stored``), and
  ``POST /api/sensitivity/{task}/base-weights`` back-fills tasks registered
  before that existed. The consensus card offers that action when it finds no
  stored weighting.
* The consensus card refetches when a run stores weights, and names the columns
  it ranked over.

Sensitivity analysis on mixed-type datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Sensitivity runs (Sobol, Morris, delta) and correlation pruning no longer
  crash on registered partitions with text columns — a churn CSV produced
  ``ValueError: could not convert string to float: 'Hargrave'`` from the SVR
  inside ``SensAprx``, and the equivalent numpy error from ``corrcoef`` for
  pruning. ``api.features.numeric_train_xy`` projects a partition onto its
  numeric columns, drops rows with non-finite values (reporting
  ``rows_dropped``), and the results list the excluded columns; a partition with
  no numeric feature answers HTTP 400 instead of failing as a background job.
* ``SensAprx`` resolves the delta index across SALib versions: 1.6 replaced
  ``delta`` with ``delta_balanced`` / ``delta_raw`` / ``delta_step``, so
  docker-built containers raised ``KeyError: 'delta'`` while a 1.5 dev venv
  worked. The estimator used is reported as ``estimator`` in the job result.
* The Feature Analysis page labels the ranking as numeric-only, names the
  excluded columns, and shows the same note under correlation pruning.

Users, sessions and sharing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **User accounts** (``api/auth.py``): username + password with scrypt digests,
  server-side sessions in an ``HttpOnly; SameSite=Strict`` cookie (only token
  digests are stored, so sessions are revocable), and per-user API keys shown
  once at creation. Login is throttled per username and client address, unknown
  usernames cost the same scrypt work, and failures never log the supplied
  secret.
* **Ownership and sharing** (``api/acl.py``): the task is the ACL unit with
  ordered roles ``viewer`` < ``editor`` < ``owner``; projects own themselves.
  Every task-scoped router enforces access through one router-level dependency,
  list endpoints (``/api/tasks``, ``/api/projects``, ``/api/jobs``) are filtered,
  strangers get 404 rather than 403, and ``POST /api/access/adopt`` lets an
  administrator take ownership of pre-auth material.
* **Endpoints**: ``/api/auth/{status,login,logout,me,password,keys,users}`` and
  ``/api/access/{tasks,projects,adopt}``.
* **First administrator**: ``python -m api.auth_cli create-admin`` or the
  idempotent ``SKSURROGATE_ADMIN_USERNAME``/``SKSURROGATE_ADMIN_PASSWORD``
  bootstrap; the Compose prod service forwards both plus
  ``SKSURROGATE_SESSION_TTL`` and ``SKSURROGATE_SESSION_COOKIE_SECURE``.
* **Web UI**: login gate above every route, an identity badge with sign-out in
  the top bar, and Settings cards for account password, API keys, sharing of the
  selected task (with adoption for admins) and user management.
* **Compatibility**: with neither ``SKSURROGATE_API_KEY`` nor
  ``SKSURROGATE_REQUIRE_AUTH`` the API stays open and single-user; the shared key
  keeps working as an admin-equivalent break-glass principal.
* **Audit trail** (``api/audit.py``): identity events (logins including denied and
  throttled attempts, logouts, password changes, API-key creation/revocation,
  account changes) are appended to ``logs/auth.jsonl`` and exposed to
  administrators at ``GET /api/auth/audit``. Entries carry the key prefix and
  username, never a password, session token or full key.
* **Docs**: new :doc:`auth-and-sharing` chapter (toctree + control-plane
  cross-link) and a Docker/readme paragraph on the new variables.

API authentication
~~~~~~~~~~~~~~~~~~

* Startup validation for the shared-key API: when ``SKSURROGATE_API_KEY`` is
  unset, the API logs a warning at startup; setting
  ``SKSURROGATE_REQUIRE_AUTH=1|true|yes`` makes it refuse to start without a
  key — for deployments that expose the port beyond a trusted network (the
  Docker entrypoint binds all interfaces). The Compose file forwards the
  variable into the prod container.
* The web UI's **Settings / Audit** page now reports the server's
  authentication state: a warning when the server requires an API key this
  browser has not stored, an error note when the stored key is rejected (401),
  and — when no key is stored at all — a reminder that the server runs without
  key enforcement.
* Regression tests for the shared-key middleware (HTTP header styles,
  WebSocket handshake policy, ``/api/health`` exemption, SPA fallback) and the
  startup validation, including fresh-interpreter import pins.

Security
~~~~~~~~

* Caller-supplied task names, model versions, dataset partitions, request IDs
  and job IDs are validated before use as filesystem path segments: values
  containing path separators, control characters or ``..`` are rejected with
  HTTP 400, and bundle and partition joins are resolved and proven to stay
  inside the storage root. Previously a keyed caller could read arbitrary
  ``*.csv`` files through a traversal ``partition`` and load any ``*.bundle``
  file on the host (``joblib``) through a traversal ``model_version``.
* With ``SKSURROGATE_API_KEY`` configured, ``/openapi.json``, ``/docs`` and
  ``/redoc`` now require the key as well; previously the full API schema was
  public even on a keyed server.
* The shared key is compared in constant time (``hmac.compare_digest``), and
  rejected requests are logged at ``WARNING`` with method, path and client
  host — never the supplied value.

Fixes
~~~~~

* EOA checkpoints are resumed only when compatible: a checkpoint written for a
  different gene set is ignored instead of replaying incompatible parents, and
  a recorded checkpoint directory that no longer exists no longer replaces the
  caller's directory. Previously a stale checkpoint could crash the next save
  or the first evaluation — e.g. after a Docker run had recorded an absolute
  ``/src/...`` path.

Logging and telemetry
~~~~~~~~~~~~~~~~~~~~~

* New ``AML.to_audit_dict()``: a JSON-safe audit bundle (schema version 2) for
  pipeline searches, mirroring ``SurrogateRandomCV.to_audit_dict()`` — one
  entry per evaluated candidate with the schema-version-1 bundles of each of
  its surrogate stages embedded, plus AML-level settings, budget aggregates,
  and an aggregated sampling-source tag list. ``json.dumps`` succeeds without
  a custom encoder; works before ``fit`` as well.
* ``AML`` now records the per-candidate stage audit trail (new
  ``stage_audits_history_``, parallel to ``evaluation_history_``) backing that
  export, and forwards its ``callbacks`` parameter to every surrogate stage —
  one callback list receives trial events from all searches of a run. No
  existing result key changes.

Evaluation page shows registry aliases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The Evaluation page now displays the named model aliases set on the Registry
  & Deployment page: the Bundle metrics table gains an **Alias** column beside
  the Model version (a muted dash when no alias has been set), and the Metric
  comparison charts — overview bars and per-metric learning-curve lines — label
  their series with the alias(es) instead of the raw version hash. Versions
  without a named alias are labelled exactly as before, so existing views are
  unchanged; lifecycle pointers (``latest``, ``production``, …) are not used
  here because they move between versions on promotion/rollback.
* The **Nested CV** and **Diagnostic curves** bundle selectors now label each
  option *alias - version* when the bundle has a named alias (several aliases
  joined by ", ") and show the plain full version otherwise — the same
  convention as the Explain page's model dropdown. Picking an aliased option
  still submits its exact version id, so runs resolve identically no matter how
  the bundle is displayed.

Model alias form validates names before submission
==================================================

* The **Model aliases** card on Registry & Deployment now checks the name
  against the registry grammar as you type — lowercase letters/digits with '.',
  '_' or '-' after the first character, max 64 characters — and flags reserved
  lifecycle pointer names (``latest``, ``candidate``, …), disabling Save until
  the name is legal. An invalid name no longer round-trips to a raw API 400; an
  inline hint under the input states the rule either way. The 400 message itself
  now includes examples (e.g. ``rf-v1`` or ``baseline-2026-11``) so any remaining
  backend rejection is actionable.

Explain page model dropdown shows registry aliases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The **Model** card on Explain now lists each bundle in its version dropdown
  as *alias - version* when a named alias has been set (several aliases joined
  by ", ") and the plain full version otherwise. Selecting an option still
  submits the exact version id, so explanation, interval and ALE calls resolve
  identically; lifecycle pointers are not used for labels.

Model Bundles list shows registry aliases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The **Bundles** table on the Model Bundles page gains an **Alias** column
  beside the model version, showing the named alias(es) set on the Registry &
  Deployment page (a muted dash when none has been set). The version cell still
  holds the exact version id, so Inspect and every other action resolve that
  id as-is.

Registry, Quality Gates and audit tables show registry aliases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The bundle/version selectors on the Quality Gates page (**Bundle (model
  version)**) and the Registry & Deployment page (**Register a bundle**,
  **Promote**, **Rollback**, and **Artifact cleanup**'s "Model version
  (predictions)") now label each option *alias - version* when a named alias
  has been set and show the plain version otherwise; the selected ``value`` is
  still the exact version id, so checks, registration, promotion, rollback and
  artifact deletion resolve identically, and the " — register first" flag on
  unregistered versions is preserved after the label.
* The **Audit log** table — on Registry & Deployment and under **Settings /
  Audit** — gains an **Alias** column that shows the named alias(es) for each
  event's model version (a muted dash when none is set).

Inference model pickers show registry aliases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The **Model version** dropdowns on the Inference page — in **Predict
  console**, **Batch run**, and the **Serving** card — label each option
  *alias - version* when a named alias has been set and show the plain version
  otherwise. The selected ``value`` is still the exact version id, so a
  per-request prediction or batch run resolves that id exactly as before.

Pre-flight prediction validation checks the model's feature schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``preflight: true`` on the inference/serving prediction endpoints previously
  validated the input frame against the task's full registered *dataset*
  schema. That over-strictly rejected any model trained on a curated or
  downstream feature subset of the raw table — e.g. an SPRA model consuming 18
  ``spr_*`` features was refused for lacking the other 47 unrelated columns of
  the original schema. Pre-flight now validates the frame against the *model
  bundle's* feature schema (the columns the estimator actually consumes), so a
  frame holding exactly those features passes and a genuinely missing feature
  is still reported as ``Prediction schema mismatch``. ``predict_batch``
  remains the strict gate for feature presence, ordering and dtypes. The
  "Validate before run" checkbox (Predict console, Batch run, Serving) works
  for subset-trained models as documented.

Serving card shows the reachable endpoint; aliases label every version on Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* When a model is being served, the **Serving** card now shows the **reachable
  endpoint** — the absolute ``POST /api/serving/{task}/predict`` URL (built
  from the effective API base) that a deployment client points at, so an
  operator can see exactly where the served model answers without hunting for
  it.
* Alias-aware labels now apply everywhere the Inference page names a model, not
  just the Model version dropdown: the Serving badge, the "frozen at / follows
  alias" hint, the model-version details, the Predict-console and Batch-run
  results, the "Answered by" line and the pin history. Each shows
  *alias - version* when a named alias is set and the plain version otherwise;
  option values and resolution keep the exact version id.

Inference dropdowns align whether or not the best-bundle chip shows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The model-picker row bottom-aligned its fields, so the **Model version**
  dropdown that carries the "★ best" chip sat higher than the **Resolve by**
  dropdown beside it (in Predict console, Batch run and Serving). The picker
  row now top-aligns its fields, keeping **Resolve by** / **Model version** /
  **Alias** controls level in all three cards with or without the chip.

[0.3.0] - 2026-11-08
--------------------

Web UI log browser and severity-aware logging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* New **Logs** page in the control plane: captured log entries per background
  job, newest first, with time, severity badge, job, logger and message.
  Severity chips toggle each level (showing live counts, with **Warnings & above**
  and **All levels** shortcuts); job, kind, free-text and row-count filters
  narrow the list, and selecting a job follows it live. Clicking an entry shows
  its trial/phase metadata or the full failure traceback. A **Logs** action on
  each job row opens the page scoped to that job.
* New ``api/logs.py`` capture layer: a ``logging.Handler`` bound to the
  ``SKSurrogate`` and ``sksurrogate.api`` loggers through a ``ContextVar``, so
  concurrent jobs never mix lines and API request handling is not captured.
  Entries are kept in a 2,000-entry ring buffer per job and appended to
  ``<SKSURROGATE_API_HOME>/logs/<job_id>.jsonl``; history survives a restart.
  Job records gain a ``log_counts`` summary.
* New endpoints: ``GET /api/logs`` (cross-job browse with ``level`` minimum
  severity or ``levels`` subset, ``kind``, ``q``, ``limit``), ``GET
  /api/logs/{job_id}`` (tail with ``since_seq``) and
  ``WS /api/logs/{job_id}/stream`` (incremental frames). Unknown levels return
  HTTP 422. The existing job status stream is unchanged.
* Dataset and project scoping: every entry carries ``task_name`` and the
  endpoint accepts ``task_name=`` to restrict to one dataset/task, plus
  ``project_id=`` which resolves the project's linked ``job`` artifacts and
  scans only those jobs (404 for unknown projects, empty result with
  ``job_count: 0`` for a project without linked jobs). The Logs page gained
  matching **Dataset / task** and **Project** selectors, both reflected in the
  URL for shareable scoped views.
* Severity is now used by the core, not just INFO/DEBUG: ``WARNING`` for a
  failed surrogate fit (logged before the exception propagates), for termination
  after too many non-improving iterations, and for one aggregated
  surrogate-failure line per search; ``ERROR`` for a non-finite surrogate
  prediction, an ``AML`` search that cannot fit any candidate, a failed user
  callback, and a failed job with its traceback. Repeated per-trial failures are
  logged at WARNING for the first five occurrences per job and status, then at
  DEBUG, so a systemic failure cannot flood the log.
* ``_logutil.emit`` no longer duplicates WARNING-or-above records: with logging
  unconfigured, ``logging.lastResort`` writes them to stderr and the stdout
  fallback is skipped. INFO/DEBUG console parity is unchanged.
* ``AML`` accepts ``callbacks=None`` and forwards it to the underlying
  ``SurrogateRandomCV``, so API-launched experiments emit per-trial telemetry
  (``trial``, ``phase``, ``status``) into the job log.
* Tests: ``tests/test_api_logs.py`` (lifecycle capture, severity and free-text
  filters, invalid-level rejection, failure traceback detail, concurrency
  isolation, JSONL persistence, warning cap, experiment trial events) and three
  new core tests for WARNING routing, fit-failure logging and no-progress
  termination.
* Docs: new log-browsing section in ``controlplane.rst`` (endpoint and entry
  contract, persistence, retention), the Logs page in ``ui-manual.rst`` with a
  figure, the severity table plus the stderr rule in ``telemetry.rst``, a
  ``14. Logs`` step in the UI walkthrough (severity, dataset/task and project
  scoping, REST equivalents), an "Observe what a run did" section in the readme,
  and ``callbacks`` documented in the ``SurrogateRandomCV``/``AML`` class
  docstrings so the generated API reference covers it.

Explanations: accumulated local effects and reference-subset stability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* New control-plane endpoint ``POST /api/evaluation/{task}/{version}/ale``
  computes accumulated local effects (ALE) for up to six numeric features —
  a supplement to partial dependence when features are dependent. The
  contract is documented in ``docs/mltrace.rst``: equal-frequency quantile bins
  with duplicate-edge collapse, deterministic left-merging of sparse bins
  (``min_bin_fraction`` default 0.01, reported in the response), non-finite
  rows excluded per curve, nominal categorical and (near-)constant features
  reported under ``skipped`` with a reason, one scalar response per curve
  (``predict`` for regression, labelled positive-class probability or decision
  value for binary classifiers, multiclass rejected), and non-causal
  interpretation labels. Feature selection defaults to the top-k permutation
  importance on the same reference sample and can be overridden explicitly.
* The Explain UI gained an **Accumulated local effects (ALE)** card with
  10/20/30-bin controls, per-bin row counts, skipped-feature reasons, and the
  non-causal limitation list. ALE is implemented as pure NumPy in
  ``api/explanations.py`` — no new dependency and no change to the
  ``SKSurrogate`` core.
* Explain responses now carry an optional ``partial_dependence_stability``
  block when 3–5 reference samples are requested: each primary PDP grid is
  re-evaluated on every distinct subset with direct model calls, and the block
  reports per-grid mean, sample SD, min, and max plus per-subset means. ALE
  curves report the same descriptive per-bin summary. These are
  sample-sensitivity diagnostics, not confidence bands; ICE stability is
  explicitly out of the API contract (individual ICE curves remain a plotting
  feature). PDP charts in the UI render the spread as dashed mean ± SD lines.
* Documentation: the ALE contract replaces the earlier "not exposed" note in
  ``docs/mltrace.rst``; the Explain section of ``docs/ui-manual.rst`` and the
  walkthrough describe the new card and the stability bands; the walkthrough's
  interval example payload now matches the endpoint (``alpha``, not
  ``coverage``).
* Controlled-DGP validation (Stage D1) gained an ALE ground-truth probe:
  ``benchmarks/dgps.py`` compares the accumulated curve against the exact
  noise-free response of the ``correlated`` and ``shift`` processes
  (including the known quadratic term), matching to floating-point precision.
* New row-level explanation endpoint
  ``POST /api/evaluation/{task}/{version}/explain/row`` (Stage D3, R9a):
  ordered-occlusion chain decomposition for one stored row against a reference
  mean or an explicit baseline row. Contributions telescope exactly to the
  model-response change (residual and tolerance reported), ``orders`` in 1–10
  draws distinct seeded feature orders and reports per-feature mean, sample SD,
  min, and max as disclosed order-dependence (not a confidence interval).
  Binary classifiers use the labelled scalar response; multiclass is rejected.
  A subset of features explains the hybrid row and is marked
  ``subset_hybrid``. Pure NumPy in ``api/explanations.py``; contract in
  ``AI_LAYER02_D3_ROW_LEVEL_CONTRACT.md`` and ``docs/mltrace.rst``.
* Explain UI: new **Row-level explanation** card with row index, 1/5/10 feature
  orders, a contribution bar chart, a mean/SD/min/max table, baseline and row
  predictions, and the reconstruction residual. Docs cover the card in the UI
  manual and walkthrough, including a new Explain-page figure
  (``docs/images/ui-explain-explanations.png``). Reconstruction residuals and
  tolerances are rendered in scientific notation instead of rounding to ``0``.
* D1 exit hook closed for the landed Stage C methods: ``benchmarks/dgps.py``
  now also checks average partial dependence and its reference-subset summary
  against the exact known response curves (``pdp_true_response_max_abs_error``
  and ``pdp_subset_shape_max_deviation`` at floating-point precision on the
  ``correlated`` and ``shift`` processes); DESIGN.md v1.17.


External references for research projects (Stage B1 tier 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Projects can now link an external citation as a fourth artifact kind,
  ``reference``: ``{kind, url, title?, doi?, label?}``. The URL must be an
  absolute ``http(s)`` URL of at most 2,048 characters, the API never fetches
  it, and references dedupe by URL. Non-reference kinds reject ``url``/``doi``
  fields, and reference entries reject ``task_name``/``artifact_id``; bad shapes
  are a 422.
* Reference snapshots use ``sksurrogate.project-artifact-snapshot.v2`` and store
  the URL/title/DOI fields only (``content_embedded: false``). The project
  summary reports them as ``exists: true`` with ``snapshot_status:
  "unverified"`` because reachability is never checked. References do not
  advance the workflow checklist, but they can satisfy the evidence requirement
  of a finding or conclusion and cannot be removed via ``PUT`` while referenced.
* Projects UI: the **Linked artifacts** card gained an **Add external reference**
  form (URL, title, DOI, label); linked references render as external links with
  their DOI and ``snapshot unverified`` badge and appear in the evidence picker.
* Docs: ``research-projects.rst`` documents the kind, validation, snapshot, and
  evidence semantics; ``ui-manual.rst`` covers the form. Tests:
  ``tests/test_projects_endpoints.py`` adds reference roundtrip/validation,
  evidence/removal-protection, backward-compatible shapes, and an HTTP-surface
  test.

LLM-assisted project synthesis (opt-in, Stage D2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* New optional workflow that drafts a short, **cited** project note from
  already computed facts. A user-supplied, OpenAI-compatible chat-completions
  endpoint is configured through ``SKSURROGATE_LLM_BASE_URL`` /
  ``SKSURROGATE_LLM_MODEL`` (plus optional key, timeout, and token budget);
  without configuration the feature returns a ``503`` and sends nothing. It is
  implemented over the standard library — no new dependency — and the core
  package remains untouched.
* ``POST /api/synthesis/projects/{project_id}`` queues a ``project-synthesis``
  job whose result carries the draft, its citations, the exact aggregated
  ``computed_facts`` payload, provider metadata, and a
  ``generated_interpretation`` label with ``causal_effect: false``.
  ``GET /api/synthesis/status`` reports configuration without exposing
  credentials.
* Sources are aggregates only: dataset profile summaries (``S1``), an optional
  job's whitelisted configuration and scalar metrics (``S2``), bundle metadata
  (``S3``), explanation summaries (``S4``), and stored monitoring alerts
  (``S5``). Raw rows, file paths, model objects, and credentials never leave the
  process. Drafts must cite supplied ids; unknown citations, uncited drafts,
  empty drafts, and over-length drafts fail the job with a named reason.
* Project notes gained optional generated provenance: ``source``
  (``human``/``generated``), ``model``, ``citations``, ``sources``,
  ``computed_facts``, and ``synthesis_job_id``. Generated notes are rejected
  without a model id, without citations, with citations that do not match a
  supplied fact, or with facts above 30,000 characters. Drafts are never saved
  automatically.
* Projects UI: new **Draft synthesis** card (provider status, task/bundle/job
  selectors from linked artifacts, explanation toggle, optional instruction)
  that renders the draft as *Generated interpretation* with citation chips and a
  computed-facts table; generated notes in the Notes list carry a badge,
  citations, model id, and collapsible facts. The Jobs kind filter includes
  ``project-synthesis``. Contract and data-handling review in
  ``AI_LAYER02_D2_LLM_SYNTHESIS_SCOPE.md``; docs updated in
  ``research-projects.rst``, ``ui-manual.rst``, and ``controlplane.rst``.
* Tests (``tests/test_api_synthesis.py``) use a local stub provider: disabled
  configuration, cited happy path with aggregate-only prompt assertions,
  unknown/uncited citation rejection, provider HTTP error mapping, and generated
  note provenance.


Acquisition readiness assessment (Stage D4, analysis only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added ``benchmarks/design_readiness.py``, a reproducible equal-budget harness
  comparing the shipped GP-LCB design policy with expected improvement (EI),
  probability of improvement (PI), and a noise-aware (Matern + WhiteKernel)
  variant on three synthetic 2-D objectives, with paired seeds, identical
  candidate pools, optional Gaussian observation noise, per-evaluation records,
  and a versioned report (``sksurrogate.design-readiness.v1``). CLI:
  ``python -m benchmarks.design_readiness --out DIR``.
* Result recorded in ``AI_LAYER02_D4_READINESS_ASSESSMENT.md`` and
  ``benchmarks/DESIGN.md`` §15: at 20 equal evaluations the shipped LCB and the
  alternatives all beat random by roughly an order of magnitude; EI/PI were
  neutral without noise; EI showed a weak (6 wins / 4 losses / 5 ties), not
  significant edge under synthetic noise; PI was worse under noise; and the
  fixed-length-scale noise-aware model regressed on the oscillatory objective.
  No production acquisition code, endpoint, or UI change ships, and D4 remains
  gated on a real expensive-experiment user.
* Tests: ``tests/test_benchmark_design_readiness.py`` (9 tests) pin equal-budget
  record shape, strategy validation, seeded determinism, observation-noise
  recording, noise-aware metadata, and the CLI report contract.


Benchmarks: slim-environment dataset loading and per-evaluation records (schema v2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Benchmark datasets now load from the local CSV cache without requiring the optional
  ``openml`` dependency: ``benchmarks/datasets.py`` checks the warm cache first and only imports
  ``openml`` on a genuine download, so cached datasets work in slim environments (e.g. CI-mirror
  containers) that do not install the ``[benchmark]`` extra; the actionable install-hint error on a
  cache miss is unchanged.
* Benchmark run records are now schema v2: every per-evaluation entry carries the full candidate
  configuration (``params``, all keys of the shared search space), each run JSON is stamped with
  ``schema_version``, and the directory manifest records ``record_schema_version``. Archive readers
  (``benchmarks/plot.py``, ``benchmarks/analyze_surrogates.py``) stay tolerant of legacy v1 runs that
  lack both fields, so archived runs remain re-plottable — and usable for acquisition-function
  comparisons — without re-running them. See ``benchmarks/DESIGN.md`` (v1.14).

Rule-rejection fixes for forbidden/conditional experiment rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Fixed a silent name mismatch that disabled ``forbidden`` and ``conditional``
  experiment rules in ``AML`` pipeline search: rules written with bare API-level
  parameter names (e.g. ``l1_ratio``) never matched the step-prefixed keys of a
  candidate pipeline's search space (e.g. ``stp_0__l1_ratio``), so banned
  candidates reached scikit-learn and failed there instead of being rejected by
  rule. Rules are now expanded per candidate pipeline to every declaring step,
  list values match "any of these values", callables pass through unchanged, and
  rules naming a parameter the pipeline does not declare keep the legacy skip
  behaviour.
* ``SurrogateRandomCV`` compares forbidden and conditional rule values with one
  scalar-or-list contract (a list/tuple/set of expected values matches when the
  candidate's value is any of them), and the final refit now mirrors trial-level
  conditioning so conditionally inactive parameters are not decoded back into a
  model the reported score never saw.
* When every candidate of an ``AML`` run is rejected by rule, both search paths
  fail with an explicit error naming the cause ("every candidate was rejected")
  instead of surfacing the underlying fit failure; per-candidate history records
  carry a ``rule_rejected`` flag, and each surrogate stage's trial history is
  profiled to tell rule rejections apart from ordinary failures.

pandas 3.x compatibility in fold-local preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``FoldLocalDataPreprocessor`` no longer routes plain string columns through
  TF-IDF under pandas >= 3, where they are inferred as a first-class
  ``StringDtype``: only explicitly declared text columns (``astype("string")`` /
  ``pandas.StringDtype()``) receive TF-IDF vectorisation; inferred strings fall
  back to one-hot encoding like object columns. The flatten step also passes
  ``feature_names_out="one-to-one"`` for current scikit-learn versions.

Launch visibility (docs and assets only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Readme: new "Understand what you tuned" section covering model explanations
  (permutation importance + partial-dependence plots, control plane and web UI)
  and sensitivity analysis (Sobol/Morris/δ indices), linking to the existing
  documentation guides; wording is scoped to descriptive claims per the launch
  plan guardrails.
* New figure ``docs/images/stability_contour.png`` — surrogate posterior mean of
  cross-validated accuracy over a two-dimensional hyperparameter slice on
  credit-g, showing local curvature in hyperparameter space (sharp peak vs
  plateau contrast) — plus its reproducible generator script
  ``docs/images/make_stability_contour.py`` (package public API only, no new
  dependencies).

Dependencies and packaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added ``lightgbm>=4.6`` and ``xgboost>=2.1`` to the package requirements: both
  are part of the default AML candidate pool; lightgbm was previously listed in
  ``requirements.txt`` only, xgboost is newly declared.

Docker and web UI upgrade
~~~~~~~~~~~~~~~~~~~~~~~~~

* Docker: multi-target ``Dockerfile`` (``dev`` with hot reload, ``prod`` with
  the built UI served by FastAPI on one port, non-root, healthcheck, ``/data``
  volume, ``EXTRAS`` build arg); compose profiles ``dev`` and ``prod``;
  ``SKSURROGATE_MODE`` and ``SKSURROGATE_WEBUI_DIST``; CI ``docker-smoke`` job.
* API: ``/api/experiments`` accepts ``preset``, ``strategy`` and resumes with
  ``AML(resume=True)``; new ``GET /api/experiments/presets``,
  ``POST /api/evaluation/{task}/{version}/explain`` and ``/intervals``,
  ``POST /api/design/latin-hypercube`` and ``/suggest-next``, and
  ``GET /api/projects/{id}/summary``.
* Web UI: Quick start experiments, Explain, Experiment Design and Projects
  pages; navigation grouped by workflow stage; Dashboard getting-started guide.

Walkthrough, validation fix and UI polish
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Docs: new ``docs/walkthrough.rst``, an end-to-end diabetes tutorial with
  screenshots and the matching REST and Python calls; sample CSVs
  ``docs/walkthrough-diabetes-{train,validation}.csv``; a "Run the web UI and
  API with Docker" section in the readme.
* Fix: ``mltrack.validate_data`` and the inference preflight no longer reject
  unseen values in continuous (float) columns as "unknown categories".
* ``/api/health`` returns the library version; Settings shows a Server card;
  the lineage rail also appears on Inference, Explain and Projects.
* Web UI: metric cards no longer wrap digit by digit; Explain labels every
  feature; Evaluation shows a muted note for classification-only curves.

JOSS submission and documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Opened the JOSS submission (``paper/paper.md``, ``paper/paper.bib``) as PR
  #8633 on ``openjournals/joss-papers``; added ``CITATION.cff`` pointing at the
  software record DOI 10.5281/zenodo.23631823 and the technical note
  ``docs/technical-note-rbf-gp-default.md``.
* D-track Hilbert-space surrogate study: new benchmark modules
  (``benchmarks/dtrack/{d1_approx,d2_compare,analyze}.py``), generated report
  ``docs/benchmarks/dtrack.md``, and standalone write-up
  ``docs/dtrack-study.md``; raw results (D1: 3816 cells, D2: 2160 cells) are
  archived at DOI 10.5281/zenodo.23652613.
* Phase C README rewrite with PyPI metadata polish; ORCID and affiliation
  filled in across docs; benchmark archive pointer updated to the v0.2.0
  software record (DOI 10.5281/zenodo.23631823).
* WebUI dependency bump: vite 8.3.4.

Scientific workflows and diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* AML categorical preprocessing is now learned within each cross-validation
  training fold. Added the reusable ``FoldLocalDataPreprocessor``; the eager
  whole-frame ``DataPreprocess`` API remains available, with leakage caveats
  documented for use before validation or cross-validation.
* Surrogate-search diagnostics now identify the selected response surface and
  record fit and local-optimization outcomes. Surrogate fit or prediction
  failures are not silently replaced by random search or another regressor.
* Added bounded permutation-importance and partial-dependence/ICE diagnostics,
  paired repeated-CV comparisons, and single-target regression diagnostics.
  Explanations describe model behavior, not causality; reference data, feature
  selection, and sampling bounds should be chosen deliberately.
* Added seeded Latin-hypercube designs and Gaussian-process LCB candidate
  recommendations for bounded continuous domains. Recommendations are not
  objective evaluations or experiment execution, and the benchmark evidence
  does not establish general or physical-world performance.
* Added split-conformal regression and conformalized quantile-regression
  intervals with disjoint calibration data. Quantile-interval integration in
  the UI/API remains deferred; coverage assumptions and limitations are
  documented in the Python guide.
* Added chronological splitter support for nested evaluation and quantile
  regression with pinball-loss evaluation in the Python/API and Evaluation UI
  workflows.
* Added API research projects that store questions and notes and reference
  existing dataset partitions, model bundles, and jobs without copying their
  artifacts. The API notebook demonstrates following those references.
* Added bounded mixed-type dataset profiles and reproducibility exports for
  API-trained bundles, including training-partition lineage, configuration,
  environment, and evaluation details. Added executable offline and Python API
  client notebooks; Jupyter remains optional.
* Sensitive-feature scans now include bounded, local checks for email,
  formatted US SSN, and formatted US phone patterns; reports contain counts,
  not matched values. External retraining triggers support bounded,
  deduplicated scheduler/webhook events. Events must be timezone-aware and no
  more than ten minutes old; data events must match the registered training
  partition fingerprint. Deduplication is process-local, and automated
  promotion is limited to threshold-gated staging.
* Bundle promotion checks schema compatibility by default; a mandatory metric
  threshold policy is not configured for ordinary promotion. Automated
  retraining promotion is separate and requires at least one finite metric
  threshold before staging.

API
~~~

* Added explicit ``AML(preset="classification"|"regression")`` quick starts
  using scikit-learn tree ensembles with task-appropriate scoring and bounded
  per-candidate evaluation budgets. Custom and legacy configurations remain
  available.
* API simplification (no removals): ``SurrogateRandomCV`` accepts scikit-learn
  names ``param_distributions`` and ``n_iter`` (``params``/``max_iter`` remain),
  search spaces may be shorthand (``(a, b)`` tuple -> ``Integer``/``Real``,
  list -> ``Categorical``; see ``as_space``), and fitted searches expose
  ``best_params_``. ``AML`` is now a ``BaseEstimator`` (repr, ``clone``);
  ``clone`` of ``SurrogateRandomCV``/``AML`` no longer fails. Fixed
  ``AML()`` default config referencing the removed ``sksurrogate`` module name.
  ``AML.fit(X, y, strategy="eoa")`` equals ``eoa_fit``; ``AML(stacking={...})``
  overrides the ``stack_*`` arguments; ``sampling="box"|"sphere"|"compact"``
  is accepted wherever a sampler class was.
  Fixed ``subgroup_performance_report(metric="precision")``, which computed
  recall, and made ``SurrogateRandomCV.best_score_`` follow scikit-learn's sign
  (higher is better; the loss is still in ``best_estimator_score``).
  ``cv_results_`` gains ``rank_test_score`` and ``loss``; ``mean_test_score`` is
  now higher-is-better, and ``best_index_`` is set. ``AML`` keeps the best
  surrogate stage instead of the last (which could be an unfitted ``inf``
  result), and ``StackingEstimator.set_params`` no longer ignores ``cv``/``n_jobs``.
  ``AML(resume=False)``: fits no longer silently resume a stale checkpoint (set ``resume=True`` for the old behaviour).
  EOA now accepts ``parents_prop``, ``elites_prop``, and
  ``max_generations``; the legacy spellings remain supported.
  ``ModelBundle.save``/``ModelBundle.load`` mirror ``save_bundle``/``load_bundle``.
  Removed redundant ``fit_transform`` overrides in ``SensAprx`` and
  ``CorrelationThreshold`` (inherited from ``TransformerMixin``).

* ``RegisterData`` accepts an optional ``preprocess_kwargs`` mapping, forwarded
  to ``DataPreprocess`` (e.g. ``bin_threshold``, ``cat_threshold``,
  ``force_impute``) when encoding is enabled; defaults are unchanged.

Logging and per-trial telemetry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Core modules ``structsearch.py``, ``eoa.py``, and ``aml.py`` now emit progress
  through the standard library ``logging`` (module loggers in the
  ``SKSurrogate`` namespace) instead of bare ``print()`` calls. Default console
  output is unchanged byte-for-byte: when no handler is configured anywhere in
  the logger's propagation chain, ``SKSurrogate._logutil.emit`` still writes the
  message to stdout; once a handler exists that stdout copy stops, so integrating
  pipelines see exactly one copy. ``verbose`` semantics are untouched
  (``>0`` → INFO, ``>1`` → DEBUG; EOA's per-generation line stays ungated INFO).
* ``SurrogateSearch.surrogate_fit_history_`` records gain a new ``fit_score``
  key: the fitted surrogate's own in-sample ``score`` on the data it was fitted
  to (``None`` when the regressor has no callable ``score`` or returns a
  non-finite value). Existing keys are unchanged.
* New ``SurrogateSearch.sampling_sources_`` parallel list tags every evaluation
  as ``init``, ``explore``, or ``surrogate_exploit``, appended in lockstep with
  ``evaluated`` and stored in the pickle checkpoint; checkpoints written before
  this key existed still load, with an empty tag list rather than invented
  provenance.
* New ``callbacks=None`` parameter on ``SurrogateRandomCV``: a callable or
  iterable of callables, each receiving one event dict per evaluated candidate
  (the trial record plus a zero-based ``trial`` index and its ``phase``),
  delivered at every record site — forbidden, zero-feature, pruned, and
  completed/failed. A callback that raises is logged and skipped; it cannot
  abort a search. ``get_params``/``clone`` compatibility is preserved.
* New ``SurrogateRandomCV.to_audit_dict()``: a JSON-safe audit bundle
  (``schema_version: 1``) carrying the estimator class and parameters, the
  encoded search space, scoring/CV summaries, budget and termination state, all
  trial records with trial indices, surrogate diagnostics, and sampling
  provenance; ``json.dumps()`` succeeds without a custom encoder.
* Documentation: new :doc:`telemetry` chapter covering logger names and level
  mapping, the unconfigured-stdout fallback, the ``fit_score`` and
  sampling-source records, the ``callbacks`` event schema (including the
  seed-trial ``phase = None`` caveat), and the ``to_audit_dict()`` bundle; the
  hyperparameter-optimization chapter links to it.

[0.2.0] - 2026-10-24
--------------------

Response-surface default change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``SurrogateSearch`` now defaults its response surface to a stock, un-tuned RBF
  Gaussian process (`sklearn.gaussian_process.GaussianProcessRegressor`) instead of the
  legacy cubic Hilbert polynomial (`HilbertRegressor(deg=3)`). Evidence: equal-budget
  benchmark suite -- the RBF-GP surface attains top mean CV on all six datasets at both
  B=30 and B=60, with seed-level win rates of 20W/17T/5L (B=30) and 25W/13T/4L (B=60)
  against Hilbert (two-sided exact binomial p approx. 4e-3 / 1e-4); the un-tuned RBF-KRR
  alternative never wins a seed at either budget. See ``benchmarks/DESIGN.md`` sec. 12;
  raw per-run records are archived at DOI 10.5281/zenodo.23629024 (software:
  DOI 10.5281/zenodo.23631823). The surrogate-firing floor (`MinEvals`) is unchanged, so
  the firing behavior matches that benchmark configuration exactly.
* Restore the v0.1 behaviour with `regressor=HilbertRegressor(deg=...)`; no new dependency
  (scikit-learn was already required).

Dependencies and packaging
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Lower-bound dependency pins in ``setup.py``/``requirements.txt`` exclude the
  vulnerable ranges flagged by Dependabot (numpy <1.22, scipy <1.10,
  scikit-learn <1.5, joblib <1.2, tqdm <4.66.3, fastapi <=0.109.0, lightgbm
  <4.6, uvicorn <0.11.7, python-multipart <0.0.31); bounds sit far below the
  versions CI currently installs and no upper pins were added.
* WebUI lockfile: source-map-js pinned to a non-vulnerable version
  (CVE-2026-93749).

[0.1.1] - 2026-10-11
--------------------

Republished from 0.1.0 with canonical MIT license metadata; code is identical
to 0.1.0.

Control-plane API and Web UI (branch ``ui``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added a FastAPI control plane under ``api/`` covering datasets, bundles,
  quality gates, registry/deployment, inference, monitoring, sensitivity,
  experiments, background jobs with WebSocket progress streaming, retraining,
  and a lineage traceability endpoint.
* Added a React/Vite web UI under ``webui/`` with pages for every pipeline
  stage, live job tracking, a responsive shell, and route-level code splitting.
* Added opt-in shared-key API authentication via the ``SKSURROGATE_API_KEY``
  environment variable (accepted through the ``X-API-Key`` header, Bearer
  tokens, or an ``api_key`` query parameter on WebSocket routes); unset keeps
  all endpoints open for local use.
* Made CORS origins configurable via ``SKSURROGATE_API_CORS_ORIGINS``, defaulting
  to local-only origins instead of a wildcard; the web UI stores and sends an
  API key from its Settings page.
* Added learning-curve evaluation: ``GET /api/evaluation/{task}/metrics`` lists
  the metrics relevant to the task's problem family, and
  ``GET /api/evaluation/{task}/learning-curves`` computes cross-validated
  training/held-out curves per bundle for one metric via scikit-learn. The
  Evaluation page's "Metric comparison" card now has one tab per metric on top
  of the side-by-side overview chart.
* Experiment searches gained surrogate-assisted mode
  (``surrogate_mode``/``surrogate_itrs``), conditional parameters
  (per-parameter ``depends_on``) and top-level ``forbidden`` rules, a per-run
  Pareto frontier (score vs. candidate duration) and ``top_pipelines`` in the
  job result, plus a ``POST /api/experiments/{task}/optimize-pipeline`` endpoint
  that tunes one explicit component sequence into a bundle. The Experiments page
  gained a search-strategy radio, live JSON validation hints, a "Top pipelines"
  table with per-row optimization, and a Pareto scatter chart.
* Added the ``stacking`` directive to experiment search spaces
  (``res``/``probs``/``decision``/``cv``/``n_jobs``), which configures the
  out-of-fold ``StackingEstimator`` wrappers ``AML`` applies to intermediate
  estimators; ``AML`` gained the matching ``stack_cv``/``stack_n_jobs``
  constructor parameters.

Bug fixes
~~~~~~~~~

* ``SurrogateRandomCV`` no longer refits a rejected combination. The optimizer's
  solution can land inside a forbidden region (or in one whose folds all
  failed), and refitting those settings either raised — e.g. ``lbfgs`` with
  ``penalty='l1'`` — or silently returned a banned model. The best *completed*
  trial is used instead, and the refit is skipped entirely when no trial
  completed.
* Fixed the experiments stage failing with ``ValueError: too many values to
  unpack (expected 2)``: the stored cross-validation spec (a JSON dictionary)
  was handed straight to ``AML``, and scikit-learn's ``check_cv`` treated that
  dictionary as an iterable of ``(train, test)`` folds and unpacked its keys.
  The API now rebuilds the splitter with ``SKSurrogate.mltrace.build_cv`` while
  keeping the JSON spec for the bundle audit event, and ``AML`` normalizes any
  dictionary ``cv`` the same way so no caller can repeat the mistake.
* ``EOA`` now rejects ``num_parents`` larger than the population size with a
  message naming both values, instead of failing later inside
  ``random.sample`` with ``Sample larger than population or is negative``.
* ``SKSurrogate.aml.default_config`` is defined at module level again, so
  ``AML()`` works without an explicit ``config``.
* Inference on a stored partition no longer fails with ``400: extra columns:
  target``. A registered partition is training data, so it carries the target
  label while the bundle schema holds only features; ``predict_batch`` validates
  strictly and rejected the label outright. The control-plane API now projects a
  stored partition onto the bundle schema (which also normalizes column order)
  and reports the dropped columns as ``ignored_columns``. Inline ``rows`` are
  still validated exactly as sent, since they represent the serving payload.

Phase 10: Documentation and release hardening
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added an offline-to-deployment lifecycle example covering bundles, registry
  promotion, batch inference, and drift reporting.
* Added a full synthetic classification workflow covering nested evaluation,
  quality gates, approval-gated promotion, fairness, delayed labels, and
  runtime monitoring.
* Added a full synthetic regression workflow covering nested evaluation, R2
  quality gates, approval-gated promotion, numeric drift, delayed-label
  MAE/MSE, and runtime monitoring.
* Added SQLite database and EOA checkpoint migration and rollback guidance.
* Documented public API coverage, supported runtime policy, benchmark thresholds,
  release checks, and the compatibility matrix.
* Added opt-in benchmarks for bundle loading, batch inference, and drift reports.

Phase 9: Distributed execution and CI/CD integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added durable local and optional Dask execution backends.
* Added bundle quality gates, triggered retraining, approval-gated deployment,
  and rollback workflows.
* Added Phase 9 operational documentation.

Phase 8: Governance, fairness, and security
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added audit events, ownership metadata, sensitive-feature reports, fairness
  and subgroup reports, and retention-aware artifact deletion.

Phase 7: Data quality, drift, and performance monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added schema, quality, distribution, range, prediction, delayed-label, and
  runtime monitoring reports with configurable alerts.

Phase 6: Batch and online inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added schema-validated batch prediction, the batch CLI, and a dependency-free
  WSGI serving adapter with health and readiness endpoints.

Phase 5: Model bundles and registry lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added portable model bundles, atomic persistence, compatibility checks,
  registry aliases, promotion and rollback history, and MLflow-compatible export.

Earlier phases
~~~~~~~~~~~~~~

* Phases 1 through 4 added dataset identity and schema contracts, explicit
  evaluation protocols, resource governance, search-space validation, and
  structured failure handling.
