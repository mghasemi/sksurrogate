==============================
Notebook workflows
==============================

The repository includes five executable notebooks under
``examples/notebooks/``. They use seeded synthetic data and bounded evaluation
settings. Data, splits, imputation, the ``AML`` search (start point and candidate
estimators) and the CV splitters are all seeded, so two runs of a notebook report the
same metrics; only wall-clock fields and the model bundle's build-time digest differ.
No notebook server or widget service is part of the SKSurrogate runtime.

Offline scientific workflow
===========================

Run ``scientific_workflow.ipynb`` from the repository root with Jupyter
installed in the active environment::

    jupyter lab examples/notebooks/scientific_workflow.ipynb

The notebook profiles a dataset, runs a small classification preset, compares
using repeated nested cross-validation, and computes bounded permutation
importance on an explicitly held-out validation partition. It exports
``data-profile-columns.csv``, ``target-associations.csv``,
``performance-summary.csv``, ``permutation-importance.csv``,
``eda-formal-tests.csv``, ``eda-effect-sizes.csv``, ``eda-stability.csv``,
``eda-quality-flags.csv``, ``eda-report.md``, ``analysis-metadata.json``, and a
300-dpi ``workflow-summary.png`` under ``scientific-workflow-report/`` (created in
the current working directory).
The tables and metadata record profile summaries, metric, model, seed, sample
sizes, repeated outer scores, and importance reference partition. The figure
keeps repeated-CV results separate from the held-out validation score;
importance error bars show permutation-repeat standard deviations, not
confidence intervals. The small example confidence interval is illustrative,
seeds do not guarantee identical search results across environments, and the
notebook does not make causal claims.

Python API client workflow
==========================

Run ``python_api_workflow.ipynb`` after starting the API in a separate
terminal::

    python -m uvicorn api.main:app --host 127.0.0.1 --port 8013
    jupyter lab examples/notebooks/python_api_workflow.ipynb

The notebook uses Python's standard-library HTTP client to upload two
partitions, inspect a profile, submit and poll an AutoML job, launch repeated
nested-CV evaluation, and retrieve the run-configuration JSON and Markdown
report. It then creates a research project linking the registered partition,
search/evaluation jobs, and selected bundle, and follows those references to
verify that the linked job, dataset fingerprint, and run configuration agree.
Set ``SKSURROGATE_API_URL`` and, when API-key authentication is enabled,
``SKSURROGATE_API_KEY`` in the notebook environment. ``SKSURROGATE_TASK_NAME``
can select a task name; otherwise each run creates a timestamped one. Uploaded
data, jobs, and bundles persist in the API's configured storage root. Use only
disposable tasks and non-sensitive data when trying the example.

End-to-end task workflows
=========================

Three notebooks take one task each from a raw frame to a shipped model:

``01_classification_end_to_end.ipynb``
    Rare-event classification (about 12 % positive): profiling, MAR diagnostics,
    Little's test, a measured imputation bake-off, a leakage demonstration, an
    ``AML`` search, repeated nested cross-validation, threshold and lift analysis,
    drift and subgroup reports, bundle quality gate, and serving through
    ``predict_batch``.

``02_regression_end_to_end.ipynb``
    Continuous target in physical units with multiplicative noise: the same
    skeleton plus bootstrap intervals on the metrics, residual and
    heteroscedasticity diagnostics, split-conformal prediction intervals with a
    coverage-by-decile check, and one- and two-dimensional partial dependence.

``03_time_series_end_to_end.ipynb``
    Daily demand with weekly and annual seasonality: ``TemporalSpec`` and
    validation, the refused shuffled split, ``ForecastFeatureBuilder``,
    ``make_supervised`` and ``assert_no_leakage``, a rolling-origin ``backtest``
    against the naive baselines, interval calibration from backtest residuals, a
    horizon sweep at fixed evaluation length, window-mode ``drift_report``, and
    roll-forward ``serve_forecast`` from the saved bundle.

Each writes ``examples/notebooks/<task>-report/`` with ``report.md``, a branded
``report.pdf`` (LuaLaTeX optional), every table as CSV, ``analysis-metadata.json``
and the figures. Those directories are gitignored: run the notebook to regenerate
them. The shared helpers are ``notebook_data.py`` (the generators plus an OpenML
opt-in via ``SKSURROGATE_NOTEBOOK_DATA=openml``) and ``notebook_report.py``
(figure export, Markdown tables, and ``compile_pdf``, which reports a missing
LaTeX installation instead of failing the run).

Jupyter and ``ipykernel`` are optional user-side tools; they are not added as
package dependencies. All five notebooks store source code rather than rendered
screenshots or generated outputs, so they remain portable and reviewable.
