# End-to-end notebooks

Three worked examples that take one task from raw frame to a shipped, monitored,
documented model. Each notebook is self-contained: it generates its own data, runs the
whole workflow, and writes its report into a sibling directory.

| Notebook | Task | Headline artefact |
| --- | --- | --- |
| [`01_classification_end_to_end.ipynb`](01_classification_end_to_end.ipynb) | Rare-event classification (12 % positive class) | Nested CV with repeats, threshold/lift/calibration analysis, drift + subgroup reports, bundle → `/predict`, branded PDF |
| [`02_regression_end_to_end.ipynb`](02_regression_end_to_end.ipynb) | Continuous target in physical units, heteroscedastic noise | Bootstrap metric intervals, residual diagnostics, split-conformal intervals with coverage-by-decile, 2-D partial dependence, PDF |
| [`03_time_series_end_to_end.ipynb`](03_time_series_end_to_end.ipynb) | Daily demand, 7-day horizon | Temporal contract, leakage-safe features, walk-forward backtest against naive baselines, horizon sweep, roll-forward serving, PDF |

## Running them

```bash
# from the repository root, with the project environment active
jupyter lab examples/notebooks/01_classification_end_to_end.ipynb
```

Requirements beyond the library: `jupyterlab`/`ipykernel` (dev tooling, not a runtime
dependency), `matplotlib`, plus `lightgbm` for the boosted-tree candidate in notebooks 2
and 3. LaTeX (`lualatex`) is optional: without it the report Markdown, CSVs, JSON and PNGs
are still written and the PDF step reports why it was skipped.

Each notebook writes `examples/notebooks/<task>-report/` (gitignored) containing:

```
report.md            the narrative report
report.pdf           the same report, branded, compiled with lualatex
*.csv                every table that appears in the report
analysis-metadata.json   metrics, intervals, seed, protocol, limitations
*.png / *.pdf        figures for display and for the PDF
.notebook-work/      the tracker database and model bundle
```

The notebooks ship **without stored outputs** so the diffs stay readable; run one to
regenerate them. Every one of them is executed headlessly before release, with a fixed seed
(`SEED = 7`) that reaches the data generator, the train/validation split, the imputer, the
search (its start point and every candidate estimator) and the CV splitters. Two runs of the
same notebook therefore produce the same metrics and the same tables; only wall-clock fields
and the bundle's build-time digest (`model_version`) differ.

## Data sources

Synthetic by default, which keeps the notebooks offline, fast and reproducible:

* classification — synthetic subscription churn frame with MAR missingness, a duplicated
  column, an identifier column and a deliberately rare positive class;
* regression — synthetic building energy intensity with a non-linear interaction,
  multiplicative noise and MCAR gaps;
* time series — synthetic daily demand with weekly and annual seasonality, a promotion
  flag, and gaps in both columns.

Set `SKSURROGATE_NOTEBOOK_DATA=openml` to switch to an OpenML frame instead (needs network;
`SKSURROGATE_OPENML_KEY` is read when present). The workflow is unchanged; the target,
units and the caveats are not.

## What these notebooks are *not*

They are methods demonstrations, not benchmark results: the search budgets are small, the
frames are small, and every hyperparameter was chosen for runtime rather than accuracy. The
limitations section at the end of each notebook states what its numbers do and do not
support.

## Shared helpers

* `notebook_data.py` — the three generators plus the OpenML switch, each returning a
  `Dataset(frame, target, source, description)` with notes on how it was built.
* `notebook_report.py` — figure saving (PNG for display, PDF for the report template),
  Markdown table rendering, and `compile_pdf`, which degrades to a note when LaTeX is
  missing instead of failing the notebook.
