==============================
Notebook workflows
==============================

The repository includes two executable notebooks under
``examples/notebooks/``. They use seeded synthetic data and bounded evaluation
settings; exact search/model results can still vary across runs or environments.
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

Jupyter and ``ipykernel`` are optional user-side tools; they are not added as
package dependencies. Both notebooks store source code rather than rendered
screenshots or generated outputs, so they remain portable and reviewable.
