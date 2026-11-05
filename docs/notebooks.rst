==============================
Notebook workflows
==============================

The repository includes two executable notebooks under
``examples/notebooks/``. They use fixed random seeds, synthetic data, and
bounded evaluation settings; no notebook server or widget service is part of
the SKSurrogate runtime.

Offline scientific workflow
===========================

Run ``scientific_workflow.ipynb`` from the repository root with Jupyter
installed in the active environment::

    jupyter lab examples/notebooks/scientific_workflow.ipynb

The notebook profiles a dataset, runs a small classification preset, compares
using repeated nested cross-validation, and computes bounded permutation
importance on an explicitly held-out validation partition. It does not make
causal claims, and its small example confidence interval is illustrative.

Python API client workflow
==========================

Run ``python_api_workflow.ipynb`` after starting the API in a separate
terminal::

    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
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
