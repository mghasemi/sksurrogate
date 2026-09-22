Phase 10: Documentation and Release Hardening
==============================================

The end-to-end example in ``examples/example04_end_to_end.py`` demonstrates
the core offline-to-deployment lifecycle:

1. Train and validate a scikit-learn model.
2. Capture its input schema and validation metric in a ``ModelBundle``.
3. Save and reload the bundle using atomic persistence.
4. Register and promote the model to the ``production`` alias.
5. Run schema-validated batch predictions with audit metadata.
6. Compare serving data with the training reference using ``drift_report``.

Run it from the repository root with::

   .venv/bin/python examples/example04_end_to_end.py

For a full synthetic classification workflow, including nested evaluation,
quality gates, approval-gated promotion, fairness, delayed labels, and runtime
monitoring, run::

   .venv/bin/python examples/example05_full_classification.py

For the equivalent synthetic regression workflow, including R2 quality gates,
numeric drift, delayed-label MAE/MSE, and runtime monitoring, run::

   .venv/bin/python examples/example06_full_regression.py

All files are written to a temporary directory. The example therefore leaves
no model, registry, or prediction artifacts behind after it exits.

The example is intentionally small enough to run as a documentation smoke
test. The focused regression suite covers the same APIs in more detail::

   .venv/bin/python -m unittest discover -s tests -v

Migrating Existing State
========================

SQLite tracking databases and EOA checkpoints are local artifacts. There is no
in-place migration command, so preserve the original files before opening them
with a new version of SKSurrogate::

   cp mltrack.db mltrack.db.backup
   cp -a example01-checkpoints example01-checkpoints.backup

Existing SQLite databases can be reopened with the same task name and
``db_name``. The tracker creates any tables that are missing and retains the
existing task, metadata, models, metrics, plots, data, and saved-model rows::

   from SKSurrogate import mltrack

   tracker = mltrack("my-task", db_name="mltrack.db")
   print(tracker.GetMetadata())
   print(tracker.ListModels())
   tracker.close()

Use the backup to roll back if the database cannot be opened or if an
application-level validation fails::

   mv mltrack.db mltrack.db.failed
   mv mltrack.db.backup mltrack.db

EOA checkpoints contain a ``checkpoint_version`` field. A checkpoint with an
unsupported version is rejected rather than being partially loaded. Resume a
compatible checkpoint by using the original checkpoint directory and task name
with the same search configuration::

   from SKSurrogate import AML

   search = AML(config=config, length=3, check_point="example01-checkpoints/")
   search.eoa_fit(X, y, max_generation=10, num_parents=10)

If a checkpoint cannot be resumed, restore the backup and start a new search
in a new directory. Do not overwrite the original checkpoint while diagnosing
compatibility or serialization failures.

Legacy models saved through ``mltrack.PreserveModel`` remain recoverable with
``RecoverModel`` when their Python and dependency versions can deserialize the
stored joblib payload. For deployment, recover and validate the model first,
then create a new ``ModelBundle`` with an explicit schema, metrics, owner, and
model version. Keep the legacy database as the audit source until the bundle
has passed prediction and compatibility checks; registry promotion and rollback
operate on bundle versions, not on legacy SQLite rows.

API Reference
=============

The :doc:`code` reference covers the package-root exports and every public
implementation module, including the bundle, inference, monitoring,
execution, CI, retraining, and deployment APIs added in Phases 5 through 9.

Performance Benchmarks
======================

Run the opt-in benchmark from the repository root::

   .venv/bin/python examples/benchmark_phase10.py --check

The benchmark reports median wall-clock time over five repetitions for bundle
loading, batch prediction of 400 rows, and drift reporting for six numeric
features. Each operation has a two-second threshold. Import and model-training
time are excluded so the thresholds focus on the portable operational paths;
use ``--repetitions`` to increase sampling when investigating a regression.

Supported Runtime Policy
========================

The release validation baseline is Python 3.11, matching the Read the Docs
build configuration. Python 3.13 is also exercised in the repository virtual
environment; Python 3.12 is expected to be compatible but is not currently a
release-validation job. New Python releases become supported only after the
full test suite, runnable examples, benchmark checks, documentation build, and
package installation pass on that interpreter.

Runtime dependencies are currently specified as a compatible, unpinned set in
``requirements.txt`` and ``setup.py``. A release records the resolved versions
used for validation, while downstream applications should lock their own
environment. In particular, serialized model bundles and legacy checkpoints
should be validated with the same Python, NumPy, pandas, scikit-learn, and
joblib major versions used to create them; the bundle loader rejects recorded
dependency mismatches when strict compatibility checking is enabled.
