Release Checklist and Compatibility Matrix
===========================================

Release checklist
-----------------

Before publishing a release:

* Run ``.venv/bin/python -m unittest discover -s tests -v``.
* Run each runnable example, including ``example04_end_to_end.py``.
* Run ``.venv/bin/python examples/benchmark_phase10.py --check``.
* Build the Sphinx documentation with ``make -C docs html`` and resolve warnings.
* Install the package into a clean virtual environment and verify the public
  imports and ``sksurrogate-batch-predict`` entry point.
* Validate a saved bundle in a fresh process with strict dependency checking.
* Exercise registry promotion, approval, rollback, and artifact deletion.
* Check that SQLite databases and EOA checkpoints from the previous release
  remain readable, or publish migration instructions for any incompatibility.
* Review sensitive metadata, audit records, dependency versions, and changelog
  entries before tagging the release.

Compatibility matrix
--------------------

+----------------------+----------------------+------------------------------+
| Area                 | Validated baseline  | Release expectation          |
+======================+======================+==============================+
| Python               | 3.11, 3.13          | 3.11 and 3.13 supported     |
+----------------------+----------------------+------------------------------+
| Python 3.12          | Not currently run   | Expected compatible; verify  |
+----------------------+----------------------+------------------------------+
| NumPy/pandas/sklearn | Current test env    | Record exact versions        |
+----------------------+----------------------+------------------------------+
| SQLite tracking DB   | Existing schema     | Reopen and smoke-test        |
+----------------------+----------------------+------------------------------+
| EOA checkpoints      | ``checkpoint_version=1`` | Resume or document upgrade |
+----------------------+----------------------+------------------------------+
| Model bundles        | ``format_version=1`` | Strict dependency validation |
+----------------------+----------------------+------------------------------+

The exact resolved dependency versions used for a release must be recorded in
release artifacts. Applications should lock their own dependency environment;
serialized models should not be moved between incompatible major versions
without a prediction and loadability check.
