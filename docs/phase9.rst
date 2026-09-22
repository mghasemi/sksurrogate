========================================
Phase 9: Execution and CI/CD Operations
========================================

Phase 9 provides the operational pieces needed to move a search or retraining
workflow beyond one process: durable trial execution, optional distributed
submission, model quality gates, retraining triggers, and approval-controlled
deployment.

Execution backends
==================

``ExecutionBackend`` defines the common lifecycle for queued trials. A trial
contains parameters, a callable, optional metadata, and a task name. Backends
persist or dispatch the following states:

* ``queued``: submitted but not claimed;
* ``running``: claimed by a worker;
* ``completed``: finished with a persisted result;
* ``failed``: finished with an error and traceback.

The interface includes ``submit_trial``, ``get_trial``, ``mark_running``,
``mark_completed``, ``mark_failed``, ``execute_pending``, ``claim_trial``,
``claim_next_trial``, ``list_pending``, ``list_results``, and
``resume_incomplete_trials``. A worker ID can be attached to claims and results
to make ownership visible in the stored state.

Local durable execution
-----------------------

``LocalProcessExecutionBackend`` is appropriate for a single host, a scheduled
job, or a lightweight worker pool. It stores one JSON state file per task,
serializes callables with ``cloudpickle`` when available, writes state through
an atomic replacement, and uses a lock around local state changes.

.. code-block:: python

    from SKSurrogate import LocalProcessExecutionBackend

    backend = LocalProcessExecutionBackend(
        "artifacts/execution",
        task_name="customer-churn-search",
    )
    trial_id = backend.submit_trial(
        {"max_depth": 8},
        fn=lambda params: {"score": 0.91, "params": params},
        metadata={"source": "nightly-search"},
    )

    # A worker claims the oldest queued trial and executes it.
    claimed = backend.claim_next_trial(worker_id="worker-1")
    if claimed is not None:
        try:
            result = claimed["fn"](claimed["params"])
            backend.mark_completed(
                claimed["trial_id"], result=result, worker_id="worker-1"
            )
        except Exception as error:
            backend.mark_failed(
                claimed["trial_id"], error=str(error), worker_id="worker-1"
            )

For simple sequential work, ``execute_pending`` performs the claim, execution,
and persistence loop and returns the executed record snapshots:

.. code-block:: python

    completed_or_failed = backend.execute_pending(
        worker_id="local-worker",
        limit=10,
    )
    completed = backend.list_results(status="completed")
    pending = backend.list_pending()

``claim_next_trial`` selects the oldest queued record. ``claim_trial`` returns
``None`` when another worker owns a running trial or when the trial is no longer
claimable. ``resume_incomplete_trials`` returns queued, running, failed, and
timed-out records so a restarted worker can decide what to retry. Results and
failures remain queryable through ``list_results`` and ``get_trial``.

Dask execution
--------------

``DaskExecutionBackend`` uses a Dask-compatible client for asynchronous execution
while retaining the local durable trial store for lifecycle state. Dask is an
optional dependency and is not imported when an already-created client is
injected. This keeps local installations lightweight and makes the adapter easy
to test or integrate with an existing Dask scheduler.

.. code-block:: python

    from dask.distributed import Client
    from SKSurrogate import DaskExecutionBackend

    client = Client("tcp://scheduler:8786")
    backend = DaskExecutionBackend(
        "artifacts/execution",
        task_name="customer-churn-search",
        client=client,
    )
    backend.submit_trial(
        {"max_depth": 8},
        fn=evaluate_candidate,
        metadata={"fold": 0},
    )
    scheduled = backend.execute_pending(worker_id="dask-worker", limit=20)

Each submitted future updates the durable record through a completion callback.
Successful futures become ``completed`` with their result; exceptions become
``failed`` with an error message and traceback. Use the same listing and resume
methods as the local backend. The durable store is the recovery record; Dask
cluster configuration, worker scaling, and scheduler lifecycle remain the
responsibility of the deployment environment.

CI model quality gates
======================

``check_bundle_quality`` validates a ``ModelBundle`` or a saved bundle path and
returns a JSON-compatible report with ``passed``, ``checks``, and ``failures``.
It can enforce:

* exact schema compatibility through ``expected_schema``;
* minimum numeric metric values through ``metric_thresholds``;
* maximum serialized artifact size through ``max_bundle_size_bytes``.

The size check requires a filesystem path because an in-memory bundle has no
serialized size. Bundle loading for this check does not require the current
runtime dependency versions to match; dependency policy can be handled by the
CI environment separately.

.. code-block:: python

    from SKSurrogate import check_bundle_quality, assert_bundle_quality

    quality = check_bundle_quality(
        "artifacts/customer-churn.bundle",
        expected_schema=feature_schema,
        metric_thresholds={"validation_accuracy": 0.90},
        max_bundle_size_bytes=25 * 1024 * 1024,
    )
    if not quality["passed"]:
        print(quality["failures"])

    # Raise BundleQualityGateError to fail a CI job.
    assert_bundle_quality(
        "artifacts/customer-churn.bundle",
        expected_schema=feature_schema,
        metric_thresholds={"validation_accuracy": 0.90},
    )

``BundleQualityGateError`` is raised by ``assert_bundle_quality`` when any gate
fails. A missing metric, a non-numeric metric, a schema mismatch, or an artifact
above the configured size is a failure. A passing report can be supplied to the
deployment approval gate.

Triggered retraining
====================

``RetrainingJob`` is scheduler-agnostic. The caller supplies a trainer callback
that accepts a context dictionary and returns a ``ModelBundle``. The job fills in
the configured task name when the trainer leaves it unset, registers the bundle
with ``ModelRegistry``, and can optionally promote it to a lifecycle state.

.. code-block:: python

    from SKSurrogate import ModelRegistry, RetrainingJob

    registry = ModelRegistry("artifacts/registry")
    job = RetrainingJob(
        trainer=train_customer_churn_model,
        registry=registry,
        task_name="customer-churn",
        promotion_state="staging",
    )

    scheduled = job.run_scheduled(
        lambda context: context["hour"] == 2,
        context={"hour": 2, "reason": "nightly"},
    )
    data_triggered = job.run_on_data(
        lambda context: context["new_rows"] >= 1000,
        context={"new_rows": 2400, "reason": "data-arrival"},
    )

A false trigger returns ``{"status": "skipped", ...}`` and does not call the
trainer. A satisfied trigger returns ``{"status": "completed", ...}`` with the
new model version and promotion state. The trainer must return a ``ModelBundle``
for the configured task; mismatched task names and invalid return values fail
before registry promotion. Cron, Airflow, event queues, and data watchers can
call these methods without being dependencies of the library.

Approval-gated deployment and rollback
======================================

``DeploymentApprovalGate`` adds a deployment policy around ``ModelRegistry``.
It requires a configured number of unique approvers and optionally requires a
passing quality report before calling the registry's normal promotion method.
Every approval is persisted as an audit event with the approver, target state,
model version, and timestamp.

.. code-block:: python

    from SKSurrogate import DeploymentApprovalGate

    gate = DeploymentApprovalGate(registry, required_approvals=2)
    promoted = gate.promote(
        "customer-churn",
        model_version,
        "production",
        approvers=["ml-owner", "risk-owner"],
        quality_report=quality,
    )

    rollback = gate.rollback(
        "customer-churn",
        "production",
        model_version="previous-production-version",
    )

Fewer than the required number of unique non-empty approvers raises
``PermissionError``. A supplied quality report with ``passed`` set to false
raises ``ValueError``. Promotion and rollback continue to use the registry's
lifecycle aliases and history, so ``registry.audit_log(task_name)`` provides a
single trace of approvals, promotions, and rollbacks.

Operational use cases
=====================

* **Nightly local search:** submit candidate evaluations to a local backend,
  let a scheduled worker execute pending trials, and resume the state file after
  an interrupted host process.
* **Distributed search:** submit the same trial records to a Dask scheduler while
  keeping durable local status and result records for recovery and inspection.
* **Pull-request model validation:** load a candidate bundle in CI, verify its
  schema, enforce validation metrics, and reject oversized artifacts before
  registration or deployment.
* **Scheduled retraining:** invoke ``run_scheduled`` from cron or an orchestrator
  and promote successful output to ``staging`` for later review.
* **Data-triggered retraining:** invoke ``run_on_data`` from an ingestion or drift
  event when enough new rows or a monitored condition is available.
* **Controlled production release:** pass the quality report and independent
  approver identities to ``DeploymentApprovalGate``; use its rollback method to
  restore a known registry version while retaining the audit trail.

Safety and boundaries
=====================

Callable serialization and bundle loading execute Python objects and therefore
must only consume trusted artifacts. Dask is optional and must be installed in
the environment that creates a client. The execution backends provide durable
state and dispatch semantics, but they do not provide distributed locking across
separate filesystem mounts or replace scheduler-level authentication and
network security. Approval identities are recorded as supplied; integrating an
external identity provider remains an application concern.
