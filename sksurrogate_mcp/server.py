"""MCP server exposing the SKSurrogate HTTP control plane as agent tools.

Phase 0 shipped three discovery tools (``health``, ``list_tasks``,
``task_overview``); Phase 1 adds the core loop: dataset registration and
preprocessing (``register_dataset``, ``dataset_preflight``, ``impute_missing``,
``set_ignored_columns``, ``dataset_preview``), experiment submission
(``run_experiment``, ``optimize_pipeline``), bounded job handling
(``job_wait``, ``job_diagnose``, ``stop_job``, ``resume_job``) and the generic
``api_request`` escape hatch, validated against an embedded OpenAPI snapshot.

Every tool is one curated endpoint (or a small read-only composite of them).
No computation is re-implemented here, so every call passes through the
deployment's authentication, ACL roles and audit trail exactly as if a human
had made it. Mutating tools state their effect in their description; expensive
actions (impute materialization) are explicit calls only — nothing chains by
itself.

Configuration comes from the environment: ``SKSURROGATE_API_URL`` (required,
e.g. ``http://127.0.0.1:8099``) and ``SKSURROGATE_API_KEY`` (optional; sent as
the ``X-API-Key`` header). The key is never logged or echoed.

Run with the console script ``sksurrogate-mcp`` (stdio transport, the default)
or ``python -m sksurrogate_mcp.server``. stdout is reserved for the MCP stdio
protocol; diagnostics go to stderr.

Error contract: expected API failures (4xx/5xx answers, unreachable host) are
returned as structured payloads — ``{"error": "<message>"}`` plus a numeric
``status_code`` when the deployment answered non-2xx — rather than raised, so
an agent always receives a greppable answer. Only unexpected internal
exceptions mark the MCP result itself as an error.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, get_type_hints as _get_type_hints

import httpx

DEFAULT_TIMEOUT_S = 60.0

#: Job statuses that end a wait (mirrors api/jobs.py terminal handling:
#: completed/failed are final; paused/interrupted stop the worker but keep a
#: resumable record).
TERMINAL_STATUSES = ("completed", "failed", "paused", "interrupted")

#: Statuses of jobs the worker pool still owns — used for the duplicate-live-job
#: guard (G4) on submit tools. Mirrors api/routers/jobs.py ACTIVE_STATUSES.
ACTIVE_STATUSES = ("queued", "running")

#: Legal stop_job modes; mirrors STOP_MODES in api/control.py.
STOP_MODES = ("pause", "cancel")

#: G6: dataset_preview is a capped sample only — the endpoint itself has no row
#: cap, so the MCP layer clamps it (row-level export stays out of scope).
PREVIEW_LIMIT_MAX = 100
DEFAULT_PREVIEW_LIMIT = 20

#: G1: job_wait runs a bounded poll loop inside the server. Default cap 120 s;
#: hard ceiling keeps a mis-set argument from stalling an agent turn for long.
JOB_WAIT_DEFAULT_TIMEOUT_S = 120.0
JOB_WAIT_MAX_TIMEOUT_S = 900.0
DEFAULT_POLL_INTERVAL_S = 5.0

#: job_diagnose failure rollup: tail length kept from the recorded traceback and
#: cap on log entries returned (the API itself allows up to 2000).
DIAGNOSE_ERROR_TAIL_CHARS = 4000
DIAGNOSE_LOG_MAX = 200

ALLOWED_API_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")

#: Curated-tool → control-plane endpoint templates. Consumed by the spec-drift
#: test (Phase 3) that fails if any hardcoded path disappears from the live
#: OpenAPI spec; each entry is (METHOD, "/api/.../{param}").
TOOL_ENDPOINTS: dict[str, tuple[tuple[str, str], ...]] = {
    "health": (("GET", "/api/health"),),
    "list_tasks": (("GET", "/api/tasks"),),
    "task_overview": (("GET", "/api/datasets/{task_name}"),),
    "experiment_options": (
        ("GET", "/api/experiments/presets"),
        ("GET", "/api/experiments/components"),
        ("GET", "/api/experiments/scoring-options"),
    ),
    "register_dataset": (("POST", "/api/datasets/{task_name}/register"),),
    "dataset_preflight": (("GET", "/api/datasets/{task_name}/preflight"),),
    "impute_missing": (("POST", "/api/datasets/{task_name}/impute"),),
    "set_ignored_columns": (("PUT", "/api/datasets/{task_name}/ignored-columns"),),
    "dataset_preview": (
        ("GET", "/api/datasets/{task_name}"),
        ("GET", "/api/datasets/{task_name}/{partition}/preview"),
    ),
    "run_experiment": (
        ("GET", "/api/jobs/active"),
        ("POST", "/api/experiments/{task_name}/run"),
    ),
    "optimize_pipeline": (
        ("GET", "/api/jobs/active"),
        ("POST", "/api/experiments/{task_name}/optimize-pipeline"),
    ),
    "job_wait": (("GET", "/api/jobs/{job_id}"),),
    "job_diagnose": (
        ("GET", "/api/jobs/{job_id}"),
        ("GET", "/api/logs/{job_id}"),
    ),
    "stop_job": (("POST", "/api/jobs/{job_id}/stop"),),
    "resume_job": (("POST", "/api/jobs/{job_id}/resume"),),
}

_SNAPSHOT_PATH = Path(__file__).resolve().parent / "openapi_ops.json"


class ControlPlaneError(RuntimeError):
    """Unreachable deployment or a non-2xx / malformed API answer."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _detail(response: httpx.Response) -> str:
    """Bounded human-readable detail for an error response body.

    FastAPI errors are shaped ``{"detail": ...}``; keep the message short so it
    stays useful as a tool-error string in front of an LLM.
    """
    try:
        body = response.json()
    except ValueError:
        text = response.text
    else:
        if isinstance(body, dict) and "detail" in body:
            detail = body["detail"]
            text = detail if isinstance(detail, str) else json.dumps(detail)
        else:
            text = json.dumps(body)
    return text[:1000]


class ControlPlaneClient:
    """Bounded HTTP client bound to one SKSurrogate control-plane deployment."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        if not base_url:
            raise ValueError("base_url is required")
        self.base_url = base_url.rstrip("/")
        # AuthMiddleware (api/main.py) accepts the key as X-API-Key or
        # Authorization: Bearer; *** header is the canonical non-browser form.
        headers = {} if api_key in (None, "") else {"X-API-Key": api_key}
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout_s)

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        json_body: Any = None,
        data: Optional[dict] = None,
        files: Optional[Any] = None,
    ) -> Any:
        """One HTTP round trip; raises :class:`ControlPlaneError` on failure.

        ``params`` → query string, ``json_body`` → JSON body, ``data``/``files``
        → multipart/form-data (used by dataset registration).
        """
        try:
            response = self._http.request(
                method, path, params=params, json=json_body, data=data, files=files
            )
        except httpx.HTTPError as exc:
            raise ControlPlaneError(f"control plane {self.base_url} unreachable: {exc}") from exc
        if response.status_code >= 400:
            raise ControlPlaneError(
                f"{method.upper()} {path} -> HTTP {response.status_code}: {_detail(response)}",
                status_code=response.status_code,
            )
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise ControlPlaneError(f"{method.upper()} {path} returned a non-JSON body") from exc

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs)


def _call(client: ControlPlaneClient, method: str, path: str, **request_kwargs: Any) -> Any:
    """Run one control-plane call with the shared error contract.

    Expected API failures (4xx/5xx from the deployment, unreachable host,
    malformed bodies) come back as ``{"error": "<message>"}`` — plus a numeric
    ``status_code`` when the deployment answered non-2xx — instead of raising,
    so an agent always receives a structured, greppable payload. Only truly
    unexpected exceptions propagate (the MCP SDK then marks the result as an
    error).
    """
    try:
        data = client.request(method, path, **request_kwargs)
    except ControlPlaneError as exc:
        payload: dict[str, Any] = {"error": str(exc)}
        if exc.status_code is not None:
            payload["status_code"] = exc.status_code
        return payload
    return data if data is not None else {}


# ---------------------------------------------------------------------------
# Tool implementations (plain functions; ``client`` bound at registration)
# ---------------------------------------------------------------------------


def health(client: ControlPlaneClient) -> dict:
    """Check connectivity to the SKSurrogate control plane.

    Returns {"status": "ok", "version": "<core version>"}. Call this first
    after any configuration change; a failure here means the base URL or
    credential is wrong, not that one specific tool is broken. On failure
    returns an {"error": ...} payload (with status_code when the deployment
    answered).
    """
    return _call(client, "GET", "/api/health")


def list_tasks(client: ControlPlaneClient) -> dict:
    """List task names visible to this credential on the deployment.

    A task appears if it has a dataset directory, an mltrace database file,
    or persisted job records; with auth enforced only tasks the caller owns
    or is shared are returned. Read-only. Use the returned names as the
    ``task_name`` argument of other tools. On failure returns an
    {"error": ...} payload.
    """
    return _call(client, "GET", "/api/tasks")


def task_overview(client: ControlPlaneClient, task_name: str) -> dict:
    """Registered schema and metadata for one dataset (no raw rows).

    Returns the registered column schema, target column, declared problem
    type with its provenance, temporal spec, dataset fingerprint, feature
    counts and partition history. Raises HTTP 404 when no dataset is
    registered under ``task_name`` — the tool then returns an {"error": ...}
    payload; call ``list_tasks`` to check names. Row-level previews are
    deliberately not part of this tool (see ``dataset_preview``).

    Args:
        task_name: Task name as returned by ``list_tasks``.
    """
    return _call(client, "GET", f"/api/datasets/{task_name}")


def experiment_options(client: ControlPlaneClient) -> dict:
    """Discovery for experiments: presets, component catalog and scoring metrics.

    Read-only composite of three control-plane endpoints. Returns
    {"presets": [...], "components": {...}, "scoring_options": {...}} — the
    quick-start presets (classification/regression with their default search
    spaces), the curated component library ready for ``optimize_pipeline``'s
    config, and every scorer the deployment accepts (groups + defaults). Call
    this before ``run_experiment`` or ``optimize_pipeline`` to choose valid
    values. On failure returns an {"error": ...} payload.
    """
    presets = _call(client, "GET", "/api/experiments/presets")
    if isinstance(presets, dict) and "error" in presets:
        return presets
    components = _call(client, "GET", "/api/experiments/components")
    scoring = _call(client, "GET", "/api/experiments/scoring-options")
    merged: dict[str, Any] = {"presets": presets}
    if isinstance(components, dict) and "error" in components:
        merged["components_error"] = components.pop("error")
    if isinstance(scoring, dict) and "error" in scoring:
        merged["scoring_options_error"] = scoring.pop("error")
    return {**merged, "components": components, "scoring_options": scoring}


def register_dataset(
    client: ControlPlaneClient,
    task_name: str,
    target: str,
    csv_path: str,
    partition: str = "train",
    type_overrides: Optional[dict] = None,
    binarize_label: bool = False,
    ignored_columns: Optional[list[str]] = None,
    temporal: Optional[dict] = None,
    problem_type: Optional[str] = None,
) -> dict:
    """Register a local CSV file as the task's dataset (uploads to the deployment).

    Mutating: creates/updates the registered dataset under ``task_name`` and
    stores the uploaded CSV as partition ``partition``. The control plane
    parses it, deduces column types, validates that ``target`` exists with at
    least one non-missing value, and (optionally) binarizes a 0/1 label into
    binary classification. Use ``dataset_preflight`` afterwards to see the
    advisory checks before running an experiment. On failure returns an
    {"error": ...} payload (422 for bad CSV/target names, 409 on conflicts).

    Args:
        task_name: Task name; created if new.
        target: Name of the prediction target column in the CSV.
        csv_path: Absolute path to a local .csv file to upload.
        partition: Partition name for this upload (default "train").
        type_overrides: Optional {column: type} overrides, e.g. {"city": "categorical"}.
        binarize_label: True only for a 0/1-encoded classification target.
        ignored_columns: Optional list of columns to exclude from all modeling paths.
        temporal: Optional JSON object describing time/series structure (opt-in).
        problem_type: Declare "classification" or "regression" explicitly (else inferred).
    """
    path = Path(csv_path)
    if not path.is_file():
        return {"error": f"CSV file not found locally: {csv_path}"}
    if path.suffix.lower() != ".csv":
        return {"error": "only .csv files are accepted by the control plane; convert before registering"}
    data = {
        "target": target,
        "partition": partition,
        "binarize_label": str(bool(binarize_label)).lower(),
    }
    if type_overrides is not None:
        data["type_overrides"] = json.dumps(type_overrides)
    if ignored_columns:
        data["ignored_columns"] = json.dumps(list(ignored_columns))
    if temporal is not None:
        data["temporal"] = json.dumps(temporal)
    if problem_type:
        data["problem_type"] = problem_type
    with path.open("rb") as handle:
        return _call(
            client,
            "POST",
            f"/api/datasets/{task_name}/register",
            data=data,
            files={"file": (path.name, handle, "text/csv")},
        )


def dataset_preflight(
    client: ControlPlaneClient,
    task_name: str,
    partition: Optional[str] = None,
    max_rows: int = 5000,
) -> dict:
    """Advisory pre-flight checks for a registered dataset before an experiment.

    Read-only. Inspects up to ``max_rows`` rows of the (resolved) partition and
    reports column types, missingness, target sanity, leakage suspects and
    other warnings — nothing is changed. Use after ``register_dataset`` or
    ``impute_missing`` to decide whether cleanup is needed. On failure returns
    an {"error": ...} payload.

    Args:
        task_name: Registered task name.
        partition: Partition to check (default: the registered one, "train" when present).
        max_rows: Row cap for the scan (default 5000).
    """
    params: dict[str, Any] = {"max_rows": int(max_rows)}
    if partition:
        params["partition"] = partition
    return _call(client, "GET", f"/api/datasets/{task_name}/preflight", params=params)


def impute_missing(
    client: ControlPlaneClient,
    task_name: str,
    method: str,
    partition: Optional[str] = None,
    columns: Optional[list[str]] = None,
    params: Optional[dict] = None,
    output_partition: Optional[str] = None,
    preview_only: bool = False,
    drop_missing_target: bool = False,
) -> dict:
    """Impute missing values into a NEW partition; the source is untouched.

    Mutating (unless preview_only): computes imputed rows for ``partition``
    and stores them as ``output_partition`` (default "<partition>_imputed"),
    registered in the task's partition history. The returned payload carries
    the new partition name — pass it to ``run_experiment(train_partition=...)``
    or ``optimize_pipeline``; do not assume a fixed suffix. Use
    preview_only=True first to get the comparison report without writing. On
    failure returns an {"error": ...} payload (409 when the output partition
    already exists).

    Args:
        task_name: Registered task name.
        method: Imputation method accepted by the control plane, e.g. "simple".
        partition: Source partition (default: the registered one).
        columns: Restrict imputation to these columns (default: all with missing values).
        params: Method-specific parameters as a JSON object.
        output_partition: Name for the new partition (default "<partition>_imputed").
        preview_only: True → return the report only, write nothing.
        drop_missing_target: Drop rows whose target is missing instead of imputing it.
    """
    body: dict[str, Any] = {"method": method}
    if partition is not None:
        body["partition"] = partition
    if columns is not None:
        body["columns"] = list(columns)
    if params is not None:
        body["params"] = params
    if output_partition is not None:
        body["output_partition"] = output_partition
    body["preview_only"] = bool(preview_only)
    body["drop_missing_target"] = bool(drop_missing_target)
    return _call(client, "POST", f"/api/datasets/{task_name}/impute", json_body=body)


def set_ignored_columns(
    client: ControlPlaneClient, task_name: str, columns: list[str]
) -> dict:
    """Replace the dataset's ignored-column list (excluded from all modeling paths).

    Mutating but reversible: ignored columns stay in the stored partitions and
    fingerprints remain valid; experiments, optimization, baselines, analysis,
    profiles and drift monitoring drop them. The target column cannot be
    ignored and unknown names are rejected. Returns the normalized list plus
    the effective feature count. On failure returns an {"error": ...} payload.

    Args:
        task_name: Registered task name.
        columns: Full replacement list of column names to ignore (empty list clears it).
    """
    if not isinstance(columns, list) or any(not isinstance(c, str) for c in columns):
        return {"error": "columns must be a list of column-name strings"}
    return _call(
        client,
        "PUT",
        f"/api/datasets/{task_name}/ignored-columns",
        json_body={"columns": list(columns)},
    )


def dataset_preview(
    client: ControlPlaneClient, task_name: str, partition: Optional[str] = None, limit: int = 20
) -> dict:
    """Capped sample of stored rows for a partition (read-only; never full export).

    Returns at most ``limit`` rows (hard cap 100 — row-level bulk export is out
    of scope by design) with their column names. Intended to sanity-check a
    registration or imputation result, not to analyze data. When ``partition``
    is omitted the task's registered partition ("train" when present, else the
    first stored one) is resolved via the dataset metadata. On failure returns
    an {"error": ...} payload (404 for unknown task/partition).

    Args:
        task_name: Registered task name.
        partition: Partition to preview (default: resolved from task_overview).
        limit: Rows to fetch, clamped to 1..100 (default 20).
    """
    bounded_limit = max(1, min(int(limit), PREVIEW_LIMIT_MAX))
    if not partition:
        meta = _call(client, "GET", f"/api/datasets/{task_name}")
        partitions = meta.get("partitions") if isinstance(meta, dict) and "error" not in meta else None
        if not isinstance(partitions, dict) or not partitions:
            return {
                "error": (
                    f"could not resolve a default partition for task %r; pass one explicitly "
                    "(see the 'partitions' map from task_overview)" % task_name
                )
            }
        partition = "train" if "train" in partitions else sorted(partitions)[0]
    return _call(
        client,
        "GET",
        f"/api/datasets/{task_name}/{partition}/preview",
        params={"limit": bounded_limit},
    )


def _find_live_job(client: ControlPlaneClient, task_name: str, kind: str) -> Optional[dict]:
    """First job of ``kind`` actually running on the deployment for this task.

    Uses GET /api/jobs/active (live worker registry), never persisted status —
    a stale "queued" record from a crashed process must not block resubmission.
    The guard is advisory: if the liveness lookup itself fails, it yields None
    and submission proceeds (the API itself allows concurrent jobs).
    """
    active = _call(client, "GET", "/api/jobs/active", params={"task_name": task_name})
    items = active.get("jobs") if isinstance(active, dict) else None
    if not isinstance(items, list):
        return None
    for record in items:
        if (
            isinstance(record, dict)
            and record.get("kind") == kind
            and record.get("status") in ACTIVE_STATUSES
        ):
            return record
    return None


def run_experiment(
    client: ControlPlaneClient,
    task_name: str,
    preset: Optional[str] = None,
    config: Optional[dict] = None,
    strategy: str = "eoa",
    length: int = 2,
    max_generations: int = 3,
    num_parents: int = 4,
    train_partition: Optional[str] = None,
    scoring: Optional[str] = None,
    random_state: Optional[int] = None,
    surrogate_estimator: str = "gpr",
    surrogate_itrs: int = 30,
    surrogate_sampling: str = "BoxSample",
    surrogate_min_random_evals: int = 10,
    forbidden: Optional[list] = None,
    backend: str = "local",
    label: Optional[str] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Submit an evolutionary/exhaustive pipeline-search job (returns immediately).

    Mutating: queues a long-running experiment job on the task; the response is
    a job handle {"job_id", "display_name", "status": "queued"} — wait on it
    with ``job_wait``. Duplicate guard (G4): if a live experiment job already
    runs for this task, submission is refused and the existing job id returned
    instead; stop or wait on that one first. Give either ``preset`` or
    ``config``, not both (with neither, the preset follows the task's declared
    problem type). See ``experiment_options`` for valid presets/components/metrics.

    Args:
        task_name: Registered task name.
        preset: "classification" or "regression" quick-start search.
        config: Custom search space {estimator_class: {param: spec}} instead of a preset.
        strategy: "eoa" (default) or "exhaustive".
        length: Maximum pipeline length for the search.
        max_generations: EOA generations (budget knob).
        num_parents: Parents per generation (EOA only).
        train_partition: Partition to train on (default: the registered one).
        scoring: Metric name, e.g. "accuracy" or "neg_root_mean_squared_error".
        random_state: Seed for reproducibility.
        surrogate_estimator: "gpr", "krr" or "svr".
        surrogate_itrs: Surrogate-search iterations (budget knob; default 30).
        surrogate_sampling: "BoxSample", "CompactSample" or "SphereSample".
        surrogate_min_random_evals: Random initial evaluations before exploitation.
        forbidden: Disallowed [[param, value], ...] combinations.
        backend: "local" (default) or "dask".
        label: Optional human name for the job and its bundle.
        run_id: Optional caller-chosen idempotency identifier.
    """
    existing = _find_live_job(client, task_name, "experiment")
    if existing is not None:
        return {
            "error": (
                "A live experiment job already exists for task %r: %s (status %s). "
                "Wait on it with job_wait or stop it with stop_job before submitting a duplicate."
                % (task_name, existing.get("job_id"), existing.get("status"))
            ),
            "duplicate_job_id": existing.get("job_id"),
        }
    body: dict[str, Any] = {
        "strategy": strategy,
        "length": int(length),
        "max_generations": int(max_generations),
        "num_parents": int(num_parents),
        "surrogate_estimator": surrogate_estimator,
        "surrogate_itrs": int(surrogate_itrs),
        "surrogate_sampling": surrogate_sampling,
        "surrogate_min_random_evals": int(surrogate_min_random_evals),
        "backend": backend,
    }
    if preset:
        body["preset"] = preset
    if config is not None:
        body["config"] = config
    if train_partition:
        body["train_partition"] = train_partition
    if scoring:
        body["scoring"] = scoring
    if random_state is not None:
        body["random_state"] = int(random_state)
    if forbidden is not None:
        body["forbidden"] = forbidden
    if label:
        body["label"] = label
    if run_id:
        body["run_id"] = run_id
    return _call(client, "POST", f"/api/experiments/{task_name}/run", json_body=body)


def optimize_pipeline(
    client: ControlPlaneClient,
    task_name: str,
    seq: list[str],
    config: Optional[dict] = None,
    train_partition: Optional[str] = None,
    scoring: Optional[str] = None,
    random_state: Optional[int] = None,
    label: Optional[str] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Submit a fixed-pipeline optimization job over one pipeline sequence.

    Mutating: queues a long-running optimize_pipeline job that searches the
    parameter space of the given component sequence (no architecture search).
    The response is a job handle — wait on it with ``job_wait``. Duplicate
    guard (G4): refused while a live optimize_pipeline job runs for this task.
    Component names and their parameter specs come from ``experiment_options``.

    Args:
        task_name: Registered task name.
        seq: Non-empty ordered list of estimator class names, e.g. ["sklearn.ensemble.RandomForestRegressor"].
        config: Search space {estimator_class: {param: spec}}; defaults per component when omitted.
        train_partition: Partition to train on (default: the registered one).
        scoring: Metric name for the search objective.
        random_state: Seed for reproducibility.
        label: Optional human name for the job and its bundle.
        run_id: Optional caller-chosen idempotency identifier.
    """
    if not isinstance(seq, list) or not seq or any(not isinstance(name, str) for name in seq):
        return {"error": "seq must be a non-empty list of estimator class names"}
    existing = _find_live_job(client, task_name, "optimize_pipeline")
    if existing is not None:
        return {
            "error": (
                "A live optimize_pipeline job already exists for task %r: %s (status %s). "
                "Wait on it with job_wait or stop it with stop_job before submitting a duplicate."
                % (task_name, existing.get("job_id"), existing.get("status"))
            ),
            "duplicate_job_id": existing.get("job_id"),
        }
    body: dict[str, Any] = {"seq": list(seq)}
    if config is not None:
        body["config"] = config
    if train_partition:
        body["train_partition"] = train_partition
    if scoring:
        body["scoring"] = scoring
    if random_state is not None:
        body["random_state"] = int(random_state)
    if label:
        body["label"] = label
    if run_id:
        body["run_id"] = run_id
    return _call(client, "POST", f"/api/experiments/{task_name}/optimize-pipeline", json_body=body)


def job_wait(
    client: ControlPlaneClient, job_id: str, timeout_s: float = JOB_WAIT_DEFAULT_TIMEOUT_S, poll_interval_s: float = DEFAULT_POLL_INTERVAL_S
) -> dict:
    """Wait (bounded) until a job reaches a terminal status, then return its record.

    Polls GET /api/jobs/{job_id} at ~``poll_interval_s`` cadence for up to
    ``timeout_s`` seconds (default cap 120 s; hard ceiling 900 s). Terminal =
    completed, failed, paused or interrupted: the full job record is returned,
    including ``result`` on success and the traceback in ``error`` on failure.
    If the deadline passes first, a {"job_id", "status", "timed_out": true, ...}
    payload comes back instead — call again to keep waiting or inspect with
    ``job_diagnose``; an agent never has to hand-poll. On API-level failures
    (unknown job, unreachable host) returns an {"error": ...} payload.

    Args:
        job_id: Job id from a submit tool's response.
        timeout_s: Max seconds to wait (default 120; clamped to 1..900).
        poll_interval_s: Seconds between polls (clamped to 0.5..30; default 5).
    """
    bounded_timeout = max(1.0, min(float(timeout_s), JOB_WAIT_MAX_TIMEOUT_S))
    interval = max(0.5, min(float(poll_interval_s), 30.0))
    start = time.monotonic()
    deadline = start + bounded_timeout
    while True:
        record = _call(client, "GET", f"/api/jobs/{job_id}")
        # A live job record carries an (empty) "error" field; only API-level
        # failure payloads from _call lack the job identity — distinguish them.
        if isinstance(record, dict) and "error" in record and "job_id" not in record:
            return record
        status = (record or {}).get("status")
        if status in TERMINAL_STATUSES:
            return record
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {
                "job_id": job_id,
                "status": status,
                "timed_out": True,
                "elapsed_s": round(time.monotonic() - start, 2),
                "hint": (
                    "The job had not reached a terminal status within the wait budget. "
                    "Call job_wait again to keep waiting, or inspect it with job_diagnose."
                ),
            }
        time.sleep(min(remaining, interval))


def job_diagnose(client: ControlPlaneClient, job_id: str, log_limit: int = 50) -> dict:
    """Failure rollup for one job: its record plus the tail of its logs.

    Read-only composite of GET /api/jobs/{job_id} and GET /api/logs/{job_id}.
    Returns {"job": <record>, "log_counts": {level: n}, "recent_logs": [...]}.
    On a failed job the recorded traceback is kept (tail only, bounded) — read
    its last lines for the root cause. Use after ``job_wait`` returns a non-
    completed status, or mid-run to check progress. On failure returns an
    {"error": ...} payload.

    Args:
        job_id: Job id from a submit tool's response.
        log_limit: Max recent log entries to include (default 50; clamped to 1..200).
    """
    record = _call(client, "GET", f"/api/jobs/{job_id}")
    # A transport-level error payload from _call lacks a job_id; a *failed*
    # job's genuine record carries an "error" traceback and must be diagnosed.
    if isinstance(record, dict) and "error" in record and "job_id" not in record:
        return {**record, "note": "Could not fetch the job record; no logs included."}
    bounded_limit = max(1, min(int(log_limit), DIAGNOSE_LOG_MAX))
    logs = _call(client, "GET", f"/api/logs/{job_id}", params={"limit": bounded_limit})
    entries: list[dict] = []
    counts: dict[str, int] = {}
    log_error: Optional[str] = None
    if isinstance(logs, dict) and "error" in logs:
        log_error = logs["error"]
    elif isinstance(logs, dict):
        raw_entries = logs.get("entries") or []
        entries = [
            {
                "ts": item.get("ts"),
                "level": item.get("level"),
                "message": item.get("message"),
            }
            for item in raw_entries[-bounded_limit:]
            if isinstance(item, dict)
        ]
        counts = logs.get("counts") or {}
    job_record = record if isinstance(record, dict) else {"job_id": job_id}
    error_text = job_record.get("error")
    if isinstance(error_text, str) and len(error_text) > DIAGNOSE_ERROR_TAIL_CHARS:
        job_record["error"] = (
            "... [traceback truncated to its tail] ...\n" + error_text[-DIAGNOSE_ERROR_TAIL_CHARS:]
        )
    payload: dict[str, Any] = {
        "job": job_record,
        "log_counts": counts,
        "recent_logs": entries,
    }
    if log_error is not None:
        payload["logs_error"] = log_error
    return payload


def stop_job(client: ControlPlaneClient, job_id: str, mode: str = "pause") -> dict:
    """Stop a running/queued job. Mutating: interrupts the worker's execution.

    mode="pause" (default) keeps checkpoints so ``resume_job`` can continue the
    search later; mode="cancel" is final — no resume afterwards. The control
    plane answers with the job id and resulting status. On failure returns an
    {"error": ...} payload (409 when the job is not running in this process).

    Args:
        job_id: Job id from a submit tool's response or job list.
        mode: "pause" (default, resumable) or "cancel" (final).
    """
    if mode not in STOP_MODES:
        return {"error": "mode must be one of %s" % ", ".join(STOP_MODES)}
    return _call(client, "POST", f"/api/jobs/{job_id}/stop", json_body={"mode": mode})


def resume_job(client: ControlPlaneClient, job_id: str) -> dict:
    """Resume a paused/interrupted/failed experiment from its EOA checkpoint.

    Mutating: submits a NEW job that continues the search from the stored
    checkpoint — the response is a fresh job handle {"job_id", ...}; wait on
    that new id with ``job_wait`` (the original record stays as history). Only
    jobs of kind experiment/optimize_pipeline with an EOA checkpoint can be
    resumed; otherwise an {"error": ...} payload comes back (409 when no
    checkpoint exists or a resume is already active for it).

    Args:
        job_id: Job id to resume.
    """
    return _call(client, "POST", f"/api/jobs/{job_id}/resume")


def api_request(
    client: ControlPlaneClient, method: str, path: str, body: Optional[dict] = None
) -> dict:
    """Generic escape hatch: one raw control-plane request, validated by snapshot.

    For endpoints without a curated tool (registry promotion, inference,
    projects, bundles, ...). The METHOD+PATH must exist in the embedded OpenAPI
    operation snapshot of this package version — unknown routes are refused
    with an explanatory payload instead of being guessed at; regenerate the
    snapshot if the deployment runs a newer core. Path parameters are supplied
    by substituting literal values (e.g. /api/jobs/<job_id>). The JSON response
    body is returned verbatim, or {"error": ...} on expected failures — exactly
    like every curated tool.

    Args:
        method: One of GET, POST, PUT, PATCH, DELETE (case-insensitive).
        path: Absolute API path starting with /api/, parameters substituted, e.g. /api/health.
        body: Optional JSON object body for non-GET methods (never for GET).
    """
    normalized = str(method).upper()
    if normalized not in ALLOWED_API_METHODS:
        return {"error": f"method must be one of {', '.join(ALLOWED_API_METHODS)}"}
    path_value = str(path)
    if not path_value.startswith("/api/"):
        return {"error": "path must start with /api/ (control-plane routes only)"}
    if normalized == "GET" and body is not None:
        return {"error": "body is only allowed on non-GET requests"}
    snapshot = _load_openapi_snapshot()
    if isinstance(snapshot, dict):
        return {**snapshot, "error_hint": (
            "The embedded OpenAPI snapshot is missing or unreadable — reinstall "
            "the sksurrogate[mcp] package."
        )}
    operations, version_note = snapshot
    if not _path_in_snapshot(operations, normalized, path_value):
        return {
            "error": (
                f"{normalized} {path_value} is not in the embedded OpenAPI snapshot "
                f"({len(operations)} operations generated from core {version_note}). "
                "The route may have been removed or renamed; use a curated tool for "
                "its area, or regenerate sksurrogate_mcp/openapi_ops.json if the "
                "deployment runs a newer core."
            )
        }
    return _call(client, normalized, path_value, json_body=body)


# ---------------------------------------------------------------------------
# OpenAPI snapshot (embedded route validation for api_request / drift tests)
# ---------------------------------------------------------------------------

_SNAPSHOT_CACHE: Optional[tuple[list[dict], str]] = None
_SNAPSHOT_ERROR: Optional[str] = None


def _load_openapi_snapshot() -> Any:
    """Load the embedded operation list once; a dict payload on failure."""
    global _SNAPSHOT_CACHE, _SNAPSHOT_ERROR
    if _SNAPSHOT_CACHE is not None and _SNAPSHOT_ERROR is None:
        return _SNAPSHOT_CACHE
    try:
        with open(_SNAPSHOT_PATH, encoding="utf-8") as handle:
            payload = json.load(handle)
        operations = [op for op in payload.get("operations", []) if isinstance(op, dict)]
        version_note = str(payload.get("generated_from_version", "unknown"))
    except (OSError, ValueError) as exc:
        _SNAPSHOT_ERROR = f"could not read the embedded OpenAPI snapshot: {exc}"
        return {"error": _SNAPSHOT_ERROR}
    _SNAPSHOT_CACHE = (operations, version_note)
    return _SNAPSHOT_CACHE


def _path_in_snapshot(operations: list[dict], method: str, path_value: str) -> bool:
    """Template match: {param} segments accept exactly one non-empty literal."""
    wanted_segments = [s for s in path_value.split("/") if s != ""]
    for op in operations:
        if str(op.get("method", "")).upper() != method:
            continue
        template_segments = [s for s in str(op.get("path", "")).split("/") if s != ""]
        if len(template_segments) != len(wanted_segments):
            continue
        if all(
            t == w or (t.startswith("{") and t.endswith("}") and bool(w))
            for t, w in zip(template_segments, wanted_segments)
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# MCP server assembly
# ---------------------------------------------------------------------------

#: Tool functions exposed by the Phase 1 server, in presentation order. The
#: registry test asserts on this list without importing the optional SDK.
TOOLS: tuple = (
    health,
    list_tasks,
    task_overview,
    experiment_options,
    register_dataset,
    dataset_preflight,
    impute_missing,
    set_ignored_columns,
    dataset_preview,
    run_experiment,
    optimize_pipeline,
    job_wait,
    job_diagnose,
    stop_job,
    resume_job,
    api_request,
)


def create_server(client: ControlPlaneClient):
    """Build the MCPServer with every tool bound to ``client``.

    Kept separate from :func:`main` so tests can construct a server against a
    stubbed client without touching the environment. The MCP SDK import is
    deferred here (not at module top) so importing sksurrogate_mcp does not
    require the optional extra — only running the server does. Verified against
    MCP Python SDK 2.x, where FastMCP was renamed to MCPServer.
    """
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        name="sksurrogate",
        title="SKSurrogate control plane",
        description=(
            "Tools for driving a running SKSurrogate deployment: dataset registration, "
            "preprocessing, pipeline search, job handling and model lifecycle. Bound to "
            f"{client.base_url}."
        ),
    )

    def _bind(tool_fn):
        """Wrap a module-level tool for MCPServer registration.

        The first parameter (``client``) is bound as an argument but must not
        appear in the exposed input schema: pydantic builds schemas from the
        inspected signature and annotations, so we reproduce them without
        ``client`` — deliberately NOT via functools.wraps, whose ``__wrapped__``
        attribute would leak ``client`` back into inspect.
        """
        import inspect as _inspect

        sig = _inspect.signature(tool_fn)
        # server.py uses PEP 563 (from __future__ import annotations), so raw
        # parameter annotations are *strings*; resolve them against this module's
        # globals before rebuilding the exposed signature.
        hints = _get_type_hints(tool_fn)
        params_new: list[_inspect.Parameter] = []
        for p in list(sig.parameters.values())[1:]:  # drop the bound client
            if p.annotation is _inspect.Parameter.empty:
                params_new.append(p)
            else:
                resolved = hints.get(p.name, Any)
                params_new.append(
                    _inspect.Parameter(p.name, p.kind, default=p.default, annotation=resolved)
                )
        new_sig = sig.replace(parameters=params_new)

        def wrapper(*args: Any, **kwargs: Any):
            return tool_fn(client, *args, **kwargs)

        wrapper.__name__ = tool_fn.__name__
        wrapper.__qualname__ = f"create_server.<locals>.{tool_fn.__name__}"
        wrapper.__doc__ = tool_fn.__doc__
        annotations = {p.name: p.annotation for p in params_new if p.annotation is not _inspect.Parameter.empty}
        if sig.return_annotation is not _inspect.Parameter.empty:
            annotations["return"] = hints.get("return", Any)
        wrapper.__annotations__ = annotations
        # inspect.signature() honours an explicit __signature__; FunctionType has no
        # typed slot for it, so route the assignment through object.
        object.__setattr__(wrapper, "__signature__", new_sig)
        return wrapper

    for tool_fn in TOOLS:
        server.tool(description=tool_fn.__doc__)(_bind(tool_fn))

    return server


def main(argv: Optional[list] = None) -> int:
    base_url = os.environ.get("SKSURROGATE_API_URL", "").strip()
    api_key = os.environ.get("SKSURROGATE_API_KEY") or None
    if not base_url:
        # stderr only: stdout is reserved for the MCP stdio protocol.
        print(
            "sksurrogate-mcp: SKSURROGATE_API_URL is required (e.g. http://127.0.0.1:8099); "
            "optionally set SKSURROGATE_API_KEY for deployments that enforce auth.",
            file=sys.stderr,
        )
        return 2

    client = ControlPlaneClient(base_url, api_key)
    server = create_server(client)
    server.run()  # stdio transport (default)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
