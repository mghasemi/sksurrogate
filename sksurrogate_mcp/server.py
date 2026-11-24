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

import asyncio
import json
import math
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

#: Phase-2 analysis methods accepted by POST /api/sensitivity/{task}/run —
#: mirrors api/routers/sensitivity.py (pre-validation only; the API re-checks).
SENSITIVITY_METHODS = ("sobol", "morris", "delta-mmnt")

#: Registry lifecycle states in promotion order — mirrors REGISTRY_STATES from
#: SKSurrogate/modelbundle.py (pre-validation only; the gate re-checks).
REGISTRY_STATES = ("candidate", "validated", "staging", "production", "archived")

#: Log entries sampled per job_wait poll to derive progress notifications (G2).
PROGRESS_LOG_LIMIT = 100

#: Tools whose bound wrapper exposes an injected MCP Context for progress reporting.
_CONTEXT_TOOLS = frozenset({"job_wait"})

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
    "job_wait": (
        ("GET", "/api/jobs/{job_id}"),
        ("GET", "/api/logs/{job_id}"),
    ),
    "job_diagnose": (
        ("GET", "/api/jobs/{job_id}"),
        ("GET", "/api/logs/{job_id}"),
    ),
    "stop_job": (("POST", "/api/jobs/{job_id}/stop"),),
    "resume_job": (("POST", "/api/jobs/{job_id}/resume"),),
    # Phase 2 — analysis
    "sensitivity_analysis": (
        ("GET", "/api/jobs/active"),
        ("POST", "/api/sensitivity/{task_name}/run"),
    ),
    "top_features": (("GET", "/api/sensitivity/{task_name}/top-features"),),
    "explain_model": (("POST", "/api/evaluation/{task_name}/{model_version}/explain"),),
    "explain_row": (
        ("POST", "/api/evaluation/{task_name}/{model_version}/explain/row"),
    ),
    "evaluation_metrics": (("GET", "/api/evaluation/{task_name}/metrics"),),
    "nested_cv": (
        ("GET", "/api/jobs/active"),
        ("POST", "/api/evaluation/{task_name}/nested-cv"),
    ),
    # Phase 2 — lifecycle
    "registry_summary": (("GET", "/api/registry/{task_name}/summary"),),
    "promote_model": (("POST", "/api/registry/{task_name}/promote"),),
    "quality_gate_check": (
        ("POST", "/api/quality-gates/{task_name}/{model_version}/check"),
    ),
    "predict": (("POST", "/api/inference/{task_name}/predict"),),
    "predict_batch": (("POST", "/api/inference/{task_name}/predict-batch"),),
    # Phase 2 — research record (append-only semantics live in the API)
    "project_create": (("POST", "/api/projects"),),
    "project_update": (
        ("GET", "/api/projects/{project_id}"),
        ("PUT", "/api/projects/{project_id}"),
    ),
    "project_note": (("POST", "/api/projects/{project_id}/notes"),),
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


def _progress_summary(logs_payload: Any) -> dict[str, Any]:
    """Distill trial progress from a GET /api/logs/{job_id} payload (G2).

    Trial entries carry structured fields — ``trial`` (int), ``phase`` (str) and
    ``score`` (float in the scorer's own units; api/logs.py flips the search
    loss back before storing, so higher is better by construction). Plain-text
    lines are ignored. Entries arrive newest first, so the latest phase is the
    first one carrying a value. Returns {} when nothing structured was found.
    """
    entries = logs_payload.get("entries") if isinstance(logs_payload, dict) else None
    if not isinstance(entries, list):
        return {}
    max_trial = 0
    best_score: Optional[float] = None
    phase: Optional[str] = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        trial = entry.get("trial")
        if isinstance(trial, int) and not isinstance(trial, bool) and trial > max_trial:
            max_trial = trial
        score = entry.get("score")
        if (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score))
            and (best_score is None or float(score) > best_score)
        ):
            best_score = float(score)
        if phase is None:
            candidate_phase = entry.get("phase")
            if isinstance(candidate_phase, str) and candidate_phase:
                phase = candidate_phase
    summary: dict[str, Any] = {}
    if max_trial:
        summary["max_trial"] = max_trial
    if best_score is not None:
        summary["best_score"] = best_score
    if phase:
        summary["phase"] = phase
    return summary


async def _report_wait_progress(
    ctx, client: ControlPlaneClient, job_id: str, status: Optional[str], elapsed_s: float, budget_s: float
) -> None:
    """Emit one MCP progress notification for a live job (G2). Never raises.

    ``ctx`` is the SDK-injected Context under a real MCP request and None when
    a tool function is called directly (unit tests); any failure while sampling
    logs or reporting is swallowed so progress can never break the wait loop.
    """
    if ctx is None:
        return
    try:
        logs = _call(client, "GET", f"/api/logs/{job_id}", params={"limit": PROGRESS_LOG_LIMIT})
        summary = _progress_summary(logs)
        parts = [f"{status} after {elapsed_s:.0f}s" if status else f"after {elapsed_s:.0f}s"]
        if "max_trial" in summary:
            parts.append(f"trial {summary['max_trial']}")
        if "phase" in summary:
            parts.append(f"phase {summary['phase']}")
        if "best_score" in summary:
            parts.append(f"best score {summary['best_score']:.4g}")
        await ctx.report_progress(progress=elapsed_s, total=budget_s, message=" · ".join(parts))
    except Exception:  # progress is best-effort; the wait loop must survive it
        pass


async def job_wait(
    client: ControlPlaneClient,
    job_id: str,
    timeout_s: float = JOB_WAIT_DEFAULT_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    ctx=None,
) -> dict:
    """Wait (bounded) until a job reaches a terminal status, then return its record.

    Polls GET /api/jobs/{job_id} at ~``poll_interval_s`` cadence for up to
    ``timeout_s`` seconds (default cap 120 s; hard ceiling 900 s). Terminal =
    completed, failed, paused or interrupted: the full job record is returned,
    including ``result`` on success and the traceback in ``error`` on failure.
    While the job is live, each poll samples its log feed and emits an MCP
    progress notification (trial count, phase, best score in scorer units) so a
    client can show live status without per-turn polling (G2). If the deadline
    passes first, a {"job_id", "status", "timed_out": true, ...} payload comes
    back instead — call again to keep waiting or inspect with ``job_diagnose``;
    an agent never has to hand-poll. On API-level failures (unknown job,
    unreachable host) returns an {"error": ...} payload.

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
        await _report_wait_progress(ctx, client, job_id, status, time.monotonic() - start, bounded_timeout)
        await asyncio.sleep(min(remaining, interval))


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


# ---------------------------------------------------------------------------
# Phase 2 — analysis / lifecycle / research record
# ---------------------------------------------------------------------------


def sensitivity_analysis(
    client: ControlPlaneClient,
    task_name: str,
    method: str = "sobol",
    n_features_to_select: int = 5,
    train_partition: Optional[str] = None,
) -> dict:
    """Submit a global sensitivity-analysis job (returns immediately).

    Mutating: queues a long-running feature-importance analysis with the chosen
    method — "sobol" (default), "morris" or "delta-mmnt". The response is a job
    handle {"job_id", ...} — wait on it with ``job_wait``; afterwards read the
    ranked importances with ``top_features``. Duplicate guard (G4): refused
    while a live sensitivity job runs for this task. On failure returns an
    {"error": ...} payload.

    Args:
        task_name: Registered task name (needs stored data, no fitted model).
        method: "sobol" (default), "morris" or "delta-mmnt".
        n_features_to_select: Top-k features the analysis focuses on (default 5; min 1).
        train_partition: Partition to analyse (default: the registered one).
    """
    if method not in SENSITIVITY_METHODS:
        return {"error": f"method must be one of {', '.join(SENSITIVITY_METHODS)}"}
    existing = _find_live_job(client, task_name, "sensitivity")
    if existing is not None:
        return {
            "error": (
                "A live sensitivity job already exists for task %r: %s (status %s). "
                "Wait on it with job_wait or stop it with stop_job before submitting a duplicate."
                % (task_name, existing.get("job_id"), existing.get("status"))
            ),
            "duplicate_job_id": existing.get("job_id"),
        }
    body: dict[str, Any] = {
        "method": method,
        "n_features_to_select": max(1, int(n_features_to_select)),
    }
    if train_partition:
        body["train_partition"] = train_partition
    return _call(client, "POST", f"/api/sensitivity/{task_name}/run", json_body=body)


def top_features(client: ControlPlaneClient, task_name: str, num: int = 10) -> dict:
    """Consensus top features across the stored sensitivity weightings (read-only).

    Ranks features by how often they appear in the top ``num`` of every stored
    weighting column for the task — populated only after a sensitivity run has
    persisted its weights. When nothing is stored yet, returns {"available":
    false, "hint": ...} rather than an error: submit one with
    ``sensitivity_analysis`` first. On failure returns an {"error": ...} payload
    (404 when no dataset is registered for the task).

    Args:
        task_name: Registered task name.
        num: Top-k window per weighting column (default 10; min 1).
    """
    return _call(
        client, "GET", f"/api/sensitivity/{task_name}/top-features", params={"num": max(1, int(num))}
    )


def explain_model(
    client: ControlPlaneClient,
    task_name: str,
    model_version: str,
    partition: Optional[str] = None,
    max_rows: int = 500,
    n_repeats: int = 5,
    top_k: int = 3,
) -> dict:
    """Row-stable global explanation of one stored bundle (permutation importance + PDP).

    Read-only but computationally heavier than a preview: shuffles each feature
    to measure its score drop and draws partial-dependence curves for the most
    important ones. Use ``registry_summary`` first to find valid model versions.
    On failure returns an {"error": ...} payload (404 unknown bundle, 422 when
    the partition cannot support the requested reference sample).

    Args:
        task_name: Registered task name.
        model_version: Stored bundle version id (see registry_summary or a job result).
        partition: Reference-data partition (default: the registered one).
        max_rows: Reference-sample size, clamped to 20..1000 (default 500).
        n_repeats: Permutation repeats per feature, clamped to 1..20 (default 5).
        top_k: PDP curves for the top-k features, clamped to 0..6 (default 3; 0 = none).
    """
    body: dict[str, Any] = {
        "max_rows": max(20, min(int(max_rows), 1000)),
        "n_repeats": max(1, min(int(n_repeats), 20)),
        "top_k": max(0, min(int(top_k), 6)),
    }
    if partition:
        body["partition"] = partition
    return _call(client, "POST", f"/api/evaluation/{task_name}/{model_version}/explain", json_body=body)


def explain_row(
    client: ControlPlaneClient,
    task_name: str,
    model_version: str,
    row_index: int = 0,
    features: Optional[list[str]] = None,
    baseline: str = "reference_mean",
    partition: Optional[str] = None,
    baseline_row_index: Optional[int] = None,
    orders: int = 1,
) -> dict:
    """Local explanation for ONE input row against a reference baseline (read-only).

    Decomposes the model's output difference between one specific row and the
    baseline into per-feature contributions — with ``orders=2`` also pairwise
    interactions. The row is addressed by position in the resolved partition;
    use ``dataset_preview`` to check which rows that is. On failure returns an
    {"error": ...} payload (404 unknown bundle, 400 when a feature is missing
    from the partition or the baseline cannot be built).

    Args:
        task_name: Registered task name.
        model_version: Stored bundle version id.
        row_index: Position of the explained row in the partition (default 0; min 0).
        features: Restrict to these feature names (default: all features).
        baseline: "reference_mean" (finite mean / mode per column) or "row".
        partition: Partition holding the explained row (default: the registered one).
        baseline_row_index: Required when ``baseline`` is "row" — position of the reference row.
        orders: Interaction order, clamped to 1..10 (default 1 = main effects only).
    """
    if baseline not in ("reference_mean", "row"):
        return {"error": "baseline must be one of reference_mean, row"}
    body: dict[str, Any] = {
        "row_index": max(0, int(row_index)),
        "baseline": baseline,
        "orders": max(1, min(int(orders), 10)),
    }
    if features is not None:
        body["features"] = list(features)
    if partition:
        body["partition"] = partition
    if baseline_row_index is not None:
        body["baseline_row_index"] = max(0, int(baseline_row_index))
    return _call(client, "POST", f"/api/evaluation/{task_name}/{model_version}/explain/row", json_body=body)


def evaluation_metrics(
    client: ControlPlaneClient, task_name: str, model_version: Optional[str] = None
) -> dict:
    """Metric options relevant to a task's problem family (read-only).

    Returns the metrics available for the task plus their default — probe one
    bundle via ``model_version`` or let the API scan the task's stored bundles.
    Use before choosing ``scoring`` for experiments. On failure returns an
    {"error": ...} payload (404 when no dataset is registered).

    Args:
        task_name: Registered task name.
        model_version: Optional bundle version to probe instead of scanning all.
    """
    params: dict[str, Any] = {}
    if model_version:
        params["model_version"] = model_version
    return _call(client, "GET", f"/api/evaluation/{task_name}/metrics", params=params)


def nested_cv(
    client: ControlPlaneClient,
    task_name: str,
    alias: Optional[str] = None,
    model_version: Optional[str] = None,
    inner_cv: int = 2,
    outer_cv: int = 2,
    scoring: Optional[str] = None,
    repeats: int = 1,
    train_partition: Optional[str] = None,
) -> dict:
    """Submit a nested cross-validation evaluation job (returns immediately).

    Mutating: queues a long-running outer/inner CV evaluation of one stored
    bundle — give ``alias`` or ``model_version``, exactly one (both or neither
    is refused here before any request goes out). The response is a job handle;
    wait on it with ``job_wait`` and the completed record carries the metric
    distributions. Duplicate guard (G4): refused while a live nested_cv job
    runs for this task. On failure returns an {"error": ...} payload.

    Args:
        task_name: Registered task name.
        alias: Registry state alias of the bundle to evaluate, e.g. "staging".
        model_version: Stored bundle version id (mutually exclusive with ``alias``).
        inner_cv: Inner CV folds, clamped to 2..10 (default 2).
        outer_cv: Outer CV folds, clamped to 2..10 (default 2).
        scoring: Metric name override (default: the task's declared metric).
        repeats: Repeat the full evaluation this many times, clamped to 1..5 (default 1).
        train_partition: Partition to evaluate on (default: the registered one).
    """
    if bool(alias) == bool(model_version):
        return {"error": "exactly one of alias or model_version is required"}
    existing = _find_live_job(client, task_name, "nested-cv")
    if existing is not None:
        return {
            "error": (
                "A live nested_cv job already exists for task %r: %s (status %s). "
                "Wait on it with job_wait or stop it with stop_job before submitting a duplicate."
                % (task_name, existing.get("job_id"), existing.get("status"))
            ),
            "duplicate_job_id": existing.get("job_id"),
        }
    body: dict[str, Any] = {
        "inner_cv": max(2, min(int(inner_cv), 10)),
        "outer_cv": max(2, min(int(outer_cv), 10)),
        "repeats": max(1, min(int(repeats), 5)),
    }
    if alias:
        body["alias"] = alias
    else:
        body["model_version"] = model_version
    if scoring:
        body["scoring"] = scoring
    if train_partition:
        body["train_partition"] = train_partition
    return _call(client, "POST", f"/api/evaluation/{task_name}/nested-cv", json_body=body)


def registry_summary(client: ControlPlaneClient, task_name: str) -> dict:
    """Model-registry summary for a task (read-only).

    Lists the stored bundles with their state aliases (candidate / staging /
    production), sizes and registered quality reports — where to look before
    promoting or predicting. On failure returns an {"error": ...} payload (404
    when no dataset is registered for the task).

    Args:
        task_name: Registered task name.
    """
    return _call(client, "GET", f"/api/registry/{task_name}/summary")


def promote_model(
    client: ControlPlaneClient,
    task_name: str,
    model_version: str,
    state: str,
    approvers: Optional[list[str]] = None,
    quality_report: Optional[dict] = None,
    required_approvals: int = 1,
) -> dict:
    """Move a stored bundle to another registry lifecycle state. Mutating.

    States in promotion order: candidate → validated → staging → production
    (plus "archived"). The control plane enforces the promotion rules — e.g.
    moving into "production" requires an attached passing quality report and
    enough recorded approvers, so check one first with ``quality_gate_check``.
    Returns the updated bundle record; on failure returns an {"error": ...}
    payload (422 when a rule is violated, 404 for unknown task/bundle).

    Args:
        task_name: Registered task name.
        model_version: Stored bundle version id to move.
        state: Target state — "candidate", "validated", "staging", "production" or "archived".
        approvers: Approver identifiers recorded with the transition (default: none).
        quality_report: Quality report object to attach if one is not yet on the bundle.
        required_approvals: Distinct approvers the gate counts (min 1; default 1).
    """
    if state not in REGISTRY_STATES:
        return {"error": f"state must be one of {', '.join(REGISTRY_STATES)}"}
    body: dict[str, Any] = {
        "model_version": model_version,
        "state": state,
        "approvers": list(approvers or ()),
        "required_approvals": max(1, int(required_approvals)),
    }
    if quality_report is not None:
        body["quality_report"] = quality_report
    return _call(client, "POST", f"/api/registry/{task_name}/promote", json_body=body)


def quality_gate_check(
    client: ControlPlaneClient,
    task_name: str,
    model_version: str,
    metric_thresholds: Optional[dict] = None,
    max_bundle_size_bytes: Optional[int] = None,
) -> dict:
    """Run the deployment's quality gate against one stored bundle (read-only).

    Re-fits nothing: evaluates the stored bundle against ``metric_thresholds``
    (e.g. {"accuracy": 0.8}) and an optional size cap, returning a report with
    ``passed`` plus per-check detail. A passing report is what promotes into
    "production" — run this before calling ``promote_model`` there. On failure
    returns an {"error": ...} payload (404 unknown bundle).

    Args:
        task_name: Registered task name.
        model_version: Stored bundle version id to check.
        metric_thresholds: {metric: minimum value} pairs; defaults to the gate's own thresholds when omitted.
        max_bundle_size_bytes: Optional upper bound on serialized bundle size (bytes).
    """
    body: dict[str, Any] = {}
    if metric_thresholds is not None:
        body["metric_thresholds"] = dict(metric_thresholds)
    if max_bundle_size_bytes is not None:
        body["max_bundle_size_bytes"] = int(max_bundle_size_bytes)
    return _call(
        client, "POST", f"/api/quality-gates/{task_name}/{model_version}/check", json_body=body
    )


def predict(
    client: ControlPlaneClient,
    task_name: str,
    rows: list[dict],
    alias: Optional[str] = None,
    model_version: Optional[str] = None,
) -> dict:
    """Predict on in-memory rows with one stored bundle (read-only; nothing persisted).

    Give ``alias`` or ``model_version``, exactly one — the alias form resolves
    through the registry state at call time ("staging", "production", ...), so
    predictions follow promotions automatically. Rows are dicts keyed by feature
    column name; they pass through untouched, so each row must carry exactly the
    bundle's schema columns (check them with ``dataset_preview`` first) — missing
    or unknown columns fail strict validation. Returns per-row predictions plus
    class probabilities when the bundle is a classifier. On failure returns an
    {"error": ...} payload (404 unknown bundle/alias, 422 on schema mismatch).

    Args:
        task_name: Registered task name.
        rows: One dict per row — {column: value}, columns matching the bundle's schema.
        alias: Registry state alias to resolve (e.g. "staging").
        model_version: Stored bundle version id (mutually exclusive with ``alias``).
    """
    if bool(alias) == bool(model_version):
        return {"error": "exactly one of alias or model_version is required"}
    body: dict[str, Any] = {"rows": list(rows)}
    if alias:
        body["alias"] = alias
    else:
        body["model_version"] = model_version
    return _call(client, "POST", f"/api/inference/{task_name}/predict", json_body=body)


def predict_batch(
    client: ControlPlaneClient,
    task_name: str,
    partition: str,
    alias: Optional[str] = None,
    model_version: Optional[str] = None,
) -> dict:
    """Score an ENTIRE stored partition with one bundle (read-only; result persisted).

    Give ``alias`` or ``model_version``, exactly one. Predicts every row of the
    named partition and stores a per-row prediction file in the task's
    inference folder — use for offline evaluation on held-out partitions; for
    ad-hoc rows use ``predict`` instead (it persists nothing). On failure
    returns an {"error": ...} payload (404 unknown bundle/partition/alias, 422
    when a column cannot be decoded by the stored preprocessing).

    Args:
        task_name: Registered task name.
        partition: Stored partition to score in full.
        alias: Registry state alias to resolve (e.g. "staging").
        model_version: Stored bundle version id (mutually exclusive with ``alias``).
    """
    if bool(alias) == bool(model_version):
        return {"error": "exactly one of alias or model_version is required"}
    body: dict[str, Any] = {"partition": partition}
    if alias:
        body["alias"] = alias
    else:
        body["model_version"] = model_version
    return _call(client, "POST", f"/api/inference/{task_name}/predict-batch", json_body=body)


def project_create(
    client: ControlPlaneClient,
    name: str,
    description: Optional[str] = None,
    research_questions: Optional[list[str]] = None,
    artifacts: Optional[list[dict]] = None,
) -> dict:
    """Create a research-project record (the benchmark-record loop). Mutating.

    Projects are append-only evidence records: created here with an optional
    description, open research questions and initial artifact references, then
    extended via ``project_update`` / ``project_note`` — the API never deletes
    evidence artifacts once linked. Returns the stored record including its
    ``project_id`` (use it in every later call). On failure returns an {"error":
    ...} payload (409 when a project with this name already exists).

    Args:
        name: Unique project name.
        description: Free-text context for what the project studies.
        research_questions: Open questions to track (each 1..1000 non-whitespace chars).
        artifacts: Initial evidence artifact references (see docs/research-projects.rst shapes).
    """
    body: dict[str, Any] = {"name": name}
    if description is not None:
        body["description"] = description
    if research_questions is not None:
        body["research_questions"] = list(research_questions)
    if artifacts is not None:
        body["artifacts"] = list(artifacts)
    return _call(client, "POST", "/api/projects", json_body=body)


def project_update(
    client: ControlPlaneClient,
    project_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    research_questions: Optional[list[str]] = None,
    artifacts: Optional[list[dict]] = None,
) -> dict:
    """Update one project record with read-modify-write semantics. Mutating.

    Fetches the current record, applies only the provided fields, and writes it
    back — omitted fields are preserved untouched (the API itself also refuses
    to let artifacts used as research evidence be unlinked). ``research_questions``
    is a flat list of question strings; structured progress — records with
    hypothesis statuses (proposed / under_investigation / supported /
    contradicted / inconclusive), findings and conclusions — lives on separate
    project endpoints, reachable via ``api_request``. Returns the updated
    record; on failure returns an {"error": ...} payload (404 unknown project,
    422 invalid field shapes).

    Args:
        project_id: Project's ``project_id`` from its create/get response (32-char hex string).
        name: New unique name (rename), 1..200 chars.
        description: Replace the free-text context (max 4000 chars).
        research_questions: Replacement list of open questions, each 1..1000 non-whitespace chars.
        artifacts: Artifact references to add / re-snapshot (shape per docs/research-projects.rst).
    """
    current = _call(client, "GET", f"/api/projects/{project_id}")
    if isinstance(current, dict) and "error" in current:
        return {**current, "note": "Could not fetch the project record; nothing was written."}
    body: dict[str, Any] = {"name": str(current.get("name") or "")}
    if name is not None:
        body["name"] = name
    if description is not None:
        body["description"] = description
    elif current.get("description") is not None:
        body["description"] = current.get("description")
    questions = research_questions if research_questions is not None else (current.get("research_questions") or [])
    body["research_questions"] = list(questions)
    artifacts_out = artifacts if artifacts is not None else (current.get("artifacts") or [])
    body["artifacts"] = list(artifacts_out)
    return _call(client, "PUT", f"/api/projects/{project_id}", json_body=body)


def project_note(
    client: ControlPlaneClient,
    project_id: str,
    text: str,
    source: str = "human",
    model: Optional[str] = None,
    citations: Optional[list[str]] = None,
    sources: Optional[list[dict]] = None,
    computed_facts: Optional[list[dict]] = None,
) -> dict:
    """Append a note to one project record (append-only). Mutating.

    Notes are never edited or deleted afterwards; ``source`` marks who wrote it —
    "human" (default) keeps the human-review semantics. Generated notes carry
    explicit provenance and the API enforces it: set ``model``, provide at least
    one citation, and every citation must match an ``id`` of a fact in
    ``computed_facts`` (facts are structured values attached to the note; total
    JSON budget 30k chars). Returns the created note record; on failure returns
    an {"error": ...} payload (404 unknown project, 422 blank text or broken
    provenance chain).

    Args:
        project_id: Project's ``project_id`` from its create/get response (32-char hex string).
        text: Note body — non-blank after stripping, max 10000 characters.
        source: "human" (default) or "generated".
        model: Model name; required by the API when ``source`` is "generated".
        citations: Ids of computed facts cited by this note (max 40).
        sources: Free-form source records attached to the note (list of dicts, max 20).
        computed_facts: Structured facts {id, ...} that citations may point at (max 12).
    """
    if source not in ("human", "generated"):
        return {"error": "source must be one of human, generated"}
    body: dict[str, Any] = {"text": text, "source": source}
    if model is not None:
        body["model"] = model
    if citations is not None:
        body["citations"] = list(citations)
    if sources is not None:
        body["sources"] = list(sources)
    if computed_facts is not None:
        body["computed_facts"] = list(computed_facts)
    return _call(client, "POST", f"/api/projects/{project_id}/notes", json_body=body)


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

#: Tool functions exposed by the server, in presentation order (grouped by
#: process stage: discovery, data, experiment, jobs, analysis, lifecycle,
#: research record; the escape hatch last). The registry test asserts on this
#: list without importing the optional SDK.
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
    sensitivity_analysis,
    top_features,
    explain_model,
    explain_row,
    evaluation_metrics,
    nested_cv,
    registry_summary,
    promote_model,
    quality_gate_check,
    predict,
    predict_batch,
    project_create,
    project_update,
    project_note,
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
    # Re-exported for _bind only: the SDK's context-injection detection matches
    # parameters annotated with this exact class (find_context_parameter).
    from mcp.server.mcpserver.context import Context as MCPContext

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
        attribute would leak ``client`` back into inspect. For tools listed in
        ``_CONTEXT_TOOLS`` (G2 progress reporting) an extra trailing ``ctx``
        parameter is injected from the SDK's Context under a real request; it is
        excluded from the schema too, and direct calls without one simply see
        no progress notifications.
        """
        import inspect as _inspect

        sig = _inspect.signature(tool_fn)
        # server.py uses PEP 563 (from __future__ import annotations), so raw
        # parameter annotations are *strings*; resolve them against this module's
        # globals before rebuilding the exposed signature.
        hints = _get_type_hints(tool_fn)
        wants_context = tool_fn.__name__ in _CONTEXT_TOOLS
        skip = {"client"} | ({"ctx"} if wants_context else set())
        params_new: list[_inspect.Parameter] = []
        for p in sig.parameters.values():
            if p.name in skip:  # drop the bound client / injected ctx
                continue
            if p.annotation is _inspect.Parameter.empty:
                params_new.append(p)
            else:
                resolved = hints.get(p.name, Any)
                params_new.append(
                    _inspect.Parameter(p.name, p.kind, default=p.default, annotation=resolved)
                )
        new_sig = sig.replace(parameters=params_new)

        if wants_context and _inspect.iscoroutinefunction(tool_fn):

            async def ctx_wrapper(*args: Any, ctx=None, **kwargs: Any):
                # The SDK injects a live mcp.server.mcpserver.context.Context as
                # the "ctx" kwarg when this tool runs under an MCP request (see
                # FunctionTool / find_context_parameter); it is excluded from the
                # input schema by _bind and passed through unchanged. Direct calls
                # without one simply see no progress notifications.
                return await tool_fn(client, *args, **kwargs, ctx=ctx)

            wrapper = ctx_wrapper
        else:

            def plain_wrapper(*args: Any, **kwargs: Any):
                return tool_fn(client, *args, **kwargs)

            wrapper = plain_wrapper

        wrapper.__name__ = tool_fn.__name__
        wrapper.__qualname__ = f"create_server.<locals>.{tool_fn.__name__}"
        wrapper.__doc__ = tool_fn.__doc__
        annotations = {p.name: p.annotation for p in params_new if p.annotation is not _inspect.Parameter.empty}
        if sig.return_annotation is not _inspect.Parameter.empty:
            annotations["return"] = hints.get("return", Any)
        if wants_context:
            # find_context_parameter matches this annotation on the registered
            # function, which keeps "ctx" out of the tool's input schema and
            # makes the SDK inject a live Context at call time.
            annotations["ctx"] = MCPContext
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
