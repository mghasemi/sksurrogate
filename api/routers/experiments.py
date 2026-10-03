"""AutoML pipeline search endpoints, wrapping ``SKSurrogate.AML.eoa_fit``.

Runs as a background job (see ``api.jobs``): AML/EOA searches can take a
long time, so the request only submits the search and returns a job id.
Fitting logic is shared with the Bundles and Retraining routers via
``api.training``.
"""

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from SKSurrogate import save_bundle

from ..config import settings
from ..deps import bad_request
from ..jobs import dask_available, job_manager
from ..training import fit_experiment_bundle, fit_pipeline_bundle, standard_scoring_options

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


class ParamSpec(BaseModel):
    type: str  # "real" | "integer" | "categorical"
    low: float | None = None
    high: float | None = None
    items: list | None = None
    depends_on: dict[str, list] | None = None  # {other_param: [allowed values]} conditional rule


#: A search-space entry is either a range spec or a plain scalar — the latter only
#: appears under the reserved ``"stacking"`` directive (see ``api.training``).
ParamValue = ParamSpec | bool | int | None
SearchSpace = dict[str, dict[str, ParamValue]]


def _dump_config(config):
    """Turn validated request config values back into plain JSON (specs or scalars)."""
    dumped = {}
    for estimator, params in (config or {}).items():
        row = {}
        for name, spec in params.items():
            row[name] = spec.model_dump(exclude_none=True) if isinstance(spec, BaseModel) else spec
        dumped[estimator] = row
    return dumped


class RunExperimentRequest(BaseModel):
    config: SearchSpace
    length: int = 2
    max_generation: int = 3
    num_parents: int = 4
    train_partition: str = "train"
    scoring: str = "accuracy"
    random_state: int | None = None
    owner: str | None = None
    run_id: str | None = None
    surrogate_mode: bool = False
    surrogate_itrs: int | None = None  # per-surrogate iteration budget (surrogate mode only)
    forbidden: list[list] | None = None  # [["param", value], ...] disallowed combinations
    backend: Literal["local", "dask"] = "local"


@router.get("/scoring-options")
def scoring_options():
    """Standard scikit-learn scorers that can serve as the AML/EOA optimization objective.

    The chosen ``scoring`` value is handed to ``AML`` and becomes the metric the
    evolutionary search maximizes, so this is the direct source of truth for the
    Experiments/Retraining "Scoring" dropdown.
    """
    backends = ["local"]
    if dask_available():
        backends.append("dask")
    return {"groups": standard_scoring_options(), "default": "accuracy", "backends": backends}


def submit_experiment(task_name, body, *, checkpoint_job_id=None, run_id=None):
    if body.backend == "dask" and not dask_available():
        raise HTTPException(status_code=503, detail="Dask backend requested but dask.distributed is not installed")
    if body.surrogate_itrs is not None and body.surrogate_itrs < 1:
        raise bad_request("surrogate_itrs must be a positive integer")
    config = _dump_config(body.config)

    def make_job(job_id):
        def _run():
            bundle, evaluation_history, top_pipelines, pareto = fit_experiment_bundle(
                task_name,
                config=config,
                checkpoint_dir=settings.task_checkpoint_dir(task_name, checkpoint_job_id or job_id),
                length=body.length,
                max_generation=body.max_generation,
                num_parents=body.num_parents,
                train_partition=body.train_partition,
                scoring=body.scoring,
                random_state=body.random_state,
                owner=body.owner,
                run_id=run_id or body.run_id or job_id,
                surrogate_mode=body.surrogate_mode,
                surrogate_itrs=body.surrogate_itrs,
                forbidden=body.forbidden,
            )
            bundles_dir = settings.task_bundles_dir(task_name)
            bundles_dir.mkdir(parents=True, exist_ok=True)
            bundle_path = save_bundle(bundle, bundles_dir / (bundle.model_version + ".bundle"))
            return {
                "model_version": bundle.model_version,
                "bundle_path": str(bundle_path),
                "train_score": bundle.metrics.get("train_score"),
                "evaluation_history": evaluation_history,
                "top_pipelines": top_pipelines,
                "pareto": pareto,
            }

        return _run

    def metadata_for_job(job_id):
        effective_run_id = run_id or body.run_id or job_id
        request = body.model_dump(mode="json")
        request["run_id"] = effective_run_id
        return {
            "resume": {
                "request": request,
                "checkpoint_job_id": checkpoint_job_id or job_id,
                "run_id": effective_run_id,
            }
        }

    job_id = job_manager.submit(
        task_name,
        "experiment",
        make_job,
        backend=body.backend,
        metadata_factory=metadata_for_job,
    )
    return {"job_id": job_id, "status": "queued"}


@router.post("/{task_name}/run")
def run_experiment(task_name: str, body: RunExperimentRequest):
    """Submit an AML/EOA pipeline search job over the given search-space config."""
    return submit_experiment(task_name, body)


def resume_experiment(record):
    """Re-submit a failed experiment against its existing EOA checkpoint."""
    resume = record["resume"]
    body = RunExperimentRequest.model_validate(resume["request"])
    return submit_experiment(
        record["task_name"],
        body,
        checkpoint_job_id=resume["checkpoint_job_id"],
        run_id=resume["run_id"],
    )


class OptimizePipelineRequest(BaseModel):
    seq: list[str]  # ordered component names, e.g. ["sklearn.linear_model.LogisticRegression"]
    config: SearchSpace | None = None  # optional per-component search space
    train_partition: str = "train"
    scoring: str = "accuracy"
    random_state: int | None = None
    owner: str | None = None
    run_id: str | None = None


@router.post("/{task_name}/optimize-pipeline")
def optimize_pipeline(task_name: str, body: OptimizePipelineRequest):
    """Submit a job that optimizes one explicit pipeline structure (no search over structures)."""
    if not body.seq:
        raise bad_request("seq must name at least one component")
    config = _dump_config(body.config)

    def make_job(job_id):
        def _run():
            bundle = fit_pipeline_bundle(
                task_name,
                seq=body.seq,
                config=config,
                checkpoint_dir=settings.task_checkpoint_dir(task_name, job_id),
                train_partition=body.train_partition,
                scoring=body.scoring,
                random_state=body.random_state,
                owner=body.owner,
                run_id=body.run_id,
            )
            bundles_dir = settings.task_bundles_dir(task_name)
            bundles_dir.mkdir(parents=True, exist_ok=True)
            bundle_path = save_bundle(bundle, bundles_dir / (bundle.model_version + ".bundle"))
            return {
                "model_version": bundle.model_version,
                "bundle_path": str(bundle_path),
                "train_score": bundle.metrics.get("train_score"),
                "pipeline": list(body.seq),
            }

        return _run

    job_id = job_manager.submit(task_name, "optimize_pipeline", make_job)
    return {"job_id": job_id, "status": "queued"}
