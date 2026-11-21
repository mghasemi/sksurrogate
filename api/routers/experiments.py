"""AutoML pipeline search endpoints, wrapping ``SKSurrogate.AML.eoa_fit``.

Runs as a background job (see ``api.jobs``): AML/EOA searches can take a
long time, so the request only submits the search and returns a job id.
Fitting logic is shared with the Bundles and Retraining routers via
``api.training``.
"""

from typing import Literal

from fastapi import APIRouter, HTTPException, Depends
from pydantic import AliasChoices, BaseModel, Field, model_validator

from SKSurrogate import save_bundle

from ..config import settings
from ..safety import safe_segment
from ..deps import bad_request, open_tracker, task_access
from ..partitions import resolve_partition
from ..problem_type import declared_problem_type
from ..jobs import dask_available, job_manager
from ..training import SURROGATE_FAMILIES, SURROGATE_SAMPLERS, fit_experiment_bundle, fit_pipeline_bundle, standard_scoring_options

router = APIRouter(prefix="/api/experiments", tags=["experiments"], dependencies=[Depends(task_access)])


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
    config: SearchSpace | None = None  # omit when a preset is given
    preset: Literal["classification", "regression"] | None = None
    strategy: Literal["eoa", "exhaustive"] = "eoa"  # how the pipeline space is searched
    length: int = 2
    max_generations: int = Field(
        default=3,
        validation_alias=AliasChoices("max_generations", "max_generation"),
    )
    num_parents: int = 4
    #: Omitted → the task's registered partition ("train" when present, else the
    #: only/first one).
    train_partition: str | None = None
    scoring: str | None = None  # defaults to the preset's metric, else accuracy
    random_state: int | None = None
    owner: str | None = None
    run_id: str | None = None
    surrogate_estimator: str = "gpr"  # one regressor applied to every candidate pipeline (gpr|krr|svr)
    surrogate_itrs: int = 30  # surrogate-search iterations
    surrogate_sampling: str = "BoxSample"  # BoxSample | CompactSample | SphereSample
    surrogate_min_random_evals: int = 10  # random initial sample size before surrogate exploitation
    forbidden: list[list] | None = None  # [["param", value], ...] disallowed combinations
    backend: Literal["local", "dask"] = "local"
    label: str | None = None  # optional human name for the job and its bundle

    @model_validator(mode="before")
    @classmethod
    def reject_conflicting_generation_names(cls, values):
        # Neither a config nor a preset is allowed: the preset then defaults to the
        # task's declared problem type (regression stays regression), which is the
        # whole point of storing that declaration.
        if isinstance(values, dict) and values.get("config") and values.get("preset"):
            raise ValueError("Provide either a config or a preset, not both.")
        if (
            isinstance(values, dict)
            and "max_generations" in values
            and "max_generation" in values
        ):
            raise ValueError("Use only 'max_generations'; do not send both spellings.")
        return values


#: Quick-start presets mirror ``AML(preset=...)``; each is a small, bounded search.
PRESETS = [
    {
        "value": "classification",
        "label": "Classification",
        "description": "Random forest and extra trees with a bounded search; scored by accuracy.",
        "default_scoring": "accuracy",
    },
    {
        "value": "regression",
        "label": "Regression",
        "description": "Random forest and extra trees with a bounded search; scored by negative RMSE.",
        "default_scoring": "neg_root_mean_squared_error",
    },
]


@router.get("/presets")
def presets():
    """Quick-start presets for the Experiments page (no search space to write)."""
    return {"presets": PRESETS}


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


def _effective_scoring(body, preset=None):
    if body.scoring:
        return body.scoring
    defaults = {preset_entry["value"]: preset_entry["default_scoring"] for preset_entry in PRESETS}
    return defaults.get(preset if preset is not None else body.preset, "accuracy")


def _declared_preset(task_name):
    """The task's declared problem type as a Quick-start preset, or ``None``.

    A task registered before the declaration existed has no stored type; the
    caller keeps the historical default in that case.
    """
    try:
        with open_tracker(task_name) as tracker:
            return declared_problem_type(tracker.GetMetadata())
    except Exception:  # pragma: no cover - a missing task keeps the default
        return None


def submit_experiment(task_name, body, *, checkpoint_job_id=None, run_id=None):
    if body.backend == "dask" and not dask_available():
        raise HTTPException(status_code=503, detail="Dask backend requested but dask.distributed is not installed")
    if body.surrogate_itrs < 1:
        raise bad_request("surrogate_itrs must be a positive integer")
    if body.surrogate_min_random_evals < 1:
        raise bad_request("surrogate_min_random_evals must be a positive integer")
    if body.surrogate_estimator not in SURROGATE_FAMILIES:
        raise bad_request(
            "unknown surrogate estimator: %s (choose from %s)"
            % (body.surrogate_estimator, ", ".join(SURROGATE_FAMILIES))
        )
    if body.surrogate_sampling not in SURROGATE_SAMPLERS:
        raise bad_request(
            "unknown surrogate sampling: %s (choose from %s)"
            % (body.surrogate_sampling, ", ".join(SURROGATE_SAMPLERS))
        )
    config = _dump_config(body.config) if body.config else None
    # Quick start without an explicit preset follows the task's declared problem
    # type, so a regression task no longer silently runs a classification search.
    # An explicitly supplied config — even ``{}`` — is the caller's choice and is
    # never replaced by a preset.
    preset = body.preset
    if body.config is None and preset is None:
        preset = _declared_preset(task_name)
        if preset is None:
            # A request-shape problem: keep the historical 422 contract.
            raise HTTPException(
                status_code=422,
                detail=(
                    "Provide a search-space config or a preset (this task has no "
                    "declared problem type to default from)."
                ),
            )

    train_partition = resolve_partition(task_name, body.train_partition)

    def make_job(job_id):
        def _run():
            bundle, evaluation_history, top_pipelines, pareto = fit_experiment_bundle(
                task_name,
                config=config,
                checkpoint_dir=settings.task_checkpoint_dir(task_name, checkpoint_job_id or job_id),
                length=body.length,
                max_generation=body.max_generations,
                num_parents=body.num_parents,
                train_partition=train_partition,
                scoring=_effective_scoring(body, preset),
                random_state=body.random_state,
                owner=body.owner,
                run_id=run_id or body.run_id or job_id,
                surrogate_estimator=body.surrogate_estimator,
                surrogate_itrs=body.surrogate_itrs,
                surrogate_sampling=body.surrogate_sampling,
                surrogate_min_random_evals=body.surrogate_min_random_evals,
                forbidden=body.forbidden,
                preset=preset,
                strategy=body.strategy,
                resume=checkpoint_job_id is not None,
                label=body.label,
            )
            bundles_dir = settings.task_bundles_dir(task_name)
            bundles_dir.mkdir(parents=True, exist_ok=True)
            bundle_path = save_bundle(bundle, bundles_dir / (safe_segment(bundle.model_version, "model version") + ".bundle"))
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
        label=body.label,
    )
    return _submitted(job_id)


def _submitted(job_id):
    """Submit response: the stable id plus the human name the UI can render at once."""
    record = job_manager.get(job_id) or {}
    return {
        "job_id": job_id,
        "display_name": record.get("display_name"),
        "status": "queued",
    }


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
    label: str | None = None  # optional human name for the job and its bundle


@router.post("/{task_name}/optimize-pipeline")
def optimize_pipeline(task_name: str, body: OptimizePipelineRequest):
    """Submit a job that optimizes one explicit pipeline structure (no search over structures)."""
    return submit_optimize_pipeline(task_name, body)


def submit_optimize_pipeline(task_name, body, *, checkpoint_job_id=None, run_id=None):
    """Submit (or re-submit) one pipeline-structure optimization job.

    ``checkpoint_job_id`` points a resumed run at the original job's checkpoint
    directory, mirroring :func:`submit_experiment`, so pause/resume keeps the
    same search state.
    """
    if not body.seq:
        raise bad_request("seq must name at least one component")
    config = _dump_config(body.config)
    effective_run_id = run_id or body.run_id
    train_partition = resolve_partition(task_name, body.train_partition)

    def make_job(job_id):
        def _run():
            bundle = fit_pipeline_bundle(
                task_name,
                seq=body.seq,
                config=config,
                checkpoint_dir=settings.task_checkpoint_dir(task_name, checkpoint_job_id or job_id),
                train_partition=train_partition,
                scoring=body.scoring,
                random_state=body.random_state,
                owner=body.owner,
                run_id=effective_run_id or job_id,
                resume=checkpoint_job_id is not None,
                label=body.label,
            )
            bundles_dir = settings.task_bundles_dir(task_name)
            bundles_dir.mkdir(parents=True, exist_ok=True)
            bundle_path = save_bundle(bundle, bundles_dir / (safe_segment(bundle.model_version, "model version") + ".bundle"))
            return {
                "model_version": bundle.model_version,
                "bundle_path": str(bundle_path),
                "train_score": bundle.metrics.get("train_score"),
                "pipeline": list(body.seq),
            }

        return _run

    def metadata_for_job(job_id):
        request = body.model_dump(mode="json")
        effective = effective_run_id or job_id
        request["run_id"] = effective
        return {
            "resume": {
                "request": request,
                "checkpoint_job_id": checkpoint_job_id or job_id,
                "run_id": effective,
            }
        }

    job_id = job_manager.submit(
        task_name,
        "optimize_pipeline",
        make_job,
        metadata_factory=metadata_for_job,
        label=body.label,
    )
    return _submitted(job_id)


def resume_optimize_pipeline(record):
    """Re-submit a paused/interrupted pipeline optimization against its checkpoint."""
    resume = record["resume"]
    body = OptimizePipelineRequest.model_validate(resume["request"])
    return submit_optimize_pipeline(
        record["task_name"],
        body,
        checkpoint_job_id=resume["checkpoint_job_id"],
        run_id=resume["run_id"],
    )
