"""AutoML pipeline search endpoints, wrapping ``SKSurrogate.AML.eoa_fit``.

Runs as a background job (see ``api.jobs``): AML/EOA searches can take a
long time, so the request only submits the search and returns a job id.
Fitting logic is shared with the Bundles and Retraining routers via
``api.training``.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from SKSurrogate import save_bundle

from ..config import settings
from ..jobs import job_manager
from ..training import fit_experiment_bundle, standard_scoring_options

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


class ParamSpec(BaseModel):
    type: str  # "real" | "integer" | "categorical"
    low: float | None = None
    high: float | None = None
    items: list | None = None


class RunExperimentRequest(BaseModel):
    config: dict[str, dict[str, ParamSpec]]
    length: int = 2
    max_generation: int = 3
    num_parents: int = 4
    train_partition: str = "train"
    scoring: str = "accuracy"
    random_state: int | None = None
    owner: str | None = None
    run_id: str | None = None


@router.get("/scoring-options")
def scoring_options():
    """Standard scikit-learn scorers that can serve as the AML/EOA optimization objective.

    The chosen ``scoring`` value is handed to ``AML`` and becomes the metric the
    evolutionary search maximizes, so this is the direct source of truth for the
    Experiments/Retraining "Scoring" dropdown.
    """
    return {"groups": standard_scoring_options(), "default": "accuracy"}


@router.post("/{task_name}/run")
def run_experiment(task_name: str, body: RunExperimentRequest):
    """Submit an AML/EOA pipeline search job over the given search-space config."""
    config = {
        estimator: {name: spec.model_dump(exclude_none=True) for name, spec in params.items()}
        for estimator, params in body.config.items()
    }

    def make_job(job_id):
        def _run():
            bundle, evaluation_history = fit_experiment_bundle(
                task_name,
                config=config,
                checkpoint_dir=settings.task_checkpoint_dir(task_name, job_id),
                length=body.length,
                max_generation=body.max_generation,
                num_parents=body.num_parents,
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
                "evaluation_history": evaluation_history,
            }

        return _run

    job_id = job_manager.submit(task_name, "experiment", make_job)
    return {"job_id": job_id, "status": "queued"}
