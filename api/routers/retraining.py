"""Retraining endpoints, wrapping ``SKSurrogate.RetrainingJob``.

Triggers must stay constrained JSON, not arbitrary code (see docs/ui-plan.md
section 4): "always" runs unconditionally, "scheduled"/"on_data" evaluate a
plain boolean (``due``) supplied by the caller (e.g. an external scheduler or
data-watcher) instead of executing a callback from the request body.
Runs as a background job since the underlying trainer may be a full AML
search.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from SKSurrogate import ModelRegistry, RetrainingJob

from ..config import settings
from ..deps import bad_request
from ..jobs import job_manager
from ..routers.experiments import ParamSpec
from ..training import fit_baseline_bundle, fit_experiment_bundle

router = APIRouter(prefix="/api/retraining", tags=["retraining"])


class BaselineTrainerConfig(BaseModel):
    kind: str = "baseline"
    estimator: str = "linear_regression"
    train_partition: str = "train"
    validation_partition: str | None = "validation"


class ExperimentTrainerConfig(BaseModel):
    kind: str = "experiment"
    config: dict[str, dict[str, ParamSpec]]
    length: int = 2
    max_generation: int = 3
    num_parents: int = 4
    train_partition: str = "train"
    scoring: str = "accuracy"
    random_state: int | None = None


class RunRetrainingRequest(BaseModel):
    trainer: BaselineTrainerConfig | ExperimentTrainerConfig
    trigger: str = "always"  # "always" | "scheduled" | "on_data"
    due: bool = True
    promotion_state: str | None = None
    context: dict = {}
    owner: str | None = None
    run_id: str | None = None


def _make_trainer(task_name, trainer_config, checkpoint_dir, owner, run_id):
    if trainer_config.kind == "baseline":
        def _train(context):
            return fit_baseline_bundle(
                task_name,
                estimator=trainer_config.estimator,
                train_partition=trainer_config.train_partition,
                validation_partition=trainer_config.validation_partition,
                owner=owner,
                run_id=run_id,
            )

        return _train
    if trainer_config.kind == "experiment":
        def _train(context):
            config = {
                estimator: {name: spec.model_dump(exclude_none=True) for name, spec in params.items()}
                for estimator, params in trainer_config.config.items()
            }
            bundle, _ = fit_experiment_bundle(
                task_name,
                config=config,
                checkpoint_dir=checkpoint_dir,
                length=trainer_config.length,
                max_generation=trainer_config.max_generation,
                num_parents=trainer_config.num_parents,
                train_partition=trainer_config.train_partition,
                scoring=trainer_config.scoring,
                random_state=trainer_config.random_state,
                owner=owner,
                run_id=run_id,
            )
            return bundle

        return _train
    raise bad_request("trainer.kind must be 'baseline' or 'experiment'")


@router.post("/{task_name}/run")
def run_retraining(task_name: str, body: RunRetrainingRequest):
    """Submit a retraining job: fits a bundle (if triggered) and registers it."""
    if body.trigger not in {"always", "scheduled", "on_data"}:
        raise bad_request("trigger must be one of always, scheduled, on_data")

    def make_job(job_id):
        checkpoint_dir = settings.task_checkpoint_dir(task_name, job_id)
        trainer = _make_trainer(task_name, body.trainer, checkpoint_dir, body.owner, body.run_id)
        registry = ModelRegistry(settings.registry_dir)
        job = RetrainingJob(trainer, registry, task_name, promotion_state=body.promotion_state)
        trigger_fn = None if body.trigger == "always" else (lambda context: body.due)

        def _run():
            return job.run(trigger=trigger_fn, context=body.context)

        return _run

    job_id = job_manager.submit(task_name, "retraining", make_job)
    return {"job_id": job_id, "status": "queued"}
