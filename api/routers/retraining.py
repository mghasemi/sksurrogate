"""Retraining endpoints, wrapping ``SKSurrogate.RetrainingJob``.

Triggers must stay constrained JSON, not arbitrary code (see docs/ui-plan.md
section 4): "always" runs unconditionally, "scheduled"/"on_data" evaluate a
plain boolean (``due``) supplied by the caller (e.g. an external scheduler or
data-watcher) instead of executing a callback from the request body.
Runs as a background job since the underlying trainer may be a full AML
search. ``/trigger`` is the webhook/scheduler entry point; it validates event
freshness, registered data identity, idempotency, and quality-gated promotion.
"""

import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Depends
from pydantic import AliasChoices, BaseModel, Field, model_validator

from SKSurrogate import ModelRegistry, RetrainingJob

from ..config import settings
from ..deps import bad_request, not_found, open_tracker, task_access
from ..jobs import dask_available, job_manager
from ..routers.experiments import ParamValue, _dump_config
from ..training import fit_baseline_bundle, fit_experiment_bundle

router = APIRouter(prefix="/api/retraining", tags=["retraining"], dependencies=[Depends(task_access)])


class BaselineTrainerConfig(BaseModel):
    kind: str = "baseline"
    estimator: str = "linear_regression"
    train_partition: str = "train"
    validation_partition: str | None = "validation"


class ExperimentTrainerConfig(BaseModel):
    kind: str = "experiment"
    config: dict[str, dict[str, ParamValue]]
    length: int = 2
    max_generations: int = Field(
        default=3,
        validation_alias=AliasChoices("max_generations", "max_generation"),
    )
    num_parents: int = 4
    train_partition: str = "train"
    scoring: str = "accuracy"
    random_state: int | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_conflicting_generation_names(cls, values):
        if (
            isinstance(values, dict)
            and "max_generations" in values
            and "max_generation" in values
        ):
            raise ValueError("Use only 'max_generations'; do not send both spellings.")
        return values


class RunRetrainingRequest(BaseModel):
    trainer: BaselineTrainerConfig | ExperimentTrainerConfig
    trigger: str = "always"  # "always" | "scheduled" | "on_data"
    due: bool = True
    promotion_state: str | None = None
    context: dict = {}
    owner: str | None = None
    run_id: str | None = None
    backend: Literal["local", "dask"] = "local"


class TriggerRetrainingRequest(BaseModel):
    trainer: BaselineTrainerConfig | ExperimentTrainerConfig
    trigger: Literal["scheduled", "on_data"]
    event_id: str = Field(min_length=1, max_length=128)
    occurred_at: datetime
    due: bool = True
    data_partition: str | None = None
    data_fingerprint: str | None = None
    promotion_state: Literal["staging"] = "staging"
    metric_thresholds: dict[str, float] = Field(min_length=1)
    owner: str | None = None
    run_id: str | None = None
    backend: Literal["local", "dask"] = "local"


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
            config = _dump_config(trainer_config.config)
            bundle, _, _, _ = fit_experiment_bundle(
                task_name,
                config=config,
                checkpoint_dir=checkpoint_dir,
                length=trainer_config.length,
                max_generation=trainer_config.max_generations,
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
    if body.backend == "dask" and not dask_available():
        raise HTTPException(status_code=503, detail="Dask backend requested but dask.distributed is not installed")

    def make_job(job_id):
        checkpoint_dir = settings.task_checkpoint_dir(task_name, job_id)
        trainer = _make_trainer(task_name, body.trainer, checkpoint_dir, body.owner, body.run_id)
        registry = ModelRegistry(settings.registry_dir)
        job = RetrainingJob(trainer, registry, task_name, promotion_state=body.promotion_state)
        trigger_fn = None if body.trigger == "always" else (lambda context: body.due)

        def _run():
            return job.run(trigger=trigger_fn, context=body.context)

        return _run

    job_id = job_manager.submit(task_name, "retraining", make_job, backend=body.backend)
    return {"job_id": job_id, "status": "queued"}


@router.post("/{task_name}/trigger")
def trigger_retraining(task_name: str, body: TriggerRetrainingRequest):
    """Accept a fresh scheduler/data event and enqueue an idempotent retraining job."""
    if not re.fullmatch(r"[A-Za-z0-9._:-]+", body.event_id):
        raise bad_request("event_id may contain only letters, numbers, '.', '_', ':', and '-'")
    if body.occurred_at.tzinfo is None or body.occurred_at.utcoffset() is None:
        raise bad_request("occurred_at must include a timezone")
    now = datetime.now(timezone.utc)
    age = now - body.occurred_at.astimezone(timezone.utc)
    if age < -timedelta(seconds=30):
        raise bad_request("trigger event timestamp is in the future")
    if age > timedelta(minutes=10):
        raise HTTPException(status_code=409, detail="trigger event is stale")
    if body.trigger == "scheduled" and (body.data_partition or body.data_fingerprint):
        raise bad_request("scheduled triggers must not include data partition fields")
    if body.trigger == "on_data":
        if not body.due:
            raise bad_request("on_data triggers do not accept due=false")
        if not body.data_partition or not body.data_fingerprint:
            raise bad_request("on_data triggers require data_partition and data_fingerprint")
        if body.trainer.train_partition != body.data_partition:
            raise bad_request(
                "on_data data_partition must match the trainer's train_partition"
            )
        if not settings.mltrace_db_path(task_name).exists():
            raise not_found("No registered dataset for task %r" % task_name)
        with open_tracker(task_name) as tracker:
            partition_record = tracker.dataset_splits().get(body.data_partition)
            if partition_record is None:
                raise not_found(
                    "No registered dataset partition %r for task %r"
                    % (body.data_partition, task_name)
                )
            try:
                tracker.get_dataframe(partition=body.data_partition)
            except ValueError:
                raise not_found(
                    "No registered dataset partition %r for task %r"
                    % (body.data_partition, task_name)
                )
            if partition_record.get("fingerprint") != body.data_fingerprint:
                raise HTTPException(
                    status_code=409,
                    detail="trigger data fingerprint does not match the registered partition",
                )

    if not body.metric_thresholds or any(
        not math.isfinite(value) for value in body.metric_thresholds.values()
    ):
        raise bad_request("metric_thresholds must contain finite minimum values")
    if body.backend == "dask" and not dask_available():
        raise HTTPException(
            status_code=503,
            detail="Dask backend requested but dask.distributed is not installed",
        )

    trigger_context = {
        "trigger": body.trigger,
        "event_id": body.event_id,
        "occurred_at": body.occurred_at.astimezone(timezone.utc).isoformat(),
        "due": body.due,
        "data_partition": body.data_partition,
        "data_fingerprint": body.data_fingerprint,
    }
    idempotency_key = (
        "event:%s" % body.event_id
        if body.trigger == "scheduled"
        else "data:%s:%s" % (body.data_partition, body.data_fingerprint)
    )
    request_payload = body.model_dump(
        exclude={"event_id", "occurred_at"},
        mode="json",
    )
    request_fingerprint = hashlib.sha256(
        json.dumps(request_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    def make_job(job_id):
        checkpoint_dir = settings.task_checkpoint_dir(task_name, job_id)
        trainer = _make_trainer(
            task_name, body.trainer, checkpoint_dir, body.owner, body.run_id
        )
        registry = ModelRegistry(settings.registry_dir)
        job = RetrainingJob(
            trainer,
            registry,
            task_name,
            promotion_state=body.promotion_state,
            quality_metric_thresholds=body.metric_thresholds,
        )
        return lambda: job.run(
            trigger=lambda context: context["due"],
            context=trigger_context,
        )

    def audit_metadata(_job_id):
        return {
            "trigger_event": trigger_context,
            "promotion_state": body.promotion_state,
            "metric_thresholds": body.metric_thresholds,
        }

    try:
        job_id, created = job_manager.submit_idempotent(
            task_name,
            "retraining",
            idempotency_key,
            request_fingerprint,
            make_job,
            backend=body.backend,
            metadata_factory=audit_metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    record = job_manager.get(job_id)
    return {
        "job_id": job_id,
        "status": record["status"] if record is not None else "queued",
        "duplicate": not created,
    }
