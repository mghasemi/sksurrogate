"""Stage D2 — opt-in LLM-assisted project synthesis endpoints.

``GET /api/synthesis/status`` reports whether a provider is configured.
``POST /api/synthesis/projects/{project_id}`` submits a background job that
gathers bounded computed facts, asks the user-supplied OpenAI-compatible
provider for a cited draft, validates the citations, and returns the draft
without persisting it. Persisting is a separate explicit note call carrying
provenance (``POST /api/projects/{project_id}/notes`` with
``source="generated"``).
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from ..deps import project_access
from ..jobs import job_manager
from ..synthesis import (
    SynthesisError,
    build_messages,
    chat_completion,
    provider_config,
    validate_draft,
)
from ..synthesis_facts import gather_facts
from ..training import get_registered_target

router = APIRouter(prefix="/api/synthesis", tags=["synthesis"], dependencies=[Depends(project_access)])


class SynthesisRequest(BaseModel):
    task_name: str = Field(min_length=1, max_length=128)
    model_version: str | None = Field(default=None, max_length=128)
    job_id: str | None = Field(default=None, max_length=64)
    partition: str = Field(default="train", max_length=128)
    include_explanations: bool = True
    include_drift: bool = False
    instruction: str | None = Field(default=None, max_length=500)

    @field_validator("task_name", "partition")
    @classmethod
    def validate_identifiers(cls, value):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("task_name and partition must not be blank")
        return cleaned

    @field_validator("instruction")
    @classmethod
    def validate_instruction(cls, value):
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


@router.get("/status")
def synthesis_status():
    """Report whether LLM synthesis is configured (never returns credentials)."""
    provider = provider_config()
    return {
        "provider": provider.public(),
        "enabled": provider.configured,
        "instructions": {
            "SKSURROGATE_LLM_BASE_URL": "OpenAI-compatible base URL, e.g. https://api.openai.com/v1",
            "SKSURROGATE_LLM_MODEL": "Model name sent in the chat-completions request",
            "SKSURROGATE_LLM_API_KEY": "Optional bearer token (never echoed)",
            "SKSURROGATE_LLM_TIMEOUT": "Optional request timeout in seconds (default 60)",
            "SKSURROGATE_LLM_MAX_TOKENS": "Optional output token budget (default 700)",
        },
    }


@router.post("/projects/{project_id}")
def submit_synthesis(project_id: str, body: SynthesisRequest):
    """Queue a cited-draft synthesis job for one research project."""
    from .projects import _read_project  # local import avoids an import cycle

    _read_project(project_id)
    provider = provider_config()
    if not provider.configured:
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM synthesis is not configured. Set SKSURROGATE_LLM_BASE_URL and "
                "SKSURROGATE_LLM_MODEL (and SKSURROGATE_LLM_API_KEY if required) on the "
                "API process, then retry. No request was sent anywhere."
            ),
        )
    get_registered_target(body.task_name)  # 404 when the task is not registered

    request = body.model_dump(mode="json")

    def _task():
        try:
            facts = gather_facts(
                body.task_name,
                model_version=body.model_version,
                job_id=body.job_id,
                partition=body.partition,
                include_explanations=body.include_explanations,
                include_drift=body.include_drift,
            )
            messages = build_messages(facts, body.instruction)
            text = chat_completion(messages, provider)
            draft = validate_draft(text, facts)
        except SynthesisError as exc:
            raise RuntimeError(str(exc)) from exc
        return {
            "project_id": project_id,
            "task_name": body.task_name,
            "model_version": body.model_version,
            "source_job_id": body.job_id,
            "draft": draft,
            "sources": [
                {"id": fact["id"], "kind": fact["kind"], "label": fact["label"]}
                for fact in facts
            ],
            "computed_facts": facts,
            "provider": provider.public(),
            "interpretation": {
                "category": "generated_interpretation",
                "generated_by": "llm_provider",
                "causal_effect": False,
                "limitations": [
                    "The draft is model-generated prose about the supplied computed facts; it is not a computed result and can be wrong or incomplete.",
                    "Only aggregated, stored facts were sent to the provider — no raw dataset rows or credentials.",
                    "Citations identify which computed source supports a sentence, not a causal relationship.",
                    "Persisting the draft as a project note is an explicit user action; generated notes are labelled and keep their provenance.",
                ],
            },
        }

    job_id = job_manager.submit(
        body.task_name,
        "project-synthesis",
        lambda submitted_id: _task,
        metadata_factory=lambda submitted_id: {"request": request, "project_id": project_id},
    )
    return {"job_id": job_id, "status": "queued", "provider": provider.public()}
