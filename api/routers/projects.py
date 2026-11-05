"""Persistent research-project metadata linked to existing API artifacts."""

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from ..config import settings
from ..deps import bad_request, not_found, open_tracker
from ..jobs import job_manager

router = APIRouter(prefix="/api/projects", tags=["research-projects"])
_PROJECT_LOCK = threading.RLock()
_PROJECT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
_TASK_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_PARTITION_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_JOB_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


class ProjectArtifactReference(BaseModel):
    """Reference an existing dataset partition, model bundle, or background job."""

    kind: Literal["dataset_partition", "model_bundle", "job"]
    task_name: str = Field(min_length=1, max_length=128)
    artifact_id: str = Field(min_length=1, max_length=128)
    label: str | None = Field(default=None, max_length=200)


class ProjectFields(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    research_questions: list[str] = Field(default_factory=list, max_length=50)
    artifacts: list[ProjectArtifactReference] = Field(default_factory=list, max_length=100)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        if not value.strip():
            raise ValueError("name must not be blank")
        return value.strip()

    @field_validator("research_questions")
    @classmethod
    def validate_questions(cls, values):
        if any(not question.strip() or len(question) > 1000 for question in values):
            raise ValueError("research questions must contain 1 to 1000 non-whitespace characters")
        return [question.strip() for question in values]


class CreateProjectRequest(ProjectFields):
    pass


class UpdateProjectRequest(ProjectFields):
    pass


class AddProjectNoteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        if not value.strip():
            raise ValueError("note text must not be blank")
        return value.strip()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _project_path(project_id):
    if not isinstance(project_id, str) or not _PROJECT_ID_PATTERN.fullmatch(project_id):
        raise not_found("No research project %r found" % project_id)
    return settings.projects_dir / (project_id + ".json")


def _read_project(project_id):
    path = _project_path(project_id)
    if not path.is_file():
        raise not_found("No research project %r found" % project_id)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_project(project):
    path = _project_path(project["project_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    with _PROJECT_LOCK:
        temporary_path.write_text(
            json.dumps(project, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(path)


def _validate_artifacts(artifacts):
    seen = set()
    for artifact in artifacts:
        key = (artifact.kind, artifact.task_name, artifact.artifact_id)
        if key in seen:
            raise bad_request("artifact references must be unique within a project")
        seen.add(key)
        if not _TASK_NAME_PATTERN.fullmatch(artifact.task_name):
            raise bad_request("artifact task_name contains unsupported path characters")

        if artifact.kind == "dataset_partition":
            if not _PARTITION_PATTERN.fullmatch(artifact.artifact_id):
                raise bad_request("dataset partition reference has an invalid artifact_id")
            if not settings.mltrace_db_path(artifact.task_name).is_file():
                raise bad_request("dataset partition reference does not exist")
            with open_tracker(artifact.task_name) as tracker:
                if artifact.artifact_id not in tracker.dataset_splits():
                    raise bad_request("dataset partition reference does not exist")
        elif artifact.kind == "model_bundle":
            if not _ARTIFACT_ID_PATTERN.fullmatch(artifact.artifact_id):
                raise bad_request("model bundle reference has an invalid artifact_id")
            bundle_name = artifact.artifact_id + ".bundle"
            candidates = (
                settings.task_bundles_dir(artifact.task_name) / bundle_name,
                settings.registry_dir / artifact.task_name / bundle_name,
            )
            if not any(path.is_file() for path in candidates):
                raise bad_request("model bundle reference does not exist")
        elif artifact.kind == "job":
            if not _JOB_ID_PATTERN.fullmatch(artifact.artifact_id):
                raise bad_request("job reference has an invalid artifact_id")
            record = job_manager.get(artifact.artifact_id)
            if record is None or record.get("task_name") != artifact.task_name:
                raise bad_request("job reference does not exist for the specified task")


def _project_record(project_id, body, *, created_at=None, notes=None):
    timestamp = _now()
    return {
        "schema": "sksurrogate.research-project.v1",
        "project_id": project_id,
        "name": body.name,
        "description": body.description,
        "research_questions": body.research_questions,
        "artifacts": [item.model_dump() for item in body.artifacts],
        "notes": notes or [],
        "created_at": created_at or timestamp,
        "updated_at": timestamp,
    }


@router.get("")
def list_projects():
    """List project metadata without copying linked dataset or run contents."""
    with _PROJECT_LOCK:
        projects = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in settings.projects_dir.glob("*.json")
            if path.is_file()
        ]
    return {"projects": sorted(projects, key=lambda project: project["updated_at"], reverse=True)}


@router.post("")
def create_project(body: CreateProjectRequest):
    """Create a project with questions and references to existing API artifacts."""
    _validate_artifacts(body.artifacts)
    project = _project_record(uuid.uuid4().hex, body)
    _write_project(project)
    return project


@router.get("/{project_id}")
def get_project(project_id: str):
    return _read_project(project_id)


@router.put("/{project_id}")
def update_project(project_id: str, body: UpdateProjectRequest):
    _validate_artifacts(body.artifacts)
    with _PROJECT_LOCK:
        existing = _read_project(project_id)
        project = _project_record(
            project_id,
            body,
            created_at=existing["created_at"],
            notes=existing.get("notes", []),
        )
        _write_project(project)
    return project


@router.post("/{project_id}/notes")
def add_project_note(project_id: str, body: AddProjectNoteRequest):
    with _PROJECT_LOCK:
        project = _read_project(project_id)
        project.setdefault("notes", []).append(
            {
                "note_id": uuid.uuid4().hex,
                "text": body.text,
                "created_at": _now(),
            }
        )
        project["updated_at"] = _now()
        _write_project(project)
    return project
