"""Persistent research-project metadata linked to existing API artifacts."""

import hashlib
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

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


class ProjectEvidenceReference(BaseModel):
    relation: Literal["supports", "contradicts"]
    artifact: ProjectArtifactReference


class AddProjectRecordRequest(BaseModel):
    kind: Literal["hypothesis", "analysis", "finding", "conclusion"]
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=10000)
    evidence: list[ProjectEvidenceReference] = Field(default_factory=list, max_length=20)
    research_question: str | None = Field(default=None, max_length=1000)
    status: Literal[
        "proposed", "under_investigation", "supported", "contradicted", "inconclusive"
    ] | None = None

    @field_validator("title", "text")
    @classmethod
    def validate_nonblank(cls, value):
        if not value.strip():
            raise ValueError("record title and text must not be blank")
        return value.strip()

    @field_validator("research_question")
    @classmethod
    def validate_research_question(cls, value):
        if value is not None and not value.strip():
            raise ValueError("research_question must not be blank")
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def require_claim_evidence(self):
        if self.kind in ("finding", "conclusion") and not self.evidence:
            raise ValueError("findings and conclusions require at least one evidence reference")
        if self.kind == "hypothesis":
            if self.status is None:
                self.status = "proposed"
        elif self.status is not None or self.research_question is not None:
            raise ValueError("status and research_question are only valid for hypothesis records")
        return self


class AddProjectEvidenceRequest(BaseModel):
    relation: Literal["supports", "contradicts"]
    artifact: ProjectArtifactReference


class UpdateProjectHypothesisStatusRequest(BaseModel):
    status: Literal[
        "proposed", "under_investigation", "supported", "contradicted", "inconclusive"
    ]


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
    source: Literal["human", "generated"] = "human"
    model: str | None = Field(default=None, max_length=200)
    citations: list[str] = Field(default_factory=list, max_length=40)
    sources: list[dict] = Field(default_factory=list, max_length=20)
    computed_facts: list[dict] = Field(default_factory=list, max_length=12)
    synthesis_job_id: str | None = Field(default=None, max_length=64)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        if not value.strip():
            raise ValueError("note text must not be blank")
        return value.strip()

    @model_validator(mode="after")
    def validate_provenance(self):
        if self.source != "generated":
            return self
        if not self.citations:
            raise ValueError("generated notes must carry at least one source citation")
        if not self.model:
            raise ValueError("generated notes must record the provider model")
        known = {str(fact.get("id")) for fact in self.computed_facts if fact.get("id")}
        unknown = [citation for citation in self.citations if citation not in known]
        if unknown:
            raise ValueError(
                "citations %s have no matching computed fact" % ", ".join(unknown)
            )
        if len(json.dumps(self.computed_facts, default=str)) > 30000:
            raise ValueError("computed facts exceed the 30000-character storage budget")
        return self


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
    project = json.loads(path.read_text(encoding="utf-8"))
    _normalize_hypothesis_statuses(project)
    return project


def _normalize_hypothesis_statuses(project):
    for record in project.get("records", []):
        if record.get("kind") != "hypothesis" or record.get("status") is not None:
            continue
        history = record.get("status_history")
        if history:
            status = history[-1]["status"]
        else:
            status = "proposed"
            history = [
                {
                    "status": status,
                    "changed_at": record.get("created_at", project.get("created_at")),
                }
            ]
        record["status"] = status
        record["status_history"] = history


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


def _stable_fingerprint(value):
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _artifact_snapshot(artifact):
    if artifact.kind == "dataset_partition":
        with open_tracker(artifact.task_name) as tracker:
            partition = tracker.dataset_splits().get(artifact.artifact_id)
        details = {
            key: partition.get(key)
            for key in (
                "fingerprint",
                "ingestion_timestamp",
                "rows",
                "columns",
                "target",
                "feature_count",
            )
        }
    elif artifact.kind == "model_bundle":
        from .bundles import run_configuration

        configuration = run_configuration(artifact.task_name, artifact.artifact_id)
        details = {
            "model_version": configuration["model_version"],
            "created_at": configuration["created_at"],
            "run_id": configuration["run_id"],
            "lineage": configuration["lineage"],
            "run": configuration["run"],
            "dependencies": configuration["dependencies"],
        }
    else:
        job = job_manager.get(artifact.artifact_id)
        if job is None:
            raise bad_request("job reference no longer exists")
        configuration = {
            key: value
            for key, value in job.items()
            if key not in ("status", "updated_at", "result", "error")
        }
        details = {
            "configuration_fingerprint": _stable_fingerprint(configuration),
            "result_fingerprint": (
                _stable_fingerprint(job["result"])
                if job.get("result") is not None
                else None
            ),
        }

    identity = {
        "kind": artifact.kind,
        "task_name": artifact.task_name,
        "artifact_id": artifact.artifact_id,
    }
    details = json.loads(
        json.dumps(details, ensure_ascii=False, sort_keys=True, default=str)
    )
    return {
        "schema": "sksurrogate.project-artifact-snapshot.v1",
        "captured_at": _now(),
        "identity": identity,
        "fingerprint": _stable_fingerprint({"identity": identity, "details": details}),
        "details": details,
    }


def _serialize_artifacts(artifacts, existing_artifacts=()):
    prior = {
        (item["kind"], item["task_name"], item["artifact_id"]): item
        for item in existing_artifacts
    }
    serialized = []
    for artifact in artifacts:
        item = artifact.model_dump()
        key = (artifact.kind, artifact.task_name, artifact.artifact_id)
        previous = prior.get(key, {})
        item["snapshot"] = previous.get("snapshot") or _artifact_snapshot(artifact)
        serialized.append(item)
    return serialized


def _project_record(
    project_id,
    body,
    *,
    created_at=None,
    notes=None,
    records=None,
    existing_artifacts=(),
):
    timestamp = _now()
    return {
        "schema": "sksurrogate.research-project.v2",
        "project_id": project_id,
        "name": body.name,
        "description": body.description,
        "research_questions": body.research_questions,
        "artifacts": _serialize_artifacts(body.artifacts, existing_artifacts),
        "notes": notes or [],
        "records": records or [],
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
    for project in projects:
        project.setdefault("records", [])
        _normalize_hypothesis_statuses(project)
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
    project = _read_project(project_id)
    project.setdefault("records", [])
    return project


@router.get("/{project_id}/summary")
def project_summary(project_id: str):
    """Workflow progress for a project: which linked artifacts still exist, plus a stage checklist."""
    project = _read_project(project_id)
    artifacts = []
    analysis_job_completed = False
    modeling_job_completed = False
    job_result_available = False
    for item in project["artifacts"]:
        exists, status = True, None
        try:
            _validate_artifacts([ProjectArtifactReference(**item)])
        except HTTPException as exc:
            if exc.status_code not in (400, 404):
                raise
            exists = False
        if item["kind"] == "job" and exists:
            job = job_manager.get(item["artifact_id"]) or {}
            status = job.get("status")
            if status == "completed":
                job_result_available = job_result_available or job.get("result") is not None
                if job.get("kind") in ("sensitivity", "nested-cv"):
                    analysis_job_completed = True
                elif job.get("kind") in (
                    "experiment",
                    "optimize_pipeline",
                    "retraining",
                ):
                    modeling_job_completed = True
        snapshot_status = "missing" if not exists else "unverified"
        if exists and item.get("snapshot"):
            current_snapshot = _artifact_snapshot(ProjectArtifactReference(**item))
            snapshot_status = (
                "current"
                if current_snapshot["fingerprint"] == item["snapshot"].get("fingerprint")
                else "changed"
            )
        artifacts.append(
            {
                **item,
                "exists": exists,
                "status": status,
                "snapshot_status": snapshot_status,
            }
        )
    present = lambda kind: any(a["kind"] == kind and a["exists"] for a in artifacts)
    records = project.get("records", [])
    stages = [
        {
            "stage": "question",
            "label": "Research question defined",
            "done": bool(project.get("research_questions")),
        },
        {
            "stage": "hypothesis",
            "label": "Hypothesis recorded",
            "done": any(record.get("kind") == "hypothesis" for record in records),
        },
        {"stage": "data", "label": "Data registered", "done": present("dataset_partition")},
        {
            "stage": "analysis",
            "label": "Analysis completed",
            "done": analysis_job_completed,
        },
        {
            "stage": "model",
            "label": "Modeling run completed",
            "done": modeling_job_completed,
        },
        {
            "stage": "result",
            "label": "Result available",
            "done": job_result_available or present("model_bundle"),
        },
        {
            "stage": "conclusion",
            "label": "Conclusion recorded",
            "done": any(record.get("kind") == "conclusion" for record in records),
        },
    ]
    return {"project_id": project_id, "artifacts": artifacts, "stages": stages}


@router.post("/{project_id}/artifacts")
def add_project_artifact(project_id: str, body: ProjectArtifactReference):
    """Append one artifact link without replacing other project metadata."""
    with _PROJECT_LOCK:
        project = _read_project(project_id)
        artifact_key = (body.kind, body.task_name, body.artifact_id)
        if any(
            (item["kind"], item["task_name"], item["artifact_id"]) == artifact_key
            for item in project["artifacts"]
        ):
            return project
        if len(project["artifacts"]) >= 100:
            raise bad_request("a project cannot contain more than 100 artifact references")
        _validate_artifacts([body])
        project["artifacts"].append(
            {**body.model_dump(), "snapshot": _artifact_snapshot(body)}
        )
        project["updated_at"] = _now()
        _write_project(project)
    return project


@router.put("/{project_id}")
def update_project(project_id: str, body: UpdateProjectRequest):
    _validate_artifacts(body.artifacts)
    with _PROJECT_LOCK:
        existing = _read_project(project_id)
        linked_artifacts = {
            (item.kind, item.task_name, item.artifact_id) for item in body.artifacts
        }
        required_artifacts = {
            (
                evidence["artifact"]["kind"],
                evidence["artifact"]["task_name"],
                evidence["artifact"]["artifact_id"],
            )
            for record in existing.get("records", [])
            for evidence in record.get("evidence", [])
        }
        if not required_artifacts.issubset(linked_artifacts):
            raise bad_request("artifact references used as research evidence cannot be removed")
        project = _project_record(
            project_id,
            body,
            created_at=existing["created_at"],
            notes=existing.get("notes", []),
            records=existing.get("records", []),
            existing_artifacts=existing.get("artifacts", []),
        )
        _write_project(project)
    return project


@router.post("/{project_id}/records")
def add_project_record(project_id: str, body: AddProjectRecordRequest):
    with _PROJECT_LOCK:
        project = _read_project(project_id)
        if (
            body.research_question is not None
            and body.research_question not in project.get("research_questions", [])
        ):
            raise bad_request("hypothesis research_question must match a question in this project")
        artifact_keys = {
            (item["kind"], item["task_name"], item["artifact_id"])
            for item in project.get("artifacts", [])
        }
        evidence_keys = set()
        for evidence in body.evidence:
            artifact = evidence.artifact
            key = (artifact.kind, artifact.task_name, artifact.artifact_id)
            if key not in artifact_keys:
                raise bad_request("research evidence must reference an artifact linked to this project")
            evidence_key = (evidence.relation, *key)
            if evidence_key in evidence_keys:
                raise bad_request("research evidence references must be unique within a record")
            evidence_keys.add(evidence_key)

        created_at = _now()
        record = {
            "record_id": uuid.uuid4().hex,
            "kind": body.kind,
            "title": body.title,
            "text": body.text,
            "evidence": [
                item.model_dump() for item in body.evidence
            ],
            "created_at": created_at,
        }
        if body.research_question is not None:
            record["research_question"] = body.research_question
        if body.kind == "hypothesis":
            record["status"] = body.status
            record["status_history"] = [
                {"status": body.status, "changed_at": created_at}
            ]
        records = project.setdefault("records", [])
        if len(records) >= 1000:
            raise bad_request("a project cannot contain more than 1000 research records")
        records.append(record)
        project["updated_at"] = _now()
        _write_project(project)
    return project


@router.post("/{project_id}/records/{record_id}/evidence")
def add_project_record_evidence(
    project_id: str,
    record_id: str,
    body: AddProjectEvidenceRequest,
):
    with _PROJECT_LOCK:
        project = _read_project(project_id)
        record = next(
            (
                item
                for item in project.get("records", [])
                if item.get("record_id") == record_id
            ),
            None,
        )
        if record is None:
            raise not_found("No research record %r found in project %r" % (record_id, project_id))
        artifact = body.artifact
        artifact_key = (artifact.kind, artifact.task_name, artifact.artifact_id)
        linked_artifact_keys = {
            (item["kind"], item["task_name"], item["artifact_id"])
            for item in project.get("artifacts", [])
        }
        if artifact_key not in linked_artifact_keys:
            raise bad_request("research evidence must reference an artifact linked to this project")
        evidence = record.setdefault("evidence", [])
        evidence_key = (body.relation, *artifact_key)
        if any(
            (
                item["relation"],
                item["artifact"]["kind"],
                item["artifact"]["task_name"],
                item["artifact"]["artifact_id"],
            )
            == evidence_key
            for item in evidence
        ):
            raise bad_request("research evidence references must be unique within a record")
        if len(evidence) >= 20:
            raise bad_request("a research record cannot contain more than 20 evidence references")
        evidence.append(body.model_dump())
        project["updated_at"] = _now()
        _write_project(project)
    return project


@router.post("/{project_id}/records/{record_id}/status")
def update_project_hypothesis_status(
    project_id: str,
    record_id: str,
    body: UpdateProjectHypothesisStatusRequest,
):
    with _PROJECT_LOCK:
        project = _read_project(project_id)
        record = next(
            (
                item
                for item in project.get("records", [])
                if item.get("record_id") == record_id
            ),
            None,
        )
        if record is None:
            raise not_found("No research record %r found in project %r" % (record_id, project_id))
        if record.get("kind") != "hypothesis":
            raise bad_request("only hypothesis records have a status")
        if record.get("status") == body.status:
            return project
        changed_at = _now()
        record["status"] = body.status
        record.setdefault("status_history", []).append(
            {"status": body.status, "changed_at": changed_at}
        )
        project["updated_at"] = changed_at
        _write_project(project)
    return project


@router.post("/{project_id}/notes")
def add_project_note(project_id: str, body: AddProjectNoteRequest):
    with _PROJECT_LOCK:
        project = _read_project(project_id)
        note: dict = {
            "note_id": uuid.uuid4().hex,
            "text": body.text,
            "created_at": _now(),
            "source": body.source,
        }
        if body.source == "generated":
            note.update(
                {
                    "model": body.model,
                    "citations": list(body.citations),
                    "sources": list(body.sources),
                    "computed_facts": list(body.computed_facts),
                    "synthesis_job_id": body.synthesis_job_id,
                }
            )
        project.setdefault("notes", []).append(note)
        project["updated_at"] = _now()
        _write_project(project)
    return project
