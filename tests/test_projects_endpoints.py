import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sklearn.dummy import DummyRegressor

from SKSurrogate import ModelBundle, save_bundle
from api.config import settings
from api.deps import open_tracker
from api.jobs import job_manager
from api.routers import bundles as bundles_router
from api.routers import datasets as datasets_router
from api.routers import jobs as jobs_router
from api.routers import projects as projects_router
from api.routers.projects import (
    AddProjectEvidenceRequest,
    AddProjectRecordRequest,
    AddProjectNoteRequest,
    CreateProjectRequest,
    ProjectArtifactReference,
    ProjectEvidenceReference,
    UpdateProjectHypothesisStatusRequest,
    UpdateProjectRequest,
)


TASK = "research-project-tests"
BUNDLE_VERSION = "bundle-v1"
JOB_ID = "0123456789abcdef0123456789abcdef"
EXPERIMENT_JOB_ID = "fedcba9876543210fedcba9876543210"


class TestResearchProjectEndpoints(unittest.TestCase):
    def setUp(self):
        self._original_root = settings.root
        self._temporary_directory = tempfile.TemporaryDirectory()
        settings.root = Path(self._temporary_directory.name)
        settings.ensure()
        frame = pd.DataFrame({"feature": [0.0, 1.0, 2.0], "target": [0.0, 1.0, 2.0]})
        with open_tracker(TASK) as tracker:
            tracker.RegisterData(frame, "target", partition="train")

        model = DummyRegressor().fit(frame[["feature"]], frame["target"])
        bundle = ModelBundle(
            model,
            task_name=TASK,
            model_version=BUNDLE_VERSION,
            run_id="baseline-run",
            metadata={
                "reproducibility": {
                    "data": {
                        "train_partition": "train",
                        "train_fingerprint": "test-fingerprint",
                    },
                    "configuration": {"estimator": "DummyRegressor"},
                }
            },
        )
        bundle_path = settings.task_bundles_dir(TASK) / (BUNDLE_VERSION + ".bundle")
        save_bundle(bundle, bundle_path)
        job_manager._write(
            {
                "job_id": JOB_ID,
                "task_name": TASK,
                "kind": "sensitivity",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        )
        job_manager._write(
            {
                "job_id": EXPERIMENT_JOB_ID,
                "task_name": TASK,
                "kind": "experiment",
                "status": "completed",
                "result": {"model_version": BUNDLE_VERSION},
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        )

    def tearDown(self):
        settings.root = self._original_root
        self._temporary_directory.cleanup()

    def _artifact(self, kind, artifact_id, label=None):
        return ProjectArtifactReference(
            kind=kind,
            task_name=TASK,
            artifact_id=artifact_id,
            label=label,
        )

    def test_create_list_update_and_append_note_roundtrip(self):
        project = projects_router.create_project(
            CreateProjectRequest(
                name="Material degradation",
                description="Track analyses of the pilot dataset.",
                research_questions=["Does temperature affect degradation?"],
                artifacts=[self._artifact("dataset_partition", "train")],
            )
        )
        created_at = project["created_at"]
        noted = projects_router.add_project_note(
            project["project_id"],
            AddProjectNoteRequest(text="The first fit suggests a nonlinear effect."),
        )
        updated = projects_router.update_project(
            project["project_id"],
            UpdateProjectRequest(
                name="Material degradation study",
                description="Updated scope.",
                research_questions=["Is the temperature relationship nonlinear?"],
                artifacts=[self._artifact("model_bundle", BUNDLE_VERSION, "baseline")],
            ),
        )

        self.assertEqual(updated["created_at"], created_at)
        self.assertEqual(updated["name"], "Material degradation study")
        self.assertEqual(updated["research_questions"], ["Is the temperature relationship nonlinear?"])
        self.assertEqual(updated["artifacts"][0]["kind"], "model_bundle")
        self.assertEqual(updated["notes"], noted["notes"])
        self.assertEqual(updated["notes"][0]["text"], "The first fit suggests a nonlinear effect.")
        self.assertEqual(projects_router.get_project(project["project_id"]), updated)
        self.assertEqual(projects_router.list_projects()["projects"], [updated])

    def test_research_records_link_supporting_and_contradicting_project_evidence(self):
        project = projects_router.create_project(
            CreateProjectRequest(
                name="Evidence records",
                artifacts=[
                    self._artifact("dataset_partition", "train"),
                    self._artifact("model_bundle", BUNDLE_VERSION),
                ],
            )
        )
        updated = projects_router.add_project_record(
            project["project_id"],
            AddProjectRecordRequest(
                kind="finding",
                title="Model behavior is inconclusive",
                text="The two linked sources point in different directions.",
                evidence=[
                    {
                        "relation": "supports",
                        "artifact": {
                            "kind": "dataset_partition",
                            "task_name": TASK,
                            "artifact_id": "train",
                        },
                    },
                    {
                        "relation": "contradicts",
                        "artifact": {
                            "kind": "model_bundle",
                            "task_name": TASK,
                            "artifact_id": BUNDLE_VERSION,
                        },
                    },
                ],
            ),
        )

        record = updated["records"][0]
        self.assertEqual(record["kind"], "finding")
        self.assertEqual(
            [item["relation"] for item in record["evidence"]],
            ["supports", "contradicts"],
        )
        self.assertEqual(
            record["evidence"][0]["artifact"]["artifact_id"],
            "train",
        )
        run_configuration = bundles_router.run_configuration(TASK, BUNDLE_VERSION)
        self.assertEqual(
            record["evidence"][1]["artifact"]["artifact_id"],
            run_configuration["model_version"],
        )
        self.assertEqual(run_configuration["run_id"], "baseline-run")
        self.assertEqual(
            projects_router.get_project(project["project_id"])["records"],
            updated["records"],
        )

        with self.assertRaises(HTTPException) as context:
            projects_router.update_project(
                project["project_id"],
                UpdateProjectRequest(
                    name="Evidence records",
                    artifacts=[self._artifact("dataset_partition", "train")],
                ),
            )
        self.assertEqual(context.exception.status_code, 400)

    def test_research_record_evidence_must_be_project_linked_and_claims_need_evidence(self):
        project = projects_router.create_project(
            CreateProjectRequest(name="Evidence validation")
        )
        with self.assertRaises(HTTPException) as context:
            projects_router.add_project_record(
                project["project_id"],
                AddProjectRecordRequest(
                    kind="analysis",
                    title="Unlinked source",
                    text="This evidence is not part of this project.",
                    evidence=[
                        {
                            "relation": "supports",
                            "artifact": {
                                "kind": "dataset_partition",
                                "task_name": TASK,
                                "artifact_id": "train",
                            },
                        }
                    ],
                ),
            )
        self.assertEqual(context.exception.status_code, 400)

        with self.assertRaises(ValidationError):
            AddProjectRecordRequest(
                kind="conclusion",
                title="Unsupported conclusion",
                text="No evidence is attached.",
            )

    def test_project_update_preserves_research_records(self):
        project = projects_router.create_project(
            CreateProjectRequest(name="Record preservation")
        )
        recorded = projects_router.add_project_record(
            project["project_id"],
            AddProjectRecordRequest(
                kind="hypothesis",
                title="Initial hypothesis",
                text="A candidate relationship may exist.",
            ),
        )
        updated = projects_router.update_project(
            project["project_id"],
            UpdateProjectRequest(name="Renamed project"),
        )

        self.assertEqual(updated["records"], recorded["records"])

    def test_hypothesis_status_updates_append_history_without_rewriting_evidence(self):
        project = projects_router.create_project(
            CreateProjectRequest(
                name="Hypothesis lifecycle",
                research_questions=["Does temperature affect degradation?"],
                artifacts=[self._artifact("dataset_partition", "train")],
            )
        )
        created = projects_router.add_project_record(
            project["project_id"],
            AddProjectRecordRequest(
                kind="hypothesis",
                title="Temperature response",
                text="Temperature may have a nonlinear relationship with degradation.",
                research_question="Does temperature affect degradation?",
            ),
        )
        original = created["records"][0]
        self.assertEqual(original["status"], "proposed")
        self.assertEqual(original["evidence"], [])
        self.assertEqual(
            original["research_question"],
            "Does temperature affect degradation?",
        )
        self.assertEqual(len(original["status_history"]), 1)

        with_evidence = projects_router.add_project_record_evidence(
            project["project_id"],
            original["record_id"],
            AddProjectEvidenceRequest(
                relation="supports",
                artifact=self._artifact("dataset_partition", "train"),
            ),
        )
        original = with_evidence["records"][0]
        self.assertEqual(original["evidence"][0]["relation"], "supports")

        updated = projects_router.update_project_hypothesis_status(
            project["project_id"],
            original["record_id"],
            UpdateProjectHypothesisStatusRequest(status="under_investigation"),
        )
        revised = updated["records"][0]
        self.assertEqual(revised["status"], "under_investigation")
        self.assertEqual(
            [entry["status"] for entry in revised["status_history"]],
            ["proposed", "under_investigation"],
        )
        self.assertEqual(revised["evidence"], original["evidence"])
        self.assertEqual(revised["text"], original["text"])

        unchanged = projects_router.update_project_hypothesis_status(
            project["project_id"],
            original["record_id"],
            UpdateProjectHypothesisStatusRequest(status="under_investigation"),
        )
        self.assertEqual(
            len(unchanged["records"][0]["status_history"]),
            2,
        )

        with self.assertRaises(HTTPException) as context:
            projects_router.update_project_hypothesis_status(
                project["project_id"],
                "0" * 32,
                UpdateProjectHypothesisStatusRequest(status="supported"),
            )
        self.assertEqual(context.exception.status_code, 404)

        with self.assertRaises(ValidationError):
            AddProjectRecordRequest(
                kind="analysis",
                title="Analysis",
                text="A descriptive analysis.",
                status="supported",
            )

        with self.assertRaises(HTTPException) as context:
            projects_router.add_project_record_evidence(
                project["project_id"],
                original["record_id"],
                AddProjectEvidenceRequest(
                    relation="supports",
                    artifact=self._artifact("dataset_partition", "train"),
                ),
            )
        self.assertEqual(context.exception.status_code, 400)

        with self.assertRaises(HTTPException) as context:
            projects_router.add_project_record_evidence(
                project["project_id"],
                original["record_id"],
                AddProjectEvidenceRequest(
                    relation="supports",
                    artifact=self._artifact("model_bundle", BUNDLE_VERSION),
                ),
            )
        self.assertEqual(context.exception.status_code, 400)

        with self.assertRaises(HTTPException) as context:
            projects_router.add_project_record(
                project["project_id"],
                AddProjectRecordRequest(
                    kind="hypothesis",
                    title="Unlinked question",
                    text="This question is not part of this project.",
                    research_question="Does pressure affect degradation?",
                ),
            )
        self.assertEqual(context.exception.status_code, 400)

    def test_legacy_hypotheses_receive_proposed_status_and_can_be_updated(self):
        project = projects_router.create_project(
            CreateProjectRequest(name="Legacy hypothesis")
        )
        recorded = projects_router.add_project_record(
            project["project_id"],
            AddProjectRecordRequest(
                kind="hypothesis",
                title="Legacy hypothesis",
                text="A previously recorded hypothesis.",
            ),
        )
        record = recorded["records"][0]
        path = projects_router._project_path(project["project_id"])
        stored_project = json.loads(path.read_text(encoding="utf-8"))
        stored_record = stored_project["records"][0]
        stored_record.pop("status")
        stored_record.pop("status_history")
        path.write_text(json.dumps(stored_project), encoding="utf-8")

        legacy = projects_router.get_project(project["project_id"])["records"][0]
        self.assertEqual(legacy["status"], "proposed")
        self.assertEqual(
            legacy["status_history"],
            [{"status": "proposed", "changed_at": record["created_at"]}],
        )
        updated = projects_router.update_project_hypothesis_status(
            project["project_id"],
            record["record_id"],
            UpdateProjectHypothesisStatusRequest(status="supported"),
        )
        self.assertEqual(updated["records"][0]["status"], "supported")
        self.assertEqual(
            [entry["status"] for entry in updated["records"][0]["status_history"]],
            ["proposed", "supported"],
        )

    def test_artifact_snapshots_are_preserved_and_report_changed_or_missing_sources(self):
        project = projects_router.create_project(
            CreateProjectRequest(
                name="Snapshot tracking",
                artifacts=[
                    self._artifact("dataset_partition", "train"),
                    self._artifact("model_bundle", BUNDLE_VERSION),
                ],
            )
        )
        original_snapshots = {
            artifact["artifact_id"]: artifact["snapshot"]
            for artifact in project["artifacts"]
        }
        self.assertIn("fingerprint", original_snapshots["train"])
        self.assertEqual(
            original_snapshots[BUNDLE_VERSION]["details"]["run_id"],
            "baseline-run",
        )

        updated = projects_router.update_project(
            project["project_id"],
            UpdateProjectRequest(
                name="Snapshot tracking",
                artifacts=[
                    self._artifact("dataset_partition", "train", "renamed label"),
                    self._artifact("model_bundle", BUNDLE_VERSION),
                ],
            ),
        )
        self.assertEqual(
            [artifact["snapshot"] for artifact in updated["artifacts"]],
            [original_snapshots["train"], original_snapshots[BUNDLE_VERSION]],
        )

        changed_frame = pd.DataFrame(
            {"feature": [0.0, 1.0, 2.0], "target": [10.0, 11.0, 12.0]}
        )
        with open_tracker(TASK) as tracker:
            tracker.RegisterData(changed_frame, "target", partition="train")

        summary = projects_router.project_summary(project["project_id"])
        snapshot_status = {
            artifact["artifact_id"]: artifact["snapshot_status"]
            for artifact in summary["artifacts"]
        }
        self.assertEqual(snapshot_status["train"], "changed")
        self.assertEqual(snapshot_status[BUNDLE_VERSION], "current")

        (settings.task_bundles_dir(TASK) / (BUNDLE_VERSION + ".bundle")).unlink()
        summary = projects_router.project_summary(project["project_id"])
        missing_bundle = next(
            artifact
            for artifact in summary["artifacts"]
            if artifact["artifact_id"] == BUNDLE_VERSION
        )
        self.assertFalse(missing_bundle["exists"])
        self.assertEqual(missing_bundle["snapshot_status"], "missing")

    def test_legacy_records_without_stored_snapshots_report_unverified(self):
        project = projects_router.create_project(
            CreateProjectRequest(
                name="Legacy snapshot records",
                artifacts=[
                    self._artifact("dataset_partition", "train"),
                    self._artifact("model_bundle", BUNDLE_VERSION),
                ],
            )
        )
        path = projects_router._project_path(project["project_id"])
        stored_project = json.loads(path.read_text(encoding="utf-8"))
        for artifact in stored_project["artifacts"]:
            del artifact["snapshot"]
        path.write_text(json.dumps(stored_project), encoding="utf-8")

        summary = projects_router.project_summary(project["project_id"])
        by_artifact = {artifact["artifact_id"]: artifact for artifact in summary["artifacts"]}
        self.assertTrue(by_artifact["train"]["exists"])
        self.assertEqual(by_artifact["train"]["snapshot_status"], "unverified")
        self.assertTrue(by_artifact[BUNDLE_VERSION]["exists"])
        self.assertEqual(by_artifact[BUNDLE_VERSION]["snapshot_status"], "unverified")

    def test_project_edits_never_rebaseline_stored_snapshots(self):
        project = projects_router.create_project(
            CreateProjectRequest(
                name="No rebaseline",
                artifacts=[
                    self._artifact("dataset_partition", "train"),
                    self._artifact("model_bundle", BUNDLE_VERSION),
                ],
            )
        )
        original_snapshots = {
            artifact["artifact_id"]: artifact["snapshot"] for artifact in project["artifacts"]
        }

        changed_frame = pd.DataFrame(
            {"feature": [0.0, 1.0, 2.0], "target": [10.0, 11.0, 12.0]}
        )
        with open_tracker(TASK) as tracker:
            tracker.RegisterData(changed_frame, "target", partition="train")

        updated = projects_router.update_project(
            project["project_id"],
            UpdateProjectRequest(
                name="No rebaseline",
                artifacts=[
                    self._artifact("dataset_partition", "train"),
                    self._artifact("model_bundle", BUNDLE_VERSION),
                ],
            ),
        )
        stored_snapshots = {
            artifact["artifact_id"]: artifact["snapshot"] for artifact in updated["artifacts"]
        }
        self.assertEqual(stored_snapshots, original_snapshots)

        summary = projects_router.project_summary(project["project_id"])
        snapshot_status = {
            artifact["artifact_id"]: artifact["snapshot_status"]
            for artifact in summary["artifacts"]
        }
        self.assertEqual(snapshot_status["train"], "changed")
        self.assertEqual(snapshot_status[BUNDLE_VERSION], "current")

    def test_project_can_reference_existing_dataset_bundle_and_analysis_job(self):
        links = [
            self._artifact("dataset_partition", "train"),
            self._artifact("model_bundle", BUNDLE_VERSION),
            self._artifact("job", JOB_ID),
        ]
        project = projects_router.create_project(
            CreateProjectRequest(name="Linked artifacts", artifacts=links)
        )
        linked_dataset = datasets_router.get_dataset_metadata(TASK)
        linked_run = bundles_router.run_configuration(TASK, BUNDLE_VERSION)
        linked_job = jobs_router.get_job(JOB_ID)

        self.assertEqual(
            [artifact["kind"] for artifact in project["artifacts"]],
            ["dataset_partition", "model_bundle", "job"],
        )
        self.assertEqual(project["artifacts"][2]["artifact_id"], JOB_ID)
        self.assertIn("train", linked_dataset["partitions"])
        self.assertEqual(linked_run["run_id"], "baseline-run")
        self.assertEqual(
            linked_run["run"]["data"]["train_fingerprint"],
            "test-fingerprint",
        )
        self.assertEqual(linked_job["job_id"], JOB_ID)

    def test_add_project_artifact_appends_idempotently_and_preserves_snapshots(self):
        project = projects_router.create_project(
            CreateProjectRequest(
                name="Append artifact",
                artifacts=[self._artifact("dataset_partition", "train")],
            )
        )
        original_snapshot = project["artifacts"][0]["snapshot"]

        updated = projects_router.add_project_artifact(
            project["project_id"],
            self._artifact("job", JOB_ID, "sensitivity"),
        )
        repeated = projects_router.add_project_artifact(
            project["project_id"],
            self._artifact("job", JOB_ID, "sensitivity"),
        )

        self.assertEqual(len(updated["artifacts"]), 2)
        self.assertEqual(updated["artifacts"][0]["snapshot"], original_snapshot)
        self.assertEqual(updated["artifacts"][1]["artifact_id"], JOB_ID)
        self.assertIsNone(updated["artifacts"][1]["snapshot"]["details"]["result_fingerprint"])
        self.assertEqual(repeated, updated)

    def test_add_project_artifact_captures_model_bundle_snapshot(self):
        project = projects_router.create_project(
            CreateProjectRequest(name="Produced bundle")
        )

        updated = projects_router.add_project_artifact(
            project["project_id"],
            self._artifact("model_bundle", BUNDLE_VERSION),
        )

        bundle_reference = updated["artifacts"][0]
        self.assertEqual(bundle_reference["kind"], "model_bundle")
        self.assertEqual(bundle_reference["snapshot"]["details"]["run_id"], "baseline-run")
        self.assertEqual(
            bundle_reference["snapshot"]["identity"]["artifact_id"],
            BUNDLE_VERSION,
        )

    def test_add_project_artifact_captures_registered_partition_fingerprint(self):
        project = projects_router.create_project(
            CreateProjectRequest(name="Registered partition")
        )

        updated = projects_router.add_project_artifact(
            project["project_id"],
            self._artifact("dataset_partition", "train"),
        )

        with open_tracker(TASK) as tracker:
            fingerprint = tracker.dataset_splits()["train"]["fingerprint"]
        artifact = updated["artifacts"][0]
        self.assertEqual(artifact["artifact_id"], "train")
        self.assertEqual(artifact["snapshot"]["details"]["fingerprint"], fingerprint)
        self.assertEqual(
            projects_router.project_summary(project["project_id"])["artifacts"][0][
                "snapshot_status"
            ],
            "current",
        )

    def test_project_modeling_stage_only_counts_completed_modeling_jobs(self):
        for job_id, expected in (
            (JOB_ID, False),
            (EXPERIMENT_JOB_ID, True),
        ):
            with self.subTest(job_id=job_id):
                project = projects_router.create_project(
                    CreateProjectRequest(
                        name="Modeling progress",
                        artifacts=[self._artifact("job", job_id)],
                    )
                )
                summary = projects_router.project_summary(project["project_id"])
                modeling_stage = next(
                    stage for stage in summary["stages"] if stage["stage"] == "model"
                )
                self.assertEqual(modeling_stage["done"], expected)

    def test_project_analysis_stage_counts_completed_analysis_jobs(self):
        project = projects_router.create_project(
            CreateProjectRequest(
                name="Analysis progress",
                artifacts=[self._artifact("job", JOB_ID)],
            )
        )
        summary = projects_router.project_summary(project["project_id"])
        analysis_stage = next(
            stage for stage in summary["stages"] if stage["stage"] == "analysis"
        )
        modeling_stage = next(
            stage for stage in summary["stages"] if stage["stage"] == "model"
        )
        self.assertTrue(analysis_stage["done"])
        self.assertFalse(modeling_stage["done"])

    def test_project_result_stage_requires_a_job_result_or_linked_bundle(self):
        project = projects_router.create_project(
            CreateProjectRequest(
                name="Result availability",
                artifacts=[self._artifact("job", JOB_ID)],
            )
        )
        summary = projects_router.project_summary(project["project_id"])
        result_stage = next(
            stage for stage in summary["stages"] if stage["stage"] == "result"
        )
        self.assertFalse(result_stage["done"])

        updated = projects_router.update_project(
            project["project_id"],
            UpdateProjectRequest(
                name="Result availability",
                artifacts=[
                    self._artifact("job", EXPERIMENT_JOB_ID),
                ],
            ),
        )
        result_stage = next(
            stage
            for stage in projects_router.project_summary(updated["project_id"])["stages"]
            if stage["stage"] == "result"
        )
        self.assertTrue(result_stage["done"])

        updated = projects_router.update_project(
            updated["project_id"],
            UpdateProjectRequest(
                name="Result availability",
                artifacts=[self._artifact("model_bundle", BUNDLE_VERSION)],
            ),
        )
        result_stage = next(
            stage
            for stage in projects_router.project_summary(updated["project_id"])["stages"]
            if stage["stage"] == "result"
        )
        self.assertTrue(result_stage["done"])

    def test_project_summary_tracks_research_progression(self):
        project = projects_router.create_project(
            CreateProjectRequest(
                name="Research progression",
                research_questions=["Does temperature affect degradation?"],
                artifacts=[
                    self._artifact("dataset_partition", "train"),
                    self._artifact("job", JOB_ID),
                    self._artifact("model_bundle", BUNDLE_VERSION),
                ],
            )
        )
        projects_router.add_project_record(
            project["project_id"],
            AddProjectRecordRequest(
                kind="hypothesis",
                title="Temperature response",
                text="The response may be nonlinear.",
            ),
        )
        summary = projects_router.project_summary(project["project_id"])
        stages = {stage["stage"]: stage["done"] for stage in summary["stages"]}
        self.assertEqual(
            list(stages),
            ["question", "hypothesis", "data", "analysis", "model", "result", "conclusion"],
        )
        self.assertEqual(
            stages,
            {
                "question": True,
                "hypothesis": True,
                "data": True,
                "analysis": True,
                "model": False,
                "result": True,
                "conclusion": False,
            },
        )

    def test_artifact_references_must_exist_and_match_the_task(self):
        invalid = (
            self._artifact("dataset_partition", "validation"),
            self._artifact("model_bundle", "missing-version"),
            self._artifact("job", "abcdef0123456789abcdef0123456789"),
        )
        for artifact in invalid:
            with self.subTest(kind=artifact.kind, artifact_id=artifact.artifact_id):
                with self.assertRaises(HTTPException) as context:
                    projects_router.create_project(
                        CreateProjectRequest(name="Invalid link", artifacts=[artifact])
                    )
                self.assertEqual(context.exception.status_code, 400)

        with self.assertRaises(HTTPException) as context:
            projects_router.create_project(
                CreateProjectRequest(
                    name="Wrong task",
                    artifacts=[
                        ProjectArtifactReference(
                            kind="job",
                            task_name="another-task",
                            artifact_id=JOB_ID,
                        )
                    ],
                )
            )
        self.assertEqual(context.exception.status_code, 400)

    def test_duplicate_and_path_like_artifact_references_are_rejected(self):
        artifact = self._artifact("dataset_partition", "train")
        with self.assertRaises(HTTPException) as context:
            projects_router.create_project(
                CreateProjectRequest(name="Duplicate", artifacts=[artifact, artifact])
            )
        self.assertEqual(context.exception.status_code, 400)

        unsafe = ProjectArtifactReference(
            kind="model_bundle",
            task_name="../outside",
            artifact_id="bundle-v1",
        )
        with self.assertRaises(HTTPException) as context:
            projects_router.create_project(
                CreateProjectRequest(name="Unsafe", artifacts=[unsafe])
            )
        self.assertEqual(context.exception.status_code, 400)

    def test_missing_and_malformed_project_ids_return_not_found(self):
        for project_id in ("0" * 32, "../project"):
            with self.subTest(project_id=project_id):
                with self.assertRaises(HTTPException) as context:
                    projects_router.get_project(project_id)
                self.assertEqual(context.exception.status_code, 404)

    def test_project_routes_are_registered_in_the_api(self):
        from api.main import app

        paths = app.openapi()["paths"]
        self.assertIn("/api/projects", paths)
        self.assertIn("/api/projects/{project_id}", paths)
        self.assertIn("/api/projects/{project_id}/artifacts", paths)
        self.assertIn("/api/projects/{project_id}/notes", paths)
        self.assertIn("/api/projects/{project_id}/records", paths)
        self.assertIn(
            "/api/projects/{project_id}/records/{record_id}/evidence",
            paths,
        )
        self.assertIn("/api/projects/{project_id}/records/{record_id}/status", paths)

    def test_project_record_endpoint_roundtrips_multiple_evidence_refs(self):
        from api.main import app

        with TestClient(app) as client:
            project_response = client.post(
                "/api/projects",
                json={
                    "name": "HTTP evidence flow",
                    "artifacts": [
                        {
                            "kind": "dataset_partition",
                            "task_name": TASK,
                            "artifact_id": "train",
                        },
                        {
                            "kind": "model_bundle",
                            "task_name": TASK,
                            "artifact_id": BUNDLE_VERSION,
                        },
                    ],
                },
            )
            self.assertEqual(project_response.status_code, 200, project_response.text)
            project_id = project_response.json()["project_id"]

            artifact_response = client.post(
                "/api/projects/%s/artifacts" % project_id,
                json={
                    "kind": "job",
                    "task_name": TASK,
                    "artifact_id": JOB_ID,
                    "label": "sensitivity",
                },
            )
            self.assertEqual(artifact_response.status_code, 200, artifact_response.text)
            self.assertEqual(
                [item["kind"] for item in artifact_response.json()["artifacts"]],
                ["dataset_partition", "model_bundle", "job"],
            )
            duplicate_response = client.post(
                "/api/projects/%s/artifacts" % project_id,
                json={
                    "kind": "job",
                    "task_name": TASK,
                    "artifact_id": JOB_ID,
                    "label": "sensitivity",
                },
            )
            self.assertEqual(duplicate_response.status_code, 200, duplicate_response.text)
            self.assertEqual(len(duplicate_response.json()["artifacts"]), 3)

            hypothesis_response = client.post(
                "/api/projects/%s/records" % project_id,
                json={
                    "kind": "hypothesis",
                    "title": "Temperature response",
                    "text": "Temperature may have a nonlinear relationship with degradation.",
                },
            )
            self.assertEqual(hypothesis_response.status_code, 200, hypothesis_response.text)
            hypothesis = hypothesis_response.json()["records"][0]
            self.assertEqual(hypothesis["status"], "proposed")
            status_response = client.post(
                "/api/projects/%s/records/%s/status" % (project_id, hypothesis["record_id"]),
                json={"status": "under_investigation"},
            )
            self.assertEqual(status_response.status_code, 200, status_response.text)
            updated_hypothesis = status_response.json()["records"][0]
            self.assertEqual(updated_hypothesis["text"], hypothesis["text"])
            self.assertEqual(updated_hypothesis["status"], "under_investigation")
            self.assertEqual(
                [item["status"] for item in updated_hypothesis["status_history"]],
                ["proposed", "under_investigation"],
            )
            evidence_response = client.post(
                "/api/projects/%s/records/%s/evidence"
                % (project_id, hypothesis["record_id"]),
                json={
                    "relation": "supports",
                    "artifact": {
                        "kind": "dataset_partition",
                        "task_name": TASK,
                        "artifact_id": "train",
                    },
                },
            )
            self.assertEqual(evidence_response.status_code, 200, evidence_response.text)
            self.assertEqual(
                evidence_response.json()["records"][0]["evidence"][0]["artifact"]["artifact_id"],
                "train",
            )

            record_response = client.post(
                "/api/projects/%s/records" % project_id,
                json={
                    "kind": "finding",
                    "title": "Inputs and run are linked",
                    "text": "Both source artifacts are present and identified.",
                    "evidence": [
                        {
                            "relation": "supports",
                            "artifact": {
                                "kind": "dataset_partition",
                                "task_name": TASK,
                                "artifact_id": "train",
                            },
                        },
                        {
                            "relation": "contradicts",
                            "artifact": {
                                "kind": "model_bundle",
                                "task_name": TASK,
                                "artifact_id": BUNDLE_VERSION,
                            },
                        },
                    ],
                },
            )
            self.assertEqual(record_response.status_code, 200, record_response.text)
            record = next(
                item
                for item in record_response.json()["records"]
                if item["kind"] == "finding"
            )
            self.assertEqual(len(record["evidence"]), 2)

            summary_response = client.get("/api/projects/%s/summary" % project_id)
            self.assertEqual(summary_response.status_code, 200, summary_response.text)
            self.assertTrue(
                all(
                    artifact["snapshot_status"] == "current"
                    for artifact in summary_response.json()["artifacts"]
                )
            )

    # ------------------------------------------------- B1 tier-1: references

    def _reference(self, url="https://example.org/paper", **kwargs):
        return ProjectArtifactReference(kind="reference", url=url, **kwargs)

    def test_reference_artifact_roundtrip_snapshot_and_summary(self):
        project = projects_router.create_project(
            CreateProjectRequest(
                name="Literature-backed project",
                artifacts=[
                    self._reference(
                        title="A measured paper",
                        doi="10.1000/xyz",
                        label="literature",
                    )
                ],
            )
        )
        artifact = project["artifacts"][0]
        self.assertEqual(artifact["kind"], "reference")
        self.assertEqual(artifact["url"], "https://example.org/paper")
        self.assertNotIn("task_name", artifact)
        self.assertNotIn("artifact_id", artifact)
        snapshot = artifact["snapshot"]
        self.assertEqual(snapshot["schema"], "sksurrogate.project-artifact-snapshot.v2")
        self.assertEqual(
            snapshot["identity"],
            {"kind": "reference", "url": "https://example.org/paper"},
        )
        self.assertFalse(snapshot["details"]["content_embedded"])
        self.assertEqual(snapshot["details"]["fields"]["doi"], "10.1000/xyz")

        summary = projects_router.project_summary(project["project_id"])
        linked = summary["artifacts"][0]
        self.assertTrue(linked["exists"])
        self.assertEqual(linked["snapshot_status"], "unverified")
        stages = {stage["stage"]: stage["done"] for stage in summary["stages"]}
        self.assertFalse(stages["data"] or stages["model"] or stages["result"])

        # Idempotent for the same URL, additive for a different one.
        same = projects_router.add_project_artifact(
            project["project_id"], self._reference()
        )
        self.assertEqual(len(same["artifacts"]), 1)
        other = projects_router.add_project_artifact(
            project["project_id"], self._reference(url="https://example.org/other")
        )
        self.assertEqual(len(other["artifacts"]), 2)

    def test_reference_validation_rejects_non_urls_and_mixed_shapes(self):
        for bad in (
            "ftp://example.org/x",
            "example.org/x",
            "https:///x",
            "http://",
            "",
            "https://example.org/" + "a" * 2048,
        ):
            with self.assertRaises(ValidationError):
                ProjectArtifactReference(kind="reference", url=bad)
        with self.assertRaises(ValidationError):
            ProjectArtifactReference(
                kind="reference", url="https://example.org/x", task_name=TASK
            )
        with self.assertRaises(ValidationError):
            ProjectArtifactReference(
                kind="dataset_partition",
                task_name=TASK,
                artifact_id="train",
                url="https://example.org/x",
            )
        with self.assertRaises(ValidationError):
            ProjectArtifactReference(kind="job", task_name=TASK)
        with self.assertRaises(HTTPException):
            projects_router.create_project(
                CreateProjectRequest(
                    name="Duplicate references",
                    artifacts=[self._reference(), self._reference(label="again")],
                )
            )

    def test_reference_can_back_evidence_and_cannot_be_removed(self):
        project = projects_router.create_project(
            CreateProjectRequest(
                name="Cited finding",
                artifacts=[self._reference(doi="10.1000/xyz")],
            )
        )
        original_snapshot = project["artifacts"][0]["snapshot"]
        updated = projects_router.add_project_record(
            project["project_id"],
            AddProjectRecordRequest(
                kind="finding",
                title="Corroborated result",
                text="The independent study agrees [S1].",
                evidence=[
                    ProjectEvidenceReference(
                        relation="supports",
                        artifact=self._reference(doi="10.1000/xyz"),
                    )
                ],
            ),
        )
        record = updated["records"][0]
        self.assertEqual(record["evidence"][0]["artifact"]["kind"], "reference")
        self.assertEqual(record["evidence"][0]["artifact"]["url"], "https://example.org/paper")

        # Re-PUT keeps the stored snapshot and allows label edits.
        renamed = projects_router.update_project(
            project["project_id"],
            UpdateProjectRequest(
                name="Cited finding",
                artifacts=[self._reference(doi="10.1000/xyz", label="renamed")],
            ),
        )
        self.assertEqual(renamed["artifacts"][0]["snapshot"], original_snapshot)
        self.assertEqual(renamed["artifacts"][0]["label"], "renamed")

        # A reference used as evidence cannot be removed via PUT.
        with self.assertRaises(HTTPException):
            projects_router.update_project(
                project["project_id"],
                UpdateProjectRequest(name="Cited finding", artifacts=[]),
            )

    def test_artifact_shapes_stay_backward_compatible(self):
        project = projects_router.create_project(
            CreateProjectRequest(
                name="Legacy shape",
                artifacts=[self._artifact("dataset_partition", "train")],
            )
        )
        artifact = project["artifacts"][0]
        self.assertEqual(artifact["kind"], "dataset_partition")
        self.assertNotIn("url", artifact)
        self.assertNotIn("doi", artifact)
        self.assertNotIn("title", artifact)
        self.assertEqual(
            artifact["snapshot"]["schema"],
            "sksurrogate.project-artifact-snapshot.v1",
        )

    def test_reference_http_surface(self):
        from api.main import app

        with TestClient(app) as client:
            created = client.post(
                "/api/projects",
                json={
                    "name": "HTTP references",
                    "artifacts": [
                        {
                            "kind": "reference",
                            "url": "https://example.org/paper",
                            "title": "Paper",
                        }
                    ],
                },
            )
            self.assertEqual(created.status_code, 200, created.text)
            project = created.json()
            self.assertEqual(project["artifacts"][0]["kind"], "reference")

            bad = client.post(
                "/api/projects",
                json={
                    "name": "Bad scheme",
                    "artifacts": [{"kind": "reference", "url": "ftp://example.org/x"}],
                },
            )
            self.assertEqual(bad.status_code, 422)

            appended = client.post(
                "/api/projects/%s/artifacts" % project["project_id"],
                json={"kind": "reference", "url": "https://example.org/second"},
            )
            self.assertEqual(appended.status_code, 200, appended.text)
            self.assertEqual(len(appended.json()["artifacts"]), 2)

            summary = client.get(
                "/api/projects/%s/summary" % project["project_id"]
            ).json()
            self.assertEqual(
                [item["snapshot_status"] for item in summary["artifacts"]],
                ["unverified", "unverified"],
            )

if __name__ == "__main__":
    unittest.main()
