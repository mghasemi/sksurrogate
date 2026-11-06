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
    AddProjectRecordRequest,
    AddProjectNoteRequest,
    CreateProjectRequest,
    ProjectArtifactReference,
    UpdateProjectRequest,
)


TASK = "research-project-tests"
BUNDLE_VERSION = "bundle-v1"
JOB_ID = "0123456789abcdef0123456789abcdef"


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

    def test_artifact_references_must_exist_and_match_the_task(self):
        invalid = (
            self._artifact("dataset_partition", "validation"),
            self._artifact("model_bundle", "missing-version"),
            self._artifact("job", "fedcba9876543210fedcba9876543210"),
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
        self.assertIn("/api/projects/{project_id}/notes", paths)
        self.assertIn("/api/projects/{project_id}/records", paths)

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
            record = record_response.json()["records"][0]
            self.assertEqual(len(record["evidence"]), 2)

            summary_response = client.get("/api/projects/%s/summary" % project_id)
            self.assertEqual(summary_response.status_code, 200, summary_response.text)
            self.assertTrue(
                all(
                    artifact["snapshot_status"] == "current"
                    for artifact in summary_response.json()["artifacts"]
                )
            )


if __name__ == "__main__":
    unittest.main()
