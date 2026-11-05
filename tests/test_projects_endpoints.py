import tempfile
import unittest
from pathlib import Path

import pandas as pd
from fastapi import HTTPException
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


if __name__ == "__main__":
    unittest.main()
