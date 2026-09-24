"""Tests for the Phase 4 experiment enhancements (docs/ui-gap-implementation-plan.md).

Covers the new search plumbing behind ``POST /api/experiments/{task}/run`` and the
new single-structure endpoint:

- 4.1 surrogate-assisted search (``surrogate_mode`` / ``surrogate_itrs``)
- 4.2 conditional parameters (``depends_on``) and forbidden rules (``forbidden``)
- 4.3 Pareto frontier of the search (``pareto`` in the job result)
- 4.4 top pipelines (``top_pipelines``) and ``POST /{task}/optimize-pipeline``
- 4.5 the ``stacking`` directive (``res``/``probs``/``decision``/``cv``/``n_jobs``)

The router functions are called directly against a temporary storage root, with
the small Galaxy3 dataset under ``data/`` registered as ``train`` / ``validation``
partitions — matching the phase 1-3 test pattern.
"""

import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

from api.config import settings
from api.deps import open_tracker
from api.jobs import job_manager
from api.routers import experiments as experiments_router
from api.routers.experiments import (
    OptimizePipelineRequest,
    ParamSpec,
    RunExperimentRequest,
)
from api.training import build_conditional_map, build_forbidden_rules, extract_stacking

#: Small binary-classification dataset shipped with the repo (1600 rows, 21 columns).
GALAXY3_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "Galaxy3-[GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz].tabular"
)

TASK = "phase4-endpoints"
TARGET = "target"

#: How long to wait for a background job before failing the test.
JOB_TIMEOUT_SECONDS = 300.0

#: A deliberately small search space: one structure of length 1, one generation.
TINY_CONFIG = {
    "sklearn.linear_model.LogisticRegression": {
        "C": {"type": "real", "low": 0.1, "high": 1.0},
    },
}

TWO_STEP_CONFIG = {
    "sklearn.tree.DecisionTreeClassifier": {
        "max_depth": {"type": "integer", "low": 1, "high": 3},
    },
    "sklearn.linear_model.LogisticRegression": {
        "C": {"type": "real", "low": 0.1, "high": 1.0},
    },
}


def _wait_for_job(job_id, timeout=JOB_TIMEOUT_SECONDS):
    """Poll the persisted job record until it reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = job_manager.get(job_id)
        if record is not None and record["status"] in {"completed", "failed"}:
            return record
        time.sleep(0.25)
    raise AssertionError("job %s did not finish within %.0fs" % (job_id, timeout))


class TestPhase4PureHelpers(unittest.TestCase):
    """The config→AML conversions behind 4.2 and 4.5, without touching storage."""

    def test_build_search_param_strips_depends_on(self):
        from SKSurrogate import Real

        from api.training import build_search_param

        spec = {"type": "real", "low": 0.1, "high": 1.0, "depends_on": {"penalty": ["elasticnet"]}}
        self.assertIsInstance(build_search_param(spec), Real)

    def test_build_search_param_rejects_non_object_spec(self):
        from api.training import build_search_param

        with self.assertRaises(HTTPException) as ctx:
            build_search_param(0.5)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_build_conditional_map_collects_depends_on_entries(self):
        config = {
            "sklearn.linear_model.LogisticRegression": {
                "penalty": {"type": "categorical", "items": ["l1", "l2"]},
                "l1_ratio": {
                    "type": "real",
                    "low": 0.0,
                    "high": 1.0,
                    "depends_on": {"penalty": ["elasticnet"]},
                },
            }
        }
        self.assertEqual(build_conditional_map(config), {"l1_ratio": {"penalty": ["elasticnet"]}})

    def test_build_forbidden_rules_converts_pairs(self):
        self.assertEqual(build_forbidden_rules([["penalty", "l1"]]), ({"penalty": "l1"},))
        # Values are compared against the parameter's actual value, so numbers work too.
        self.assertEqual(build_forbidden_rules([["max_depth", 1]]), ({"max_depth": 1},))
        self.assertEqual(build_forbidden_rules(None), ())

    def test_build_forbidden_rules_rejects_malformed_pairs(self):
        with self.assertRaises(HTTPException) as ctx:
            build_forbidden_rules([["penalty"]])
        self.assertEqual(ctx.exception.status_code, 400)

        with self.assertRaises(HTTPException) as ctx:
            build_forbidden_rules([["penalty", {"nested": "mapping"}]])
        self.assertEqual(ctx.exception.status_code, 400)

    def test_extract_stacking_maps_fields_and_drops_directive(self):
        config = {
            "sklearn.linear_model.LogisticRegression": {"C": {"type": "real", "low": 0.1, "high": 1.0}},
            "stacking": {"res": False, "cv": 3},
        }
        remaining, kwargs = extract_stacking(config)
        self.assertNotIn("stacking", remaining)
        self.assertEqual(kwargs, {"stack_res": False, "stack_cv": 3})

    def test_extract_stacking_validates_values(self):
        with self.assertRaises(HTTPException) as ctx:
            extract_stacking({"stacking": {"cv": 1}})
        self.assertEqual(ctx.exception.status_code, 400)

        with self.assertRaises(HTTPException) as ctx:
            extract_stacking({"stacking": {"res": "yes"}})
        self.assertEqual(ctx.exception.status_code, 400)

    def test_extract_stacking_without_directive_is_a_noop(self):
        config = {"sklearn.linear_model.LogisticRegression": {}}
        remaining, kwargs = extract_stacking(config)
        self.assertEqual(remaining, config)
        self.assertEqual(kwargs, {})


class TestPhase4Endpoints(unittest.TestCase):
    """Happy paths and error paths for the Phase 4 experiment endpoints."""

    @classmethod
    def setUpClass(cls):
        cls._original_root = settings.root
        cls._frame = pd.read_csv(GALAXY3_PATH, sep="\t")

    @classmethod
    def tearDownClass(cls):
        settings.root = cls._original_root

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        settings.root = Path(self._tmp.name)
        settings.ensure()

        # The dataset is ordered by class, so take a fixed number of rows per
        # class for both partitions instead of a contiguous slice.
        per_class = 150
        grouped = self._frame.groupby(TARGET, group_keys=False)
        self._train = grouped.head(per_class).sort_index().reset_index(drop=True)
        self._validation = grouped.tail(50).sort_index().reset_index(drop=True)
        self.assertEqual(len(self._train[TARGET].unique()), 2)
        self.assertEqual(len(self._validation[TARGET].unique()), 2)
        dataset_dir = settings.task_dataset_dir(TASK)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        self._train.to_csv(dataset_dir / "train.csv", index=False)
        self._validation.to_csv(dataset_dir / "validation.csv", index=False)
        with open_tracker(TASK) as tracker:
            tracker.RegisterData(self._train, TARGET, partition="train")
            tracker.RegisterData(self._validation, TARGET, partition="validation")

    def tearDown(self):
        self._tmp.cleanup()

    # ------------------------------------------------------------ 4.1-4.3 search

    def test_run_experiment_returns_top_pipelines_and_pareto(self):
        body = RunExperimentRequest(
            config={name: {p: ParamSpec(**spec) for p, spec in params.items()} for name, params in TINY_CONFIG.items()},
            length=1,
            max_generation=1,
            num_parents=1,
        )
        submitted = experiments_router.run_experiment(TASK, body)
        self.assertEqual(submitted["status"], "queued")

        record = _wait_for_job(submitted["job_id"])
        self.assertEqual(record["status"], "completed", record["error"])
        result = record["result"]

        self.assertIn("model_version", result)
        self.assertTrue(Path(result["bundle_path"]).exists())
        self.assertTrue(result["top_pipelines"])
        self.assertEqual(set(result["top_pipelines"][0]), {"pipeline", "score"})
        self.assertEqual(result["top_pipelines"][0]["pipeline"], ["sklearn.linear_model.LogisticRegression"])
        # Scores are reported in the scorer's own units and are finite for valid candidates.
        self.assertGreaterEqual(result["top_pipelines"][0]["score"], 0.0)
        self.assertLessEqual(result["top_pipelines"][0]["score"], 1.0)
        self.assertIn("pareto", result)
        for point in result["pareto"] or []:
            self.assertEqual(set(point), {"pipeline", "score", "duration"})

    def test_run_experiment_surrogate_mode_records_audit_event(self):
        body = RunExperimentRequest(
            config={name: {p: ParamSpec(**spec) for p, spec in params.items()} for name, params in TINY_CONFIG.items()},
            length=1,
            max_generation=1,
            num_parents=1,
            surrogate_mode=True,
            surrogate_itrs=5,
        )
        submitted = experiments_router.run_experiment(TASK, body)
        record = _wait_for_job(submitted["job_id"])
        self.assertEqual(record["status"], "completed", record["error"])

        from SKSurrogate import load_bundle

        bundle = load_bundle(Path(record["result"]["bundle_path"]))
        events = [event for event in bundle.audit_events if event["event_type"] == "experiment"]
        self.assertTrue(events)
        self.assertTrue(events[-1]["details"]["surrogate_mode"])
        self.assertEqual(events[-1]["details"]["surrogate_itrs"], 5)

    def test_run_experiment_excludes_forbidden_candidates(self):
        config = {
            "sklearn.linear_model.LogisticRegression": {
                "penalty": {"type": "categorical", "items": ["l1"]},
            },
            "sklearn.tree.DecisionTreeClassifier": {
                "max_depth": {"type": "integer", "low": 1, "high": 3},
            },
        }
        body = RunExperimentRequest(
            config={name: {p: ParamSpec(**spec) for p, spec in params.items()} for name, params in config.items()},
            length=1,
            max_generation=1,
            num_parents=1,
            forbidden=[["penalty", "l1"]],
        )
        submitted = experiments_router.run_experiment(TASK, body)
        record = _wait_for_job(submitted["job_id"])
        self.assertEqual(record["status"], "completed", record["error"])

        # Every LogisticRegression candidate is banned, so only the tree survives.
        best = record["result"]["top_pipelines"][0]
        self.assertEqual(best["pipeline"], ["sklearn.tree.DecisionTreeClassifier"])
        self.assertIsNotNone(best["score"])

        from SKSurrogate import load_bundle

        bundle = load_bundle(Path(record["result"]["bundle_path"]))
        event = [e for e in bundle.audit_events if e["event_type"] == "experiment"][-1]
        self.assertEqual(event["details"]["forbidden"], [["penalty", "l1"]])

    def test_run_experiment_records_conditional_parameters(self):
        config = {
            "sklearn.linear_model.LogisticRegression": {
                "penalty": {"type": "categorical", "items": ["l2"]},
                "C": {"type": "real", "low": 0.1, "high": 1.0, "depends_on": {"penalty": ["elasticnet"]}},
            }
        }
        body = RunExperimentRequest(
            config={name: {p: ParamSpec(**spec) for p, spec in params.items()} for name, params in config.items()},
            length=1,
            max_generation=1,
            num_parents=1,
        )
        submitted = experiments_router.run_experiment(TASK, body)
        record = _wait_for_job(submitted["job_id"])
        self.assertEqual(record["status"], "completed", record["error"])

        from SKSurrogate import load_bundle

        bundle = load_bundle(Path(record["result"]["bundle_path"]))
        event = [e for e in bundle.audit_events if e["event_type"] == "experiment"][-1]
        self.assertEqual(event["details"]["conditional"], {"C": {"penalty": ["elasticnet"]}})

    def test_run_experiment_accepts_non_string_forbidden_values(self):
        body = RunExperimentRequest(
            config={
                "sklearn.tree.DecisionTreeClassifier": {
                    "max_depth": ParamSpec(type="integer", low=1, high=3),
                }
            },
            length=1,
            max_generation=1,
            num_parents=1,
            forbidden=[["max_depth", 1]],
        )
        submitted = experiments_router.run_experiment(TASK, body)
        record = _wait_for_job(submitted["job_id"])
        self.assertEqual(record["status"], "completed", record["error"])

        from SKSurrogate import load_bundle

        bundle = load_bundle(Path(record["result"]["bundle_path"]))
        event = [e for e in bundle.audit_events if e["event_type"] == "experiment"][-1]
        self.assertEqual(event["details"]["forbidden"], [["max_depth", 1]])

    def test_run_experiment_rejects_malformed_forbidden_rules(self):
        body = RunExperimentRequest(
            config={name: {p: ParamSpec(**spec) for p, spec in params.items()} for name, params in TINY_CONFIG.items()},
            length=1,
            max_generation=1,
            num_parents=1,
            forbidden=[["penalty"]],
        )
        submitted = experiments_router.run_experiment(TASK, body)
        record = _wait_for_job(submitted["job_id"])
        self.assertEqual(record["status"], "failed")
        self.assertIn("forbidden", record["error"])

    def test_run_experiment_rejects_non_positive_surrogate_itrs(self):
        body = RunExperimentRequest(
            config={name: {p: ParamSpec(**spec) for p, spec in params.items()} for name, params in TINY_CONFIG.items()},
            surrogate_mode=True,
            surrogate_itrs=0,
        )
        with self.assertRaises(HTTPException) as ctx:
            experiments_router.run_experiment(TASK, body)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_run_experiment_honours_stacking_directive(self):
        config = {
            "sklearn.linear_model.LogisticRegression": {"C": ParamSpec(type="real", low=0.1, high=1.0)},
            "stacking": {"res": False, "probs": True, "decision": False, "cv": 2, "n_jobs": 1},
        }
        body = RunExperimentRequest(config=config, length=1, max_generation=1, num_parents=1)
        submitted = experiments_router.run_experiment(TASK, body)
        record = _wait_for_job(submitted["job_id"])
        self.assertEqual(record["status"], "completed", record["error"])

        from SKSurrogate import load_bundle

        bundle = load_bundle(Path(record["result"]["bundle_path"]))
        event = [e for e in bundle.audit_events if e["event_type"] == "experiment"][-1]
        self.assertEqual(
            event["details"]["stacking"],
            {"stack_res": False, "stack_probs": True, "stack_decision": False, "stack_cv": 2, "stack_n_jobs": 1},
        )

    # -------------------------------------------------------- 4.4 optimize-pipeline

    def test_optimize_pipeline_saves_a_bundle_for_the_given_structure(self):
        body = OptimizePipelineRequest(
            seq=["sklearn.linear_model.LogisticRegression"],
            config={"sklearn.linear_model.LogisticRegression": {"C": ParamSpec(type="real", low=0.1, high=1.0)}},
        )
        submitted = experiments_router.optimize_pipeline(TASK, body)
        self.assertEqual(submitted["status"], "queued")

        record = _wait_for_job(submitted["job_id"])
        self.assertEqual(record["status"], "completed", record["error"])
        result = record["result"]
        self.assertEqual(result["pipeline"], ["sklearn.linear_model.LogisticRegression"])
        self.assertTrue(Path(result["bundle_path"]).exists())
        self.assertIsInstance(result["train_score"], float)

        from SKSurrogate import load_bundle

        bundle = load_bundle(Path(result["bundle_path"]))
        event = [e for e in bundle.audit_events if e["event_type"] == "pipeline_optimization"][-1]
        self.assertEqual(event["details"]["pipeline"], ["sklearn.linear_model.LogisticRegression"])

        # Sanity check the bundle is servable: it scores on the validation partition.
        validation = pd.read_csv(settings.task_dataset_dir(TASK) / "validation.csv")
        X = validation.drop(columns=[TARGET]).values
        y = validation[TARGET].values
        self.assertGreaterEqual(float(bundle.model.score(X, y)), 0.0)

    def test_optimize_pipeline_rejects_empty_sequence(self):
        with self.assertRaises(HTTPException) as ctx:
            experiments_router.optimize_pipeline(TASK, OptimizePipelineRequest(seq=[]))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_optimize_pipeline_two_step_structure_stacks_the_intermediate_estimator(self):
        body = OptimizePipelineRequest(
            seq=["sklearn.tree.DecisionTreeClassifier", "sklearn.linear_model.LogisticRegression"],
            config={
                name: {p: ParamSpec(**spec) for p, spec in params.items()}
                for name, params in TWO_STEP_CONFIG.items()
            },
        )
        submitted = experiments_router.optimize_pipeline(TASK, body)
        record = _wait_for_job(submitted["job_id"])
        self.assertEqual(record["status"], "completed", record["error"])

        from SKSurrogate import load_bundle

        bundle = load_bundle(Path(record["result"]["bundle_path"]))
        step_names = [name for name, _ in bundle.model.steps]
        self.assertEqual(step_names, ["stp_0", "stp_1"])
        from SKSurrogate import StackingEstimator

        self.assertIsInstance(bundle.model.steps[0][1], StackingEstimator)


if __name__ == "__main__":
    unittest.main()
