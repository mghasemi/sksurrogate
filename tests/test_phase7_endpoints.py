"""Tests for persisted monitoring alerts and their Dashboard endpoint."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from api.config import settings
from api.deps import append_monitor_alert, append_monitor_record
from api.routers import monitoring as monitoring_router
from api.routers.monitoring import DriftRequest, PredictionDriftRequest, PredictionSource


class TestPhase7MonitoringAlerts(unittest.TestCase):
    def setUp(self):
        self._original_root = settings.root
        self._tmp = tempfile.TemporaryDirectory()
        settings.root = Path(self._tmp.name)
        settings.ensure()

    def tearDown(self):
        settings.root = self._original_root
        self._tmp.cleanup()

    def test_drift_check_persists_dashboard_alerts(self):
        task = "alerts-task"
        dataset_dir = settings.task_dataset_dir(task)
        dataset_dir.mkdir(parents=True)
        pd.DataFrame({"feature": [1, 2]}).to_csv(dataset_dir / "reference.csv", index=False)
        pd.DataFrame({"feature": [8, 9]}).to_csv(dataset_dir / "current.csv", index=False)
        report = {
            "model_version": "model-v1",
            "alerts": [{"type": "numeric_drift", "column": "feature", "value": 0.8}],
        }

        with patch.object(monitoring_router, "drift_report", return_value=report):
            returned = monitoring_router.check_drift(
                task,
                DriftRequest(
                    reference_partition="reference",
                    current_partition="current",
                    model_version="model-v1",
                    psi_threshold=0.2,
                ),
            )

        self.assertIs(returned, report)
        alerts = monitoring_router.monitoring_alerts(task=task, limit=20)["alerts"]
        self.assertEqual(len(alerts), 1)
        self.assertEqual(
            {key: alerts[0][key] for key in ("task", "version", "type", "severity", "detail")},
            {
                "task": task,
                "version": "model-v1",
                "type": "numeric_drift",
                "severity": "warning",
                "detail": "feature, value=0.8, threshold=0.2",
            },
        )
        self.assertTrue(alerts[0]["timestamp"])

    def test_feed_aggregates_failures_across_tasks_and_applies_limit(self):
        append_monitor_alert("task-a", "model-a", "numeric_drift", "feature, value=0.9")
        append_monitor_record("task-b", "model-b", latency_ms=0, rows=3, success=False)
        append_monitor_alert("task-c", "model-c", "range_drift", "feature, value=0.5")

        all_alerts = monitoring_router.monitoring_alerts(limit=2)["alerts"]
        self.assertEqual(len(all_alerts), 2)
        self.assertEqual(
            {
                alert["type"]
                for alert in monitoring_router.monitoring_alerts(task="task-b", limit=20)["alerts"]
            },
            {"inference_error"},
        )
        self.assertEqual(monitoring_router.monitoring_alerts(task="missing", limit=20)["alerts"], [])
        for alert in all_alerts:
            self.assertEqual(
                set(alert),
                {"timestamp", "task", "version", "type", "severity", "detail"},
            )

    def test_prediction_drift_check_persists_its_alerts(self):
        report = {
            "model_version": "model-v2",
            "alerts": [{"type": "numeric_drift", "column": "prediction", "value": 0.6}],
        }
        body = PredictionDriftRequest(
            reference_source=PredictionSource(values=[0, 1, 0, 1]),
            current_source=PredictionSource(values=[1, 2, 1, 2]),
        )
        with patch.object(monitoring_router, "prediction_distribution_report", return_value=report):
            monitoring_router.check_prediction_drift("prediction-task", "model-v2", body)
        alerts = monitoring_router.monitoring_alerts(task="prediction-task", limit=20)["alerts"]
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["version"], "model-v2")
        self.assertEqual(alerts[0]["detail"], "prediction, value=0.6, threshold=0.2")

    def test_alert_route_is_registered(self):
        from api.main import app

        operation = app.openapi()["paths"]["/api/monitoring/alerts"]["get"]
        self.assertEqual(operation["operationId"], "monitoring_alerts_api_monitoring_alerts_get")


if __name__ == "__main__":
    unittest.main()
