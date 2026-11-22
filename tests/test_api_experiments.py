"""API tests for the quick-start experiment options (preset / strategy)."""

import time

import numpy as np
import pandas as pd
import importlib.util

import pytest

if importlib.util.find_spec("httpx") is None:
    # Some hosts ship the httpx2 distribution instead (module `httpx2`);
    # TestClient works with either, so only skip when neither is present.
    pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    return TestClient(app)


def _register(client, task):
    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 3))
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    frame["y"] = (X[:, 0] + 0.3 * rng.normal(size=60) > 0).astype(int)
    response = client.post(
        "/api/datasets/%s/register" % task,
        data={"target": "y", "partition": "train"},
        files={"file": ("t.csv", frame.to_csv(index=False), "text/csv")},
    )
    assert response.status_code == 200, response.text


def _wait(client, job_id):
    for _ in range(240):
        record = client.get("/api/jobs/" + job_id).json()
        if record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def test_presets_listed(client):
    values = [p["value"] for p in client.get("/api/experiments/presets").json()["presets"]]
    assert values == ["classification", "regression"]


def test_preset_default_configs_track_the_library(client):
    """The Advanced-tab pre-fill served per preset stays aligned with the library.

    Regression mirrors ``SKSurrogate.aml._preset_config`` via the shared
    ``PRESET_PARAM_SPECS`` table; classification keeps the historical editor default.
    """
    from SKSurrogate.aml import PRESET_ESTIMATOR_NAMES, PRESET_PARAM_SPECS

    presets = {p["value"]: p for p in client.get("/api/experiments/presets").json()["presets"]}

    regression_space = presets["regression"]["default_config"]
    assert set(regression_space) == {
        "sklearn.ensemble.%sRegressor" % name for name in PRESET_ESTIMATOR_NAMES
    }
    for row in regression_space.values():
        assert row == PRESET_PARAM_SPECS

    classification_space = presets["classification"]["default_config"]
    assert set(classification_space) == {
        "sklearn.linear_model.LogisticRegression",
        "sklearn.tree.DecisionTreeClassifier",
        "sklearn.ensemble.RandomForestClassifier",
    }


def test_config_and_preset_are_exclusive(client):
    body = {"preset": "classification", "config": {"sklearn.tree.DecisionTreeClassifier": {}}}
    assert client.post("/api/experiments/t/run", json=body).status_code == 422
    assert client.post("/api/experiments/t/run", json={}).status_code == 422


def test_preset_run_produces_bundle(client):
    _register(client, "quick")
    body = {"preset": "classification", "strategy": "exhaustive", "length": 1, "random_state": 0}
    job = client.post("/api/experiments/quick/run", json=body).json()["job_id"]
    record = _wait(client, job)
    assert record["status"] == "completed", record
    assert record["result"]["model_version"]


def test_legacy_config_eoa_still_works(client):
    _register(client, "legacy")
    body = {
        "config": {
            "sklearn.tree.DecisionTreeClassifier": {"max_depth": {"type": "integer", "low": 2, "high": 5}},
            "sklearn.linear_model.LogisticRegression": {"C": {"type": "real", "low": 0.1, "high": 5}},
        },
        "length": 1,
        "max_generations": 1,
        "num_parents": 2,
        "random_state": 0,
    }
    job = client.post("/api/experiments/legacy/run", json=body).json()["job_id"]
    record = _wait(client, job)
    assert record["status"] == "completed", record


def test_spa_fallback(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<div id='root'></div>")
    monkeypatch.setenv("SKSURROGATE_WEBUI_DIST", str(dist))
    import importlib

    import api.main as main

    reloaded = importlib.reload(main)
    try:
        c = TestClient(reloaded.app)
        assert "root" in c.get("/experiments").text
        assert c.get("/api/health").json()["status"] == "ok"
        assert c.get("/api/health").json()["version"]
        assert c.get("/api/missing").status_code == 404
    finally:
        monkeypatch.delenv("SKSURROGATE_WEBUI_DIST")
        importlib.reload(main)


def _register_regression(client, task):
    rng = np.random.RandomState(1)
    X = rng.normal(size=(120, 3))
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    frame["y"] = 3 * X[:, 0] + 0.2 * rng.normal(size=120)
    client.post(
        "/api/datasets/%s/register" % task,
        data={"target": "y", "partition": "train"},
        files={"file": ("t.csv", frame.to_csv(index=False), "text/csv")},
    )


def _quick_bundle(client, task, preset):
    body = {"preset": preset, "strategy": "exhaustive", "length": 1, "random_state": 0}
    record = _wait(client, client.post("/api/experiments/%s/run" % task, json=body).json()["job_id"])
    assert record["status"] == "completed", record
    return record["result"]["model_version"], record["job_id"]


def test_explain_intervals_and_project_summary(client):
    _register_regression(client, "reg")
    version, job_id = _quick_bundle(client, "reg", "regression")
    explained = client.post("/api/evaluation/reg/%s/explain" % version, json={"max_rows": 100, "n_repeats": 2})
    assert explained.status_code == 200, explained.text
    data = explained.json()
    assert data["importance"][0]["feature"] == "a"
    assert data["partial_dependence"] and data["partial_dependence"][0]["grid"]
    assert data["reference_sample"] == {
        "partition": "train",
        "source_rows": 120,
        "used_rows": 100,
        "sampling": "random_without_replacement",
    }
    assert data["settings"] == {
        "max_rows": 100,
        "n_repeats": 2,
        "top_k": 3,
        "stability_samples": 1,
        "random_state": 0,
    }
    assert data["reference_sample_stability"] is None
    assert data["interpretation"]["importance"]["category"] == "model_attribution"
    assert data["interpretation"]["partial_dependence"]["category"] == "fitted_model_response_profile"
    assert data["interpretation"]["importance"]["causal_effect"] is False
    assert data["interpretation"]["partial_dependence"]["causal_effect"] is False
    assert data["interpretation"]["limitations"]
    repeated_explanation = client.post(
        "/api/evaluation/reg/%s/explain" % version,
        json={"max_rows": 100, "n_repeats": 2},
    ).json()
    assert repeated_explanation["importance"] == data["importance"]
    assert repeated_explanation["partial_dependence"] == data["partial_dependence"]
    assert repeated_explanation["reference_sample"] == data["reference_sample"]
    alternative_reference = client.post(
        "/api/evaluation/reg/%s/explain" % version,
        json={"max_rows": 100, "n_repeats": 2, "random_state": 1},
    ).json()
    assert alternative_reference["reference_sample"] == data["reference_sample"]
    assert alternative_reference["settings"]["random_state"] == 1
    assert (
        alternative_reference["importance"] != data["importance"]
        or alternative_reference["partial_dependence"] != data["partial_dependence"]
    )
    stability_request = {
        "max_rows": 100,
        "n_repeats": 2,
        "stability_samples": 3,
        "random_state": 0,
    }
    stability_response = client.post(
        "/api/evaluation/reg/%s/explain" % version,
        json=stability_request,
    )
    assert stability_response.status_code == 200, stability_response.text
    stability_result = stability_response.json()["reference_sample_stability"]
    assert stability_result["sample_count"] == 3
    assert stability_result["sample_size"] == 100
    assert stability_result["permutation_repeats_per_sample"] == 2
    assert stability_result["sampling"] == "random_without_replacement"
    assert {item["feature"] for item in stability_result["summary"]} == {"a", "b", "c"}
    assert all(item["sample_std"] >= 0.0 for item in stability_result["summary"])
    assert np.isclose(
        sum(item["top_rank_frequency"] for item in stability_result["summary"]),
        1.0,
    )
    repeated_stability = client.post(
        "/api/evaluation/reg/%s/explain" % version,
        json=stability_request,
    ).json()["reference_sample_stability"]
    assert repeated_stability == stability_result
    all_rows_explanation = client.post(
        "/api/evaluation/reg/%s/explain" % version,
        json={"max_rows": 500, "n_repeats": 2},
    ).json()
    assert all_rows_explanation["reference_sample"] == {
        "partition": "train",
        "source_rows": 120,
        "used_rows": 120,
        "sampling": "all_rows",
    }
    insufficient_reference_rows = client.post(
        "/api/evaluation/reg/%s/explain" % version,
        json={"max_rows": 500, "n_repeats": 2, "stability_samples": 2},
    )
    assert insufficient_reference_rows.status_code == 400
    assert "smaller max_rows" in insufficient_reference_rows.json()["detail"]
    assert any(
        "not estimated by default" in limitation
        for limitation in data["interpretation"]["limitations"]
    )

    sensitivity_response = client.post(
        "/api/sensitivity/reg/run",
        json={"method": "sobol", "n_features_to_select": 2, "train_partition": "train"},
    )
    assert sensitivity_response.status_code == 200, sensitivity_response.text
    sensitivity_job_id = sensitivity_response.json()["job_id"]
    sensitivity_job = _wait(client, sensitivity_job_id)
    assert sensitivity_job["status"] == "completed", sensitivity_job
    assert sensitivity_job["request"] == {
        "method": "sobol",
        "n_features_to_select": 2,
        "train_partition": "train",
    }
    sensitivity_result = sensitivity_job["result"]
    assert sensitivity_result["interpretation"]["category"] == "input_sensitivity"
    assert sensitivity_result["interpretation"]["causal_effect"] is False
    assert sensitivity_result["evaluation_protocol"] == {
        "train_partition": "train",
        "train_fingerprint": client.get("/api/datasets/reg").json()["partitions"]["train"]["fingerprint"],
        "row_count": 120,
        "feature_columns": ["a", "b", "c"],
        # Numeric-only projection: this fixture is all-numeric, so nothing is
        # excluded and no row is dropped. Non-numeric partitions are covered in
        # tests/test_sensitivity_numeric.py.
        "excluded_columns": [],
        "rows_dropped": 0,
        # A tabular task analyses its own columns; a temporal one analyses the
        # generated design (tests/test_sensitivity_temporal.py).
        "feature_source": "partition_columns",
        "temporal": None,
    }

    correlation = client.get("/api/sensitivity/reg/correlation-matrix").json()
    assert correlation["interpretation"]["category"] == "descriptive_association"
    assert correlation["interpretation"]["measure"] == "Pearson correlation"
    assert correlation["interpretation"]["causal_effect"] is False

    intervals = client.post(
        "/api/evaluation/reg/%s/intervals" % version,
        json={"alpha": 0.2, "group_by": "a", "max_groups": 2},
    )
    assert intervals.status_code == 200, intervals.text
    interval_result = intervals.json()
    assert 0.0 <= interval_result["empirical_coverage"] <= 1.0
    assert interval_result["coverage_interval"]["method"] == "Wilson score interval"
    assert interval_result["coverage_interval"]["lower"] <= interval_result["empirical_coverage"]
    assert interval_result["empirical_coverage"] <= interval_result["coverage_interval"]["upper"]
    assert interval_result["mean_interval_width"] is not None
    assert interval_result["mean_interval_score"] is not None
    assert interval_result["calibration_source_partition"] == "train"
    assert interval_result["fit_rows"] > 0
    assert "random holdout" in interval_result["holdout_partition"]
    assert interval_result["group_by"] == "a"
    assert len(interval_result["subgroup_coverage"]["groups"]) <= 3

    project = client.post(
        "/api/projects",
        json={
            "name": "p",
            "research_questions": ["Which features influence the response?"],
            "artifacts": [
                {"kind": "dataset_partition", "task_name": "reg", "artifact_id": "train"},
                {"kind": "job", "task_name": "reg", "artifact_id": job_id},
                {"kind": "job", "task_name": "reg", "artifact_id": sensitivity_job_id, "label": "Sensitivity analysis"},
                {"kind": "model_bundle", "task_name": "reg", "artifact_id": version},
            ],
        },
    ).json()
    hypothesis_record = client.post(
        "/api/projects/%s/records" % project["project_id"],
        json={
            "kind": "hypothesis",
            "title": "Feature response hypothesis",
            "text": "The input features have unequal influence on the response.",
            "research_question": "Which features influence the response?",
        },
    )
    assert hypothesis_record.status_code == 200, hypothesis_record.text
    sensitivity_record = client.post(
        "/api/projects/%s/records" % project["project_id"],
        json={
            "kind": "analysis",
            "title": "Sensitivity analysis",
            "text": "The linked job records the exact input partition and method.",
            "evidence": [
                {
                    "relation": "supports",
                    "artifact": {
                        "kind": "job",
                        "task_name": "reg",
                        "artifact_id": sensitivity_job_id,
                    },
                }
            ],
        },
    )
    assert sensitivity_record.status_code == 200, sensitivity_record.text
    linked_sensitivity = client.get("/api/jobs/" + sensitivity_job_id).json()
    assert linked_sensitivity["request"] == sensitivity_job["request"]
    assert linked_sensitivity["result"] == sensitivity_result
    summary = client.get("/api/projects/%s/summary" % project["project_id"]).json()
    sensitivity_snapshot = next(
        artifact["snapshot"]
        for artifact in summary["artifacts"]
        if artifact["artifact_id"] == sensitivity_job_id
    )
    assert sensitivity_snapshot["details"]["result_fingerprint"]
    conclusion = client.post(
        "/api/projects/%s/records" % project["project_id"],
        json={
            "kind": "conclusion",
            "title": "Sensitivity result recorded",
            "text": "The completed sensitivity run is retained as project evidence.",
            "evidence": [
                {
                    "relation": "supports",
                    "artifact": {
                        "kind": "job",
                        "task_name": "reg",
                        "artifact_id": sensitivity_job_id,
                    },
                }
            ],
        },
    )
    assert conclusion.status_code == 200, conclusion.text
    summary = client.get("/api/projects/%s/summary" % project["project_id"]).json()
    assert all(stage["done"] for stage in summary["stages"])


def test_intervals_reject_classifier(client):
    _register(client, "cls")
    version, _ = _quick_bundle(client, "cls", "classification")
    response = client.post("/api/evaluation/cls/%s/intervals" % version, json={})
    assert response.status_code == 422
    # The refusal should read as guidance: what the model is and how to proceed.
    detail = response.json()["detail"]
    assert "is a Classification model" in detail
    assert "only defined for numeric regression targets" in detail
    assert "Select or train a regression bundle" in detail
    # The refusal is machine-readable too, so clients can branch on it without
    # parsing the human-facing text.
    assert response.headers["X-Reason"] == "intervals_require_regression_bundle"


def test_design_endpoints(client):
    variables = [{"name": "x", "low": 0, "high": 1}, {"name": "t", "low": 10, "high": 20}]
    lhs = client.post("/api/design/latin-hypercube", json={"variables": variables, "n_samples": 5}).json()
    assert lhs["columns"] == ["x", "t"] and len(lhs["points"]) == 5
    samples = lhs["points"]
    values = [p[0] ** 2 + p[1] / 100 for p in samples]
    nxt = client.post("/api/design/suggest-next", json={"variables": variables, "samples": samples, "values": values})
    assert nxt.status_code == 200, nxt.text
    bad = client.post("/api/design/latin-hypercube", json={"variables": [{"name": "x", "low": 2, "high": 1}]})
    assert bad.status_code == 400


def test_a_regression_search_reports_its_error_in_natural_units(client):
    """Stored scores are scikit-learn units (-RMSE); the payload says so, once.

    The search minimizes the real error: a trial's loss is +RMSE, and because the
    stored score is the higher-is-better form the table shows -RMSE and the best
    candidate is the *largest* (least negative) value. Without this block a reader
    sees a column of negative numbers and concludes the search preferred the biggest
    error.
    """
    _register_regression(client, "reg-signs")
    body = {
        "preset": "regression",
        "strategy": "exhaustive",
        "length": 1,
        "scoring": "neg_root_mean_squared_error",
        "random_state": 0,
    }
    record = _wait(client, client.post("/api/experiments/reg-signs/run", json=body).json()["job_id"])
    assert record["status"] == "completed", record
    result = record["result"]
    assert result["scoring"] == "neg_root_mean_squared_error"
    assert result["score_metric"] == {
        "scoring": "neg_root_mean_squared_error",
        "label": "RMSE",
        "negated": True,
        "higher_is_better": False,
    }
    scores = [row["score"] for row in result["top_pipelines"] if row["score"] is not None]
    assert scores and all(score <= 0.0 for score in scores)
    # Highest stored value first == smallest RMSE first.
    assert scores == sorted(scores, reverse=True)
    assert [-score for score in scores] == sorted(-score for score in scores)


def test_scoring_display_covers_metrics_outside_the_curated_list():
    from api.training import scoring_display

    assert scoring_display("accuracy")["label"] == "Accuracy"
    assert scoring_display("accuracy")["negated"] is False
    assert scoring_display("r2")["label"] == "R²"
    assert scoring_display("r2")["higher_is_better"] is True
    # A neg_* metric absent from the curated list still drops the prefix.
    fallback = scoring_display("neg_mean_squared_log_error")
    assert fallback == {
        "scoring": "neg_mean_squared_log_error",
        "label": "mean squared log error",
        "negated": True,
        "higher_is_better": False,
    }
    assert scoring_display(None)["label"] == "Score"


def test_optimize_pipeline_without_a_scorer_uses_the_task_family(client):
    """Omitting ``scoring`` must not fall back to a classification default.

    The Optimize this structure button sends only the structure, and the request
    model's historical default was ``accuracy``; on a regression target every fold
    raised ("continuous is not supported") and the job answered only "every
    candidate was rejected".
    """
    _register_regression(client, "reg-opt")
    response = client.post(
        "/api/experiments/reg-opt/optimize-pipeline",
        json={"seq": ["sklearn.tree.DecisionTreeRegressor"]},
    )
    assert response.status_code == 200, response.text
    record = _wait(client, response.json()["job_id"])
    assert record["status"] == "completed", record.get("error")
    assert record["result"]["train_score"] is not None


def test_optimize_pipeline_names_the_scorer_when_nothing_can_be_scored(client):
    _register_regression(client, "reg-bad-scorer")
    response = client.post(
        "/api/experiments/reg-bad-scorer/optimize-pipeline",
        json={"seq": ["sklearn.tree.DecisionTreeRegressor"], "scoring": "accuracy"},
    )
    assert response.status_code == 200, response.text
    record = _wait(client, response.json()["job_id"])
    assert record["status"] == "failed"
    assert "scoring 'accuracy'" in record["error"]
    assert "problem type" in record["error"]


# ---------------------------------------------------------------------------
# Component catalog (visual search-space editor)
# ---------------------------------------------------------------------------


def test_components_endpoint_shape(client):
    res = client.get("/api/experiments/components")
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"components"}
    rows = body["components"]
    assert isinstance(rows, list) and len(rows) >= 25
    for row in rows:
        assert set(row) <= {"fqn", "name", "role", "task", "description", "params", "preselected"}
        assert row["role"] in ("estimator", "transformer", "sampler")
        if row["role"] == "transformer":
            assert "task" not in row
        else:  # estimator or sampler rows are task-tagged (samplers: classification only)
            assert row.get("task") in ("classification", "regression")
        for spec in row["params"].values():
            assert spec["type"] in ("real", "integer", "categorical")


def test_components_split_by_task(client):
    rows = client.get("/api/experiments/components").json()["components"]
    clf = [r for r in rows if r.get("task") == "classification"]
    reg = [r for r in rows if r.get("task") == "regression"]
    # exact curated counts: 8 classifiers + 3 samplers, 8 regressors
    assert len(clf) == 11 and len(reg) == 8
    # a classifier FQN must never appear under regression (and vice versa)
    clf_names = {r["fqn"] for r in clf if r["role"] == "estimator"}
    reg_names = {r["fqn"] for r in reg}
    assert not (clf_names & reg_names)


def test_gradient_boosting_in_catalog(client):
    rows = client.get("/api/experiments/components").json()["components"]
    by_fqn = {r["fqn"]: r for r in rows}
    for fqn, task in (
        ("xgboost.XGBClassifier", "classification"),
        ("lightgbm.LGBMClassifier", "classification"),
        ("xgboost.XGBRegressor", "regression"),
        ("lightgbm.LGBMRegressor", "regression"),
    ):
        row = by_fqn[fqn]
        assert row["task"] == task and row.get("preselected") is True
        params = set(row["params"])
        assert {"n_estimators", "learning_rate"} <= params, f"{fqn} missing core GBM knobs"
        assert len(params) >= 9, f"{fqn}: expected the full significant-knob set"


def test_resamplers_classification_only(client):
    rows = client.get("/api/experiments/components").json()["components"]
    samplers = {r["fqn"]: r for r in rows if r["role"] == "sampler"}
    assert set(samplers) == {
        "imblearn.over_sampling.SMOTE",
        "imblearn.over_sampling.ADASYN",
        "imblearn.under_sampling.RandomUnderSampler",
    }
    for row in samplers.values():
        assert row["task"] == "classification"


def test_sk_surrogate_transformers_present(client):
    rows = client.get("/api/experiments/components").json()["components"]
    by_fqn = {r["fqn"]: r for r in rows}
    sens = by_fqn["SKSurrogate.SensAprx"]
    assert sens["role"] == "transformer" and "task" not in sens
    assert sens["params"]["method"]["items"] == ["sobol", "morris", "delta-mmnt"]
    corr = by_fqn["SKSurrogate.CorrelationThreshold"]
    assert corr["role"] == "transformer"
    assert corr["params"]["threshold"]["type"] == "real"


def test_rgf_catalog_entries(client):
    rows = client.get("/api/experiments/components").json()["components"]
    by_fqn = {r["fqn"]: r for r in rows}
    clf = by_fqn["rgf.sklearn.RGFClassifier"]["params"]
    # underscore forms are the only ones rgf 3.12 accepts (fit-probed)
    assert clf["algorithm"]["items"] == ["RGF", "RGF_Opt", "RGF_Sib"]
    reg = by_fqn["rgf.sklearn.RGFRegressor"]["params"]
    assert set(reg) >= {"algorithm", "reg_depth"}


def test_rf_et_specs_match_preset_table(client):
    from SKSurrogate.aml import PRESET_PARAM_SPECS

    rows = client.get("/api/experiments/components").json()["components"]
    by_fqn = {r["fqn"]: r for r in rows}
    for fqn in (
        "sklearn.ensemble.RandomForestClassifier",
        "sklearn.ensemble.ExtraTreesClassifier",
        "sklearn.ensemble.RandomForestRegressor",
        "sklearn.ensemble.ExtraTreesRegressor",
    ):
        assert by_fqn[fqn]["params"] == PRESET_PARAM_SPECS, f"{fqn} drifted from the preset table"


def test_preselected_sets(client):
    rows = client.get("/api/experiments/components").json()["components"]
    for task, expected in (
        ("classification", {
            "sklearn.ensemble.RandomForestClassifier",
            "xgboost.XGBClassifier",
            "lightgbm.LGBMClassifier",
        }),
        ("regression", {
            "sklearn.ensemble.RandomForestRegressor",
            "xgboost.XGBRegressor",
            "lightgbm.LGBMRegressor",
        }),
    ):
        got = {r["fqn"] for r in rows if r.get("task") == task and r.get("preselected")}
        assert got == expected, f"preselected set for {task} drifted: {got}"
