import io
import json
import time
import zipfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

from api.config import settings
from api.main import app
from api.reports import ReportCompilationError, eda_latex


def test_report_tables_preserve_escaped_pipes_and_format_scientific_numbers():
    from api.eda import _markdown_table
    from api.reports import report_latex

    markdown = "## Values\n" + _markdown_table(
        ["Name", "Value"], [["a|b_%", "1.230e-05"], ["x", r"\input{bad}"]],
    )
    source = report_latex(markdown, "task", "title", "subtitle", "today")
    assert r"\begin{longtblr}" in source
    assert r"a|b\_\allowbreak{}\%" in source
    assert r"\ensuremath{1.230\times 10^{-5}}" in source
    assert r"\textbackslash{}input\{bad\}" in source


def test_explain_appendix_groups_complete_diagnostic_values_safely():
    from api.reports import explain_appendix_latex

    models = [{"model_version": r"m_%\input{bad}", "diagnostics": {
        "Prediction intervals": {
            "status": "unavailable", "detail": "Not applicable",
            "points": [{"actual": 1.25, "prediction": 2, "lower": None, "upper": 3},
                       {"actual": 4, "prediction": 5, "lower": 0, "upper": 6}],
            "curves": [{"feature": "a|b", "grid": [1, 2], "average": [0.1, None], "counts": [3]}],
            "errors": [], "interpretation": {"limitations": ["Not causal. " * 20]},
        },
    }}]
    source = "\n".join(explain_appendix_latex(models))
    assert r"m\_\%\textbackslash{}input\{bad\}" in source
    assert r"\begin{longtblr}" in source and "bg=skTealTint" in source
    assert r"Index & actual & prediction & lower & upper \\" in source
    assert r"0 & 1.25 & 2 & null & 3 \\" in source
    assert "Diagnostic details / curves [0] / indexed arrays" in source
    assert r"Index & grid & average & counts \\" in source
    assert r"1 & 2 & null & not present \\" in source
    assert "a|b" in source and "Not applicable" in source
    assert r"\begin{sknote}[Limitations]" in source
    assert r"Diagnostic details / errors" in source and r"\noindent []\par" in source


def test_figures_use_report_values_and_skip_unavailable_data(monkeypatch):
    from matplotlib.figure import Figure
    from api import report_plots

    axes = []

    def inspect(draw, size):
        fig = Figure(figsize=size)
        draw(fig)
        axes.append(fig.axes)
        return b"%PDF-figure"

    monkeypatch.setattr(report_plots, "_figure", inspect)
    eda = {
        "missing": {"columns": [{"column": "a", "missing_pct": 0.25},
                                {"column": "b", "missing_pct": 0.0}]},
        "column_profiles": [{"column": "a", "kind": "numeric",
                             "distribution": {"min": 0, "q25": 1, "median": 2, "q75": 3, "max": 8}}],
        "target_composition": {"classes": [{"label": "yes", "count": 7}]},
    }
    figures = report_plots.eda_figures(eda)
    assert set(figures) == {"figures/eda-missingness.pdf", "figures/eda-distributions.pdf",
                            "figures/eda-target.pdf"}
    assert [patch.get_width() for patch in axes[0][0].patches] == [0, 25]
    assert list(axes[1][0].lines[0].get_xdata()) == [1, 0]
    assert list(axes[1][0].lines[1].get_xdata()) == [3, 8]
    assert list(axes[1][0].lines[4].get_xdata()) == [2, 2]
    assert [patch.get_width() for patch in axes[2][0].patches] == [7]
    assert report_plots.eda_figures({"column_profiles": [{"kind": "numeric",
            "distribution": {"min": None}}]}) == {}

    axes.clear()
    models = [{"diagnostics": {
        "Permutation importance and partial dependence": {
            "importance": [{"feature": "a", "mean": 0.4, "std": 0.1}],
            "partial_dependence": [{"feature": "a", "grid": [1, 2, 3], "average": [4, None, 6]}]},
        "Accumulated local effects": {"curves": [{"feature": "a", "grid": [1, 2], "accumulated": [-1, 1]}]},
        "Row-level explanation": {"contributions": [{"feature": "a", "mean": -0.3}]},
        "Prediction intervals": {"points": [
            {"actual": 2, "prediction": 3, "lower": 1, "upper": 4},
            {"actual": 2, "prediction": 3, "lower": None, "upper": None}]}},
    }, {"diagnostics": {"Prediction intervals": {"status": "unavailable"}}}]
    figures = report_plots.explain_figures(models)
    assert set(figures) == {"figures/model-0-importance.pdf", "figures/model-0-pdp.pdf",
                            "figures/model-0-ale.pdf", "figures/model-0-row.pdf",
                            "figures/model-0-intervals.pdf"}
    assert axes[0][0].patches[0].get_width() == 0.4
    assert list(axes[1][0].lines[0].get_xdata()) == [1, 2, 3]
    assert axes[3][0].patches[0].get_width() == -0.3
    assert len(axes[4][0].collections[0].get_segments()) == 1


def test_eda_heatmaps_preserve_methods_missing_pairs_and_sensitivity(monkeypatch):
    import numpy as np
    from matplotlib.figure import Figure
    from api import report_plots

    plots = []

    def inspect(draw, size):
        fig = Figure(figsize=size)
        draw(fig)
        plots.append(fig.axes[0])
        return b"%PDF-figure"

    monkeypatch.setattr(report_plots, "_figure", inspect)
    payload = {
        "feature_associations": [
            {"feature_a": "a", "feature_b": "b", "method": "pearson", "value": 0.8},
            {"feature_a": "b", "feature_b": "c", "method": "pearson", "value": -0.5},
            {"feature_a": "a", "feature_b": "c", "method": "correlation_ratio", "value": 0.9},
        ],
        "target_associations": [
            {"feature": "a", "method": "pearson", "value": -0.7},
            {"feature": "b", "method": "pearson", "value": None},
            {"feature": "c", "method": "correlation_ratio", "value": 0.4},
        ],
        "report_sensitivity": {"method": "morris", "feature_columns": ["a", "c"],
                               "weights": [0.2, 0.7]},
    }
    figures = report_plots.eda_figures(payload)
    assert set(figures) == {"figures/eda-correlation.pdf", "figures/eda-target-pearson.pdf",
                            "figures/eda-target-correlation-ratio.pdf", "figures/eda-sensitivity.pdf"}
    matrix = plots[0].images[0].get_array()
    assert matrix[0, 1] == 0.8 and matrix[1, 0] == 0.8
    assert matrix[1, 2] == -0.5 and np.ma.is_masked(matrix[0, 2])
    assert plots[1].images[0].get_array()[0, 0] == -0.7
    assert plots[2].images[0].get_array()[0, 0] == 0.4
    assert list(plots[3].images[0].get_array().flatten()) == [0.7, 0.2]
    assert report_plots.eda_figures({"feature_associations": [
        {"feature_a": "a", "feature_b": "b", "method": "pearson", "value": None}]}) == {}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "root", tmp_path)
    settings.ensure()
    with TestClient(app) as client:
        yield client


def test_eda_latex_escapes_user_supplied_tex():
    source = eda_latex(
        "# Report\n## Missingness\n- column: `x_1%` \\input{secret}; ≤ 100\n",
        "task_1", "train", "2026-01-01",
    )
    assert r"\section{Missingness}" in source
    assert r"x\_1\%" in source
    assert r"\textbackslash{}input\{secret\}" in source
    assert r"\ensuremath{\leq}" in source
    assert r"\sksubtitle{Dataset task\_1; partition train}" in source


def test_eda_download_uses_selected_partition_and_propagates_failures(client, monkeypatch):
    frame = pd.DataFrame({"x": list(range(30)), "target": list(range(30))})
    for partition in ("train", "other"):
        response = client.post(
            "/api/datasets/example/register",
            data={"target": "target", "partition": partition},
            files={"file": ("data.csv", frame.to_csv(index=False), "text/csv")},
        )
        assert response.status_code == 200, response.text

    calls = []

    def compile_stub(source, *, figures):
        calls.append((source, figures))
        return b"%PDF-1.7"

    monkeypatch.setattr("api.routers.datasets.compile_report", compile_stub)
    pdf = client.get("/api/datasets/example/other/eda-report/pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content == b"%PDF-1.7"
    assert "partition other" in calls[0][0]
    assert "figures/eda-missingness.pdf" in calls[0][1]
    archive = client.get("/api/datasets/example/other/eda-report/zip")
    assert archive.status_code == 200
    with zipfile.ZipFile(io.BytesIO(archive.content)) as package:
        assert {"report.tex", "report.md", "sk.sty", "sk-brand.sty",
                "logo/sk-mark.tikz", "logo/cover-surface.tikz"} <= set(package.namelist())
        assert b"partition other" in package.read("report.tex")
        for name in calls[0][1]:
            assert package.read(name).startswith(b"%PDF-")
            assert name.encode() in package.read("report.tex")

    def fail(source, *, figures):
        raise ReportCompilationError("LuaLaTeX unavailable")

    monkeypatch.setattr("api.routers.datasets.compile_report", fail)
    response = client.get("/api/datasets/example/other/eda-report/pdf")
    assert response.status_code == 503
    assert "LuaLaTeX unavailable" in response.json()["detail"]
    assert client.get("/api/datasets/example/other/eda-report/zip").status_code == 200


def test_eda_download_includes_only_matching_sensitivity_and_correlation(client, monkeypatch):
    from api.deps import open_tracker
    from api.routers import datasets

    frame = pd.DataFrame({"a": range(40), "b": range(39, -1, -1), "target": range(40)})
    response = client.post("/api/datasets/example/register", data={"target": "target", "partition": "train"},
                           files={"file": ("data.csv", frame.to_csv(index=False), "text/csv")})
    assert response.status_code == 200, response.text
    with open_tracker("example") as tracker:
        fingerprint = tracker.dataset_splits()["train"]["fingerprint"]
    result = {
        "method": "sobol", "feature_columns": ["a", "b"], "weights": [0.7, 0.3],
        "evaluation_protocol": {"train_partition": "train", "train_fingerprint": fingerprint,
                                "excluded_columns": [], "row_count": 40, "rows_dropped": 0},
    }
    valid = {"job_id": "valid", "created_at": "2026-01-01", "kind": "sensitivity",
             "status": "completed", "result": result}
    stale = {**valid, "job_id": "stale", "result": {**result, "evaluation_protocol": {
        **result["evaluation_protocol"], "train_fingerprint": "old"}}}
    monkeypatch.setattr(datasets.job_manager, "list", lambda task: [stale, valid])
    changed_features = {"columns": ["a", "b", "target"], "target": "target", "ignored_columns": ["b"]}
    assert datasets._report_sensitivity("example", "train", changed_features) is None
    assert datasets._report_sensitivity("example", "other", changed_features) is None
    archive = client.get("/api/datasets/example/train/eda-report/zip")
    assert archive.status_code == 200, archive.text
    with zipfile.ZipFile(io.BytesIO(archive.content)) as package:
        for name in ("figures/eda-correlation.pdf", "figures/eda-target-pearson.pdf",
                     "figures/eda-sensitivity.pdf"):
            assert package.read(name).startswith(b"%PDF-")
            assert name.encode() in package.read("report.tex")
        source = package.read("report.tex")
        assert b"job: valid" in source and b"job: stale" not in source
        assert b"not model importance or sensitivity" in source
        assert b"rows used: 40; rows dropped: 0" in source

    monkeypatch.setattr(datasets.job_manager, "list", lambda task: [stale])
    archive = client.get("/api/datasets/example/train/eda-report/zip")
    with zipfile.ZipFile(io.BytesIO(archive.content)) as package:
        assert "figures/eda-sensitivity.pdf" not in package.namelist()
        assert b"No completed sensitivity job matches" in package.read("report.tex")


def test_explain_report_runs_every_diagnostic_and_preserves_unavailable_results(client, monkeypatch):
    from api.routers import reports

    calls = []

    def diagnostic(name):
        def run(task, version, request):
            calls.append((name, version, request))
            if name == "intervals" and version == "classifier":
                raise HTTPException(
                    status_code=422, detail="Regression only",
                    headers={"X-Reason": "intervals_require_regression_bundle"},
                )
            result = {"model_version": version, "interpretation": {"limitations": ["not causal"]}}
            if name == "explain_bundle":
                result.update(importance=[{"feature": "x|1", "mean": 0.1, "std": 0.02}],
                              partial_dependence=[{"feature": "x|1", "grid": [0, 1], "average": [2, 3]}])
            elif name == "accumulated_local_effects_bundle":
                result["curves"] = [{"feature": "x|1", "grid": [0, 1], "accumulated": [-1, 1],
                                     "counts": [10], "excluded_rows": 0}]
            elif name == "explain_row":
                result["contributions"] = [{"feature": "x|1", "mean": 0.5, "sample_std": 0.1}]
            else:
                result.update(nominal_coverage=0.8, points=[
                    {"actual": 2, "prediction": 3, "lower": 1, "upper": 4},
                    {"actual": 2, "prediction": 3, "lower": None, "upper": None}])
            return result
        return run

    monkeypatch.setattr(reports, "_bundle_versions", lambda task: ["regressor", "classifier"])
    monkeypatch.setattr(reports.explain, "_features_and_target", lambda task, partition: (pd.DataFrame(columns=["x", "z"]), []))
    for name in ("explain_bundle", "accumulated_local_effects_bundle", "explain_row", "prediction_intervals"):
        monkeypatch.setattr(reports.explain, name, diagnostic("intervals" if name == "prediction_intervals" else name))
    monkeypatch.setattr(reports, "compile_report", lambda source, *, figures: b"%PDF-1.7\n" + source.encode())
    submitted = client.post(
        "/api/reports/example/explain",
        json={"max_rows": 100, "stability_samples": 3, "ale_bins": 10,
              "row_index": 2, "row_orders": 5, "alpha": 0.2, "group_by": "region"},
    )
    assert submitted.status_code == 200, submitted.text
    job_id = submitted.json()["job_id"]
    for _ in range(200):
        job = client.get("/api/jobs/" + job_id).json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "completed", job.get("error")
    assert job["result"]["model_count"] == 2
    assert job["result"]["diagnostic_failures"] == 1
    assert len(calls) == 8
    by_name = {name: request for name, version, request in calls if version == "regressor"}
    assert by_name["explain_bundle"].max_rows == 100
    assert by_name["accumulated_local_effects_bundle"].n_bins == 10
    assert by_name["explain_row"].row_index == 2
    assert by_name["explain_row"].orders == 2
    assert by_name["intervals"].group_by == "region"
    assert client.get("/api/reports/other/%s/pdf" % job_id).status_code == 404
    pdf = client.get("/api/reports/example/%s/pdf" % job_id)
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF-")
    package = client.get("/api/reports/example/%s/zip" % job_id)
    with zipfile.ZipFile(io.BytesIO(package.content)) as archive:
        results = json.loads(archive.read("results.json"))
        assert len(results["models"]) == 2
        failed = results["models"][1]["diagnostics"]["Prediction intervals"]
        assert failed["status"] == "unavailable"
        assert failed["reason"] == "intervals_require_regression_bundle"
        adjustment = results["models"][0]["diagnostics"]["Row-level explanation"]["report_order_adjustment"]
        assert adjustment["requested_orders"] == 5 and adjustment["effective_orders"] == 2
        source = archive.read("report.tex")
        assert b"Regression only" in source
        assert b"x|1" in source and b"longtblr" in source
        assert b"Nominal coverage" in source and b"1 - \\alpha" in source
        for suffix in ("importance", "pdp", "ale", "row", "intervals"):
            name = "figures/model-0-%s.pdf" % suffix
            assert archive.read(name).startswith(b"%PDF-")
            assert name.encode() in source
        assert "figures/model-1-intervals.pdf" not in archive.namelist()


def test_explain_report_requires_models_and_does_not_expose_unfinished_jobs(client, monkeypatch):
    from api.routers import reports

    monkeypatch.setattr(reports, "_bundle_versions", lambda task: [])
    assert client.post("/api/reports/example/explain", json={}).status_code == 400
    assert client.get("/api/reports/example/not-a-job/pdf").status_code == 404
