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
    assert r"\begin{longtable}" in source and r"\rowcolor{skTealTint}" in source
    assert r"\textcolor{skTealInk}{actual}" in source
    assert r"0 & 1.25 & 2 & null & 3 \\" in source
    assert "Diagnostic details / curves [0] / indexed arrays" in source
    assert r"\textcolor{skTealInk}{grid}" in source
    assert r"1 & 2 & null & not present \\" in source
    assert "a|b" in source and "Not applicable" in source
    assert r"\begin{sknote}[Limitations]" in source
    assert r"Diagnostic details / errors" in source and r"\noindent []\par" in source


def test_explain_appendix_previews_large_collections_with_original_indices():
    from api.reports import explain_appendix_latex

    models = [{"model_version": "m", "diagnostics": {"Test": {
        "arrays": {"x": list(range(51)), "y": list(range(49))},
        "points": [{"value": i} for i in range(51)],
        "values": list(range(50)),
        "complex": [{"nested": {"value": i}} for i in range(51)],
        "many_fields": {"key%d" % i: i for i in range(51)},
    }}}]
    source = "\n".join(explain_appendix_latex(models))
    assert source.count("Showing the first 25 and last 25 of 51 entries.") == 4
    assert "25 & 25 & 25" not in source
    assert r"26 & 26 & 26 \\" in source
    assert r"50 & 50 & not present \\" in source
    assert "complex [24]" in source and "complex [26]" in source
    assert "complex [25]" not in source
    assert "key24" in source and "key26" in source and "key25" not in source
    assert r"25 & 25 \\" in source  # A list of exactly 50 is not truncated.
    assert r"\texttt{results.json}" in source


def test_large_explain_appendix_compiles_without_losing_archive_data():
    import shutil
    import subprocess
    from api.reports import compile_report, explain_appendix_latex, report_latex, source_archive
    from api.routers.reports import _markdown

    if not shutil.which("lualatex"):
        pytest.skip("LuaLaTeX is required to compile the PDF")
    values = list(range(1500))
    models = [{"model_version": "m", "diagnostics": {"Classification evaluation curves": {
        "cumulative_gain": {"percentages": values, "gains_class0": values, "gains_class1": values},
        "lift": {"percentages": values, "lifts_class0": values, "lifts_class1": values},
    }}}]
    source = report_latex("## Model: m\n", "task", "title", "subtitle", "today",
                          appendix_latex=explain_appendix_latex(models))
    pdf = compile_report(source)
    assert pdf.startswith(b"%PDF-")
    assert source.count("Showing the first 25 and last 25 of 1500 entries.") == 2
    if shutil.which("pdftotext"):
        text = subprocess.run(["pdftotext", "-layout", "-", "-"], input=pdf,
                              capture_output=True, check=True).stdout.decode()
        assert "first 25 and last 25 of 1500 entries" in text
        assert "1499" in text
    markdown = _markdown("task", "today", models)
    archive = source_archive(source, markdown, data_json=json.dumps({"models": models}))
    with zipfile.ZipFile(io.BytesIO(archive)) as package:
        stored = json.loads(package.read("results.json"))
        assert stored["models"][0]["diagnostics"]["Classification evaluation curves"]["lift"]["percentages"] == values
        assert b"[1496-1499]: 1496, 1497, 1498, 1499" in package.read("report.md")
        assert package.read("report.tex").decode() == source


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


def test_evaluation_figures_plot_scores_and_keep_missing_values(monkeypatch):
    import numpy as np
    from matplotlib.figure import Figure
    from api import report_plots

    plots = {}

    def inspect(draw, size):
        fig = Figure(figsize=size)
        draw(fig)
        plots[len(plots)] = fig
        return b"%PDF-figure"

    monkeypatch.setattr(report_plots, "_figure", inspect)
    models = [{"model_version": "m", "diagnostics": {
        "Stored model metrics": {"metrics": {"accuracy": 0.75}},
        "Learning curves": {"curves": [
            {"scoring_label": "Accuracy", "series": {
                "train_sizes": [10, 20], "train_scores_mean": [None, None],
                "train_scores_std": [None, None], "validation_scores_mean": [0.5, 0.7],
                "validation_scores_std": [0.1, None]}},
            {"scoring_label": "Unavailable", "series": {
                "train_sizes": [10, 20], "validation_scores_mean": [None, None]}},
        ]},
        "Classification evaluation curves": {
            "roc": {"fpr": [0, 0.2, 1], "tpr": [0, 0.8, 1]},
            "calibration": {"mean_predicted_value": [0.2], "fraction_of_positives": [0.3],
                            "histogram_counts": [2, 3], "histogram_edges": [0, 0.5, 1]},
            "lift": {"percentages": [0.5, 1], "lifts_class0": [1.5, 1],
                     "lifts_class1": [0.5, 1], "class0": "no", "class1": "yes"},
        },
    }}]
    figures = report_plots.explain_figures(models)
    assert set(figures) == {"figures/evaluation-metric-0.pdf", "figures/model-0-learning-0.pdf",
                            "figures/model-0-evaluation-curves.pdf"}
    assert plots[0].axes[0].patches[0].get_width() == 0.75
    learning = plots[1].axes[0]
    assert np.isnan(learning.lines[0].get_ydata()).all()
    assert list(learning.lines[1].get_ydata()) == [0.5, 0.7]
    dashboard = plots[2].axes
    assert list(dashboard[0].lines[1].get_ydata()) == [0, 0.8, 1]
    assert [patch.get_height() for patch in dashboard[3].patches] == [2, 3]
    assert not dashboard[4].get_visible() and not dashboard[5].get_visible()


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
    from types import SimpleNamespace
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from api.routers import reports

    calls = []
    evaluation_calls = []

    def load_bundle(task, version):
        return SimpleNamespace(
            model=LinearRegression() if version == "regressor" else LogisticRegression(),
            metrics={"score": 0.8 if version == "regressor" else 0.7},
            metadata={"reproducibility": {"metric_scope": "training"}},
            created_at="2026-01-01", dataset_fingerprint="original", run_id="run",
            model_version=version,
        )

    def learning(bundle, *, task_name, scoring, pinball_alpha, train_partition, n_points):
        evaluation_calls.append((bundle.model_version, scoring, pinball_alpha))
        if bundle.model_version == "classifier" and scoring == "f1":
            raise ValueError("f1 unavailable")
        return {
            "model_version": bundle.model_version, "train_sizes": [10, 20],
            "train_scores_mean": [0.8, 0.9], "train_scores_std": [0.1, 0.1],
            "validation_scores_mean": [0.6, 0.7], "validation_scores_std": [0.2, 0.1],
            "train_sizes_fraction": [0.1, 0.2],
        }

    def curves(task, version, bins):
        return {
            "evaluation_protocol": {"pool_partition": "train", "pool_fingerprint": "current",
                                    "note": "The slice may overlap training data."},
            "n_test_samples": 10, "bins": bins,
            "roc": {"fpr": [0, 1], "tpr": [0, 1], "auc": 0.5},
            "calibration": {"mean_predicted_value": [0.2, 0.8], "fraction_of_positives": [0.1, 0.9],
                            "histogram_counts": [4, 6], "histogram_edges": [0, 0.5, 1]},
            "cumulative_gain": {"percentages": [0, 1], "gains_class0": [0, 1], "gains_class1": [0, 1],
                                "class0": "no", "class1": "yes"},
            "lift": {"percentages": [0.5, 1], "lifts_class0": [2, 1], "lifts_class1": [0, 1],
                     "class0": "no", "class1": "yes"},
            "errors": [],
        }

    jobs = [
        {"kind": "nested-cv", "status": "completed", "job_id": "nested", "created_at": "2026-01-01",
         "request": {"outer_cv": 3}, "result": {"model_version": "regressor", "mean_outer_score": 0.4,
                                                   "fold_metrics": [{"outer_fold": 1, "outer_score": 0.3},
                                                                    {"outer_fold": 2, "outer_score": 0.5}],
                                                   "evaluation_protocol": {"train_fingerprint": "old"}}},
        {"kind": "nested-cv", "status": "failed", "job_id": "failed", "created_at": "2026-01-02",
         "result": {"model_version": "classifier"}},
        {"kind": "experiment", "status": "completed", "job_id": "experiment", "created_at": "2026-01-01",
         "result": {"train_score": 0.95, "evaluation_history": [{}, {}]}},
    ]

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
    monkeypatch.setattr(reports.evaluation, "_load_bundle", load_bundle)
    monkeypatch.setattr(reports.evaluation, "bundle_curves", curves)
    monkeypatch.setattr(reports, "compute_learning_curve", learning)
    monkeypatch.setattr(reports, "get_stored_cv_spec", lambda task: {"type": "KFold", "n_splits": 3})
    monkeypatch.setattr(reports.job_manager, "list", lambda task: jobs)
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
    assert job["result"]["diagnostic_failures"] == 4
    assert ("regressor", "neg_mean_pinball_loss", 0.5) in evaluation_calls
    assert ("classifier", "f1", None) in evaluation_calls
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
        regression = results["models"][0]["diagnostics"]
        classification = results["models"][1]["diagnostics"]
        assert regression["Stored model metrics"]["reproducibility"]["metric_scope"] == "training"
        assert regression["Learning curves"]["cv_spec"]["n_splits"] == 3
        assert regression["Nested cross-validation"]["result"]["evaluation_protocol"]["train_fingerprint"] == "old"
        assert regression["Classification evaluation curves"]["status"] == "unavailable"
        assert classification["Nested cross-validation"]["status"] == "unavailable"
        assert next(c for c in classification["Learning curves"]["curves"] if c["scoring"] == "f1")["error"] == "ValueError: f1 unavailable"
        assert results["experiment_history"][0]["history_length"] == 2
        failed = results["models"][1]["diagnostics"]["Prediction intervals"]
        assert failed["status"] == "unavailable"
        assert failed["reason"] == "intervals_require_regression_bundle"
        adjustment = results["models"][0]["diagnostics"]["Row-level explanation"]["report_order_adjustment"]
        assert adjustment["requested_orders"] == 5 and adjustment["effective_orders"] == 2
        source = archive.read("report.tex")
        assert b"Regression only" in source
        assert b"x|1" in source and b"longtblr" in source
        assert b"Nominal coverage" in source and b"1 - \\alpha" in source
        assert b"The slice may overlap training data" in source
        assert b"historical results are not recomputed" in source
        assert b"f1 unavailable" in source and b"experiment" in source
        for name in ("figures/evaluation-metric-0.pdf", "figures/model-0-learning-0.pdf",
                     "figures/model-0-nested-cv.pdf", "figures/model-1-evaluation-curves.pdf"):
            assert archive.read(name).startswith(b"%PDF-")
            assert name.encode() in source
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


def test_evaluation_report_records_unavailable_scores_and_preserves_cancellation(client, monkeypatch):
    from types import SimpleNamespace
    from sklearn.linear_model import LogisticRegression
    from SKSurrogate.structsearch import SearchStopped
    from api.routers import reports

    monkeypatch.setattr(reports.evaluation, "_load_bundle", lambda task, version: SimpleNamespace(
        model=LogisticRegression(), metadata={}, metrics={}, created_at="now",
        dataset_fingerprint=None, run_id=None,
    ))
    monkeypatch.setattr(reports, "get_stored_cv_spec", lambda task: None)
    monkeypatch.setattr(reports, "learning_curve_metric_options", lambda family: {
        "metrics": [{"value": "accuracy", "higher_is_better": True}],
    })
    monkeypatch.setattr(reports, "compute_learning_curve", lambda *args, **kwargs: {
        "train_sizes": [2], "validation_scores_mean": [None],
    })
    def fail_curves(*args, **kwargs):
        raise ValueError("no classifier scores")

    monkeypatch.setattr(reports.evaluation, "bundle_curves", fail_curves)
    diagnostics = reports._evaluation_diagnostics("example", "model", [])
    curve = diagnostics["Learning curves"]["curves"][0]
    assert curve["error"] == "No finite validation scores were produced."
    assert curve["series"]["validation_scores_mean"] == [None]
    assert diagnostics["Classification evaluation curves"]["detail"] == "ValueError: no classifier scores"
    monkeypatch.setattr(reports.control, "stop_requested", lambda: True)
    with pytest.raises(SearchStopped):
        reports._evaluation_diagnostics("example", "model", [])


def test_real_model_evaluation_report_compiles(client, monkeypatch):
    import shutil
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from SKSurrogate import ModelBundle, save_bundle
    from api.deps import open_tracker
    from api.routers import reports

    if not shutil.which("lualatex"):
        pytest.skip("LuaLaTeX is required to compile the PDF")
    X, y = make_classification(n_samples=90, n_features=4, random_state=4)
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    frame["target"] = y
    response = client.post(
        "/api/datasets/example/register", data={"target": "target", "partition": "train"},
        files={"file": ("data.csv", frame.to_csv(index=False), "text/csv")},
    )
    assert response.status_code == 200, response.text
    with open_tracker("example") as tracker:
        fingerprint = tracker.dataset_splits()["train"]["fingerprint"]
    directory = settings.task_bundles_dir("example")
    directory.mkdir(parents=True, exist_ok=True)
    for version, estimator in (("classifier", LogisticRegression(max_iter=300)), ("regressor", LinearRegression())):
        model = estimator.fit(frame.drop(columns="target"), y)
        bundle = ModelBundle(model, task_name="example", model_version=version,
                             schema={key: "float" for key in ("a", "b", "c", "d")},
                             metrics={"score": model.score(frame.drop(columns="target"), y)})
        save_bundle(bundle, directory / (version + ".bundle"))
    for name in ("explain_bundle", "accumulated_local_effects_bundle", "explain_row", "prediction_intervals"):
        monkeypatch.setattr(reports.explain, name, lambda task, version, request: {})
    reports._make_report("example", ["classifier", "regressor"], reports.ExplainReportRequest(), "real")
    root = settings.task_report_dir("example", "real")
    assert (root / "report.pdf").read_bytes().startswith(b"%PDF-")
    with zipfile.ZipFile(root / "report.zip") as archive:
        data = json.loads(archive.read("results.json"))
        classifier, regressor = [item["diagnostics"] for item in data["models"]]
        assert classifier["Learning curves"]["train_fingerprint"] == fingerprint
        assert classifier["Classification evaluation curves"]["evaluation_protocol"]["pool_fingerprint"] == fingerprint
        assert classifier["Classification evaluation curves"]["roc"]["auc"] is not None
        assert regressor["Classification evaluation curves"]["status"] == "unavailable"
        assert len(classifier["Learning curves"]["curves"]) >= 5
        assert any(c["scoring"] == "neg_mean_pinball_loss" and c["pinball_alpha"] == 0.5
                   for c in regressor["Learning curves"]["curves"])
        source = archive.read("report.tex")
        for name in ("figures/model-0-learning-0.pdf", "figures/model-1-learning-0.pdf",
                     "figures/model-0-evaluation-curves.pdf", "figures/evaluation-metric-0.pdf"):
            assert archive.read(name).startswith(b"%PDF-")
            assert name.encode() in source
