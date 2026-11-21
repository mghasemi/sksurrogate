"""Background generation and task-scoped downloads of explainability reports."""

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import FileResponse

from SKSurrogate.structsearch import SearchStopped

from .. import control, logs
from ..columns import read_ignored_columns
from ..config import settings
from ..deps import bad_request, not_found, open_tracker, task_access
from ..eda import _format_number, _markdown_table
from ..jobs import job_manager
from ..partitions import resolve_partition
from ..report_plots import explain_figures
from ..reports import compile_report, explain_appendix_latex, figure_latex, report_latex, source_archive
from ..training import (
    compute_learning_curve, get_stored_cv_spec, learning_curve_metric_label,
    learning_curve_metric_options, problem_family,
)
from . import evaluation, explain
from .evaluation import _bundle_versions

router = APIRouter(prefix="/api/reports", tags=["reports"], dependencies=[Depends(task_access)])


class ExplainReportRequest(BaseModel):
    max_rows: int = Field(default=500, ge=20, le=1000)
    stability_samples: int = Field(default=1, ge=1, le=5)
    ale_bins: int = Field(default=20, ge=5, le=50)
    row_index: int = Field(default=0, ge=0)
    row_orders: int = Field(default=5, ge=1, le=10)
    alpha: float = Field(default=0.1, gt=0, lt=1)
    group_by: str | None = None


def _check_stop():
    if control.stop_requested():
        raise SearchStopped("Report generation cancelled")


def _flatten(value, prefix=""):
    """Render every returned field, including curve points and diagnostic caveats."""
    if isinstance(value, dict):
        if not value:
            yield "%s: {}" % prefix
        for key, item in value.items():
            path = "%s / %s" % (prefix, key) if prefix else str(key)
            yield from _flatten(item, path)
    elif isinstance(value, list):
        if not value:
            yield "%s: []" % prefix
        elif all(not isinstance(item, (dict, list)) for item in value):
            for start in range(0, len(value), 8):
                label = "%s [%d-%d]" % (prefix, start, min(start + 8, len(value)) - 1)
                yield "%s: %s" % (
                    label, ", ".join(json.dumps(item, ensure_ascii=False) for item in value[start:start + 8])
                )
        else:
            for index, item in enumerate(value):
                yield from _flatten(item, "%s [%d]" % (prefix, index))
    else:
        yield "%s: %s" % (prefix, json.dumps(value, ensure_ascii=False))


def _markdown(task_name, generated_at, models, history=None):
    lines = [
        "# Explainability report",
        "- Task: %s" % task_name,
        "- Generated: %s (UTC)" % generated_at,
        "- All explanations describe fitted models, not causal effects.",
        "- Each diagnostic uses the settings and reference scope recorded below.",
    ]
    for model in models:
        lines.extend(["", "## Model: %s" % model["model_version"]])
        for name, result in model["diagnostics"].items():
            lines.extend(["", "### %s" % name])
            lines.extend("- " + line for line in _flatten(result))
    lines.extend(["", "## Experiment evaluation history"])
    lines.extend("- " + line for line in _flatten(history or []))
    return "\n".join(lines) + "\n"


def _summary_markdown(task_name, generated_at, models, history=None):
    lines = [
        "# Explainability report",
        "- Task: %s" % task_name,
        "- Generated: %s (UTC)" % generated_at,
        "- Model scores and response scales need not be comparable.",
        "- Explain response profiles require at least two finite grid/response pairs; interval plots require finite preview points.",
        "- The appendix previews large collections; complete diagnostic results, settings, sampling information and errors are in results.json in the source ZIP.",
    ]
    lines.extend(["", "## Model metric comparison",
                  "- Stored bundle metrics are shown as recorded; differing evaluation protocols and data may make comparisons invalid."])
    metric_rows = [[model["model_version"], metric, _format_number(value)]
                   for model in models
                   for metric, value in (model["diagnostics"].get("Stored model metrics", {}).get("metrics") or {}).items()]
    if metric_rows:
        lines.append(_markdown_table(["Model", "Metric", "Stored value"], metric_rows))
    else:
        lines.append("- No stored metrics are available.")
    for model in models:
        lines.extend(["", "## Model: %s" % model["model_version"]])
        for name, result in model["diagnostics"].items():
            lines.extend(["", "### %s" % name])
            if result.get("status") == "unavailable":
                lines.append("- Unavailable: %s" % result.get("detail"))
                continue
            sample = result.get("reference_sample")
            if sample:
                lines.append("- Reference: %s partition; %s of %s rows; %s."
                             % (sample.get("partition"), sample.get("used_rows"),
                                sample.get("source_rows"), sample.get("sampling")))
            if name == "Stored model metrics":
                lines.append("- Stored bundle metrics; their evaluation scope is determined by the recorded training provenance and may differ between models.")
                metrics = result.get("metrics") or {}
                if metrics:
                    lines.append(_markdown_table(
                        ["Metric", "Stored value"],
                        [[key, _format_number(value)] for key, value in metrics.items()],
                    ))
                else:
                    lines.append("- No stored metrics are available.")
            elif name == "Learning curves":
                lines.append("- Cross-validation on the %s partition using the task's stored splitter (or default 3-fold); estimator clones are fitted at each size."
                             % (result.get("train_partition") or "train"))
                rows = []
                for curve in result.get("curves") or []:
                    if curve.get("error"):
                        lines.append("- %s unavailable: %s" % (curve["scoring_label"], curve["error"]))
                    else:
                        series = curve["series"]
                        rows.append([
                            curve["scoring_label"], series["train_sizes"][-1],
                            _format_number(series["train_scores_mean"][-1]),
                            _format_number(series["validation_scores_mean"][-1]),
                            _format_number(series["validation_scores_std"][-1]),
                        ])
                if rows:
                    lines.append("- At the largest training size: fold mean scores and validation standard deviation.")
                    lines.append(_markdown_table(["Metric", "Rows per fold", "Train mean", "Validation mean", "Validation SD"], rows))
            elif name == "Classification evaluation curves":
                lines.append("- " + result["evaluation_protocol"]["note"])
                lines.append("- Partition: %s; 25%% deterministic slice; rows: %s; bins: %s."
                             % (result["evaluation_protocol"]["pool_partition"], result["n_test_samples"], result["bins"]))
                if result.get("roc"):
                    lines.append("- ROC AUC: %s." % _format_number(result["roc"].get("auc")))
            elif name == "Nested cross-validation":
                lines.append("- Latest completed nested-CV job; historical results are not recomputed and may refer to a different dataset version. See the detailed output for fingerprints and settings.")
                nested = result["result"]
                lines.append(_markdown_table(
                    ["Metric", "Value"],
                    [[key, _format_number(nested.get(key))]
                     for key in ("mean_outer_score", "final_validation_score", "repeat_count")],
                ))
            elif name == "Permutation importance and partial dependence":
                importance = result.get("importance") or []
                if importance:
                    lines.append(_markdown_table(
                        ["Feature", "Mean score decrease", "Repeat SD"],
                        [[item.get("feature"), _format_number(item.get("mean")),
                          _format_number(item.get("std"))] for item in importance],
                    ))
                else:
                    lines.append("- No permutation importance values were available.")
            elif name == "Accumulated local effects":
                curves = result.get("curves") or []
                if curves:
                    lines.append(_markdown_table(
                        ["Feature", "Bins", "Excluded rows"],
                        [[curve.get("feature"), len(curve.get("counts") or []),
                          curve.get("excluded_rows")] for curve in curves],
                    ))
                else:
                    lines.append("- No ALE curves were available.")
            elif name == "Row-level explanation":
                reconstruction = result.get("reconstruction") or {}
                if reconstruction:
                    lines.append("- Baseline response: %s; explained response: %s; sum of contributions: %s."
                                 % tuple(_format_number(reconstruction.get(key)) for key in
                                         ("baseline_prediction", "explained_prediction", "sum_contributions")))
                contributions = result.get("contributions") or []
                if contributions:
                    lines.append(_markdown_table(
                        ["Feature", "Contribution", "Order SD"],
                        [[item.get("feature"), _format_number(item.get("mean")),
                          _format_number(item.get("sample_std"))] for item in contributions],
                    ))
                else:
                    lines.append("- No row contributions were available.")
            elif name == "Prediction intervals":
                fields = (
                    ("Nominal coverage", "nominal_coverage"),
                    ("Empirical coverage", "empirical_coverage"),
                    ("Mean interval width", "mean_interval_width"),
                    ("Finite intervals", "finite_interval_count"),
                    ("Unbounded intervals", "unbounded_interval_count"),
                    ("Holdout rows", "holdout_rows"),
                )
                lines.append(_markdown_table(
                    ["Metric", "Value"],
                    [[label, _format_number(result.get(key))] for label, key in fields],
                ))
                if result.get("coverage_assumption"):
                    lines.append("- Coverage assumption: %s" % result["coverage_assumption"])
            for key in ("errors", "skipped"):
                for item in result.get(key) or []:
                    lines.append("- %s: %s" % (key.capitalize(), "; ".join(
                        "%s: %s" % (field, value) for field, value in item.items()
                    )))
            for limitation in (result.get("interpretation") or {}).get("limitations") or []:
                lines.append("- %s" % limitation)
    lines.extend(["", "## Experiment evaluation history",
                  "- Completed task-level experiment jobs; these are not necessarily comparable to model metrics."])
    if history:
        lines.append(_markdown_table(
            ["Job", "Created", "Train score", "History length"],
            [[item["job_id"], item["created_at"], _format_number(item["train_score"]),
              item["history_length"]] for item in history],
        ))
    else:
        lines.append("- No completed experiment jobs are available.")
    return "\n".join(lines) + "\n"


def _figure_inserts(models, figures):
    inserts = {}
    for metric_index in range(6):
        filename = "figures/evaluation-metric-%d.pdf" % metric_index
        if filename in figures:
            inserts.setdefault(("Model metric comparison", None), []).extend(figure_latex(
                filename,
                "Stored metric for up to 12 models, using the first six metrics. "
                "Values may have different evaluation provenance and are not necessarily comparable.",
            ))
    for index, model in enumerate(models):
        section = "Model: %s" % model["model_version"]
        specs = (
            ("Classification evaluation curves", "evaluation-curves",
             "Classification diagnostic curves and predicted-score histogram on the deterministic 25% slice. The fitted bundle was not refit and the slice may overlap its training data."),
            ("Nested cross-validation", "nested-cv",
             "Outer-fold scores from the first repeat of the latest completed nested-CV job; the dashed line is the reported mean outer score. Check the historical settings and data fingerprints."),
            ("Permutation importance and partial dependence", "importance",
             "Permutation score sensitivity for up to 12 features, measured as the decrease in estimator score after shuffling. Error bars show the standard deviation across repeats."),
            ("Permutation importance and partial dependence", "pdp",
             "Partial-dependence profiles for up to three available features, using the reference sample. Features have separate axes."),
            ("Accumulated local effects", "ale",
             "Accumulated local effects for up to three available numeric features, using the reference sample. Features have separate axes."),
            ("Row-level explanation", "row",
             "Largest absolute row contributions, up to 12 features, relative to the stated baseline."),
            ("Prediction intervals", "intervals",
             "Up to 40 finite intervals from the holdout preview; unbounded or non-finite preview points are omitted."),
        )
        for subsection, suffix, caption in specs:
            filename = "figures/model-%d-%s.pdf" % (index, suffix)
            if filename in figures:
                inserts.setdefault((section, subsection), []).extend(figure_latex(filename, caption))
        learning = model["diagnostics"].get("Learning curves") or {}
        for curve_index, curve in enumerate(learning.get("curves") or []):
            filename = "figures/model-%d-learning-%d.pdf" % (index, curve_index)
            if filename in figures:
                caption = ("Cross-validated learning curve for %s on the %s partition. "
                           "Training and validation scores are fold means; error bars show fold standard deviations."
                           % (curve["scoring_label"], learning.get("train_partition") or "train"))
                inserts.setdefault((section, "Learning curves"), []).extend(figure_latex(filename, caption))
        global_result = model["diagnostics"].get("Permutation importance and partial dependence") or {}
        if global_result.get("importance"):
            inserts.setdefault((section, "Permutation importance and partial dependence"), []).insert(
                0, r"\[ I_j = s(\hat f, X, y) - s(\hat f, X_{\pi(j)}, y) \]"
                   r"\noindent Here $s$ is the estimator score and $\pi(j)$ shuffles feature $j$."
            )
        row = model["diagnostics"].get("Row-level explanation") or {}
        if row.get("reconstruction"):
            inserts.setdefault((section, "Row-level explanation"), []).insert(
                0, r"\[ \hat f(x_{\mathrm{explained}}) = \hat f(x_{\mathrm{baseline}}) + \sum_j c_j \]"
            )
        interval = model["diagnostics"].get("Prediction intervals") or {}
        if interval.get("nominal_coverage") is not None:
            inserts.setdefault((section, "Prediction intervals"), []).insert(
                0, r"\[ \text{Nominal coverage} = 1 - \alpha \]"
            )
    return inserts


def _evaluation_diagnostics(task_name, version, jobs, partition=None):
    bundle = evaluation._load_bundle(task_name, version)
    family = problem_family(bundle.model)
    partition = resolve_partition(task_name, partition)
    metadata = bundle.metadata if isinstance(bundle.metadata, dict) else {}
    stored = {
        "family": family,
        "metrics": bundle.metrics or {},
        "created_at": bundle.created_at,
        "dataset_fingerprint": bundle.dataset_fingerprint,
        "run_id": bundle.run_id,
        "reproducibility": metadata.get("reproducibility"),
        "recorded_evaluations": metadata.get("evaluations", []),
    }
    with open_tracker(task_name) as tracker:
        train_fingerprint = (tracker.dataset_splits().get(partition) or {}).get("fingerprint")
    learning = {"family": family, "train_partition": partition,
                "train_fingerprint": train_fingerprint, "ignored_columns": read_ignored_columns(task_name),
                "cv_spec": get_stored_cv_spec(task_name), "curves": []}
    for metric in learning_curve_metric_options(family)["metrics"]:
        _check_stop()
        scoring = metric["value"]
        alpha = 0.5 if metric.get("requires_alpha") else None
        logs.log_event("INFO", "Computing learning curve %s for %s" % (scoring, version))
        curve = {"scoring": scoring, "scoring_label": learning_curve_metric_label(scoring, alpha),
                 "pinball_alpha": alpha, "higher_is_better": metric["higher_is_better"]}
        try:
            series = compute_learning_curve(
                bundle, task_name=task_name, scoring=scoring, pinball_alpha=alpha,
                train_partition=partition, n_points=5,
            )
            curve["series"] = series
            if not any(value is not None for value in series["validation_scores_mean"]):
                curve["error"] = "No finite validation scores were produced."
        except SearchStopped:
            raise
        except Exception as exc:  # Match the Evaluation page's per-bundle fit/scorer errors.
            curve["error"] = "%s: %s" % (type(exc).__name__, exc)
        if curve.get("error"):
            logs.log_event("WARNING", "Learning curve %s unavailable for %s: %s"
                           % (scoring, version, curve["error"]))
        learning["curves"].append(curve)
    if not learning["curves"]:
        learning["status"] = "unavailable"
        learning["detail"] = "No learning-curve metrics are available for this estimator family."
    _check_stop()
    if family == "Classification":
        try:
            curves = evaluation.bundle_curves(task_name, version, bins=10)
        except HTTPException as exc:
            if exc.status_code >= 500:
                raise
            curves = {"status": "unavailable", "detail": exc.detail, "http_status": exc.status_code}
        except (ValueError, AttributeError) as exc:
            curves = {"status": "unavailable", "detail": "%s: %s" % (type(exc).__name__, exc)}
        if curves.get("status") == "unavailable":
            logs.log_event("WARNING", "Classification curves unavailable for %s: %s" % (version, curves["detail"]))
    else:
        curves = {"status": "unavailable", "detail": "Classification diagnostic curves require a classifier.",
                  "by_design": True, "reason": "curves_require_classifier"}
    nested = next((job for job in jobs if job.get("kind") == "nested-cv"
                   and job.get("status") == "completed"
                   and (job.get("result") or {}).get("model_version") == version), None)
    nested_result = ({"job_id": nested["job_id"], "created_at": nested["created_at"],
                      "request": nested.get("request"), "result": nested["result"]}
                     if nested else {"status": "unavailable",
                                     "detail": "No completed nested-CV job is available for this model.",
                                     "by_design": True, "reason": "no_nested_cv_job"})
    return {
        "Stored model metrics": stored,
        "Learning curves": learning,
        "Classification evaluation curves": curves,
        "Nested cross-validation": nested_result,
    }


def _explained_feature_count(task_name, partition):
    """Width of the feature space the explain diagnostics attribute, or ``None``.

    ``explain._explain_frame`` is the single definition of that space: a tabular
    bundle reads the partition's feature columns, a temporal bundle reads the
    *generated* forecast design because its raw partition carries no feature
    column of its own. Counting raw columns instead clamped a temporal report's
    row orders from a 0-feature frame, and reading the literal ``"train"``
    partition 404'd a task stored as ``main`` before any diagnostic ran.

    ``None`` means the space could not be established (e.g. a partition with no
    feature column); the caller then keeps the requested orders and lets each
    diagnostic report its own error instead of failing the whole report.
    """
    try:
        features, _, _ = explain._explain_frame(task_name, partition)
    except HTTPException as exc:
        logs.log_event(
            "WARNING", "Cannot size the explainable feature space of %r on partition %r: %s"
            % (task_name, partition, exc.detail),
        )
        return None
    return int(features.shape[1])


def _make_report(task_name, versions, body, job_id):
    partition = resolve_partition(task_name, None)
    feature_count = _explained_feature_count(task_name, partition)
    if feature_count is None:
        effective_orders = body.row_orders
    else:
        effective_orders = min(body.row_orders, math.factorial(min(feature_count, 4)))
        if effective_orders != body.row_orders:
            logs.log_event(
                "INFO", "Using %d distinct row orders instead of %d: only %d exist for %d features"
                % (effective_orders, body.row_orders, effective_orders, feature_count),
            )
    diagnostics = (
        ("Permutation importance and partial dependence", explain.explain_bundle,
         explain.ExplainRequest(max_rows=body.max_rows, stability_samples=body.stability_samples)),
        ("Accumulated local effects", explain.accumulated_local_effects_bundle,
         explain.AleRequest(max_rows=body.max_rows, stability_samples=body.stability_samples, n_bins=body.ale_bins)),
        ("Row-level explanation", explain.explain_row,
         explain.RowExplainRequest(row_index=body.row_index, orders=effective_orders)),
        ("Prediction intervals", explain.prediction_intervals,
         explain.IntervalRequest(alpha=body.alpha, group_by=body.group_by)),
    )
    models = []
    failures = 0
    not_applicable = 0
    jobs = job_manager.list(task_name)
    history = [
        {"job_id": job["job_id"], "created_at": job["created_at"],
         "train_score": (job.get("result") or {}).get("train_score"),
         "history_length": len((job.get("result") or {}).get("evaluation_history") or [])}
        for job in jobs if job.get("kind") == "experiment" and job.get("status") == "completed"
    ]
    for version in versions:
        model = {"model_version": version,
                 "diagnostics": _evaluation_diagnostics(task_name, version, jobs, partition)}
        # ``by_design`` marks a diagnostic that cannot apply to this bundle (a
        # regression model has no classification curves; a model with no nested-CV
        # job has no nested-CV result). Those are absences, not failures, and used
        # to inflate the reported count into "15 unavailable diagnostics" on a
        # perfectly healthy report.
        model_failures = sum(
            1 for value in model["diagnostics"].values()
            if value.get("status") == "unavailable" and not value.get("by_design")
        )
        model_not_applicable = sum(
            1 for value in model["diagnostics"].values()
            if value.get("status") == "unavailable" and value.get("by_design")
        )
        failures += model_failures
        not_applicable += model_not_applicable
        failures += sum(1 for curve in model["diagnostics"]["Learning curves"]["curves"] if curve.get("error"))
        for name, function, request in diagnostics:
            _check_stop()
            logs.log_event("INFO", "Computing %s for %s" % (name, version))
            try:
                result = function(task_name, version, request)
                if name == "Row-level explanation" and effective_orders != body.row_orders:
                    result["report_order_adjustment"] = {
                        "requested_orders": body.row_orders,
                        "effective_orders": effective_orders,
                        "reason": "fewer distinct feature orders exist",
                    }
            except HTTPException as exc:
                if exc.status_code >= 500:
                    raise
                failures += 1
                result = {"status": "unavailable", "http_status": exc.status_code, "detail": exc.detail}
                if exc.headers and exc.headers.get("X-Reason"):
                    result["reason"] = exc.headers["X-Reason"]
                logs.log_event("WARNING", "%s unavailable for %s: %s" % (name, version, exc.detail))
            model["diagnostics"][name] = result
        models.append(model)
    _check_stop()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = {"task_name": task_name, "generated_at": generated_at,
            "settings": body.model_dump(mode="json"), "models": models, "experiment_history": history}
    markdown = _markdown(task_name, generated_at, models, history)
    summary = _summary_markdown(task_name, generated_at, models, history)
    figures = explain_figures(models)
    _check_stop()
    source = report_latex(
        summary, task_name, "Model Explainability Report",
        "Task %s; %d model(s); %s partition" % (task_name, len(versions), partition), generated_at,
        inserts=_figure_inserts(models, figures), appendix_latex=explain_appendix_latex(models),
    )
    archive = source_archive(source, markdown, data_json=json.dumps(data, ensure_ascii=False, indent=2),
                             figures=figures)
    logs.log_event("INFO", "Compiling explainability report")
    pdf = compile_report(source, figures=figures)
    _check_stop()
    destination = settings.task_report_dir(task_name, job_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".report-", dir=destination.parent) as temp:
        from pathlib import Path

        root = Path(temp)
        (root / "report.pdf").write_bytes(pdf)
        (root / "report.zip").write_bytes(archive)
        _check_stop()
        os.rename(root, destination)
    return {"model_count": len(models), "diagnostic_failures": failures,
            "diagnostics_not_applicable": not_applicable, "generated_at": generated_at}


@router.post("/{task_name}/explain")
def submit_explain_report(task_name: str, body: ExplainReportRequest):
    """Run all Explain diagnostics for the task's current local bundles."""
    versions = _bundle_versions(task_name)
    if not versions:
        raise bad_request("No model bundles are available for this task.")
    job_id = job_manager.submit(
        task_name, "explain_report",
        lambda job_id: lambda: _make_report(task_name, versions, body, job_id),
        metadata_factory=lambda job_id: {"request": body.model_dump(mode="json"), "model_versions": versions},
    )
    return {"job_id": job_id, "status": "queued"}


@router.get("/{task_name}/{job_id}/{format}")
def download_explain_report(task_name: str, job_id: str, format: Literal["pdf", "zip"]):
    """Download only a completed report owned by this task."""
    record = job_manager.get(job_id)
    if not record or record.get("task_name") != task_name or record.get("kind") != "explain_report":
        raise not_found("No explainability report found for this task.")
    if record.get("status") != "completed":
        raise HTTPException(status_code=409, detail="This report is not complete.")
    path = settings.task_report_dir(task_name, job_id) / ("report." + format)
    if not path.is_file():
        raise not_found("The completed report file is unavailable.")
    return FileResponse(
        path, media_type="application/pdf" if format == "pdf" else "application/zip",
        filename="explainability-report." + format,
    )
