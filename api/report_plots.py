"""Headless, bounded figures derived from the same data as the report."""

import math
import threading
from io import BytesIO
from numbers import Real

import matplotlib
import numpy as np
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure
from matplotlib.text import Text

_LOCK = threading.RLock()
_INK = "#0B0F14"
_SLATE = "#3A4552"
_BLUE = "#2B44A8"
_TEAL = "#17727E"
_EMBER = "#B4521A"
_GRID = "#D9DEE5"


def _number(value):
    if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _label(value, limit=34):
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _figure(draw, size):
    # Matplotlib's font and renderer caches are process-wide; report jobs run in threads.
    with _LOCK:
        fig = Figure(figsize=size, layout="constrained")
        FigureCanvasPdf(fig)
        draw(fig)
        for text in fig.findobj(match=Text):
            text.set_parse_math(False)
        buffer = BytesIO()
        fig.savefig(buffer, format="pdf", metadata={"CreationDate": None})
        return buffer.getvalue()


def _axes(ax):
    ax.set_axisbelow(True)
    ax.grid(color=_GRID, linewidth=0.6, alpha=0.8)
    ax.tick_params(labelsize=8, colors=_SLATE)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.xaxis.label.set_color(_SLATE)
    ax.yaxis.label.set_color(_SLATE)
    ax.title.set_color(_INK)


def _heatmap(labels, columns, values, *, cmap, vmin, vmax, colorbar_label):
    def draw(fig):
        ax = fig.subplots()
        colors = matplotlib.colormaps[cmap].copy()
        colors.set_bad("#e8eaed")
        masked = np.ma.masked_invalid(np.asarray(values, dtype=float))
        image = ax.imshow(masked, cmap=colors, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(columns)), [_label(name, 22) for name in columns],
                      rotation=45 if len(columns) > 1 else 0, ha="right" if len(columns) > 1 else "center")
        ax.set_yticks(range(len(labels)), [_label(name, 28) for name in labels])
        ax.tick_params(labelsize=8, length=0)
        for row in range(len(labels)):
            for col in range(len(columns)):
                value = masked[row, col]
                if np.ma.is_masked(value):
                    continue
                rgb = image.cmap(image.norm(value))[:3]
                color = "white" if sum(x * y for x, y in zip(rgb, (0.2126, 0.7152, 0.0722))) < 0.5 else _INK
                ax.text(col, row, "%.2g" % value, ha="center", va="center", color=color, fontsize=8)
        fig.colorbar(image, ax=ax, label=colorbar_label, shrink=0.8)

    return _figure(draw, (7, max(2.5, 0.38 * len(labels) + (2 if len(columns) > 1 else 1.2))))


def _eda_association_figures(payload):
    figures = {}
    pairs = [item for item in payload.get("feature_associations") or []
             if item.get("method") == "pearson" and _number(item.get("value")) is not None]
    strength = {}
    for pair in pairs:
        for feature in (pair["feature_a"], pair["feature_b"]):
            strength[feature] = max(strength.get(feature, 0), abs(pair["value"]))
    labels = sorted(strength, key=lambda feature: (-strength[feature], feature))[:10]
    if len(labels) >= 2:
        indices = {label: index for index, label in enumerate(labels)}
        matrix = np.full((len(labels), len(labels)), np.nan)
        np.fill_diagonal(matrix, 1)
        for pair in pairs:
            if pair["feature_a"] in indices and pair["feature_b"] in indices:
                row, col = indices[pair["feature_a"]], indices[pair["feature_b"]]
                matrix[row, col] = matrix[col, row] = pair["value"]
        figures["figures/eda-correlation.pdf"] = _heatmap(
            labels, labels, matrix, cmap="RdBu_r", vmin=-1, vmax=1, colorbar_label="Pearson r",
        )
    for method in ("pearson", "correlation_ratio", "cramers_v"):
        entries = [item for item in payload.get("target_associations") or []
                   if item.get("method") == method and _number(item.get("value")) is not None]
        entries.sort(key=lambda item: (-abs(item["value"]), item["feature"]))
        entries = entries[:12]
        if entries:
            figures["figures/eda-target-%s.pdf" % method.replace("_", "-")] = _heatmap(
                [item["feature"] for item in entries], [method.replace("_", " ")],
                [[item["value"]] for item in entries],
                cmap="RdBu_r" if method == "pearson" else "YlGnBu",
                vmin=-1 if method == "pearson" else 0, vmax=1, colorbar_label="Association",
            )
    sensitivity = payload.get("report_sensitivity")
    if sensitivity:
        entries = [(name, _number(weight)) for name, weight in
                   zip(sensitivity["feature_columns"], sensitivity["weights"])]
        entries = sorted((item for item in entries if item[1] is not None),
                         key=lambda item: (-abs(item[1]), item[0]))[:12]
        if entries:
            max_abs = max(abs(value) for _, value in entries) or 1
            signed = any(value < 0 for _, value in entries)
            figures["figures/eda-sensitivity.pdf"] = _heatmap(
                [name for name, _ in entries], [sensitivity["method"]],
                [[value] for _, value in entries],
                cmap="RdBu_r" if signed else "YlGnBu", vmin=-max_abs if signed else 0,
                vmax=max_abs, colorbar_label="Sensitivity weight",
            )
    return figures


def eda_figures(payload):
    """Return filename -> PDF for the selected partition's available summaries."""
    figures = {}
    columns = [
        item for item in (payload.get("missing") or {}).get("columns", [])
        if _number(item.get("missing_pct")) is not None
    ]
    columns.sort(key=lambda item: item["missing_pct"], reverse=True)
    if columns:
        shown = columns[:12]

        def draw_missing(fig):
            ax = fig.subplots()
            entries = list(reversed(shown))
            ax.barh(range(len(entries)), [100 * c["missing_pct"] for c in entries], color=_TEAL)
            ax.set_yticks(range(len(entries)), [_label(c["column"]) for c in entries])
            ax.set(xlabel="Missing values (% of partition rows)", xlim=(0, 100))
            _axes(ax)

        figures["figures/eda-missingness.pdf"] = _figure(
            draw_missing, (7, max(2.4, 0.32 * len(shown) + 1.1)),
        )

    numeric = []
    for column in payload.get("column_profiles") or []:
        dist = column.get("distribution") or {}
        values = [_number(dist.get(key)) for key in ("min", "q25", "median", "q75", "max")]
        if column.get("kind") == "numeric" and all(value is not None for value in values):
            numeric.append((column.get("column"), values))
    if numeric:
        shown = numeric[:6]

        def draw_distributions(fig):
            axes = fig.subplots(len(shown), 1, squeeze=False)
            for ax, (name, values) in zip(axes[:, 0], shown):
                low, q1, median, q3, high = values
                ax.bxp([{"whislo": low, "q1": q1, "med": median, "q3": q3,
                         "whishi": high, "fliers": []}], vert=False, showfliers=False,
                       patch_artist=True, boxprops={"facecolor": "#DCEEF0", "edgecolor": _TEAL},
                       medianprops={"color": _EMBER, "linewidth": 1.5})
                ax.set_yticks([])
                ax.set_title(_label(name), loc="left", fontsize=9)
                _axes(ax)

        figures["figures/eda-distributions.pdf"] = _figure(
            draw_distributions, (7, max(2.5, 1.05 * len(shown))),
        )

    classes = (payload.get("target_composition") or {}).get("classes") or []
    classes = [item for item in classes if _number(item.get("count")) is not None][:10]
    if classes:
        def draw_classes(fig):
            ax = fig.subplots()
            entries = list(reversed(classes))
            ax.barh(range(len(entries)), [c["count"] for c in entries], color=_BLUE)
            ax.set_yticks(range(len(entries)), [_label(c["label"]) for c in entries])
            ax.set_xlabel("Rows")
            _axes(ax)

        figures["figures/eda-target.pdf"] = _figure(
            draw_classes, (7, max(2.4, 0.34 * len(classes) + 1.1)),
        )
    figures.update(_eda_association_figures(payload))
    return figures


def explain_figures(models):
    """Return per-model figures without mixing scores from different estimators."""
    figures = {}
    for index, model in enumerate(models):
        diagnostics = model.get("diagnostics") or {}
        global_result = diagnostics.get("Permutation importance and partial dependence") or {}
        importance = [item for item in global_result.get("importance", [])
                      if _number(item.get("mean")) is not None][:12]
        if importance:
            def draw_importance(fig):
                ax = fig.subplots()
                for position, item in enumerate(reversed(importance)):
                    ax.barh(position, item["mean"], color=_BLUE)
                    std = _number(item.get("std"))
                    if std is not None and std >= 0:
                        ax.errorbar(item["mean"], position, xerr=std, color=_SLATE,
                                    capsize=2, linewidth=0.8)
                ax.set_yticks(range(len(importance)), [_label(item["feature"]) for item in reversed(importance)])
                ax.axvline(0, color=_INK, linewidth=0.7)
                ax.set_xlabel("Score decrease after permutation")
                _axes(ax)

            figures["figures/model-%d-importance.pdf" % index] = _figure(
                draw_importance, (7, max(2.5, 0.34 * len(importance) + 1.1)),
            )

        for suffix, result, series, value_key, ylabel in (
            ("pdp", global_result, "partial_dependence", "average", "Average model response"),
            ("ale", diagnostics.get("Accumulated local effects") or {}, "curves", "accumulated", "Accumulated local effect"),
        ):
            curves = []
            for curve in result.get(series, []):
                pairs = [(_number(x), _number(y)) for x, y in zip(curve.get("grid", []), curve.get(value_key, []))]
                if sum(x is not None and y is not None for x, y in pairs) >= 2:
                    curves.append((curve.get("feature"), pairs))
            curves = curves[:3]
            if curves:
                def draw_curves(fig):
                    axes = fig.subplots(len(curves), 1, squeeze=False)
                    for ax, (name, pairs) in zip(axes[:, 0], curves):
                        x = [x if x is not None else float("nan") for x, _ in pairs]
                        y = [y if y is not None else float("nan") for _, y in pairs]
                        ax.plot(x, y, color=_TEAL, linewidth=1.6, marker=".", markersize=3)
                        ax.set_title(_label(name), loc="left", fontsize=9)
                        ax.set_xlabel("Feature value")
                        ax.set_ylabel(ylabel, fontsize=8)
                        _axes(ax)

                figures["figures/model-%d-%s.pdf" % (index, suffix)] = _figure(
                    draw_curves, (7, max(2.5, 2.0 * len(curves))),
                )

        row = diagnostics.get("Row-level explanation") or {}
        contributions = [item for item in row.get("contributions", [])
                         if _number(item.get("mean")) is not None][:12]
        if contributions:
            def draw_contributions(fig):
                ax = fig.subplots()
                entries = list(reversed(contributions))
                ax.barh(range(len(entries)), [c["mean"] for c in entries],
                        color=[_TEAL if c["mean"] >= 0 else _EMBER for c in entries])
                ax.set_yticks(range(len(entries)), [_label(c["feature"]) for c in entries])
                ax.axvline(0, color=_INK, linewidth=0.7)
                ax.set_xlabel("Contribution to model response")
                _axes(ax)

            figures["figures/model-%d-row.pdf" % index] = _figure(
                draw_contributions, (7, max(2.5, 0.34 * len(contributions) + 1.1)),
            )

        intervals = diagnostics.get("Prediction intervals") or {}
        points = [p for p in intervals.get("points", [])
                  if all(_number(p.get(key)) is not None for key in ("actual", "prediction", "lower", "upper"))][:40]
        if points:
            def draw_intervals(fig):
                ax = fig.subplots()
                for position, point in enumerate(points):
                    ax.vlines(position, point["lower"], point["upper"], color=_TEAL, alpha=0.6, linewidth=1)
                ax.scatter(range(len(points)), [p["prediction"] for p in points],
                           color=_BLUE, s=12, label="Prediction", zorder=3)
                ax.scatter(range(len(points)), [p["actual"] for p in points],
                           color=_EMBER, s=12, label="Actual", zorder=3)
                ax.set(xlabel="Finite holdout preview (display order)", ylabel="Target")
                ax.legend(fontsize=8)
                _axes(ax)

            figures["figures/model-%d-intervals.pdf" % index] = _figure(draw_intervals, (7, 3.2))
    return figures
