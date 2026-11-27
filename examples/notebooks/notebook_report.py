"""Report plumbing shared by the end-to-end example notebooks.

Three things every notebook in this directory needs, and that are worth keeping
in one place so the examples cannot drift from the report renderer they claim to
use:

* ``markdown_table`` — turn a DataFrame into the pipe-table syntax that both the
  Markdown report and :func:`api.reports.report_latex` understand;
* ``save_figure`` / ``figure_of`` — persist a figure as PNG for the artifact
  directory and as PDF for the branded report (the report template only accepts
  ``figures/<slug>.pdf``);
* ``compile_pdf`` — compile the branded PDF through the same
  :mod:`api.reports` code path the control plane uses, returning
  ``(pdf_bytes, note)`` so a missing LuaLaTeX installation downgrades the
  deliverable to Markdown instead of breaking the notebook.
"""

from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from pathlib import Path

_SLUG = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    """Lower-case, hyphen-separated file stem (``"Per-horizon MASE"`` -> ``per-horizon-mase``)."""
    return _SLUG.sub("-", str(text).strip().lower()).strip("-") or "figure"


def _cell(value, digits: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return "n/a"
        if float(value).is_integer() and abs(value) < 1e6:
            return str(int(value))
        return ("%%.%df" % digits) % value
    return str(value).replace("|", r"\|").replace("\n", " ")


def markdown_table(frame, columns=None, digits: int = 3, max_rows: int = 25) -> str:
    """Render a DataFrame (or list of dicts) as a Markdown pipe table.

    Cells are stringified deterministically — floats to ``digits`` places, NaN and
    infinities as ``n/a``, booleans as ``yes``/``no`` — because the LaTeX
    renderer escapes what it is given and a raw ``numpy`` repr would leak into
    the PDF. Pipes inside cell text are escaped so the table cannot break.
    """
    rows = list(frame) if not hasattr(frame, "columns") else None
    if rows is not None:
        headers = list(rows[0].keys()) if rows else []
        data = [[row.get(header) for header in headers] for row in rows]
    else:
        headers = list(frame.columns if columns is None else columns)
        data = frame.loc[:, headers].head(max_rows).itertuples(index=False, name=None)
        data = [list(row) for row in data]
    if not headers:
        return "_(no rows)_"
    lines = ["| " + " | ".join(str(header) for header in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in data[:max_rows]:
        lines.append("| " + " | ".join(_cell(value, digits) for value in row) + " |")
    if len(data) > max_rows:
        lines.append(
            "| _%d more rows in the CSV artifact_ |%s"
            % (len(data) - max_rows, " |" * (len(headers) - 1))
        )
    return "\n".join(lines)


def figure_of(result):
    """Return the figure behind a tracker plot call.

    ``mltrack`` plot methods return either the ``pyplot`` module (most of them)
    or an ``Axes`` (:meth:`plot_cumulative_gain`, :meth:`plot_lift_curve`), and
    older helpers return a ``Figure`` directly. Normalising here keeps the
    notebook cells free of that trivia.
    """
    if hasattr(result, "gcf"):  # pyplot module
        return result.gcf()
    if hasattr(result, "figure"):  # Axes
        return result.figure
    if hasattr(result, "savefig"):  # Figure
        return result
    raise TypeError("cannot find a matplotlib figure on %r" % type(result).__name__)


def figure_pdf(figure) -> bytes:
    """Serialise a figure as PDF bytes (the report template's only figure format)."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="pdf", bbox_inches="tight")
    return buffer.getvalue()


def save_figure(figure, directory, stem: str, dpi: int = 200) -> Path:
    """Write the figure as PNG (artifacts) and PDF (report) into ``directory``.

    Returns the PNG path. The PDF sibling is what
    :func:`api.reports.compile_report` accepts, so the report cell never has to
    re-plot a figure to get a vector copy.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = slug(stem)
    path = directory / (name + ".png")
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    figure.savefig(directory / (name + ".pdf"), format="pdf", bbox_inches="tight")
    return path


def compile_pdf(
    markdown: str,
    *,
    task: str,
    title: str,
    subtitle: str,
    figures=None,
    inserts=None,
    generated_at: str | None = None,
) -> tuple[bytes | None, str]:
    """Compile the branded PDF from report Markdown.

    ``figures`` maps ``"figures/<slug>.pdf"`` to PDF bytes; ``inserts`` maps a
    ``(section, subsection_or_None)`` pair to the LaTeX snippet inserted under
    that heading (see :func:`api.reports.figure_latex`). Returns
    ``(pdf_bytes, note)``; on a missing or failing LaTeX installation the bytes
    are ``None`` and the note explains why, so a notebook run never dies at the
    reporting step.
    """
    generated_at = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        from api.reports import ReportCompilationError, compile_report, report_latex
    except Exception as exc:  # pragma: no cover - repo layout changed
        return None, "report renderer unavailable: %s" % exc
    try:
        source = report_latex(
            markdown,
            task,
            title,
            subtitle,
            generated_at,
            inserts=inserts or {},
        )
        pdf = compile_report(source, figures=figures or {})
    except ReportCompilationError as exc:
        return None, "PDF skipped: %s" % exc
    except FileNotFoundError as exc:  # lualatex missing
        return None, "PDF skipped: %s" % exc
    return pdf, "compiled %d KB PDF" % (len(pdf) // 1024)


def write_text(path: Path, text: str) -> Path:
    """Write UTF-8 text to ``path`` (creating parents) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
