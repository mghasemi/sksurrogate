"""Render and compile reports using the SKSurrogate document style."""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)
_TEMPLATE = Path(__file__).parent / "report_template"
_TEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "%": r"\%",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "≤": r"\ensuremath{\leq}",
    "≥": r"\ensuremath{\geq}",
    "χ": r"\ensuremath{\chi}",
    "δ": r"\ensuremath{\delta}",
    "Δ": r"\ensuremath{\Delta}",
    "±": r"\ensuremath{\pm}",
    "∞": r"\ensuremath{\infty}",
    "≈": r"\ensuremath{\approx}",
    "×": r"\ensuremath{\times}",
}


class ReportCompilationError(RuntimeError):
    """A report could not be compiled by the local LaTeX installation."""


def _escape(text: str) -> str:
    return "".join(
        _TEX_SPECIAL.get(char, char) if char.isprintable() else " "
        for char in str(text)
    )


def _table_cells(line):
    return [cell.strip().replace(r"\|", "|").replace(r"\\", "\\").replace("`", "")
            for cell in re.split(r"(?<!\\)\|", line.strip()[1:-1])]


def _table_value(value):
    scientific = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)[eE]([+-]?\d+)", value)
    if scientific:
        return r"\ensuremath{" + scientific.group(1) + r"\times 10^{" + str(int(scientific.group(2))) + "}}"
    return _escape(value).replace(r"\_", r"\_\allowbreak{}")


def _markdown_latex(markdown, inserts=None):
    body = []
    lines = markdown.splitlines()
    section = None
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or re.fullmatch(r"[| :\-]+", line):
            continue
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            section = line[3:]
            body.append(r"\section{" + _escape(section) + "}")
            body.extend((inserts or {}).get((section, None), []))
        elif line.startswith("### "):
            subsection = line[4:]
            body.append(r"\subsection{" + _escape(subsection) + "}")
            body.extend((inserts or {}).get((section, subsection), []))
        elif line.startswith("|") and line.endswith("|"):
            table = [_table_cells(line)]
            while index < len(lines) and lines[index].strip().startswith("|"):
                next_line = lines[index].strip()
                index += 1
                if not re.fullmatch(r"[| :\-]+", next_line):
                    table.append(_table_cells(next_line))
            count = len(table[0])
            body.append(r"\begin{longtblr}[entry=none,label=none]{width=\linewidth,colspec={" +
                        " ".join("X[l]" for _ in range(count)) +
                        r"},rowhead=1,hline{1,Z}={0.9pt,skInk},hline{2}={0.5pt,skInk},"
                        r"hline{3-Z}={0.4pt,skHair!80},row{1}={bg=skFaint,font=\bfseries}}")
            for row in table:
                # The producer supplies rectangular tables; keeping extra cells
                # joined avoids malformed LaTeX for arbitrary Markdown input.
                cells = row[:count - 1] + [" | ".join(row[count - 1:])]
                cells += [""] * (count - len(cells))
                body.append(" & ".join(_table_value(cell) for cell in cells) + r" \\")
            body.append(r"\end{longtblr}")
        else:
            line = line.replace("`", "")
            body.append(r"\noindent " + _escape(line) + r"\par\smallskip")
    return body


def figure_latex(name, caption):
    return [
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=\linewidth,height=0.70\textheight,keepaspectratio]{" + name + "}",
        r"\caption{" + _escape(caption) + "}",
        r"\end{figure}",
    ]


def _detail_table(headers, rows, *, field_values=False):
    if field_values:
        columns = r"X[3,l] X[7,l]"
    elif headers[0] == "Index":
        columns = "X[1,l] " + " ".join("X[3,l]" for _ in headers[1:])
    else:
        columns = " ".join("X[l]" for _ in headers)
    lines = [
        r"\begingroup\small" if field_values else r"\begingroup\footnotesize",
        r"\begin{longtblr}[entry=none,label=none]{width=\linewidth,colspec={" + columns +
        r"},rowhead=1,hline{1,Z}={0.9pt,skInk},hline{2}={0.5pt,skInk},"
        r"hline{3-Z}={0.4pt,skHair!80},row{1}={bg=skTealTint,font=\bfseries,fg=skTealInk}}",
    ]
    for row in [headers, *rows]:
        lines.append(" & ".join(_table_value(str(value)) for value in row) + r" \\")
    lines.extend([r"\end{longtblr}", r"\endgroup"])
    return lines


def _detail_scalar(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _detail_heading(path):
    return r"\par\medskip\noindent\textcolor{skTealInk}{\textbf{" + _escape(path) + r"}}\par"


def _detail_value(value, path):
    """Render every diagnostic value in bounded-width, breakable groups."""
    lines = [_detail_heading(path)]
    if isinstance(value, dict):
        if not value:
            return lines + [r"\noindent \{\}\par"]
        rows = []
        nested = []
        arrays = []
        for key, item in value.items():
            if (isinstance(item, list) and item
                    and all(not isinstance(entry, (dict, list, str)) for entry in item)):
                arrays.append((key, item))
            elif isinstance(item, (dict, list)) or (isinstance(item, str) and len(item) > 160):
                nested.append((key, item))
            else:
                rows.append([key, _detail_scalar(item)])
        if rows:
            lines.extend(_detail_table(["Field", "Value"], rows, field_values=True))
        if len(arrays) > 1:
            lines.extend([_detail_heading(path + " / indexed arrays"),
                          r"\noindent Arrays are aligned by index only; absent entries are marked "
                          r"``not present'' and are distinct from null values.\par"])
            for start in range(0, len(arrays), 3):
                group = arrays[start:start + 3]
                table_rows = [
                    [index] + [_detail_scalar(array[index]) if index < len(array) else "not present"
                               for _, array in group]
                    for index in range(max(len(array) for _, array in group))
                ]
                lines.extend(_detail_table(["Index"] + [key for key, _ in group], table_rows))
        else:
            nested.extend(arrays)
        for key, item in nested:
            lines.extend(_detail_value(item, path + " / " + str(key)))
    elif isinstance(value, list):
        if not value:
            return lines + [r"\noindent []\par"]
        if all(isinstance(item, dict) and item for item in value):
            keys = list(dict.fromkeys(key for item in value for key in item))
            if (len(keys) <= 5 and all(not isinstance(entry, (dict, list))
                                       and len(_detail_scalar(entry)) <= 100
                                       for item in value for entry in item.values())):
                rows = [[index] + [_detail_scalar(item[key]) if key in item else "not present"
                                   for key in keys] for index, item in enumerate(value)]
                return lines + _detail_table(["Index"] + keys, rows)
        if path.endswith(" / limitations") and all(isinstance(item, str) for item in value):
            lines.append(r"\begin{sknote}[Limitations]")
            for index, item in enumerate(value):
                lines.append(r"\noindent\textbf{" + str(index) + r".} " + _escape(item) + r"\par")
            return lines + [r"\end{sknote}"]
        if all(not isinstance(item, (dict, list)) and len(_detail_scalar(item)) <= 100
               for item in value):
            return lines + _detail_table(
                ["Index", "Value"], [[index, _detail_scalar(item)] for index, item in enumerate(value)],
                field_values=True,
            )
        for index, item in enumerate(value):
            lines.extend(_detail_value(item, "%s [%d]" % (path, index)))
    else:
        lines.append(r"\noindent " + _escape(_detail_scalar(value)) + r"\par")
    return lines


def explain_appendix_latex(models):
    """Present complete model diagnostics as grouped tables and readable notes."""
    body = []
    for model in models:
        body.append(r"\section{Model: " + _escape(model["model_version"]) + "}")
        for name, result in model["diagnostics"].items():
            body.append(r"\subsection{" + _escape(name) + "}")
            body.extend(_detail_value(result, "Diagnostic details"))
    return body


def report_latex(markdown: str, task: str, title: str, subtitle: str, generated_at: str,
                 *, inserts=None, appendix_latex=None) -> str:
    """Translate report Markdown to a branded LaTeX document."""
    body = _markdown_latex(markdown, inserts)
    if appendix_latex:
        body.extend([r"\clearpage", r"\appendix", r"\section*{Detailed diagnostic output}",
                     r"\addcontentsline{toc}{section}{Detailed diagnostic output}"])
        body.extend(appendix_latex)
    return "\n".join([
        r"\documentclass[11pt,letterpaper]{article}",
        r"\usepackage{sk}",
        r"\skdoctype{Report}",
        r"\sktitle[SKSurrogate]{" + _escape(title) + "}",
        r"\sksubtitle{" + _escape(subtitle) + "}",
        r"\skclient{" + _escape(task) + "}",
        r"\skauthor{SKSurrogate}",
        r"\skversion{Generated report}",
        r"\skdate{" + _escape(generated_at) + "}",
        r"\begin{document}",
        r"\skcover",
        r"\tableofcontents",
        r"\clearpage",
        r"\sloppy",
        r"\begin{sknote}[Reading this report]",
        r"Figures are descriptive summaries, not evidence of causal effects. "
        r"Figure labels may be shortened. Refer to the tables, diagnostic limitations and "
        r"captions for scope and exclusions.",
        r"\end{sknote}",
        *body,
        r"\end{document}",
        "",
    ])


def eda_latex(markdown: str, task: str, partition: str, generated_at: str, *, figures=None,
              sensitivity=None) -> str:
    figures = figures or {}
    inserts = {
        ("Missingness", None): [
            r"\[ \text{Missing percentage}_j = 100\,\frac{n_{\mathrm{missing},j}}{n_{\mathrm{rows}}}"
            r"\quad (n_{\mathrm{rows}}>0) \]",
        ],
        ("Distributions", None): [
            r"\[ \operatorname{IQR} = Q_{0.75} - Q_{0.25} \]",
        ],
    }
    for section, name, caption in (
        ("Missingness", "figures/eda-missingness.pdf",
         "Highest missingness, up to 12 columns, using the full partition. Shortened labels end in an ellipsis."),
        ("Target composition", "figures/eda-target.pdf",
         "Target class counts over the full partition; at most 10 classes from the bounded composition are shown."),
        ("Distributions", "figures/eda-distributions.pdf",
         "First six numeric feature summaries from the profiled sample. Boxes show quartiles and median; whiskers show the sample minimum and maximum, not outlier fences. Each feature has its own scale."),
        ("Associations", "figures/eda-target-pearson.pdf",
         "Feature-target Pearson correlations on the profiled sample, up to 12 features ranked by absolute association. These describe association, not model importance or sensitivity."),
        ("Associations", "figures/eda-target-correlation-ratio.pdf",
         "Feature-target correlation ratios on the profiled sample, up to 12 features ranked by association. These describe association, not model importance or sensitivity."),
        ("Associations", "figures/eda-target-cramers-v.pdf",
         "Feature-target Cramer's V on the profiled sample, up to 12 features ranked by association. These describe association, not model importance or sensitivity."),
        ("Associations", "figures/eda-correlation.pdf",
         "Pearson correlations for up to 10 numeric features on the profiled sample, selected by strongest absolute pairwise correlation. Each pair uses its available rows; grey cells are unavailable or outside the capped pair list. Categorical and mixed pairs are excluded."),
        ("Associations", "figures/eda-sensitivity.pdf",
         "Sensitivity weights from the latest matching completed sensitivity job, up to 12 features ranked by absolute weight. Values describe a surrogate over generated input ranges, not feature-target correlation or causal effects."),
    ):
        if name in figures:
            inserts.setdefault((section, None), []).extend(figure_latex(name, caption))
    association_notes = inserts.setdefault(("Associations", None), [])
    if any("figures/eda-target-" in name or name in {
        "figures/eda-correlation.pdf", "figures/eda-sensitivity.pdf",
    } for name in figures):
        association_notes.append(r"\noindent Heatmap cell labels are rounded to two significant digits.\par")
    if "figures/eda-correlation.pdf" not in figures:
        association_notes.append(r"\noindent No Pearson correlation heatmap is available for this partition.\par")
    if sensitivity:
        protocol = sensitivity["evaluation_protocol"]
        excluded = ", ".join(item["column"] for item in protocol["excluded_columns"]) or "none"
        detail = ("Method: %s; job: %s; submitted: %s; partition: %s; "
                  "dataset fingerprint: %s; rows used: %s; rows dropped: %s; excluded columns: %s."
                  % (sensitivity["method"], sensitivity["job_id"], sensitivity["created_at"],
                     protocol["train_partition"], protocol["train_fingerprint"], protocol["row_count"],
                     protocol["rows_dropped"], excluded))
        if "figures/eda-sensitivity.pdf" not in figures:
            detail += " No finite sensitivity weights are available to plot."
        association_notes.extend([r"\begin{sknote}[Sensitivity provenance]", _escape(detail), r"\end{sknote}"])
    else:
        association_notes.extend([
            r"\begin{sknote}[Sensitivity unavailable]",
            r"No completed sensitivity job matches this partition, dataset fingerprint and active features. "
            r"Run sensitivity analysis for this partition to include its heatmap.",
            r"\end{sknote}",
        ])
    return report_latex(
        markdown, task, "Statistical and Exploratory Data Analysis",
        "Dataset %s; partition %s" % (task, partition), generated_at,
        inserts=inserts,
    )


def _figure_paths(figures):
    for name, content in (figures or {}).items():
        if not re.fullmatch(r"figures/[a-z0-9-]+\.pdf", name):
            raise ValueError("Invalid report figure path: %r" % name)
        yield name, content


def source_archive(source: str, markdown: str, *, data_json: str | None = None, figures=None) -> bytes:
    """Package the report source and its style assets without retaining data."""
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("report.tex", source)
        output.writestr("report.md", markdown)
        if data_json is not None:
            output.writestr("results.json", data_json)
        for name, content in _figure_paths(figures):
            output.writestr(name, content)
        output.writestr(
            "README.txt",
            "Build report.tex twice with LuaLaTeX. The SKSurrogate style requires "
            "Inter, Inter Display, and Fira Mono or Fira Code; Libertinus Math or "
            "Latin Modern Math; and the TeX packages used by sk.sty.\n",
        )
        for name in ("sk.sty", "sk-brand.sty", "logo/sk-mark.tikz", "logo/cover-surface.tikz"):
            output.write(_TEMPLATE / name, name)
    return archive.getvalue()


def compile_report(source: str, *, figures=None) -> bytes:
    """Compile a PDF in a temporary directory, without retaining report data."""
    executable = shutil.which("lualatex")
    if executable is None:
        raise ReportCompilationError("LuaLaTeX is not installed or is not on PATH.")
    with tempfile.TemporaryDirectory(prefix="sksurrogate-report-") as directory:
        root = Path(directory)
        shutil.copytree(_TEMPLATE, root, dirs_exist_ok=True)
        (root / "report.tex").write_text(source, encoding="utf-8")
        for name, content in _figure_paths(figures):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        command = [
            executable, "-no-shell-escape", "-interaction=nonstopmode",
            "-halt-on-error", "report.tex",
        ]
        try:
            for _ in range(2):
                result = subprocess.run(
                    command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env={**os.environ, "TEXMFVAR": str(root / "texmf-var")},
                    timeout=120, check=False,
                )
                if result.returncode:
                    logger.error("LaTeX report compilation failed: %s", result.stdout.decode("utf-8", "replace")[-8000:])
                    raise ReportCompilationError("LaTeX compilation failed; check the server logs.")
        except subprocess.TimeoutExpired as exc:
            raise ReportCompilationError("LaTeX compilation timed out.") from exc
        return (root / "report.pdf").read_bytes()
