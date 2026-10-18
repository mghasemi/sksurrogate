"""Aggregation and plotting for benchmark results (DESIGN.md section 5).

Reads a results directory produced by :mod:`benchmarks.run` and writes, next to
the raw data:

* ``figures/<dataset>_best_sofar.png`` — seed-averaged best-so-far CV curves per method;
* ``target_table.csv`` / ``target_table.md`` — median evaluations needed to reach
  95% of each run's own final best CV score (the primary metric), with the share
  of seeds that converged within budget.

Usage::

    python -m benchmarks.plot <results_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: A run counts as "converged" once its running-best CV score reaches this
#: fraction of the run's own final best CV score (DESIGN.md section 6: 95%).
TARGET_FRACTION = 0.95


def _load_records(results_dir: Path) -> list[dict]:
    """Load per-run JSON files (the jsonl log is a fallback for crash recovery)."""
    records = []
    json_files = sorted(results_dir.glob("*__*.json"))
    if not json_files and (results_dir / "results.jsonl").exists():
        with open(results_dir / "results.jsonl", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
    else:
        for path in json_files:
            with open(path, encoding="utf-8") as fh:
                records.append(json.loads(fh.read()))
    return [r for r in records if r.get("status") == "ok"]


def _best_sofar_curve(record: dict) -> np.ndarray | None:
    """Running-best CV score across evaluations (None if nothing completed)."""
    out = []
    running = -np.inf
    started = False
    for rec in record["records"]:
        s = rec.get("cv_score")
        if s is not None and np.isfinite(s):
            running = max(running, float(s))
            started = True
        out.append(np.nan if not started else running)
    return np.array(out) if started else None


def evals_to_target(record: dict, fraction: float = TARGET_FRACTION) -> float | None:
    """First evaluation index k where running-best >= fraction * final best.

    Returns ``None`` when the run never reaches the target within budget.
    """
    scored = [rec for rec in record["records"] if rec.get("cv_score") is not None and np.isfinite(rec["cv_score"])]
    if not scored:
        return None
    final_best = max(float(rec["cv_score"]) for rec in scored)
    target = fraction * final_best

    running = -np.inf
    for rec in record["records"]:
        s = rec.get("cv_score")
        if s is not None and np.isfinite(s):
            running = max(running, float(s))
            if running >= target:
                return float(rec["k"])
    return None


def build_target_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        k = evals_to_target(record)
        scored = [float(r["cv_score"]) for r in record["records"] if r.get("cv_score") is not None]
        final_best = max(scored) if scored else np.nan
        rows.append(
            {
                "dataset": record["dataset"],
                "seed": record["seed"],
                "method": record["method"],
                "budget": record["budget"],
                "final_cv_best": final_best,
                "evals_to_target": k if k is not None else float(record["budget"]) + 1.0,
                "converged_within_budget": k is not None,
            }
        )
    return pd.DataFrame(rows)


def _plot_curves(records: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)

    datasets = sorted({r["dataset"] for r in records})
    methods_order = ["sksurrogate", "srs-no-surrogate", "random", "optuna-tpe", "skopt"]
    colors = {
        "sksurrogate": "#d62728",
        "srs-no-surrogate": "#ff7f0e",
        "random": "#1f77b4",
        "optuna-tpe": "#2ca02c",
        "skopt": "#9467bd",
    }

    for dataset in datasets:
        per_method: dict[str, list[np.ndarray]] = {}
        budget_max = 0
        for record in records:
            if record["dataset"] != dataset:
                continue
            curve = _best_sofar_curve(record)
            if curve is None:
                continue
            per_method.setdefault(record["method"], []).append(curve)
            budget_max = max(budget_max, len(curve))

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
        for method in methods_order:
            if method not in per_method:
                continue
            curves = np.vstack(per_method[method])
            mean_curve = np.nanmean(curves, axis=0)
            xs = np.arange(1, len(mean_curve) + 1)
            ax.plot(xs, mean_curve, label=f"{method} (n={len(curves)} seeds)", color=colors.get(method), lw=2)

        ax.set_xlabel("Model evaluations")
        ax.set_ylabel("Best-so-far CV accuracy")
        ax.set_title(f"Equal-budget search comparison — {dataset}")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        out_path = figures / f"{dataset}_best_sofar.png"
        fig.savefig(out_path)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    if not argv or len(argv) < 1:
        print(__doc__)
        return 2
    results_dir = Path(argv[0])
    if not results_dir.is_dir():
        print(f"error: {results_dir} is not a directory", file=sys.stderr)
        return 2

    records = _load_records(results_dir)
    if not records:
        print("no completed runs found in results dir")
        return 1

    table = build_target_table(records)
    table.to_csv(results_dir / "target_table.csv", index=False)

    # Aggregate: median evals-to-target per (method, dataset).
    agg = (
        table.groupby(["dataset", "method"])["evals_to_target"]
        .agg(median="median", converged_share=lambda s: float(np.mean(s <= table.loc[s.index, "budget"])))
        .reset_index()
    )

    md_lines = [
        "| dataset | method | median evals to 95%-of-best (budget B) | share of seeds converged within budget |",
        "|---|---|---|---|",
    ]
    for row in agg.itertuples():
        md_lines.append(
            f"| {row.dataset} | {row.method} | {row.median:.1f} / {int(table.loc[(table['dataset'] == row.dataset) & (table['method'] == row.method), 'budget'].iloc[0])} "
            f"| {row.converged_share:.2f} |"
        )
    md_lines.append("")
    md_lines.append(f"Primary metric: first evaluation index whose running-best CV score reaches {TARGET_FRACTION:.0%} of the run's own final best; values > budget mean not converged within budget.")

    (results_dir / "target_table.md").write_text("\n".join(md_lines), encoding="utf-8")
    _plot_curves(records, results_dir)

    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
