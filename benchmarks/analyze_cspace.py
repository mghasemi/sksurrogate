"""Analysis for the pipeline-space benchmark (DESIGN-cspace.md §5–§6).

Inputs are ``benchmarks/results/cspace-*`` directories written by
:mod:`benchmarks.run_cspace`. Outputs are markdown tables and best-so-far figures;
the numbers are computed from per-cell records, never re-derived from aggregates.

Statistics follow DESIGN-cspace.md §6 exactly:

* unit of analysis is the ``(dataset, seed)`` cell, paired across methods;
* primary test is the two-sided exact sign test with ties *excluded and counted*;
* Wilcoxon signed-rank as a sensitivity check;
* Cliff's delta and a bootstrap 95% CI of the mean paired difference;
* Holm-Bonferroni within the family "each arm vs the reference arm", per dataset;
* cross-dataset inference is a sign test over per-dataset means, explicitly
  descriptive at n = 6 datasets (no Friedman/Nemenyi claim).

Usage::

    python -m benchmarks.analyze_cspace DIR [DIR ...] --protocol B --ref srs-cspace
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

__all__ = [
    "load_records",
    "best_sofar",
    "time_averaged_best",
    "evals_to_gap",
    "paired_statistics",
    "holm",
    "group_means",
    "win_loss",
    "main",
]

DEFAULT_GAP = 0.01
BOOTSTRAP_DRAWS = 10_000


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_records(dirs: Iterable[Path]) -> list[dict[str, Any]]:
    """All per-cell records from one or more result directories."""
    records: list[dict[str, Any]] = []
    for directory in dirs:
        directory = Path(directory)
        manifest = directory / "run_manifest.json"
        protocol = None
        if manifest.exists():
            protocol = json.loads(manifest.read_text()).get("protocol")
        for path in sorted(directory.glob("*.json")):
            if path.name == "run_manifest.json":
                continue
            payload = json.loads(path.read_text())
            payload.setdefault("protocol", protocol)
            payload["_dir"] = directory.name
            records.append(payload)
    return records


def _is_ok(record: dict) -> bool:
    return record.get("status") == "ok" and record.get("final_test_accuracy") is not None


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #

def _curve(record: dict) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative time and running-best CV for one run."""
    history = record.get("records") or []
    times, bests = [], []
    running = -np.inf
    elapsed = 0.0
    for item in history:
        elapsed += float(item.get("wall_s") or 0.0)
        score = item.get("cv_score")
        if score is not None and score > running:
            running = float(score)
        times.append(elapsed)
        bests.append(running if np.isfinite(running) else np.nan)
    return np.asarray(times), np.asarray(bests)


def best_sofar(record: dict) -> tuple[np.ndarray, np.ndarray]:
    """(cumulative time, running best) with leading NaNs dropped."""
    times, bests = _curve(record)
    if times.size == 0:
        return times, bests
    first = int(np.argmax(np.isfinite(bests)))
    return times[first:], bests[first:]


def time_averaged_best(record: dict, horizon: float) -> Optional[float]:
    """Trapezoid mean of best-so-far CV over ``[0, horizon]``.

    Higher is better and comparable across arms that ran a different number of
    evaluations: an arm that finds a good pipeline early scores high even if it
    finished fewer evaluations.
    """
    times, bests = best_sofar(record)
    if times.size == 0:
        return None
    times = np.concatenate([[0.0], times, [horizon]])
    bests = np.concatenate([[bests[0]], bests, [bests[-1]]])
    times = np.clip(times, 0.0, horizon)
    order = np.argsort(times)
    return float(np.trapezoid(bests[order], times[order]) / horizon)


def evals_to_gap(record: dict, gap: float = DEFAULT_GAP) -> Optional[float]:
    """Smallest k whose running best is within ``gap`` of the run's own final best."""
    history = [item for item in (record.get("records") or []) if item.get("cv_score") is not None]
    if not history:
        return None
    final = max(float(item["cv_score"]) for item in history)
    running = -np.inf
    for item in history:
        running = max(running, float(item["cv_score"]))
        if final - running <= gap:
            return float(item["k"])
    return None


# --------------------------------------------------------------------------- #
# Paired statistics
# --------------------------------------------------------------------------- #

def _bootstrap_ci(differences: np.ndarray, draws: int = BOOTSTRAP_DRAWS, seed: int = 20261026) -> tuple[float, float]:
    if differences.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(differences, size=(draws, differences.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) - P(a < b) over all pairs (0 = no effect, ±1 = complete dominance)."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjustment within one family of comparisons."""
    items = [(name, p) for name, p in pvalues.items() if p is not None and np.isfinite(p)]
    items.sort(key=lambda pair: pair[1])
    adjusted: dict[str, float] = {}
    m = len(items)
    for rank, (name, p) in enumerate(items):
        adjusted[name] = float(min(1.0, p * (m - rank)))
    for name, p in pvalues.items():
        adjusted.setdefault(name, float("nan"))
    return adjusted


def paired_statistics(
    table: pd.DataFrame, reference: str, metric: str = "final_test_accuracy"
) -> pd.DataFrame:
    """Per-dataset paired comparison of every method against ``reference``.

    ``table`` needs ``dataset``, ``seed``, ``method``, and ``metric`` columns.
    """
    from scipy import stats

    rows = []
    for dataset, frame in table.groupby("dataset"):
        pivot = frame.pivot_table(index="seed", columns="method", values=metric, aggfunc="mean")
        if reference not in pivot.columns:
            continue
        for method in pivot.columns:
            if method == reference:
                continue
            pair = pivot[[method, reference]].dropna()
            if pair.empty:
                continue
            differences = (pair[method] - pair[reference]).to_numpy(dtype=float)
            wins = int(np.sum(differences > 1e-12))
            losses = int(np.sum(differences < -1e-12))
            ties = int(len(differences) - wins - losses)
            decisive = wins + losses
            sign_p = float(stats.binomtest(wins, decisive, 0.5).pvalue) if decisive else float("nan")
            try:
                wilcoxon_p = float(stats.wilcoxon(differences, zero_method="wilcox").pvalue)
            except ValueError:
                wilcoxon_p = float("nan")
            low, high = _bootstrap_ci(differences)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "reference": reference,
                    "n_pairs": int(len(differences)),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "decisive": decisive,
                    "mean_diff": float(np.mean(differences)),
                    "ci_low": low,
                    "ci_high": high,
                    "cliffs_delta": _cliffs_delta(
                        pair[method].to_numpy(dtype=float), pair[reference].to_numpy(dtype=float)
                    ),
                    "sign_p": sign_p,
                    "wilcoxon_p": wilcoxon_p,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    adjusted = {}
    for _, group in result.groupby("dataset"):
        adjusted.update(holm(dict(zip(group["method"], group["sign_p"]))))
    result["sign_p_holm"] = result["method"].map(adjusted)
    return result


def win_loss(table: pd.DataFrame, method: str, reference: str, metric: str = "final_test_accuracy") -> dict[str, int]:
    """Seed-level win/tie/loss counts pooled across datasets."""
    wins = ties = losses = 0
    for _, frame in table.groupby("dataset"):
        pivot = frame.pivot_table(index="seed", columns="method", values=metric, aggfunc="mean")
        if method not in pivot or reference not in pivot:
            continue
        difference = (pivot[method] - pivot[reference]).dropna()
        wins += int((difference > 1e-12).sum())
        losses += int((difference < -1e-12).sum())
        ties += int((difference.abs() <= 1e-12).sum())
    return {"wins": wins, "ties": ties, "losses": losses}


def group_means(table: pd.DataFrame, metric: str = "final_test_accuracy") -> tuple[pd.DataFrame, pd.DataFrame]:
    means = table.pivot_table(index="dataset", columns="method", values=metric, aggfunc="mean")
    stds = table.pivot_table(index="dataset", columns="method", values=metric, aggfunc="std")
    return means.round(4), stds.round(4)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _markdown(frame: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    if frame.empty:
        return "_(empty)_"
    header = "| " + " | ".join(str(col) for col in frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    lines = [header, divider]
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("" if not np.isfinite(value) else floatfmt.format(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _coverage(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        extra = record.get("extra") or {}
        rows.append(
            {
                "dataset": record.get("dataset"),
                "method": record.get("method"),
                "n_evals": record.get("n_evals"),
                "wall_s": record.get("total_wall_s"),
                "invalid_rate": extra.get("invalid_rate"),
                "duplicate_rate": extra.get("duplicate_rate"),
                "engaged_at": extra.get("engaged_at"),
                "n_surface_fits": extra.get("n_surface_fits"),
                "best_clf": (record.get("best_params") or {}).get("clf"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.groupby("method", dropna=False).agg(
        n_cells=("n_evals", "size"),
        median_evals=("n_evals", "median"),
        median_wall_s=("wall_s", "median"),
        mean_invalid=("invalid_rate", "mean"),
        mean_duplicate=("duplicate_rate", "mean"),
        engaged_cells=("n_surface_fits", lambda series: int((series.fillna(0) > 0).sum())),
    ).round(4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--protocol", choices=["A", "B", "C"], default=None, help="filter to one protocol")
    parser.add_argument("--ref", default="srs-cspace", help="reference arm for paired tests")
    parser.add_argument("--abs-gap", type=float, default=DEFAULT_GAP)
    parser.add_argument("--out", type=Path, default=None, help="write a markdown report here")
    args = parser.parse_args(argv)

    records = load_records(args.dirs)
    if args.protocol:
        records = [record for record in records if record.get("protocol") == args.protocol]
    ok = [record for record in records if _is_ok(record)]
    if not ok:
        print("no successful records")
        return 1

    # One horizon for every arm, so time-averaged best-so-far is comparable
    # (per-run total_wall would give a slower arm a longer, easier integration window).
    horizon = max(float(record.get("total_wall_s") or 0.0) for record in ok)
    table = pd.DataFrame(
        [
            {
                "dataset": record.get("dataset"),
                "seed": record.get("seed"),
                "method": record.get("method"),
                "protocol": record.get("protocol"),
                "final_test_accuracy": record.get("final_test_accuracy"),
                "final_test_balanced": record.get("final_test_balanced"),
                "final_cv_best": record.get("final_cv_best"),
                "evals_to_gap": evals_to_gap(record, args.abs_gap),
                "time_avg_best": time_averaged_best(record, horizon=horizon),
            }
            for record in ok
        ]
    )

    means, stds = group_means(table)
    paired = paired_statistics(table, args.ref)
    coverage = _coverage(ok)
    pooled = {
        method: win_loss(table, method, args.ref)
        for method in sorted(table["method"].unique())
        if method != args.ref
    }

    chunks = [
        f"# C-space results ({args.protocol or 'all protocols'})\n",
        f"Reference arm: `{args.ref}`. Unit of analysis: (dataset, seed); "
        f"{table['seed'].nunique()} seeds × {table['dataset'].nunique()} datasets.\n",
        "## Mean final test accuracy\n",
        _markdown(means.reset_index()),
        "\n## Standard deviation across seeds\n",
        _markdown(stds.reset_index()),
        "\n## Paired comparison vs reference (per dataset)\n",
        _markdown(paired[["dataset", "method", "n_pairs", "wins", "ties", "losses", "mean_diff",
                          "ci_low", "ci_high", "cliffs_delta", "sign_p", "sign_p_holm"]]),
        "\n## Pooled seed-level win/tie/loss vs reference\n",
        _markdown(pd.DataFrame([{"method": k, **v} for k, v in pooled.items()])),
        "\n## Eval efficiency and budget accounting\n",
        _markdown(
            table.groupby("method").agg(
                median_evals_to_gap=("evals_to_gap", "median"),
                mean_time_avg_best=("time_avg_best", "mean"),
            ).round(4).reset_index()
        ),
        "\n## Coverage, validity, engagement\n",
        _markdown(coverage.reset_index()),
    ]
    report = "\n".join(chunks)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
