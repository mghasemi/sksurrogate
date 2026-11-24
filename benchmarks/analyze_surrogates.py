"""Surrogate-regressor comparison (DESIGN.md v1.8).

Compares the response-surface arms of the SRS benchmark — default cubic Hilbert
(``sksurrogate``), ``GaussianProcessRegressor`` (``sksurrogate-gp``) and
``KernelRidge`` (``sksurrogate-krr``) — against each other and the standard
baselines, at equal budget. Records may come from several results directories
(canonical suites + the v1.8 arms run); within a given dataset/budget/seed the
records are merged by method name.

Usage::

    python -m benchmarks.analyze_surrogates DIR [DIR ...] [--budget 30] \
        [--abs-gap 0.01] [--out PATH.csv]

Prints: (a) per-dataset mean final-CV / test-accuracy table with the primary
gap metric; (b) seed-level win/tie/loss of each new arm vs the Hilbert arm on
final CV; (c) writes a long-format CSV next to the first input directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .plot import _load_records, evals_to_target
from .stats import holm_correction, paired_report_row


def load_merged(dirs: list[Path], budget: int) -> list[dict]:
    """Load ok-status records from all dirs; drop (dataset, seed, method) duplicates."""
    seen: set[tuple[str, str, int]] = set()
    merged: list[dict] = []
    for d in dirs:
        for r in _load_records(d):
            if r.get("budget") != budget:
                continue
            key = (r["dataset"], str(r["seed"]), r["method"])
            if key in seen:
                continue  # first directory wins (pass canonical dirs first)
            seen.add(key)
            merged.append(r)
    return merged


def build_table(records: list[dict], abs_gap: float) -> pd.DataFrame:
    rows = []
    for r in records:
        scored = [float(x["cv_score"]) for x in r["records"] if x.get("cv_score") is not None]
        k = evals_to_target(r, mode="gap", abs_gap=abs_gap)
        rows.append(
            {
                "dataset": r["dataset"],
                "seed": r["seed"],
                "method": r["method"],
                "final_cv_best": max(scored) if scored else np.nan,
                "evals_to_target_g01": k if k is not None else float(r["budget"]) + 1.0,
                "test_acc": r.get("final_test_accuracy"),
                "surrogate": (r.get("extra") or {}).get("surrogate", ""),
            }
        )
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame, abs_gap: float) -> pd.DataFrame:
    agg = table.groupby(["dataset", "method"]).agg(
        mean_cv=("final_cv_best", "mean"),
        sd_cv=("final_cv_best", "std"),
        mean_test_acc=("test_acc", "mean"),
        median_evals_to_target=("evals_to_target_g01", "median"),
    ).reset_index()
    return agg.sort_values(["dataset", "method"])


def win_loss(table: pd.DataFrame, method_new: str, method_ref: str) -> dict[str, int]:
    """Seed-level comparison of final CV scores on shared (dataset, seed) pairs."""
    a = table[table["method"] == method_new].set_index(["dataset", "seed"])["final_cv_best"]
    b = table[table["method"] == method_ref].set_index(["dataset", "seed"])["final_cv_best"]
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).dropna()
    return {
        "n": int(len(d)),
        "wins": int((d > 1e-9).sum()),
        "ties": int((np.abs(d) <= 1e-9).sum()),
        "losses": int((d < -1e-9).sum()),
    }


def wall_best(record: dict, wall_budget: float):
    """Best CV score reached within ``wall_budget`` seconds of candidate time.

    Uses the sum of per-evaluation ``wall_s`` (the search's own candidate time,
    excluding the final held-out evaluation). Returns ``None`` when the record has
    no per-evaluation timing (e.g. the archived skopt arm), because inventing a
    wall-clock position would be worse than reporting the arm as unavailable.
    """
    cumulative = 0.0
    best = None
    for entry in record.get("records") or []:
        seconds = entry.get("wall_s")
        if seconds is None:
            return None
        cumulative += float(seconds)
        if cumulative > wall_budget:
            break
        score = entry.get("cv_score")
        if score is not None:
            best = float(score) if best is None else max(best, float(score))
    return best


def build_wall_table(records: list[dict], wall_budget: float) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "dataset": record["dataset"],
                "seed": record["seed"],
                "method": record["method"],
                "wall_best_cv": wall_best(record, wall_budget),
                "total_wall_s": record.get("total_wall_s"),
            }
        )
    return pd.DataFrame(rows)


def paired_stats(table: pd.DataFrame, value_column: str, baseline: str) -> pd.DataFrame:
    """Paired per-dataset and aggregate rows for every method against ``baseline``.

    Each row is one method's paired comparison on the shared (dataset, seed)
    pairs: mean delta, 95% percentile bootstrap CI, W/T/L, Wilcoxon p, Cliff's
    delta, and a descriptive ``distinguishable`` flag. The aggregate row pools
    every dataset pair; it is the interpretable quantity (see DESIGN §16.3), and
    the Holm correction is applied across methods on the aggregate p-values.
    """
    if baseline not in set(table["method"]):
        return pd.DataFrame()
    pivot = table.pivot_table(
        index=["dataset", "seed"], columns="method", values=value_column
    )
    if baseline not in pivot.columns:
        return pd.DataFrame()
    rows = []
    aggregate_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in pivot.columns:
        if method == baseline:
            continue
        pair = pivot[[baseline, method]].dropna()
        if len(pair) < 2:
            continue
        for dataset, block in pair.groupby(level=0):
            row = paired_report_row(
                method,
                block[method].to_numpy(),
                block[baseline].to_numpy(),
            )
            rows.append({"scope": dataset, "method": method, **row})
        aggregate_pairs[method] = (pair[method].to_numpy(), pair[baseline].to_numpy())
    # Aggregate: every (dataset, seed) pair pooled; Cliff's delta and the
    # bootstrap CI are computed on the raw paired scores, not on the deltas.
    for method, (treatment, control) in aggregate_pairs.items():
        if len(treatment) < 2:
            continue
        row = paired_report_row(method, treatment, control)
        rows.append({"scope": "aggregate", "method": method, **row})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    aggregate = frame[frame["scope"] == "aggregate"]
    if not aggregate.empty:
        holm = holm_correction(list(aggregate["wilcoxon_p"]))
        indexed = aggregate.set_index("method")
        indexed["p_holm"] = holm["adjusted"]
        indexed["reject_holm"] = holm["reject"]
        frame.loc[frame["scope"] == "aggregate", "p_holm"] = indexed["p_holm"].to_numpy()
        frame.loc[frame["scope"] == "aggregate", "reject_holm"] = indexed["reject_holm"].to_numpy()
    return frame


def plot_curves(records: list[dict], out_dir: Path, methods: list[str]) -> None:
    """Seed-averaged best-so-far curves for the requested methods (DESIGN v1.8 figure)."""
    from .plot import _best_sofar_curve

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)
    colors = {
        "sksurrogate": "#d62728",
        "sksurrogate-gp": "#17becf",
        "sksurrogate-krr": "#e377c2",
        "random": "#1f77b4",
        "optuna-tpe": "#2ca02c",
    }
    for dataset in sorted({r["dataset"] for r in records}):
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
        for method in methods:
            if method not in per_method or method not in colors:
                continue
            curves = np.vstack(per_method[method])
            mean_curve = np.nanmean(curves, axis=0)
            xs = np.arange(1, len(mean_curve) + 1)
            ax.plot(xs, mean_curve, label=f"{method} (n={len(curves)} seeds)", color=colors[method], lw=2)
        ax.set_xlabel("Model evaluations")
        ax.set_ylabel("Best-so-far CV accuracy")
        ax.set_title(f"Surrogate surface comparison — {dataset}")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(figures / f"{dataset}_surrogate_comparison.png")
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", help="results directories (canonical first for dedupe priority)")
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--abs-gap", type=float, default=0.01)
    parser.add_argument("--out", default=None, help="output CSV path (default: <first dir>/surrogate_comparison.csv)")
    parser.add_argument(
        "--currency",
        choices=("evals", "wall"),
        default="evals",
        help="primary table currency: evaluation budget (default) or wall-clock budget",
    )
    parser.add_argument(
        "--wall-budget",
        type=float,
        default=60.0,
        help="candidate-time budget in seconds for --currency wall (default 60)",
    )
    parser.add_argument(
        "--baseline",
        default="sksurrogate-gp",
        help="method every paired comparison is against (default sksurrogate-gp)",
    )
    parser.add_argument(
        "--paired-out",
        default=None,
        help="paired-statistics CSV path (default: <last dir>/paired_<currency>_B<budget>.csv)",
    )
    args = parser.parse_args(argv)

    dirs = [Path(d) for d in args.dirs]
    records = load_merged(dirs, budget=args.budget)
    if not records:
        print(f"no ok-status records with budget={args.budget} found")
        return 1

    table = build_table(records, abs_gap=args.abs_gap)
    agg = summarize(table, args.abs_gap)

    # Artifacts go to the LAST directory (the new arms' run), never into a sealed
    # canonical baseline passed first for dedupe priority.
    out_base = dirs[-1]
    out_path = Path(args.out) if args.out else out_base / f"surrogate_comparison_B{args.budget}.csv"
    agg.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    print(f"\n=== B={args.budget} | mean final CV (sd) / test acc / median evals to gap {args.abs_gap:g} ===")
    show = agg.pivot_table(index="dataset", columns="method", values=["mean_cv", "mean_test_acc"])
    for metric in ["mean_cv", "mean_test_acc"]:
        print(f"\n-- {metric} --")
        print(show[metric].round(4).to_string())

    gap = agg.pivot_table(index="dataset", columns="method", values="median_evals_to_target")
    print("\n-- median evals within abs-gap of own final best (secondary, self-referential; DESIGN sec. 6) --")
    print(gap.round(1).to_string())

    if args.currency == "wall":
        wall_table = build_wall_table(records, args.wall_budget)
        print(
            f"\n=== wall-clock budget {args.wall_budget:g}s of candidate time | "
            "best CV reached within the budget ==="
        )
        wall_summary = wall_table.groupby(["dataset", "method"]).agg(
            mean_wall_best=("wall_best_cv", "mean"),
            median_wall_best=("wall_best_cv", "median"),
            mean_total_wall_s=("total_wall_s", "mean"),
        )
        print(wall_summary.round(4).to_string())
        paired_input = wall_table.rename(columns={"wall_best_cv": "final_cv_best"})
        currency = "wall"
    else:
        paired_input = table
        currency = "evals"

    paired = paired_stats(paired_input, "final_cv_best", args.baseline)
    if not paired.empty:
        print(f"\n=== paired vs {args.baseline} ({currency}) — aggregate is the interpretable row ===")
        display_columns = [
            column
            for column in (
                "scope", "method", "n_pairs", "mean_delta", "ci_low", "ci_high",
                "wins", "ties", "losses", "wilcoxon_p", "cliffs_delta",
                "distinguishable", "p_holm", "reject_holm",
            )
            if column in paired.columns
        ]
        print(paired[display_columns].to_string(index=False))
        paired_path = (
            Path(args.paired_out)
            if args.paired_out
            else out_base / f"paired_{currency}_B{args.budget}.csv"
        )
        paired.to_csv(paired_path, index=False)
        print(f"\npaired CSV: {paired_path}")

    print("\n=== seed-level vs Hilbert arm (final CV) ===")
    ref = "sksurrogate"
    for new in ["sksurrogate-gp", "sksurrogate-krr"]:
        if new not in set(table["method"]):
            continue
        wl = win_loss(table, new, ref)
        print(f"{new} vs {ref}: n={wl['n']}  wins={wl['wins']} ties={wl['ties']} losses={wl['losses']}")

    plot_curves(records, out_base, methods=[ref, "sksurrogate-gp", "sksurrogate-krr", "random"])
    print(f"\nCSV: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
