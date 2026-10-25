"""D-track analysis: turn raw D1/D2 CSVs into regime-map tables + figures + a
Markdown report.

Reads ``benchmarks/results/dtrack/d1/d1_aggregate.csv`` and
``.../d2/d2_aggregate.csv`` and writes:

- ``docs/benchmarks/dtrack.md``  — the D-track report (regime map, findings)
- ``docs/benchmarks/figures/dtrack_*.png`` — convergence, stability, noise,
  and regime-map figures

Usage::

    python -m benchmarks.dtrack.analyze
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
D1_CSV = REPO_ROOT / "benchmarks" / "results" / "dtrack" / "d1" / "d1_aggregate.csv"
D2_CSV = REPO_ROOT / "benchmarks" / "results" / "dtrack" / "d2" / "d2_aggregate.csv"
FIG_DIR = REPO_ROOT / "docs" / "benchmarks" / "figures"
OUT_MD = REPO_ROOT / "docs" / "benchmarks" / "dtrack.md"

SURROGATES = ["hilbert", "gp", "rf"]
SURROGATE_LABEL = {"hilbert": "Hilbert(deg=3)", "gp": "GP(RBF)", "rf": "RF(100)"}


def _loglog_slope(ns, rmses):
    """Least-squares slope of log(RMSE) vs log(n) (negative = convergence)."""
    ns = np.asarray(ns, dtype=float)
    rmses = np.asarray(rmses, dtype=float)
    mask = (ns > 0) & (rmses > 0) & np.isfinite(rmses)
    if mask.sum() < 2:
        return float("nan")
    x = np.log(ns[mask])
    y = np.log(rmses[mask])
    return float(np.polyfit(x, y, 1)[0])


def analyze_d1(df: pd.DataFrame) -> str:
    ok = df[df["ok"]].copy()
    md = ["## D1 — Hilbert approximation behavior\n"]
    md.append("### Convergence rate vs sample size (log-log slope of RMSE vs n)\n")
    md.append("| family | dim | deg | slope |")
    md.append("|--------|-----|-----|-------|")
    conv = ok[(ok["sigma"] == 0.0) & (ok["seed"] == 0)].sort_values("n")
    for (fam, d, deg), sub in conv.groupby(["family", "dim", "deg"]):
        slope = _loglog_slope(sub["n"], sub["rmse"])
        md.append(f"| {fam} | {d} | {deg} | {slope:.2f} |")
    md.append("")

    md.append("### Small-n stability (mean +/- std RMSE, repeated draws)\n")
    md.append("| family | dim | deg | n | mean | std | fail_rate |")
    md.append("|--------|-----|-----|---|------|-----|-----------|")
    small = ok[ok["n"].isin([10, 20, 30])]
    for (fam, d, deg, n), sub in small.groupby(["family", "dim", "deg", "n"]):
        md.append(f"| {fam} | {d} | {deg} | {n} | {sub['rmse'].mean():.4f} | "
                  f"{sub['rmse'].std():.4f} | {1.0 - sub['ok'].mean():.2f} |")
    md.append("")

    md.append("### Noise robustness (RMSE at n=100 vs noise sigma)\n")
    md.append("| family | dim | deg | sigma=0 | sigma=0.01 | sigma=0.05 | sigma=0.1 |")
    md.append("|--------|-----|-----|---------|------------|------------|-----------|")
    noise = ok[ok["n"] == 100]
    for (fam, d, deg), sub in noise.groupby(["family", "dim", "deg"]):
        row = {sig: sub[sub["sigma"] == sig]["rmse"].mean() for sig in [0.0, 0.01, 0.05, 0.1]}
        md.append(f"| {fam} | {d} | {deg} | {row[0.0]:.4f} | {row[0.01]:.4f} | "
                  f"{row[0.05]:.4f} | {row[0.1]:.4f} |")
    md.append("")
    return "\n".join(md)


def analyze_d2(df: pd.DataFrame) -> str:
    ok = df[df["ok"]].copy()
    md = ["## D2 — Surrogate head-to-head + regime map\n"]
    md.append("Mean RMSE by (family, dim, n, sigma, surrogate); **bold** = winner.\n")
    g = ok.groupby(["family", "dim", "n", "sigma", "surrogate"])["rmse"].mean().reset_index()
    g["winner"] = g.groupby(["family", "dim", "n", "sigma"])["rmse"].transform("min")
    g["is_winner"] = np.isclose(g["rmse"], g["winner"])

    md.append("| family | dim | n | sigma | hilbert | gp | rf | winner |")
    md.append("|--------|-----|---|-------|---------|----|----|--------|")
    for (fam, d, n, sig), sub in g.groupby(["family", "dim", "n", "sigma"]):
        vals = {s: sub[sub["surrogate"] == s]["rmse"].iloc[0] for s in SURROGATES}
        win = sub[sub["is_winner"]]["surrogate"].tolist()
        win_s = ",".join(SURROGATE_LABEL[w] for w in win)
        md.append(f"| {fam} | {d} | {n} | {sig:.2f} | {vals['hilbert']:.4f} | "
                  f"{vals['gp']:.4f} | {vals['rf']:.4f} | {win_s} |")
    md.append("")

    # win counts by family
    md.append("### Winner counts by objective family\n")
    md.append("| family | hilbert | gp | rf |")
    md.append("|--------|---------|----|----|")
    for fam, sub in g.groupby("family"):
        wins = sub[sub["is_winner"]]["surrogate"].value_counts()
        md.append(f"| {fam} | {wins.get('hilbert', 0)} | {wins.get('gp', 0)} | {wins.get('rf', 0)} |")
    md.append("")
    return "\n".join(md)


def make_figures(df1: pd.DataFrame, df2: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    made = []

    # D1 convergence: RMSE vs n (log-log) for deg=3, clean, per family/dim
    ok1 = df1[(df1["ok"]) & (df1["deg"] == 3) & (df1["sigma"] == 0.0) & (df1["seed"] == 0)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, d in zip(axes, [1, 2, 3]):
        for fam, sub in ok1[ok1["dim"] == d].groupby("family"):
            sub = sub.sort_values("n")
            ax.plot(sub["n"], sub["rmse"], marker="o", label=fam)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"dim={d}"); ax.set_xlabel("n (log)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("RMSE (log)")
    axes[0].legend(fontsize=8)
    fig.suptitle("D1: Hilbert(deg=3) convergence — RMSE vs sample size (clean)")
    fig.tight_layout()
    p = FIG_DIR / "dtrack_d1_convergence.png"
    fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))

    # D2 regime map heatmap: winner per (family, dim, n, sigma)
    ok2 = df2[df2["ok"]]
    g = ok2.groupby(["family", "dim", "n", "sigma", "surrogate"])["rmse"].mean().reset_index()
    g["winner"] = g.groupby(["family", "dim", "n", "sigma"])["rmse"].transform("min")
    g["is_winner"] = np.isclose(g["rmse"], g["winner"])
    win = g[g["is_winner"]].copy()
    win["cell"] = win["family"] + " d" + win["dim"].astype(str) + " n" + win["n"].astype(str) + \
        " s" + win["sigma"].astype(str)
    # pivot winner -> surrogate code
    wmap = win.set_index("cell")["surrogate"].to_dict()
    cells = sorted(wmap.keys())
    codes = {"hilbert": 0, "gp": 1, "rf": 2}
    mat = np.array([[codes[wmap[c]] for c in cells]])
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(cells)), 2.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=2)
    ax.set_xticks(range(len(cells))); ax.set_xticklabels(cells, rotation=90, fontsize=7)
    ax.set_yticks([])
    cbar = fig.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Hilbert", "GP", "RF"])
    ax.set_title("D2 regime map: winning surrogate per (family, dim, n, sigma)")
    fig.tight_layout()
    p = FIG_DIR / "dtrack_d2_regime_map.png"
    fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))

    return made


def main() -> int:
    df1 = pd.read_csv(D1_CSV)
    df2 = pd.read_csv(D2_CSV)
    md = ["# D-track: Hilbert-space surrogate study\n"]
    md.append(f"_Generated {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')} from "
              f"`benchmarks/results/dtrack/` (raw dirs git-ignored)._\n")
    md.append(analyze_d1(df1))
    md.append(analyze_d2(df2))
    figs = make_figures(df1, df2)
    md.append("## Figures\n")
    for f in figs:
        md.append(f"- `{Path(f).name}`")
    md.append("")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print("figures:", *[Path(f).name for f in figs], sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
