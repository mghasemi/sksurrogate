"""Cost model for the v2.0 benchmark matrix (DESIGN.md §16.4).

The full v2.0 matrix is compute-bounded, not code-bounded: the pilot's job is to
measure what one cell costs before anyone launches hundreds of them. This module
measures the two additive components of an SRS cell and combines them with the
matrix size:

1. **Candidate evaluation** — one ``StratifiedKFold(5)`` fit of the tuned
   estimator on a dataset. Independent of the parameter-space dimension; measured
   per (estimator family, dataset).
2. **Surrogate overhead** — one Gaussian-process fit on ``n`` evaluated points
   plus one bounded L-BFGS-B acquisition step, measured per parameter-space
   dimension and per evaluation count. Only surrogate arms pay it, and only for
   the evaluations after engagement.

The extrapolation is deliberately explicit about its assumptions: it multiplies
per-component medians, not a fitted model. It is a planning instrument, not a
result; no benchmark claim depends on it.

Usage::

    python -m benchmarks.cost_model --datasets ionosphere,credit-g \
        --out benchmarks/DESIGN-v2-cost.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .datasets import load_dataset

REPEATS = 3


@dataclass(frozen=True)
class Family:
    """One estimator family in the v2.0 matrix, with a representative candidate."""

    name: str
    make: Callable[[], object]
    space_dim: int
    note: str = ""


def _families() -> dict[str, Family]:
    return {
        "rf": Family(
            "rf",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", RandomForestClassifier(n_estimators=50, random_state=0)),
                ]
            ),
            3,
            "v1.9 anchor",
        ),
        "svc": Family(
            "svc",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel="rbf", C=1.0, gamma="scale")),
                ]
            ),
            3,
            "kernel, scale-sensitive",
        ),
        "hgb": Family(
            "hgb",
            lambda: Pipeline(
                [
                    ("model", HistGradientBoostingClassifier(max_iter=100, random_state=0)),
                ]
            ),
            5,
            "trees, binning",
        ),
        "logreg": Family(
            "logreg",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000)),
                ]
            ),
            3,
            "linear",
        ),
    }


def _median_seconds(fn, repeats: int = REPEATS) -> float:
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - started)
    return float(np.median(timings))


def measure_evaluation(family: Family, X, y, *, cv: int = 5) -> float:
    """One candidate-model evaluation (5-fold CV) in seconds, median of repeats."""
    splitter = StratifiedKFold(n_splits=cv, shuffle=False)

    def run():
        scores = cross_val_score(
            family.make(), X, y, cv=splitter, scoring="accuracy", n_jobs=1
        )
        if not np.all(np.isfinite(scores)):  # pragma: no cover - data dependent
            raise RuntimeError("non-finite CV score in cost probe")

    return _median_seconds(run)


def _surrogate_point(dim: int, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, dim))
    y = np.sin(X.sum(axis=1)) + 0.05 * rng.normal(size=n)
    return X, y


def measure_surrogate(dim: int, n: int, *, seed: int = 0) -> dict[str, float]:
    """GP fit and one bounded acquisition step, in seconds, median of repeats."""
    from sklearn.gaussian_process import GaussianProcessRegressor

    X, y = _surrogate_point(dim, n, seed)

    def fit_only():
        GaussianProcessRegressor().fit(X, y)

    fit_seconds = _median_seconds(fit_only)

    gp = GaussianProcessRegressor().fit(X, y)
    bounds = [(-1.0, 1.0)] * dim

    def one_acquisition():
        x0 = np.zeros(dim)

        def objective(point):
            mean, std = gp.predict(point.reshape(1, -1), return_std=True)
            # Lower confidence bound is what the shipped search minimizes.
            return float(mean[0] - 0.1 * std[0])

        minimize(objective, x0, bounds=bounds, method="L-BFGS-B")

    acquisition_seconds = _median_seconds(one_acquisition)
    return {
        "fit_s": fit_seconds,
        "acquisition_s": acquisition_seconds,
        "n_points": n,
        "dim": dim,
    }


def _dataset_family_assignment():
    """Which estimator family each v2.0 dataset-cell is planned against.

    Mirrors §16.2: the four real datasets anchor four families, with RF kept as
    the v1.9 anchor everywhere.
    """
    return {
        "ionosphere": ["rf", "svc", "logreg"],
        "credit-g": ["rf", "svc", "hgb"],
        "spambase": ["hgb", "logreg"],
        "optdigits": ["svc", "hgb"],
        # Fallback anchor when credit-g is unavailable (bundled, no download).
        "galaxy3": ["rf", "hgb"],
    }


def estimate_matrix(
    evaluation_s: dict[tuple[str, str], float],
    surrogate_s: dict[int, dict[str, float]],
    *,
    datasets=(),
    budgets=(30, 60, 120),
    seeds: int = 7,
    arms: int = 9,
    surrogate_arms: int = 4,
    engagement_floor: int = 20,
) -> dict:
    """Extrapolate the full matrix from measured component medians.

    Assumptions are explicit because this is a planning estimate: every real-data
    cell spends its full budget on candidate evaluations; surrogate arms add one
    fit+acquisition per post-engagement evaluation; the synthetic-function matrix
    is costed as a separate, cheaper block. The expensive-objective arm is costed
    at 1 s/evaluation as the protocol specifies.
    """
    rows = []
    for dataset in datasets:
        for family in _dataset_family_assignment().get(dataset, ["rf"]):
            key = (dataset, family)
            if key not in evaluation_s:
                continue
            per_eval = evaluation_s[key]
            for budget in budgets:
                guided = max(0, budget - engagement_floor)
                overhead = (
                    guided
                    * (
                        surrogate_s[3]["fit_s"]
                        + surrogate_s[3]["acquisition_s"]
                    )
                    if family
                    else 0.0
                )
                per_run = budget * per_eval + (overhead if family else 0.0)
                rows.append(
                    {
                        "dataset": dataset,
                        "family": family,
                        "budget": budget,
                        "per_eval_s": round(per_eval, 4),
                        "per_surrogate_run_s": round(per_run, 2),
                    }
                )
    # Non-surrogate arms (random, TPE, skopt, RandomizedSearchCV, flaml) cost only
    # candidate evaluations; the SRS arms pay the overhead above.
    total_real = 0.0
    for row in rows:
        total_real += row["per_surrogate_run_s"] * surrogate_arms * seeds
        total_real += (
            row["budget"] * row["per_eval_s"] * max(arms - surrogate_arms, 0) * seeds
        )
    # High-dimensional continuous cells (§16.2c): parameter dimension is the only
    # change; candidate evaluation cost is the dataset anchor's, while the
    # surrogate pays the d=10/30 fit+acquisition measured above. Assume 2 dims x
    # 4 datasets x 3 budgets x 6 arms (4 surrogate + 2 random).
    total_high_dim = 0.0
    for dim in (10, 30):
        if dim not in surrogate_s:
            continue
        per_eval = evaluation_s.get((datasets[0] if datasets else "ionosphere", "rf"), 0.01)
        for budget in budgets:
            guided = max(0, budget - engagement_floor)
            per_run = budget * per_eval + guided * (
                surrogate_s[dim]["fit_s"] + surrogate_s[dim]["acquisition_s"]
            )
            total_high_dim += per_run * 4 * len(budgets) * seeds * surrogate_arms
            total_high_dim += budget * per_eval * 4 * len(budgets) * seeds * 2
    total_real += total_high_dim
    # Synthetic matrix: 6 functions x 3 seeds x 2 budgets x 6 arms, evaluated at
    # the cheap RF family cost (objective evaluation is milliseconds; the search
    # overhead dominates and is bounded by the same surrogate measurements).
    synthetic_per_eval = evaluation_s.get((datasets[0] if datasets else "ionosphere", "rf"), 0.01)
    synthetic_evals = 6 * 3 * (50 + 100) * 6
    total_synthetic = synthetic_evals * max(synthetic_per_eval, 0.005)
    # One expensive-objective arm in the design: 1 s/eval, 4 datasets x 7 seeds x 2 budgets.
    expensive = 1.0 * 4 * seeds * (30 + 60) * 2
    return {
        "rows": rows,
        "real_matrix_s": total_real,
        "high_dim_s": total_high_dim,
        "synthetic_s": total_synthetic,
        "expensive_arm_s": expensive,
        "total_s": total_real + total_synthetic + expensive,
        "assumptions": {
            "budgets": list(budgets),
            "seeds": seeds,
            "arms": arms,
            "surrogate_arms": surrogate_arms,
            "engagement_floor": engagement_floor,
            "synthetic_functions": 6,
            "expensive_objective_s_per_eval": 1.0,
        },
    }


def _machine_info() -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpus": __import__("os").cpu_count(),
        "stamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def render_markdown(evaluation_s, surrogate_s, estimate, datasets) -> str:
    lines = [
        "# Benchmark v2.0 cost model",
        "",
        "Status: **measured component costs + explicit extrapolation** (not a benchmark result).",
        "",
        "Machine: `%s`" % json.dumps(_machine_info(), sort_keys=True),
        "",
        "## Candidate evaluation (5-fold CV, median of %d)" % REPEATS,
        "",
        "| dataset | family | seconds / evaluation |",
        "|---|---|---:|",
    ]
    for (dataset, family), seconds in sorted(evaluation_s.items()):
        lines.append("| %s | %s | %.4f |" % (dataset, family, seconds))
    lines.extend(
        [
            "",
            "## Surrogate overhead (GP fit + one bounded LCB step, median of %d)" % REPEATS,
            "",
            "| parameter dim | evaluated points | fit (s) | acquisition (s) |",
            "|---:|---:|---:|---:|",
        ]
    )
    for dim, row in sorted(surrogate_s.items()):
        lines.append(
            "| %d | %d | %.4f | %.4f |"
            % (dim, row["n_points"], row["fit_s"], row["acquisition_s"])
        )
    lines.extend(
        [
            "",
            "## Per-cell cost (surrogate arm)",
            "",
            "| dataset | family | budget | per-eval (s) | one surrogate run (s) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in estimate["rows"]:
        lines.append(
            "| %(dataset)s | %(family)s | %(budget)d | %(per_eval_s).4f | %(per_surrogate_run_s).2f |"
            % row
        )
    totals = estimate
    lines.extend(
        [
            "",
            "## Extrapolated full matrix",
            "",
            "| block | CPU-hours |",
            "|---|---:|",
            "| real-data matrix | %.2f |" % (totals["real_matrix_s"] / 3600.0),
            "| (of which high-dimensional cells) | %.2f |" % (totals["high_dim_s"] / 3600.0),
            "| synthetic functions | %.2f |" % (totals["synthetic_s"] / 3600.0),
            "| expensive-objective arm | %.2f |" % (totals["expensive_arm_s"] / 3600.0),
            "| **total (serial)** | **%.2f** |" % (totals["total_s"] / 3600.0),
            "",
            "Assumptions: %s" % json.dumps(totals["assumptions"], sort_keys=True),
            "",
            "Wall-clock scales with the number of parallel workers; this table is serial CPU time.",
            "",
            "## Gate",
            "",
            "**No full matrix runs until the owner records a go/no-go decision here.**",
            "",
            "Decision: _(pending)_",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="ionosphere,credit-g")
    parser.add_argument("--out", default="benchmarks/DESIGN-v2-cost.md")
    parser.add_argument("--budgets", default="30,60,120")
    parser.add_argument("--seeds", type=int, default=7)
    parser.add_argument("--smoke", action="store_true", help="tiny measurement set")
    parser.add_argument(
        "--note",
        default="",
        help="markdown note appended verbatim (pilot substitutions, credentials, decisions)",
    )
    args = parser.parse_args(argv)

    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    budgets = tuple(int(value) for value in args.budgets.split(",") if value.strip())
    families = _families()

    evaluation_s = {}
    for dataset in datasets:
        X, y = load_dataset(dataset)
        for family_name in _dataset_family_assignment().get(dataset, ["rf"]):
            evaluation_s[(dataset, family_name)] = measure_evaluation(
                families[family_name], X, y
            )
            print(
                "[%s/%s] %.4f s/eval" % (dataset, family_name, evaluation_s[(dataset, family_name)]),
                flush=True,
            )

    dims = (3,) if args.smoke else (3, 6, 10, 30)
    points = (30,) if args.smoke else (30, 60)
    surrogate_s = {}
    for dim in dims:
        for n in points:
            surrogate_s[dim] = measure_surrogate(dim, n)
            print("[surrogate d=%d n=%d] %s" % (dim, n, surrogate_s[dim]), flush=True)
            break  # fit+acquisition at a single n; dim is the cost driver

    estimate = estimate_matrix(
        evaluation_s,
        surrogate_s,
        datasets=datasets,
        budgets=budgets,
        seeds=args.seeds,
    )
    destination = Path(args.out)
    document = render_markdown(evaluation_s, surrogate_s, estimate, datasets)
    if args.note:
        document += "\n## Notes\n\n" + args.note.strip() + "\n"
    destination.write_text(document + "\n", encoding="utf-8")
    print("wrote %s" % destination)
    print("estimated serial hours: %.2f" % (estimate["total_s"] / 3600.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
