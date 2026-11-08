"""Controlled synthetic data-generating processes with known ground truth (Stage D1).

This module is the Stage D1 deliverable of ``AI_LAYER02_REMAINING_PLAN.md``
(item R6): a small seeded set of *parametric* data-generating processes whose
structure, coefficients, and noise model are exactly known, plus a validation
harness that scores a method panel against them and writes a versioned JSON
report. See ``benchmarks/DESIGN.md`` section 14 for the contract.

Relationship to ``SKSurrogate/synthdat.py`` (the "scaffolding" cited by R6):
``synthdat.SynthData`` synthesizes data that *resembles an existing dataset*
by resampling its marginal or joint structure — it has no ground-truth
function and exposes no seed. That is the wrong tool for method validation,
where one needs a specified process with exact truth values (and seeded
repeatability). This module therefore extends the synthetic-data capability
family rather than reusing ``SynthData``'s resampling machinery: it follows
the same conventions (typed DataFrame output, named feature columns) and is
intended to sit alongside it as the "known-process" half of the family. A
future categorical variant can map values through ``synthdat.SynthCat`` items;
none of the current DGPs need it.

Design rules (inherited from the suite):

- Every draw is seeded via ``numpy.random.default_rng(seed)`` and bit-
  reproducible across processes within a fixed NumPy version.
- The harness is *descriptive*: repeated draws produce point estimates of
  error; nothing here makes significance or generalization claims (see the
  cross-cutting rules in ``AI_LAYER02_REMAINING_PLAN.md``).
- Report records carry an explicit schema version so future readers can be
  tolerant, mirroring the benchmark run-record convention.

Run directly::

    python -m benchmarks.dgps --dgps interactions shift --seeds 0 1 --out DIR
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

#: Report schema version for the validation JSON written by :func:`validate`.
REPORT_SCHEMA_VERSION = "sksurrogate.dgp-validation.v1"

# ---------------------------------------------------------------------------
# Shared DGP constants (fixed ground-truth parameters)
# ---------------------------------------------------------------------------

#: Baseline additive noise standard deviation shared by all DGPs.
NOISE_SIGMA = 0.5

#: Fixed intercept used by every DGP's target function.
INTERCEPT = 0.5

_GAMMA_INTERACTION = 1.5  # interactions: coefficient of the x0*x1 term

_RHO = 0.9  # correlated: within-pair correlation
_RHO_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (3, 4))

_SIGMA0_HET = 0.5  # heteroscedastic: base noise scale
_SCALE_C_HET = 2.0  # heteroscedastic: |x2| growth of the noise scale
_HET_DEPENDS_ON = "x2"

_KAPPA_QUAD = 1.0  # shift: coefficient of the x2**2 term
_SHIFT_MU = 2.0  # shift: mean offset applied to every test feature


def _feature_names(p: int) -> list[str]:
    return [f"x{i}" for i in range(p)]


@dataclass(frozen=True)
class DGPSample:
    """One seeded draw from a named DGP, with train/test/iid splits and truth.

    ``X_test``/``y_test`` are the evaluation set; for the ``shift`` DGP they
    come from the *shifted* distribution, for all others from the same process
    as training. ``X_iid``/``y_iid`` always come i.i.d. from the training
    distribution and serve as the in-distribution reference (degradation
    baseline) regardless of DGP.
    """

    name: str
    seed: int
    p: int
    n_train: int
    n_test: int
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    X_iid: pd.DataFrame
    y_iid: np.ndarray
    truth: Mapping[str, Any]


# ---------------------------------------------------------------------------
# DGP definitions (private generators + public registry)
# ---------------------------------------------------------------------------

def _linear_target(X: pd.DataFrame, beta: dict[str, float]) -> np.ndarray:
    """INTERCEPT + sum of named coefficients over columns of ``X``."""
    y = np.full(len(X), INTERCEPT, dtype=float)
    for name, c in beta.items():
        y += c * X[name].to_numpy()
    return y


def _dgp_interactions(rng: np.random.Generator, n: int, p: int):
    """y = 0.5 + 2.0*x0 - 1.5*x1 + 1.0*x2 + 1.5*(x0*x1) + eps, x ~ N(0, I).

    Nonlinear interaction with a sparse active set (3 of p features active);
    a linear model cannot represent the x0*x1 term exactly.
    """
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=_feature_names(p))
    beta = {"x0": 2.0, "x1": -1.5, "x2": 1.0}
    y = _linear_target(X, beta) + _GAMMA_INTERACTION * X["x0"].to_numpy() * X["x1"].to_numpy()
    y = y + rng.normal(0.0, NOISE_SIGMA, size=n)
    truth = {
        "kind": "interactions",
        "intercept": INTERCEPT,
        "linear_coefs": beta,
        "interaction_terms": ["x0", "x1"],
        "interaction_coef": _GAMMA_INTERACTION,
        "noise_sigma": NOISE_SIGMA,
    }
    return X, y, truth


def _dgp_correlated(rng: np.random.Generator, n: int, p: int):
    """y = 0.5 + beta . x + eps with block-correlated predictors.

    corr(x0, x1) = corr(x3, x4) = 0.9 (all other pairs independent); all six
    named coefficients are nonzero, so per-coefficient recovery is
    ill-conditioned even though prediction stays easy. Requires p >= 5.
    """
    if p < 5:
        raise ValueError("the correlated DGP requires p >= 5")
    cov = np.eye(p)
    for i, j in _RHO_PAIRS:
        cov[i, j] = cov[j, i] = _RHO
    X = pd.DataFrame(rng.multivariate_normal(np.zeros(p), cov, size=n), columns=_feature_names(p))
    beta = {"x0": 1.2, "x1": -0.8, "x2": 0.6, "x3": 1.0, "x4": -1.2, "x5": 0.4}
    y = _linear_target(X, beta) + rng.normal(0.0, NOISE_SIGMA, size=n)
    truth = {
        "kind": "correlated",
        "intercept": INTERCEPT,
        "linear_coefs": beta,
        "correlations": {f"{_feature_names(p)[i]}:{_feature_names(p)[j]}": _RHO for i, j in _RHO_PAIRS},
        "noise_sigma": NOISE_SIGMA,
    }
    return X, y, truth


def _dgp_heteroscedastic(rng: np.random.Generator, n: int, p: int):
    """y = 0.5 + beta . x + eps with noise scale depending on |x2|.

    sigma(x) = SIGMA0 * (1 + C * |x2|); a homoscedastic fit's residuals carry
    the heteroscedastic structure in their squared magnitudes. Requires
    p >= 3.
    """
    if p < 3:
        raise ValueError("the heteroscedastic DGP requires p >= 3")
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=_feature_names(p))
    beta = {"x0": 0.9, "x1": -0.7, "x2": 0.5, "x3": -0.4, "x4": 0.8, "x5": -0.3}
    y = _linear_target(X, beta)
    sigma = _SIGMA0_HET * (1.0 + _SCALE_C_HET * np.abs(X[_HET_DEPENDS_ON].to_numpy()))
    y = y + rng.normal(0.0, 1.0, size=n) * sigma
    truth = {
        "kind": "heteroscedastic",
        "intercept": INTERCEPT,
        "linear_coefs": beta,
        "noise": {"sigma0": _SIGMA0_HET, "scale_c": _SCALE_C_HET, "depends_on": _HET_DEPENDS_ON},
    }
    return X, y, truth


def _dgp_shift(rng: np.random.Generator, n: int, p: int):
    """y = 0.5 + beta . x + KAPPA * x2**2 + eps; train at N(0,I), test shifted.

    A quadratic term makes the covariate shift (mean +SHIFT_MU on every
    feature) consequential for models that cannot represent it: under the
    shifted distribution the missing x2**2 curvature correlates with x2 in a
    way absent at the origin. The conditional function itself is unchanged —
    this probes *model* degradation, not label shift.
    """
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=_feature_names(p))
    beta = {"x0": 1.2, "x1": -0.8, "x2": 0.6, "x3": 1.0, "x4": -1.2, "x5": 0.4}
    y = _linear_target(X, beta) + _KAPPA_QUAD * X["x2"].to_numpy() ** 2
    y = y + rng.normal(0.0, NOISE_SIGMA, size=n)
    truth = {
        "kind": "shift",
        "intercept": INTERCEPT,
        "linear_coefs": beta,
        "quadratic": {"term": "x2", "coef": _KAPPA_QUAD},
        "noise_sigma": NOISE_SIGMA,
        "test_mean_shift": _SHIFT_MU,
    }
    return X, y, truth


_DGP_GENERATORS = {
    "interactions": _dgp_interactions,
    "correlated": _dgp_correlated,
    "heteroscedastic": _dgp_heteroscedastic,
    "shift": _dgp_shift,
}

#: Public ordering for default runs.
KNOWN_DGPS: tuple[str, ...] = ("interactions", "correlated", "heteroscedastic", "shift")


def sample_dgp(
    name: str,
    n_train: int = 400,
    n_test: int = 200,
    p: int = 6,
    seed: int = 0,
) -> DGPSample:
    """Draw one seeded train/test/iid sample from DGP ``name``.

    Raises :class:`ValueError` for unknown names (listing the known ones).
    The training draw uses ``seed``; test and iid draws use fixed per-seed
    offsets so every part of a sample is reproducible from one seed value.
    """
    if name not in _DGP_GENERATORS:
        raise ValueError(f"unknown DGP {name!r}; known DGPs are: {', '.join(KNOWN_DGPS)}")
    gen = _DGP_GENERATORS[name]

    rng_train = np.random.default_rng(seed)
    X_train, y_train, truth = gen(rng_train, n_train, p)

    rng_test = np.random.default_rng(seed + 1_000_003)
    if name == "shift":
        # Test rows: same Gaussian feature draw shifted by the known mean offset.
        base = pd.DataFrame(
            rng_test.normal(size=(n_test, p)) + _SHIFT_MU * np.ones(p), columns=_feature_names(p)
        )
        y_test = (
            _linear_target(base, truth["linear_coefs"])
            + truth["quadratic"]["coef"] * base[truth["quadratic"]["term"]].to_numpy() ** 2
            + rng_test.normal(0.0, NOISE_SIGMA, size=n_test)
        )
        X_test = base
    else:
        X_test, y_test, _ = gen(rng_test, n_test, p)

    # In-distribution reference always drawn from the TRAINING process.
    rng_iid = np.random.default_rng(seed + 2_000_003)
    X_iid, y_iid, _ = gen(rng_iid, n_test, p)

    return DGPSample(
        name=name,
        seed=seed,
        p=p,
        n_train=n_train,
        n_test=n_test,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_iid=X_iid,
        y_iid=y_iid,
        truth=truth,
    )


# ---------------------------------------------------------------------------
# Method panel and validation harness
# ---------------------------------------------------------------------------

#: Default panel order for :func:`validate` and the CLI.
DEFAULT_METHODS: tuple[str, ...] = ("ridge", "random-forest", "kernel-ridge")


def _make_method(name: str, seed: int):
    """Instantiate a fresh estimator from its panel name (seeded where relevant)."""
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "random-forest":
        return RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=seed)
    if name == "kernel-ridge":
        return KernelRidge()
    raise ValueError(f"unknown method {name!r}; known methods are: {', '.join(DEFAULT_METHODS)}")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _ground_truth_checks(sample: DGPSample) -> dict[str, Any]:
    """Descriptive ground-truth probes specific to each DGP's kind.

    Each probe uses a plain ridge fit (alpha = 1) as the reference linear model
    and reports point estimates only — no significance claims.
    """
    Xtr = sample.X_train.to_numpy()
    checks: dict[str, Any] = {}

    if sample.name == "interactions":
        Xte = sample.X_test.to_numpy()
        plain = Ridge(alpha=1.0).fit(Xtr, sample.y_train)
        expanded_design_tr = np.column_stack([Xtr, Xtr[:, 0] * Xtr[:, 1]])
        expanded = Ridge(alpha=1.0).fit(expanded_design_tr, sample.y_train)
        gamma_hat = float(expanded.coef_[-1])
        checks["interaction_coef_abs_error"] = abs(gamma_hat - sample.truth["interaction_coef"])
        checks["interaction_coef_estimate"] = gamma_hat
        expanded_design_te = np.column_stack([Xte, Xte[:, 0] * Xte[:, 1]])
        checks["rmse_plain_ridge_test"] = _rmse(sample.y_test, plain.predict(Xte))
        checks["rmse_expanded_ridge_test"] = _rmse(sample.y_test, expanded.predict(expanded_design_te))

    elif sample.name == "correlated":
        fit = Ridge(alpha=1.0).fit(Xtr, sample.y_train)
        coefs = {f"x{i}": float(v) for i, v in enumerate(fit.coef_)}
        devs = [abs(coefs[nm] - c) for nm, c in sample.truth["linear_coefs"].items() if nm in coefs]
        checks["coef_mean_abs_deviation"] = float(np.mean(devs))
        checks["coef_max_abs_deviation"] = float(max(devs))

    elif sample.name == "heteroscedastic":
        fit = Ridge(alpha=1.0).fit(Xtr, sample.y_train)
        resid2_iid = (sample.y_iid - fit.predict(sample.X_iid.to_numpy())) ** 2
        noise_spec = sample.truth["noise"]
        sigma2 = (
            noise_spec["sigma0"]
            * (1.0 + noise_spec["scale_c"] * np.abs(sample.X_iid[noise_spec["depends_on"]].to_numpy()))
        ) ** 2
        checks["corr_squared_residuals_vs_predicted_variance"] = float(np.corrcoef(resid2_iid, sigma2)[0, 1])

    elif sample.name == "shift":
        # Per-method degradation ratios are reported by the harness itself.
        pass

    return checks


def validate(
    dgps: Sequence[str] = KNOWN_DGPS,
    methods: Sequence[str] = DEFAULT_METHODS,
    n_train: int = 400,
    n_test: int = 200,
    p: int = 6,
    seeds: Sequence[int] = (0, 1),
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Score each method on each DGP per seed and assemble a versioned report.

    All summaries are descriptive point estimates across the given seeds; no
    significance claims are made or implied. When ``out_dir`` is provided the
    report JSON (``dgp_validation.json``) is written there.
    """
    for name in dgps:
        if name not in _DGP_GENERATORS:
            raise ValueError(f"unknown DGP {name!r}; known DGPs are: {', '.join(KNOWN_DGPS)}")
    for m in methods:
        _make_method(m, 0)  # raises early on unknown method names

    results: dict[str, Any] = {}
    started = time.time()
    for name in dgps:
        per_seed: dict[str, Any] = {}
        for seed in seeds:
            sample = sample_dgp(name, n_train=n_train, n_test=n_test, p=p, seed=seed)
            row: dict[str, Any] = {"truth": sample.truth}
            row["methods"] = {}
            for m in methods:
                model = _make_method(m, seed)
                model.fit(sample.X_train.to_numpy(), sample.y_train)
                y_test_pred = model.predict(sample.X_test.to_numpy())
                y_iid_pred = model.predict(sample.X_iid.to_numpy())
                rmse_test = _rmse(sample.y_test, y_test_pred)
                rmse_iid = _rmse(sample.y_iid, y_iid_pred)
                row["methods"][m] = {
                    "rmse_test": rmse_test,
                    "r2_test": _r2(sample.y_test, y_test_pred),
                    "rmse_iid": rmse_iid,
                    "degradation_ratio": (rmse_test / rmse_iid) if rmse_iid > 0 else float("nan"),
                }
            row["ground_truth_checks"] = _ground_truth_checks(sample)
            per_seed[str(seed)] = row
        results[name] = {"per_seed": per_seed}

    # Cross-seed means for the common method metrics (descriptive convenience).
    for name, entry in results.items():
        methods_summary: dict[str, Any] = {}
        for m in methods:
            keys = ("rmse_test", "r2_test", "rmse_iid", "degradation_ratio")
            methods_summary[m] = {
                k: float(np.mean([entry["per_seed"][s]["methods"][m][k] for s in entry["per_seed"]]))
                for k in keys
            }
        results[name]["mean_across_seeds"] = {"methods": methods_summary}

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": n_train,
        "n_test": n_test,
        "p": p,
        "seeds": list(seeds),
        "methods": {m: type(_make_method(m, 0)).__name__ for m in methods},
        "results": results,
        "wall_seconds": round(time.time() - started, 2),
        "note": (
            "Descriptive validation on controlled synthetic data with known ground truth; "
            "no significance or generalization claims. See benchmarks/DESIGN.md section 14."
        ),
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "dgp_validation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate methods against controlled synthetic DGPs (Stage D1).")
    parser.add_argument("--dgps", nargs="+", choices=list(KNOWN_DGPS), default=list(KNOWN_DGPS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--p", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument(
        "--out",
        default=str(Path("benchmarks/results/dgps") / stamp),
        help="output directory for dgp_validation.json (default: benchmarks/results/dgps/<UTC stamp>)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = validate(
        dgps=args.dgps,
        methods=args.methods,
        n_train=args.n_train,
        n_test=args.n_test,
        p=args.p,
        seeds=args.seeds,
        out_dir=args.out,
    )

    print(f"{'dgp':<16} {'method':<14} {'rmse_test':>10} {'r2_test':>9} {'rmse_iid':>9} {'degrad.':>8}")
    for name in args.dgps:
        summary = report["results"][name]["mean_across_seeds"]["methods"]
        for m in args.methods:
            s = summary[m]
            print(
                f"{name:<16} {m:<14} {s['rmse_test']:>10.4f} {s['r2_test']:>9.3f} "
                f"{s['rmse_iid']:>9.4f} {s['degradation_ratio']:>8.2f}"
            )
    print(f"\nreport written to: {Path(args.out) / 'dgp_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
