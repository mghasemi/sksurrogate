"""Arm runners for the pipeline-space benchmark (DESIGN-cspace.md §3–§4).

Every arm searches the *same* frozen C-space (``benchmarks.cspace``) on the *same*
precomputed CV folds, under one of two currencies:

* **Protocol A** — equal wall-clock ``wall`` seconds (primary);
* **Protocol B** — exactly ``budget`` candidate evaluations (secondary).

Records use :class:`benchmarks.methods.RunRecord` so the existing analysis and
plotting conventions carry over unchanged.

Deliberate implementation choices (each one is a fairness guarantee):

``min_evals=20`` for every SRS arm
    ``SurrogateSearch`` raises ``MinEvals`` to ``C(p+3, p)`` when it installs its
    own default regressor — 364 at this space's 11 coordinates — so the surrogate
    would never fire and the arm would silently be random search. Passing an
    explicit ``regressor=`` skips that clamp; passing an explicit ``min_evals``
    makes engagement identical across the three SRS arms.

``max_iter=4*budget`` / ``10_000`` and ``max_evals`` as the binding cap
    ``max_iter`` counts loop iterations and ``x0`` is evaluated in ``__init__``;
    failed local optimisations of the response surface consume iterations without
    evaluating anything (DESIGN.md v1.1/v1.7). With ``max_iter == budget`` a run
    can stop early; the evaluation cap must be the only binding constraint.

precomputed fold list
    One ``list[tuple[np.ndarray, np.ndarray]]`` per cell is passed to
    ``cross_val_score(..., cv=splits)`` and to ``SurrogateRandomCV(cv=...)``, so
    fold membership cannot differ between arms.

``threadpool_limits(1)``
    Wall-clock is a primary axis; SVC/LogReg scale with BLAS threads while tree
    ensembles do not.

``_EngagementProbe``
    Wraps the response surface to record the evaluation index at which the
    surrogate actually fires (Gate G4). ``SurrogateRandomCV`` does not expose it.
"""

from __future__ import annotations

import contextlib
import math
import random as _pyrandom
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from benchmarks.cspace import (
    CLF_GRID,
    to_sklearn,
    to_srs_space,
    srs_categorical_columns,
    srs_conditional,
    sample_uniform,
)
from benchmarks.methods import RunRecord
from benchmarks.pipelines import MixedCategoricalSurface, PipelineFactory

__all__ = [
    "METHODS",
    "PROTOCOLS",
    "train_test_split_frame",
    "run_random_cspace",
    "run_optuna_tpe_cspace",
    "run_optuna_rand_cspace",
    "run_srs_cspace",
    "run_srs_onehot_cspace",
    "run_srs_nosurr_cspace",
    "run_flaml",
    "run_autogluon",
]

#: Minimum number of evaluations before the response surface is fitted, identical
#: for all three SRS arms (DESIGN-cspace.md §4).
SRS_MIN_EVALS = 20


# --------------------------------------------------------------------------- #
# Shared machinery
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def _one_thread():
    """Pin BLAS/OpenMP to one thread for the duration of a fit (fairness)."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl ships with sklearn
        yield
        return
    with threadpool_limits(limits=1):
        yield


class _EngagementProbe(RegressorMixin, BaseEstimator):
    """Transparent wrapper that records when the response surface is fitted.

    :param regressor: the real surface (GP, mixed categorical surface, ...)
    :param state: mutable dict shared with the runner; ``state["evals"]`` is the
        number of candidate evaluations completed so far.
    """

    def __init__(self, regressor=None, state=None, counter=None):
        self.regressor = regressor
        self.state = state
        self.counter = counter

    def fit(self, X, y):
        state = self.state if self.state is not None else {}
        if "engaged_at" not in state:
            # ``counter`` is wired to ``len(search.evaluation_history_)`` after the
            # search object exists, so this is the true evaluation index at which
            # the surface first fires, not the configured threshold.
            state["engaged_at"] = int(self.counter()) if self.counter is not None else int(state.get("evals", 0))
        state["n_fits"] = state.get("n_fits", 0) + 1
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, y)
        return self

    def predict(self, X):
        return self.regressor_.predict(X)


def train_test_split_frame(X, y, seed: int):
    """One stratified 80/20 split per (dataset, seed), shared by every arm."""
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


def cv_splits(y, seed: int, n_splits: int = 5):
    """Precomputed stratified folds — identical for every arm on a given cell."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return splitter, list(splitter.split(np.zeros(len(y)), y))


def _evaluate(cfg: dict, Xtr, ytr, splits, seed: int) -> float:
    """One candidate = one 5-fold CV fit of the concrete pipeline."""
    model = to_sklearn(cfg, random_state=seed)
    with _one_thread():
        scores = cross_val_score(model, Xtr, ytr, cv=splits, scoring="accuracy", n_jobs=1)
    return float(np.mean(scores))


def _final_test_scores(cfg: dict, Xtr, ytr, Xte, yte, seed: int):
    """Refit the chosen configuration once and score it on the held-out split."""
    if not cfg:
        return None, None
    model = to_sklearn(cfg, random_state=seed)
    with _one_thread():
        model.fit(Xtr, ytr)
        prediction = model.predict(Xte)
    return float(accuracy_score(yte, prediction)), float(balanced_accuracy_score(yte, prediction))


def _record(
    method: str,
    seed: int,
    budget: Optional[int],
    cfg: dict,
    records: list[dict],
    total_wall: float,
    seeds_row: dict,
    Xtr, ytr, Xte, yte,
    extra: dict,
) -> RunRecord:
    completed = [item for item in records if item.get("cv_score") is not None]
    best = max(completed, key=lambda item: item["cv_score"]) if completed else None
    best_cfg = dict(cfg)
    test_acc, test_bal = _final_test_scores(best_cfg, Xtr, ytr, Xte, yte, seed)
    families = [item.get("family") for item in records if item.get("family")]
    extra = dict(extra)
    extra.setdefault("budget_evals", budget)
    extra.setdefault("budget_wall_s", None)
    extra["family_mix"] = {name: families.count(name) for name in sorted(set(families))}
    extra["invalid"] = sum(1 for item in records if item.get("cv_score") is None)
    extra["invalid_rate"] = (extra["invalid"] / len(records)) if records else None
    keys = [tuple(sorted((k, str(v)) for k, v in item["_cfg"].items())) for item in records]
    extra["duplicate_rate"] = (len(keys) - len(set(keys))) / len(keys) if keys else None
    return RunRecord(
        method=method,
        dataset="",  # filled by the caller
        seed=seed,
        budget=int(budget) if budget is not None else 0,
        n_estimators=50,
        status="ok" if best is not None else "error",
        error=None if best is not None else "no completed evaluations",
        n_evals=len(records),
        records=[{k: v for k, v in item.items() if not k.startswith("_")} for item in records],
        best_params=best_cfg,
        final_cv_best=best["cv_score"] if best is not None else None,
        final_test_accuracy=test_acc,
        final_test_balanced=test_bal,
        total_wall_s=total_wall,
        extra={**seeds_row, **extra},
    )


def _item(k: int, score: Optional[float], wall: Optional[float], cfg: dict, status: str = "complete"):
    return {
        "k": k,
        "cv_score": score,
        "wall_s": wall,
        "status": status,
        "family": cfg.get("clf"),
        "_cfg": cfg,
    }


# --------------------------------------------------------------------------- #
# Protocol A/B: search-strategy arms
# --------------------------------------------------------------------------- #

def run_random_cspace(Xtr, ytr, Xte, yte, *, budget=None, wall=None, seed=0) -> RunRecord:
    """Uniform random sampling of valid configs (search control)."""
    rng = np.random.default_rng(seed)
    _, splits = cv_splits(ytr, seed)
    records: list[dict] = []
    started = time.perf_counter()
    if budget is None and wall is None:
        raise ValueError("random-cspace needs either budget or wall")
    while True:
        elapsed = time.perf_counter() - started
        if budget is not None and len(records) >= budget:
            break
        if budget is None and wall is not None and elapsed >= wall:
            break
        cfg = sample_uniform(rng)
        t0 = time.perf_counter()
        try:
            score = _evaluate(cfg, Xtr, ytr, splits, seed)
        except Exception as exc:  # noqa: BLE001 - record, never abort the suite
            records.append(_item(len(records), None, time.perf_counter() - t0, cfg, f"failed: {exc.__class__.__name__}"))
            continue
        records.append(_item(len(records), score, time.perf_counter() - t0, cfg))
    total = time.perf_counter() - started
    completed = [item for item in records if item["cv_score"] is not None]
    best_cfg = dict(max(completed, key=lambda item: item["cv_score"])["_cfg"]) if completed else {}
    return _record("random-cspace", seed, budget, best_cfg, records, total,
                   {"protocol": "A" if budget is None else "B", "budget_wall_s": wall},
                   Xtr, ytr, Xte, yte, {})


def _run_optuna(Xtr, ytr, Xte, yte, *, budget, wall, seed, sampler_kind: str, method: str) -> RunRecord:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from benchmarks.cspace import optuna_suggest

    if sampler_kind == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed)
    else:
        sampler = optuna.samplers.RandomSampler(seed=seed)

    _, splits = cv_splits(ytr, seed)
    records: list[dict] = []
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        cfg = optuna_suggest(trial)
        t0 = time.perf_counter()
        try:
            score = _evaluate(cfg, Xtr, ytr, splits, seed)
        except Exception:  # noqa: BLE001 - Optuna records the failure as a trial state
            records.append(_item(len(records), None, time.perf_counter() - t0, cfg, "failed"))
            raise optuna.TrialPruned() from None
        records.append(_item(len(records), score, time.perf_counter() - t0, cfg))
        return score

    started = time.perf_counter()
    study.optimize(objective, n_trials=budget if budget is not None else 10**6, timeout=wall, show_progress_bar=False)
    total = time.perf_counter() - started

    if budget is not None and len(study.trials) != budget:
        return RunRecord(
            method=method, dataset="", seed=seed, budget=int(budget), n_estimators=50,
            status="error",
            error=f"trial-count parity violated: {len(study.trials)} != {budget}",
            total_wall_s=total,
        )

    completed = [item for item in records if item["cv_score"] is not None]
    best_cfg = max(completed, key=lambda item: item["cv_score"])["_cfg"] if completed else {}
    return _record(method, seed, budget, best_cfg, records, total,
                   {"protocol": "A" if budget is None else "B"}, Xtr, ytr, Xte, yte,
                   {"sampler": sampler_kind})


def run_optuna_tpe_cspace(Xtr, ytr, Xte, yte, *, budget=None, wall=None, seed=0) -> RunRecord:
    """Optuna TPE with native conditional branching (HPO baseline)."""
    return _run_optuna(Xtr, ytr, Xte, yte, budget=budget, wall=wall, seed=seed,
                       sampler_kind="tpe", method="optuna-tpe-cspace")


def run_optuna_rand_cspace(Xtr, ytr, Xte, yte, *, budget=None, wall=None, seed=0) -> RunRecord:
    """Optuna RandomSampler — separates "TPE is smart" from "Optuna is a loop"."""
    return _run_optuna(Xtr, ytr, Xte, yte, budget=budget, wall=wall, seed=seed,
                       sampler_kind="random", method="optuna-rand-cspace")


def _run_srs(
    Xtr, ytr, Xte, yte, *, budget, wall, seed, method: str, regressor_kind: str, min_evals: int,
) -> RunRecord:
    from SKSurrogate.structsearch import SurrogateRandomCV

    cv, _ = cv_splits(ytr, seed)
    state: dict[str, Any] = {"evals": 0}

    def make_surface():
        if regressor_kind == "mixed":
            return MixedCategoricalSurface(
                categorical=srs_categorical_columns(), embedding="circular"
            )
        from sklearn.gaussian_process import GaussianProcessRegressor

        return GaussianProcessRegressor()

    surface = _EngagementProbe(regressor=make_surface(), state=state)
    task_name = f"cspace_{method.replace('-', '_')}_s{seed}_{int(time.time())}"
    search = SurrogateRandomCV(
        PipelineFactory(random_state=seed),
        to_srs_space(),
        max_iter=(4 * budget) if budget is not None else 10_000,
        max_evals=budget,
        min_evals=min_evals,
        regressor=surface,           # explicit surface => no C(p+3,p) engagement clamp
        conditional=srs_conditional(),
        scoring="accuracy",
        cv=cv,
        refit=False,
        verbose=0,
        random_state=seed,
        n_jobs=1,
        warm_start=False,
        Continue=False,
        task_name=task_name,
        time_limit=wall,
    )
    # SRS draws x0 from CPython's module-level RNG (ignoring random_state) and
    # checkpoints to <task_name>.pkl in the CWD; pin the former, remove the latter.
    surface.counter = lambda: len(search.evaluation_history_)
    saved_state = _pyrandom.getstate()
    _pyrandom.seed(1_000_003 + seed)
    started = time.perf_counter()
    try:
        with _one_thread():
            search.fit(Xtr, ytr)
    finally:
        _pyrandom.setstate(saved_state)
        checkpoint = Path(task_name + ".pkl")
        if checkpoint.exists():
            checkpoint.unlink()
    total = time.perf_counter() - started

    records: list[dict] = []
    for index, item in enumerate(search.evaluation_history_):
        raw = item.get("score")
        score = None if raw is None else float(raw)
        records.append(
            _item(index, score, float(item.get("duration", 0.0) or 0.0),
                  dict(item.get("params", {})),
                  "complete" if score is not None else (item.get("status") or "failed"))
        )

    completed = [item for item in records if item["cv_score"] is not None]
    best_cfg = dict(max(completed, key=lambda item: item["cv_score"])["_cfg"]) if completed else {}
    return _record(
        method, seed, budget, best_cfg, records, total,
        {"protocol": "A" if budget is None else "B"},
        Xtr, ytr, Xte, yte,
        {
            "budget_wall_s": wall,
            "surrogate": regressor_kind,
            "surrogate_class": type(surface.regressor).__name__,
            "min_evals": min_evals,
            "engaged_at": state.get("engaged_at"),
            "n_surface_fits": state.get("n_fits", 0),
            "n_iterations": int(getattr(search.OPTIM, "iteration", 0) or 0) if getattr(search, "OPTIM", None) else None,
            "termination_reason": getattr(search, "termination_reason", None),
            "failure_summary": getattr(search, "failure_summary_", {}),
        },
    )


def run_srs_cspace(Xtr, ytr, Xte, yte, *, budget=None, wall=None, seed=0) -> RunRecord:
    """Primary arm: SRS with its default-RBF-equivalent GP surface.

    ``GaussianProcessRegressor()`` is what ``SurrogateSearch`` installs by default
    since v0.2; passing it explicitly is what avoids the combinatorial
    ``MinEvals`` clamp (DESIGN-cspace.md §0, fact 3).
    """
    return _run_srs(Xtr, ytr, Xte, yte, budget=budget, wall=wall, seed=seed,
                    method="srs-cspace", regressor_kind="gp", min_evals=SRS_MIN_EVALS)


def run_srs_onehot_cspace(Xtr, ytr, Xte, yte, *, budget=None, wall=None, seed=0) -> RunRecord:
    """H2 arm: same search with a mixed categorical surface (encoding fix)."""
    return _run_srs(Xtr, ytr, Xte, yte, budget=budget, wall=wall, seed=seed,
                    method="srs-onehot-cspace", regressor_kind="mixed", min_evals=SRS_MIN_EVALS)


def run_srs_nosurr_cspace(Xtr, ytr, Xte, yte, *, budget=None, wall=None, seed=0) -> RunRecord:
    """Ablation: ``min_evals`` beyond the cap, so the surface never engages.

    For Protocol A the cap is time, not evaluations, so the threshold is set to a
    value no run can reach in ``wall`` seconds (``10**9``) — the search then
    degenerates to pure random sampling through the identical SRS machinery.
    """
    threshold = int(budget) if budget is not None else 10**9
    return _run_srs(Xtr, ytr, Xte, yte, budget=budget, wall=wall, seed=seed,
                    method="srs-nosurr-cspace", regressor_kind="gp", min_evals=threshold)


# --------------------------------------------------------------------------- #
# Protocol C: system-level arms (different object of study — own table)
# --------------------------------------------------------------------------- #

def run_flaml(Xtr, ytr, Xte, yte, *, budget=None, wall=120, seed=0) -> RunRecord:
    """FLAML AutoML — chooses its own pipelines; only final accuracy is comparable."""
    from flaml import AutoML

    # FLAML's boosters reject feature names containing "[", "]", "<". Our one-hot
    # columns inherit raw category values (credit-g yields 'checking_status_0<=X<200'),
    # so 7/7 credit-g cells previously failed with
    #   ValueError: feature_names must be string, and may not contain [, ] or <
    # Names are opaque to every learner, so renaming both frames with the same
    # positional map removes a harness artifact without touching the data.
    rename = {column: f"f{index}" for index, column in enumerate(Xtr.columns)}
    Xtr = Xtr.rename(columns=rename)
    Xte = Xte.rename(columns=rename)

    automl = AutoML()
    started = time.perf_counter()
    automl.fit(
        X_train=Xtr, y_train=ytr, task="classification", metric="accuracy",
        time_budget=int(wall), verbose=0, seed=seed, eval_method="cv", n_jobs=1,
    )
    total = time.perf_counter() - started
    prediction = automl.predict(Xte)
    accuracy = float(accuracy_score(yte, prediction))
    balanced = float(balanced_accuracy_score(yte, prediction))
    return RunRecord(
        method="flaml", dataset="", seed=seed, budget=int(wall), n_estimators=50,
        status="ok", n_evals=None,
        best_params={"estimator": getattr(automl, "best_estimator", None),
                     "config": getattr(automl, "best_config", None)},
        final_cv_best=None, final_test_accuracy=accuracy, final_test_balanced=balanced,
        total_wall_s=total,
        extra={"protocol": "C", "best_loss": getattr(automl, "best_loss", None),
               "n_evals": None},
    )


def run_autogluon(Xtr, ytr, Xte, yte, *, budget=None, wall=120, seed=0) -> RunRecord:
    """AutoGluon-Tabular (medium_quality) — portfolio system, Protocol C only."""
    import tempfile

    from autogluon.tabular import TabularPredictor

    label = "__target__"
    train = Xtr.copy()
    train[label] = ytr.to_numpy()
    problem_type = "binary" if ytr.nunique() == 2 else "multiclass"
    with tempfile.TemporaryDirectory() as path:
        predictor = TabularPredictor(label=label, path=path, verbosity=0, problem_type=problem_type)
        started = time.perf_counter()
        predictor.fit(train, presets="medium_quality", time_limit=float(wall), num_cpus=1)
        total = time.perf_counter() - started
        prediction = predictor.predict(Xte).to_numpy()
    accuracy = float(accuracy_score(yte, prediction))
    balanced = float(balanced_accuracy_score(yte, prediction))
    return RunRecord(
        method="autogluon", dataset="", seed=seed, budget=int(wall), n_estimators=50,
        status="ok", n_evals=None,
        best_params={"models": None},
        final_cv_best=None, final_test_accuracy=accuracy, final_test_balanced=balanced,
        total_wall_s=total,
        extra={"protocol": "C", "n_evals": None},
    )


# --------------------------------------------------------------------------- #
# Registries
# --------------------------------------------------------------------------- #

METHODS: dict[str, Callable[..., RunRecord]] = {
    "random-cspace": run_random_cspace,
    "optuna-tpe-cspace": run_optuna_tpe_cspace,
    "optuna-rand-cspace": run_optuna_rand_cspace,
    "srs-cspace": run_srs_cspace,
    "srs-onehot-cspace": run_srs_onehot_cspace,
    "srs-nosurr-cspace": run_srs_nosurr_cspace,
    "flaml": run_flaml,
    "autogluon": run_autogluon,
}

PROTOCOLS: dict[str, list[str]] = {
    "A": ["random-cspace", "optuna-tpe-cspace", "optuna-rand-cspace",
          "srs-cspace", "srs-onehot-cspace", "srs-nosurr-cspace"],
    "B": ["random-cspace", "optuna-tpe-cspace", "optuna-rand-cspace",
          "srs-cspace", "srs-onehot-cspace", "srs-nosurr-cspace"],
    "C": ["flaml", "autogluon", "srs-cspace"],
}
