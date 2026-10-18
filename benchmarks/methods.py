"""Equal-budget method runners for the benchmark suite (DESIGN.md section 2).

All methods search the *same* discrete parameter space and run exactly
``budget`` candidate-model evaluations on an identical training split with
an identical stratified 5-fold CV splitter, so differences in the results are
attributable to the search strategy alone.

Space design notes:

* Every parameter is a **discrete categorical list**, shared verbatim by all
  methods (SKSurrogate wraps each list in :class:`SKSurrogate.structsearch.Categorical`).
  This makes the candidate sets bit-identical across baselines; with-replacement
  duplicates, when they occur, are part of a method's behavior and are counted
  as evaluations honestly.
* The ``random`` arm draws uniformly at random (with replacement) from these
  lists — algorithmically identical to
  ``sklearn.model_selection.RandomizedSearchCV(param_grid=SPACE, n_iter=budget)``
  for categorical list spaces, but implemented directly so that per-evaluation
  wall time is logged with the same semantics as SKSurrogate's built-in
  ``evaluation_history_``.

Per-evaluation records are ``(k, cv_score, wall_s)``; entries without a score
(invalid/pruned/failed trials) keep their position in the curve with
``cv_score=None`` so evaluation counting stays honest.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#: Shared discrete search space (DESIGN.md section 2). 14 x 8 x 2 = 224 configs.
SPACE: dict[str, list[Any]] = {
    "rf__max_depth": list(range(3, 17)),
    "rf__min_samples_leaf": [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25],
    "rf__criterion": ["gini", "entropy"],
}

N_CONFIGS = int(np.prod([len(v) for v in SPACE.values()]))


def make_pipeline(seed: int, n_estimators: int = 50) -> Pipeline:
    """Pipeline under tuning: StandardScaler + RandomForest (DESIGN.md section 2)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=n_estimators, random_state=seed)),
        ]
    )


@dataclasses.dataclass
class RunRecord:
    """Outcome of one (method, dataset, seed) benchmark run."""

    method: str
    dataset: str
    seed: int
    budget: int
    n_estimators: int
    status: str  # "ok" | "error" | "skipped_archived"
    error: Optional[str] = None
    n_evals: int = 0
    #: list of {"k": int, "cv_score": float|None, "wall_s": float|None} in eval order
    records: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    best_params: dict[str, Any] = dataclasses.field(default_factory=dict)
    final_cv_best: Optional[float] = None
    final_test_accuracy: Optional[float] = None
    final_test_balanced: Optional[float] = None
    total_wall_s: Optional[float] = None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True, default=str)


def _cv_splitter() -> StratifiedKFold:
    """Unshuffled stratified 5-fold — matches ``check_cv(5)`` inside SKSurrogate."""
    return StratifiedKFold(n_splits=5, shuffle=False)


def _evaluate_candidate(params: dict[str, Any], Xtr, ytr, n_estimators: int, seed: int) -> float:
    pipe = make_pipeline(seed, n_estimators).set_params(**params)
    scores = cross_val_score(pipe, Xtr, ytr, cv=_cv_splitter(), scoring="accuracy", n_jobs=1)
    return float(np.mean(scores))


def _final_test_scores(
    best_params: dict[str, Any], Xtr, ytr, Xte, yte, seed: int, n_estimators: int
) -> tuple[Optional[float], Optional[float]]:
    """Refit the best candidate on the full training split; score on held-out test."""
    if not best_params:
        return None, None
    pipe = make_pipeline(seed, n_estimators).set_params(**best_params).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    return float(accuracy_score(yte, pred)), float(balanced_accuracy_score(yte, pred))


def _best_completed(records: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The completed record with the highest CV score (None if none completed)."""
    completed = [r for r in records if r.get("cv_score") is not None]
    if not completed:
        return None
    return max(completed, key=lambda r: r["cv_score"])


# --------------------------------------------------------------------------- #
# Method runners
# --------------------------------------------------------------------------- #

def run_sksurrogate(Xtr, ytr, Xte, yte, budget: int, seed: int, n_estimators: int = 50) -> RunRecord:
    """Arm B — SKSurrogate surrogate-guided search (primary method).

    Requests the default engagement ``min_evals = max(5, budget // 5)``, but note the
    constructor clamp in :class:`SKSurrogate.structsearch.SurrogateSearch`: with the
    default cubic Hilbert regressor the effective threshold is
    ``max(requested, C(n_params + 3, n_params))`` — for this 3-parameter space that is
    **20**, so the surrogate only engages once 20 points have been evaluated (records
    k >= budget - 10 at the suite budget B = 30; it never engages when budget < 21).
    See DESIGN.md v1.4 errata.
    """
    return _run_srs(Xtr, ytr, Xte, yte, budget, seed, n_estimators, min_evals=None, task_tag="srs")


def run_srs_no_surrogate(Xtr, ytr, Xte, yte, budget: int, seed: int, n_estimators: int = 50) -> RunRecord:
    """Arm A — same SRS outer loop with ``min_evals = budget`` (DESIGN.md section 9).

    The response-surface approximation never engages; search degenerates to pure
    random sampling through the identical SRS machinery, isolating exactly the
    surrogate's contribution. Everything else (space, estimator, seeds, budgets)
    is bit-identical to :func:`run_sksurrogate`.
    """
    return _run_srs(
        Xtr, ytr, Xte, yte, budget, seed, n_estimators,
        min_evals=budget, task_tag="srss", method_name="srs-no-surrogate",
    )


def _run_srs(
    Xtr, ytr, Xte, yte, budget: int, seed: int, n_estimators: int = 50,
    min_evals: int | None = None, task_tag: str = "srs", method_name: str = "sksurrogate",
) -> RunRecord:
    """Shared runner for both SRS ablation arms; ``min_evals=None`` selects default engagement."""
    from SKSurrogate.structsearch import Categorical, SurrogateRandomCV

    if min_evals is None:
        min_evals = min(max(5, budget // 5), budget)
    params = {name: Categorical(values) for name, values in SPACE.items()}
    # Hard evaluation cap: SRS's `max_iter` counts loop iterations (inclusive
    # off-by-one) and the x0 initial point is evaluated in __init__, so a naive
    # max_iter=budget would spend budget+2 evaluations. `max_evals` is checked
    # before every iteration and caps *total* objective calls exactly.
    # warm_start/Continue must be OFF: by default SRS persists state to a .pkl
    # in the CWD after every iteration and silently restores it, which would
    # contaminate independent benchmark runs (verified 2026-10-17).
    # `min_evals` is passed through verbatim: None -> default request (Arm B), budget ->
    # surrogate never engages within the cap (Arm A; DESIGN.md section 9). NOTE: the
    # SurrogateSearch constructor clamps the effective threshold up to C(n_params+3, n_params)
    # for the default cubic Hilbert regressor — 20 in this space (DESIGN.md v1.4 errata), so
    # Arm B engages only once 20 points are evaluated regardless of the requested value.
    task_name = f"{task_tag}_bench_s{seed}_{int(time.time())}"
    search = SurrogateRandomCV(
        estimator=make_pipeline(seed, n_estimators),
        params=params,
        max_iter=budget,
        max_evals=budget,
        min_evals=min_evals,
        scoring=None,
        cv=5,
        refit=False,
        verbose=0,
        random_state=seed,
        n_jobs=1,
        warm_start=False,
        Continue=False,
        task_name=task_name,
    )
    # SRS draws its initial point x0 from Python's *module-level* `random`
    # state inside SurrogateRandomCV.fit (structsearch.py), which ignores our
    # per-run random_state — so across processes, and between the two arms of
    # one process, x0 would differ. Pin it deterministically per seed (same salt
    # for both arms => identical starting point) and restore the caller's global
    # state afterwards; everything else in SRS uses the seeded `self.rng`.
    import random as _pyrandom

    _global_rng_state = _pyrandom.getstate()
    _pyrandom.seed(1_000_003 + seed)
    started = time.perf_counter()
    try:
        search.fit(Xtr, ytr)
    finally:
        _pyrandom.setstate(_global_rng_state)
    total_wall = time.perf_counter() - started

    # SRS writes a state checkpoint to the CWD after every iteration; remove it
    # so no cross-run state survives (warm_start/Continue are off regardless).
    _pkl = Path(task_name + ".pkl")
    if _pkl.exists():
        _pkl.unlink()

    records: list[dict[str, Any]] = []
    for k, item in enumerate(search.evaluation_history_):
        raw_score = item.get("score")
        # structsearch stores scores negated (minimization convention).
        score = None if raw_score is None else float(-raw_score)
        records.append(
            {
                "k": k,
                "cv_score": score,
                "wall_s": float(item.get("duration", 0.0)),
                "status": item.get("status"),
                "_params": dict(item.get("params", {})),
            }
        )

    best = _best_completed(records)
    best_params = dict(best["_params"]) if best is not None else {}
    test_acc, test_bal = _final_test_scores(best_params, Xtr, ytr, Xte, yte, seed, n_estimators)

    return RunRecord(
        method=method_name,
        dataset="",  # filled by caller
        seed=seed,
        budget=budget,
        n_estimators=n_estimators,
        status="ok" if best is not None else "error",
        error=None if best is not None else "no completed evaluations in history",
        n_evals=len(records),
        records=[{key: value for key, value in r.items() if key != "_params"} for r in records],
        best_params=best_params,
        final_cv_best=float(-min(i["score"] for i in search.evaluation_history_ if i.get("score") is not None))
        if any(i.get("score") is not None for i in search.evaluation_history_)
        else None,
        final_test_accuracy=test_acc,
        final_test_balanced=test_bal,
        total_wall_s=total_wall,
        extra={
            "termination_reason": getattr(search, "termination_reason", None),
            "failure_summary": getattr(search, "failure_summary_", {}),
        },
    )


def run_random(Xtr, ytr, Xte, yte, budget: int, seed: int, n_estimators: int = 50) -> RunRecord:
    """Uniform random search over the shared space (RandomizedSearchCV-equivalent)."""
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for k in range(budget):
        params = {name: values[int(rng.integers(len(values)))] for name, values in SPACE.items()}
        started = time.perf_counter()
        try:
            score = _evaluate_candidate(params, Xtr, ytr, n_estimators, seed)
            status = "complete"
        except Exception as exc:  # noqa: BLE001 - record and continue like SRS does
            score = None
            status = f"failed: {exc.__class__.__name__}: {exc}"
        wall = time.perf_counter() - started
        records.append({"k": k, "cv_score": score, "wall_s": wall, "status": status, "_params": params})

    best = _best_completed(records)
    best_params = dict(best.pop("_params")) if best is not None else {}
    test_acc, test_bal = _final_test_scores(best_params, Xtr, ytr, Xte, yte, seed, n_estimators)

    return RunRecord(
        method="random",
        dataset="",  # filled by caller
        seed=seed,
        budget=budget,
        n_estimators=n_estimators,
        status="ok" if best is not None else "error",
        error=None if best is not None else "no completed evaluations",
        n_evals=len(records),
        records=[{key: value for key, value in r.items() if key != "_params"} for r in records],
        best_params=best_params,
        final_cv_best=best["cv_score"] if best is not None else None,
        final_test_accuracy=test_acc,
        final_test_balanced=test_bal,
    )


def run_optuna_tpe(Xtr, ytr, Xte, yte, budget: int, seed: int, n_estimators: int = 50) -> RunRecord:
    """Optuna TPE over the shared space."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "optuna is required for the optuna-tpe arm. Install it with `pip install sksurrogate[benchmark]`."
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {name: trial.suggest_categorical(name, values) for name, values in SPACE.items()}
        return _evaluate_candidate(params, Xtr, ytr, n_estimators, seed)

    started = time.perf_counter()
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    total_wall = time.perf_counter() - started

    # DESIGN.md risk #2: trial count must equal candidate fits exactly.
    if len(study.trials) != budget:
        return RunRecord(
            method="optuna-tpe",
            dataset="",  # filled by caller
            seed=seed,
            budget=budget,
            n_estimators=n_estimators,
            status="error",
            error=f"trial-count parity violated: {len(study.trials)} trials != budget {budget}",
            total_wall_s=total_wall,
        )

    records: list[dict[str, Any]] = []
    for k, trial in enumerate(study.trials):
        wall = None
        if trial.datetime_start is not None and trial.datetime_complete is not None:
            wall = (trial.datetime_complete - trial.datetime_start).total_seconds()
        score = None if trial.value is None else float(trial.value)
        records.append({"k": k, "cv_score": score, "wall_s": wall})

    best_trial = study.best_trial
    best_params = dict(best_trial.params) if best_trial is not None and best_trial.value is not None else {}
    test_acc, test_bal = _final_test_scores(best_params, Xtr, ytr, Xte, yte, seed, n_estimators)

    return RunRecord(
        method="optuna-tpe",
        dataset="",  # filled by caller
        seed=seed,
        budget=budget,
        n_estimators=n_estimators,
        status="ok" if best_trial is not None and best_trial.value is not None else "error",
        error=None if (best_trial is not None and best_trial.value is not None) else "no completed trials",
        n_evals=len(records),
        records=records,
        best_params=best_params,
        final_cv_best=float(best_trial.value) if best_trial is not None and best_trial.value is not None else None,
        final_test_accuracy=test_acc,
        final_test_balanced=test_bal,
        total_wall_s=total_wall,
    )


def run_skopt(Xtr, ytr, Xte, yte, budget: int, seed: int, n_estimators: int = 50) -> RunRecord:
    """scikit-optimize baseline — conditional on the (archived) dependency importing.

    If scikit-optimize no longer installs/imports against current sklearn, this
    returns a ``skipped_archived`` record instead of failing the suite; the skip
    itself is recorded data (DESIGN.md section 2).
    """
    try:
        from skopt import BayesSearchCV
    except Exception as exc:  # noqa: BLE001 - any import/ABI breakage counts
        return RunRecord(
            method="skopt",
            dataset="",  # filled by caller
            seed=seed,
            budget=budget,
            n_estimators=n_estimators,
            status="skipped_archived",
            error=f"{exc.__class__.__name__}: {exc}",
            extra={"archived": True},
        )

    started = time.perf_counter()
    search = BayesSearchCV(
        estimator=make_pipeline(seed, n_estimators),
        search_spaces={name: list(values) for name, values in SPACE.items()},
        n_iter=budget,
        cv=5,
        refit=False,
        random_state=seed,
        n_jobs=1,
    )
    try:
        search.fit(Xtr, ytr)
    except Exception as exc:  # noqa: BLE001 - archived dependency breaking at fit time is a skip, not a failure
        return RunRecord(
            method="skopt",
            dataset="",  # filled by caller
            seed=seed,
            budget=budget,
            n_estimators=n_estimators,
            status="skipped_archived",
            error=f"fit failed (archived dependency): {exc.__class__.__name__}: {exc}",
            total_wall_s=time.perf_counter() - started,
            extra={"archived": True},
        )
    total_wall = time.perf_counter() - started

    # BayesSearchCV has no per-evaluation wall clock; curve order comes from cv_results_.
    records: list[dict[str, Any]] = []
    for k in range(len(search.cv_results_["mean_test_score"])):
        score = float(search.cv_results_["mean_test_score"][k])
        records.append({"k": k, "cv_score": score, "wall_s": None})

    best_idx = int(search.best_index_)
    best_params = {name: search.cv_results_["params"][best_idx][name] for name in SPACE}
    test_acc, test_bal = _final_test_scores(best_params, Xtr, ytr, Xte, yte, seed, n_estimators)

    return RunRecord(
        method="skopt",
        dataset="",  # filled by caller
        seed=seed,
        budget=budget,
        n_estimators=n_estimators,
        status="ok" if records else "error",
        error=None if records else "no candidates evaluated",
        n_evals=len(records),
        records=records,
        best_params=best_params,
        final_cv_best=float(max(r["cv_score"] for r in records)) if records else None,
        final_test_accuracy=test_acc,
        final_test_balanced=test_bal,
        total_wall_s=total_wall,
    )


METHODS: dict[str, Any] = {
    "sksurrogate": run_sksurrogate,
    "srs-no-surrogate": run_srs_no_surrogate,
    "random": run_random,
    "optuna-tpe": run_optuna_tpe,
    "skopt": run_skopt,
}
