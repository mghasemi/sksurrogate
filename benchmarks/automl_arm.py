"""Wall-clock-budgeted AutoML arm (complements the equal-evaluation protocol).

The published benchmark suite (``benchmarks/design`` / ``docs/benchmarks.md``)
compares *search strategies* at an equal number of candidate-model evaluations.
That currency cannot express AutoML systems, which decide their own number of
internal fits (stacking, bagging, multi-model portfolios). This arm therefore
uses a **second, explicitly different protocol**:

* one identical stratified 80/20 split per (dataset, seed), test set never seen
  during search;
* every method receives the same **wall-clock budget** ``B`` seconds;
* the reported number is final held-out accuracy (and balanced accuracy) of the
  model the method chose, plus how many candidate evaluations it managed to run.

Methods
-------
``sksurrogate``   SurrogateRandomCV over the suite's shared discrete space,
                  ``time_limit=B`` (SKSurrogate's own wall-clock stop).
``optuna``        Optuna TPE over the same space, ``timeout=B``.
``autogluon``     TabularPredictor, ``presets="medium_quality"``, ``time_limit=B``.
``flaml``         AutoML, ``time_budget=B``.

This is a *pilot-scale* harness: it is honest about what it measures (wall-clock
quality, not search efficiency) and it is not a substitute for the equal-budget
suite. Usage::

    python benchmarks/automl_arm.py --datasets ionosphere credit-g galaxy3 \
        --seeds 0 1 --budget 45 --out benchmarks/results/automl_arm/pilot.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.datasets import load_dataset
from benchmarks.methods import SPACE, make_pipeline


def _split(X: pd.DataFrame, y: pd.Series, seed: int):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


def _scores(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def run_sksurrogate(Xtr, ytr, Xte, yte, budget, seed):
    from SKSurrogate import Categorical, SurrogateRandomCV

    params = {name: Categorical(list(values)) for name, values in SPACE.items()}
    search = SurrogateRandomCV(
        make_pipeline(seed),
        params,
        max_iter=10_000,
        min_evals=10,
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        scoring="accuracy",
        time_limit=float(budget),
        random_state=seed,
        n_jobs=1,
        verbose=0,
    )
    started = time.perf_counter()
    search.fit(Xtr, ytr)
    wall = time.perf_counter() - started
    pred = search.predict(Xte)
    best = {k.replace("rf__", ""): v for k, v in (search.best_params_ or {}).items()}
    return {
        "wall_s": wall,
        "n_evals": len(search.evaluation_history_),
        "cv_score": float(search.best_score_),
        "best_params": best,
        **_scores(yte, pred),
    }


def run_optuna(Xtr, ytr, Xte, yte, budget, seed):
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    best = {"value": -np.inf, "params": None}
    counter = {"n": 0}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)

    def objective(trial):
        params = {
            name.replace("rf__", ""): trial.suggest_categorical(name, list(values))
            for name, values in SPACE.items()
        }
        counter["n"] += 1
        score = cross_val_score(
            make_pipeline(seed, n_estimators=50).set_params(**params),
            Xtr,
            ytr,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        ).mean()
        if score > best["value"]:
            best["value"], best["params"] = float(score), params
        return score

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    started = time.perf_counter()
    study.optimize(objective, timeout=float(budget), n_trials=10_000)
    wall = time.perf_counter() - started
    model = make_pipeline(seed, n_estimators=50).set_params(**best["params"])
    model.fit(Xtr, ytr)
    return {
        "wall_s": wall,
        "n_evals": counter["n"],
        "cv_score": best["value"],
        "best_params": best["params"],
        **_scores(yte, model.predict(Xte)),
    }


def run_autogluon(Xtr, ytr, Xte, yte, budget, seed):
    import tempfile

    from autogluon.tabular import TabularPredictor

    train = Xtr.copy()
    train["__target__"] = ytr.to_numpy()
    with tempfile.TemporaryDirectory() as path:
        predictor = TabularPredictor(
            label="__target__", path=path, verbosity=0, problem_type="binary"
            if ytr.nunique() == 2
            else ("multiclass" if ytr.nunique() > 2 else "binary")
        )
        started = time.perf_counter()
        predictor.fit(train, presets="medium_quality", time_limit=float(budget))
        wall = time.perf_counter() - started
        pred = predictor.predict(Xte)
    return {
        "wall_s": wall,
        "n_evals": None,  # AutoGluon does not expose a comparable evaluation count
        "cv_score": None,
        "best_params": None,
        **_scores(yte, pred.to_numpy()),
    }


def _flaml_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """FLAML rejects feature names containing ``[ ] <`` (credit-g's one-hot names).

    Rename to positional names for the FLAML arm only; every method still sees
    the same values, and the rename keeps the comparison valid.
    """
    return frame.set_axis([f"f{i}" for i in range(frame.shape[1])], axis=1)


def run_flaml(Xtr, ytr, Xte, yte, budget, seed):
    from flaml import AutoML

    Xtr, Xte = _flaml_safe(Xtr), _flaml_safe(Xte)
    automl = AutoML()
    started = time.perf_counter()
    automl.fit(
        X_train=Xtr,
        y_train=ytr,
        task="classification",
        metric="accuracy",
        time_budget=int(budget),
        verbose=0,
        seed=seed,
        eval_method="cv",
    )
    wall = time.perf_counter() - started
    pred = automl.predict(Xte)
    return {
        "wall_s": wall,
        "n_evals": None,
        "cv_score": float(getattr(automl, "best_loss", float("nan")))
        if hasattr(automl, "best_loss")
        else None,
        "best_params": {"estimator": getattr(automl, "best_estimator", None)},
        **_scores(yte, pred),
    }


RUNNERS = {
    "sksurrogate": run_sksurrogate,
    "optuna": run_optuna,
    "autogluon": run_autogluon,
    "flaml": run_flaml,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["ionosphere", "credit-g", "galaxy3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--budget", type=float, default=45.0, help="wall-clock seconds per fit")
    parser.add_argument("--methods", nargs="+", default=list(RUNNERS))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = []
    for name in args.datasets:
        X, y = load_dataset(name)
        for seed in args.seeds:
            Xtr, Xte, ytr, yte = _split(X, y, seed)
            for method in args.methods:
                runner = RUNNERS[method]
                try:
                    result = runner(Xtr, ytr, Xte, yte, args.budget, seed)
                    status = "ok"
                except Exception as exc:  # keep the record; never silently drop
                    result = {"error": "%s: %s" % (exc.__class__.__name__, exc)}
                    status = "failed"
                record = {
                    "dataset": name,
                    "seed": seed,
                    "method": method,
                    "budget_s": args.budget,
                    "status": status,
                    **result,
                }
                records.append(record)
                acc = record.get("accuracy")
                print(
                    "%-10s seed=%d %-12s %s%s"
                    % (
                        name,
                        seed,
                        method,
                        ("acc=%.4f" % acc) if acc is not None else "FAILED",
                        (" wall=%.1fs evals=%s" % (record.get("wall_s", float("nan")), record.get("n_evals"))),
                    ),
                    flush=True,
                )

    payload = {
        "protocol": "wall-clock budget (NOT the equal-evaluation protocol)",
        "budget_s": args.budget,
        "datasets": args.datasets,
        "seeds": args.seeds,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": records,
    }
    out = args.out
    if out is None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out = Path("benchmarks/results/automl_arm") / ("%s.json" % stamp)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print("\nwrote %s (%d records)" % (out, len(records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
