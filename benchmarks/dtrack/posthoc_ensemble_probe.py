"""Probe small holdout-selected probability ensembles on bundled classifiers.

Run with ``python -m benchmarks.dtrack.posthoc_ensemble_probe``. Candidate
ranking uses training-only cross-validation; equal-weight subsets are selected
on a separate calibration partition and evaluated once on an untouched test
partition. This is an evidence probe, not a production ensemble API.
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASETS = {
    "breast_cancer": load_breast_cancer,
    "iris": load_iris,
    "wine": load_wine,
}


def _candidates(seed: int):
    return {
        "extra_trees": ExtraTreesClassifier(n_estimators=100, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)
        ),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=seed),
    }


def _one_run(X, y, seed: int):
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_calibration, X_test, y_calibration, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.5,
        random_state=seed + 100,
        stratify=y_remaining,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    ranked = []
    for name, estimator in _candidates(seed).items():
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="accuracy")
        ranked.append((float(scores.mean()), name, estimator))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    top = ranked[:3]

    fitted = []
    for cv_score, name, estimator in top:
        estimator.fit(X_train, y_train)
        fitted.append((cv_score, name, estimator))

    best_estimator = fitted[0][2]
    best_test_score = float(
        accuracy_score(y_test, best_estimator.predict(X_test))
    )
    candidates = []
    for size in range(1, len(fitted) + 1):
        for members in itertools.combinations(fitted, size):
            calibration_probabilities = np.mean(
                [estimator.predict_proba(X_calibration) for _, _, estimator in members],
                axis=0,
            )
            calibration_score = float(
                accuracy_score(
                    y_calibration,
                    best_estimator.classes_[
                        np.argmax(calibration_probabilities, axis=1)
                    ],
                )
            )
            test_probabilities = np.mean(
                [estimator.predict_proba(X_test) for _, _, estimator in members],
                axis=0,
            )
            test_score = float(
                accuracy_score(
                    y_test,
                    best_estimator.classes_[np.argmax(test_probabilities, axis=1)],
                )
            )
            candidates.append(
                (
                    calibration_score,
                    size,
                    tuple(name for _, name, _ in members),
                    test_score,
                )
            )

    selected = max(candidates, key=lambda row: (row[0], -row[1], row[2]))
    return {
        "seed": seed,
        "cv_best_model": fitted[0][1],
        "cv_best_score": fitted[0][0],
        "best_model_test_accuracy": best_test_score,
        "calibration_selected_members": list(selected[2]),
        "calibration_accuracy": selected[0],
        "selected_test_accuracy": selected[3],
        "test_accuracy_delta": selected[3] - best_test_score,
    }


def run_probe(dataset: str, seeds: int = 10):
    if dataset not in DATASETS:
        raise ValueError("Unknown dataset %r; choose from %s" % (dataset, ", ".join(DATASETS)))
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
        raise ValueError("seeds must be a positive integer")
    bunch = DATASETS[dataset](return_X_y=True)
    X, y = bunch
    runs = [_one_run(X, y, seed) for seed in range(seeds)]
    deltas = np.asarray([run["test_accuracy_delta"] for run in runs])
    return {
        "dataset": dataset,
        "seed_count": seeds,
        "runs": runs,
        "summary": {
            "mean_test_accuracy_delta": float(deltas.mean()),
            "median_test_accuracy_delta": float(np.median(deltas)),
            "wins": int(np.sum(deltas > 0)),
            "ties": int(np.sum(deltas == 0)),
            "losses": int(np.sum(deltas < 0)),
            "multi_model_selection_rate": float(
                np.mean(
                    [
                        len(run["calibration_selected_members"]) > 1
                        for run in runs
                    ]
                )
            ),
        },
        "protocol": (
            "Top three models ranked by training-only 3-fold CV; equal-weight "
            "singletons and subsets selected by calibration accuracy; one "
            "evaluation on an untouched test split. No test-set selection."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), action="append")
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args(argv)
    reports = [
        run_probe(dataset, seeds=args.seeds)
        for dataset in (args.dataset or sorted(DATASETS))
    ]
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
