"""Declared problem type for a task (regression vs classification).

The control plane used to leave this implicit: the target column was stored at
registration, but "classification or regression?" was decided later — by the
Quick-start preset, by whichever estimator was configured, or by inferring the
family of an already-fitted bundle. Screens could therefore disagree, and a
regression task's first experiment silently defaulted to a classification preset.

This module makes the choice explicit and durable:

* :func:`infer_problem_type` derives a *default* from the target column with a
  conservative, explainable rule;
* the declaration is stored in task metadata as ``problem_type`` together with
  ``problem_type_source`` (``"inferred"`` or ``"user"``) and
  ``problem_type_reason``;
* :func:`declared_problem_type` reads it back, falling back to inference for tasks
  registered before the field existed (so nothing breaks).

Time structure is a separate axis (``temporal_spec``): a forecasting task is still
a regression problem — numeric target, regression metrics — so a temporal task is
pinned to ``"regression"`` and asking for classification there is refused rather
than silently scored as one.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

PROBLEM_TYPES = ("regression", "classification")

#: Metadata keys owned by this module.
PROBLEM_TYPE_KEY = "problem_type"
PROBLEM_TYPE_SOURCE_KEY = "problem_type_source"
PROBLEM_TYPE_REASON_KEY = "problem_type_reason"
PROBLEM_TYPE_INFERRED_KEY = "problem_type_inferred"


def infer_problem_type(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    """Infer the default problem type from the target column.

    The rule is deliberately conservative, because a wrong default is worse than
    asking the user:

    * boolean target → classification (two levels by construction);
    * non-numeric dtype (object/category/string) → classification;
    * integer target with exactly two distinct values → classification, since a
      0/1 integer column is a classification label in practice. The reason string
      says so, so counts mislabelled this way can be corrected with an override;
    * everything else numeric → regression (continuous measurements and integer
      counts alike).

    :returns: ``{"problem_type", "reason", "cardinality", "dtype"}``.
    """
    if target not in frame.columns:
        raise ValueError("target column %r is not present in the frame" % target)
    series = frame[target].dropna()
    cardinality = int(series.nunique())
    dtype = str(frame[target].dtype)
    if pd.api.types.is_bool_dtype(frame[target]):
        return {
            "problem_type": "classification",
            "reason": "boolean target (two levels by construction)",
            "cardinality": cardinality,
            "dtype": dtype,
        }
    if not pd.api.types.is_numeric_dtype(frame[target]):
        return {
            "problem_type": "classification",
            "reason": "non-numeric target (dtype %s)" % dtype,
            "cardinality": cardinality,
            "dtype": dtype,
        }
    if pd.api.types.is_integer_dtype(frame[target]) and cardinality == 2:
        return {
            "problem_type": "classification",
            "reason": "integer target with exactly 2 distinct values",
            "cardinality": cardinality,
            "dtype": dtype,
        }
    return {
        "problem_type": "regression",
        "reason": "numeric target (dtype %s, %d distinct values)" % (dtype, cardinality),
        "cardinality": cardinality,
        "dtype": dtype,
    }


def problem_type_metadata(
    frame: pd.DataFrame,
    target: str,
    *,
    override: str | None = None,
    temporal: bool = False,
    label_encoded: bool = False,
) -> dict[str, Any]:
    """Metadata block describing the task's problem type.

    ``override`` is what the user explicitly chose; without it the inference wins.
    Both the resolved value and the inference are stored, so the Datasets page can
    say "inferred: classification — overridden to regression".

    ``temporal`` pins the result to ``"regression"``: a forecasting task is scored
    with regression metrics regardless of how the label is coded, so a binary
    target (which the inference would call classification) becomes a regression on
    0/1 values, with the reason saying so. A *user* override to classification on a
    temporal task is still refused by the callers via
    :func:`ensure_temporal_is_regression`.

    ``label_encoded`` says the target was binarized/ordinal-encoded at
    registration: the stored column is then a float 0/1 code, which the dtype rule
    alone would call regression, so the declaration keeps the classification
    meaning the user asked for (source stays ``"inferred"``, the reason says why).
    """
    inferred = infer_problem_type(frame, target)
    if override is not None:
        if override not in PROBLEM_TYPES:
            raise ValueError("problem_type must be one of %s" % ", ".join(PROBLEM_TYPES))
        resolved = override
        source = "user"
    elif label_encoded and inferred["problem_type"] != "classification":
        resolved = "classification"
        source = "inferred"
    else:
        resolved = inferred["problem_type"]
        source = "inferred"
    if source == "user":
        reason = "set by the user"
    elif label_encoded and inferred["problem_type"] != "classification":
        reason = "target was binarized/ordinal-encoded at registration"
    else:
        reason = inferred["reason"]
    if temporal and resolved == "classification":
        resolved = "regression"
        source = "inferred"
        reason = "temporal task is regression-shaped (%s)" % inferred["reason"]
    return {
        PROBLEM_TYPE_KEY: resolved,
        PROBLEM_TYPE_SOURCE_KEY: source,
        PROBLEM_TYPE_REASON_KEY: reason,
        PROBLEM_TYPE_INFERRED_KEY: inferred["problem_type"],
    }


def declared_problem_type(metadata: dict[str, Any]) -> str | None:
    """The task's declared problem type, or ``None`` for legacy tasks.

    Deliberately does **not** re-infer from an estimator here: callers that need a
    fallback should use the dataset columns, and this function returning ``None``
    is the signal for "this task predates the declaration".
    """
    value = (metadata or {}).get(PROBLEM_TYPE_KEY)
    return value if value in PROBLEM_TYPES else None


def ensure_temporal_is_regression(problem_type: str | None) -> None:
    """Refuse classification on a temporal task.

    The feature builder casts the target to float and the rolling-origin backtest
    scores MAE/RMSE/MASE, so a "classification" temporal task would be a label
    pretending to be a forecast. Raising here keeps the metadata honest.
    """
    if problem_type == "classification":
        raise ValueError(
            "a temporal task is regression-shaped: the feature builder casts the "
            "target to float and the backtest scores regression metrics. Encode the "
            "label as numeric and declare problem_type='regression', or remove the "
            "temporal spec."
        )


__all__ = [
    "PROBLEM_TYPES",
    "PROBLEM_TYPE_KEY",
    "PROBLEM_TYPE_SOURCE_KEY",
    "PROBLEM_TYPE_REASON_KEY",
    "PROBLEM_TYPE_INFERRED_KEY",
    "declared_problem_type",
    "ensure_temporal_is_regression",
    "infer_problem_type",
    "problem_type_metadata",
]
