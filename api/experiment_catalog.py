"""Curated component catalog for the visual search-space editor.

Served by ``GET /api/experiments/components`` (see :mod:`api.routers.experiments`).
Estimators are tagged with their task type so the web UI can filter the library;
transformers and samplers carry no tag or a fixed classification tag respectively.
Parameter specs use exactly the ``ParamSpec`` shapes accepted in a search-space
config, so what the editor shows is what it submits — including pinned scalars
(plain values), which are legal entries.

Rows with ``"preselected": True`` seed the visual editor on first open for that
task (see decision D6). The RF/ExtraTrees rows deliberately reuse
:data:`SKSurrogate.aml.PRESET_PARAM_SPECS` so the visual defaults and the preset
payloads cannot drift apart.

Parameters are included only when they materially affect performance *and* map
to a spec-convertible type (real / integer / categorical). Boolean flags
(``SVC.probability``, ``SimpleImputer.add_indicator``), string-valued objectives
(``XGB*.objective``, task-coupled) and non-spec values such as SMOTE's
``sampling_strategy="auto"`` are deliberately absent: they stay available by
pinning scalars in Raw mode.
"""

from __future__ import annotations

from SKSurrogate.aml import PRESET_PARAM_SPECS

# One row per library entry. ``task`` is set for estimators and samplers only;
# transformers appear under both problem types.
EXPERIMENT_COMPONENTS: list[dict[str, object]] = [
    # --- classification estimators --------------------------------------
    {
        "fqn": "sklearn.linear_model.LogisticRegression",
        "name": "Logistic Regression",
        "role": "estimator",
        "task": "classification",
        "description": "Linear baseline; strong on small, well-scaled data.",
        "params": {
            "C": {"type": "real", "low": 0.01, "high": 10},
            "penalty": {"type": "categorical", "items": ["l2", "l1"]},
        },
    },
    {
        "fqn": "sklearn.svm.SVC",
        "name": "Support Vector Classifier",
        "role": "estimator",
        "task": "classification",
        "description": "Kernel classifier; effective on small-to-medium data.",
        "params": {
            "C": {"type": "real", "low": 0.01, "high": 10},
            "kernel": {"type": "categorical", "items": ["rbf", "linear"]},
            "gamma": {"type": "categorical", "items": ["scale", "auto"]},
        },
    },
    {
        "fqn": "sklearn.tree.DecisionTreeClassifier",
        "name": "Decision Tree (clf)",
        "role": "estimator",
        "task": "classification",
        "description": "Interpretable single tree; weak alone, useful as a baseline letter.",
        "params": {
            "max_depth": {"type": "integer", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "integer", "low": 1, "high": 20},
        },
    },
    {
        "fqn": "sklearn.ensemble.RandomForestClassifier",
        "name": "Random Forest (clf)",
        "role": "estimator",
        "task": "classification",
        "preselected": True,
        "description": "Bagged trees; the default workhorse for tabular classification.",
        "params": dict(PRESET_PARAM_SPECS),
    },
    {
        "fqn": "sklearn.ensemble.ExtraTreesClassifier",
        "name": "Extra Trees (clf)",
        "role": "estimator",
        "task": "classification",
        "description": "Random splits on both sides; often trades a little accuracy for speed.",
        "params": dict(PRESET_PARAM_SPECS),
    },
    {
        "fqn": "xgboost.XGBClassifier",
        "name": "XGBoost (clf)",
        "role": "estimator",
        "task": "classification",
        "preselected": True,
        "description": "Regularised gradient boosting; a top-3 tabular workhorse.",
        "params": {
            "n_estimators": {"type": "integer", "low": 50, "high": 200},
            "max_depth": {"type": "integer", "low": 2, "high": 10},
            "learning_rate": {"type": "real", "low": 0.01, "high": 0.3},
            "subsample": {"type": "real", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "real", "low": 0.6, "high": 1.0},
            "min_child_weight": {"type": "real", "low": 1, "high": 10},
            "gamma": {"type": "real", "low": 0, "high": 5},
            "reg_alpha": {"type": "real", "low": 0, "high": 10},
            "reg_lambda": {"type": "real", "low": 0, "high": 10},
        },
    },
    {
        "fqn": "lightgbm.LGBMClassifier",
        "name": "LightGBM (clf)",
        "role": "estimator",
        "task": "classification",
        "preselected": True,
        "description": "Leaf-wise gradient boosting; fast, strong on medium/large tables.",
        "params": {
            "n_estimators": {"type": "integer", "low": 50, "high": 200},
            "num_leaves": {"type": "integer", "low": 4, "high": 128},
            "learning_rate": {"type": "real", "low": 0.01, "high": 0.3},
            "max_depth": {"type": "integer", "low": 2, "high": 10},
            "min_child_samples": {"type": "integer", "low": 1, "high": 100},
            "subsample": {"type": "real", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "real", "low": 0.6, "high": 1.0},
            "reg_alpha": {"type": "real", "low": 0, "high": 10},
            "reg_lambda": {"type": "real", "low": 0, "high": 10},
            "boosting_type": {"type": "categorical", "items": ["gbdt", "dart", "goss"]},
        },
    },
    {
        "fqn": "rgf.sklearn.RGFClassifier",
        "name": "RGF (clf)",
        "role": "estimator",
        "task": "classification",
        "description": "Random Gradient Forest; competitive on structured data. `algorithm` uses the verified underscore forms.",
        "params": {
            "algorithm": {"type": "categorical", "items": ["RGF", "RGF_Opt", "RGF_Sib"]},
            "max_leaf": {"type": "categorical", "items": [8, 16, 32, 64]},
            "n_tree_search": {"type": "integer", "low": 1, "high": 10},
            "learning_rate": {"type": "real", "low": 0.05, "high": 0.3},
            "reg_depth": {"type": "integer", "low": 1, "high": 5},
        },
    },
    # --- regression estimators ------------------------------------------
    {
        "fqn": "sklearn.linear_model.Ridge",
        "name": "Ridge Regression",
        "role": "estimator",
        "task": "regression",
        "description": "Linear baseline with L2 regularisation.",
        "params": {"alpha": {"type": "real", "low": 0.01, "high": 100}},
    },
    {
        "fqn": "sklearn.svm.SVR",
        "name": "Support Vector Regressor",
        "role": "estimator",
        "task": "regression",
        "description": "Kernel regressor; effective on small-to-medium data.",
        "params": {
            "C": {"type": "real", "low": 0.01, "high": 10},
            "kernel": {"type": "categorical", "items": ["rbf", "linear"]},
            "epsilon": {"type": "real", "low": 0, "high": 0.5},
        },
    },
    {
        "fqn": "sklearn.tree.DecisionTreeRegressor",
        "name": "Decision Tree (reg)",
        "role": "estimator",
        "task": "regression",
        "description": "Interpretable single tree; weak alone, useful as a baseline letter.",
        "params": {
            "max_depth": {"type": "integer", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "integer", "low": 1, "high": 20},
        },
    },
    {
        "fqn": "sklearn.ensemble.RandomForestRegressor",
        "name": "Random Forest (reg)",
        "role": "estimator",
        "task": "regression",
        "preselected": True,
        "description": "Bagged trees; the default workhorse for tabular regression.",
        "params": dict(PRESET_PARAM_SPECS),
    },
    {
        "fqn": "sklearn.ensemble.ExtraTreesRegressor",
        "name": "Extra Trees (reg)",
        "role": "estimator",
        "task": "regression",
        "description": "Random splits on both sides; often trades a little accuracy for speed.",
        "params": dict(PRESET_PARAM_SPECS),
    },
    {
        "fqn": "xgboost.XGBRegressor",
        "name": "XGBoost (reg)",
        "role": "estimator",
        "task": "regression",
        "preselected": True,
        "description": "Regularised gradient boosting; a top-3 tabular workhorse.",
        "params": {
            "n_estimators": {"type": "integer", "low": 50, "high": 200},
            "max_depth": {"type": "integer", "low": 2, "high": 10},
            "learning_rate": {"type": "real", "low": 0.01, "high": 0.3},
            "subsample": {"type": "real", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "real", "low": 0.6, "high": 1.0},
            "min_child_weight": {"type": "real", "low": 1, "high": 10},
            "gamma": {"type": "real", "low": 0, "high": 5},
            "reg_alpha": {"type": "real", "low": 0, "high": 10},
            "reg_lambda": {"type": "real", "low": 0, "high": 10},
        },
    },
    {
        "fqn": "lightgbm.LGBMRegressor",
        "name": "LightGBM (reg)",
        "role": "estimator",
        "task": "regression",
        "preselected": True,
        "description": "Leaf-wise gradient boosting; fast, strong on medium/large tables.",
        "params": {
            "n_estimators": {"type": "integer", "low": 50, "high": 200},
            "num_leaves": {"type": "integer", "low": 4, "high": 128},
            "learning_rate": {"type": "real", "low": 0.01, "high": 0.3},
            "max_depth": {"type": "integer", "low": 2, "high": 10},
            "min_child_samples": {"type": "integer", "low": 1, "high": 100},
            "subsample": {"type": "real", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "real", "low": 0.6, "high": 1.0},
            "reg_alpha": {"type": "real", "low": 0, "high": 10},
            "reg_lambda": {"type": "real", "low": 0, "high": 10},
            "boosting_type": {"type": "categorical", "items": ["gbdt", "dart", "goss"]},
        },
    },
    {
        "fqn": "rgf.sklearn.RGFRegressor",
        "name": "RGF (reg)",
        "role": "estimator",
        "task": "regression",
        "description": "Random Gradient Forest; competitive on structured data. `algorithm` uses the verified underscore forms.",
        "params": {
            "algorithm": {"type": "categorical", "items": ["RGF", "RGF_Opt", "RGF_Sib"]},
            "max_leaf": {"type": "categorical", "items": [8, 16, 32, 64]},
            "n_tree_search": {"type": "integer", "low": 1, "high": 10},
            "learning_rate": {"type": "real", "low": 0.05, "high": 0.3},
            "reg_depth": {"type": "integer", "low": 1, "high": 5},
        },
    },
    # --- resamplers (classification only; AML places samplers first) ----
    {
        "fqn": "imblearn.over_sampling.SMOTE",
        "name": "SMOTE",
        "role": "sampler",
        "task": "classification",
        "description": "Synthesises minority samples. `sampling_strategy` is a real ratio here; the string \"auto\" stays in Raw mode.",
        "params": {
            "k_neighbors": {"type": "integer", "low": 3, "high": 15},
            "sampling_strategy": {"type": "real", "low": 0.5, "high": 1.0},
        },
    },
    {
        "fqn": "imblearn.over_sampling.ADASYN",
        "name": "ADASYN",
        "role": "sampler",
        "task": "classification",
        "description": "Synthesises minority samples, concentrated where the class boundary is hardest.",
        "params": {
            "n_neighbors": {"type": "integer", "low": 3, "high": 20},
            "sampling_strategy": {"type": "real", "low": 0.5, "high": 1.0},
        },
    },
    {
        "fqn": "imblearn.under_sampling.RandomUnderSampler",
        "name": "Random Under-Sampler",
        "role": "sampler",
        "task": "classification",
        "description": "Drops majority-class samples; cheap way to balance very skewed targets.",
        "params": {"sampling_strategy": {"type": "real", "low": 0.3, "high": 1.0}},
    },
    # --- transformers (shared by both problem types) ---------------------
    {
        "fqn": "sklearn.preprocessing.StandardScaler",
        "name": "Standard Scaler",
        "role": "transformer",
        "description": "Zero-mean, unit-variance scaling. No searchable parameters; the engine already scales numerics per fold for presets.",
        "params": {},
    },
    {
        "fqn": "sklearn.preprocessing.RobustScaler",
        "name": "Robust Scaler",
        "role": "transformer",
        "description": "Median/IQR scaling; tolerant of outliers. No searchable parameters.",
        "params": {},
    },
    {
        "fqn": "sklearn.impute.SimpleImputer",
        "name": "Simple Imputer",
        "role": "transformer",
        "description": "Mean/median/most-frequent imputation for missing values.",
        "params": {"strategy": {"type": "categorical", "items": ["mean", "median", "most_frequent"]}},
    },
    {
        "fqn": "sklearn.decomposition.PCA",
        "name": "PCA",
        "role": "transformer",
        "description": "Variance-based dimensionality reduction. Integer component counts here; fractional targets (e.g. 0.95) via Raw mode.",
        "params": {"n_components": {"type": "integer", "low": 2, "high": 32}},
    },
    {
        "fqn": "sklearn.feature_selection.SelectKBest",
        "name": "Select K Best",
        "role": "transformer",
        "description": "Univariate feature ranking; keeps the top-k by score.",
        "params": {"k": {"type": "integer", "low": 2, "high": 50}},
    },
    {
        "fqn": "sklearn.feature_selection.VarianceThreshold",
        "name": "Variance Threshold",
        "role": "transformer",
        "description": "Drops near-constant features.",
        "params": {"threshold": {"type": "real", "low": 0.1, "high": 5.0}},
    },
    {
        "fqn": "SKSurrogate.SensAprx",
        "name": "Sensitivity Approximation (SALib)",
        "role": "transformer",
        "description": "SKSurrogate's own sensitivity-analysis selector (Sobol/Morris/Delta). Expensive per fit — not preselected.",
        "params": {
            "n_features_to_select": {"type": "integer", "low": 3, "high": 20},
            "method": {"type": "categorical", "items": ["sobol", "morris", "delta-mmnt"]},
        },
    },
    {
        "fqn": "SKSurrogate.CorrelationThreshold",
        "name": "Correlation Threshold",
        "role": "transformer",
        "description": "SKSurrogate's own Pearson-correlation pruning selector; cheap.",
        "params": {"threshold": {"type": "real", "low": 0.5, "high": 0.95}},
    },
]

def components_for_task(task: str | None) -> list[dict[str, object]]:
    """Components whose role fits ``task``; transformers always qualify."""
    out = []
    for row in EXPERIMENT_COMPONENTS:
        if row.get("role") == "transformer" or row.get("task") == task:
            out.append(row)
    return out


def validate_catalog() -> list[str]:
    """Return a list of structural problems (empty when the table is sound)."""
    problems: list[str] = []
    seen: set[str] = set()
    for row in EXPERIMENT_COMPONENTS:
        fqn = str(row["fqn"])
        if fqn in seen:
            problems.append(f"duplicate fqn {fqn}")
        seen.add(fqn)
        if not isinstance(row.get("params"), dict):
            problems.append(f"{fqn}: params must be a dict")
            continue
        for pname, spec in row["params"].items():
            stype = spec.get("type")
            if stype in ("real", "integer"):
                if not (isinstance(spec.get("low"), (int, float)) and isinstance(spec.get("high"), (int, float))):
                    problems.append(f"{fqn}.{pname}: real/integer specs need low/high")
            elif stype == "categorical":
                if not isinstance(spec.get("items"), list) or not spec["items"]:
                    problems.append(f"{fqn}.{pname}: categorical specs need non-empty items")
            else:
                problems.append(f"{fqn}.{pname}: unknown spec type {stype!r}")
    return problems
