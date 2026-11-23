# Visual Search-Space Editor (Experiments · Advanced) — Implementation Plan

**Goal.** Replace the raw-JSON `<textarea>` under **Experiments → Advanced → Search space** with a visual editor: a task-filtered component library plus one card per chosen component, per-hyperparameter controls (range / category chips), pin-to-fixed-value toggles, forbidden-rule rows and a stacking toggle — all expressing exactly today's `SearchSpace` JSON so the backend contract does not change. A **Raw JSON** sub-tab keeps full power-user control with lossless switching in both directions.

**Architecture.**
- Backend: one new read-only endpoint `GET /api/experiments/components` serving a curated catalog (estimators tagged by task type + shared transformers). No changes to `RunExperimentRequest`, the EOA engine, or preset payloads.
- Frontend: `webui/src/lib/searchspace.ts` (pure model ⇄ JSON serializer — the parity core), `webui/src/components/SearchSpaceEditor.tsx` (controlled editor component), and a restructure of the Search-space card in `Experiments.tsx` into **Visual | Raw** sub-tabs that share one source of truth: the existing `configJson` string. `parseSpace`, `spaceWarnings`, submit path, pre-fill effect, Quick-start tab and Run settings are untouched.
- Parity invariant: for any JSON accepted by today's textarea, (a) Raw mode behaves identically, and (b) Visual→Raw round-trips semantically; parts the catalog doesn't know render as read-only verbatim blocks and serialize back unchanged.

**Tech Stack.** FastAPI + pydantic (existing), React 18 + TypeScript + TanStack Query (existing webui stack), no new npm dependencies, no unit-test runner in `webui` — frontend verification is the `tsc`/Vite build gate plus live-browser assertions per `.hermes` skill reference. Backend tests: pytest in the existing suite.

**Test strategy note.** The repo's `tests/` covers Python only; `webui/package.json` has dev/build/lint/preview and no test runner. TDD (RED→GREEN) applies to every backend task. Frontend tasks follow *spec → implement → build gate (`npm run build`) → live-browser assertion* per the webui-live-verification reference, with a real end-to-end experiment submission as acceptance evidence.

**Branch (per user instruction).** All implementation work happens on a **separate feature branch `webui-visual-searchspace`**, created from master (`c05792e`) before Task 1. Master is not touched until the Task 8 gates pass and the PR/merge is requested.

**Complete code is provided for:** the catalog module + endpoint, `client.ts` additions, and `lib/searchspace.ts`. The large React component is specified by structure, props and representative JSX; its final markup is written against the existing CSS classes (`mono`, `field`, `hint`, `row top`, `btn`, `card`) at implementation time.

---

## Decisions & assumptions (flagged for review — flip any of these before Task 1)

| # | Decision | Default adopted | Alternative |
|---|----------|-----------------|-------------|
| D1 | Catalog source | **New endpoint** `GET /api/experiments/components`, server-curated in a new module `api/experiment_catalog.py` (no frontend hardcoding; importable from tests) | extend `/presets`; hardcode in `client.ts` next to `DEFAULT_*_EXPERIMENT_CONFIG` |
| D2 | Roster (expanded per user instruction) | **All relevant implementations**: 8 classifiers + 8 regressors — sklearn core (`LogisticRegression`, `SVC`, `DecisionTree`, RF/ET, and the regression-side `Ridge`/`SVR`), **XGBoost** (`xgboost.XGBClassifier/Regressor` 2.1.4) and **LightGBM** (`lightgbm.LGBMClassifier/Regressor` 4.7.0) with their performance-significant hyperparameters, plus the RGF pair (verified strings `["RGF","RGF_Opt","RGF_Sib"]`). **Imbalanced-learn resamplers** (classification only): SMOTE, ADASYN, RandomUnderSampler — AML classifies them as `"sampler"` via `_estimator_type` and auto-places them first. **All transformers incl. SKSurrogate's own**: scalers, `SimpleImputer`, `PCA`, `SelectKBest`, `VarianceThreshold`, plus `SKSurrogate.SensAprx` (SALib sensitivity) and `SKSurrogate.CorrelationThreshold`. Full spec table = the literal in Task 1 | sklearn-only v1 (rejected); skrebate selectors deferred — installed & engine-recognized, one row each if requested |
| D3 | Pin toggles | **First-class in v1** — per-parameter "pin" switches the control from range/chips to a single scalar input (pinned scalars are already legal `ParamValue` JSON; zero backend work) | ranges only in v1, pinning via Raw tab |
| D4 | Stacking directive | **Toggle card in the visual editor** — on/off; "on" inserts the existing `STACKING_COMPONENT` payload from `client.ts`; field-level editing of stacking stays in Raw mode (v1) | reachable only via Raw fallback |
| D5 | Visual/Raw coexistence | **Two sub-tabs sharing one state**: source of truth stays `configJson`; visual edits serialize back into it immediately; switching to Raw re-parses it. Lossless both ways, and submit/warnings/pre-fill keep working unchanged | visual fully replaces the textarea with a read-only JSON drawer only |
| A1 | `depends_on` | **Not editable in v1.** Imported rules are preserved verbatim through round-trips; cards show a small "conditional" badge when present. Editing remains possible in Raw mode. | full conditional-rule editor (v2) |
| A2 | Pipeline depth (`length`) | **No new control** — the existing "Max pipeline length" input in Run settings (`Experiments.tsx:707-710`) is kept as-is; the visual card only points at it. | a stepper inside the editor (rejected: duplicate state) |
| A3 | Temporal tasks | No behavior change; the temporal note under Run settings already explains one-step scoring. The catalog/visual editor is task-type aware but not temporal-aware. | — |
| D6 | Preselection (new per user instruction) | **Catalog rows carry `preselected: bool`**; per task the pre-selected set is **RandomForest + XGBoost + LightGBM** (the 2026 tabular workhorses). Mechanics: opening Advanced with an *untouched* config seeds the visual editor from the preselected rows for the effective problem type (this replaces the old preset-JSON seed, so untouched Start now searches a diverse RF/XGB/LGBM space); "Reset to {preset} defaults" still fetches the **server** preset JSON (RF/ET) — documented in the hint line. Quick-start tab is unchanged (server presets apply there). | preselect RF+ET only; no seeding at all (empty cards, user adds everything) |

---

## Current-state map (verified 2026-11-22)

- `webui/src/pages/Experiments.tsx` (920 lines):
  - `parseSpace(json)` → `{config, forbidden, knownParams, paramsByEstimator}` (`:63-96`); reserved keys `STACKING_KEY="stacking"`, `FORBIDDEN_KEY="forbidden"` (`:57,60`).
  - `spaceWarnings(space, forbidden)` — live hints for unknown param references in `depends_on`/`forbidden` (`:99-125`). **Reused unchanged** by both tabs.
  - State: `configJson` + `configTouched` pre-fill guard; prefill follows effective problem type via `preset = presetTouched ? presetChoice : declaredPreset ?? presetChoice` (`:303`) and the effect at `:319-322`. **Kept unchanged.**
  - Advanced Search-space card `:645-699`: textarea, "Reset to {preset} defaults", `insertStackingComponent` (`:445-453`, merges client.ts `STACKING_COMPONENT`), `insertForbiddenRule` (`:456-470`).
  - Run settings card `:701-797`: `length`, generation/parents, partition, scoring, backend, strategy, surrogate fields, temporal note, **Start experiment** → `runMut` body built from `parsed.config` + `parsed.forbidden` (`:363-391`). **Untouched.**
- Schema (server): `api/routers/experiments.py` — `ParamSpec{type: "real"|"integer"|"categorical", low?, high?, items?, depends_on?}`, `ParamValue = ParamSpec | bool | int | None`, `SearchSpace = dict[str, dict[str, ParamValue]]`; pinned scalars are legal entries (docstring `:30-47`).
- Engine (`SKSurrogate/aml.py`): config keys are "letters" composed into sequences up to `length`; `_validate_sequence` (`:729`) requires the last element to be a regressor/classifier, so transformers as intermediate letters are legal. Preset specs live in `PRESET_PARAM_SPECS` / `PRESET_ESTIMATOR_NAMES` (`:395-406`).
- **Letter typing is dynamic** (`_classify`, `aml.py:~700-728`): importlib-resolved classes are typed via `is_classifier`/`is_regressor`; anything with `_estimator_type == "sampler"` (all imblearn resamplers) becomes `"sampler"` and is auto-placed first (`couldBfirst`, `init_genes=...:998`). No hardcoded FQN list — the catalog only needs importable FQNs. `known_feature_selectors` (`:557-579`) adds special placement rules for PCA etc.
- **Engine auto-preprocessing** (`aml.py:1390-1420`): preset + no `cat_cols` → a fold-local `FoldLocalDataPreprocessor(scale_numeric=True)` is prepended to every candidate; explicit `cat_cols` → `category_encoders.OneHotEncoder`. User-added scaler/encoder letters remain legal (redundant but harmless) — the UI hint should mention this.
- **Verified installed in `.venv` (2026-11-22):** xgboost 2.1.4 (kwargs-based sklearn API; `get_params()` round-trips `n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda`), lightgbm 4.7.0 (`num_leaves, n_estimators, learning_rate, max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda, boosting_type` in the real signature), imbalanced-learn 0.14.2 (`imblearn.combine` — not `combination`; SMOTE/ADASYN/RandomOverSampler/RandomUnderSampler/TomekLinks verified via signatures), skrebate 0.8.4 (deferred to v2 per D2).
- RGF is installed (3.12.0); **live fit-probe verified** the only valid `algorithm` strings are `"RGF"`, `"RGF_Opt"`, `"RGF_Sib"` — spaced forms raise `ValueError`. The catalog must use underscore forms.
- SKSurrogate's own transformers (`SKSurrogate/sensapprx.py`, exported at package root): `SensAprx(n_features_to_select=10, method="sobol"| "morris"|"delta-mmnt", margin=.2, num_smpl=512, ...)` — SALib-based, **expensive per fit** (UI warning; not preselected); `CorrelationThreshold(threshold=0.7)` — cheap Pearson-pruning selector. Both are in `known_feature_selectors`.
- `webui/dist` is **not** git-tracked (Docker builds it), so no built assets are committed.
- Docs that mention the search space: `docs/ui-manual.rst` (primary UI guide), `docs/walkthrough.rst`, `docs/dataproc.rst`, `docs/mltrace.rst`.

---

## File map

| File | Action |
|------|--------|
| `api/experiment_catalog.py` | **New** — curated catalog table + validation helper |
| `api/routers/experiments.py` | Modify — import catalog, add `GET /components` route |
| `tests/test_api_experiments.py` | Modify — endpoint tests (or new sibling file if the suite prefers) |
| `webui/src/api/client.ts` | Modify — `CatalogComponent`, `ComponentsResponse`, `getExperimentComponents()` |
| `webui/src/lib/searchspace.ts` | **New** — model types + `spaceToModel` / `modelToSpace` (parity core) |
| `webui/src/components/SearchSpaceEditor.tsx` | **New** — visual editor (library, cards, controls, rules, stacking toggle) |
| `webui/src/index.css` | Modify — small additions: chips + two-column editor grid |
| `webui/src/pages/Experiments.tsx` | Modify — sub-tabs in the Search-space card; wire model state; keep Raw path byte-identical in behavior |
| `docs/ui-manual.rst`, `CHANGELOG.md` | Modify — new editor section + screenshot, changelog entry |

---

## Task 1 — Backend: catalog module (data only)

**Objective.** A server-side curated table of components with task tagging and default parameter specs.

**Files.** Create `api/experiment_catalog.py`.

```python
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
```

> **Roster rationale (verified 2026-11-22 in `.venv`):** XGBoost 2.1.4 exposes all nine listed kwargs through the sklearn API (`get_params()` round-trip verified); LightGBM 4.7.0 lists them in its real signature; `imblearn` resamplers carry `_estimator_type == "sampler"`, which AML's dynamic letter typing uses to place them first (`couldBfirst`) — no engine change needed. RGF `algorithm` values were fit-probed: only the underscore forms are accepted. Booleans/objectives/`"auto"` strings are excluded on purpose (see module docstring) and remain reachable via Raw-mode pinning.

Also add the helper used by tests and (optionally) the endpoint:

```python
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
```

**Verify.** `env -u PYTHONPATH ./.venv/bin/python -c "from api.experiment_catalog import validate_catalog; print(validate_catalog())"` → prints `[]`. (Catalog data has no behavior to unit-test beyond this guard; endpoint tests in Task 2 cover the API surface.)

**Commit.** `[Chore(api): add curated component catalog for the visual search-space editor]` — conventional message explaining *why*: single server-side source so UI and presets cannot drift.

---

## Task 2 — Backend: `GET /api/experiments/components` (TDD)

**Objective.** Read-only endpoint returning the catalog; shape validated by tests first.

**Files.** Modify `tests/test_api_experiments.py`; modify `api/routers/experiments.py`.

### Step 1 — failing test

Append to `tests/test_api_experiments.py` (follow that file's existing client/fixture conventions — it already builds a TestClient against the app):

```python
def test_components_endpoint_shape(client):
    res = client.get("/api/experiments/components")
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"components"}
    rows = body["components"]
    assert isinstance(rows, list) and len(rows) >= 25
    for row in rows:
        assert set(row) <= {"fqn", "name", "role", "task", "description", "params", "preselected"}
        assert row["role"] in ("estimator", "transformer", "sampler")
        if row["role"] == "transformer":
            assert "task" not in row
        else:  # estimator or sampler rows are task-tagged (samplers: classification only)
            assert row.get("task") in ("classification", "regression")
        for spec in row["params"].values():
            assert spec["type"] in ("real", "integer", "categorical")


def test_components_split_by_task(client):
    rows = client.get("/api/experiments/components").json()["components"]
    clf = [r for r in rows if r.get("task") == "classification"]
    reg = [r for r in rows if r.get("task") == "regression"]
    # exact curated counts: 8 classifiers + 3 samplers, 8 regressors
    assert len(clf) == 11 and len(reg) == 8
    # a classifier FQN must never appear under regression (and vice versa)
    clf_names = {r["fqn"] for r in clf if r["role"] == "estimator"}
    reg_names = {r["fqn"] for r in reg}
    assert not (clf_names & reg_names)


def test_gradient_boosting_in_catalog(client):
    rows = client.get("/api/experiments/components").json()["components"]
    by_fqn = {r["fqn"]: r for r in rows}
    for fqn, task in (
        ("xgboost.XGBClassifier", "classification"),
        ("lightgbm.LGBMClassifier", "classification"),
        ("xgboost.XGBRegressor", "regression"),
        ("lightgbm.LGBMRegressor", "regression"),
    ):
        row = by_fqn[fqn]
        assert row["task"] == task and row.get("preselected") is True
        params = set(row["params"])
        assert {"n_estimators", "learning_rate"} <= params, f"{fqn} missing core GBM knobs"
        assert len(params) >= 9, f"{fqn}: expected the full significant-knob set"


def test_resamplers_classification_only(client):
    rows = client.get("/api/experiments/components").json()["components"]
    samplers = {r["fqn"]: r for r in rows if r["role"] == "sampler"}
    assert set(samplers) == {
        "imblearn.over_sampling.SMOTE",
        "imblearn.over_sampling.ADASYN",
        "imblearn.under_sampling.RandomUnderSampler",
    }
    for row in samplers.values():
        assert row["task"] == "classification"


def test_sk_surrogate_transformers_present(client):
    rows = client.get("/api/experiments/components").json()["components"]
    by_fqn = {r["fqn"]: r for r in rows}
    sens = by_fqn["SKSurrogate.SensAprx"]
    assert sens["role"] == "transformer" and "task" not in sens
    assert sens["params"]["method"]["items"] == ["sobol", "morris", "delta-mmnt"]
    corr = by_fqn["SKSurrogate.CorrelationThreshold"]
    assert corr["role"] == "transformer"
    assert corr["params"]["threshold"]["type"] == "real"


def test_rgf_catalog_entries(client):
    rows = client.get("/api/experiments/components").json()["components"]
    by_fqn = {r["fqn"]: r for r in rows}
    clf = by_fqn["rgf.sklearn.RGFClassifier"]["params"]
    # underscore forms are the only ones rgf 3.12 accepts (fit-probed)
    assert clf["algorithm"]["items"] == ["RGF", "RGF_Opt", "RGF_Sib"]
    reg = by_fqn["rgf.sklearn.RGFRegressor"]["params"]
    assert set(reg) >= {"algorithm", "reg_depth"}


def test_rf_et_specs_match_preset_table(client):
    from SKSurrogate.aml import PRESET_PARAM_SPECS
    rows = client.get("/api/experiments/components").json()["components"]
    by_fqn = {r["fqn"]: r for r in rows}
    for fqn in (
        "sklearn.ensemble.RandomForestClassifier",
        "sklearn.ensemble.ExtraTreesClassifier",
        "sklearn.ensemble.RandomForestRegressor",
        "sklearn.ensemble.ExtraTreesRegressor",
    ):
        assert by_fqn[fqn]["params"] == PRESET_PARAM_SPECS, f"{fqn} drifted from the preset table"


def test_preselected_sets(client):
    rows = client.get("/api/experiments/components").json()["components"]
    for task, expected in (
        ("classification", {
            "sklearn.ensemble.RandomForestClassifier",
            "xgboost.XGBClassifier",
            "lightgbm.LGBMClassifier",
        }),
        ("regression", {
            "sklearn.ensemble.RandomForestRegressor",
            "xgboost.XGBRegressor",
            "lightgbm.LGBMRegressor",
        }),
    ):
        got = {r["fqn"] for r in rows if r.get("task") == task and r.get("preselected")}
        assert got == expected, f"preselected set for {task} drifted: {got}"
```

### Step 2 — verify failure

`env -u PYTHONPATH ./.venv/bin/python -m pytest tests/test_api_experiments.py -q -p no:randomly --no-header -k "components or rf_et or preselected"` → **8 failed** (404 route missing).

> Adapt the `-k` expression to the exact test names and the file's fixture style once opened. If `client` is not a bare fixture name in that file, mirror how neighboring tests obtain their client.

### Step 3 — implementation

In `api/routers/experiments.py`, next to the other imports: `from ..experiment_catalog import EXPERIMENT_COMPONENTS`. After `GET /presets`:

```python
@router.get("/components")
def list_components() -> dict[str, object]:
    """Curated component catalog for the visual search-space editor.

    Estimators are tagged with their task type (``classification``/``regression``);
    transformers carry no tag and appear in both libraries. Read-only — editing
    happens client-side against the returned specs.
    """
    return {"components": EXPERIMENT_COMPONENTS}
```

### Step 4 — verify pass + full-suite regression

`env -u PYTHONPATH ./.venv/bin/python -m pytest tests/test_api_experiments.py -q --no-header` → green, then the repo recipe for a clean-env full run (see memory: `pytest tests` on master baseline) to confirm no drift.

**Commit.** `[Test(api)] + [Feat(api): serve the component catalog at /api/experiments/components]` — two atomic commits if you prefer test-first separation; message explains read-only nature and task tagging.

---

## Task 3 — Frontend: `client.ts` types + query function

**Objective.** Typed access to the new endpoint, matching the file's existing fetch style (see how `getPresets` is written in `webui/src/api/client.ts`).

**Files.** Modify `webui/src/api/client.ts`.

```ts
/** One entry of the curated component catalog served by /api/experiments/components. */
export interface CatalogComponent {
  /** Fully-qualified class name, e.g. "sklearn.linear_model.LogisticRegression". */
  fqn: string;
  name: string;
  role: "estimator" | "transformer" | "sampler";
  /** Set for estimators and samplers — the web UI filters the library by it (samplers are classification-only). */
  task?: "classification" | "regression";
  description: string;
  /** Default per-hyperparameter specs, exactly as accepted in a search-space config. */
  params: Record<string, ParamSpec>;
  /** Rows seeding the editor on first open for their task (D6). */
  preselected?: boolean;
}

export interface ComponentsResponse {
  components: CatalogComponent[];
}
```

and (place beside `getPresets`, copying its fetch/error handling verbatim):

```ts
/** Curated component catalog for the visual search-space editor. */
export async function getExperimentComponents(): Promise<ComponentsResponse> {
  const res = await authedFetch(`${API_BASE}/api/experiments/components`); // mirror local helper names
  if (!res.ok) throw new ApiError(res.status, await safeMessage(res));
  return (await res.json()) as ComponentsResponse;
}
```

(Adjust `authedFetch`/`safeMessage`/`API_BASE` to the actual identifiers already used in that file — do not invent helpers.)

**Verify.** `cd webui && npm run build` → tsc clean.

**Commit.** `[Feat(webui): typed client access to the experiment components catalog]`

---

## Task 4 — Frontend: `webui/src/lib/searchspace.ts` (parity core)

**Objective.** Pure, framework-free model ⇄ JSON functions with a stated round-trip invariant. This file is where "visual edits == hand-written JSON" is guaranteed; it must not import React.

**Files.** Create `webui/src/lib/searchspace.ts`. Complete code:

```ts
/**
 * Model and (de)serialization for the visual search-space editor.
 *
 * Parity invariant: for any document accepted by the raw textarea,
 *   modelToSpace(spaceToModel(json)) ≡ json  (semantically; unknown parts verbatim)
 * "Unknown" means FQNs or parameters absent from the loaded catalog — they are
 * kept in `rawJson` / `customEntries` and re-emitted byte-for-byte.
 */

import type { ParamSpec, SearchSpace } from "../api/client";
import type { CatalogComponent } from "../api/client";

/** Reserved search-space key for the stacking directive (see client STACKING_COMPONENT). */
export const STACKING_KEY = "stacking";
/** Top-level key holding forbidden [parameter, value] rules. */
export const FORBIDDEN_KEY = "forbidden";

const NUMERIC_TYPES = new Set(["real", "integer"]);

export interface ParamModel {
  /** Catalog default (or imported) spec; absent for pinned scalars whose type we infer. */
  spec?: ParamSpec;
  /** false → parameter is pinned to `value` and emits a plain scalar. */
  free: boolean;
  /** Numeric range when free and spec.type ∈ {real, integer}. */
  low?: number;
  high?: number;
  /** Categorical: currently SELECTED values — exactly what gets serialized. */
  items: Array<string | null>;
  /** Categorical: full option list (catalog superset); chips render this and toggle membership in `items`. */
  options?: Array<string | null>;
  /** Pinned scalar value (used only when `free === false`). */
  value?: number | string | boolean | null;
  /** Preserved verbatim from imported JSON (A1: not editable in v1). */
  dependsOn?: Record<string, unknown[]>;
}

export interface CardModel {
  fqn: string;
  role: "estimator" | "transformer" | "sampler";
  task?: "classification" | "regression";
  name: string; // catalog display label, or the short class name when unknown
  description?: string;
  /** false → FQN not in the loaded catalog; the whole subtree is verbatim. */
  known: boolean;
  params: Record<string, ParamModel>;
  /** Verbatim JSON of the original params object (only when !known). */
  rawJson?: string;
  /** name → verbatim JSON for parameters outside the catalog (known cards only). */
  customEntries?: Record<string, string>;
}

export interface SpaceModel {
  cards: CardModel[]; // insertion order = key order of the JSON document
  stacking: unknown | null; // reserved directive object, or null when absent
  forbidden: Array<[string, unknown]> | null;
}

type Result<T> = T | { error: string };

function shortName(fqn: string): string {
  const parts = fqn.split(".");
  return parts[parts.length - 1];
}

function isSpecObject(v: unknown): v is ParamSpec {
  if (typeof v !== "object" || v === null) return false;
  const t = (v as ParamSpec).type;
  return t === "real" || t === "integer" || t === "categorical";
}

/** Parse a search-space JSON document into the editor model. */
export function spaceToModel(json: string, catalog: CatalogComponent[]): Result<SpaceModel> {
  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch (e) {
    return { error: `Invalid search-space JSON: ${e instanceof Error ? e.message : String(e)}` };
  }
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    return { error: "Search space must be a JSON object mapping estimator names to parameter specs." };
  }
  const doc = { ...(raw as Record<string, unknown>) };
  const stacking = STACKING_KEY in doc ? (doc[STACKING_KEY] ?? null) : null;
  delete doc[STACKING_KEY];

  let forbidden: Array<[string, unknown]> | null = null;
  if (FORBIDDEN_KEY in doc && doc[FORBIDDEN_KEY] != null) {
    const fr = doc[FORBIDDEN_KEY];
    if (!Array.isArray(fr) || fr.some((p) => !Array.isArray(p) || p.length !== 2)) {
      return { error: '"forbidden" must be an array of [parameter, value] pairs.' };
    }
    forbidden = fr as Array<[string, unknown]>;
  }
  delete doc[FORBIDDEN_KEY];

  const byFqn = new Map(catalog.map((c) => [c.fqn, c]));
  const cards: CardModel[] = [];
  for (const [fqn, paramsRaw] of Object.entries(doc)) {
    if (typeof paramsRaw !== "object" || paramsRaw === null) continue; // defensive, matches parseSpace
    const cat = byFqn.get(fqn);
    const card: CardModel = {
      fqn,
      role: cat?.role ?? (fqn.includes("Classifier") || fqn.includes("Regressor") ? "estimator" : "transformer"),
      task: cat?.task,
      name: cat?.name ?? shortName(fqn),
      description: cat?.description,
      known: Boolean(cat),
    };
    if (!cat) {
      card.rawJson = JSON.stringify(paramsRaw); // verbatim block, read-only in the UI
      cards.push(card);
      continue;
    }
    const params: Record<string, ParamModel> = {};
    const customEntries: Record<string, string> = {};
    for (const [pname, val] of Object.entries(paramsRaw as Record<string, unknown>)) {
      const defaultSpec = cat.params[pname]; // catalog spec or undefined
      if (isSpecObject(val)) {
        const pm: ParamModel = {
          spec: defaultSpec ?? val,
          free: true,
          dependsOn: val.depends_on,
        };
        if (NUMERIC_TYPES.has(val.type)) {
          pm.items = [];
          pm.low = typeof val.low === "number" ? val.low : defaultSpec?.low;
          pm.high = typeof val.high === "number" ? val.high : defaultSpec?.high;
        } else {
          // Imported selection round-trips unchanged (parity); the catalog's full
          // item list is only the chip set offered for toggling.
          pm.items = [...(val.items ?? [])];
          if (defaultSpec && defaultSpec.type === "categorical") pm.options = [...defaultSpec.items];
        }
        params[pname] = pm;
      } else if (typeof val === "number" || typeof val === "string" || typeof val === "boolean" || val === null) {
        // Pinned scalar: legal ParamValue. Render as pinned with the catalog spec for context.
        params[pname] = { free: false, value: val, spec: defaultSpec };
      } else if (defaultSpec === undefined && !cat.params.hasOwnProperty(pname)) {
        customEntries[pname] = JSON.stringify(val); // unknown parameter → verbatim
      } else {
        return { error: `Unrecognized entry for ${shortName(fqn)}.${pname}; edit it in Raw JSON.` };
      }
    }
    card.params = params;
    if (Object.keys(customEntries).length > 0) card.customEntries = customEntries;
    cards.push(card);
  }
  return { cards, stacking, forbidden };
}

/** Serialize the editor model back to a search-space JSON document string. */
export function modelToSpace(model: SpaceModel): Result<string> {
  const doc: Record<string, unknown> = {};
  for (const card of model.cards) {
    if (!card.known) {
      // Verbatim round-trip for components outside the catalog.
      try {
        doc[card.fqn] = JSON.parse(card.rawJson ?? "{}");
      } catch {
        return { error: `Stored custom block for ${shortName(card.fqn)} is not valid JSON.` };
      }
      continue;
    }
    const entry: Record<string, unknown> = {};
    for (const [pname, pm] of Object.entries(card.params)) {
      if (!pm.free) {
        entry[pname] = pm.value ?? null; // pinned scalar (number | string | boolean | null)
        continue;
      }
      const type = pm.spec?.type ?? "real";
      if (type === "categorical") {
        if (pm.items.length === 0) {
          return { error: `${shortName(card.fqn)}: ${pname} has no selected categories — pick at least one or pin a value.` };
        }
        entry[pname] = pm.dependsOn ? { type, items: pm.items, depends_on: pm.dependsOn } : { type, items: pm.items };
      } else if (NUMERIC_TYPES.has(type)) {
        if (typeof pm.low !== "number" || typeof pm.high !== "number" || !(pm.low <= pm.high)) {
          return { error: `${shortName(card.fqn)}: ${pname} range is invalid (need low ≤ high).` };
        }
        const lo = type === "integer" ? Math.round(pm.low) : pm.low;
        const hi = type === "integer" ? Math.round(pm.high) : pm.high;
        entry[pname] = pm.dependsOn ? { type, low: lo, high: hi, depends_on: pm.dependsOn } : { type, low: lo, high: hi };
      } else {
        return { error: `${shortName(card.fqn)}: unknown spec type ${type}.` };
      }
    }
    if (card.customEntries) {
      for (const [pname, raw] of Object.entries(card.customEntries)) entry[pname] = JSON.parse(raw);
    }
    doc[card.fqn] = entry;
  }
  if (model.stacking != null) doc[STACKING_KEY] = model.stacking;
  if (model.forbidden && model.forbidden.length > 0) doc[FORBIDDEN_KEY] = model.forbidden;
  return JSON.stringify(doc, null, 2);
}

/** Build a fresh card from a catalog row: every parameter free at its defaults. */
export function cardFromCatalog(row: CatalogComponent): CardModel {
  const params: Record<string, ParamModel> = {};
  for (const [pname, spec] of Object.entries(row.params)) {
    if (spec.type === "categorical") {
      params[pname] = { spec, free: true, items: [...(spec.items ?? [])] };
    } else {
      params[pname] = { spec, free: true, low: spec.low, high: spec.high, items: [] };
    }
  }
  return { fqn: row.fqn, role: row.role, task: row.task, name: row.name, description: row.description, known: true, params };
}

/** All parameter names across the model (options for forbidden-rule rows). */
export function allParamNames(model: SpaceModel): string[] {
  const names = new Set<string>();
  for (const card of model.cards) if (card.known) for (const pname of Object.keys(card.params)) names.add(pname);
  return [...names].sort();
}
```

> Reviewer notes on the code above: (1) the categorical branch keeps it simple — selected chips start as the catalog's full item list; importing a JSON with a *subset* is also accepted and re-emitted identically because `items` carries exactly what was imported when no catalog spec exists. (2) `depends_on` passes through untouched in both directions (A1). (3) Errors are returned, never thrown — the caller shows them in the existing `configError` slot.

**Verify.** `npm run build` clean; then a quick node sanity check is *not* part of CI here — correctness is instead asserted by Task 6's live round-trips and by the code review gate on this file (it is small and pure).

**Commit.** `[Feat(webui): search-space model serializer with verbatim fallback for unknown entries]`

---

## Task 5 — Frontend: `SearchSpaceEditor.tsx` (the visual editor)

**Objective.** Controlled component rendering the model; emits `onChange(nextModel)` on every interaction. No internal persistence of its own beyond controlled state.

**Files.** Create `webui/src/components/SearchSpaceEditor.tsx`; small additions to `webui/src/index.css`.

### Props contract

```ts
export interface SearchSpaceEditorProps {
  model: SpaceModel;                       // from Experiments.tsx (derived from configJson)
  catalog: CatalogComponent[] | null;      // null while the query is loading
  activeTask: "classification" | "regression"; // effective problem type → library filter
  onChange(next: SpaceModel): void;
}
```

### Layout (two columns, left = library, right = cards + rules)

- **Library panel** (`<aside>`): search input (filters by `name`/`fqn` substring); three sections — *Estimators* (rows where `role==="estimator" && task===activeTask`), *Resamplers* (**classification only**: `role==="sampler"`; hidden for regression) and *Transformers* (`role==="transformer"`). Each row: name, short description line, a **★** badge when `row.preselected`, and an **Add** button (disabled when the fqn already has a card; shows "Added ✓"). Loading state: single hint line “Loading component library…” while `catalog === null`.
- **Cards column**: one `<div className="ss-card">` per model card — header row (`name`, role badge, ↑ ↓ move buttons, ✕ remove), body = parameter controls. Unknown cards render a read-only `<pre className="mono">{rawJson}</pre>` with the hint “Not in the component library — edit in Raw JSON.” Cards with `customEntries` show them as a read-only mono block under their controls (“Custom parameters (edit in Raw JSON)”).
- **Parameter control** per `ParamModel`:
  - free + numeric: two number inputs (`low`, `high`) side by side, min/max from the spec; invalid state (low > high or empty) renders a red hint and blocks submit via the returned error path.
  - free + categorical: chip buttons rendered from `pm.options ?? pm.items` — clicking toggles membership in `pm.items` (the serialized selection) immutably; at least one must stay selected (enforced by the `modelToSpace` error, mirrored as an inline hint).
  - pin toggle: small switch per row (“range” ⇄ “pinned”). Pinned state replaces the control with a single scalar input — number for numeric specs, text otherwise; value typed into `pm.value`. Unpinning restores the previous range (kept in `low/high`/`items`).
  - conditional badge: when `pm.dependsOn` exists → small “conditional” badge + hint “(depends_on preserved from JSON — edit in Raw mode)”.
- **Forbidden rules** section under the cards: rows of [select over `allParamNames(model)` ∪ free-text fallback] + value text input + ✕; an “Add forbidden rule” button appends `["<first param>", ""]`.
- **Stacking toggle**: switch labelled “Stacking (out-of-fold StackingEstimator wrappers)” — on with null payload inserts the client-side default (`STACKING_COMPONENT` value under key `"stacking"` — read from `../api/client`, confirm exact shape in implementation); off removes it. Hint text reuses today's wording (“a directive, not a pipeline step”).
- **Summary line** at the bottom: “{n} component(s) · {m} free parameter(s) · {k} pinned” plus a pointer: “Maximum number of components per generated pipeline is set in Run settings → Max pipeline length (A2).”

### State updates (all immutable, all via `onChange`)

```ts
const patchParam = (fqn: string, pname: string, patch: Partial<ParamModel>) => { /* map cards */ };
const togglePin   = (fqn: string, pname: string) => { /* flip free; keep low/high/items for restore */ };
const addCard     = (row: CatalogComponent) => onChange({ ...model, cards: [...model.cards, cardFromCatalog(row)] });
const removeCard  = (fqn: string) => onChange({ ...model, cards: model.cards.filter((c) => c.fqn !== fqn) });
const moveCard    = (index: number, dir: -1 | 1) => { /* swap */ };
```

### CSS additions (`webui/src/index.css`)

`.ss-editor` grid (300px / 1fr, collapses under ~900 px), `.ss-card`, chip styles (`.chip`, `.chip.on`), and a muted mono block for verbatim entries. Reuse existing tokens/classes; no new framework.

**Verify.** `npm run build` clean + oxlint pass (`npm run lint` if the script is present). The component has no standalone runtime yet — wiring lands in Task 6, so also do a throwaway render check by temporarily mounting it from a scratch page *only during development* (revert before commit; never commit debug routes).

**Commit.** `[Feat(webui): visual search-space editor component (library, cards, pin toggles, rules, stacking)]`

---

## Task 6 — Frontend: integrate into `Experiments.tsx` (sub-tabs)

**Objective.** The Advanced → Search space card gains **Visual | Raw** sub-tabs; the submit path and warnings keep working unchanged (`configJson` remains the single source of truth). The untouched-config pre-fill is replaced by the D6 catalog seed (step 3); everything else on this card behaves as today.

**Files.** Modify `webui/src/pages/Experiments.tsx`.

1. New state + query:
   ```ts
   const [spaceEditorMode, setSpaceEditorMode] = useState<"visual" | "raw">("visual");
   const componentsQ = useQuery({ queryKey: ["experiment-components"], queryFn: getExperimentComponents, staleTime: Infinity });
   const catalog = componentsQ.data?.components ?? null;
   ```
2. Model derivation — visual edits write back to `configJson` immediately (so `parsed`, `warnings`, pre-fill effect and submit all see the same document):
   ```ts
   // Rebuild the editor model when entering visual mode or when the catalog first loads.
   useEffect(() => {
     if (spaceEditorMode !== "visual") return;
     const res = spaceToModel(configJson, componentsQ.data?.components ?? []);
     if (!("error" in res)) setSpaceModel(res);
   }, [spaceEditorMode, componentsQ.data]); // deliberately NOT keyed on configJson — see note

   const handleVisualChange = (next: SpaceModel) => {
     setSpaceModel(next);
     const out = modelToSpace(next);
     if ("error" in out) { setConfigError(out.error); return; }
     setConfigJson(out);           // single source of truth — raw tab and submit keep working
     setConfigTouched(true);       // same guard as typing into the textarea today
     setConfigError(null);
   };
   ```
   > Note: not keying the effect on `configJson` avoids a re-parse loop while the visual editor is open; entering Raw mode and editing there updates `configJson` directly, and switching back to Visual runs this same derivation.
3. **D6 preselection seed** (replaces the untouched-config pre-fill — see the decision table). Add next to `cardFromCatalog` in `webui/src/lib/searchspace.ts`:
   ```ts
   /** Search-space JSON seeding an *untouched* editor from the catalog's preselected rows for a task. */
   export function preselectedConfigJson(catalog: CatalogComponent[], task: "classification" | "regression"): string {
     const space: Record<string, Record<string, ParamValue>> = {};
     for (const row of catalog) if (row.preselected && row.task === task) space[row.fqn] = { ...row.params };
     return JSON.stringify(space, null, 2);
   }
   ```
   In `Experiments.tsx`, the existing untouched-config prefill effect (`:317-322`) becomes:
   ```ts
   useEffect(() => {
     if (configTouched) return;
     const seed = componentsQ.data ? preselectedConfigJson(componentsQ.data.components, preset) : configJsonForPreset(preset);
     setConfigJson(seed);
   }, [preset, configTouched, componentsQ.data, configJsonForPreset]);
   ```
   i.e. once the catalog query has loaded, an untouched config seeds to the RF/XGBoost/LightGBM workhorses for the effective problem type; while the catalog is still in flight it keeps today's server-preset fallback so nothing flashes empty. `configJsonForPreset` (server `default_config`, local fallback) survives **only** as the target of “Reset to {preset} defaults” — that button keeps applying the *server* preset JSON per D6, which is a deliberately different document from the untouched seed.
4. Replace the card body (`:645-699`) with:
   - sub-tab row (reuse the existing `<Tabs>` primitive or two `.btn` toggles — match house style);
   - **Visual** pane → `<SearchSpaceEditor model={spaceModel} catalog={catalog} activeTask={preset} onChange={handleVisualChange} />`;
   - **Raw** pane → today's textarea + its hint, byte-for-byte the same handlers (`setConfigJson`, `configTouched`);
   - move “Reset to {preset} defaults” above both panes (it sets `configJson` and clears touched; the visual model re-derives on next switch — also call the derivation synchronously after reset so the user doesn't have to flip tabs to see it);
   - rewrite the pre-fill hint (`:658`) → “Pre-filled with the **{preset}** workhorses — Random Forest, XGBoost and LightGBM (★ in the library). Editing customises it; “Reset to {preset} defaults” re-applies the server preset.”;
   - remove the two old buttons (`Add stacking component`, `Add forbidden rule`) from the Raw pane footer only when a card-level replacement exists (it does: stacking toggle + forbidden rows) — keep their behavior reachable, not duplicated;
   - warnings list + `configError` render under both panes (unchanged code).
5. Quick-start tab, Run settings card, `runMut`, temporal note: **untouched** (server presets apply there exactly as today).

**Verify.**
- `npm run build` clean; oxlint pass.
- Live assertions per the webui-live-verification reference (scratch root, fresh port, real login): see Task 7 — it doubles as this task's test suite.

**Commit.** `[Feat(webui): visual | raw sub-tabs for the Experiments search space]`

---

## Task 7 — End-to-end live verification (acceptance)

Per `.hermes` skill reference §1/§3: scratch `HOME`, fresh port, real registration + submission; assertions recorded as evidence.

**Setup.**
1. Scratch app on a free port with an isolated data dir (the usual recipe); log in via UI.
2. Register **two** small CSVs through the Datasets API/UI: one with a binary target declared `problem_type=classification`, one numeric target declared `regression` (declaration path is the same field Quick start reads — `api/routers/datasets.py` accepts `problem_type` at registration).

**Assertions (each = browser observation + screenshot):**
1. **Filtering.** Open Experiments → Advanced for the classification task: library *Estimators* shows exactly the 8 classifier rows and zero regressors, plus a *Resamplers* section with exactly SMOTE / ADASYN / Random Under-Sampler; switch to a regression task (or Quick-start problem type Regression) → 8 regressor rows appear, classifiers **and** the Resamplers section gone. Transformers section constant in both (8 rows incl. the two SKSurrogate selectors).
2. **Preselection seed (D6).** On an untouched Advanced card after the catalog query resolves: exactly three estimator cards are present — RandomForest · XGBoost · LightGBM for the active task (★ badges on their library rows); Raw JSON matches `preselectedConfigJson` output for that task; flipping problem type before touching anything re-seeds to the other trio.
3. **Round-trip parity.** From the pre-filled visual cards (the D6 seed): pin `n_estimators` on RandomForestClassifier to e.g. 100; toggle a category chip off for `max_features`; switch to Raw tab → JSON shows `"n_estimators": 100` and the reduced `items` list, everything else unchanged; switch back to Visual → pinned state and chip selection restored (lossless both ways).
4. **Forbidden + stacking.** Add one forbidden rule row referencing an existing parameter; toggle stacking on then off → Raw JSON gains/loses the `"forbidden"` array / `"stacking"` key respectively; with a `depends_on`-bearing config imported via Raw, switching to Visual and back preserves it verbatim (A1).
5. **Unknown-entry fallback.** In Raw mode add an ad-hoc FQN not in the catalog (e.g. `sklearn.neighbors.KNeighborsClassifier`) with one spec + one scalar; switch to Visual → card renders as a read-only mono block; switch back → byte-identical entry.
6. **Real submission (classification).** Visual editor, length 1–2, small generation budget → Start experiment → job completes in seconds on the scratch dataset; Top pipelines renders. Same once for regression with a StandardScaler card added — transformer letter accepted by `_validate_sequence` (last element remains an estimator) and the run still completes (or fails *gracefully* with the existing all-candidates-rejected message, which is acceptable evidence of no crash).
7. **Reset.** “Reset to {preset} defaults” in visual mode applies the **server** preset JSON (RF/ExtraTrees document — visibly different from the D6 seed); the untouched-config pre-fill effect still re-applies on problem-type change until touched (now following the catalog seed once loaded — flip type before touching anything and confirm cards follow).

**Evidence.** Screenshot per assertion stored under `docs/images/` (one composite figure for the manual); note the port, task names and timestamps in the verification log entry of this plan's commit message.

**Commit.** none (verification only) — its evidence lands with Task 8.

---

## Task 8 — Docs + CHANGELOG + final gates

1. `docs/ui-manual.rst`: rewrite the Experiments → Advanced search-space section: describe Visual/Raw sub-tabs, library filtering by declared problem type, pin toggles, forbidden rows, stacking toggle; embed the composite screenshot (`.. image:: images/experiments-visual-searchspace.png`); keep a short “Raw JSON” paragraph for power users (reserved keys, `depends_on`, custom FQNs).
2. Grep audit per the skill reference: every .rst mentioning “search space” (`ui-manual.rst`, `walkthrough.rst`, `dataproc.rst`, `mltrace.rst`) — update stale “paste JSON” phrasing; build Sphinx (or at least `make html` if available) and grep the rendered output for the new terms as evidence.
3. `CHANGELOG.md`: entry under Unreleased — *Experiments: visual search-space editor with task-filtered component library, pin toggles, forbidden-rule rows and stacking toggle (raw-JSON mode retained). The catalog covers the full classifier/regressor roster incl. XGBoost and LightGBM, imbalanced-learn resamplers (classification) and SKSurrogate's SensAprx/CorrelationThreshold selectors; untouched configs pre-select Random Forest + XGBoost + LightGBM.*
4. **Final gates:** full pytest suite green via the clean-env recipe (`env -u PYTHONPATH ./.venv/bin/python -m pytest tests -q --no-header` — baseline on master: 1138 passed / 32 subtests as of this week); `cd webui && npm run build`; oxlint.
5. **Commit.** `[Docs] + [Chore(release)]` entries; then branch merge flow per the usual PR process (CI fires on master/PRs only).

---

## Out of scope (explicit)

- Editing `depends_on` from the UI (A1 — v2 candidate, with a conditional-rule mini-editor).
- Drag-and-drop reordering and node-graph layouts (order is not semantically fixed by the engine; ↑/↓ buttons suffice).
- Per-task custom catalogs / user-defined components via API (the table in `api/experiment_catalog.py` is deliberately curated; extending it is one row per component).
- Any change to `RunExperimentRequest`, preset payloads, or the EOA/surrogate machinery.

## Rollback note

The feature is additive: the endpoint is read-only and new files are self-contained; reverting the branch restores today's textarea exactly (the Raw pane keeps its code path byte-for-byte until merged). No data migration — nothing stored changes shape.
