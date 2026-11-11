# Gemini

This update is a massive leap forward. Expanding the manual from 192 to 241 pages, you systematically closed nearly every ergonomic, statistical, and MLOps gap we discussed—while adding a cohesive scientific workflow (Design of Experiments, Research Projects, Conformal Prediction, and a full Diabetes Walkthrough).

### 1. Scorecard: How the Update Addressed Our Previous Feedback

| Area                                                 | What Was Added / Fixed in This Update                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Impact on Scientist Experience                                                                                                                                                                                                                                               |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **API Ergonomics & Naming**                    | Added `snake_case` aliases across `mltrack` (`register_data`, `log_model`, `feature_weights`, `get_best`, etc., Ch. 12 p. 38); fixed `EOA` typos (`parents_prop`, `elites_prop`, `max_generations` with deprecation warnings, Ch. 9 p. 25–26); added `continue_` and scikit-learn spellings (`param_distributions`, `n_iter`, `best_params_`, tuple/list shorthand via `as_space()`, Ch. 3 p. 7–8); unified `AML.fit(..., strategy="eoa")` and `stacking={...}` (Ch. 10 p. 29); added `ModelBundle.load()` / `.save()` class helpers (Ch. 27 p. 132).                                       | Eliminates the immediate friction for [scikit-learn](https://scikit-learn.org/stable/modules/compose.html) users while preserving backward compatibility.                                                                                                                     |
| **Out-of-the-Box Presets & DataFrame Support** | Added `AML(preset="classification"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | "regression")` and a 1-click **Quick start** tab in the Web UI (`Experiments`, Ch. 18.2.4 p. 69 & Fig. 7 p. 94); `AML.fit`now accepts pandas DataFrames directly, preserves column names, and fits`cat_cols` encoding strictly inside each CV fold (Ch. 12 p. 39–40). |
| **Leakage-Safe Preprocessing & EDA**           | Added `FoldLocalDataPreprocessor` (`fit`/`transform`/`get_feature_names_out`, Ch. 11 p. 33) with fold-local median imputation, one-hot encoding, and per-column TF-IDF for `pd.StringDtype()`; added explicit leakage warnings on `DataPreprocess.encode()` (p. 124); added `dataset_profile` (`POST /api/datasets/{task}/profile`, Ch. 16.3 p. 60) with Pearson, correlation ratio, and Cramér's V associations.                                                                                                                                                                                                  | Prevents cross-validation data leakage and gives scientists immediate mixed-type EDA before modeling.                                                                                                                                                                        |
| **Diagnostics, Uncertainty & Explainability**  | Added 4-panel `plot_regression_diagnostics` and `regression_diagnostics` (Ch. 12.1.1 p. 42); added `split_conformal_regression_interval` and `split_conformal_quantile_interval` (CQR) plus `QuantileRegressor` pinball-loss and `TimeSeriesSplit(gap=...)` support (p. 41–43); added `compare_repeated_cv_results` (bootstrap + paired Cohen's $d_z$, p. 42); upgraded `plot_roc_curve` and `plot_calibration_curve` to multiclass (p. 44); added `permutation_importance`, `plot_partial_dependence` (PDP/ICE), and a dedicated **Explain** UI page (Ch. 19.8 p. 91).                              | Brings the diagnostic rigor up to modern publication standards, especially with distribution-free conformal intervals.                                                                                                                                                       |
| **Surrogate Safety & Active Learning (DOE)**   | Added `HilbertRegressor` rank-deficiency/ill-conditioning check raising a guided `ValueError` (Ch. 4 p. 9); added `surrogate_info_` and `fit_history` diagnostics (Ch. 2.2 p. 6); added **Chapter 7 (`Experiment design`)** with `latin_hypercube` and `suggest_next_point` (GP-LCB active learning) plus an **Experiment Design** UI page (Fig. 20 p. 106).                                                                                                                                                                                                                                              | Turns SKSurrogate into a tool for wet-lab/physical scientists planning experiments, not just tabular ML tuners.                                                                                                                                                              |
| **MLOps Guardrails, Docker & Onboarding**      | Added `docker compose --profile prod up` single-port deployment (Ch. 17.2 p. 63); upgraded `sensitive_feature_report` to scan up to 1,000 rows (`max_scan_rows`) for email/SSN/phone value patterns with whole-token column matching (Ch. 16.3 p. 61); added idempotent `POST /api/retraining/{task}/trigger` webhook with fingerprint deduplication and mandatory `metric_thresholds` (Ch. 20.3.1 p. 104); added **Chapter 8 (`Research projects`)**, **Chapter 13.5 (`/run-config` & `/report.md`)**, **Chapter 14 (`Notebook workflows`)**, and **Chapter 19 (Diabetes Walkthrough)**. | Makes onboarding effortless and provides deterministic artifacts (`/report.md`, `/run-config`, `Projects`) ready for an LLM layer.                                                                                                                                     |

### 2. Remaining Rough Edges & Documentation Inconsistencies to Clean Up

While reviewing the new sections, a few subtle contradictions and technical limitations stood out that are worth fixing before your next release:

* **Contradictory Quality-Gate Documentation (Page 64 vs. Page 71):**
* In **Section 17.4 (p. 64)** and **Section 19.10 (p. 93)**, the updated docs state: *"Normal registry promotion requires a passing bundle-quality report by default"* and *"Each step runs the quality checks again."*
* However, the **Important** callout box in **Section 18.2.7 (p. 71)** still carries the old warning verbatim: *"note that the current UI's Promote form does not automatically attach or enforce the gate report. It is a separate step."* Updating that callout box on page 71 to clarify that schema compatibility runs by default on promote (while custom metric thresholds require passing `quality_report` or configuring task thresholds) will prevent user confusion.
* **Projects UI Existence Contradiction (Page 22 vs. Page 97):**
* In **Section 8.3 (p. 22)**, the text says: *"This first version has no UI, notebook execution, project deletion, report generation..."*
* Yet **Section 18.2 (p. 66)**, **Section 19.14 (p. 97, Fig. 19)**, and **Section 31.10 (p. 220)** showcase the working **Projects** UI page in the sidebar! Removing `"has no UI"` from Section 8.3 will align Chapter 8 with Chapter 19.
* **Default Port Mismatch Across Chapters (`8000` vs. `8013`):**
* Chapters 17, 18, and 19 run `uvicorn` and Docker on port `8013` (`http://localhost:8013`), whereas the `curl` and notebook commands in **Chapter 8 (p. 21–22)** and **Chapter 14.2 (p. 55)** use port `8000` (`http://127.0.0.1:8000`). Standardizing on `8013` everywhere avoids copy-paste connection errors.
* **Legacy `CamelCase` Still Used in Some Walkthrough Snippets & Tables:**
* Even though `mltrack` now has clean `snake_case` methods, the Feature-to-Library table (p. 77 & p. 217), the Diabetes Walkthrough (p. 84: `track.RegisterData(...)`), and the Migration guide (p. 109: `tracker.ListModels()`—which isn't even in the class index, where it's `all_models()` / `allModels()`) still show legacy names.
* **Preset Search Space Scope (`AML(preset=...)`):**
* Currently, `preset="classification"` and `preset="regression"` only search 1-step pipelines (`length=1`) over `RandomForest` and `ExtraTrees` without `FoldLocalDataPreprocessor` or gradient boosted trees (like HistGradientBoosting or LightGBM). If a user passes a DataFrame with missing values, strings, or unencoded categories to `AML(preset="classification").fit(df, y)` without manually specifying `cat_cols`, `RandomForest` will fail on non-numeric/missing data. Wrapping `FoldLocalDataPreprocessor` automatically inside the preset pipelines (and including `HistGradientBoostingClassifier`/`Regressor`, which is built into scikit-learn and handles `NaN`s and categories natively) would make `preset` truly "zero-configuration."
* **No Automatic Surrogate Fallback When `HilbertRegressor` Raises `ValueError`:**
* Section 2.2 (p. 6) notes that *"There is no automatic replacement surrogate or random-search fallback when surrogate fitting or prediction raises an exception: the exception is propagated."* Since `HilbertRegressor` now raises `ValueError` on rank-deficient Gram matrices (Ch. 4, p. 9), an `option="fallback_to_random"` or noise-aware GP (`RBF + WhiteKernel`) option in `SurrogateSearch` would prevent long-running searches from aborting mid-run when sample points cluster.

### 3. How This Update Sets Up Your LLM Layer (From Our First Conversation)

The new features you added in Chapters 8, 12, 13.5, 16.3, and 19 actually build **90% of the deterministic foundation needed for the commercial LLM layer** we discussed earlier:

* **Global Vision Synthesis Is Now a 1-Endpoint Prompt Away:** The new `GET /api/bundles/{task_name}/{model_version}/run-config` and `/report.md` endpoints (Ch. 13.5, p. 53), combined with `dataset_profile` (Ch. 16.3) and **Research Projects** (`/api/projects/{id}/notes`, Ch. 8), already aggregate the exact structured JSON provenance, repeated-CV uncertainty intervals, and methodology limitations that an LLM needs. You can add an `/api/projects/{id}/synthesize` endpoint that feeds `dataset_profile` + `run-config` + `explain` (permutation importance & PDP summaries) + `drift_report` into an LLM to auto-write executive research notes or answer questions like *"Did our new feature engineering answer Research Question #2 in this project?"*
* **Sample-Level Outcome Explanations Can Piggyback on `/explain` and `/intervals`:** The new **Explain** page (Ch. 19.8, p. 91) already computes global permutation importance, partial dependence (PDP/ICE), and split-conformal prediction intervals (`Prediction ± 88`). To complete your first LLM goal (*"explain outcomes on samples"*), add a local row-explanation endpoint (`POST /api/inference/{task}/explain-row`) that computes local [SHAP](https://github.com/shap/shap) values (or 1D ICE slice deltas + conformal bounds for that specific row) and uses the LLM to translate them into a grounded sentence (e.g., *"Predicted progression is 213.8 ± 88.2 (90% conformal interval), driven primarily by BMI (+0.06) and S5 (+0.02) sitting in the top decile of the training distribution"*).
* **Unstructured-to-Tabular Extraction Can Plug Directly into `FoldLocalDataPreprocessor`:** Now that `FoldLocalDataPreprocessor` (Ch. 11, p. 33) cleanly routes `pd.StringDtype()` columns to per-column TF-IDF inside each CV fold, you have the exact hook needed for your third LLM goal. Adding an optional `text_extractor="llm_schema"` mode (using [Instructor](https://github.com/567-labs/instructor) or local structured generation) inside `FoldLocalDataPreprocessor` or before `mltrack.register_data` would let users turn raw text columns into typed numeric/categorical features while keeping the extraction schema locked inside the `ModelBundle` contract.

# ChatGPT


# SKSurrogate 0.2.0 — review of the updated version

This is a meaningful update. My assessment is considerably more positive than it would have been from the previous documentation.

The most important development is that you have started addressing the gap we discussed: moving SKSurrogate beyond an AutoML and model-operations library toward a research-oriented scientific ML environment.

Three additions stand out:

1. Research projects: persistent research questions, notes, and references to datasets, jobs, and model bundles.
2. Scientific experiment design: Latin-hypercube initialization and sequential surrogate-based recommendations.
3. A more complete scientific evaluation workflow: repeated nested cross-validation, permutation importance, partial dependence, and split-conformal prediction intervals.

The notebook workflows and the expanded end-to-end UI walkthrough also make the product more accessible to scientists.

However, I see an important distinction: the update adds several components of a scientific workflow, but they are not yet integrated into a coherent research environment.

I will explain what has improved, what remains weak, and what I would prioritize next.

## 1. What has improved substantially

| Area                   | Previous assessment                       | Updated assessment                                          |
| ---------------------- | ----------------------------------------- | ----------------------------------------------------------- |
| Research organization  | Missing                                   | Initial implementation                                      |
| Experiment design      | Basic optimization-oriented functionality | Sequential recommendation added                             |
| Scientific notebooks   | Not sufficiently exposed                  | Two executable workflows                                    |
| Model interpretation   | Limited                                   | Permutation importance and PDP                              |
| Predictive uncertainty | Insufficiently exposed                    | Split-conformal regression intervals                        |
| Model evaluation       | Basic evaluation workflow                 | Repeated nested CV and learning curves                      |
| Reproducibility        | Partial                                   | Stronger configuration, fingerprint, and dependency records |
| Web workflow           | Primarily operational                     | More complete scientific-to-operational walkthrough         |
| Execution              | Limited                                   | Durable jobs, Dask, checkpoints, CI/CD                      |

These are not merely documentation improvements. The new chapters describe concrete functionality and interfaces.

### 1.1 Research projects: an important conceptual addition

Chapter 8 introduces a project abstraction containing:

- A project name and description.
- Research questions.
- References to dataset partitions.
- References to model bundles.
- References to background jobs.
- Timestamped notes.

I particularly like that projects reference existing artifacts instead of copying them. This avoids unnecessary duplication and preserves the distinction between research organization, experiment tracking, and model storage.

However, the documentation explicitly identifies several limitations:

- No project UI.
- No project deletion.
- No notebook execution.
- No project-level report generation.
- No authorization model beyond the existing optional API-key middleware.
- References are pointers rather than immutable snapshots.

This is a sensible first implementation, but I would not yet describe it as a research management system.

My recommendation: develop this abstraction further rather than replacing it.

The next step should be to make a project the organizing context for scientific work, not merely a container for metadata.

For example, a project should eventually connect:

`Question → Hypothesis → Dataset → Analysis → Experiment → Result → Conclusion`

The most important missing element is the ability to record the relationship between an investigation and the evidence supporting or contradicting it.

### 1.2 Experiment design: potentially one of your strongest differentiators

Chapter 7 is an interesting addition.

The new functionality includes:

- Seeded Latin-hypercube sampling.
- Sequential candidate recommendation.
- Gaussian-process surrogate modeling.
- Lower confidence bound acquisition.
- Explicit bounds and resource limits.
- Returned surrogate mean, standard deviation, acquisition value, and metadata.

I like the explicit separation between proposing an experiment and actually evaluating it. SKSurrogate does not pretend to control instruments or execute experiments.

There is an opportunity here to develop a genuinely useful scientific experiment-design environment.

However, I would distinguish the current implementation from a general Bayesian optimization framework.

The documented implementation uses a fixed Matérn 5/2 kernel with unit length scales after normalization, and does not optimize kernel hyperparameters. The candidate is selected from a bounded Latin-hypercube pool.

That is a useful bounded sequential-design baseline, but it is not yet a comprehensive treatment of Bayesian experimental design.

I would eventually add:

- Expected improvement.
- Probability of improvement.
- Upper/lower confidence bounds.
- Noise-aware Gaussian processes.
- Kernel and hyperparameter selection.
- Replicate recommendations for estimating experimental noise.
- Multi-objective and constrained optimization.
- Batch acquisition.
- Multi-fidelity experimental design.

I would also expose the distinction between exploration and exploitation explicitly.

My assessment: this feature deserves investment. It connects your existing surrogate optimization work to a scientific use case that is more distinctive than conventional AutoML.

### 1.3 Notebook workflows: exactly the right direction

Chapter 14 introduces two executable notebooks:

- `scientific_workflow.ipynb`
- `python_api_workflow.ipynb`

The first demonstrates profiling, classification, repeated nested cross-validation, and permutation importance.

The second demonstrates an API-driven workflow, including dataset registration, experiment submission, evaluation, reports, and research project creation.

This is a good architectural decision.

You have avoided making Jupyter a runtime dependency and instead use it as an external scientific interface. That is preferable to building a proprietary notebook environment.

I would expand this into a collection of carefully maintained scientific templates, but the current approach is correct.

### 1.4 Explainability and uncertainty: a significant improvement

The new walkthrough, particularly Chapter 19.8, is one of the most important parts of the update.

It documents:

- Permutation importance.
- Partial dependence.
- Split-conformal regression prediction intervals.

This addresses a substantial part of my earlier criticism.

The conformal interval implementation is especially useful because it provides something beyond a conventional point prediction.

However, I would strengthen this functionality in three ways:

1. Coverage diagnostics: report empirical coverage alongside nominal coverage, ideally with confidence intervals.
2. Conditional coverage: investigate whether coverage deteriorates across important subgroups or regions of the feature space.
3. Interval evaluation: report interval width and appropriate interval scoring metrics, not just coverage.

Also, be careful with the interpretation of the documented 90% interval example. An observed coverage of 87.9% on 66 observations is not sufficient evidence that the procedure is well calibrated. The sample is small, and the uncertainty around the empirical coverage is substantial.

The interface should communicate that distinction.

## 2. What remains insufficient for a scientist-friendly product

This is where I would focus the next development cycle.

### 2.1 Exploratory data analysis is still too shallow

Highest priority

The updated walkthrough shows dataset preview, schema validation, target statistics, and some feature analysis.

That is a good start, but it is not yet a proper scientific EDA environment.

I would add a dedicated profiling and exploration subsystem.

| Missing or underdeveloped capability | Suggested functionality                                        |
| ------------------------------------ | -------------------------------------------------------------- |
| Distribution analysis                | Histograms, KDE, ECDF, quantiles                               |
| Descriptive statistics               | Skewness, kurtosis, robust statistics                          |
| Missingness                          | Missingness maps, patterns, associations                       |
| Outliers                             | Robust statistical and multivariate detection                  |
| Feature relationships                | Pearson, Spearman, Kendall, mutual information                 |
| Group comparisons                    | Distribution comparisons across experimental groups            |
| Data leakage                         | Suspicious target relationships and temporal leakage checks    |
| Multicollinearity                    | VIF, condition numbers, correlation clustering                 |
| Data quality                         | Duplicate observations, constant features, inconsistent values |

I would not simply add more plots. Each analysis should have a computational result, an explanation of its assumptions, and a reproducible record.

For example, a correlation analysis should distinguish linear association from general dependence. A distribution comparison should state which test was used and what assumptions apply.

This is a foundational requirement for the product's scientific identity.

### 2.2 Statistical inference needs to become a first-class subsystem

Highest priority

Repeated nested cross-validation is valuable, but it does not by itself provide a complete statistical model-comparison framework.

A scientist should be able to compare two candidate models and investigate:

- The distribution of performance differences across repeated splits.
- The variability due to data partitioning.
- The stability of feature importance.
- Whether the observed performance difference is practically meaningful.
- Whether multiple comparisons have inflated the apparent winner.
- Whether the results generalize to an independent population.

I would introduce a dedicated statistical inference module, separate from model evaluation.

My suggested additions:

| Topic                  | Methods                                                            |
| ---------------------- | ------------------------------------------------------------------ |
| Resampling inference   | Bootstrap, BCa bootstrap                                           |
| Model comparison       | Paired permutation tests, appropriate paired resampling procedures |
| Effect sizes           | Standardized effects, Cliff's delta                                |
| Multiple comparisons   | Holm, Benjamini–Hochberg                                          |
| Distribution testing   | KS, Anderson–Darling, permutation tests                           |
| Calibration            | Reliability diagrams, Brier score, calibration error               |
| Stability              | Feature-selection and explanation stability                        |
| Prediction uncertainty | Conformal methods and interval scoring                             |

One technical caution: avoid treating cross-validation folds as independent observations when applying ordinary statistical tests. The dependence between training sets matters. The system should make its comparison procedure explicit and use methods appropriate to the evaluation design.

This would be a major improvement over reporting a leaderboard of scores.

### 2.3 Interpretability needs more than permutation importance and PDP

High priority

The current explainability implementation is a good initial layer, but it is not yet sufficiently comprehensive.

I would add:

- SHAP for local and global attribution.
- ALE plots.
- ICE curves.
- Interaction strength and interaction plots.
- Error analysis and misclassified-observation exploration.
- Explanation stability under resampling.
- Subgroup-specific explanations.

There is also a technical point worth emphasizing.

Permutation importance can be misleading when features are strongly correlated. PDP can produce misleading results when it evaluates combinations of features that are rare or impossible in the observed data.

For scientific use, I would make ALE and conditional analysis particularly important additions.

More importantly, I would introduce a distinction between:

- Predictive association.
- Model attribution.
- Sensitivity to input perturbations.
- Causal effects.

These are different mathematical objects. The interface should not conflate them.

### 2.4 The research project abstraction needs an actual research interface

High priority

I would prioritize this over adding more deployment features.

The current project abstraction is a metadata store. It does not yet provide a scientist with a place to conduct and organize an investigation.

I would introduce four additional concepts:

Hypotheses

A hypothesis should have a statement, status, and supporting or contradicting evidence.

Analyses

An analysis should record its inputs, methodology, parameters, outputs, and provenance.

Findings

A finding should summarize an observation and link to the analysis that supports it.

Research artifacts

These could include notebooks, figures, tables, datasets, reports, and external references.

A project could then look like:

Project: Material degradation

Research workspace

Hypothesis H1

Temperature has a nonlinear relationship with degradation rate.

Under investigation

Evidence

- EDA: observed nonlinear association.
- Model analysis: nonlinear model improves validation performance.
- ALE: response increases sharply above a temperature threshold.
- Sensitivity: temperature has a high total-effect index.

Next investigation

Determine whether the apparent threshold persists under resampling and across pressure conditions.

This is the kind of structure that makes the product useful across a research project rather than only during model training.

### 2.5 The unstructured-data capability is still missing

This remains an important part of our earlier discussion.

I do not see a developed document-to-tabular workflow in this release.

I would retain it as a strategic feature, but I would not put it ahead of EDA, inference, and research organization.

When implemented, it should provide:

- PDF, DOCX, spreadsheet, and scanned-document ingestion.
- Schema-driven extraction.
- Unit and measurement normalization.
- Source-page and passage provenance.
- Confidence scores.
- Human validation.
- Duplicate and contradiction detection.
- Export to a typed dataset.

A particularly useful scientific workflow would be extracting measurements from experimental reports and combining them with structured laboratory or sensor data.

That could become a strong differentiator, especially in industrial science.

## 3. My assessment of the architecture and product direction

The update also makes some architectural strengths and weaknesses clearer.

### What I like

- The separation between the Python library and web control plane.
- The use of persistent background jobs.
- The Dask execution backend.
- Checkpointing and resumability.
- Dataset fingerprints and model-bundle metadata.
- MLflow-compatible export.
- Optional dependencies and external Jupyter integration.
- Explicit API contracts and bounded execution.

These are sensible decisions for an open-source project.

I also appreciate the documentation's explicit treatment of limitations. For example, the experiment-design module says it proposes points but does not evaluate them, and the research-project module describes its current storage and concurrency limitations.

That kind of technical honesty is valuable.

### What concerns me

A. Research projects are stored in JSON files.

The documentation recommends a single API process for writes because the store is not a distributed multi-worker database.

That is acceptable for a lightweight initial implementation, but I would eventually move project metadata to SQLite or another transactional store. It would make concurrent updates, relationships, and future querying much easier.

B. Artifact references are not immutable.

A project can point to a dataset partition, job, or model bundle that subsequently changes or disappears.

For reproducible research, I would prefer immutable artifact identities and versioned references, including the relevant dataset fingerprint and configuration snapshot.

C. The scientific workflow is fragmented across modules.

The functionality exists in several places:

- `mltrack`
- Sensitivity analysis
- Experiment design
- Evaluation
- Model bundles
- Research projects
- Notebook workflows

These are reasonable implementation boundaries, but the user experience should connect them.

A scientist should not have to understand the internal module architecture to move from a research question to a defensible conclusion.

D. The UI is still primarily an ML lifecycle interface.

The Chapter 19 walkthrough is a major improvement, but its overall sequence remains:

Dataset → Feature Analysis → Experiments → Model Bundles → Evaluation → Explain → Quality Gates → Deployment → Monitoring.

That is a sensible engineering workflow.

It is not quite a scientific workflow.

I would keep this operational workflow, but introduce a separate Research workspace rather than trying to force all scientific activities into the existing navigation.

## 4. A few technical observations worth addressing

There are also some details in the documentation that I would refine before positioning SKSurrogate as a serious scientific tool.

### 4.1 Sensitivity analysis and feature importance are not interchangeable

The walkthrough makes a useful observation: Sobol ranks BMI relatively low, while permutation importance identifies it as the strongest predictor.

I actually like that this discrepancy is documented rather than hidden.

However, I would explain the distinction more prominently.

- Sobol indices measure contributions to output variance under a specified input distribution.
- Permutation importance measures the effect of disrupting a feature on a particular model's predictive performance.
- Correlation measures pairwise linear association.
- SHAP measures feature attribution relative to a particular attribution framework and background distribution.

These methods answer different questions. Their rankings should not be expected to agree.

I would turn this into a deliberate feature: compare analytical perspectives on the same variables.

This could be a genuinely useful scientific investigation tool.

### 4.2 Model comparison needs stronger experimental discipline

The documentation correctly warns that the experiment bundle's training score and the baseline's validation score are not directly comparable.

This is good, but I would go further.

Every model comparison should record and expose:

- Identical evaluation partitions.
- Identical scoring definitions.
- Identical preprocessing boundaries.
- Whether hyperparameter selection occurred inside or outside the evaluation loop.
- Whether the test set influenced model selection.
- Whether the compared estimates are paired.

I would make invalid comparisons difficult to perform accidentally.

### 4.3 Synthetic data generation deserves more scientific attention

The new documentation includes a `synthdata` module, but I would consider this more than a convenience feature.

Synthetic data can be useful for:

- Testing algorithms under controlled conditions.
- Evaluating optimization strategies.
- Studying sensitivity to noise.
- Investigating feature interactions.
- Testing model stability.
- Reproducing benchmark experiments.

I would eventually introduce a collection of controlled data-generating processes with known ground-truth properties.

For example, a researcher could generate datasets with a known nonlinear interaction, heteroscedastic noise, correlated predictors, or distribution shift, then test whether SKSurrogate correctly detects these phenomena.

This would also be useful for testing SKSurrogate itself.

## 5. What I would prioritize next

I would revise my previous roadmap based on what is now implemented.

Recommended next development cycle

| Priority | Work                                                         | Reason                                                   |
| -------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| 1        | Scientific EDA and profiling                                 | Establishes the foundation for all subsequent analysis   |
| 2        | Statistical inference and model comparison                   | Makes conclusions more scientifically defensible         |
| 3        | Research project UI and evidence tracking                    | Connects existing functionality into a research workflow |
| 4        | Explanation improvements: ALE, SHAP, interactions, stability | Strengthens scientific interpretation                    |
| 5        | Immutable experiment and dataset provenance                  | Makes investigations reproducible                        |
| 6        | Publication-quality reports and notebooks                    | Makes results usable outside SKSurrogate                 |
| 7        | LLM-assisted research interface                              | Makes the growing analytical ecosystem accessible        |
| 8        | Document-to-tabular extraction                               | Opens a new and commercially interesting use case        |

I would postpone symbolic regression, physics-informed modeling, and extensive domain-specific algorithms until the scientific foundation is stronger.

They are interesting directions, but I would not let them distract from the integration work.

## 6. My revised overall assessment

# SKSurrogate 0.2.0

Meaningful progress

My assessment by dimension

| Dimension                   | Assessment                                  |
| --------------------------- | ------------------------------------------- |
| AutoML and optimization     | Strong foundation                           |
| Experiment design           | Promising initial capability                |
| Model evaluation            | Good foundation                             |
| Interpretability            | Useful, but incomplete                      |
| Uncertainty quantification  | Promising initial capability                |
| Reproducibility             | Good foundation, needs immutable provenance |
| Research management         | Early implementation                        |
| Scientific data exploration | Still underdeveloped                        |
| Statistical inference       | Still underdeveloped                        |
| Research-oriented UX        | Improving, but fragmented                   |
| Production lifecycle        | Relatively mature for the project's scope   |

My main conclusion is that the project has reached a point where integration is more valuable than adding another large collection of algorithms.

You now have enough building blocks to create a meaningful research-oriented workflow. The next challenge is to make them work together.

I would also make one strategic adjustment to the product's identity.

Rather than positioning SKSurrogate as an AutoML platform with scientific features, I would position it as:

An open-source scientific machine learning environment for data exploration, surrogate-based optimization, experimental design, and model discovery.

That identity accommodates the existing AutoML functionality without making it the entire product.

And it gives the project a more defensible direction: not simply finding a model that predicts well, but helping researchers understand the systems represented by their data.

Documentation reviewed: SKSurrogate Documentation 0.2.0, dated November 5, 2026. In particular, Chapters 7–8, 14, 18–19, and 20; the screenshots on pages 80–91 illustrate the expanded dataset, experiment, evaluation, and explanation workflows.&#x20;

SKSurrogate.pdf

My strongest recommendation: make the next major release revolve around a unified Research Workspace, backed by better EDA and statistical inference. That would make the additions in 0.2.0 feel like parts of one product rather than a collection of increasingly capable modules.
