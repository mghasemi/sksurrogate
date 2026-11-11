# Gemini:

A close reading of the **SKSurrogate v0.2.0** documentation—especially the code reference (Chapter 23), the UI manual (Chapter 15), and the benchmark post-mortems (Chapters 24–26)—shows a library with unusually strong MLOps bones and admirable scientific honesty, but several friction points that hold back day-to-day data scientists.

To make SKSurrogate much more appealing and scientist-friendly, the biggest gaps to close fall into five areas:

### 1. Developer Ergonomics & API Consistency

Data scientists judge a Python library within the first five minutes of writing code. Currently, SKSurrogate has high boilerplate and inconsistent API conventions:

* **No "Batteries-Included" Default Search Spaces:** To run `AML`, a user must manually construct a 30-to-50-line nested `config` dictionary with fully-qualified string import paths (e.g., `"sklearn.linear_model.LogisticRegression": {"C": Real(1e-6, 10.0), ...}`). If a scientist just wants to run `AML(preset="medium_quality", time_limit=300).fit(df, target="class")`, they cannot—they have to define every estimator, preprocessor, and hyperparameter bound by hand.
* **Inconsistent Naming & Typos in Keyword Arguments:** Across the public API, naming conventions mix `CamelCase`, `snake_case`, and capitalized kwargs. For example, `mltrack` mixes `RegisterData()`, `LogModel()`, `FeatureWeights()`, and `TopFeatures()` with `nested_cv_evaluation()`, `validate_data()`, and `get_dataframe()`, plus mixed-case methods like `allModels()` and `getBest()`. Meanwhile, `EOA` uses misspelled parameter names (`parents_porp` and `elits_porp` instead of `prop`, and singular `max_generation` instead of `max_generations`), and `SurrogateRandomCV` uses a capitalized `Continue=False` parameter.
* **Loss of DataFrame Column Names in `AML`:** While `mltrack.RegisterData` accepts a pandas DataFrame, `AML.fit(X, y)` and `AML.eoa_fit(X, y)` take raw NumPy `.values` arrays (as shown in Chapter 8), and categorical columns must be passed by integer index (`cat_cols=[5]`). Stripping DataFrame column names before pipeline search makes downstream feature selection (`SensAprx`, `ReliefF`) and error debugging much harder to read.

### 2. Preprocessing Leakage & Modern Tabular Wrangling

Chapter 9 (`DataProcess`) aims to automate data preparation, but its current design introduces pitfalls that experienced ML practitioners will immediately flag:

* **Full-Dataset Imputation and Encoding Outside CV:** In the Chapter 9 workflow (`A = DataPreprocess(df); A.deduce_types(); A.encode()`), `OneHotEncoder`, `OrdinalEncoder`, `DateTime2Num`, and `IterativeImputer()` are fitted on the entire DataFrame before cross-validation splits occur. Fitting a multivariate `IterativeImputer` on the full dataset leaks validation and test distribution signal into the training folds.
* **Lack of Column-Specific Pipeline Routing:** Instead of transforming the DataFrame upfront or applying `StandardScaler` / `Normalizer` across every column indiscriminately in `AML`, `DataPreprocess` should output a native [scikit-learn `ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)` or integrate with [skrub](https://skrub-data.org/stable/documentation.html) (`TableVectorizer`) so numeric, categorical, datetime, and text transformations happen strictly *inside* each cross-validation fold.
* **Incomplete Unstructured & High-Cardinality Support:** In `DataPreprocess`, `text_processor=None` is a bare placeholder, image data is listed as a `"To Do"` (Section 23.14.1), and high-cardinality strings are ordinal-encoded alphabetically (page 29), which Chapter 24 (Finding 3) proved degrades surrogate guidance and tree/linear model performance compared to out-of-fold target encoding or native LightGBM categorical handling.

### 3. Search Speed, Surrogate Math, and Ensembling

SKSurrogate's own benchmarks (Chapters 24–26) identify the exact algorithmic bottlenecks that frustrate scientists on tight iteration loops:

* **Slow Turnaround ("Hours if Not Days"):** Chapter 1 warns that `AML` *"is not designed to result in outstanding results in a few minutes, rather it needs hours if not days."* Chapter 24 explains why: fitting the Gaussian Process (GP) surrogate scales as $O(n^3)$, consuming **49% of a 120-second budget in search overhead alone**, which allows portfolio AutoML tools like [FLAML](https://github.com/microsoft/FLAML) to beat every search arm on 5 of 6 datasets at equal wall-clock time.
* **Known Surrogate Fragilities Not Yet Auto-Handled:** Chapter 26 proves three critical regime failures:
1. `HilbertRegressor` crashes with a singular Gram matrix whenever sample count $n < \binom{d+\text{deg}}{d}$ (failing 100% of runs at $d=3, \text{deg}=3, n=10$).
2. The stock RBF `GaussianProcessRegressor` has no noise term (`alpha=1e-10`), causing RMSE to explode from $\approx 0$ to $2.9$ on noisy CV evaluations.
3. `RandomForestRegressor` is the most robust generalist under noise and trigonometric landscapes.
Yet `SurrogateRandomCV` still defaults to an un-tuned, noise-intolerant RBF-GP rather than a noise-aware GP (`RBF() + WhiteKernel()`) or an automatic fallback to Random Forest when the Gram matrix is ill-conditioned.


* **No Post-Hoc Ensembling or Zero-Shot Foundation Models:** `AML` evaluates dozens of pipelines and stores the top $n$ via `A.get_top(n)`, but `A.predict(X)` only uses the single `best_estimator_`. Adding fast post-hoc greedy ensemble selection (Caruana ensembling) over the top-$k$ already-fitted pipelines—and including zero-shot tabular foundation models like [TabPFN](https://github.com/PriorLabs/tabpfn) or multi-fidelity pruning like [Optuna](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)'s Hyperband—would dramatically improve out-of-the-box accuracy in under 60 seconds.

### 4. Scientific Diagnostics & Explainability Gaps

While `sensapprx` (Sobol, Morris, $\delta$-index) is mathematically sophisticated, several everyday diagnostic tools that scientists rely on are missing:

* **Binary-Only Diagnostic Plots:** In `mltrack`, `plot_roc_curve`, `plot_cumulative_gain`, and `plot_lift_curve` only work for binary classification (raising a `ValueError` on multiclass tasks, p. 112), and there are **zero diagnostic plots for regression** (no predicted-vs-actual scatter, residual distribution, or Q-Q plots).
* **No Local Attribution or Fast Tree Explainability:** Variance-based Sobol and Morris sensitivity analyses require hundreds of model evaluations (`num_smpl=512`), making them slow on wide datasets (the API even rejects heatmaps with $>60$ features with HTTP 400 on page 15). Integrating [SHAP](https://github.com/shap/shap) (`TreeExplainer` and `LinearExplainer`) would provide instant global and sample-level explanations for wide datasets without the 60-feature ceiling.

### 5. Unfinished MLOps & Control-Plane Guardrails

The FastAPI + React UI is a major selling point, but the manual candidly notes several v1 shortcuts that need hardening before teams can trust it in production:

| Current Gap in Manual | Where Documented | Scientist / MLOps Fix Needed |
| --- | --- | --- |
| **Unenforced UI Quality Gates** | Chapter 15.2.7 (p. 59) | Wire the UI **Promote** button to require a passing `check_bundle_quality` report by default rather than leaving it optional. |
| **In-Memory-Only `InferenceMonitor**` | Chapter 13.2 (p. 49) | Persist latency, error rate, and row throughput into `api/state.db` (SQLite) so serving metrics survive process restarts. |
| **Passive Retraining Triggers** | Chapter 15.2.11 (p. 63) | Add a lightweight background scheduler (or webhook listener) so `scheduled` and `on_data` triggers fire automatically when drift crosses a PSI threshold. |
| **Name-Only PII Detection** | Chapter 15.2.10 (p. 62) | Upgrade `sensitive_feature_report` from a regex scan on column names to also inspect cell values (e.g., detecting regex patterns for emails, SSNs, or phone numbers inside unnamed columns). |
| **Notebook-Native Visualization** | Chapter 14 & 15 | Render interactive HTML widgets (like scikit-learn's `estimator_html_repr` and Plotly charts) directly inside Jupyter notebooks so scientists don't have to spin up `uvicorn` + `npm run dev` just to inspect runs. |

If you prioritize **(1) built-in default search presets (`config="auto"`), (2) a noise-aware GP/RF surrogate ensemble, and (3) `ColumnTransformer`-based preprocessing inside CV**, SKSurrogate will feel immediately intuitive to scikit-learn users while preserving all of its unique MLOps lineage and sensitivity analysis strengths.

# ChatGPT:

# What SKSurrogate is currently lacking to become a scientist-friendly product

My assessment is that SKSurrogate has developed considerably more infrastructure than a typical AutoML library, but it still needs a stronger scientific workflow layer.

There is an important distinction between building software that can train, evaluate, deploy, and monitor ML models and building software that helps scientists conduct investigations.

A scientist generally does not approach a dataset with the sole objective of maximizing predictive accuracy. They want to:

- Formulate and investigate hypotheses.
- Understand the structure of their data.
- Discover relationships and unexpected phenomena.
- Determine whether findings are statistically meaningful.
- Quantify uncertainty.
- Compare competing explanations.
- Reproduce and communicate their findings.

I would use these requirements as the benchmark for SKSurrogate's next stage.

I have separated the assessment into three categories:

1. Capabilities that appear to be missing or insufficiently exposed in the manual.
2. Scientific capabilities that would substantially improve the product.
3. Product and workflow improvements that would make it pleasant to use in actual research.

An important qualification: I am assessing the documented capabilities, not claiming that every feature below is absent from the implementation. Some may exist without being covered in the manual.

## 1. Scientific capabilities: the biggest gaps

### 1.1 Exploratory data analysis (EDA)

High priority

This is probably the most important missing layer before model training.

Currently, SKSurrogate has dataset registration, schema validation, data type inference, preprocessing, and feature analysis. Its documented `DataProcess` module supports numerical, categorical, ordinal, Boolean, datetime, and label types, along with encoding and imputation.

But a scientist needs much more than preprocessing.

I would introduce a dedicated Scientific Data Exploration module.

| Capability              | What it should provide                                           |
| ----------------------- | ---------------------------------------------------------------- |
| Descriptive statistics  | Mean, median, variance, quantiles, skewness, kurtosis            |
| Distribution analysis   | Histograms, density estimates, empirical distributions           |
| Dependence analysis     | Pearson, Spearman, Kendall, mutual information                   |
| Missingness analysis    | Missingness patterns and associations                            |
| Outlier detection       | Statistical and model-based anomaly detection                    |
| Multicollinearity       | VIF, correlation structures, condition numbers                   |
| Distribution comparison | Compare populations, groups, and experimental conditions         |
| Target analysis         | Class balance, target distribution, target-feature relationships |

I would also include an automatic EDA report.

For example, after uploading a dataset, the scientist receives:

- Dataset characteristics.
- Potential data-quality problems.
- Important distributional properties.
- Potentially redundant features.
- Unexpected correlations.
- Potential target leakage.
- Recommendations for subsequent analysis.

The LLM can then help investigate the findings.

Why this matters: Scientists should understand their data before an optimization algorithm starts searching for a predictive model.

### 1.2 Statistical inference and uncertainty quantification

Very high priority

This is where I see a particularly important distinction between a conventional AutoML system and a scientific ML system.

SKSurrogate already documents nested cross-validation, fold-level predictions, repeated-CV confidence intervals, learning curves, and calibration plots. These are valuable foundations.

However, I would expand this into a comprehensive statistical inference layer.

I would prioritize:

| Capability             | Suggested methods                                   |
| ---------------------- | --------------------------------------------------- |
| Confidence intervals   | Bootstrap, BCa bootstrap                            |
| Model comparison       | Paired tests, permutation tests                     |
| Effect sizes           | Cohen's d, Cliff's delta, standardized effects      |
| Distribution testing   | KS, Anderson-Darling, permutation-based tests       |
| Association testing    | Pearson, Spearman, Kendall, mutual information      |
| Multiple testing       | Holm, Benjamini-Hochberg                            |
| Prediction uncertainty | Prediction intervals, conformal prediction          |
| Model stability        | Bootstrap stability and feature-selection frequency |

For example, suppose SKSurrogate identifies a model that achieves 94% accuracy and another that achieves 93%.

A conventional AutoML interface reports the winner.

A scientist-friendly interface should answer:

- Is the difference statistically meaningful?
- How variable are the estimates?
- Does the conclusion hold across different random seeds?
- Is one model more stable?
- Is the performance difference practically meaningful?

This is a much more useful way to approach model selection.

I would especially investigate conformal prediction. It provides a principled framework for uncertainty-aware predictions under appropriate exchangeability assumptions, and it is highly compatible with an existing scikit-learn-oriented architecture.

### 1.3 Experiment design

High priority

I think this is an underrated opportunity.

Scientists frequently have limited experimental budgets. They want to know which experiments, measurements, or observations would be most informative.

SKSurrogate already has surrogate optimization and sensitivity analysis. These could be extended toward:

- Design of experiments (DoE).
- Latin hypercube sampling.
- Factorial designs.
- Response surface methodology.
- Bayesian optimization for expensive experiments.
- Active learning.
- Sequential experimental design.
- Optimal experimental design using Fisher information.

Imagine asking:

> I can perform only 30 additional experiments. Which parameter combinations should I investigate to maximize the information gained about the system?

This is a natural extension of surrogate-based optimization.

It also creates a bridge between conventional ML and scientific computing.

### 1.4 Scientific interpretability

Very high priority

Your existing Morris, Sobol, and moment-independent sensitivity analysis is an excellent starting point.

But I would make the interpretability ecosystem more comprehensive.

| Capability             | Why it matters                                           |
| ---------------------- | -------------------------------------------------------- |
| SHAP                   | Local and global feature attribution                     |
| Partial dependence     | Average marginal relationships                           |
| ICE                    | Individual conditional relationships                     |
| ALE                    | Feature effects with reduced extrapolation problems      |
| Interaction analysis   | Discover nonlinear interactions                          |
| Permutation importance | Model-agnostic feature importance                        |
| Surrogate explanations | Approximate complex models with interpretable models     |
| Stability analysis     | Determine whether explanations persist across resampling |

One thing I would emphasize: distinguish between model interpretation and scientific discovery.

For instance, Sobol indices describe how output variance is apportioned under a specified input distribution. They do not automatically establish causal relationships or physical mechanisms.

A scientist-friendly system should make these assumptions explicit.

### 1.5 More scientific problem classes

Medium-high priority

The documented workflow is predominantly organized around conventional tabular classification and regression.

I would consider expanding support for:

- Time-series forecasting.
- Survival analysis and time-to-event modeling.
- Count regression.
- Multi-output regression.
- Ordinal classification.
- Clustering.
- Anomaly detection.
- Quantile regression.
- Functional data analysis.

I would not implement all of these immediately.

Time series, survival analysis, and quantile regression would be particularly useful additions.

They would also require appropriate evaluation protocols rather than simply plugging new estimators into the existing search machinery.

## 2. Scientific data engineering

This is the second major area I would invest in.

### 2.1 A more sophisticated data understanding layer

The existing `DataProcess` module is useful, but its documented functionality is relatively conventional.

For example, it recognizes common tabular types, supports encoding and imputation, and allows a text processor to be supplied. However, text columns must be explicitly declared, and image processing is listed as future work in the manual.

I would develop a separate data intelligence subsystem.

Its responsibilities would include:

1. Semantic column identification.
2. Automatic identification of units and measurement scales.
3. Detection of identifiers versus meaningful numerical variables.
4. Detection of categorical and ordinal variables.
5. Recognition of dates, temporal sequences, and repeated measurements.
6. Document extraction and entity resolution.
7. Detection of duplicate and contradictory observations.
8. Data provenance and extraction confidence.

For scientific applications, units deserve particular attention.

For example:

| Variable    | Value | Unit |
| ----------- | ----- | ---- |
| Temperature | 350   | K    |
| Pressure    | 1.5   | MPa  |
| Duration    | 120   | s    |

The system should understand that measurements with different units cannot be compared naively.

I would also investigate integration with domain-specific formats such as:

- HDF5.
- NetCDF.
- FITS.
- Parquet.
- Scientific CSV and Excel datasets.

The goal should be to make heterogeneous scientific data accessible without requiring every user to write custom preprocessing code.

### 2.2 Data provenance and lineage

Very high priority

You already have dataset fingerprints, experiment tracking, model versions, and lineage concepts.

I would extend these into scientific data provenance.

Every result should be traceable to:

- Original dataset.
- Dataset version.
- Data transformations.
- Feature definitions.
- Excluded observations.
- Target construction.
- Training and validation partitions.
- Model configuration.
- Random seeds.
- Software and dependency versions.
- Evaluation protocol.

I would also record the provenance of extracted values from unstructured documents.

This is essential if SKSurrogate is intended for serious research.

A scientist should be able to return to a result six months later and reconstruct exactly how it was obtained.

## 3. Research workflow and experiment management

This is perhaps the largest product-level gap.

Your existing MLTrace is useful for recording model training, metrics, plots, and experiment history. The newer architecture also provides jobs, checkpoints, model lineage, and registry management.

But a scientific investigation is not necessarily a collection of model-training runs.

It is a sequence of questions, hypotheses, analyses, and conclusions.

I would introduce a first-class concept: Research Projects.

### 3.1 Research projects and hypotheses

A project might look like this:

Research project

In progress

Relationship between operating conditions and material degradation

Research questions

1. Does temperature influence degradation rate?
2. Is the relationship nonlinear?
3. Does pressure interact with temperature?
4. Can degradation be predicted reliably?

Investigations

1. Exploratory analysis

   Distribution analysis, correlation, missingness.

   Completed
2. Sensitivity analysis

   Morris and Sobol analysis.

   Completed
3. Predictive modeling

   Compare regression pipelines.

   In progress
4. Interaction investigation

   Temperature-pressure interactions.

   Pending

Research notes

- Temperature appears to have a nonlinear association with degradation.
- The pressure interaction requires further investigation.

This is not simply an experiment tracker.

It gives scientists a persistent workspace for organizing their research.

### 3.2 Notebooks and interactive analysis

High priority

I would make Jupyter integration a first-class feature.

Scientists already work with:

- JupyterLab.
- VS Code.
- Python scripts.
- Interactive plotting.
- LaTeX and Markdown.
- Git repositories.

SKSurrogate should integrate with these workflows rather than try to replace them.

I would provide:

- Reusable notebook templates.
- Interactive visualization of analysis results.
- Exportable experiment configurations.
- A Python API for every major UI operation.
- Automatic generation of reproducible notebooks from UI workflows.
- Notebook execution and result tracking.
- Markdown and LaTeX report generation.

A particularly useful capability would be:

Generate a complete, executable research notebook from an experiment.

The notebook should reconstruct the analysis, not merely display screenshots or copied results.

### 3.3 Publication-quality reporting

Scientists need to communicate their findings.

I would provide automatic report generation in:

- Jupyter notebooks.
- Markdown.
- HTML.
- PDF.
- LaTeX.

The reports should include:

- Methodology.
- Dataset description.
- Experimental configuration.
- Statistical results.
- Figures and tables.
- Uncertainty estimates.
- Limitations.
- Reproducibility information.

I would also make plots customizable, including mathematical notation, units, confidence bands, and publication-quality vector exports.

This is a relatively practical way to deliver value without requiring the user to adopt an entirely new research environment.

## 4. What the user interface is missing

I would be careful here: more dashboards do not necessarily make a product more scientist-friendly.

The interface should reflect the scientific process.

I would reorganize the product around the following workflow:

## Proposed SKSurrogate research workspace

1. Project and research questions

   Define the investigation, objectives, hypotheses, and datasets.
2. Data exploration

   Inspect distributions, relationships, data quality, and provenance.
3. Data preparation

   Extract, transform, validate, and document data.
4. Scientific analysis

   Statistical tests, sensitivity analysis, and experiment design.
5. Predictive modeling

   Pipeline search, hyperparameter optimization, and model comparison.
6. Model investigation

   Explain predictions, interactions, errors, and global behavior.
7. Results and publication

   Reproducible notebooks, reports, and research artifacts.

I would retain the existing deployment, registry, monitoring, and retraining functionality as a separate operational workflow.

That functionality is valuable, but it should not dominate the experience of someone using SKSurrogate for research rather than production deployment.

### A conversational interface should be embedded throughout

I would not make the LLM a separate chatbot page.

Instead, every research object should have an analytical conversational interface.

For example:

- Dataset: "Why are these variables highly correlated?"
- Feature: "Does this variable exhibit a nonlinear relationship with the target?"
- Experiment: "Why did the surrogate search outperform random search?"
- Model: "Where does this model fail?"
- Prediction: "Why was this observation classified differently?"
- Research project: "What evidence do we have for our second hypothesis?"

The LLM should have access to the project's actual computational artifacts, not just a textual summary.

## 5. A particularly interesting opportunity: scientific ML rather than generic AutoML

There is a direction I would seriously consider for SKSurrogate.

Your mathematical background and the library's existing surrogate optimization and sensitivity analysis capabilities suggest that you could develop a more distinctive identity around scientific machine learning.

This would mean supporting not only predictive modeling, but also the investigation of mathematical relationships between inputs and outputs.

Some interesting capabilities:

| Scientific capability          | Potential functionality                                      |
| ------------------------------ | ------------------------------------------------------------ |
| Symbolic regression            | Discover approximate analytical expressions                  |
| Equation discovery             | Identify candidate governing relationships                   |
| Dimensional analysis           | Detect dimensionally inconsistent models                     |
| Physical constraints           | Incorporate known physical laws into modeling                |
| Monotonicity constraints       | Enforce known directional relationships                      |
| Shape-constrained regression   | Enforce convexity, concavity, or other structural properties |
| Multi-fidelity optimization    | Combine cheap approximations with expensive evaluations      |
| Uncertainty-aware optimization | Optimize while accounting for predictive uncertainty         |
| Sensitivity-guided discovery   | Identify important variables and interactions                |
| Active learning                | Select the next most informative observations                |

I would not rush into implementing symbolic regression or physics-informed neural networks simply because they are fashionable.

However, the combination of surrogate optimization, Hilbert-space approximation, sensitivity analysis, and scientific constraints could become a distinctive research-oriented offering.

In particular, your Hilbert-space work could provide an interesting foundation for structured function approximation and surrogate modeling.

## 6. Prioritization: what I would actually implement

Here is my suggested roadmap, considering development cost, scientific value, and differentiation.

My recommended priorities

| Priority | Capability                                 | Value     | Effort      |
| -------- | ------------------------------------------ | --------- | ----------- |
| P0       | Scientific EDA                             | Very high | Medium      |
| P0       | Statistical inference and uncertainty      | Very high | Medium-high |
| P0       | Reproducibility and provenance             | Very high | Medium      |
| P0       | SHAP, ALE, PDP, interaction analysis       | Very high | Medium      |
| P1       | Jupyter integration                        | High      | Medium      |
| P1       | Research projects and hypotheses           | High      | Medium      |
| P1       | Publication-quality reports                | High      | Low-medium  |
| P1       | LLM research assistant                     | Very high | Medium-high |
| P1       | Document-to-tabular extraction             | Very high | High        |
| P2       | Experiment design and active learning      | Very high | High        |
| P2       | Time-series and survival analysis          | High      | High        |
| P2       | Symbolic regression and equation discovery | High      | High        |
| P2       | Physics-informed and constrained modeling  | High      | Very high   |

I would implement P0 before investing heavily in the LLM layer.

There is little value in having an LLM produce sophisticated scientific explanations if the underlying analytical capabilities are incomplete.

## 7. What I would deliberately avoid

There are some tempting directions that I would postpone.

1. Building another general-purpose AutoML interface. The market is already crowded.
2. Implementing dozens of algorithms. Breadth of algorithms is less valuable than depth of analysis.
3. Making the LLM responsible for mathematical calculations. Keep numerical and statistical computations deterministic and independently verifiable.
4. Building a proprietary notebook environment. Integrate with Jupyter instead.
5. Overemphasizing MLOps. Keep the existing deployment capabilities, but don't let them define the scientific product.
6. Adding every scientific domain at once. Build a general foundation and then add domain-specific extensions.

## 8. My overall recommendation

I see three possible identities for SKSurrogate:

1\. AutoML platform

Optimize predictive models and automate ML pipelines.

Your existing foundation.

2\. AI-assisted data science platform

Explore data, prepare datasets, build models, interpret results, and generate reports.

The most practical next evolution.

3\. Scientific discovery platform

Investigate mathematical relationships, discover patterns, quantify uncertainty, design experiments, and develop predictive or explanatory models.

The most differentiated long-term direction.

I would aim for the second in the short term and the third in the long term.

My suggested development sequence would be:

1. Make SKSurrogate scientifically rigorous: EDA, statistical inference, uncertainty quantification, explainability, and reproducibility.
2. Make it research-friendly: Jupyter integration, research projects, hypotheses, and publication-quality reports.
3. Add the LLM layer: Conversational access to the analytical capabilities, not merely a chatbot.
4. Expand data intelligence: Unstructured document extraction, semantic schema inference, and dataset construction.
5. Develop scientific discovery capabilities: Experiment design, active learning, symbolic regression, and physically constrained modeling.

### One final strategic observation

I think the most important conceptual change is this:

Stop thinking of SKSurrogate as a system that finds the best model for a dataset. Think of it as a system that helps scientists extract, test, and communicate knowledge from data.

That shift changes what you prioritize, how you design the interface, which algorithms you implement, and ultimately how you position the product commercially.

And importantly, it gives SKSurrogate a more distinctive identity than simply being another AutoML framework.