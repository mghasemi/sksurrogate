Changelog
=========

Unreleased
----------

Web UI log browser and severity-aware logging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* New **Logs** page in the control plane: captured log entries per background
  job, newest first, with time, severity badge, job, logger and message.
  Severity chips toggle each level (showing live counts, with **Warnings only**
  and **All levels** shortcuts); job, kind, free-text and row-count filters
  narrow the list, and selecting a job follows it live. Clicking an entry shows
  its trial/phase metadata or the full failure traceback. A **Logs** action on
  each job row opens the page scoped to that job.
* New ``api/logs.py`` capture layer: a ``logging.Handler`` bound to the
  ``SKSurrogate`` and ``sksurrogate.api`` loggers through a ``ContextVar``, so
  concurrent jobs never mix lines and API request handling is not captured.
  Entries are kept in a 2,000-entry ring buffer per job and appended to
  ``<SKSURROGATE_API_HOME>/logs/<job_id>.jsonl``; history survives a restart.
  Job records gain a ``log_counts`` summary.
* New endpoints: ``GET /api/logs`` (cross-job browse with ``level`` minimum
  severity or ``levels`` subset, ``kind``, ``q``, ``limit``), ``GET
  /api/logs/{job_id}`` (tail with ``since_seq``) and
  ``WS /api/logs/{job_id}/stream`` (incremental frames). Unknown levels return
  HTTP 422. The existing job status stream is unchanged.
* Severity is now used by the core, not just INFO/DEBUG: ``WARNING`` for a
  failed surrogate fit (logged before the exception propagates), for termination
  after too many non-improving iterations, and for one aggregated
  surrogate-failure line per search; ``ERROR`` for a non-finite surrogate
  prediction, an ``AML`` search that cannot fit any candidate, a failed user
  callback, and a failed job with its traceback. Repeated per-trial failures are
  logged at WARNING for the first five occurrences per job and status, then at
  DEBUG, so a systemic failure cannot flood the log.
* ``_logutil.emit`` no longer duplicates WARNING-or-above records: with logging
  unconfigured, ``logging.lastResort`` writes them to stderr and the stdout
  fallback is skipped. INFO/DEBUG console parity is unchanged.
* ``AML`` accepts ``callbacks=None`` and forwards it to the underlying
  ``SurrogateRandomCV``, so API-launched experiments emit per-trial telemetry
  (``trial``, ``phase``, ``status``) into the job log.
* Tests: ``tests/test_api_logs.py`` (lifecycle capture, severity and free-text
  filters, invalid-level rejection, failure traceback detail, concurrency
  isolation, JSONL persistence, warning cap, experiment trial events) and three
  new core tests for WARNING routing, fit-failure logging and no-progress
  termination.
* Docs: new log-browsing section in ``controlplane.rst`` (endpoint and entry
  contract, persistence, retention), the Logs page in ``ui-manual.rst`` with a
  figure, and the severity table plus the stderr rule in ``telemetry.rst``.

[0.3.0] - 2026-11-08
--------------------

Explanations: accumulated local effects and reference-subset stability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* New control-plane endpoint ``POST /api/evaluation/{task}/{version}/ale``
  computes accumulated local effects (ALE) for up to six numeric features —
  a supplement to partial dependence when features are dependent. The
  contract is documented in ``docs/mltrace.rst``: equal-frequency quantile bins
  with duplicate-edge collapse, deterministic left-merging of sparse bins
  (``min_bin_fraction`` default 0.01, reported in the response), non-finite
  rows excluded per curve, nominal categorical and (near-)constant features
  reported under ``skipped`` with a reason, one scalar response per curve
  (``predict`` for regression, labelled positive-class probability or decision
  value for binary classifiers, multiclass rejected), and non-causal
  interpretation labels. Feature selection defaults to the top-k permutation
  importance on the same reference sample and can be overridden explicitly.
* The Explain UI gained an **Accumulated local effects (ALE)** card with
  10/20/30-bin controls, per-bin row counts, skipped-feature reasons, and the
  non-causal limitation list. ALE is implemented as pure NumPy in
  ``api/explanations.py`` — no new dependency and no change to the
  ``SKSurrogate`` core.
* Explain responses now carry an optional ``partial_dependence_stability``
  block when 3–5 reference samples are requested: each primary PDP grid is
  re-evaluated on every distinct subset with direct model calls, and the block
  reports per-grid mean, sample SD, min, and max plus per-subset means. ALE
  curves report the same descriptive per-bin summary. These are
  sample-sensitivity diagnostics, not confidence bands; ICE stability is
  explicitly out of the API contract (individual ICE curves remain a plotting
  feature). PDP charts in the UI render the spread as dashed mean ± SD lines.
* Documentation: the ALE contract replaces the earlier "not exposed" note in
  ``docs/mltrace.rst``; the Explain section of ``docs/ui-manual.rst`` and the
  walkthrough describe the new card and the stability bands; the walkthrough's
  interval example payload now matches the endpoint (``alpha``, not
  ``coverage``).
* Controlled-DGP validation (Stage D1) gained an ALE ground-truth probe:
  ``benchmarks/dgps.py`` compares the accumulated curve against the exact
  noise-free response of the ``correlated`` and ``shift`` processes
  (including the known quadratic term), matching to floating-point precision.
* New row-level explanation endpoint
  ``POST /api/evaluation/{task}/{version}/explain/row`` (Stage D3, R9a):
  ordered-occlusion chain decomposition for one stored row against a reference
  mean or an explicit baseline row. Contributions telescope exactly to the
  model-response change (residual and tolerance reported), ``orders`` in 1–10
  draws distinct seeded feature orders and reports per-feature mean, sample SD,
  min, and max as disclosed order-dependence (not a confidence interval).
  Binary classifiers use the labelled scalar response; multiclass is rejected.
  A subset of features explains the hybrid row and is marked
  ``subset_hybrid``. Pure NumPy in ``api/explanations.py``; contract in
  ``AI_LAYER02_D3_ROW_LEVEL_CONTRACT.md`` and ``docs/mltrace.rst``.
* Explain UI: new **Row-level explanation** card with row index, 1/5/10 feature
  orders, a contribution bar chart, a mean/SD/min/max table, baseline and row
  predictions, and the reconstruction residual. Docs cover the card in the UI
  manual and walkthrough, including a new Explain-page figure
  (``docs/images/ui-explain-explanations.png``). Reconstruction residuals and
  tolerances are rendered in scientific notation instead of rounding to ``0``.
* D1 exit hook closed for the landed Stage C methods: ``benchmarks/dgps.py``
  now also checks average partial dependence and its reference-subset summary
  against the exact known response curves (``pdp_true_response_max_abs_error``
  and ``pdp_subset_shape_max_deviation`` at floating-point precision on the
  ``correlated`` and ``shift`` processes); DESIGN.md v1.17.


External references for research projects (Stage B1 tier 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Projects can now link an external citation as a fourth artifact kind,
  ``reference``: ``{kind, url, title?, doi?, label?}``. The URL must be an
  absolute ``http(s)`` URL of at most 2,048 characters, the API never fetches
  it, and references dedupe by URL. Non-reference kinds reject ``url``/``doi``
  fields, and reference entries reject ``task_name``/``artifact_id``; bad shapes
  are a 422.
* Reference snapshots use ``sksurrogate.project-artifact-snapshot.v2`` and store
  the URL/title/DOI fields only (``content_embedded: false``). The project
  summary reports them as ``exists: true`` with ``snapshot_status:
  "unverified"`` because reachability is never checked. References do not
  advance the workflow checklist, but they can satisfy the evidence requirement
  of a finding or conclusion and cannot be removed via ``PUT`` while referenced.
* Projects UI: the **Linked artifacts** card gained an **Add external reference**
  form (URL, title, DOI, label); linked references render as external links with
  their DOI and ``snapshot unverified`` badge and appear in the evidence picker.
* Docs: ``research-projects.rst`` documents the kind, validation, snapshot, and
  evidence semantics; ``ui-manual.rst`` covers the form. Tests:
  ``tests/test_projects_endpoints.py`` adds reference roundtrip/validation,
  evidence/removal-protection, backward-compatible shapes, and an HTTP-surface
  test.

LLM-assisted project synthesis (opt-in, Stage D2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* New optional workflow that drafts a short, **cited** project note from
  already computed facts. A user-supplied, OpenAI-compatible chat-completions
  endpoint is configured through ``SKSURROGATE_LLM_BASE_URL`` /
  ``SKSURROGATE_LLM_MODEL`` (plus optional key, timeout, and token budget);
  without configuration the feature returns a ``503`` and sends nothing. It is
  implemented over the standard library — no new dependency — and the core
  package remains untouched.
* ``POST /api/synthesis/projects/{project_id}`` queues a ``project-synthesis``
  job whose result carries the draft, its citations, the exact aggregated
  ``computed_facts`` payload, provider metadata, and a
  ``generated_interpretation`` label with ``causal_effect: false``.
  ``GET /api/synthesis/status`` reports configuration without exposing
  credentials.
* Sources are aggregates only: dataset profile summaries (``S1``), an optional
  job's whitelisted configuration and scalar metrics (``S2``), bundle metadata
  (``S3``), explanation summaries (``S4``), and stored monitoring alerts
  (``S5``). Raw rows, file paths, model objects, and credentials never leave the
  process. Drafts must cite supplied ids; unknown citations, uncited drafts,
  empty drafts, and over-length drafts fail the job with a named reason.
* Project notes gained optional generated provenance: ``source``
  (``human``/``generated``), ``model``, ``citations``, ``sources``,
  ``computed_facts``, and ``synthesis_job_id``. Generated notes are rejected
  without a model id, without citations, with citations that do not match a
  supplied fact, or with facts above 30,000 characters. Drafts are never saved
  automatically.
* Projects UI: new **Draft synthesis** card (provider status, task/bundle/job
  selectors from linked artifacts, explanation toggle, optional instruction)
  that renders the draft as *Generated interpretation* with citation chips and a
  computed-facts table; generated notes in the Notes list carry a badge,
  citations, model id, and collapsible facts. The Jobs kind filter includes
  ``project-synthesis``. Contract and data-handling review in
  ``AI_LAYER02_D2_LLM_SYNTHESIS_SCOPE.md``; docs updated in
  ``research-projects.rst``, ``ui-manual.rst``, and ``controlplane.rst``.
* Tests (``tests/test_api_synthesis.py``) use a local stub provider: disabled
  configuration, cited happy path with aggregate-only prompt assertions,
  unknown/uncited citation rejection, provider HTTP error mapping, and generated
  note provenance.


Acquisition readiness assessment (Stage D4, analysis only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added ``benchmarks/design_readiness.py``, a reproducible equal-budget harness
  comparing the shipped GP-LCB design policy with expected improvement (EI),
  probability of improvement (PI), and a noise-aware (Matern + WhiteKernel)
  variant on three synthetic 2-D objectives, with paired seeds, identical
  candidate pools, optional Gaussian observation noise, per-evaluation records,
  and a versioned report (``sksurrogate.design-readiness.v1``). CLI:
  ``python -m benchmarks.design_readiness --out DIR``.
* Result recorded in ``AI_LAYER02_D4_READINESS_ASSESSMENT.md`` and
  ``benchmarks/DESIGN.md`` §15: at 20 equal evaluations the shipped LCB and the
  alternatives all beat random by roughly an order of magnitude; EI/PI were
  neutral without noise; EI showed a weak (6 wins / 4 losses / 5 ties), not
  significant edge under synthetic noise; PI was worse under noise; and the
  fixed-length-scale noise-aware model regressed on the oscillatory objective.
  No production acquisition code, endpoint, or UI change ships, and D4 remains
  gated on a real expensive-experiment user.
* Tests: ``tests/test_benchmark_design_readiness.py`` (9 tests) pin equal-budget
  record shape, strategy validation, seeded determinism, observation-noise
  recording, noise-aware metadata, and the CLI report contract.


Benchmarks: slim-environment dataset loading and per-evaluation records (schema v2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Benchmark datasets now load from the local CSV cache without requiring the optional
  ``openml`` dependency: ``benchmarks/datasets.py`` checks the warm cache first and only imports
  ``openml`` on a genuine download, so cached datasets work in slim environments (e.g. CI-mirror
  containers) that do not install the ``[benchmark]`` extra; the actionable install-hint error on a
  cache miss is unchanged.
* Benchmark run records are now schema v2: every per-evaluation entry carries the full candidate
  configuration (``params``, all keys of the shared search space), each run JSON is stamped with
  ``schema_version``, and the directory manifest records ``record_schema_version``. Archive readers
  (``benchmarks/plot.py``, ``benchmarks/analyze_surrogates.py``) stay tolerant of legacy v1 runs that
  lack both fields, so archived runs remain re-plottable — and usable for acquisition-function
  comparisons — without re-running them. See ``benchmarks/DESIGN.md`` (v1.14).

Rule-rejection fixes for forbidden/conditional experiment rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Fixed a silent name mismatch that disabled ``forbidden`` and ``conditional``
  experiment rules in ``AML`` pipeline search: rules written with bare API-level
  parameter names (e.g. ``l1_ratio``) never matched the step-prefixed keys of a
  candidate pipeline's search space (e.g. ``stp_0__l1_ratio``), so banned
  candidates reached scikit-learn and failed there instead of being rejected by
  rule. Rules are now expanded per candidate pipeline to every declaring step,
  list values match "any of these values", callables pass through unchanged, and
  rules naming a parameter the pipeline does not declare keep the legacy skip
  behaviour.
* ``SurrogateRandomCV`` compares forbidden and conditional rule values with one
  scalar-or-list contract (a list/tuple/set of expected values matches when the
  candidate's value is any of them), and the final refit now mirrors trial-level
  conditioning so conditionally inactive parameters are not decoded back into a
  model the reported score never saw.
* When every candidate of an ``AML`` run is rejected by rule, both search paths
  fail with an explicit error naming the cause ("every candidate was rejected")
  instead of surfacing the underlying fit failure; per-candidate history records
  carry a ``rule_rejected`` flag, and each surrogate stage's trial history is
  profiled to tell rule rejections apart from ordinary failures.

pandas 3.x compatibility in fold-local preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``FoldLocalDataPreprocessor`` no longer routes plain string columns through
  TF-IDF under pandas >= 3, where they are inferred as a first-class
  ``StringDtype``: only explicitly declared text columns (``astype("string")`` /
  ``pandas.StringDtype()``) receive TF-IDF vectorisation; inferred strings fall
  back to one-hot encoding like object columns. The flatten step also passes
  ``feature_names_out="one-to-one"`` for current scikit-learn versions.

Launch visibility (docs and assets only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Readme: new "Understand what you tuned" section covering model explanations
  (permutation importance + partial-dependence plots, control plane and web UI)
  and sensitivity analysis (Sobol/Morris/δ indices), linking to the existing
  documentation guides; wording is scoped to descriptive claims per the launch
  plan guardrails.
* New figure ``docs/images/stability_contour.png`` — surrogate posterior mean of
  cross-validated accuracy over a two-dimensional hyperparameter slice on
  credit-g, showing local curvature in hyperparameter space (sharp peak vs
  plateau contrast) — plus its reproducible generator script
  ``docs/images/make_stability_contour.py`` (package public API only, no new
  dependencies).

Dependencies and packaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added ``lightgbm>=4.6`` and ``xgboost>=2.1`` to the package requirements: both
  are part of the default AML candidate pool; lightgbm was previously listed in
  ``requirements.txt`` only, xgboost is newly declared.

Docker and web UI upgrade
~~~~~~~~~~~~~~~~~~~~~~~~~

* Docker: multi-target ``Dockerfile`` (``dev`` with hot reload, ``prod`` with
  the built UI served by FastAPI on one port, non-root, healthcheck, ``/data``
  volume, ``EXTRAS`` build arg); compose profiles ``dev`` and ``prod``;
  ``SKSURROGATE_MODE`` and ``SKSURROGATE_WEBUI_DIST``; CI ``docker-smoke`` job.
* API: ``/api/experiments`` accepts ``preset``, ``strategy`` and resumes with
  ``AML(resume=True)``; new ``GET /api/experiments/presets``,
  ``POST /api/evaluation/{task}/{version}/explain`` and ``/intervals``,
  ``POST /api/design/latin-hypercube`` and ``/suggest-next``, and
  ``GET /api/projects/{id}/summary``.
* Web UI: Quick start experiments, Explain, Experiment Design and Projects
  pages; navigation grouped by workflow stage; Dashboard getting-started guide.

Walkthrough, validation fix and UI polish
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Docs: new ``docs/walkthrough.rst``, an end-to-end diabetes tutorial with
  screenshots and the matching REST and Python calls; sample CSVs
  ``docs/walkthrough-diabetes-{train,validation}.csv``; a "Run the web UI and
  API with Docker" section in the readme.
* Fix: ``mltrack.validate_data`` and the inference preflight no longer reject
  unseen values in continuous (float) columns as "unknown categories".
* ``/api/health`` returns the library version; Settings shows a Server card;
  the lineage rail also appears on Inference, Explain and Projects.
* Web UI: metric cards no longer wrap digit by digit; Explain labels every
  feature; Evaluation shows a muted note for classification-only curves.

JOSS submission and documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Opened the JOSS submission (``paper/paper.md``, ``paper/paper.bib``) as PR
  #8633 on ``openjournals/joss-papers``; added ``CITATION.cff`` pointing at the
  software record DOI 10.5281/zenodo.23631823 and the technical note
  ``docs/technical-note-rbf-gp-default.md``.
* D-track Hilbert-space surrogate study: new benchmark modules
  (``benchmarks/dtrack/{d1_approx,d2_compare,analyze}.py``), generated report
  ``docs/benchmarks/dtrack.md``, and standalone write-up
  ``docs/dtrack-study.md``; raw results (D1: 3816 cells, D2: 2160 cells) are
  archived at DOI 10.5281/zenodo.23652613.
* Phase C README rewrite with PyPI metadata polish; ORCID and affiliation
  filled in across docs; benchmark archive pointer updated to the v0.2.0
  software record (DOI 10.5281/zenodo.23631823).
* WebUI dependency bump: vite 8.3.4.

Scientific workflows and diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* AML categorical preprocessing is now learned within each cross-validation
  training fold. Added the reusable ``FoldLocalDataPreprocessor``; the eager
  whole-frame ``DataPreprocess`` API remains available, with leakage caveats
  documented for use before validation or cross-validation.
* Surrogate-search diagnostics now identify the selected response surface and
  record fit and local-optimization outcomes. Surrogate fit or prediction
  failures are not silently replaced by random search or another regressor.
* Added bounded permutation-importance and partial-dependence/ICE diagnostics,
  paired repeated-CV comparisons, and single-target regression diagnostics.
  Explanations describe model behavior, not causality; reference data, feature
  selection, and sampling bounds should be chosen deliberately.
* Added seeded Latin-hypercube designs and Gaussian-process LCB candidate
  recommendations for bounded continuous domains. Recommendations are not
  objective evaluations or experiment execution, and the benchmark evidence
  does not establish general or physical-world performance.
* Added split-conformal regression and conformalized quantile-regression
  intervals with disjoint calibration data. Quantile-interval integration in
  the UI/API remains deferred; coverage assumptions and limitations are
  documented in the Python guide.
* Added chronological splitter support for nested evaluation and quantile
  regression with pinball-loss evaluation in the Python/API and Evaluation UI
  workflows.
* Added API research projects that store questions and notes and reference
  existing dataset partitions, model bundles, and jobs without copying their
  artifacts. The API notebook demonstrates following those references.
* Added bounded mixed-type dataset profiles and reproducibility exports for
  API-trained bundles, including training-partition lineage, configuration,
  environment, and evaluation details. Added executable offline and Python API
  client notebooks; Jupyter remains optional.
* Sensitive-feature scans now include bounded, local checks for email,
  formatted US SSN, and formatted US phone patterns; reports contain counts,
  not matched values. External retraining triggers support bounded,
  deduplicated scheduler/webhook events. Events must be timezone-aware and no
  more than ten minutes old; data events must match the registered training
  partition fingerprint. Deduplication is process-local, and automated
  promotion is limited to threshold-gated staging.
* Bundle promotion checks schema compatibility by default; a mandatory metric
  threshold policy is not configured for ordinary promotion. Automated
  retraining promotion is separate and requires at least one finite metric
  threshold before staging.

API
~~~

* Added explicit ``AML(preset="classification"|"regression")`` quick starts
  using scikit-learn tree ensembles with task-appropriate scoring and bounded
  per-candidate evaluation budgets. Custom and legacy configurations remain
  available.
* API simplification (no removals): ``SurrogateRandomCV`` accepts scikit-learn
  names ``param_distributions`` and ``n_iter`` (``params``/``max_iter`` remain),
  search spaces may be shorthand (``(a, b)`` tuple -> ``Integer``/``Real``,
  list -> ``Categorical``; see ``as_space``), and fitted searches expose
  ``best_params_``. ``AML`` is now a ``BaseEstimator`` (repr, ``clone``);
  ``clone`` of ``SurrogateRandomCV``/``AML`` no longer fails. Fixed
  ``AML()`` default config referencing the removed ``sksurrogate`` module name.
  ``AML.fit(X, y, strategy="eoa")`` equals ``eoa_fit``; ``AML(stacking={...})``
  overrides the ``stack_*`` arguments; ``sampling="box"|"sphere"|"compact"``
  is accepted wherever a sampler class was.
  Fixed ``subgroup_performance_report(metric="precision")``, which computed
  recall, and made ``SurrogateRandomCV.best_score_`` follow scikit-learn's sign
  (higher is better; the loss is still in ``best_estimator_score``).
  ``cv_results_`` gains ``rank_test_score`` and ``loss``; ``mean_test_score`` is
  now higher-is-better, and ``best_index_`` is set. ``AML`` keeps the best
  surrogate stage instead of the last (which could be an unfitted ``inf``
  result), and ``StackingEstimator.set_params`` no longer ignores ``cv``/``n_jobs``.
  ``AML(resume=False)``: fits no longer silently resume a stale checkpoint (set ``resume=True`` for the old behaviour).
  EOA now accepts ``parents_prop``, ``elites_prop``, and
  ``max_generations``; the legacy spellings remain supported.
  ``ModelBundle.save``/``ModelBundle.load`` mirror ``save_bundle``/``load_bundle``.
  Removed redundant ``fit_transform`` overrides in ``SensAprx`` and
  ``CorrelationThreshold`` (inherited from ``TransformerMixin``).

* ``RegisterData`` accepts an optional ``preprocess_kwargs`` mapping, forwarded
  to ``DataPreprocess`` (e.g. ``bin_threshold``, ``cat_threshold``,
  ``force_impute``) when encoding is enabled; defaults are unchanged.

Logging and per-trial telemetry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Core modules ``structsearch.py``, ``eoa.py``, and ``aml.py`` now emit progress
  through the standard library ``logging`` (module loggers in the
  ``SKSurrogate`` namespace) instead of bare ``print()`` calls. Default console
  output is unchanged byte-for-byte: when no handler is configured anywhere in
  the logger's propagation chain, ``SKSurrogate._logutil.emit`` still writes the
  message to stdout; once a handler exists that stdout copy stops, so integrating
  pipelines see exactly one copy. ``verbose`` semantics are untouched
  (``>0`` → INFO, ``>1`` → DEBUG; EOA's per-generation line stays ungated INFO).
* ``SurrogateSearch.surrogate_fit_history_`` records gain a new ``fit_score``
  key: the fitted surrogate's own in-sample ``score`` on the data it was fitted
  to (``None`` when the regressor has no callable ``score`` or returns a
  non-finite value). Existing keys are unchanged.
* New ``SurrogateSearch.sampling_sources_`` parallel list tags every evaluation
  as ``init``, ``explore``, or ``surrogate_exploit``, appended in lockstep with
  ``evaluated`` and stored in the pickle checkpoint; checkpoints written before
  this key existed still load, with an empty tag list rather than invented
  provenance.
* New ``callbacks=None`` parameter on ``SurrogateRandomCV``: a callable or
  iterable of callables, each receiving one event dict per evaluated candidate
  (the trial record plus a zero-based ``trial`` index and its ``phase``),
  delivered at every record site — forbidden, zero-feature, pruned, and
  completed/failed. A callback that raises is logged and skipped; it cannot
  abort a search. ``get_params``/``clone`` compatibility is preserved.
* New ``SurrogateRandomCV.to_audit_dict()``: a JSON-safe audit bundle
  (``schema_version: 1``) carrying the estimator class and parameters, the
  encoded search space, scoring/CV summaries, budget and termination state, all
  trial records with trial indices, surrogate diagnostics, and sampling
  provenance; ``json.dumps()`` succeeds without a custom encoder.
* Documentation: new :doc:`telemetry` chapter covering logger names and level
  mapping, the unconfigured-stdout fallback, the ``fit_score`` and
  sampling-source records, the ``callbacks`` event schema (including the
  seed-trial ``phase = None`` caveat), and the ``to_audit_dict()`` bundle; the
  hyperparameter-optimization chapter links to it.

[0.2.0] - 2026-10-24
--------------------

Response-surface default change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``SurrogateSearch`` now defaults its response surface to a stock, un-tuned RBF
  Gaussian process (`sklearn.gaussian_process.GaussianProcessRegressor`) instead of the
  legacy cubic Hilbert polynomial (`HilbertRegressor(deg=3)`). Evidence: equal-budget
  benchmark suite -- the RBF-GP surface attains top mean CV on all six datasets at both
  B=30 and B=60, with seed-level win rates of 20W/17T/5L (B=30) and 25W/13T/4L (B=60)
  against Hilbert (two-sided exact binomial p approx. 4e-3 / 1e-4); the un-tuned RBF-KRR
  alternative never wins a seed at either budget. See ``benchmarks/DESIGN.md`` sec. 12;
  raw per-run records are archived at DOI 10.5281/zenodo.23629024 (software:
  DOI 10.5281/zenodo.23631823). The surrogate-firing floor (`MinEvals`) is unchanged, so
  the firing behavior matches that benchmark configuration exactly.
* Restore the v0.1 behaviour with `regressor=HilbertRegressor(deg=...)`; no new dependency
  (scikit-learn was already required).

Dependencies and packaging
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Lower-bound dependency pins in ``setup.py``/``requirements.txt`` exclude the
  vulnerable ranges flagged by Dependabot (numpy <1.22, scipy <1.10,
  scikit-learn <1.5, joblib <1.2, tqdm <4.66.3, fastapi <=0.109.0, lightgbm
  <4.6, uvicorn <0.11.7, python-multipart <0.0.31); bounds sit far below the
  versions CI currently installs and no upper pins were added.
* WebUI lockfile: source-map-js pinned to a non-vulnerable version
  (CVE-2026-93749).

[0.1.1] - 2026-10-11
--------------------

Republished from 0.1.0 with canonical MIT license metadata; code is identical
to 0.1.0.

Control-plane API and Web UI (branch ``ui``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added a FastAPI control plane under ``api/`` covering datasets, bundles,
  quality gates, registry/deployment, inference, monitoring, sensitivity,
  experiments, background jobs with WebSocket progress streaming, retraining,
  and a lineage traceability endpoint.
* Added a React/Vite web UI under ``webui/`` with pages for every pipeline
  stage, live job tracking, a responsive shell, and route-level code splitting.
* Added opt-in shared-key API authentication via the ``SKSURROGATE_API_KEY``
  environment variable (accepted through the ``X-API-Key`` header, Bearer
  tokens, or an ``api_key`` query parameter on WebSocket routes); unset keeps
  all endpoints open for local use.
* Made CORS origins configurable via ``SKSURROGATE_API_CORS_ORIGINS``, defaulting
  to local-only origins instead of a wildcard; the web UI stores and sends an
  API key from its Settings page.
* Added learning-curve evaluation: ``GET /api/evaluation/{task}/metrics`` lists
  the metrics relevant to the task's problem family, and
  ``GET /api/evaluation/{task}/learning-curves`` computes cross-validated
  training/held-out curves per bundle for one metric via scikit-learn. The
  Evaluation page's "Metric comparison" card now has one tab per metric on top
  of the side-by-side overview chart.
* Experiment searches gained surrogate-assisted mode
  (``surrogate_mode``/``surrogate_itrs``), conditional parameters
  (per-parameter ``depends_on``) and top-level ``forbidden`` rules, a per-run
  Pareto frontier (score vs. candidate duration) and ``top_pipelines`` in the
  job result, plus a ``POST /api/experiments/{task}/optimize-pipeline`` endpoint
  that tunes one explicit component sequence into a bundle. The Experiments page
  gained a search-strategy radio, live JSON validation hints, a "Top pipelines"
  table with per-row optimization, and a Pareto scatter chart.
* Added the ``stacking`` directive to experiment search spaces
  (``res``/``probs``/``decision``/``cv``/``n_jobs``), which configures the
  out-of-fold ``StackingEstimator`` wrappers ``AML`` applies to intermediate
  estimators; ``AML`` gained the matching ``stack_cv``/``stack_n_jobs``
  constructor parameters.

Bug fixes
~~~~~~~~~

* ``SurrogateRandomCV`` no longer refits a rejected combination. The optimizer's
  solution can land inside a forbidden region (or in one whose folds all
  failed), and refitting those settings either raised — e.g. ``lbfgs`` with
  ``penalty='l1'`` — or silently returned a banned model. The best *completed*
  trial is used instead, and the refit is skipped entirely when no trial
  completed.
* Fixed the experiments stage failing with ``ValueError: too many values to
  unpack (expected 2)``: the stored cross-validation spec (a JSON dictionary)
  was handed straight to ``AML``, and scikit-learn's ``check_cv`` treated that
  dictionary as an iterable of ``(train, test)`` folds and unpacked its keys.
  The API now rebuilds the splitter with ``SKSurrogate.mltrace.build_cv`` while
  keeping the JSON spec for the bundle audit event, and ``AML`` normalizes any
  dictionary ``cv`` the same way so no caller can repeat the mistake.
* ``EOA`` now rejects ``num_parents`` larger than the population size with a
  message naming both values, instead of failing later inside
  ``random.sample`` with ``Sample larger than population or is negative``.
* ``SKSurrogate.aml.default_config`` is defined at module level again, so
  ``AML()`` works without an explicit ``config``.
* Inference on a stored partition no longer fails with ``400: extra columns:
  target``. A registered partition is training data, so it carries the target
  label while the bundle schema holds only features; ``predict_batch`` validates
  strictly and rejected the label outright. The control-plane API now projects a
  stored partition onto the bundle schema (which also normalizes column order)
  and reports the dropped columns as ``ignored_columns``. Inline ``rows`` are
  still validated exactly as sent, since they represent the serving payload.

Phase 10: Documentation and release hardening
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added an offline-to-deployment lifecycle example covering bundles, registry
  promotion, batch inference, and drift reporting.
* Added a full synthetic classification workflow covering nested evaluation,
  quality gates, approval-gated promotion, fairness, delayed labels, and
  runtime monitoring.
* Added a full synthetic regression workflow covering nested evaluation, R2
  quality gates, approval-gated promotion, numeric drift, delayed-label
  MAE/MSE, and runtime monitoring.
* Added SQLite database and EOA checkpoint migration and rollback guidance.
* Documented public API coverage, supported runtime policy, benchmark thresholds,
  release checks, and the compatibility matrix.
* Added opt-in benchmarks for bundle loading, batch inference, and drift reports.

Phase 9: Distributed execution and CI/CD integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added durable local and optional Dask execution backends.
* Added bundle quality gates, triggered retraining, approval-gated deployment,
  and rollback workflows.
* Added Phase 9 operational documentation.

Phase 8: Governance, fairness, and security
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added audit events, ownership metadata, sensitive-feature reports, fairness
  and subgroup reports, and retention-aware artifact deletion.

Phase 7: Data quality, drift, and performance monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added schema, quality, distribution, range, prediction, delayed-label, and
  runtime monitoring reports with configurable alerts.

Phase 6: Batch and online inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added schema-validated batch prediction, the batch CLI, and a dependency-free
  WSGI serving adapter with health and readiness endpoints.

Phase 5: Model bundles and registry lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added portable model bundles, atomic persistence, compatibility checks,
  registry aliases, promotion and rollback history, and MLflow-compatible export.

Earlier phases
~~~~~~~~~~~~~~

* Phases 1 through 4 added dataset identity and schema contracts, explicit
  evaluation protocols, resource governance, search-space validation, and
  structured failure handling.
