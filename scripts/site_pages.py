"""Body content for the Phase-3 pages, imported by build_site.py.

Every sentence here is either (a) reproduced verbatim from an approved source —
readme.rst, docs/blog-honest-benchmark.md, CITATION.cff, or the homepage's own
sections — or (b) marked COPY_PENDING and replaced by a TODO block.
"""
import html

SEP = ' \n'

TODO = ('<p class="todo">Copy pending — this page is a structural shell. Nothing here '
        'is approved text, and the page is excluded from search indexing '
        '(robots: noindex) until the copy lands.</p>')


def L(rel, href):
    """Relative link that works both at the domain root and under /sksurrogate/."""
    return rel + href.lstrip("/")


def crumb(rel, label):
    return ('<div class="crumb"><a href="%s">sksurrogate</a><span>/</span>%s</div>'
            % (rel or "./", label))


# ---------------------------------------------------------------- quickstart
QUICK = '''      <pre><code>pip install sksurrogate</code></pre>'''


def quickstart(rel):
    return '''%s
      <h1>SKSurrogate in five minutes</h1>
      <p class="lede"><code>SurrogateRandomCV</code> is a drop-in for
        <code>RandomizedSearchCV</code>; ranges are written as plain tuples (integers or floats)
        and lists (categorical values).</p>
      <div class="prose">
        <h2>Install</h2>
        <pre><code>pip install sksurrogate</code></pre>
        <p>The package requires Python 3.10 or newer and ships the search layer on top of
          scikit-learn. Nothing else is needed for the tuner; the control plane runs from a
          published Docker image when you want it.</p>

        <h2>Control plane (optional)</h2>
        <pre><code>docker run --rm -p 8013:8013 -v sksurrogate-data:/data mghasemi/sksurrogate:latest</code></pre>
        <p>The published image (<a href="https://hub.docker.com/r/mghasemi/sksurrogate" target="_blank" rel="noopener">Docker Hub ↗</a>, v0.3.0) serves the UI and API on port 8013; datasets, experiments and model bundles stay in the named volume — the image ships no user data. Prefer the latest main branch? Build from a checkout: <code>docker compose --profile prod up -d</code>.</p>

        <h2>Optimize a model</h2>
        <pre><code>from sklearn.svm import SVC
from SKSurrogate import SurrogateRandomCV

search = SurrogateRandomCV(
    SVC(),
    param_distributions={"C": (1e-3, 10.0), "kernel": ["poly", "rbf"], "degree": (1, 4)},
    n_iter=30,
).fit(X, y)
search.best_params_, search.best_score_</code></pre>
        <p>The object is a <code>BaseSearchCV</code> subclass, so <code>best_params_</code>,
          <code>best_score_</code>, <code>cv_results_</code>, refitting and scoring behave as they
          do in scikit-learn — adopting it changes one import, not your codebase. Every candidate
          evaluation is recorded, so the result is an artifact you can inspect rather than a single
          returned winner.</p>

        <h2>Automatic pipeline design</h2>
        <pre><code>from SKSurrogate import AML

AML(preset="classification").fit(X, y)</code></pre>
        <p><code>AML(preset="regression")</code> covers the regression case and needs no search
          configuration. For pandas DataFrames the preset fits numeric, categorical, datetime and
          text preprocessing inside each cross-validation fold when <code>cat_cols</code> is
          omitted; array-like inputs should already be numeric. <code>fit(X, y, strategy="eoa")</code>
          selects evolutionary search.</p>

        <h2>Where to go next</h2>
        <ul>
          <li><a href="https://sksurrogate.readthedocs.io/en/latest/surrogate.html" target="_blank" rel="noopener">Surrogate search guide ↗</a>
            — strategies, acquisition, and how the response surface is fitted.</li>
          <li><a href="https://sksurrogate.readthedocs.io/en/latest/dataproc.html" target="_blank" rel="noopener">Preprocessing guide ↗</a>
            — how presets build fold-local preprocessing.</li>
          <li><a href="https://sksurrogate.readthedocs.io/en/latest/controlplane.html" target="_blank" rel="noopener">Control plane ↗</a>
            — the API, the UI and the compose profile on port 8013.</li>
          <li><a href="%s">Benchmarks</a> — what the search claims, and where it does not win.</li>
          <li><a href="%s">Concepts</a> — what surrogate-based optimization is, and when it pays.</li>
        </ul>
      </div>''' % (crumb(rel, "quickstart"), L(rel, "/benchmarks/"), L(rel, "/concepts/"))


# ---------------------------------------------------------------- benchmarks
def benchmarks(rel):
    return '''%s
      <h1>Honest benchmark: surrogate search vs TPE at small budgets</h1>
      <p class="lede">Equal evaluation budgets, six public datasets, seven seeds per cell. TPE is
        top on five of six; the interesting result is which response surface you fit.</p>
      <div class="prose">
        <p>All methods optimize the <em>same</em> estimator
          (<code>StandardScaler → RandomForestClassifier(50)</code>) over the <em>same</em> shared
          discrete grid (224 configurations across 3 parameters), with stratified 5-fold CV inside
          every search and one shared held-out 80/20 outer split per dataset-seed. Budgets are
          measured in <strong>candidate-model evaluations</strong> — a hardware-independent
          currency. Seven seeds per cell; 6 datasets from OpenML (pinned by <code>did</code>) plus
          one bundled synthetic set, B = 30 and B = 60 evaluations.</p>

        <table>
          <thead><tr><th>Arm</th><th>What it is</th></tr></thead>
          <tbody>
            <tr><td>SRS + RBF-GP</td><td>The package's search with its v0.2 default surface (stock un-tuned <code>GaussianProcessRegressor</code>)</td></tr>
            <tr><td>SRS, no surrogate</td><td>Identical outer loop, surrogate never engages — the ablation that isolates the surrogate's contribution</td></tr>
            <tr><td>random / Optuna-TPE</td><td>Uniform sampling and TPE as the standard baselines (<code>scikit-optimize</code> is archived upstream; recorded as “not runnable” rather than silently dropped)</td></tr>
          </tbody>
        </table>

        <h2>Mean final CV accuracy, B = 30</h2>
        <table>
          <thead><tr><th>dataset</th><th class="num">SRS + RBF-GP</th><th class="num">no surrogate</th><th class="num">random</th><th class="num">Optuna-TPE</th></tr></thead>
          <tbody>
            <tr><td>arrhythmia</td><td class="num">0.7020</td><td class="num">0.6910</td><td class="num">0.6941</td><td class="num"><b>0.7036</b></td></tr>
            <tr><td>credit-g</td><td class="num">0.7423</td><td class="num">0.7391</td><td class="num">0.7405</td><td class="num"><b>0.7436</b></td></tr>
            <tr><td>galaxy3</td><td class="num">0.6473</td><td class="num">0.6448</td><td class="num">0.6484</td><td class="num"><b>0.6507</b></td></tr>
            <tr><td>ionosphere</td><td class="num"><b>0.9352</b></td><td class="num">0.9347</td><td class="num">0.9347</td><td class="num">0.9337</td></tr>
            <tr><td>optdigits</td><td class="num">0.9403</td><td class="num">0.9392</td><td class="num">0.9380</td><td class="num"><b>0.9405</b></td></tr>
            <tr><td>spambase</td><td class="num">0.9280</td><td class="num">0.9269</td><td class="num">0.9272</td><td class="num"><b>0.9286</b></td></tr>
          </tbody>
        </table>
        <p>TPE is top on five of six datasets, and the whole table spans roughly half a point — a
          RandomForest(50) over three parameters is a flat landscape where no search method
          separates. On four of six datasets, uniform random reaches its own near-best in as few as
          2 evaluations.</p>

        <h2>The caveat, in full</h2>
        <p>Seven seeds per cell; budgets of 30–60 evaluations; the fixed-pipeline suite's space is
          only three parameters wide; D-track objectives are smooth and low-dimensional
          (d ≤ 3); the wall-clock arm uses a different currency (seconds, hardware-dependent)
          and is explicitly <em>not</em> comparable to the evaluation-count tables. These are
          small-budget results, and TPE remains competitive. SKSurrogate's edge is in stability,
          surface quality and evidence — not a universal accuracy crown.</p>

        <h2>Reproduce it</h2>
        <pre><code>pip install sksurrogate
python -m benchmarks.run --dataset all --budget 30      # full five-method suite (~1–3 h laptop-scale)
python -m benchmarks.dtrack.d2_compare                  # the regime-map study</code></pre>
        <ul>
          <li>Protocol contract + errata history:
            <a href="https://github.com/mghasemi/sksurrogate/blob/master/benchmarks/DESIGN.md" target="_blank" rel="noopener">benchmarks/DESIGN.md</a></li>
          <li>Raw B=30/B=60 records (462 runs):
            <a href="https://doi.org/10.5281/zenodo.23629024" target="_blank" rel="noopener">10.5281/zenodo.23629024</a></li>
          <li>D-track raw data + report:
            <a href="https://doi.org/10.5281/zenodo.23652613" target="_blank" rel="noopener">10.5281/zenodo.23652613</a></li>
          <li>Full write-up: <a href="%s">An Honest Benchmark of My Own AutoML Package</a></li>
        </ul>
      </div>''' % (crumb(rel, "benchmarks"), L(rel, "/blog/honest-benchmark/"))


# ---------------------------------------------------------------------- cite
def cite(rel):
    return '''%s
      <h1>Cite SKSurrogate</h1>
      <p class="lede">A repository link moves; a DOI does not. Cite the versioned software archive
        for the release you actually used, and add the dataset DOIs when your claim depends on the
        benchmark numbers.</p>
      <p class="chiprow"><span class="chip">MIT licence</span>
        <span class="chip">DOI 10.5281/zenodo.23631823</span>
        <span class="chip">ORCID 0000-0002-9551-1780</span></p>
      <div class="prose">
        <p style="color:var(--faint);font-size:13px">The authoritative record is
          <a href="https://github.com/mghasemi/sksurrogate/blob/master/CITATION.cff" target="_blank" rel="noopener">CITATION.cff</a>
          at the tag you are citing — the entries below mirror it and are updated when a release is
          archived.</p>

        <h2>Software (BibLaTeX)</h2>
        <pre><code id="bib-biblatex">@software{ghasemi2026sksurrogate,
  author  = {Ghasemi, Mehdi},
  title   = {SKSurrogate: a Python toolkit for surrogate-based optimization
             of expensive functions},
  year    = {2026},
  version = {0.2.0},
  doi     = {10.5281/zenodo.23631823},
  url     = {https://github.com/mghasemi/sksurrogate},
  note    = {MIT licence}
}</code></pre>
        <p><button class="copybtn" data-copy="bib-biblatex">copy BibLaTeX</button></p>

        <h2>Software (plain BibTeX)</h2>
        <p>Does your bibliography style not know <code>@software</code>? Use <code>@misc</code> —
          the fields are the same; only the entry type and the <code>howpublished</code> line
          differ.</p>
        <pre><code id="bib-bibtex">@misc{ghasemi2026sksurrogate,
  author       = {Ghasemi, Mehdi},
  title        = {SKSurrogate: a Python toolkit for surrogate-based optimization
                  of expensive functions},
  year         = {2026},
  howpublished = {Zenodo software archive},
  version      = {0.2.0},
  doi          = {10.5281/zenodo.23631823},
  url          = {https://github.com/mghasemi/sksurrogate},
  note         = {MIT licence}
}</code></pre>
        <p><button class="copybtn" data-copy="bib-bibtex">copy BibTeX</button></p>

        <h2>Datasets behind the published benchmarks</h2>
        <table>
          <thead><tr><th>Archive</th><th>Contents</th><th>DOI</th></tr></thead>
          <tbody>
            <tr><td>Equal-budget HPO suite</td><td>Raw B=30 / B=60 records, protocol DESIGN v1.9</td><td><a href="https://doi.org/10.5281/zenodo.23629024" target="_blank" rel="noopener">10.5281/zenodo.23629024</a></td></tr>
            <tr><td>D-track regime map</td><td>Controlled synthetic study — raw data and report</td><td><a href="https://doi.org/10.5281/zenodo.23652613" target="_blank" rel="noopener">10.5281/zenodo.23652613</a></td></tr>
            <tr><td>Software archive</td><td>Source release cited above</td><td><a href="https://doi.org/10.5281/zenodo.23631823" target="_blank" rel="noopener">10.5281/zenodo.23631823</a></td></tr>
          </tbody>
        </table>

        <h2>In text</h2>
        <p>With the key above, a LaTeX citation is simply
          <code>\\cite{ghasemi2026sksurrogate}</code>, or
          <code>\\parencite{ghasemi2026sksurrogate}</code> under BibLaTeX.</p>
        <p>If your work uses the benchmark protocol rather than the search layer, cite the dataset
          archive as well — the protocol contract it implements is
          <code>benchmarks/DESIGN.md</code> in the repository.</p>

        <h2>What to cite when the release has no DOI yet</h2>
        <p>Zenodo mints DOIs per uploaded release. If you are using a version that has not been
          archived yet, cite the most recent archived version in <code>CITATION.cff</code> and say
          so in your methods section; that keeps your citation resolvable and honest about which
          build you ran. The installed version is always available from the package itself:
          <code>python -c "import sksurrogate; print(sksurrogate.__version__)"</code>.</p>
      </div>''' % crumb(rel, "cite")


# --------------------------------------------------------------------- about
def about(rel):
    return '''%s
      <h1>About SKSurrogate</h1>
      <p class="lede">Software you can audit, not just install: an open-source,
        scikit-learn-native environment for optimizing expensive objectives, where every number it
        reports is traceable to a protocol, a seed and an archived record.</p>
      <p class="chiprow"><span class="chip">MIT licence</span>
        <span class="chip">Python 3 · scikit-learn</span>
        <span class="chip">v0.3.0</span>
        <span class="chip">open development</span></p>
      <div class="prose">
        <h2>What it is</h2>
        <p>SKSurrogate implements surrogate-guided optimization for objectives that are expensive
          to evaluate: each evaluation costs minutes rather than milliseconds, so the question is
          not how many candidates you can try but how much you learn from each one. The package
          builds a response surface over the space you have sampled, then uses that surface to
          decide where to spend the next evaluation.</p>
        <p>Around that core it provides the machinery a research or production workflow needs:
          dataset registration with identity, a drop-in <code>BaseSearchCV</code> tuner, statistical
          evaluation with nested cross-validation and conformal intervals, sensitivity and
          dependence analysis, experiment design for expensive runs, a control plane that executes
          and logs searches, and a research workspace that keeps each claim attached to the
          evidence that supports it.</p>

        <h2>What it is not</h2>
        <div class="prose">
          <p><strong>Not a managed cloud service.</strong> Self-hosted and single-machine by
            design — local API or one compose profile. If you need multi-node training or GPU
            fleets, that is a different category of tool.</p>
          <p><strong>No LLM magic in the box.</strong> The shipped tooling is deterministic. A
            bounded, opt-in cited-note synthesis (your endpoint, aggregate facts only, enforced
            citations) already runs inside it; a conversational research assistant is roadmap, not
            this release.</p>
          <p><strong>Not a black-box AutoML zoo.</strong> The search space is your pipeline plus
            the ranges you declare, and every evaluation is recorded — results are auditable
            artifacts, not opaque picks from a fixed model zoo.</p>
        </div>
        <p>What is dated and gated lives on the <a href="%s">roadmap</a>.</p>

        <h2>How it is developed</h2>
        <p>Development happens in the open on
          <a href="https://github.com/mghasemi/sksurrogate" target="_blank" rel="noopener">GitHub</a>.
          The benchmark suite is part of the repository, not a marketing appendix: a frozen protocol
          contract, pre-registered hypotheses with explicit falsifiers, and raw run records archived
          with DOIs so a reader can check the advertised results without rerunning anything.</p>
        <p>That has two consequences. The changelog is public and versioned, documentation lives at a
          stable address on Read the Docs instead of being regenerated per release, and when the
          data says the package does not beat a baseline, that gets published too — see
          <a href="%s">the honest benchmark post</a>.</p>

        <h2>Who maintains it</h2>
        <p>SKSurrogate is written and maintained by <strong>Mehdi Ghasemi</strong>, a
          scientist-mathematician. Contact is by email or through the issue tracker — see
          <a href="%s">contact</a>. There is no support contract and no service-level agreement
          attached to the project; it is maintained software, not a hosted product.</p>
        <p class="mono" style="font-size:11px">
          <a href="https://github.com/mghasemi" target="_blank" rel="noopener">GitHub ↗</a> ·
          <a href="https://scholar.google.com/citations?user=1PlVtHsAAAAJ" target="_blank" rel="noopener">Google Scholar ↗</a> ·
          <a href="https://orcid.org/0000-0002-9551-1780" target="_blank" rel="noopener">ORCID ↗</a> ·
          <a href="https://mghasemi.net" target="_blank" rel="noopener">mghasemi.net ↗</a></p>

        <h2>Licence</h2>
        <p>SKSurrogate is released under the <strong>MIT licence</strong>. You may use it
          commercially, modify it and redistribute it, provided the copyright notice and permission
          notice are retained. The canonical wording is the
          <a href="https://opensource.org/license/mit" target="_blank" rel="noopener">MIT licence</a>.</p>
      </div>''' % (crumb(rel, "about"), L(rel, "/roadmap/"),
                   L(rel, "/blog/honest-benchmark/"), L(rel, "/contact/"))


# ------------------------------------------------------------------- roadmap
def roadmap(rel):
    return '''%s
      <h1>Roadmap</h1>
      <p class="lede">Dated and gated. An item says “next” or “later” — never “it does that today”
        unless it shipped.</p>
      <div class="prose">
        <h2>Next</h2>
        <p>Quantile / conformal-interval diagnostics in the UI and API — the Python split-conformal
          protocol already ships; the calibration-data contract for serving them is the next
          step.</p>
        <h2>Later</h2>
        <p>LLM-assisted synthesis beyond bounded cited notes — toward a human-gated,
          citation-enforced research assistant. Shipped only with its gates intact.</p>
        <h2>Not on the roadmap</h2>
        <p>A managed cloud service, a fixed model zoo, or a chat assistant inside the box. See
          <a href="%s">what SKSurrogate is not</a>.</p>
      </div>''' % (crumb(rel, "roadmap"), L(rel, "/about/"))


# ---------------------------------------------------------------------- blog
def blog(rel):
    return '''%s
      <h1>Notes and benchmarks</h1>
      <p class="lede">Write-ups of the studies behind the product page: the protocol, the seeds, the
        results that supported the hypothesis, and the ones that refuted it.</p>
      <ul class="postlist">
        <li>
          <a href="%s">
            <div class="date">26 October 2026 · benchmarks</div>
            <h2>An Honest Benchmark of My Own AutoML Package</h2>
            <p>At equal evaluation budgets, surrogate-guided search does not beat Optuna-TPE on
              accuracy. What <em>is</em> real: the response surface matters more than the search
              mechanism, and the most useful finding was a defect in our own pipeline — ordinal
              encoding was silently masking the surrogate's advantage.</p>
          </a>
        </li>
      </ul>''' % (crumb(rel, "blog"), L(rel, "/blog/honest-benchmark/"))


# ------------------------------------------------------------------- contact
def contact(rel):
    return '''%s
      <h1>Contact</h1>
      <p class="lede">One address, three destinations. SKSurrogate is maintained software, not a
        hosted service — there is no sales team and no support desk.</p>
      <div class="prose">
        <p style="font-size:17px;color:var(--ink)">Email:
          <a href="mailto:mehdi.ghasemi@gmail.com">mehdi.ghasemi@gmail.com</a></p>

        <h2>Bug reports</h2>
        <p>Use the <a href="https://github.com/mghasemi/sksurrogate/issues" target="_blank" rel="noopener">GitHub
          issue tracker</a>. A report that can be reproduced gets fixed; one that cannot gets a
          question. Please include:</p>
        <ul>
          <li>the version — <code>python -c "import sksurrogate; print(sksurrogate.__version__)"</code></li>
          <li>your Python and scikit-learn versions</li>
          <li>a minimal snippet or dataset shape that triggers the problem, and the full traceback</li>
          <li>what you expected versus what happened</li>
        </ul>

        <h2>Usage questions</h2>
        <p>The <a href="https://sksurrogate.readthedocs.io/en/latest/intro.html" target="_blank" rel="noopener">documentation</a>
          answers most installation, API and control-plane questions, and the
          <a href="%s">benchmark post</a> explains what the package does and does not claim. If
          something is genuinely unclear after that, open an issue with the <code>question</code>
          label or send email — documentation gaps are treated as bugs.</p>

        <h2>Corrections to published results</h2>
        <p>Numbers on this site come from the benchmark suite in the repository. If you find an
          arithmetic error, a mismatched table, or a claim the archived records do not support,
          email or open an issue with the dataset, budget and seed cell you are checking. Material
          corrections are recorded in the changelog rather than applied silently.</p>

        <h2>Security</h2>
        <p>Do not open a public issue for a security problem. Email
          <a href="mailto:mehdi.ghasemi@gmail.com?subject=%%5Bsecurity%%5D%%20sksurrogate">mehdi.ghasemi@gmail.com</a>
          with a subject line beginning <code>[security]</code>, including the affected version, the
          impact, and a reproduction if one exists.</p>

        <h2>What this project does not offer</h2>
        <p>No support contracts, no service-level agreements, no paid consulting, and no hosted demo
          to book. The software is MIT-licensed and self-service by design.</p>

        <h2>This website</h2>
        <p>The site is static HTML with no cookies, no analytics and no third-party trackers, and it
          serves its own fonts. Pages are served by GitHub Pages, which may log requests under
          <a href="https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement" target="_blank" rel="noopener">GitHub's
          privacy statement</a>. Nothing else is collected.</p>
      </div>''' % (crumb(rel, "contact"), L(rel, "/blog/honest-benchmark/"))


# ----------------------------------------------------------------------- faq

# ---------------------------------------------------------------- concepts
def concepts_index(rel):
    return '''%s
      <h1>Concepts</h1>
      <p class="lede">Three ideas do the work in SKSurrogate. Each is defined here in the terms the
        documentation uses, with the evidence the package publishes on its own behaviour.</p>
      <div class="cards">
        <a class="card" href="%s">
          <div class="k">optimization</div>
          <h3>Surrogate-based optimization</h3>
          <p>Fit an approximate model to expensive evaluations and use it to choose the next
            point — when that saves evaluations, and when it does not.</p>
        </a>
        <a class="card" href="%s">
          <div class="k">uncertainty</div>
          <h3>Split-conformal intervals</h3>
          <p>Finite-sample marginal coverage from a calibration split, and the assumptions that
            make the guarantee fail.</p>
        </a>
        <a class="card" href="%s">
          <div class="k">evaluation</div>
          <h3>Nested cross-validation</h3>
          <p>Keep model selection separate from the estimate of generalisation, so the score you
            report is not the score you optimised.</p>
        </a>
      </div>''' % (crumb(rel, "concepts"),
                   L(rel, "/concepts/surrogate-optimization/"),
                   L(rel, "/concepts/conformal-intervals/"),
                   L(rel, "/concepts/nested-cross-validation/"))


def concept_surrogate(rel):
    return '''%s
      <h1>What is surrogate-based optimization?</h1>
      <p class="lede">Surrogate-based optimization fits an approximate model — the surrogate — to
        the evaluations of an expensive function, then uses that model to decide where to evaluate
        next. The cost of a prediction is negligible compared with the cost of an evaluation, so the
        search happens on the model and only the chosen points are paid for.</p>
      <p class="chiprow"><span class="chip">also: surrogate modelling, SBO</span>
        <span class="chip">budget: number of evaluations</span>
        <span class="chip">default surface: RBF Gaussian process</span></p>
      <div class="prose">
        <h2>The mechanism</h2>
        <p>For a function <code>f</code> that can only be evaluated a limited number of times, one iteration
          is: sample a point in the feasible set; evaluate <code>f</code> there; add the pair to the evaluation
          set <code>E</code>; fit a surrogate to <code>E</code>; minimise the surrogate to propose the next point;
          evaluate <code>f</code> at that proposal. Repeat until the budget is exhausted, then return the best
          evaluated pair. The package's own description of this loop, including the
          <code>MinEvals</code> random-evaluation phase before the surrogate engages, is in the
          <a href="https://sksurrogate.readthedocs.io/en/latest/surrogate.html" target="_blank" rel="noopener">surrogate search guide</a>.</p>
        <p>Three design choices carry most of the behaviour: how the initial and subsequent points
          are sampled, which model is fitted, and how the model is minimised. SKSurrogate ships
          three samplers — a random point in the feasible set, a cube around the last point (with
          <code>init_radius</code> 2.0 and a <code>contraction</code> of 0.9 by default), and a
          sphere around the last point — and accepts user-defined sampling classes with the same
          signature. The surrogate is any scikit-learn regressor; the default is an RBF Gaussian
          process.</p>

        <h2>When it pays, and when it does not</h2>
        <p>The case for a surrogate is economic, not algorithmic: it pays when one evaluation costs
          minutes and a model fit costs milliseconds. The package publishes what happens when that
          condition is not met. On six public datasets at a budget of 30 evaluations, the
          surrogate-guided search is <em>at parity</em> with Optuna-TPE, which is top on five of the
          six; the whole table spans about half a point of accuracy. At 60 evaluations the
          surrogate arm leads all six datasets. The honest summary is that on a flat, three-parameter
          landscape your choice among these tools will not move your accuracy much — the
          <a href="%s">benchmark page</a> has the tables.</p>
        <p>What the surrogate buys instead is evaluation efficiency. On the 1,220-pipeline space the
          search reached within 1&nbsp;%% of its own final best in a median of <strong>6
          evaluations</strong>, against 11–17 for the other arms at identical final accuracy. That
          advantage does not transfer to wall-clock time: the Gaussian-process fit grows as
          O(n³), so per-fit cost rose from 3&nbsp;ms to 174&nbsp;ms across a run and 49&nbsp;%% of
          a 120&nbsp;s budget went to search overhead (22&nbsp;%% fitting, 27&nbsp;%% acquisition).</p>

        <h2>The response surface matters more than the search</h2>
        <p>Inside the same search loop, changing only the fitted surface changes the result. An
          un-tuned RBF Gaussian process beat the legacy cubic Hilbert polynomial on all six datasets
          at both budgets, with seed-level wins of 20W/17T/5L at B=30 and 25W/13T/4L at B=60 (exact
          binomial <code>p ≈ 4×10⁻³</code> and <code>p ≈ 10⁻⁴</code>). That is why v0.2 made the
          RBF Gaussian process the default and kept <code>HilbertRegressor</code> available. A
          controlled synthetic study explains the mechanism: the polynomial regressor is exact on
          in-span objectives but truncates elsewhere, the stock Gaussian process has no noise term
          and interpolates noise, and a tree ensemble is the robust generalist.</p>

        <h2>How SKSurrogate implements it</h2>
        <p><code>SurrogateRandomCV</code> is a <code>BaseSearchCV</code> subclass, so it has the
          call shape of <code>RandomizedSearchCV</code>: pass an estimator, a
          <code>param_distributions</code> mapping of tuples (numeric ranges) and lists (categorical
          values), and <code>n_iter</code>. For pipeline design, <code>AML</code> adds an
          evolutionary search over transformers and estimators
          (<code>fit(X, y, strategy="eoa")</code>) before the local surrogate refinement. Every
          evaluation is recorded, which is what makes the result an artifact rather than a single
          returned winner.</p>

        <h2>What it does not do</h2>
        <p>The surrogate is an approximation of the observed surface, not a certificate: it can be
          confidently wrong between samples, and no coverage guarantee is claimed for the search
          itself. It does not remove the need for a held-out evaluation, and it does not make a flat
          objective steep. Compare with a single-split score using
          <a href="%s">nested cross-validation</a>, and quantify deployed uncertainty with
          <a href="%s">conformal intervals</a>.</p>
      </div>'''  % (crumb(rel, "concepts <span>/</span> surrogate optimization"),
                    L(rel, "/benchmarks/"),
                    L(rel, "/concepts/nested-cross-validation/"),
                    L(rel, "/concepts/conformal-intervals/"))


def concept_conformal(rel):
    return '''%s
      <h1>What is a split-conformal prediction interval?</h1>
      <p class="lede">A split-conformal interval is a prediction interval built from the residuals
        of a model on a calibration set that was not used to fit it. It comes with a finite-sample
        <em>marginal</em> coverage guarantee — under assumptions that are stated, and that fail in
        ways worth knowing before you report a number.</p>
      <p class="chiprow"><span class="chip">guarantee: marginal</span>
        <span class="chip">requires: exchangeability</span>
        <span class="chip">calibration ≠ training</span></p>
      <div class="prose">
        <h2>The procedure</h2>
        <p>Split the labelled data into a training part and a disjoint calibration part. Fit the
          estimator on the training part only. Compute absolute residuals on the calibration part.
          Take the finite-sample corrected empirical quantile of those residuals, and report
          the predicted value ± <code>q</code> for each test point. In SKSurrogate this is
          <code>split_conformal_regression_interval(regressor, X_train, y_train, X_calibration,
          y_calibration, X_test, alpha=0.1)</code>. The estimator is cloned and fitted only on the
          training data, and the function does not accept test labels — by construction, the interval
          cannot peek at the answer it is meant to bracket.</p>

        <h2>What the guarantee covers</h2>
        <p>The marginal coverage statement assumes that calibration examples and future examples are
          exchangeable — drawn from the same distribution, in any order — and that the fitted model
          was chosen independently of the calibration set. If a hyperparameter search used the
          calibration rows, the correction is computed on data the model has already seen and the
          guarantee no longer applies. Coverage is <em>marginal</em>: it holds on average over draws,
          not per subgroup, not per region of feature space, and not under distribution shift.</p>

        <h2>Checking it in practice</h2>
        <p>A guarantee you never test is a claim, not a measurement. Given held-out labels,
          <code>prediction_interval_diagnostics(y_holdout, lower, upper, alpha=0.1)</code> reports
          nominal and empirical coverage, a Wilson score interval for the empirical coverage (which
          describes finite-sample uncertainty and is <em>not</em> the conformal guarantee), mean and
          median width, and the mean Winkler score for a central two-sided interval. An optional
          <code>groups</code> argument adds bounded per-group coverage counts and Wilson intervals;
          small groups are highly uncertain, and marginal coverage does not imply subgroup coverage.
          Unbounded intervals stay in the coverage count, while width and interval score are
          <code>None</code> if any interval is unbounded — a detail that matters when a model is
          badly miscalibrated.</p>

        <h2>Conformalized quantile regression</h2>
        <p>When two conditional-quantile estimators are available — typically at
          <code>α/2</code> and <code>1−α/2</code> — the conformal correction can widen or narrow the base
          interval according to the calibration residuals
          (<code>split_conformal_quantile_interval</code>). Predicted bounds are ordered before
          calibration, and the result includes an <code>empty_interval_mask</code>: a negative
          correction can produce an empty set when the corrected lower bound exceeds the upper one.
          The correction is infinite when the requested rank exceeds the calibration size, which is
          the honest answer to "you asked for 99.9&nbsp;%% coverage from 40 calibration points".</p>

        <h2>Limitations to state when you report one</h2>
        <ul>
          <li>Marginal, not conditional: no per-subgroup coverage claim follows.</li>
          <li>Exchangeability, not i.i.d. forever: drift voids the guarantee.</li>
          <li>The model must not be selected on the calibration split.</li>
          <li>Interval width is a property of the data and model, not a quality score; a narrow
            interval on a bad model is still narrow.</li>
        </ul>
        <p>The API reference and the full caveats are in the
          <a href="https://sksurrogate.readthedocs.io/en/latest/mltrace.html" target="_blank" rel="noopener">tracking and evaluation guide</a>;
          the deployed model bundle and its interval diagnostics are described in the
          <a href="https://sksurrogate.readthedocs.io/en/latest/controlplane.html" target="_blank" rel="noopener">control-plane guide</a>.
          The measurement half of the workflow — what a single split can and cannot tell you — is on
          <a href="%s">nested cross-validation</a>.</p>
      </div>''' % (crumb(rel, "concepts <span>/</span> conformal intervals"),
                   L(rel, "/concepts/nested-cross-validation/"))


def concept_nested(rel):
    return '''%s
      <h1>What is nested cross-validation?</h1>
      <p class="lede">Nested cross-validation keeps the step that <em>chooses</em> a model separate
        from the step that <em>estimates</em> its generalisation error. The inner loop selects;
        the outer loop scores the selection procedure on data the inner loop never saw. Without
        that separation, the reported score is the score you optimised, and it is optimistic by
        construction.</p>
      <p class="chiprow"><span class="chip">inner: selection</span>
        <span class="chip">outer: estimate</span>
        <span class="chip">repeated CV: interval, not a test</span></p>
      <div class="prose">
        <h2>Why a single split misleads</h2>
        <p>When the same folds are used to tune and to score, every configuration is evaluated on
          the same data and the winner's score inherits the selection. The bias grows with the number
          of candidates and shrinks with the size of the data — which is exactly the regime where
          expensive problems live. Nested CV answers a different question: not "how well does the
          chosen configuration score on these folds" but "how well does this <em>selection
          procedure</em> score on unseen folds".</p>

        <h2>What SKSurrogate's <code>nested_cv_evaluation</code> does</h2>
        <p><code>nested_cv_evaluation</code> computes inner-CV scores and outer-fold estimates for
          the estimator you supply, then evaluates it on the untouched validation partition. It
          stores fold-level predictions and metrics, mean outer scores, repeated-CV confidence
          intervals and the final validation score in task metadata. Mixed-type DataFrame features
          are preprocessed inside each training fold, and unseen categories are ignored at scoring
          time rather than being folded into the vocabulary — the leakage that quietly inflates a
          score. Group-aware splitters preserve group labels through the evaluation path.</p>
        <p>Two limits are stated plainly in the documentation and should be repeated here: the
          function <strong>does not search among estimators or tune hyperparameters</strong>, and it
          does not replace the search layer. It is the evaluation half — the part that scores what
          the search produced without re-using the search's own optimised number. Tuning happens in
          <code>SurrogateRandomCV</code> / <code>AML</code>; the evaluation asks how that result
          behaves on partitions it never touched.</p>

        <h2>Reading the result honestly</h2>
        <p>Results carry an evaluation protocol: registered partition fingerprints, the CV splitter
          specification, the repeat seed policy, the preprocessing path, the scoring identity and
          whether a validation set was used. When two results are compared with
          <code>compare_repeated_cv_results</code>, known mismatches are rejected and the output
          reports whether alignment could be verified — opaque custom scorers, missing fingerprints
          and older result dictionaries remain explicitly <em>unverified</em>, which is a signal, not
          a gap to paper over.</p>
        <p>The paired comparison reports the candidate-minus-reference mean and median repeat-score
          difference, a percentile bootstrap interval and paired Cohen's <code>d_z</code>. It is deliberately
          descriptive: the resampling unit is a complete repeat-level score difference, repeats reuse
          observations, and the interval summarizes variation across repeats. It is not an
          independent-sample significance test, and excluding zero is not a significance claim. BCa
          intervals do not resolve dependence between repeats either; for a confirmatory comparison,
          define the independent unit and the comparison family before looking at the data.</p>

        <h2>Time series</h2>
        <p>For ordered observations, pass <code>TimeSeriesSplit</code> for both the inner and outer
          loops and choose a <code>gap</code> that excludes observations between the training and test
          windows. Rows must already be in chronological order: the package does not infer timestamps
          or sort them, and it does not construct lag, rolling or horizon features — doing that
          without future leakage is the caller's responsibility.</p>

        <h2>What it does not settle</h2>
        <p>Nested CV estimates the procedure's error on the distribution your folds were drawn from.
          It says nothing about a shifted deployment population, and it does not produce prediction
          intervals — that is <a href="%s">split conformal</a>. Reporting one number from one split
          is still the most common way to overstate a result; the API reference is in the
          <a href="https://sksurrogate.readthedocs.io/en/latest/mltrace.html" target="_blank" rel="noopener">tracking and evaluation guide</a>.</p>
      </div>''' % (crumb(rel, "concepts <span>/</span> nested cross-validation"),
                   L(rel, "/concepts/conformal-intervals/"))

# ----------------------------------------------------------------------- faq
FAQ_QA = [
    ("What is SKSurrogate?",
     "SKSurrogate is an open-source Python package built on scikit-learn for surrogate-based optimization "
     "of expensive objectives — hyperparameter and pipeline search, statistical evaluation, sensitivity "
     "analysis and experiment design — with a control plane that runs and logs searches and a research "
     "workspace that records evidence. It is MIT licensed and installs with pip."),
    ("Is SKSurrogate a replacement for scikit-learn?",
     "No. It is a search and evaluation layer on top of scikit-learn. SurrogateRandomCV is a BaseSearchCV "
     "subclass with the same call shape as RandomizedSearchCV, and your estimators, splitters, scorers and "
     "pipelines are used as they are."),
    ("Does it work with scikit-learn pipelines and parameter grids?",
     "Yes. You pass an estimator or a pipeline plus a param_distributions mapping, with numeric ranges as "
     "tuples and categorical choices as lists. Because the class is a BaseSearchCV subclass, best_params_, "
     "best_score_, cv_results_ and refitting behave as they do in scikit-learn."),
    ("Is it faster than Optuna?",
     "Not at small budgets. On six public datasets at 30 evaluations Optuna-TPE is top on five of six and the "
     "whole table spans about half a point of accuracy; at 60 evaluations the surrogate arm leads all six. "
     "The surrogate's advantage is evaluation efficiency rather than wall-clock speed: in the pipeline-space "
     "study it reached within 1 per cent of its final best in a median of 6 evaluations against 11 to 17 for "
     "the other arms, while the Gaussian-process fit grows as O(n^3) and search overhead consumed about half "
     "of a 120-second budget."),
    ("How many evaluations does the search need?",
     "Budgets are counted in candidate-model evaluations, not seconds, so they are hardware-independent. The "
     "published equal-budget suite used 30 and 60 evaluations with seven seeds per cell; the pipeline-space "
     "study used equal wall-clock, equal 40 evaluations, and a wall-budget cap."),
    ("What surrogate model does it use by default?",
     "Since v0.2 the default response surface is a stock, un-tuned RBF GaussianProcessRegressor. The previous "
     "cubic Hilbert polynomial remains available as HilbertRegressor, and any scikit-learn regressor can be "
     "supplied. The change was made on the evidence of the equal-budget benchmark, where the Gaussian process "
     "beat the polynomial surface at both budgets."),
    ("Does it handle classification and regression?",
     "Yes. AML(preset=\"classification\") and AML(preset=\"regression\") need no search configuration. For "
     "pandas DataFrames the preset fits numeric, categorical, datetime and text preprocessing inside each "
     "cross-validation fold when cat_cols is omitted."),
    ("How does it handle categorical features?",
     "Through the preset's fold-local preprocessing, or explicitly with cat_cols. This is worth checking in "
     "your own pipeline: the benchmark study found ordinal encoding of unordered categories to be a defect "
     "in the package itself. Replacing it with a mixed categorical surface lifted best cross-validated score "
     "by at least 0.3 points on four of six datasets and moved the search above TPE on three of them."),
    ("What guarantee do the prediction intervals have?",
     "The split-conformal intervals assume that calibration and future examples are exchangeable and that "
     "the model was chosen independently of the calibration set. Under those assumptions they provide "
     "finite-sample marginal coverage — not per-subgroup or conditional coverage, and not coverage under "
     "distribution shift. Coverage is meant to be measured on held-out labels with "
     "prediction_interval_diagnostics."),
    ("Can I run it without a server or cloud account?",
     "Yes. It is a Python package. The control plane with its API and UI is optional and runs from a Docker "
     "Compose profile on a single port, 8013, with an optional API key; nothing in the loop calls an external "
     "service."),
    ("How do I cite SKSurrogate?",
     "Cite the archived release rather than the repository URL. The software archive recorded in CITATION.cff "
     "carries a DOI, and the benchmark records and the D-track study have their own DOIs. Ready-to-paste "
     "BibTeX and BibLaTeX are on the citation page."),
    ("What is SKSurrogate not?",
     "Not a managed cloud service, not a fixed model zoo, and not a chat assistant. The shipped tooling is "
     "deterministic, the search space is your pipeline plus the ranges you declare, and every evaluation is "
     "recorded. A bounded, opt-in cited-note synthesis exists; a conversational research assistant is roadmap, "
     "not this release."),
]


def faq_jsonld():
    import json as _json
    data = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {"@type": "Question", "name": q,
             "acceptedAnswer": {"@type": "Answer", "text": a}}
            for q, a in FAQ_QA
        ],
    }
    return '<script type="application/ld+json">\n%s\n</script>\n' % _json.dumps(data, ensure_ascii=False)


def faq(rel):
    blocks = "\n".join(
        "        <h2>%s</h2>\n        <p>%s</p>" % (q, a) for q, a in FAQ_QA)
    return '''%s
      <h1>SKSurrogate FAQ</h1>
      <p class="lede">Twelve questions that decide whether the package fits your problem. Every answer
        restates a documented behaviour or a published result; the source is linked where the detail
        matters.</p>
      <div class="prose">
%s
        <h2>Where to go next</h2>
        <p>The <a href="%s">quickstart</a> installs and runs the tuner; the
          <a href="%s">benchmark page</a> has the tables and the caveats; the
          <a href="%s">concepts</a> explain surrogate optimization, conformal intervals and nested
          cross-validation in the terms the documentation uses.</p>
      </div>''' % (crumb(rel, "faq"), blocks,
                   L(rel, "/quickstart/"), L(rel, "/benchmarks/"), L(rel, "/concepts/"))


# ----------------------------------------------------------------- vs-optuna
def vs_optuna(rel):
    return '''%s
      <h1>SKSurrogate vs Optuna and skopt</h1>
      <p class="lede">A comparison page is only useful if it says where the other tools win.</p>
      <div class="prose">
        <h2>Where it sits</h2>
        <table>
          <thead><tr><th>Claim</th><th>Artifact</th></tr></thead>
          <tbody>
            <tr><td><b>A stable plateau, not a brittle peak.</b> The surrogate response surface makes the neighborhood of the optimum inspectable — broad plateaus and needle-like peaks are visible at a glance before you commit a run.</td><td><a href="%s">figure</a></td></tr>
            <tr><td><b>Statistical evaluation, not a single score.</b> Repeated nested cross-validation, paired comparison protocol checks, and split-conformal intervals with coverage diagnostics.</td><td><a href="https://sksurrogate.readthedocs.io/en/latest/controlplane.html" target="_blank" rel="noopener">docs ↗</a></td></tr>
            <tr><td><b>A research workspace with receipts.</b> Projects, hypotheses with a status lifecycle, evidence records that snapshot artifacts — plus LHS + GP-LCB experiment design for expensive physical experiments.</td><td><a href="https://sksurrogate.readthedocs.io/en/latest/mltrace.html" target="_blank" rel="noopener">docs ↗</a></td></tr>
            <tr><td><b>Surrogate hybrid search, honestly benchmarked.</b> Equal-budget comparison against TPE, random and no-surrogate arms — published with the raw data DOI and a plain statement of where the surrogate loses.</td><td><a href="%s">benchmarks</a></td></tr>
            <tr><td><b>A drop-in, not a rewrite.</b> SurrogateRandomCV is a BaseSearchCV subclass — the same call shape as RandomizedSearchCV, so adopting it changes one import, not your codebase.</td><td><a href="https://github.com/mghasemi/sksurrogate/blob/master/readme.rst" target="_blank" rel="noopener">readme ↗</a></td></tr>
            <tr><td><b>Yours to run.</b> One compose profile on a single port (8013), optional API key, MIT licence — no external service in the loop.</td><td><a href="https://sksurrogate.readthedocs.io/en/latest/controlplane.html" target="_blank" rel="noopener">docs ↗</a></td></tr>
          </tbody>
        </table>
        <h2>Head-to-head</h2>
        %s
      </div>''' % (crumb(rel, "vs-optuna"), L(rel, "/#evidence"), L(rel, "/benchmarks/"), TODO)


PAGES = [
    dict(path="quickstart", title="SKSurrogate in five minutes — SKSurrogate",
         desc="Install SKSurrogate, run a surrogate-guided search over an SVM's hyperparameters, and switch to automatic pipeline design with AML — from the project readme.",
         body=quickstart, name="SKSurrogate in five minutes"),
    dict(path="benchmarks", title="Honest benchmark: surrogate search vs TPE at small budgets — SKSurrogate",
         desc="Equal-budget comparison of SKSurrogate's surrogate-guided search against Optuna-TPE, random search and a no-surrogate ablation on six public datasets, with the raw-data DOI.",
         body=benchmarks, name="Honest benchmark: surrogate search vs TPE at small budgets"),
    dict(path="concepts", title="Concepts — SKSurrogate",
         desc="Surrogate-based optimization, split-conformal prediction intervals and nested cross-validation, defined in the terms the SKSurrogate documentation uses.",
         body=concepts_index, name="Concepts"),
    dict(path="concepts/surrogate-optimization", title="What is surrogate-based optimization? — SKSurrogate",
         desc="Surrogate-based optimization fits an approximate model to expensive evaluations and uses it to choose the next point. The mechanism, when it pays, the measured evidence, and the limitations.",
         body=concept_surrogate, name="What is surrogate-based optimization?"),
    dict(path="concepts/conformal-intervals", title="What is a split-conformal prediction interval? — SKSurrogate",
         desc="How split-conformal intervals are built, what their marginal coverage guarantee assumes, how to check empirical coverage, and when the guarantee fails.",
         body=concept_conformal, name="What is a split-conformal prediction interval?"),
    dict(path="concepts/nested-cross-validation", title="What is nested cross-validation? — SKSurrogate",
         desc="Nested cross-validation separates model selection from the estimate of generalisation error. What the nested_cv_evaluation API does, what it does not, and how to read the result honestly.",
         body=concept_nested, name="What is nested cross-validation?"),
    dict(path="cite", title="Cite SKSurrogate",
         desc="How to cite SKSurrogate: BibTeX and BibLaTeX for the archived software release, plus the DOIs of the benchmark and D-track datasets.",
         body=cite, name="Cite SKSurrogate"),
    dict(path="about", title="About SKSurrogate",
         desc="What SKSurrogate is, what it deliberately is not, how it is developed, who maintains it, and the licence it ships under.",
         body=about, name="About SKSurrogate"),
    dict(path="roadmap", title="Roadmap — SKSurrogate",
         desc="What is next, what is later, and what is explicitly not on the roadmap for SKSurrogate.",
         body=roadmap, name="Roadmap"),
    dict(path="blog", title="Notes and benchmarks — SKSurrogate",
         desc="Long-form write-ups of the studies behind SKSurrogate: equal-budget benchmarks, pre-registered hypotheses, and the results that did not go our way.",
         body=blog, name="Notes and benchmarks", ogtype="website"),
    dict(path="contact", title="Contact — SKSurrogate",
         desc="How to reach the SKSurrogate maintainer: reproducible bug reports, usage questions, security disclosure, and corrections to published results.",
         body=contact, name="Contact"),
    dict(path="faq", title="SKSurrogate FAQ",
         desc="Twelve questions about SKSurrogate, answered from documented behaviour and published benchmark results: scikit-learn fit, speed versus Optuna, categorical features, conformal intervals, deployment.",
         body=faq, name="SKSurrogate FAQ", jsonld=faq_jsonld),
    dict(path="vs-optuna", title="SKSurrogate vs Optuna and skopt",
         desc="How SKSurrogate compares with Optuna and scikit-optimize on the criteria that decide an HPO tool. (Structural shell — copy pending.)",
         body=vs_optuna, name="SKSurrogate vs Optuna and skopt", noindex=True),
]
