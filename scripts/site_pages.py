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
          compose profile when you want it.</p>

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
        </ul>
      </div>''' % (crumb(rel, "quickstart"), L(rel, "/benchmarks/"))


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
          ($d \le 3$); the wall-clock arm uses a different currency (seconds, hardware-dependent)
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
def faq(rel):
    return '''%s
      <h1>SKSurrogate FAQ</h1>
      <p class="lede">Answers to the questions the documentation keeps getting asked.</p>
      <div class="prose">
        %s
      </div>''' % (crumb(rel, "faq"), TODO)


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
            <tr><td><b>Statistical evaluation, not a single score.</b> Repeated nested cross-validation, paired comparison protocol checks, and split-conformal intervals with coverage diagnostics — the “90 %% interval” is a number you can check, not an adjective.</td><td><a href="https://sksurrogate.readthedocs.io/en/latest/controlplane.html" target="_blank" rel="noopener">docs ↗</a></td></tr>
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
    # path, title, description, body, og:type, jsonld name, noindex
    ("quickstart", "SKSurrogate in five minutes — SKSurrogate",
     "Install SKSurrogate, run a surrogate-guided search over an SVM's hyperparameters, and switch to automatic pipeline design with AML — from the project readme.",
     quickstart, "article", "SKSurrogate in five minutes", False),
    ("benchmarks", "Honest benchmark: surrogate search vs TPE at small budgets — SKSurrogate",
     "Equal-budget comparison of SKSurrogate's surrogate-guided search against Optuna-TPE, random search and a no-surrogate ablation on six public datasets, with the raw-data DOI.",
     benchmarks, "article", "Honest benchmark: surrogate search vs TPE at small budgets", False),
    ("cite", "Cite SKSurrogate",
     "How to cite SKSurrogate: BibTeX and BibLaTeX for the archived software release, plus the DOIs of the benchmark and D-track datasets.",
     cite, "article", "Cite SKSurrogate", False),
    ("about", "About SKSurrogate",
     "What SKSurrogate is, what it deliberately is not, how it is developed, who maintains it, and the licence it ships under.",
     about, "article", "About SKSurrogate", False),
    ("roadmap", "Roadmap — SKSurrogate",
     "What is next, what is later, and what is explicitly not on the roadmap for SKSurrogate.",
     roadmap, "article", "Roadmap", False),
    ("blog", "Notes and benchmarks — SKSurrogate",
     "Long-form write-ups of the studies behind SKSurrogate: equal-budget benchmarks, pre-registered hypotheses, and the results that did not go our way.",
     blog, "website", "Notes and benchmarks", False),
    ("contact", "Contact — SKSurrogate",
     "How to reach the SKSurrogate maintainer: reproducible bug reports, usage questions, security disclosure, and corrections to published results.",
     contact, "article", "Contact", False),
    ("faq", "SKSurrogate FAQ",
     "Frequently asked questions about SKSurrogate. (Structural shell — copy pending.)",
     faq, "article", "SKSurrogate FAQ", True),
    ("vs-optuna", "SKSurrogate vs Optuna and skopt",
     "How SKSurrogate compares with Optuna and scikit-optimize on the criteria that decide an HPO tool. (Structural shell — copy pending.)",
     vs_optuna, "article", "SKSurrogate vs Optuna and skopt", True),
]
