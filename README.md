# sksurrogate-site

Static site for [SKSurrogate](https://github.com/mghasemi/sksurrogate), served from the `gh-pages`
branch at the branch root by GitHub Pages.

- **Canonical origin:** `https://sksurrogate.com` (the single `SITE` constant in
  `scripts/build_site.py` and `scripts/site_audit.py`). `sksurrogate.com` does not resolve yet; when
  it does, change that constant, add a root `CNAME`, and re-run the build + audit.
- **No build step at deploy time.** Pages are committed HTML. `scripts/build_site.py` is an
  authoring tool that emits the shared head/nav/footer so the pages cannot drift apart.

## Page inventory

| Path | Purpose | Generated? |
|---|---|---|
| `index.html` | Product homepage (`#product` … `#qs`). Copy is frozen — do not edit. | no |
| `quickstart/` | Five-minute path: pip install, `SurrogateRandomCV`, `AML`, links to the guides | yes |
| `benchmarks/` | Citable benchmark page: arms table + B=30 results + caveat + DOIs | yes |
| `blog/` | Post index | yes |
| `blog/honest-benchmark/` | Long-form post; body lives in `scripts/sources/honest_benchmark_body.html` | body verbatim |
| `cite/` | BibTeX / BibLaTeX, dataset DOIs, in-text example | yes |
| `about/` | What it is / what it is not / maintainer / licence (no affiliations, by owner request) | yes |
| `roadmap/` | Next / later / not on the roadmap | yes |
| `contact/` | Email, issue tracker, security disclosure, site privacy | yes |
| `faq/` | **Shell, `noindex`** — 8–12 question headings and answers pending from the advisor | yes |
| `vs-optuna/` | **Shell, `noindex`** — comparison copy pending; reuses the `#compare` rows | yes |
| `404.html` | Not-found page (`noindex`, absolute paths — see note below) | no |

## Editing rules

1. **Copy freeze on `index.html`.** `scripts/site_audit.py` hashes the text of the nine `<section>`
   blocks against `scripts/fixtures/index_text.sha256`. If a change to those sections is intended,
   edit, then delete the fixture and re-run the audit to re-freeze it.
2. **Traceable numbers only.** Benchmark figures are transcribed verbatim from
   `docs/blog-honest-benchmark.md` / `benchmarks/DESIGN.md` in the software repo. Verify with a
   table diff, not by eye.
3. **Relative links everywhere** (so the site works at the domain root *and* under
   `/sksurrogate/`). Canonical/OG URLs are absolute on `SITE`. The one exception is `404.html`,
   which GitHub Pages serves at arbitrary depths and therefore needs absolute paths.
4. **BibTeX blocks are hand-wrapped inside `{...}`** — a line break inside braces is whitespace to
   BibTeX. `@software` needs BibLaTeX, hence the `@misc` twin. Validate with `biber --tool` and
   `bibtex` before shipping.
5. **Structured data must match visible content.** Homepage: `SoftwareApplication` + `Person` +
   `WebSite` + two `Dataset` nodes. Post: `BlogPosting`. Other pages: `WebPage`. Update
   `softwareVersion` when PyPI actually serves that version — it is **0.2.0** while the footer strip
   says v0.3.0 (tag pushed, no GitHub Release yet; owner decision pending).
6. **Grid children need `min-width:0`** (see the `.hero>*, .rail>*` rule). Without it a wide `<pre>`
   or table expands the whole grid past the viewport — that is what broke 390 px before.

## Scripts

```bash
python scripts/build_site.py        # regenerate the nine pages + sitemap.xml
python scripts/make_llms_full.py    # regenerate llms-full.txt from index.html
python scripts/make_og_card.py      # regenerate assets/og-card.png (1200x630)
python scripts/site_audit.py        # 33 checks, PASS/FAIL, non-zero exit on failure
python scripts/site_audit.py --live # + HEAD-check the published URLs
```

`build_site.py` and `site_pages.py` hold the content; edit copy there, never in the emitted HTML.

## Verification

The audit covers: viewport/title/description/canonical/OG on every page; one canonical host; every
relative link resolving; JSON-LD parsing and the required `@graph` nodes; robots + sitemap
consistency (every indexable page listed, every `noindex` page excluded, `<lastmod>` present); `llms.txt`
≤ 40 lines; `<img>` width/height/alt and lazy-loading; image payload < 350 KB; OG card dimensions;
no Google Fonts requests; orphan detection; and the homepage copy hash.

Two checks need a rendering engine and are done in a browser instead:

- **390 px overflow:** `document.documentElement.scrollWidth === clientWidth` on all 12 pages
  (verified at 390 and 360 px).
- **1440 px fidelity:** section heights and pixels compared against a pre-change render. Sections
  are identical; only `footer` grew (+63 px) because the brief adds Blog/FAQ/Cite/About/Roadmap/
  Quickstart links to it.

## Known gaps (owner/advisor input needed)

- `faq/` and `vs-optuna/` are `noindex` shells: question headings and comparison copy are pending.
- Read the Docs currently publishes only `intro`, `surrogate`, `dataproc`, `mltrace`, `controlplane`,
  `ui-manual`, `monitoring`, `benchmarks`, `blog-honest-benchmark`; the readme links to
  `walkthrough.html`, `experiment-design.html` and `research-projects.html`, which 404 on the
  published build. The site links only pages that resolve — fix RTD separately.
- Version strip (v0.3.0) vs PyPI (0.2.0): untouched by design.

## Update procedure

1. `python scripts/build_site.py && python scripts/make_llms_full.py`
2. `python scripts/site_audit.py` — must be 0 failed
3. Commit, push `gh-pages`, wait for Pages `status: built`, then `--live` audit.
