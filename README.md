# sksurrogate-site

Static site for [SKSurrogate](https://github.com/mghasemi/sksurrogate), served from the `gh-pages`
branch at the branch root by GitHub Pages.

- **Canonical origin:** `https://sksurrogate.com` — live since 2026-11-10. The host appears once as
  `SITE` in `scripts/build_site.py` and `scripts/site_audit.py`; `CNAME` at the branch root pins the
  GitHub Pages custom domain. `www.sksurrogate.com` and the old
  `mghasemi.github.io/sksurrogate/` both 301 to the apex; the certificate is Let's Encrypt, covers
  apex + www, renews automatically, and HTTPS is enforced in repo settings.
  To move hosts again: change `SITE`, swap the `CNAME` contents, re-run the build, audit, push.
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
| `concepts/` | Concept hub: surrogate optimization, conformal intervals, nested CV | yes |
| `concepts/surrogate-optimization/` | What surrogate-based optimization is, when it pays, the measured evidence | yes |
| `concepts/conformal-intervals/` | Split-conformal procedure, marginal-coverage assumptions, diagnostics | yes |
| `concepts/nested-cross-validation/` | Selection vs estimation, what the API does and does not do | yes |
| `faq/` | 12 answered questions + `FAQPage` JSON-LD generated from the same list the page renders | yes |
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

- `vs-optuna/` is a `noindex` shell: the head-to-head copy is still pending.
- Concept-page copy was drafted from `readme.rst`, `docs/surrogate.rst`, `docs/mltrace.rst` and the
  benchmark post, then reviewed by the owner before it shipped; new claims need the same treatment.
- Read the Docs currently publishes only `intro`, `surrogate`, `dataproc`, `mltrace`, `controlplane`,
  `ui-manual`, `monitoring`, `benchmarks`, `blog-honest-benchmark`; the readme links to
  `walkthrough.html`, `experiment-design.html` and `research-projects.html`, which 404 on the
  published build. The site links only pages that resolve — fix RTD separately.
- Version strip (v0.3.0) vs PyPI (0.2.0): untouched by design.

## Update procedure

1. `python scripts/build_site.py && python scripts/make_llms_full.py`
2. `python scripts/site_audit.py` — must be 0 failed
3. Commit, push `gh-pages`, wait for Pages `status: built`, then `--live` audit.

## Edge / DNS (Cloudflare)

The zone is `sksurrogate.com` (id `5d88f6a942534daea5cf0498e18ba771`), plan Free, serving through
Cloudflare with GitHub Pages as origin. Current state, set 2026-11-10:

| Setting | Value |
|---|---|
| DNS | `<CNAME @ → mghasemi.github.io>` + `<CNAME www → mghasemi.github.io>`, both **proxied** |
| SSL/TLS mode | **Full (strict)** — Cloudflare validates GitHub Pages' Let's Encrypt cert |
| Min TLS | 1.2 (TLS 1.0/1.1 refused; TLS 1.3 on) |
| Always Use HTTPS | on |
| HSTS | `max-age=15552000; includeSubDomains` + `nosniff`; **preload off** |
| Redirect rule | `http_request_dynamic_redirect` entrypoint `682058bb744f44a6b71d85a1167a77dc`: `http.host eq "www.sksurrogate.com"` → 301 `https://sksurrogate.com<path>` (query preserved) |
| HTML at edge | `cf-cache-status: DYNAMIC` (not cached — deploys are visible immediately); assets `HIT` |

Two consequences worth knowing:

- **HSTS is sticky.** Browsers keep the 6-month policy after the header disappears; do not plan a
  return to plain HTTP or to a different host on this zone without accounting for that.
- **`www` no longer reaches GitHub.** Before the proxy, Pages answered `www` with its own 301; behind
  the proxy the request never leaves the edge, hence the redirect rule above. Deleting that rule
  reintroduces a 404 on `www`.

## Post-launch checklist (outside this repo)

Verified by the 2026-11-10 checks: apex + www serve over HTTPS, HTTP→HTTPS 301, `www`→apex 301 with
path and query preserved, legacy `mghasemi.github.io/sksurrogate/`→apex 301, all 11 pages 200,
sitemap/robots/llms.txt live on the apex, HSTS + `X-Content-Type-Options` present.

Done:
- GitHub "Website" field → `https://sksurrogate.com` (was Read the Docs).
- HSTS + strict TLS at the edge (see above).
- Google Search Console (**Domain** property) verified by DNS TXT at `@` — the record is live and
  unproxied; Bing imported it from Search Console. Sitemap submitted to both.

Still owner-side, because they need accounts this repo does not hold:

1. **Index monitoring** — first indexing of a new domain takes days to weeks. Watch Search Console
   → Pages and Bing's Site Explorer; expect "Discovered – currently not indexed" before "Indexed".
2. **PyPI** — `setup.py`'s `url` and `project_urls` point at GitHub; they only change with a release.
3. **External mentions** — anything written before the domain existed still points at the github.io
   URL; it now 301s, so links keep working, but the canonical reference should be updated when those
   pages are next touched.
4. **HSTS preload** — deliberately off. Only submit to `hstspreload.org` once the domain has run
   HSTS for months and `includeSubDomains` is known to be safe for every future subdomain.
