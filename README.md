# sksurrogate-site

Static marketing site for [SKSurrogate](https://github.com/mghasemi/sksurrogate), served from the
`gh-pages` branch at the branch root and intended for the custom domain `https://sksurrogate.com/`.

No framework, no build step: every page is hand-written HTML plus `assets/page.css`.

## Page inventory

| Path | Purpose |
|---|---|
| `index.html` | Product homepage (sections `#product` … `#qs`). Styles are inline here; do not move them. |
| `about.html` | What the project is, how it is developed, maintainer, licence. No affiliations listed by owner request. |
| `contact.html` | Email, GitHub Issues, security disclosure, what the project does **not** offer. |
| `cite.html` | BibTeX / BibLaTeX for the archived release, dataset DOIs, in-text example. |
| `blog/index.html` | Post index. One entry per `<li>` in `.postlist`. |
| `blog/honest-benchmark.html` | Long-form benchmark post; the numbers mirror `docs/blog-honest-benchmark.md` in the software repo. |
| `404.html` | Not-found page (self-contained, `noindex`). |
| `assets/page.css` | Shared styles for all pages except `index.html` and `404.html`. |
| `sitemap.xml`, `robots.txt` | Crawl hygiene. **Every new page must be added to `sitemap.xml`.** |

## Editing rules

1. **Every number must be traceable.** Benchmark figures come from the software repo's
   `docs/blog-honest-benchmark.md` / `benchmarks/DESIGN.md` and are transcribed verbatim. Verify a
   transcription with a table diff against the markdown source rather than re-reading by eye.
2. **Relative links.** Internal links are relative (`../about.html`, `blog/index.html`) so the site
   also works under the GitHub project URL `mghasemi.github.io/sksurrogate/` before the domain is
   live. Canonical tags and `og:url` stay absolute on `sksurrogate.com`.
3. **BibTeX blocks are hand-wrapped** inside the `{...}` field values. A line break inside braces is
   whitespace to BibTeX, so wrapped entries stay valid; long lines that overflow the code block are
   not.
4. **`@software` needs BibLaTeX.** Plain BibTeX styles do not define it, which is why `cite.html`
   also ships an `@misc` twin. Validate entries with `biber --tool` and `bibtex` before shipping.
5. **Structured data must match visible content.** Homepage carries `SoftwareApplication` +
   `Person` + `WebSite`; the post carries `BlogPosting`; about/contact carry `AboutPage` /
   `ContactPage`. Update `softwareVersion` in the homepage JSON-LD when the footer version changes.
6. **The 1440 px frame is intentional** on `index.html` (design mockup fidelity). Secondary pages use
   a fluid `max-width:1080px` column.

## What this site deliberately does not have

No pricing page, no "request a demo" funnel, no careers page, and no second copy of the
documentation (Read the Docs is the single source of truth). SKSurrogate is MIT, pip-installable and
self-service; a gated demo form would contradict the honesty positioning the copy is built on.

## Verification

```bash
env PYTHONPATH= python3 -m http.server 8099   # then open http://127.0.0.1:8099/
```

- Every internal `href`/`src` resolves (check relative to the page's own directory).
- Every `<script type="application/ld+json">` block parses as JSON.
- External URLs return 200 (`doi.org` links need a plain curl UA — Zenodo returns 403 to some
  browser UAs, which is bot protection, not a dead link).
- `sitemap.xml` lists every shipped page.

## Launch gate

`canonical`, `og:url` and `sitemap.xml` all point at `https://sksurrogate.com/`. Until the DNS flip
from the GitHub Pages project URL happens, search engines see a canonical target that does not
resolve — do not promote the project URL publicly in that window.
