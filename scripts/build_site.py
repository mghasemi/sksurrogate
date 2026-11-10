#!/usr/bin/env python3
"""Assemble the Phase-3 pages of the SKSurrogate site.

Why a generator when the site is "no build step": the secondary pages share one
<head>, one nav and one footer, and hand-copying those across files is how sites
drift. The output is plain committed HTML — nothing runs at deploy time.
`index.html` and the long-form blog post keep their hand-authored markup.

Copy rules (HANDOFF-website-seo.md):
  * only reproduce approved text — readme.rst, docs/blog-honest-benchmark.md,
    CITATION.cff, and the existing homepage sections;
  * pages marked COPY_PENDING ship as shells with robots=noindex and stay out of
    sitemap.xml until the advisor's copy lands.

Run:  python scripts/build_site.py
"""
import os
import re

SITE = "https://mghasemi.github.io/sksurrogate"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NAV = [
    ("/", "Product", False),
    ("/benchmarks/", "Benchmarks", False),
    ("/quickstart/", "Quickstart", False),
    ("/blog/", "Blog", False),
    ("/faq/", "FAQ", False),
    ("/cite/", "Cite", False),
    ("/about/", "About", False),
    ("https://sksurrogate.readthedocs.io/en/latest/intro.html", "Docs \u2197", True),
    ("https://github.com/mghasemi/sksurrogate", "GitHub \u2197", True),
]

FOOTER = [
    ("Project", [
        ("https://github.com/mghasemi/sksurrogate", "GitHub \u2197", True),
        ("https://pypi.org/project/sksurrogate/", "PyPI \u2197", True),
        ("https://sksurrogate.readthedocs.io/en/latest/", "Documentation \u2197", True),
        ("https://github.com/mghasemi/sksurrogate/blob/master/CHANGELOG.rst", "Changelog \u2197", True),
    ]),
    ("Site", [
        ("/quickstart/", "Quickstart", False),
        ("/benchmarks/", "Benchmarks", False),
        ("/blog/", "Blog", False),
        ("/faq/", "FAQ", False),
        ("/cite/", "Citation", False),
        ("/about/", "About", False),
        ("/roadmap/", "Roadmap", False),
        ("/contact/", "Contact", False),
    ]),
    ("Maintainer", [
        ("https://mghasemi.net", "mghasemi.net \u2197", True),
        ("mailto:mehdi.ghasemi@gmail.com", "mehdi.ghasemi@gmail.com", False),
        ("https://orcid.org/0000-0002-9551-1780", "ORCID \u2197", True),
    ]),
]

HEADER_TPL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
{robots}<link rel="canonical" href="{site}{path}">
<meta property="og:site_name" content="SKSurrogate">
<meta property="og:locale" content="en_US">
<meta property="og:type" content="{ogtype}">
<meta property="og:url" content="{site}{path}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:image" content="{site}/assets/og-card.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="SKSurrogate — from raw dataset to deployed, drift-monitored model. MIT, pip install sksurrogate.">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="{site}/assets/og-card.png">
<link rel="icon" href="{rel}assets/mark.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="{rel}icons/icon-180.png">
<link rel="stylesheet" href="{rel}assets/site.css">
{jsonld}</head>
<body class="sub">

<header class="topbar">
  <a class="brand" href="{rel}"><img src="{rel}assets/mark.svg" alt="">SKSurrogate</a>
  <nav>
{nav}
  </nav>
</header>

<main class="wrap">
{body}
</main>

<footer class="subfooter">
{footer}
</footer>
<div class="subfootline">
  <span>© 2019–2026 Mehdi Ghasemi · MIT licence</span>
  <span>sksurrogate v0.3.0 · scikit-learn-native</span>
</div>
{scripts}</body>
</html>
"""


def nav_html(rel):
    out = []
    for href, label, ext in NAV:
        if ext:
            out.append('    <a href="%s" target="_blank" rel="noopener">%s</a>' % (href, label))
        elif href == "/":
            out.append('    <a href="%s">%s</a>' % (rel or "./", label))
        else:
            out.append('    <a href="%s%s">%s</a>' % (rel, href.lstrip("/"), label))
    return "\n".join(out)


def footer_html(rel):
    blocks = []
    for title, links in FOOTER:
        items = []
        for href, label, ext in links:
            if href.startswith(("http", "mailto")):
                items.append('      <li><a href="%s"%s>%s</a></li>'
                             % (href, ' target="_blank" rel="noopener"' if ext else "", label))
            else:
                items.append('      <li><a href="%s%s">%s</a></li>' % (rel, href.lstrip("/"), label))
        blocks.append("  <div>\n    <h5>%s</h5>\n    <ul>\n%s\n    </ul>\n  </div>"
                      % (title, "\n".join(items)))
    return "\n".join(blocks)


def rel_for(path):
    """'../' per directory level: /quickstart/ lives at quickstart/index.html."""
    depth = len([p for p in path.strip("/").split("/") if p])
    return "../" * depth


def write_page(path, title, desc, body_fn, ogtype="article", jsonld="",
               noindex=False, scripts=""):
    rel = rel_for(path)
    html = HEADER_TPL.format(
        title=title, desc=desc, site=SITE, path=path, rel=rel, ogtype=ogtype,
        robots=('<meta name="robots" content="noindex">\n' if noindex else ""),
        jsonld=jsonld, nav=nav_html(rel), footer=footer_html(rel),
        body=body_fn(rel), scripts=scripts)
    out_dir = os.path.join(ROOT, path.strip("/"))
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, "index.html")
    with open(target, "w") as fh:
        fh.write(html)
    return target


def webpage_jsonld(path, name):
    return ('<script type="application/ld+json">\n'
            '{"@context":"https://schema.org","@type":"WebPage","@id":"%s%s",'
            '"url":"%s%s","name":"%s","isPartOf":{"@id":"%s/#website"}}\n'
            '</script>\n') % (SITE, path, SITE, path, name, SITE)


TODO = ('<p class="todo">Copy pending — this page is a structural shell. Nothing here '
        'is approved text, and the page is excluded from search indexing '
        '(robots: noindex) until the copy lands.</p>')


def crumb(rel, *parts):
    html = '<div class="crumb"><a href="%s">sksurrogate</a>' % (rel or "./")
    for p in parts:
        html += "<span>/</span>" + p
    return html + "</div>"


# ------------------------------------------------------------------ driver
def main():
    from site_pages import PAGES  # same directory as this script

    written = []
    for path, title, desc, body_fn, ogtype, name, noindex in PAGES:
        url_path = "/%s/" % path
        target = write_page(url_path, title, desc, body_fn, ogtype=ogtype,
                            jsonld=webpage_jsonld(url_path, name) if not noindex else "",
                            noindex=noindex)
        written.append((url_path, target, noindex))
        print("wrote", os.path.relpath(target, ROOT))

    # the long-form post keeps its hand-authored body (a fragment under
    # scripts/sources/); only the wrapper is rebuilt, and internal links are
    # normalised to the target depth so the build is idempotent.
    post_src = os.path.join(ROOT, "scripts", "sources", "honest_benchmark_body.html")
    if os.path.exists(post_src):
        raw = open(post_src).read().strip()

        def post_body(rel, raw=raw):
            body = raw.replace('href="../cite.html"', 'href="%scite/"' % rel)
            body = re.sub(r'href="(?:\.\./)*cite/"', 'href="%scite/"' % rel, body)
            body = re.sub(r'href="(?:\.\./)*blog/"', 'href="%sblog/"' % rel, body)
            body = re.sub(r'href="(?:\.\./)*#', 'href="%s#' % rel, body)
            return body

        title = "An Honest Benchmark of My Own AutoML Package — SKSurrogate"
        desc = ("At equal evaluation budgets on six public datasets, surrogate-guided search does "
                "not beat Optuna-TPE on accuracy. What is real: the response surface matters more "
                "than the search mechanism.")
        jsonld = ('<script type="application/ld+json">\n'
                  '{"@context":"https://schema.org","@type":"BlogPosting",'
                  '"mainEntityOfPage":"%s/blog/honest-benchmark/",'
                  '"headline":"An Honest Benchmark of My Own AutoML Package",'
                  '"description":"At equal evaluation budgets on six public datasets, '
                  'surrogate-guided search does not beat Optuna-TPE on accuracy.",'
                  '"datePublished":"2026-10-26","author":{"@type":"Person",'
                  '"name":"Mehdi Ghasemi","url":"https://mghasemi.net"}}\n</script>\n' % SITE)
        target = write_page("/blog/honest-benchmark/", title, desc, post_body,
                            ogtype="article", jsonld=jsonld)
        written.append(("/blog/honest-benchmark/", target, False))
        print("wrote", os.path.relpath(target, ROOT))

    # sitemap: only indexable pages, with lastmod from the repo's HEAD date
    import subprocess
    try:
        lastmod = subprocess.check_output(["git", "log", "-1", "--format=%cs"],
                                          cwd=ROOT).decode().strip()
    except Exception:
        lastmod = "2026-11-10"
    entries = [("/", lastmod)] + [(p, lastmod) for p, _, noindex in written if not noindex]
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for path, lm in entries:
        lines.append("  <url><loc>%s%s</loc><lastmod>%s</lastmod></url>" % (SITE, path, lm))
    lines.append("</urlset>")
    open(os.path.join(ROOT, "sitemap.xml"), "w").write("\n".join(lines) + "\n")
    print("wrote sitemap.xml (%d urls, %d noindex pages excluded)"
          % (len(entries), sum(1 for _, _, n in written if n)))


if __name__ == "__main__":
    main()
