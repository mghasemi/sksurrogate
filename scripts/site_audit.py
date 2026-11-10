#!/usr/bin/env python3
"""Machine-check the site against the SEO/AEO acceptance criteria.

Stdlib only, no network except where noted. Run from the repo root:

    python scripts/site_audit.py            # local file checks
    python scripts/site_audit.py --live     # also HEAD-check the published URLs

Prints PASS/FAIL/SKIP per assertion and exits non-zero if anything failed.
The two checks that need a rendering engine — 390 px overflow and the 1440 px
visual diff — are deliberately out of scope here and are verified in a browser
(see README, "Verification").
"""
import argparse
import glob
import hashlib
import html
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = "https://sksurrogate.com"
# Some hosts (Read the Docs, Zenodo behind doi.org) 403 the default python-urllib
# User-Agent while serving the same URL to a browser, so a live check needs one.
UA = "Mozilla/5.0 (X11; Linux x86_64) sksurrogate-site-audit/1.0"
FIXTURE = os.path.join(ROOT, "scripts", "fixtures", "index_text.sha256")

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name, ("  — " + detail) if detail else ""))
    return ok


def skip(name, why):
    results.append((name, None, why))
    print("SKIP  %s  — %s" % (name, why))


def pages():
    out = []
    for p in glob.glob(os.path.join(ROOT, "**", "*.html"), recursive=True):
        if ".git" in p or "/scripts/" in p:
            continue
        out.append(p)
    return sorted(out)


def url_path(p):
    rel = os.path.relpath(p, ROOT)
    if rel == "index.html":
        return "/"
    if rel.endswith("/index.html"):
        return "/" + rel[: -len("index.html")]
    return "/" + rel


def text_of_index():
    src = open(os.path.join(ROOT, "index.html")).read()
    m = re.search(r"<body[^>]*>", src)
    body = src[m.end():] if m else src
    body = re.sub(r"<script.*?</script>|<style.*?</style>", "", body, flags=re.S)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip()
    # the nav/footer are chrome, not marketing copy — hash the sections only
    sections = re.findall(r"<section[^>]*>.*?</section>", body, flags=re.S)
    core = " ".join(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip() for s in sections)
    return text, core


def jsonld_blocks(html_text):
    return [json.loads(m) for m in
            re.findall(r'<script type="application/ld\+json">(.*?)</script>', html_text, re.S)]


def png_size(path):
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also check published URLs")
    args = ap.parse_args()

    all_pages = pages()

    # ---- 1. head hygiene on every page ---------------------------------
    missing = {"viewport": [], "title": [], "description": [], "canonical": [],
               "og:image": [], "og:image:width": [], "og:image:height": [],
               "og:image:alt": []}
    for p in all_pages:
        if url_path(p) == "/404.html":
            continue
        s = open(p).read()
        if not re.search(r'<meta name="viewport" content="width=device-width, initial-scale=1">', s):
            missing["viewport"].append(url_path(p))
        if not re.search(r"<title>[^<]+</title>", s):
            missing["title"].append(url_path(p))
        if '<meta name="description" content="' not in s:
            missing["description"].append(url_path(p))
        if 'rel="canonical"' not in s:
            missing["canonical"].append(url_path(p))
        for key in ("og:image", "og:image:width", "og:image:height", "og:image:alt"):
            if 'property="%s"' % key not in s:
                missing[key].append(url_path(p))
    check("viewport on every page", not missing["viewport"], ", ".join(missing["viewport"]))
    check("title/description on every page",
          not missing["title"] and not missing["description"],
          ", ".join(missing["title"] + missing["description"]))
    check("canonical + og:image set on every page",
          not missing["canonical"] and not missing["og:image"],
          ", ".join(missing["canonical"] + missing["og:image"]))
    check("og:image dimensions/alt on every page",
          not (missing["og:image:width"] or missing["og:image:height"] or missing["og:image:alt"]),
          ", ".join(missing["og:image:width"] + missing["og:image:height"] + missing["og:image:alt"]))

    # ---- 2. stylesheet ------------------------------------------------
    css = open(os.path.join(ROOT, "assets", "site.css")).read()
    media = len(re.findall(r"@media", css))
    check("site.css has >= 3 breakpoints", media >= 3, "%d found" % media)
    body_rules = re.findall(r"(?:^|\n)body\s*\{([^}]*)\}", css)
    fixed = [r for r in body_rules if re.search(r"(?<![-\w])width:\s*\d+px", r)]
    check("no fixed width on body", not fixed, "; ".join(fixed))

    # ---- 3. one host, and every absolute URL resolves ------------------
    hosts = set()
    bad_abs = []
    for p in all_pages:
        s = open(p).read()
        for u in re.findall(r'(?:href|src|content)="(https?://[^"]+)"', s):
            if u.startswith(SITE):
                hosts.add("site")
                rel = u[len(SITE):].split("#")[0]
                target = os.path.join(ROOT, rel.strip("/"))
                if rel in ("", "/"):
                    ok = os.path.exists(os.path.join(ROOT, "index.html"))
                elif rel.endswith("/"):
                    ok = os.path.exists(os.path.join(target, "index.html"))
                else:
                    ok = os.path.exists(target)
                if not ok:
                    bad_abs.append("%s -> %s" % (url_path(p), u))
    check("absolute SITE_URLs resolve to files", not bad_abs, "; ".join(bad_abs[:5]))
    legacy = []
    for p in all_pages:
        if "mghasemi.github.io" in open(p).read():
            legacy.append(url_path(p))
    check("no legacy github.io references after the custom-domain switch",
          not legacy, ", ".join(legacy))

    # ---- 3b. every relative href/src resolves -------------------------
    broken = []
    for p in all_pages:
        base = os.path.dirname(p)
        for href in re.findall(r'(?:href|src)="([^"]+)"', open(p).read()):
            if href.startswith(("http", "mailto:", "#", "data:")):
                continue
            if href.startswith("/"):
                # root-absolute links only appear in 404.html, which GitHub Pages
                # serves at arbitrary depths; they carry the project path prefix
                prefix = urllib.parse.urlparse(SITE).path.rstrip("/") + "/"
                rel = href[len(prefix):] if href.startswith(prefix) else href[1:]
                path = rel.split("#")[0]
                target = os.path.join(ROOT, path)
                ok = (os.path.isfile(os.path.join(target, "index.html"))
                      if path.endswith("/") else os.path.exists(target))
                if not ok:
                    broken.append("%s -> %s" % (url_path(p), href))
                continue
            path = href.split("#")[0].split("?")[0]
            if not path:
                continue
            target = os.path.normpath(os.path.join(base, path))
            ok = (os.path.isfile(os.path.join(target, "index.html"))
                  if path.endswith("/") else os.path.exists(target))
            if not ok:
                broken.append("%s -> %s" % (url_path(p), href))
    check("every relative link resolves to a file", not broken, "; ".join(broken[:6]))

    # ---- 4. JSON-LD ---------------------------------------------------
    invalid = []
    graphs = {}
    for p in all_pages:
        for block in jsonld_blocks(open(p).read()):
            graphs.setdefault(url_path(p), []).append(block)
    for page, blocks in graphs.items():
        for b in blocks:
            if "@context" not in b:
                invalid.append(page)
    check("every JSON-LD block parses with @context", not invalid, ", ".join(set(invalid)))

    home = graphs.get("/", [{}])[0]
    nodes = home.get("@graph", []) if isinstance(home, dict) else []
    types = {n.get("@type") for n in nodes}
    check("index @graph has SoftwareApplication/Person/WebSite",
          {"SoftwareApplication", "Person", "WebSite"} <= types, str(sorted(t for t in types if t)))
    app = next((n for n in nodes if n.get("@type") == "SoftwareApplication"), {})
    person = next((n for n in nodes if n.get("@type") == "Person"), {})
    check("SoftwareApplication.license is MIT",
          app.get("license") == "https://opensource.org/licenses/MIT", str(app.get("license")))
    check("SoftwareApplication.downloadUrl is PyPI",
          app.get("downloadUrl") == "https://pypi.org/project/sksurrogate/", str(app.get("downloadUrl")))
    check("Person.sameAs has >= 3 entries", len(person.get("sameAs", [])) >= 3,
          "%d" % len(person.get("sameAs", [])))

    # programmingLanguage/codeRepository are SoftwareSourceCode properties; the
    # schema.org validator flags them as UNKNOWN_FIELD on SoftwareApplication.
    src = next((n for n in nodes if n.get("@type") == "SoftwareSourceCode"), None)
    check("SoftwareSourceCode node carries the repo and language",
          bool(src) and src.get("codeRepository") and src.get("programmingLanguage"),
          "missing isBasedOn target" if not src else str(src.get("codeRepository")))
    leaked = [k for k in ("codeRepository", "programmingLanguage") if k in app]
    check("SoftwareApplication does not carry SoftwareSourceCode fields",
          not leaked, ", ".join(leaked))

    # Google's Dataset rich result requires name + description; both were
    # missing until a validator flagged "Missing field description".
    datasets = [n for n in nodes if n.get("@type") == "Dataset"]
    incomplete = [d.get("name", "<unnamed>") for d in datasets
                  if not (d.get("description") and d.get("name") and d.get("url"))]
    check("every Dataset node has name, description and url",
          bool(datasets) and not incomplete, ", ".join(incomplete) or "%d datasets ok" % len(datasets))

    # ---- 5. robots + sitemap -----------------------------------------
    robots = open(os.path.join(ROOT, "robots.txt")).read()
    check("robots Sitemap matches SITE_URL", ("Sitemap: %s/sitemap.xml" % SITE) in robots)
    bots = ["GPTBot", "ClaudeBot", "PerplexityBot", "Google-Extended", "CCBot"]
    check("robots allows the answer-engine crawlers",
          all(b in robots for b in bots), ", ".join(b for b in bots if b not in robots))

    sitemap = open(os.path.join(ROOT, "sitemap.xml")).read()
    locs = re.findall(r"<loc>([^<]+)</loc>", sitemap)
    check("sitemap uses one host", all(l.startswith(SITE) for l in locs))
    missing_lastmod = re.findall(r"<loc>[^<]+</loc>(?!<lastmod>)", sitemap)
    check("every sitemap url has lastmod", not missing_lastmod)
    listed = {l[len(SITE):] for l in locs}
    noindex_pages = set()
    indexable = set()
    for p in all_pages:
        s = open(p).read()
        if 'content="noindex"' in s:
            noindex_pages.add(url_path(p))
        else:
            indexable.add(url_path(p))
    missing_from_sitemap = indexable - listed
    check("every indexable page is in the sitemap", not missing_from_sitemap,
          ", ".join(sorted(missing_from_sitemap)))
    check("noindex pages are excluded from the sitemap",
          not (noindex_pages & listed), ", ".join(sorted(noindex_pages & listed)))

    # ---- 6. llms.txt --------------------------------------------------
    llms_path = os.path.join(ROOT, "llms.txt")
    llms = open(llms_path).read() if os.path.exists(llms_path) else ""
    lines = [l for l in llms.splitlines() if l.strip()]
    check("llms.txt exists with <= 40 lines", 0 < len(lines) <= 40, "%d lines" % len(lines))
    check("llms-full.txt exists and is non-trivial",
          os.path.exists(os.path.join(ROOT, "llms-full.txt"))
          and os.path.getsize(os.path.join(ROOT, "llms-full.txt")) > 4000,
          "%d bytes" % os.path.getsize(os.path.join(ROOT, "llms-full.txt")))
    llms_urls = re.findall(r"https?://[^\s\)]+", llms)

    # ---- 7. images ----------------------------------------------------
    img_tags = re.findall(r"<img\b[^>]*>", open(os.path.join(ROOT, "index.html")).read())
    no_dims = [t for t in img_tags if 'width="' not in t or 'height="' not in t]
    no_alt = [t for t in img_tags if 'alt="' not in t]
    check("every <img> has width, height and alt", not no_dims and not no_alt,
          "%d without dims, %d without alt" % (len(no_dims), len(no_alt)))
    lazy_ok = sum(1 for t in img_tags if 'loading="lazy"' in t)
    check("below-the-fold images are lazy", lazy_ok >= len(img_tags) - 1,
          "%d of %d lazy" % (lazy_ok, len(img_tags)))
    webps = sorted(glob.glob(os.path.join(ROOT, "assets", "*.webp")))
    total = sum(os.path.getsize(w) for w in webps)
    check("image payload < 350 KB", total < 350 * 1024, "%.0f KB over %d webp files"
          % (total / 1024, len(webps)))

    # ---- 8. og card ---------------------------------------------------
    card = os.path.join(ROOT, "assets", "og-card.png")
    if os.path.exists(card):
        w, h = png_size(card)
        check("og-card.png is 1200x630", (w, h) == (1200, 630), "%dx%d" % (w, h))
    else:
        check("og-card.png exists", False)

    # ---- 9. no third-party font requests ------------------------------
    font_hits = []
    for p in all_pages:
        s = open(p).read()
        if "fonts.googleapis.com" in s or "fonts.gstatic.com" in s:
            font_hits.append(url_path(p))
    check("no Google Fonts requests", not font_hits, ", ".join(font_hits))
    fc = len(re.findall(r"@font-face", css))
    check("site.css declares @font-face for self-hosted fonts", fc >= 3, "%d faces" % fc)

    # ---- 10. FAQ JSON-LD must equal the visible answers ---------------
    faq_page = os.path.join(ROOT, "faq", "index.html")
    faq_blocks = [b for b in jsonld_blocks(open(faq_page).read()) if b.get("@type") == "FAQPage"]
    if not faq_blocks:
        skip("FAQPage answers match visible text",
             "faq/ ships as a noindex shell until the advisor copy lands")
    else:
        visible = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ",
                          open(faq_page).read().split("<body", 1)[1])).strip()
        mismatched = [q["name"] for q in faq_blocks[0]["mainEntity"]
                      if q["acceptedAnswer"]["text"] not in visible]
        check("FAQPage answers match visible text", not mismatched, ", ".join(mismatched))

    # ---- 11. reachability from the homepage ---------------------------
    # Breadth-first crawl over relative links: a page two hops down a hub
    # (/concepts/ -> /concepts/x/) is fine; an unreachable page is not.
    from collections import deque
    seen, queue = {"/"}, deque(["/"])
    while queue:
        cur = queue.popleft()
        src = os.path.join(ROOT, cur.strip("/"), "index.html") if cur != "/" else os.path.join(ROOT, "index.html")
        if not os.path.exists(src):
            continue
        base = os.path.dirname(src)
        for href in re.findall(r'href="([^"]+)"', open(src).read()):
            if href.startswith(("http", "mailto:", "#", "data:")):
                continue
            if href.startswith("/"):
                continue  # root-absolute links live only in 404.html
            target = os.path.normpath(os.path.join(base, href.split("#")[0]))
            if not os.path.exists(os.path.join(target, "index.html")):
                continue
            page = url_path(os.path.join(target, "index.html"))
            if page not in seen:
                seen.add(page)
                queue.append(page)
    unlinked = indexable - seen
    check("no orphaned indexable page (crawl from index)", not unlinked,
          ", ".join(sorted(unlinked)))

    # ---- 12. copy freeze on the homepage -----------------------------
    text, core = text_of_index()
    digest = hashlib.sha256(core.encode()).hexdigest()
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    if not os.path.exists(FIXTURE):
        open(FIXTURE, "w").write(digest + "\n")
        skip("homepage section copy unchanged", "fixture created at " + FIXTURE)
    else:
        expected = open(FIXTURE).read().strip()
        check("homepage section copy unchanged", digest == expected,
              "" if digest == expected else "sections differ from the frozen hash")

    # ---- 13. live URLs ------------------------------------------------
    if args.live:
        urls = [SITE + p for p in sorted(indexable)] + [SITE + "/llms.txt",
                                                        SITE + "/robots.txt",
                                                        SITE + "/sitemap.xml",
                                                        SITE + "/assets/og-card.png"]
        bad = []
        for u in urls:
            try:
                req = urllib.request.Request(u, method="GET", headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=25) as r:
                    if r.status != 200:
                        bad.append("%s %s" % (r.status, u))
            except urllib.error.HTTPError as e:
                bad.append("%s %s" % (e.code, u))
            except Exception as e:  # noqa: BLE001 - report, do not crash the audit
                bad.append("%s %s" % (type(e).__name__, u))
        check("live URLs return 200", not bad, "; ".join(bad[:6]))
        bad_llms = []
        for u in llms_urls:
            try:
                req = urllib.request.Request(u, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=25) as r:
                    if r.status != 200:
                        bad_llms.append(u)
            except Exception:  # noqa: BLE001
                bad_llms.append(u)
        check("llms.txt links resolve", not bad_llms, ", ".join(bad_llms[:5]))
    else:
        skip("live URLs return 200", "run with --live after deploy")
        skip("llms.txt links resolve", "run with --live after deploy")

    failed = [n for n, ok, _ in results if ok is False]
    print("\n%d checks, %d failed, %d skipped"
          % (len(results), len(failed), sum(1 for _, ok, _ in results if ok is None)))
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
