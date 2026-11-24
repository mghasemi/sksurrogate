#!/usr/bin/env python3
"""Archive a GitHub release to Zenodo, keeping the existing concept DOI.

Zenodo's GitHub webhook mints a *new* concept record and cannot attach to a
manually uploaded one, so enabling it would fork SKSurrogate's citation identity
(the existing chain is concept 10.5281/zenodo.23325032 with 0.1.1 and 0.2.0).
This script does the same job against the same concept instead, using the
legacy deposit API: newversion from the newest published record, replace the
copied file with the tagged source tarball, publish, verify.

Environment (set by the workflow):
    ZENODO_TOKEN        personal access token, scope deposit:write
    CONCEPT_RECID       concept record id, e.g. 23325032
    RELEASE_TAG         tag being archived, e.g. v0.8.1
    REPO                owner/name, e.g. mghasemi/sksurrogate

Local use:
    python3 zenodo_archive.py --print-metadata --tag v0.8.1   # no network, no token
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://zenodo.org/api"

TITLE = ("SKSurrogate: a Python toolkit for surrogate-based optimization of "
         "expensive functions")

DESCRIPTION = (
    "SKSurrogate is a scikit-learn-native suite for surrogate-based optimization of "
    "expensive objectives: hyperparameter search and automated pipeline design with an "
    "evolutionary outer loop and a surrogate response surface (a stock Gaussian process by "
    "default; any scikit-learn regressor is accepted). Around the search sit dataset identity "
    "and drift tracking, sensitivity analysis (Sobol, Morris, delta), conformal intervals, "
    "nested cross-validation, paired model comparison, portable model bundles with a serving "
    "pin, and a FastAPI control plane with a web UI and an MCP server for agents. Equal-budget "
    "benchmarks against Optuna-TPE, random search, FLAML and AutoGluon are published with raw "
    "records (DOI 10.5281/zenodo.23629024). MIT licensed."
)

KEYWORDS = ["surrogate optimization", "hyperparameter optimization", "automl",
            "scikit-learn", "gaussian process", "bayesian optimization", "python"]


def metadata(version, pub_date, repo):
    return {"metadata": {
        "title": TITLE,
        "upload_type": "software",
        "publication_date": pub_date,
        "description": DESCRIPTION,
        "keywords": KEYWORDS,
        "license": "mit",
        "version": version,
        "creators": [{"name": "Ghasemi, Mehdi"}],
        "access_right": "open",
        "related_identifiers": [
            {"identifier": "https://github.com/" + repo, "relation": "isDescribedBy"},
            {"identifier": "10.5281/zenodo.23629024", "relation": "references"},
        ],
    }}


def call(method, url, token, payload=None, raw=None,
         content_type="application/json", allow_404=False):
    data = json.dumps(payload).encode() if payload is not None else raw
    req = urllib.request.Request(url, data=data, method=method)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    if data is not None:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            body = r.read()
            return r.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as e:
        if allow_404 and e.code == 404:
            return 404, None
        print("HTTP %s on %s %s: %s" % (e.code, method, url, e.read()[:400]))
        raise SystemExit(1)


def public_get(url):
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def version_of_tag(tag):
    return tag[1:] if tag.startswith("v") else tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=os.environ.get("RELEASE_TAG", ""))
    ap.add_argument("--repo", default=os.environ.get("REPO", "mghasemi/sksurrogate"))
    ap.add_argument("--concept", default=os.environ.get("CONCEPT_RECID", "23325032"))
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d", time.gmtime()))
    ap.add_argument("--print-metadata", action="store_true")
    args = ap.parse_args()

    if not args.tag:
        raise SystemExit("no tag: pass --tag or set RELEASE_TAG")
    version = version_of_tag(args.tag)

    if args.print_metadata:
        print(json.dumps(metadata(version, args.date, args.repo), indent=2))
        return

    # --- 1. newest published record under the concept ---------------------
    q = urllib.parse.quote("conceptrecid:%s" % args.concept)
    listing = public_get("%s/records?q=%s&all_versions=true&sort=mostrecent&size=5"
                         % (API, q))
    hits = listing.get("hits", {}).get("hits", [])
    if not hits:
        raise SystemExit("no published record under concept %s" % args.concept)
    archived = {h["metadata"].get("version") for h in hits}
    if version in archived:
        print("::notice::%s is already archived under concept %s — nothing to do"
              % (version, args.concept))
        return
    latest = hits[0]

    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        raise SystemExit("ZENODO_TOKEN is not set")

    print("newest archived version: %s (record %s)" % (latest["metadata"].get("version"),
                                                       latest["id"]))

    # --- 2. draft the next version ---------------------------------------
    _, draft = call("POST", "%s/deposit/depositions/%s/actions/newversion"
                    % (API, latest["id"]), token)
    draft_id = draft["links"]["latest_draft"].rstrip("/").split("/")[-1]
    _, draft = call("GET", "%s/deposit/depositions/%s" % (API, draft_id), token)
    doi = draft["metadata"]["prereserve_doi"]["doi"]
    print("draft %s | reserved DOI %s" % (draft_id, doi))

    for f in draft.get("files", []):
        call("DELETE", "%s/deposit/depositions/%s/files/%s"
             % (API, draft_id, f["id"]), token)
        print("removed copied file:", f.get("filename"))

    call("PUT", "%s/deposit/depositions/%s" % (API, draft_id), token,
         payload=metadata(version, args.date, args.repo))

    # --- 3. upload the pinned tag tarball --------------------------------
    name = "%s-%s.tar.gz" % (args.repo.split("/")[-1], version)
    url = "https://codeload.github.com/%s/tar.gz/refs/tags/%s" % (args.repo, args.tag)
    with urllib.request.urlopen(url, timeout=300) as r:
        blob = r.read()
    md5 = hashlib.md5(blob).hexdigest()
    call("PUT", "%s/%s" % (draft["links"]["bucket"], name), token,
         raw=blob, content_type="application/octet-stream")
    print("uploaded %s (%d bytes, md5 %s)" % (name, len(blob), md5))

    _, after = call("GET", "%s/deposit/depositions/%s" % (API, draft_id), token)
    files = after.get("files", [])
    if len(files) != 1 or not files[0].get("checksum", "").endswith(md5):
        raise SystemExit("upload verification failed: %r" % [(f.get("filename"),
                                                             f.get("checksum")) for f in files])

    # --- 4. publish + verify ---------------------------------------------
    call("POST", "%s/deposit/depositions/%s/actions/publish" % (API, draft_id), token)
    for _ in range(30):
        _, rec = call("GET", "%s/deposit/depositions/%s" % (API, draft_id), token)
        if rec.get("state") == "done":
            break
        time.sleep(5)
    pub = public_get("%s/records/%s" % (API, draft_id))
    print("::notice::archived %s as %s" % (version, pub["doi"]))
    print("concept DOI: https://doi.org/%s" % pub["conceptdoi"])
    print("version DOI: https://doi.org/%s" % pub["doi"])
    # The public record is InvenioRDM-shaped: file entries are `key`/`size`
    # (`filename`/`filesize` on the legacy deposit endpoint) and may be wrapped
    # under `files.entries`. A wrong shape here must not fail a job whose publish
    # already succeeded.
    entries = pub.get("files", [])
    if isinstance(entries, dict):
        entries = list(entries.get("entries", {}).values())
    print("files: %s" % [(e.get("key") or e.get("filename"),
                          e.get("size") or e.get("filesize")) for e in entries])


if __name__ == "__main__":
    main()
