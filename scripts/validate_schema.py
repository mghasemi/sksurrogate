#!/usr/bin/env python3
"""Validate the live page's structured data with validator.schema.org.

The site audit is offline by design; this script is the network half. It posts the
published URL to the same validator the web UI uses and fails on any *error*
(warnings are printed but do not fail — schema.org raises non-severe type hints
that Google's rich-result tooling ignores).

    python scripts/validate_schema.py                       # https://sksurrogate.com/
    python scripts/validate_schema.py --url https://... --strict

Exit codes: 0 clean (errors == 0), 1 errors present, 2 the validator was unusable.
"""
import argparse
import json
import subprocess
import sys

SITE = "https://sksurrogate.com/"


def fetch(url):
    raw = subprocess.check_output([
        "curl", "-s", "-m", "60", "-X", "POST",
        "https://validator.schema.org/validate",
        "-H", "Content-Type: application/x-www-form-urlencoded",
        "--data-urlencode", "url=" + url,
    ])
    text = raw.decode("utf-8", "replace")
    # validator.schema.org prefixes JSON with the anti-hijack sequence )]}'
    return json.loads(text[text.index("{"):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=SITE)
    ap.add_argument("--strict", action="store_true", help="fail on warnings too")
    args = ap.parse_args()

    try:
        data = fetch(args.url)
    except Exception as exc:  # noqa: BLE001 - report, do not traceback
        print("could not query validator.schema.org: %s" % exc)
        return 2

    errors = data.get("totalNumErrors", 0)
    warnings = data.get("totalNumWarnings", 0)
    print("%s: %d objects, %d errors, %d warnings"
          % (args.url, data.get("numObjects", 0), errors, warnings))

    for group in data.get("tripleGroups", []):
        for node in group.get("nodes", []):
            types = [t.get("value") for t in node.get("types", [])]
            for prop in node.get("properties", []):
                for err in prop.get("errors", []):
                    label = err.get("errorType", "ISSUE")
                    print("  %-14s %-22s %s" % (label, types[0] if types else "?", prop.get("pred")))
            for err in node.get("errors", []):
                print("  %-14s %-22s (node-level)" % (err.get("errorType", "ISSUE"),
                                                      types[0] if types else "?"))

    if errors or (args.strict and warnings):
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
