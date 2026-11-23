"""Repository-layout guard.

Two conventions are easy to break by accident and expensive to notice later:
planning notes drifting back to the repository root / into the user docs, and
``dev-notes/`` references rotting when a note is renamed. This module turns both
into a failing test instead of a reviewer's remark.

Conventions (see ``dev-notes/README.md``):

* user documentation lives in ``docs/``; plans, handoffs, scopes and roadmaps
  live in ``dev-notes/``;
* the repository root holds project files only;
* every ``dev-notes/<file>`` path cited from code or docs exists;
* the changelog is split per release under ``docs/changelog/`` and the root
  ``CHANGELOG.rst`` is a pointer, not the content.
"""

from __future__ import annotations

import fnmatch
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Filename patterns that belong in dev-notes/, never at the repository root.
ROOT_NOTE_PATTERNS = (
    "*PLAN*.md",
    "*PLAN*.rst",
    "HANDOFF-*.md",
    "HANDOFF*.md",
    "AI_LAYER*.md",
    "IDEA.md",
    "*_SCOPE.md",
    "*ROADMAP*.md",
    "*_ROADMAP.md",
)

#: Plan-shaped filenames that must not sit in the user documentation tree.
DOCS_PLAN_PATTERNS = (
    "*-plan.md",
    "*_PLAN.md",
    "*-plans.md",
    "ui-gap-*.md",
    "mlops-roadmap.md",
    "phase9.rst",
    "phase10.rst",
    "handoff*.md",
    "HANDOFF*.md",
)

#: Deliberately published study/report pages that look plan-ish but are not.
DOCS_ALLOWED = {"benchmarks-cspace.md", "dtrack-study.md", "blog-honest-benchmark.md"}

REFERENCE_RE = re.compile(r"dev-notes/([A-Za-z0-9_.\-]+\.(?:md|rst))")
REFERENCE_SCAN_DIRS = ("api", "SKSurrogate", "tests", "benchmarks", "docs", "examples")
REFERENCE_SCAN_SUFFIXES = (".py", ".rst", ".md")


def _tracked_name(path: pathlib.Path) -> bool:
    """Only guard files that are part of the repository (ignore local drafts)."""
    return not any(part.startswith(".") for part in path.parts)


def test_no_planning_notes_at_repository_root():
    offenders = []
    for path in ROOT.iterdir():
        if not path.is_file():
            continue
        for pattern in ROOT_NOTE_PATTERNS:
            if fnmatch.fnmatch(path.name, pattern):
                offenders.append(path.name)
                break
    assert not offenders, (
        "planning notes belong in dev-notes/, not the repository root: "
        f"{sorted(offenders)}"
    )


def test_user_docs_contain_no_planning_notes():
    offenders = []
    for path in (ROOT / "docs").iterdir():
        if not path.is_file() or path.name in DOCS_ALLOWED:
            continue
        for pattern in DOCS_PLAN_PATTERNS:
            if fnmatch.fnmatch(path.name, pattern):
                offenders.append(path.name)
                break
    assert not offenders, (
        "plans do not belong in the published documentation tree "
        f"(move to dev-notes/): {sorted(offenders)}"
    )


def test_dev_notes_readme_states_the_rule():
    readme = ROOT / "dev-notes" / "README.md"
    assert readme.is_file(), "dev-notes/README.md documents what belongs there"
    assert "user documentation" in readme.read_text(encoding="utf-8").lower()


def test_every_cited_dev_note_exists():
    """``dev-notes`` paths cited in code and docs must resolve."""
    missing: list[str] = []
    for directory in REFERENCE_SCAN_DIRS:
        base = ROOT / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in REFERENCE_SCAN_SUFFIXES:
                continue
            if any(part in {"_build", "node_modules", "dist"} for part in path.parts):
                continue
            if path == pathlib.Path(__file__).resolve():
                continue  # this module's own docstrings show example paths
            text = path.read_text(encoding="utf-8", errors="ignore")
            for name in REFERENCE_RE.findall(text):
                if not (ROOT / "dev-notes" / name).is_file():
                    missing.append(f"{path.relative_to(ROOT)} -> dev-notes/{name}")
    assert not missing, "dangling dev-notes references:\n" + "\n".join(sorted(set(missing)))


def test_no_references_to_plans_inside_docs_tree():
    """Nothing may still cite a plan at its old ``docs/…`` location."""
    moved = (
        "ui-plan.md",
        "ui-gap-implementation-plan.md",
        "visual-search-space-editor-plan.md",
        "mlops-roadmap.md",
        "phase9.rst",
        "phase10.rst",
    )
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in REFERENCE_SCAN_SUFFIXES:
            continue
        if any(part in {"_build", "node_modules", "dist", ".git"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name in moved:
            if f"docs/{name}" in text:
                offenders.append(f"{path.relative_to(ROOT)} -> docs/{name}")
    assert not offenders, "plans now live in dev-notes/; update these paths:\n" + "\n".join(
        sorted(offenders)
    )


def test_changelog_is_split_per_release():
    root_changelog = ROOT / "CHANGELOG.rst"
    assert root_changelog.stat().st_size < 2048, (
        "CHANGELOG.rst is a pointer; release notes live in docs/changelog/"
    )
    text = root_changelog.read_text(encoding="utf-8")
    assert "docs/changelog/index.rst" in text

    changelog = ROOT / "docs" / "changelog"
    assert (changelog / "index.rst").is_file()
    releases = sorted(p.name for p in changelog.glob("*.rst") if p.name != "index.rst")
    assert len(releases) >= 5, f"expected one page per release, found {releases}"
    # No single page may grow without bound: the whole point of the split.
    oversized = [p.name for p in changelog.glob("*.rst") if p.stat().st_size > 40_000]
    assert not oversized, f"split these releases further: {oversized}"


def test_changelog_index_links_every_release_page():
    changelog = ROOT / "docs" / "changelog"
    index = (changelog / "index.rst").read_text(encoding="utf-8")
    pages = [p.stem for p in changelog.glob("*.rst") if p.name != "index.rst"]
    unlinked = [p for p in pages if p not in index]
    assert not unlinked, f"release pages missing from the changelog toctree: {unlinked}"
