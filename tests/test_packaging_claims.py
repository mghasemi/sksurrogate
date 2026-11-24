"""Guard the install claims in user-facing copy against what the artifacts ship.

``setup.py`` packages only ``SKSurrogate`` and ``sksurrogate_mcp``; neither the
wheel nor the sdist contains ``api/`` or ``webui/`` (the React bundle is a build
artifact and is not tracked). An instruction to ``pip install "sksurrogate[api]"``
therefore adds dependencies and no importable ``api`` module, which is the
failure this test prevents from reappearing in the docs.

The extras themselves are legitimate — ``pip install ".[api]"`` from a source
checkout needs them — so this test bans the *index-install* wording, not the extra.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# "sksurrogate[api]" / "sksurrogate[ui]" — an extra that only carries the control
# plane's dependencies. Other extras ([xgboost], [all], ...) are correct as-is.
INDEX_INSTALL_OF_PLATFORM_EXTRA = re.compile(r"sksurrogate\[(?:api|ui)\]")

DOC_PATTERNS = ("readme.rst", "docs/**/*.rst", "docs/**/*.md")


def _doc_files():
    files = [ROOT / "readme.rst"]
    for pattern in DOC_PATTERNS[1:]:
        files.extend(sorted(ROOT.glob(pattern)))
    return [path for path in files if path.is_file()]


@pytest.mark.parametrize(
    "path", _doc_files(), ids=lambda p: str(p.relative_to(ROOT)).replace("/", "-")
)
def test_docs_do_not_promise_a_pip_installable_control_plane(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    assert not INDEX_INSTALL_OF_PLATFORM_EXTRA.search(text), (
        f"{path.relative_to(ROOT)} tells the reader to install the [api]/[ui] extra "
        'from the index. Use `pip install ".[api]"` from a source checkout, or the '
        "Docker image; api/ and webui/ are not in the wheel."
    )


def test_setup_packages_exclude_the_control_plane():
    """If this fails because the platform was packaged, the docs test above can go."""
    setup_py = (ROOT / "setup.py").read_text(encoding="utf-8")
    match = re.search(r"packages=\[([^\]]*)\]", setup_py)
    assert match is not None, "setup.py has no literal packages=[...] list"
    names = set(re.findall(r'"([^"]+)"', match.group(1)))
    assert "api" not in names and "webui" not in names, (
        "api/ or webui/ is now packaged — publish the modules, then update the "
        "install wording in readme.rst, docs/architecture.rst and "
        "requirements-api.txt and delete these guards"
    )


def test_citation_records_the_shipped_version():
    """CITATION.cff is what Zenodo and the site's /cite page mirror.

    It drifted once already: the v0.8.0 tag shipped a citation file for 0.2.0,
    which made a correct citation impossible to produce from the repository.
    The DOI is deliberately not checked — it changes only when a release is
    archived, which happens after the version bump.
    """
    init_py = (ROOT / "SKSurrogate" / "__init__.py").read_text(encoding="utf-8")
    version = re.search(r'^__version__\s*=\s*"([^"]+)"', init_py, re.M).group(1)
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    cited = re.search(r'^version:\s*"?([^"\n]+)"?\s*$', cff, re.M).group(1).strip()
    assert cited == version, (
        f"CITATION.cff says version {cited}, the package says {version} — "
        "bump CITATION.cff in the same commit as the version bump"
    )
