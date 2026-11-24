# Cutting an SKSurrogate release

**Status:** in progress — this file is the procedure the 0.8.0 cut is being written
against. Where a step exists only because something broke before, the reason is
recorded so it is not dropped.

Every release medium must be cut from **one commit**. That is the whole point of
the provenance story: if PyPI, Docker Hub, GitHub and the docs disagree about what
`vX.Y.Z` means, nothing else in the release can be trusted.

## 0. Pre-flight

```bash
git switch master && git pull --ff-only
git status --porcelain          # must be empty
```

- Full suite green locally: `env -u PYTHONPATH ./.venv/bin/python -m pytest tests -q --no-header`
- CI green on the exact commit you are about to tag (the `push: master` run).
- Nothing in the release notes claims more than the measured evidence. As of 0.8.0
  that means: **no benchmark superiority claim** — `docs/benchmarks.md` is frozen at
  v1.9 and the v2.0 matrix is gated in `benchmarks/DESIGN-v2-cost.md`.

## 1. Version and its dependents

`SKSurrogate/__init__.py::__version__` is the single source of truth: `setup.py`
reads it, and `sksurrogate_mcp` derives its own version from the installed
distribution. Everything else is a copy that must be updated or regenerated.

| File | What to change | Why |
|---|---|---|
| `SKSurrogate/__init__.py` | `__version__` | the only real version |
| `sksurrogate_mcp/openapi_ops.json` | regenerate, do not hand-edit | `generated_from_version` is compared by the CI drift guard; a bump without regeneration fails the `Tests + smoke` job **by design** |
| `docs/conf.py` | `version`, `release` | Sphinx literals, not derived; drifted two releases behind before 0.8.0 |
| `readme.rst` | the Docker pin example | names a concrete tag |
| `docs/changelog/<version>.rst` | date in the title | the page is the release record |
| `docs/changelog/index.rst` | table row: real date, honest highlights | a "not published" row is better than a phantom release |
| `CITATION.cff` | `version`, `date-released`, `doi` | **only after** the Zenodo archive for this version exists — see §5 |

Regenerate the snapshot from the tree:

```bash
env -u PYTHONPATH ./.venv/bin/python scripts/generate_mcp_openapi_snapshot.py
env -u PYTHONPATH ./.venv/bin/python scripts/generate_mcp_openapi_snapshot.py --check
```

Expected output: `snapshot is current: N operations (core X.Y.Z)`.

## 2. Prove the artifact, not the working tree

The editable install in the shared venv resolves to whichever worktree installed it
and will happily report the *old* version. A local suite pass therefore says nothing
about a built wheel. Build one and install it into a throwaway venv:

```bash
python -m build --wheel --sdist            # or: pip wheel . -w dist --no-deps
python -m venv /tmp/relcheck && /tmp/relcheck/bin/pip install dist/sksurrogate-X.Y.Z-py3-none-any.whl
/tmp/relcheck/bin/python -c "import SKSurrogate; print(SKSurrogate.__version__)"
/tmp/relcheck/bin/python -c "import sksurrogate_mcp; print(sksurrogate_mcp.__version__)"
```

Both must print `X.Y.Z`. The `install-smoke` CI job does the same thing; if it is
green on the release commit, that is the evidence to cite.

> After bumping `__version__`, a **local** run of
> `tests/test_package_version_tracks_the_core_release.py` fails until the venv is
> refreshed (`pip install --no-deps -e .`). That is expected, not a regression — and
> it is exactly why the wheel check above is the real gate.

## 3. Tag, then publish

```bash
git tag -a vX.Y.Z -m "SKSurrogate X.Y.Z"
git push origin vX.Y.Z
```

Order matters: the tag must exist before the GitHub release, and the GitHub release
must exist before Zenodo archives it.

| Medium | Command / action | Verify with |
|---|---|---|
| PyPI | `python -m twine upload dist/*` | `pip index versions sksurrogate` |
| GitHub release | `gh release create vX.Y.Z --title vX.Y.Z --notes-file <notes>` | `gh release list` |
| Docker Hub | `docker build` from the **tagged commit**, then push `:X.Y.Z` **and** `:latest` | Docker Hub tag list |
| Docs | RTD build on the tag; GitHub Pages for the site | rendered page shows the right version |

Tag naming history: Docker Hub currently carries `v0.3.0` and `0.3.1` — pick one
form (bare `X.Y.Z`) and use it everywhere from 0.8.0 on.

## 4. Site and downstream copies

- `sksurrogate-site` footer prints a version (`sksurrogate v0.3.0 · ...`) and the
  honest-benchmark post pins `pip install sksurrogate # v0.2.0`. Both are copies.
- Anything that quotes a version in a blog post or FAQ answer.

## 5. Citation and DOI

`CITATION.cff` carries the **concept** DOI of the archive, not the version DOI.
Update it only after Zenodo has archived the new release, otherwise the file claims
a version the DOI does not resolve to. Until then, leave it pointing at the archived
version and note the discrepancy in the release announcement.

## 6. After the dust settles

- Delete the merged release branches: `git push origin --delete release/vX.Y.Z ...`
- Confirm the four media agree on the version (`pip index versions`, Docker Hub tags,
  `gh release list`, `git tag`).
- If the release is cumulative, say so in the notes and link the folded pages rather
  than duplicating them.
