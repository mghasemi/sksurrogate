"""Versioned experiment manifests: the reproducibility record of one fit.

A *manifest* is the single object a reader needs to answer "what exactly produced
this artifact?": which dataset partition and fingerprint, which preprocessing and
estimator configuration, which random seeds and which generator stream each came
from, which interpreter/dependency/hardware stack, which git revision, how the
model was scored and validated, and the checksum of the artifact itself.

The record is **schema v2**. Version 1 was the flat ``reproducibility`` dict
written by ``api.training._reproducibility_record`` (task, data, configuration,
transformations, evaluation_protocol, environment, known_gaps). Readers written
against v2 must keep accepting v1 rows: :meth:`ExperimentManifest.from_dict`
upgrades them instead of rejecting them, because bundles stored before the
upgrade are still loadable and still have to explain themselves.

The module deliberately depends only on the standard library plus NumPy (already
a hard dependency). Optional backends are reported when installed and omitted
otherwise, so a manifest is always serializable.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

#: Current manifest schema written by this module.
MANIFEST_SCHEMA_VERSION = 2
#: Schemas :meth:`ExperimentManifest.from_dict` accepts (v1 is upgraded in place).
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = (1, 2)

#: Packages every manifest records when installed. The list is intentionally a
#: superset of the core requirements: a reader must be able to tell whether a
#: result came from a lean install or an extras-enabled one.
RECORDED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "SKSurrogate",
)
#: Optional backends: recorded only when importable/installed, never required.
OPTIONAL_PACKAGES = ("xgboost", "lightgbm", "optuna", "scikit-optimize", "dask")


def _package_versions(packages):
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            continue
        except Exception:  # pragma: no cover - defensive: broken metadata
            continue
    return versions


def dependency_versions(extra_packages=()):
    """Return the installed versions of the core and optional packages.

    ``extra_packages`` lets a caller (e.g. the API) add the packages it actually
    uses without changing this module's default contract.
    """
    versions = _package_versions(RECORDED_PACKAGES)
    versions.update(_package_versions(tuple(OPTIONAL_PACKAGES) + tuple(extra_packages)))
    return dict(sorted(versions.items()))


def _blas_info():
    """Best-effort description of the BLAS/LAPACK build NumPy is using.

    Reproducibility claims about floating-point results are meaningless without
    this: the same code on OpenBLAS and MKL can differ in the last bits. We use
    ``threadpoolctl`` when available (it reports the *loaded* runtime libraries)
    and fall back to NumPy's build-time configuration.
    """
    try:
        from threadpoolctl import threadpool_info

        libraries = []
        for entry in threadpool_info():
            name = entry.get("internal_api") or entry.get("user_api") or "unknown"
            version = entry.get("version")
            libraries.append("%s %s" % (name, version) if version else str(name))
        if libraries:
            return {"source": "threadpoolctl", "libraries": sorted(set(libraries))}
    except Exception:
        pass
    try:
        import numpy as np

        config = np.__config__.CONFIG
        dependencies = config.get("Build Dependencies", {})
        libraries = []
        for entry in dependencies.values():
            if isinstance(entry, dict) and entry.get("name"):
                version = entry.get("version")
                libraries.append(
                    "%s %s" % (entry["name"], version) if version else entry["name"]
                )
        blas = (config.get("blas") or {}).get("name")
        lapack = (config.get("lapack") or {}).get("name")
        if blas:
            libraries.insert(0, str(blas))
        if lapack:
            libraries.insert(1, str(lapack))
        if libraries:
            return {"source": "numpy.__config__", "libraries": sorted(set(libraries))}
    except Exception:
        pass
    return {"source": "unavailable", "libraries": []}


def hardware_snapshot():
    """Describe the machine and interpreter that ran the fit."""
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "blas": _blas_info(),
    }


def _git_command(path, *args):
    executable = shutil.which("git")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, "-C", str(path), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_revision(path=None):
    """Return the git revision of ``path``'s working tree, or a null record.

    ``revision`` is the full ``HEAD`` hash; ``dirty`` is ``True`` when tracked
    files differ from ``HEAD`` (untracked files do not count — a stray cache file
    must not invalidate a run). The dict always has the same keys so downstream
    readers never need to branch on whether git was available.
    """
    root = Path(path or Path.cwd()).resolve()
    revision = _git_command(root, "rev-parse", "HEAD")
    if revision is None:
        return {"revision": None, "dirty": None, "source": "unavailable"}
    status = _git_command(root, "status", "--porcelain", "--untracked-files=no")
    return {
        "revision": revision,
        "dirty": bool(status),
        "source": "git",
    }


def canonical_json(payload):
    """Serialize ``payload`` deterministically, for hashing and digesting."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text):
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path, *, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_digest(payload):
    """SHA-256 over a canonical JSON rendering of ``payload``."""
    return sha256_text(canonical_json(payload))


class ExperimentManifest:
    """A schema-v2 reproducibility record.

    Instances are plain-data containers: every field is JSON-serializable, so a
    manifest can be written beside a bundle, returned by the API and compared by
    digest without importing the package that produced it.
    """

    def __init__(
        self,
        *,
        task_name=None,
        data=None,
        preprocessing=None,
        configuration=None,
        evaluation_protocol=None,
        seeds=None,
        search_space=None,
        evaluation_history_digest=None,
        artifacts=None,
        environment=None,
        hardware=None,
        git=None,
        known_gaps=None,
        leakage_warnings=None,
        schema_version=MANIFEST_SCHEMA_VERSION,
        created_at=None,
        extra=None,
    ):
        self.schema_version = int(schema_version)
        self.task_name = task_name
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.data = dict(data or {})
        self.preprocessing = list(preprocessing or [])
        self.configuration = dict(configuration or {})
        self.evaluation_protocol = dict(evaluation_protocol or {})
        self.seeds = dict(seeds or {})
        self.search_space = search_space
        self.evaluation_history_digest = evaluation_history_digest
        self.artifacts = list(artifacts or [])
        self.environment = dict(environment or {})
        self.hardware = dict(hardware or {})
        self.git = dict(git or {})
        self.known_gaps = list(known_gaps or [])
        self.leakage_warnings = dict(leakage_warnings or {})
        #: Free-form keys preserved from a legacy record so an upgrade never
        #: silently drops information this module does not know about.
        self.extra = dict(extra or {})

    # -- construction ---------------------------------------------------- #

    @classmethod
    def from_dict(cls, record, *, upgrade=True):
        """Build a manifest from a stored (possibly v1) record.

        :raises ValueError: when the record's ``schema_version`` is newer than
            this module supports — guessing at a future schema is worse than
            failing loudly.
        """
        if not isinstance(record, dict):
            raise TypeError("a manifest record must be a mapping")
        version = record.get("schema_version", 1)
        try:
            version = int(version)
        except (TypeError, ValueError):
            raise ValueError("manifest schema_version %r is not an integer" % (version,))
        if version > MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported manifest schema %r (this release reads up to %r)"
                % (version, MANIFEST_SCHEMA_VERSION)
            )
        if version < 1:
            raise ValueError("manifest schema_version must be >= 1, got %r" % (version,))
        if upgrade and version < MANIFEST_SCHEMA_VERSION:
            record = upgrade_manifest(record)
        environment = dict(record.get("environment") or {})
        hardware = dict(record.get("hardware") or {})
        # A v1 record kept interpreter versions flat under ``environment``;
        # v2 nests them under ``python_version`` and friends plus ``packages``.
        if "packages" not in environment:
            legacy = {
                key: value
                for key, value in environment.items()
                if key.endswith("_version") or key == "sksurrogate_version"
            }
            if legacy:
                environment.setdefault("registered", legacy)
        known = {
            "schema_version",
            "task_name",
            "created_at",
            "data",
            "preprocessing",
            "transformations",
            "configuration",
            "evaluation_protocol",
            "seeds",
            "search_space",
            "evaluation_history_digest",
            "artifacts",
            "environment",
            "hardware",
            "git",
            "known_gaps",
            "leakage_warnings",
            "manifest_digest",
        }
        extra = {
            key: value for key, value in record.items() if key not in known
        }
        manifest = cls(
            task_name=record.get("task_name"),
            data=record.get("data"),
            preprocessing=record.get("preprocessing", record.get("transformations")),
            configuration=record.get("configuration"),
            evaluation_protocol=record.get("evaluation_protocol"),
            seeds=record.get("seeds"),
            search_space=record.get("search_space"),
            evaluation_history_digest=record.get("evaluation_history_digest"),
            artifacts=record.get("artifacts"),
            environment=environment,
            hardware=hardware,
            git=record.get("git"),
            known_gaps=record.get("known_gaps"),
            leakage_warnings=record.get("leakage_warnings"),
            schema_version=MANIFEST_SCHEMA_VERSION,
            created_at=record.get("created_at"),
            extra=extra,
        )
        return manifest

    @classmethod
    def from_bundle(cls, bundle):
        """Recover the manifest a :class:`~SKSurrogate.modelbundle.ModelBundle` carries.

        Bundles written before this module existed store the v1 record under
        ``metadata["reproducibility"]``; the upgrade path accepts it unchanged.
        """
        metadata = getattr(bundle, "metadata", None) or {}
        record = metadata.get("manifest") or metadata.get("reproducibility")
        if record:
            return cls.from_dict(record)
        # Last resort: a minimal manifest that still names the artifact and its
        # dataset — better than pretending provenance exists.
        return cls(
            task_name=getattr(bundle, "task_name", None),
            data={"dataset_fingerprint": getattr(bundle, "dataset_fingerprint", None)},
            known_gaps=["no reproducibility record was stored with this bundle"],
        )

    # -- serialization ---------------------------------------------------- #

    def to_dict(self, *, include_digest=True):
        payload = {
            "schema_version": self.schema_version,
            "task_name": self.task_name,
            "created_at": self.created_at,
            "data": self.data,
            "preprocessing": self.preprocessing,
            "configuration": self.configuration,
            "evaluation_protocol": self.evaluation_protocol,
            "seeds": self.seeds,
            "search_space": self.search_space,
            "evaluation_history_digest": self.evaluation_history_digest,
            "artifacts": self.artifacts,
            "environment": self.environment,
            "hardware": self.hardware,
            "git": self.git,
            "known_gaps": self.known_gaps,
            "leakage_warnings": self.leakage_warnings,
        }
        payload.update(self.extra)
        if include_digest:
            payload["manifest_digest"] = self.digest(payload)
        return payload

    def digest(self, payload=None):
        """SHA-256 over the manifest's canonical content (excluding the digest)."""
        body = self.to_dict(include_digest=False) if payload is None else dict(payload)
        body.pop("manifest_digest", None)
        return payload_digest(body)

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def write(self, path):
        """Atomically write ``manifest.json``-shaped JSON and return the path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(self.to_json() + "\n", encoding="utf-8")
        os.replace(str(temporary), str(destination))
        return destination

    @classmethod
    def read(cls, path):
        with open(path, encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))


def upgrade_manifest(record):
    """Return the schema-v2 form of a v1 reproducibility record.

    The upgrade is lossless: every unknown key is carried forward under the same
    name, so a caller reading a v2 manifest can still find the old fields.
    """
    if not isinstance(record, dict):
        raise TypeError("a manifest record must be a mapping")
    version = int(record.get("schema_version", 1) or 1)
    if version >= MANIFEST_SCHEMA_VERSION:
        return dict(record)
    upgraded = dict(record)
    upgraded["schema_version"] = MANIFEST_SCHEMA_VERSION
    upgraded.setdefault("preprocessing", record.get("transformations", []))
    upgraded.setdefault("seeds", {})
    upgraded.setdefault("search_space", None)
    upgraded.setdefault("evaluation_history_digest", None)
    upgraded.setdefault("artifacts", [])
    # Preserve the flat v1 environment block under ``registered`` and add live
    # runtime facts; a v1 reader that asks for ``environment["numpy_version"]``
    # keeps working because the flat keys are still at the top level.
    environment = dict(record.get("environment") or {})
    live = environment_snapshot(include_git=False)
    environment.setdefault("packages", live["packages"])
    environment.setdefault("python_version", live["python_version"])
    environment["registered"] = {
        key: value
        for key, value in environment.items()
        if key.endswith("_version") and key != "python_version"
    }
    upgraded["environment"] = environment
    upgraded.setdefault("hardware", live["hardware"])
    upgraded.setdefault("git", {})
    upgraded.setdefault("leakage_warnings", {})
    return upgraded


def environment_snapshot(*, include_git=True, path=None):
    """The full runtime block: packages, interpreter/hardware and git state."""
    hardware = hardware_snapshot()
    snapshot = {
        "python_version": hardware["python_version"],
        "packages": dependency_versions(),
        "hardware": hardware,
    }
    if include_git:
        snapshot["git"] = git_revision(path)
    return snapshot


def build_manifest(
    *,
    task_name=None,
    data=None,
    preprocessing=None,
    configuration=None,
    evaluation_protocol=None,
    seeds=None,
    search_space=None,
    evaluation_history_digest=None,
    artifacts=None,
    known_gaps=None,
    leakage_warnings=None,
    registered_environment=None,
    git_path=None,
    include_git=True,
    extra=None,
):
    """Assemble a schema-v2 manifest, filling environment/hardware/git live.

    ``registered_environment`` is the package-version block recorded when the
    dataset was registered; it is preserved under ``environment["registered"]``
    so a reader can see both the registration-time and the fit-time stack.
    """
    snapshot = environment_snapshot(include_git=include_git, path=git_path)
    environment = {
        "python_version": snapshot["python_version"],
        "packages": snapshot["packages"],
    }
    if registered_environment:
        environment["registered"] = dict(registered_environment)
    return ExperimentManifest(
        task_name=task_name,
        data=data,
        preprocessing=preprocessing,
        configuration=configuration,
        evaluation_protocol=evaluation_protocol,
        seeds=seeds,
        search_space=search_space,
        evaluation_history_digest=evaluation_history_digest,
        artifacts=artifacts,
        environment=environment,
        hardware=snapshot["hardware"],
        git=snapshot.get("git", {}),
        known_gaps=known_gaps,
        leakage_warnings=leakage_warnings,
        extra=extra,
    )
