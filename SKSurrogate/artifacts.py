"""Versioned JSON artifacts and their one-way upgrades.

The control plane stores several kinds of document on disk. Each one is read by
code that may be older than the writer, so every kind carries a
``schema_version`` and is upgraded on read instead of being rejected:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Kind
     - Version field
     - Notes
   * - ``job``
     - ``schema_version``
     - Job records (``<API home>/jobs/<id>.json``). Version 1 lacked the explicit
       display/progress/compute fields; the upgrade fills defaults and never
       rewrites history.
   * - ``serving_pin``
     - ``schema_version``
     - One serving document per task. Version 1 had no ``history`` list.
   * - ``registry_index``
     - ``schema_version``
     - The model registry index. Version 1 had no version stamp.
   * - ``bundle``
     - ``format_version``
     - Owned by :mod:`SKSurrogate.modelbundle` (v1 tabular and v2 add the temporal
       block and manifest); :func:`read_artifact` delegates to the model bundle
       reader instead of duplicating that switch.
   * - ``manifest``
     - ``schema_version``
     - Owned by :mod:`SKSurrogate.manifest` (v1 ``reproducibility`` records are
       upgraded on read, newer-than-v2 is refused).

**Not** document artifacts: dataset and partition rows live in
``datasets/<task>/<partition>.csv`` plus the ``mltrace`` SQLite database, and
explanations are synchronous responses (only report jobs persist a result, and
that result lives inside a job record). Those have no version field to stamp;
inventing a sidecar for them would add a file nobody reads.

Rules:

* an artifact **newer** than this package supports raises
  :class:`ArtifactVersionError` — silently dropping unknown fields would corrupt
  a newer writer's state;
* an older artifact is upgraded by adding defaults, never by removing keys, so a
  downgrade still finds what it wrote;
* :func:`stamp` is applied on write, so the migration path is exercised only by
  artifacts written before this module existed.
"""

from __future__ import annotations

__all__ = [
    "ArtifactVersionError",
    "CURRENT_VERSIONS",
    "migrate",
    "read_artifact",
    "schema_version_of",
    "stamp",
]

#: The schema version this package writes for each document kind.
CURRENT_VERSIONS = {
    "job": 2,
    "serving_pin": 2,
    "registry_index": 2,
}


class ArtifactVersionError(ValueError):
    """Raised when an artifact is newer than this package can upgrade."""


def _migrate_job(payload: dict) -> dict:
    """v1 → v2: explicit display, progress, compute and resume fields."""
    upgraded = dict(payload)
    upgraded.setdefault("label", None)
    upgraded.setdefault("log_counts", {})
    upgraded.setdefault("compute", {})
    upgraded.setdefault("result", None)
    upgraded.setdefault("error", None)
    # ``resumable`` was derived at submit time; recompute it for an old record
    # because the saved value did not exist and a stale one would be a guess.
    upgraded.setdefault("resumable", isinstance(upgraded.get("resume"), dict))
    upgraded.setdefault("stop", None)
    upgraded.setdefault("cleanup", None)
    return upgraded


def _migrate_serving_pin(payload: dict) -> dict:
    """v1 → v2: an explicit history list and stable display fields."""
    upgraded = dict(payload)
    upgraded.setdefault("history", [])
    upgraded.setdefault("active", bool(upgraded.get("model_version")))
    upgraded.setdefault("note", None)
    upgraded.setdefault("requested", upgraded.get("model_version"))
    return upgraded


def _migrate_registry_index(payload: dict) -> dict:
    """v1 → v2: an explicit models mapping (the index had one, but unstamped)."""
    upgraded = dict(payload)
    models = upgraded.get("models")
    upgraded["models"] = models if isinstance(models, dict) else {}
    return upgraded


_MIGRATIONS = {
    "job": _migrate_job,
    "serving_pin": _migrate_serving_pin,
    "registry_index": _migrate_registry_index,
}


def schema_version_of(kind: str, payload: dict) -> int:
    """The artifact's version, treating an unstamped document as version 1."""
    if kind not in CURRENT_VERSIONS:
        raise ArtifactVersionError("unknown artifact kind %r" % (kind,))
    if not isinstance(payload, dict):
        raise ArtifactVersionError("%s artifact must be a mapping" % kind)
    return int(payload.get("schema_version") or 1)


def stamp(kind: str, payload: dict) -> dict:
    """Return a copy of ``payload`` stamped with the current version for ``kind``."""
    if kind not in CURRENT_VERSIONS:
        raise ArtifactVersionError("unknown artifact kind %r" % (kind,))
    if not isinstance(payload, dict):
        raise ArtifactVersionError("%s artifact must be a mapping" % kind)
    stamped = dict(payload)
    stamped["schema_version"] = CURRENT_VERSIONS[kind]
    return stamped


def migrate(kind: str, payload: dict) -> dict:
    """Upgrade one artifact to the current version, preserving unknown keys."""
    if kind not in _MIGRATIONS:
        raise ArtifactVersionError("unknown artifact kind %r" % (kind,))
    version = schema_version_of(kind, payload)
    latest = CURRENT_VERSIONS[kind]
    if version > latest:
        raise ArtifactVersionError(
            "%s artifact has schema_version %d, newer than the supported %d; "
            "upgrade SKSurrogate before reading it" % (kind, version, latest)
        )
    upgraded = _MIGRATIONS[kind](payload)
    upgraded["schema_version"] = latest
    return upgraded


def read_artifact(kind: str, payload: dict) -> dict:
    """Read an artifact, upgrading an older version and refusing a newer one.

    ``kind="bundle"``/``"manifest"`` are routed to their owning readers; call
    those modules directly (`ModelBundle.from_record`, `ExperimentManifest.from_dict`)
    so this function stays a dispatch for the documents it owns.
    """
    if kind in ("bundle", "manifest"):
        raise ArtifactVersionError(
            "%s artifacts are versioned by their owning module; use "
            "ModelBundle.from_record or ExperimentManifest.from_dict" % kind
        )
    return migrate(kind, payload)
