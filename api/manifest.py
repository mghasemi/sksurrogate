"""API adapter that turns a fitted experiment into a schema-v2 manifest.

``SKSurrogate.manifest`` owns the format and the environment snapshot; this
module owns the API-side assembly: which tracker metadata counts as the
registered environment, how a partition history entry becomes an artifact
descriptor, and what the seed provenance block says about the two random streams
``SurrogateRandomCV.fit`` derives from ``random_state`` (``SeedSequence.spawn``).

Keeping the adapter separate means the core library never imports ``api`` (the
layer-boundary test enforces that direction) while the platform can still emit
the full record.
"""

from __future__ import annotations

from SKSurrogate.manifest import build_manifest, payload_digest

#: Environment keys recorded by ``mltrace`` when the dataset was registered.
REGISTERED_ENVIRONMENT_KEYS = (
    "python_version",
    "numpy_version",
    "pandas_version",
    "scipy_version",
    "scikit_learn_version",
    "xgboost_version",
    "lightgbm_version",
    "sksurrogate_version",
    "git_revision",
    "git_dirty",
)


def seed_provenance(random_state):
    """Describe where a search's randomness came from.

    Phase 1 (D1) derives two independent child streams from one
    ``SeedSequence(random_state)``: ``init_candidate`` and ``sampling``. The
    manifest records the parent seed and the stream mapping so a reader can tell
    that "same seed" means "same two streams", not "global RNG state happened to
    match". ``None`` is entropy — explicitly not reproducible.
    """
    if random_state is None:
        return {
            "random_state": None,
            "reproducible": False,
            "provenance": {
                "sampling": "entropy (SeedSequence without an explicit seed)",
                "init_candidate": "entropy (SeedSequence without an explicit seed)",
            },
        }
    return {
        "random_state": random_state,
        "reproducible": True,
        "provenance": {
            "parent": "numpy.random.SeedSequence(random_state)",
            "init_candidate": "SeedSequence.spawn(2)[0]",
            "sampling": "SeedSequence.spawn(2)[1]",
        },
    }


def registered_environment(metadata):
    """Extract the registration-time package block from tracker metadata."""
    return {
        key: metadata[key]
        for key in REGISTERED_ENVIRONMENT_KEYS
        if key in metadata
    }


def partition_artifacts(metadata, train_partition):
    """Describe the training partition artifact, including its fingerprint.

    The fingerprint is the digest ``mltrace`` computed when the partition was
    registered; it identifies the exact rows the model saw without re-reading a
    potentially large CSV at fit time.
    """
    history = metadata.get("dataset_partition_history") or {}
    entry = history.get(train_partition) or {}
    fingerprint = entry.get("fingerprint")
    if not fingerprint:
        return []
    return [
        {
            "kind": "dataset_partition",
            "id": train_partition,
            "fingerprint": fingerprint,
            "rows": entry.get("rows"),
            "columns": entry.get("columns"),
            "algorithm": "mltrace.dataset_fingerprint",
        }
    ]


def build_run_manifest(
    *,
    task_name,
    metadata,
    data,
    configuration,
    transformations,
    evaluation_protocol,
    seeds=None,
    evaluation_history=None,
    leakage_warnings=None,
    known_gaps=None,
    git_path=None,
):
    """Assemble the schema-v2 manifest for one API fit.

    ``configuration`` may already carry ``random_state`` and ``search_space``;
    they are lifted into the manifest so the audit view does not depend on the
    shape of a per-kind configuration dict.
    """
    random_state = configuration.get("random_state")
    search_space = configuration.get("search_space")
    history_digest = (
        payload_digest(evaluation_history) if evaluation_history is not None else None
    )
    return build_manifest(
        task_name=task_name,
        data=data,
        preprocessing=transformations,
        configuration=configuration,
        evaluation_protocol=evaluation_protocol,
        seeds=seeds if seeds is not None else seed_provenance(random_state),
        search_space=search_space,
        evaluation_history_digest=history_digest,
        artifacts=partition_artifacts(metadata, (data or {}).get("train_partition")),
        known_gaps=known_gaps,
        leakage_warnings=leakage_warnings,
        registered_environment=registered_environment(metadata),
        git_path=git_path,
    )


def stamp_evaluation_history(record, evaluation_history):
    """Record the digest of the evaluation history and refresh the manifest digest.

    The history is produced after the search finishes, so the manifest is built
    first and stamped here rather than duplicating the manifest assembly.
    """
    record["evaluation_history_digest"] = payload_digest(evaluation_history)
    record["manifest_digest"] = payload_digest(
        {key: value for key, value in record.items() if key != "manifest_digest"}
    )
    return record
