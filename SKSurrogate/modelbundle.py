"""Portable model bundles and a small filesystem-backed model registry.

Records written by this version of the library are **format v2**: they carry an
optional ``temporal`` block (see :mod:`SKSurrogate.temporal`) describing the
design behind a forecasting bundle — its spec, target column, feature
configuration and serving strategy — so a saved temporal model can be served as
a multi-step forecast after a reload, and an optional ``manifest`` block (see
:mod:`SKSurrogate.manifest`) carrying the schema-v2 experiment manifest. Records
written by earlier releases are **format v1**; both remain loadable (v1 records
simply have no ``temporal``/``manifest`` block). :func:`save_bundle` additionally
writes a ``<name>.bundle.manifest.json`` sidecar with the SHA-256 of the exact
bundle bytes; :func:`verify_bundle` refuses a mismatched artifact.
See plan phase P6 in ``TIME_SERIES_PLAN.md`` for the temporal design.
"""

import importlib.metadata
import json
import os
import platform
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib

from .manifest import MANIFEST_SCHEMA_VERSION, sha256_file


BUNDLE_FORMAT_VERSION = 2
#: Record format versions that :func:`load_bundle` / :meth:`ModelBundle.from_record`
#: accept. v1 records (no temporal block) keep loading; anything newer than v2 is
#: refused loudly rather than guessed at.
SUPPORTED_BUNDLE_FORMAT_VERSIONS = (1, 2)
REGISTRY_STATES = {"candidate", "validated", "staging", "production", "archived"}
#: Names reserved for the lifecycle pointers maintained by promote/rollback.
LIFECYCLE_ALIAS_NAMES = frozenset(REGISTRY_STATES | {"latest"})
#: User-chosen model alias grammar: lowercase slug, max 64 characters.
ALIAS_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "password",
    "passwd",
    "secret",
    "token",
    "private_key",
    "privatekey",
    "ssn",
}


def _redact_sensitive_value(key, value):
    if isinstance(key, str):
        normalized = key.lower().replace("-", "_")
        if any(token in normalized for token in _SENSITIVE_KEYS):
            return "[REDACTED]"
    if isinstance(value, dict):
        return {inner_key: _redact_sensitive_value(inner_key, inner_value) for inner_key, inner_value in value.items()}
    if isinstance(value, list):
        return [_redact_sensitive_value(key, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_sensitive_value(key, item) for item in value)
    return value


def _dependencies():
    dependencies = {}
    for package in ("numpy", "pandas", "scikit-learn", "SKSurrogate"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return dependencies


class ModelBundle:
    """A fitted estimator and the metadata needed to reproduce its inputs.

    For tabular models ``model`` is a scikit-learn estimator (or pipeline) that
    predicts from a feature frame. For temporal forecasting models it is a
    pipeline whose first step is a :class:`~SKSurrogate.temporal.ForecastFeatureBuilder`
    and the :attr:`temporal` block records the design behind it — spec, target
    column, feature configuration and serving strategy — so the bundle can be
    reloaded and served as multi-step forecasts via :meth:`predict_forecast`.
    """

    def __init__(
        self,
        model,
        *,
        task_name=None,
        schema=None,
        preprocessing=None,
        metadata=None,
        metrics=None,
        dataset_fingerprint=None,
        config_version=None,
        model_version=None,
        dependencies=None,
        owner=None,
        run_id=None,
        sensitive_features=None,
        audit_events=None,
        interval_calibration=None,
        temporal=None,
        manifest=None,
    ):
        self.model = model
        self.task_name = task_name
        self.schema = {} if schema is None else schema
        self.preprocessing = preprocessing
        self.metadata = {} if metadata is None else metadata
        self.metrics = {} if metrics is None else metrics
        self.dataset_fingerprint = dataset_fingerprint
        self.config_version = config_version
        self.model_version = model_version or uuid.uuid4().hex
        self.dependencies = _dependencies() if dependencies is None else dependencies
        self.owner = owner
        self.run_id = run_id
        self.sensitive_features = list(sensitive_features) if sensitive_features is not None else []
        self.audit_events = list(audit_events) if audit_events is not None else []
        # Residual offsets from a rolling-origin backtest: the calibration serving
        # needs to emit an interval without recomputing anything. ``None`` means the
        # bundle has never been calibrated, and asking for an interval raises.
        self.interval_calibration = (
            dict(interval_calibration) if interval_calibration is not None else None
        )
        # Temporal design for forecasting bundles: the mapping produced by
        # ``TemporalServeState.to_dict()`` (spec, target, feature_spec, strategy).
        # Tabular bundles keep this as ``None``; :meth:`predict_forecast` raises
        # on them. Stored verbatim — temporal.py owns its own schema validation.
        if temporal is not None:
            try:
                self.temporal = dict(temporal)
            except (TypeError, ValueError):
                raise TypeError(
                    "the temporal block must be a mapping (e.g. "
                    "`TemporalServeState.to_dict()`), got %r" % type(temporal).__name__
                ) from None
        else:
            self.temporal = None
        #: Schema-v2 experiment manifest (see :mod:`SKSurrogate.manifest`). Older
        #: callers keep storing the v1 record under ``metadata["reproducibility"]``;
        #: this attribute is the additive v2 location. ``None`` means the bundle
        #: predates manifests or was built by a minimal caller.
        if manifest is not None and not isinstance(manifest, dict):
            raise TypeError(
                "manifest must be a mapping (e.g. ExperimentManifest.to_dict())"
            )
        self.manifest = dict(manifest) if manifest is not None else None
        self.created_at = datetime.now(timezone.utc).isoformat()

    def record_audit_event(self, event_type, **details):
        snapshot = {**self.metadata, **details}
        event = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": _redact_sensitive_value("details", snapshot),
        }
        self.audit_events.append(event)
        return event

    def to_record(self):
        return {
            "format_version": BUNDLE_FORMAT_VERSION,
            "model_version": self.model_version,
            "created_at": self.created_at,
            "task_name": self.task_name,
            "model": self.model,
            "preprocessing": self.preprocessing,
            "schema": self.schema,
            "metadata": _redact_sensitive_value("metadata", self.metadata),
            "metrics": self.metrics,
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_version": self.config_version,
            "dependencies": self.dependencies,
            "owner": self.owner,
            "run_id": self.run_id,
            "sensitive_features": list(self.sensitive_features),
            "audit_events": self.audit_events,
            "interval_calibration": self.interval_calibration,
            # v2: the temporal design block (None for tabular bundles). Kept at
            # the end so records stay readable by hand and diffs stay stable.
            "temporal": self.temporal,
            # Additive in the v2 format: a schema-v2 experiment manifest. Readers
            # that predate it ignore the key; new readers prefer it over the
            # legacy ``metadata["reproducibility"]`` record.
            "manifest": self.manifest,
        }

    def save(self, path):
        """Atomically write this bundle; same as :func:`save_bundle`."""
        return save_bundle(self, path)

    @classmethod
    def load(cls, path, **kwargs):
        """Load a bundle from ``path``; same as :func:`load_bundle`."""
        return load_bundle(path, **kwargs)

    @classmethod
    def from_record(cls, record):
        format_version = record.get("format_version")
        if format_version not in SUPPORTED_BUNDLE_FORMAT_VERSIONS:
            raise ValueError(
                "Unsupported model bundle format %r (supported: %s)"
                % (format_version, ", ".join(str(v) for v in SUPPORTED_BUNDLE_FORMAT_VERSIONS))
            )
        # ``temporal`` is a v2 key; v1 records simply lack it and stay tabular.
        temporal = record.get("temporal")
        bundle = cls(
            record["model"],
            task_name=record.get("task_name"),
            schema=record.get("schema"),
            preprocessing=record.get("preprocessing"),
            metadata=record.get("metadata"),
            metrics=record.get("metrics"),
            dataset_fingerprint=record.get("dataset_fingerprint"),
            config_version=record.get("config_version"),
            model_version=record.get("model_version"),
            dependencies=record.get("dependencies"),
            owner=record.get("owner"),
            run_id=record.get("run_id"),
            sensitive_features=record.get("sensitive_features"),
            audit_events=record.get("audit_events", []),
            interval_calibration=record.get("interval_calibration"),
            temporal=temporal,
            manifest=record.get("manifest"),
        )
        bundle.created_at = record.get("created_at", bundle.created_at)
        return bundle

    def predict(self, *args, **kwargs):
        """Delegate prediction to the fitted model in the bundle."""
        return self.model.predict(*args, **kwargs)

    def interval_bounds(self, predictions):
        """Apply the bundle's stored interval calibration to point predictions.

        :raises ValueError: when the bundle has no calibration — an uncalibrated
            bundle must not silently emit a zero-width interval.
        """
        if not self.interval_calibration:
            raise ValueError(
                "bundle %r has no interval calibration; run a rolling-origin "
                "backtest (mltrack.backtest) and store result['intervals']["
                "'calibration'] first" % self.model_version
            )
        from .forecast_intervals import IntervalCalibration

        calibration = IntervalCalibration.from_dict(self.interval_calibration)
        return calibration.bounds(predictions)

    def predict_forecast(self, history, horizon=None, future_exogenous=None):
        """Serve a multi-step forecast from this bundle's temporal design.

        Thin delegation to :func:`SKSurrogate.temporal.serve_forecast`. The
        bundle must carry the ``temporal`` block written by plan phase P6 in
        ``TIME_SERIES_PLAN.md`` (a v1 tabular bundle raises ``ValueError``).
        ``history`` is one series of past observations; ``horizon`` steps are
        predicted — a direct-strategy bundle answers exactly its design horizon,
        while a recursive bundle accepts any positive horizon.

        :raises ValueError: when this is a tabular bundle without a temporal
            design, the requested horizon is refused by the stored strategy, or
            the history does not satisfy the stored spec.
        """
        from .temporal import serve_forecast

        return serve_forecast(
            self, history, horizon=horizon, future_exogenous=future_exogenous
        )


def _manifest_sidecar_path(path):
    """The sidecar written beside a bundle: ``model.bundle.manifest.json``."""
    return Path(path).with_name(Path(path).name + ".manifest.json")


def _strict_bundles_default():
    return (os.environ.get("SKSURROGATE_STRICT_BUNDLES") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


class BundleIntegrityError(ValueError):
    """Raised when a bundle's bytes do not match the checksum recorded beside it."""


def _sidecar_manifest_payload(bundle, destination, checksum, size):
    """The envelope written to the sidecar: artifact identity plus the manifest.

    The checksum lives in the envelope, never inside the payload it hashes — a
    record cannot contain its own digest. The manifest block itself is the
    schema-v2 record (or a minimal honest placeholder for legacy callers).
    """
    manifest = dict(bundle.manifest) if isinstance(bundle.manifest, dict) else None
    if manifest is None:
        legacy = (bundle.metadata or {}).get("reproducibility")
        manifest = dict(legacy) if isinstance(legacy, dict) else {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "task_name": bundle.task_name,
            "data": {"dataset_fingerprint": bundle.dataset_fingerprint},
            "known_gaps": ["no reproducibility record was stored with this bundle"],
        }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "model_bundle",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": {
            "path": destination.name,
            "bytes": int(size),
            "sha256": checksum,
        },
        "manifest": manifest,
    }


def save_bundle(bundle, path):
    """Atomically write a :class:`ModelBundle` and its manifest sidecar.

    Two files land beside each other: ``<name>.bundle`` (the joblib payload) and
    ``<name>.bundle.manifest.json`` (the SHA-256 of those exact bytes plus the
    schema-v2 experiment manifest). The sidecar is written after the payload is
    atomically in place, so :func:`verify_bundle` can always trust the pair.
    """
    if not isinstance(bundle, ModelBundle):
        raise TypeError("bundle must be a ModelBundle")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=str(destination.parent), prefix=".bundle-", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        joblib.dump(bundle.to_record(), temporary)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        os.replace(str(temporary_path), str(destination))
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    checksum = sha256_file(destination)
    sidecar = _manifest_sidecar_path(destination)
    payload = _sidecar_manifest_payload(
        bundle, destination, checksum, destination.stat().st_size
    )
    sidecar_tmp = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(str(sidecar_tmp), str(sidecar))
    return destination


def verify_bundle(path, *, strict=None):
    """Verify a bundle against the checksum recorded in its manifest sidecar.

    Returns a dict describing the verdict. When the checksum does **not** match,
    :class:`BundleIntegrityError` is raised — a tampered or truncated artifact
    must never be loaded silently.

    A missing sidecar is *not* an integrity failure for bundles written before
    manifests existed. It is reported as ``verified: False`` unless ``strict``
    (or ``SKSURROGATE_STRICT_BUNDLES=1``) asks for fail-closed behaviour.
    """
    destination = Path(path)
    if strict is None:
        strict = _strict_bundles_default()
    sidecar = _manifest_sidecar_path(destination)
    if not destination.exists():
        raise FileNotFoundError("model bundle not found: %s" % destination)
    if not sidecar.exists():
        if strict:
            raise BundleIntegrityError(
                "refusing to load %s: no manifest sidecar at %s (strict mode)"
                % (destination, sidecar)
            )
        return {
            "verified": False,
            "reason": "manifest sidecar missing (bundle predates checksums)",
            "path": str(destination),
        }
    with sidecar.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    recorded = (payload.get("artifact") or {}).get("sha256")
    if not recorded:
        raise BundleIntegrityError(
            "manifest sidecar %s records no artifact checksum" % sidecar
        )
    actual = sha256_file(destination)
    if actual != recorded:
        raise BundleIntegrityError(
            "bundle checksum mismatch for %s: manifest records %s, file hashes to %s"
            % (destination, recorded, actual)
        )
    return {
        "verified": True,
        "sha256": actual,
        "path": str(destination),
        "manifest": payload.get("manifest"),
        "sidecar": str(sidecar),
    }


def load_bundle(
    path,
    *,
    expected_schema=None,
    strict_dependencies=True,
    verify=False,
):
    """Load a bundle and reject incompatible schema or runtime dependencies.

    ``verify=True`` runs :func:`verify_bundle` first (fail-closed on a checksum
    mismatch, and on a missing sidecar too — an explicit request for integrity
    must not succeed without evidence).
    """
    if verify:
        verify_bundle(path, strict=True)
    record = joblib.load(path)
    bundle = ModelBundle.from_record(record)
    if expected_schema is not None and bundle.schema != expected_schema:
        raise ValueError("Model bundle schema is incompatible with the expected schema")
    if strict_dependencies:
        current = _dependencies()
        incompatible = {
            name: (required, current.get(name))
            for name, required in bundle.dependencies.items()
            if current.get(name) != required
        }
        if incompatible:
            raise ValueError("Model bundle dependencies are incompatible: %s" % incompatible)
    return bundle


def export_mlflow(bundle, path):
    """Export a bundle using the standard MLflow model directory layout.

    MLflow remains an optional dependency. When installed, its sklearn flavor
    can consume the exported ``MLmodel`` and ``model.pkl`` files directly.
    """
    if not isinstance(bundle, ModelBundle):
        raise TypeError("bundle must be a ModelBundle")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("MLflow export destination already exists: %s" % destination)
    staging = Path(tempfile.mkdtemp(prefix=".mlflow-", dir=str(destination.parent)))
    try:
        joblib.dump(bundle.model, staging / "model.pkl")
        with (staging / "bundle.json").open("w", encoding="utf-8") as stream:
            json.dump(bundle.to_record(), stream, indent=2, sort_keys=True, default=str)
        sklearn_version = bundle.dependencies.get("scikit-learn", "unknown")
        mlmodel = "\n".join(
            [
                "artifact_path: model",
                "flavors:",
                "  python_function:",
                "    loader_module: mlflow.sklearn",
                "    model_path: model.pkl",
                "    python_version: %s" % platform.python_version(),
                "  sklearn:",
                "    pickled_model: model.pkl",
                "    sklearn_version: %s" % sklearn_version,
                "metadata:",
                "  model_version: %s" % bundle.model_version,
                "  task_name: %s" % (bundle.task_name or ""),
                "",
            ]
        )
        (staging / "MLmodel").write_text(mlmodel, encoding="utf-8")
        os.replace(str(staging), str(destination))
    except Exception:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


class ModelRegistry:
    """Filesystem registry for versioned bundles and promotion history."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "registry.json"

    def _read_index(self):
        if not self._index_path.exists():
            return {"models": {}}
        with self._index_path.open(encoding="utf-8") as stream:
            return json.load(stream)

    def _write_index(self, index):
        with tempfile.NamedTemporaryFile(
            mode="w", dir=str(self.root), prefix=".registry-", delete=False, encoding="utf-8"
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(index, temporary, indent=2, sort_keys=True)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(str(temporary_path), str(self._index_path))

    def register(self, bundle):
        """Store a new bundle as a candidate version."""
        if not isinstance(bundle, ModelBundle):
            raise TypeError("bundle must be a ModelBundle")
        task_name = bundle.task_name or "default"
        bundle.record_audit_event("register", task_name=task_name, model_version=bundle.model_version)
        task_dir = self.root / task_name
        path = task_dir / (bundle.model_version + ".bundle")
        save_bundle(bundle, path)
        index = self._read_index()
        task = index["models"].setdefault(
            task_name,
            {"versions": {}, "aliases": {}, "history": [], "audit": []},
        )
        task["versions"][bundle.model_version] = {
            "path": str(path.relative_to(self.root)),
            "state": "candidate",
            "created_at": bundle.created_at,
        }
        task["aliases"]["latest"] = bundle.model_version
        task["audit"].append({
            "event_type": "register",
            "task_name": task_name,
            "model_version": bundle.model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": {"state": "candidate"},
        })
        self._write_index(index)
        return bundle.model_version

    def promote(self, task_name, model_version, state):
        """Set a version state and update the matching lifecycle alias."""
        if state not in REGISTRY_STATES:
            raise ValueError("Unknown model registry state: %r" % state)
        index = self._read_index()
        task = index["models"].get(task_name)
        if task is None or model_version not in task["versions"]:
            raise KeyError(
                "Unknown model version %r for task %r: the bundle is not registered in this task's registry yet. Register it first (e.g. POST /api/registry/%s/register)."
                % (model_version, task_name, task_name)
            )
        previous = task["versions"][model_version]["state"]
        task["versions"][model_version]["state"] = state
        task["aliases"][state] = model_version
        task["history"].append({
            "event_type": "promote",
            "action": "promote",
            "from": previous,
            "to": state,
            "model_version": model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        task.setdefault("audit", []).append({
            "event_type": "promote",
            "action": "promote",
            "from": previous,
            "to": state,
            "model_version": model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._write_index(index)

    def record_approval(self, task_name, model_version, approver, *, state):
        """Persist an approval event for a pending lifecycle transition."""
        if not str(approver).strip():
            raise ValueError("approver must not be empty")
        index = self._read_index()
        task = index["models"].get(task_name)
        if task is None or model_version not in task["versions"]:
            raise KeyError(
                "Unknown model version %r for task %r: the bundle is not registered in this task's registry yet. Register it first (e.g. POST /api/registry/%s/register)."
                % (model_version, task_name, task_name)
            )
        event = {
            "event_type": "approval",
            "action": "approve",
            "state": state,
            "model_version": model_version,
            "approver": str(approver),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        task.setdefault("audit", []).append(event)
        self._write_index(index)
        return event

    def rollback(self, task_name, state, model_version=None):
        """Point a lifecycle alias at a prior version and record the rollback."""
        index = self._read_index()
        task = index["models"].get(task_name)
        if task is None:
            raise KeyError("Unknown model task: %r" % task_name)
        target = model_version or next(
            (item["model_version"] for item in reversed(task["history"])
             if item["action"] == "promote" and item["to"] == state),
            None,
        )
        if target is None or target not in task["versions"]:
            raise KeyError("No rollback target for state %r" % state)
        previous = task["aliases"].get(state)
        task["aliases"][state] = target
        task["history"].append({
            "event_type": "rollback",
            "action": "rollback",
            "state": state,
            "from": previous,
            "to": target,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        task.setdefault("audit", []).append({
            "event_type": "rollback",
            "action": "rollback",
            "state": state,
            "from": previous,
            "to": target,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._write_index(index)

    def audit_log(self, task_name):
        index = self._read_index()
        task = index["models"].get(task_name)
        if task is None:
            raise KeyError("Unknown model task: %r" % task_name)
        return list(task.get("audit", task.get("history", [])))

    def register_dataset(self, task_name, dataset_fingerprint, source_path, *, metadata=None):
        """Register a stored dataset artifact and keep retention metadata for the task."""
        index = self._read_index()
        task = index["models"].setdefault(
            task_name,
            {"versions": {}, "aliases": {}, "history": [], "audit": [], "datasets": {}, "predictions": {}},
        )
        dataset_dir = self.root / task_name / "datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_file = dataset_dir / (dataset_fingerprint + ".json")
        payload = {
            "task_name": task_name,
            "dataset_fingerprint": dataset_fingerprint,
            "source_path": str(source_path),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        with dataset_file.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
        task.setdefault("datasets", {})[dataset_fingerprint] = {
            "path": str(dataset_file.relative_to(self.root)),
            "source_path": str(source_path),
            "registered_at": payload["registered_at"],
            "metadata": payload["metadata"],
        }
        self._write_index(index)
        return dataset_file

    def register_prediction(self, task_name, model_version, dataset_fingerprint, source_path, *, prediction_id=None, metadata=None):
        """Register a stored prediction artifact keyed to a model and dataset."""
        index = self._read_index()
        task = index["models"].setdefault(
            task_name,
            {"versions": {}, "aliases": {}, "history": [], "audit": [], "datasets": {}, "predictions": {}},
        )
        prediction_id = prediction_id or uuid.uuid4().hex
        target_dir = self.root / task_name / "predictions" / model_version / dataset_fingerprint
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / (prediction_id + ".csv")
        source = Path(source_path)
        if source.exists() and source != target_path:
            shutil.copy2(str(source), str(target_path))
        else:
            target_path.write_text(str(source_path), encoding="utf-8")
        task.setdefault("predictions", {}).setdefault(model_version, {}).setdefault(dataset_fingerprint, {})[prediction_id] = {
            "path": str(target_path.relative_to(self.root)),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        self._write_index(index)
        return target_path

    def delete_artifacts(self, task_name, *, model_version=None, dataset_fingerprint=None):
        """Delete stored dataset and prediction artifacts by task, model, and dataset identity."""
        index = self._read_index()
        task = index["models"].get(task_name)
        if task is None:
            raise KeyError("Unknown model task: %r" % task_name)
        deleted = {"datasets": 0, "predictions": 0}

        datasets = task.get("datasets", {})
        if dataset_fingerprint is not None:
            entry = datasets.pop(dataset_fingerprint, None)
            if entry is not None:
                path = self.root / entry["path"]
                if path.exists():
                    path.unlink()
                deleted["datasets"] += 1
        else:
            for dataset_key in list(datasets):
                entry = datasets.pop(dataset_key, None)
                path = self.root / entry["path"]
                if path.exists():
                    path.unlink()
                deleted["datasets"] += 1

        predictions = task.get("predictions", {})
        if model_version is not None:
            model_predictions = predictions.get(model_version, {})
            if dataset_fingerprint is not None:
                dataset_predictions = model_predictions.pop(dataset_fingerprint, {})
                for prediction_key, entry in list(dataset_predictions.items()):
                    path = self.root / entry["path"]
                    if path.exists():
                        path.unlink()
                    deleted["predictions"] += 1
                if not model_predictions:
                    predictions.pop(model_version, None)
            else:
                for dataset_key in list(model_predictions):
                    dataset_predictions = model_predictions.pop(dataset_key, {})
                    for prediction_key, entry in list(dataset_predictions.items()):
                        path = self.root / entry["path"]
                        if path.exists():
                            path.unlink()
                        deleted["predictions"] += 1
                if not model_predictions:
                    predictions.pop(model_version, None)
        elif dataset_fingerprint is not None:
            for model_key, model_predictions in list(predictions.items()):
                dataset_predictions = model_predictions.pop(dataset_fingerprint, {})
                for prediction_key, entry in list(dataset_predictions.items()):
                    path = self.root / entry["path"]
                    if path.exists():
                        path.unlink()
                    deleted["predictions"] += 1
                if not model_predictions:
                    predictions.pop(model_key, None)
        else:
            for model_key, model_predictions in list(predictions.items()):
                for dataset_key in list(model_predictions):
                    dataset_predictions = model_predictions.pop(dataset_key, {})
                    for prediction_key, entry in list(dataset_predictions.items()):
                        path = self.root / entry["path"]
                        if path.exists():
                            path.unlink()
                        deleted["predictions"] += 1
                if not model_predictions:
                    predictions.pop(model_key, None)

        task["datasets"] = datasets
        task["predictions"] = predictions
        task.setdefault("audit", []).append({
            "event_type": "delete_artifacts",
            "task_name": task_name,
            "model_version": model_version,
            "dataset_fingerprint": dataset_fingerprint,
            "deleted": deleted,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._write_index(index)
        return deleted

    def artifacts(self, task_name):
        """List the dataset and prediction artifacts registered for a task.

        Returns a plain, JSON-serializable view with one entry per artifact
        (``kind`` is ``"dataset"`` or ``"prediction"``), each carrying the
        stored path, registration timestamp, free-form metadata and the
        on-disk size, so a retention/cleanup UI can show exactly what a call
        to :meth:`delete_artifacts` would remove. Raises KeyError if the task
        is unknown to the registry.
        """
        index = self._read_index()
        task = index["models"].get(task_name)
        if task is None:
            raise KeyError("Unknown model task: %r" % task_name)

        def _entry(kind, path, registered_at, metadata, **extra):
            absolute = self.root / path
            return {
                "kind": kind,
                "path": path,
                "exists": absolute.exists(),
                "size_bytes": absolute.stat().st_size if absolute.exists() else 0,
                "registered_at": registered_at,
                "metadata": metadata or {},
                **extra,
            }

        datasets = [
            _entry("dataset", entry["path"], entry.get("registered_at"), entry.get("metadata"),
                   dataset_fingerprint=fingerprint, source_path=entry.get("source_path"))
            for fingerprint, entry in sorted(task.get("datasets", {}).items())
        ]
        predictions = [
            _entry("prediction", entry["path"], entry.get("registered_at"), entry.get("metadata"),
                   model_version=model_version, dataset_fingerprint=dataset_fingerprint,
                   prediction_id=prediction_id)
            for model_version, by_dataset in sorted(task.get("predictions", {}).items())
            for dataset_fingerprint, by_id in sorted(by_dataset.items())
            for prediction_id, entry in sorted(by_id.items())
        ]
        return {"datasets": datasets, "predictions": predictions}

    def remove_task(self, task_name):
        """Delete a task's whole registry entry and stored versions.

        Used by the per-user data reset: the index entry and the task's registry
        directory must disappear together, or a later ``load`` would point at a
        bundle that no longer exists. Returns the removed version ids.
        """
        index = self._read_index()
        task = index["models"].pop(task_name, None)
        if task is None:
            return []
        self._write_index(index)
        shutil.rmtree(self.root / task_name, ignore_errors=True)
        return sorted(task.get("versions", {}))

    # -- user-chosen aliases -------------------------------------------
    #
    # ``aliases`` holds lifecycle pointers (``candidate``/``production``/...) that
    # promote/rollback rewrite automatically. Named aliases are a *separate* map
    # under ``named_aliases``: a stable, user-chosen handle ("baseline-2026-11")
    # for an immutable ``model_version``, so an experiment can be referred to and
    # displayed by a human name without touching the lifecycle semantics.

    @staticmethod
    def validate_alias_name(name):
        """Return ``name`` if it is a legal, non-reserved alias, else raise ValueError."""
        if not isinstance(name, str) or not ALIAS_NAME_PATTERN.match(name):
            raise ValueError(
                "Alias names must start with a lowercase letter or digit and use only "
                "'a-z', '0-9', '.', '_' or '-' (max 64 characters), e.g. 'rf-v1' or "
                "'baseline-2026-11'"
            )
        if name in LIFECYCLE_ALIAS_NAMES:
            raise ValueError("%r is a reserved lifecycle alias name" % name)
        return name

    def _task_or_raise(self, index, task_name):
        task = index["models"].get(task_name)
        if task is None:
            raise KeyError("Unknown model task: %r" % task_name)
        return task

    def named_aliases(self, task_name):
        """Return ``{name: {model_version, created_at, created_by}}`` for one task."""
        index = self._read_index()
        task = self._task_or_raise(index, task_name)
        return {
            name: dict(entry) for name, entry in task.get("named_aliases", {}).items()
        }

    def resolve_version(self, task_name, identifier):
        """Resolve an exact version id *or* a named alias to a version id.

        Exact ids win, so a version whose hex id happens to match an alias name
        is never shadowed. Raises KeyError when neither resolves.
        """
        index = self._read_index()
        task = self._task_or_raise(index, task_name)
        if identifier in task.get("versions", {}):
            return identifier
        entry = task.get("named_aliases", {}).get(identifier)
        if entry is not None:
            return entry["model_version"]
        raise KeyError(
            "Unknown model version or alias %r for task %r: check the registry's named aliases, and register a bundle first (e.g. POST /api/registry/%s/register) if this task has no registered versions yet"
            % (identifier, task_name, task_name)
        )

    def set_named_alias(self, task_name, name, model_version, *, actor=None):
        """Point a user-chosen alias at a registered version and record it in the audit log."""
        self.validate_alias_name(name)
        index = self._read_index()
        task = self._task_or_raise(index, task_name)
        if model_version not in task.get("versions", {}):
            raise KeyError(
                "Unknown model version %r for task %r: the bundle is not registered in this task's registry yet. Register it first (e.g. POST /api/registry/%s/register) to make its version addressable by alias."
                % (model_version, task_name, task_name)
            )
        timestamp = datetime.now(timezone.utc).isoformat()
        previous = task.setdefault("named_aliases", {}).get(name)
        task["named_aliases"][name] = {
            "model_version": model_version,
            "created_at": timestamp,
            "created_by": actor,
        }
        event = {
            "event_type": "alias_set",
            "action": "alias_set",
            "alias": name,
            "model_version": model_version,
            "previous_model_version": previous.get("model_version") if previous else None,
            "actor": actor,
            "timestamp": timestamp,
        }
        task["history"].append(event)
        task.setdefault("audit", []).append(dict(event))
        self._write_index(index)
        return task["named_aliases"][name]

    def delete_named_alias(self, task_name, name):
        """Remove a user-chosen alias (the version it pointed at is left untouched)."""
        index = self._read_index()
        task = self._task_or_raise(index, task_name)
        removed = task.setdefault("named_aliases", {}).pop(name, None)
        if removed is None:
            raise KeyError("Unknown model alias %r for task %r" % (name, task_name))
        event = {
            "event_type": "alias_deleted",
            "action": "alias_deleted",
            "alias": name,
            "model_version": removed.get("model_version"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        task["history"].append(event)
        task.setdefault("audit", []).append(dict(event))
        self._write_index(index)
        return removed

    def load(self, task_name, alias="production", **kwargs):
        index = self._read_index()
        task = index["models"].get(task_name)
        if task is None:
            raise KeyError("Unknown model alias %r for task %r" % (alias, task_name))
        if alias in task["aliases"]:
            version = task["aliases"][alias]
        elif alias in task.get("named_aliases", {}):
            version = task["named_aliases"][alias]["model_version"]
        else:
            raise KeyError("Unknown model alias %r for task %r" % (alias, task_name))
        return load_bundle(self.root / task["versions"][version]["path"], **kwargs)

    def load_version(self, task_name, model_version, **kwargs):
        """Load a registered version by exact id or by user-chosen alias."""
        index = self._read_index()
        task = self._task_or_raise(index, task_name)
        if model_version not in task.get("versions", {}):
            entry = task.get("named_aliases", {}).get(model_version)
            if entry is None:
                raise KeyError(
                    "Unknown model version %r for task %r" % (model_version, task_name)
                )
            model_version = entry["model_version"]
        path = self.root / task["versions"][model_version]["path"]
        return load_bundle(path, **kwargs)

    def history(self, task_name):
        index = self._read_index()
        task = index["models"].get(task_name)
        if task is None:
            raise KeyError("Unknown model task: %r" % task_name)
        return list(task.get("history", []))

    def version_state(self, task_name, model_version):
        """Return the current lifecycle state of a registered version.

        The authoritative value lives in ``versions`` (set by register/promote);
        promotion history alone is empty for freshly-registered candidates.
        Raises KeyError if the task or version is unknown to the registry.
        """
        index = self._read_index()
        task = index["models"].get(task_name)
        if task is None:
            raise KeyError("Unknown model task: %r" % task_name)
        versions = task.get("versions", {})
        if model_version not in versions:
            raise KeyError("Unknown model version %r for task %r" % (model_version, task_name))
        return versions[model_version]["state"]

    def summary(self, task_name):
        """Return the registry state of one task as a plain dict.

        Includes every lifecycle alias (``latest`` plus any promoted states)
        and the current state of each registered version, so clients can
        render the full lifecycle in a single round-trip instead of probing
        aliases one by one. Raises KeyError if the task is unknown.
        """
        index = self._read_index()
        task = index["models"].get(task_name)
        if task is None:
            raise KeyError("Unknown model task: %r" % task_name)
        return {
            "aliases": dict(task.get("aliases", {})),
            "named_aliases": {
                name: dict(entry) for name, entry in task.get("named_aliases", {}).items()
            },
            "versions": {v: meta.get("state") for v, meta in task.get("versions", {}).items()},
        }