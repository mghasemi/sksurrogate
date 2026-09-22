"""Portable model bundles and a small filesystem-backed model registry."""

import importlib.metadata
import json
import os
import platform
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib


BUNDLE_FORMAT_VERSION = 1
REGISTRY_STATES = {"candidate", "validated", "staging", "production", "archived"}


def _dependencies():
    dependencies = {}
    for package in ("numpy", "pandas", "scikit-learn", "SKSurrogate"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return dependencies


class ModelBundle:
    """A fitted estimator and the metadata needed to reproduce its inputs."""

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
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_record(self):
        return {
            "format_version": BUNDLE_FORMAT_VERSION,
            "model_version": self.model_version,
            "created_at": self.created_at,
            "task_name": self.task_name,
            "model": self.model,
            "preprocessing": self.preprocessing,
            "schema": self.schema,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_version": self.config_version,
            "dependencies": self.dependencies,
        }

    @classmethod
    def from_record(cls, record):
        if record.get("format_version") != BUNDLE_FORMAT_VERSION:
            raise ValueError(
                "Unsupported model bundle format: %r" % record.get("format_version")
            )
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
        )
        bundle.created_at = record.get("created_at", bundle.created_at)
        return bundle

    def predict(self, *args, **kwargs):
        """Delegate prediction to the fitted model in the bundle."""
        return self.model.predict(*args, **kwargs)


def save_bundle(bundle, path):
    """Atomically write a :class:`ModelBundle` and return its path."""
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
    return destination


def load_bundle(path, *, expected_schema=None, strict_dependencies=True):
    """Load a bundle and reject incompatible schema or runtime dependencies."""
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
        task_dir = self.root / task_name
        path = task_dir / (bundle.model_version + ".bundle")
        save_bundle(bundle, path)
        index = self._read_index()
        task = index["models"].setdefault(task_name, {"versions": {}, "aliases": {}, "history": []})
        task["versions"][bundle.model_version] = {
            "path": str(path.relative_to(self.root)),
            "state": "candidate",
            "created_at": bundle.created_at,
        }
        task["aliases"]["latest"] = bundle.model_version
        self._write_index(index)
        return bundle.model_version

    def promote(self, task_name, model_version, state):
        """Set a version state and update the matching lifecycle alias."""
        if state not in REGISTRY_STATES:
            raise ValueError("Unknown model registry state: %r" % state)
        index = self._read_index()
        task = index["models"].get(task_name)
        if task is None or model_version not in task["versions"]:
            raise KeyError("Unknown model version %r for task %r" % (model_version, task_name))
        previous = task["versions"][model_version]["state"]
        task["versions"][model_version]["state"] = state
        task["aliases"][state] = model_version
        task["history"].append({
            "action": "promote",
            "from": previous,
            "to": state,
            "model_version": model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._write_index(index)

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
            "action": "rollback",
            "state": state,
            "from": previous,
            "to": target,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._write_index(index)

    def load(self, task_name, alias="production", **kwargs):
        index = self._read_index()
        task = index["models"].get(task_name)
        if task is None or alias not in task["aliases"]:
            raise KeyError("Unknown model alias %r for task %r" % (alias, task_name))
        version = task["aliases"][alias]
        return load_bundle(self.root / task["versions"][version]["path"], **kwargs)

    def history(self, task_name):
        index = self._read_index()
        task = index["models"].get(task_name)
        if task is None:
            raise KeyError("Unknown model task: %r" % task_name)
        return list(task["history"])