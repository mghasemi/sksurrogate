"""Quality gates for validating model bundles in CI pipelines."""

from pathlib import Path
from numbers import Real

from .modelbundle import ModelBundle, load_bundle


class BundleQualityGateError(ValueError):
    """Raised when a model bundle fails one or more CI quality gates."""


def _load_bundle(bundle_or_path, expected_schema):
    if isinstance(bundle_or_path, ModelBundle):
        if expected_schema is not None and bundle_or_path.schema != expected_schema:
            raise ValueError("Model bundle schema is incompatible with the expected schema")
        return bundle_or_path, None
    path = Path(bundle_or_path)
    return load_bundle(path, expected_schema=expected_schema, strict_dependencies=False), path


def check_bundle_quality(
    bundle_or_path,
    *,
    expected_schema=None,
    metric_thresholds=None,
    max_bundle_size_bytes=None,
):
    """Return structured CI checks for a model bundle.

    ``metric_thresholds`` maps metric names to minimum accepted numeric values.
    A bundle path is required when ``max_bundle_size_bytes`` is configured.
    """
    checks = []
    failures = []
    try:
        bundle, path = _load_bundle(bundle_or_path, expected_schema)
        checks.append({"name": "schema_compatibility", "passed": True})
    except (OSError, TypeError, ValueError) as exc:
        checks.append({"name": "schema_compatibility", "passed": False, "error": str(exc)})
        failures.append(checks[-1])
        return {"passed": False, "checks": checks, "failures": failures}

    for metric_name, minimum in (metric_thresholds or {}).items():
        value = bundle.metrics.get(metric_name)
        passed = isinstance(value, Real) and not isinstance(value, bool) and value >= minimum
        check = {
            "name": "metric_threshold",
            "metric": metric_name,
            "minimum": minimum,
            "value": value,
            "passed": passed,
        }
        checks.append(check)
        if not passed:
            failures.append(check)

    if max_bundle_size_bytes is not None:
        size = path.stat().st_size if path is not None else None
        passed = size is not None and size <= max_bundle_size_bytes
        check = {
            "name": "bundle_size",
            "maximum": max_bundle_size_bytes,
            "value": size,
            "passed": passed,
        }
        checks.append(check)
        if not passed:
            failures.append(check)

    return {"passed": not failures, "checks": checks, "failures": failures}


def assert_bundle_quality(
    bundle_or_path,
    *,
    expected_schema=None,
    metric_thresholds=None,
    max_bundle_size_bytes=None,
):
    """Raise :class:`BundleQualityGateError` when CI checks fail."""
    report = check_bundle_quality(
        bundle_or_path,
        expected_schema=expected_schema,
        metric_thresholds=metric_thresholds,
        max_bundle_size_bytes=max_bundle_size_bytes,
    )
    if not report["passed"]:
        raise BundleQualityGateError("Model bundle failed quality gates: %s" % report["failures"])
    return report


__all__ = ["BundleQualityGateError", "assert_bundle_quality", "check_bundle_quality"]