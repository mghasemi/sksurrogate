"""Optional-dependency resolution with actionable error messages.

The base install of SKSurrogate is deliberately small: only the modules that
``import SKSurrogate`` imports at module scope (numpy, scipy, pandas, joblib,
scikit-learn, peewee) are required. Everything else is imported lazily, inside
the code path that needs it, and is installed through an extra such as
``pip install "sksurrogate[estimators]"``.

This module turns a bare ``ModuleNotFoundError`` at that lazy import into an
error that names the missing distribution *and* the extra that provides it::

    >>> from SKSurrogate.deps import require
    >>> xgb = require("xgboost")          # doctest: +SKIP
    Traceback (most recent call last):
    ...
    SKSurrogate.deps.MissingOptionalDependency: SKSurrogate needs the optional
    package 'xgboost' for the XGBoost estimator, but it is not installed.
    Install it with: pip install "sksurrogate[xgboost]"

Two entry points are provided:

``require(module, purpose=...)``
    Import an optional module or raise :class:`MissingOptionalDependency`.

``is_available(module)``
    Non-raising probe, used by the component catalog to flag rows whose backend
    is not installed so the web UI can grey them out instead of letting a job
    fail at fit time.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

__all__ = [
    "EXTRA_FOR_MODULE",
    "MissingOptionalDependency",
    "extra_for",
    "is_available",
    "require",
]

#: Mapping from importable top-level module name to the extra that provides it.
#: Keep in sync with ``extras_require`` in ``setup.py``.
EXTRA_FOR_MODULE: dict[str, str] = {
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "rgf": "rgf",
    "category_encoders": "categorical",
    "SALib": "sensitivity",
    "eli5": "explain",
    "skrebate": "explain",
    "imblearn": "imbalance",
    "matplotlib": "plots",
    "dask": "dask",
    "distributed": "dask",
    "fastapi": "api",
    "uvicorn": "api",
    "multipart": "api",
    "openml": "benchmark",
    "optuna": "benchmark",
    "scikit_posthocs": "benchmark",
}

_INSTALL_HINT = 'pip install "sksurrogate[{extra}]"'


class MissingOptionalDependency(ImportError):
    """An optional package is required by the requested feature but absent.

    Subclasses :class:`ImportError` so existing ``except ImportError`` handlers
    keep working, while ``str(exc)`` explains how to fix the situation.
    """

    def __init__(self, module: str, extra: str | None, purpose: str = "", requirement: str | None = None):
        self.module = module
        self.extra = extra
        self.purpose = purpose
        root = module.split(".")[0]
        self.requirement = requirement or root
        for_what = f" for {purpose}" if purpose else ""
        message = (
            f"SKSurrogate needs the optional package {self.requirement!r}{for_what}, "
            "but it is not installed."
        )
        if extra:
            message += f"\nInstall it with: {_INSTALL_HINT.format(extra=extra)}"
        else:
            message += f"\nInstall it with: pip install {self.requirement}"
        message += "\nOr install every supported backend at once: pip install \"sksurrogate[all]\""
        super().__init__(message)


def extra_for(module: str) -> str | None:
    """Return the extra that provides ``module``, or ``None`` if unmapped."""
    return EXTRA_FOR_MODULE.get(module.split(".")[0])


def is_available(module: str) -> bool:
    """Return whether ``module`` can be imported, without importing it.

    ``importlib.util.find_spec`` raises ``ModuleNotFoundError`` (rather than
    returning ``None``) when a *parent* of a dotted name is missing, so that is
    handled here and reported as "not available".
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # ValueError: module in sys.modules = None
        return False


def require(module: str, *, purpose: str = "", requirement: str | None = None) -> Any:
    """Import and return ``module``, or raise :class:`MissingOptionalDependency`.

    Only a *missing* module is translated; an import that fails for another
    reason (a broken install, an ABI mismatch inside an installed package) is
    re-raised untouched, because "install the extra" would be wrong advice.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0]
        if missing not in _known_roots(module):
            raise
        raise MissingOptionalDependency(
            module, extra_for(missing or module), purpose=purpose, requirement=requirement
        ) from exc


def _known_roots(module: str) -> set[str]:
    """Names a failed import of ``module`` may legitimately report as missing.

    A lazy import of ``dask.distributed`` fails with ``ModuleNotFoundError``
    naming ``distributed``, not ``dask``; both map to the ``dask`` extra. So any
    component of the dotted path counts, plus any mapped prefix of it.
    """
    parts = module.split(".")
    return {parts[0], *parts} | {key for key in EXTRA_FOR_MODULE if module.startswith(key)}
