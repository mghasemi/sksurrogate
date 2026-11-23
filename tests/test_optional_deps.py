"""Tests for the optional-dependency layer (``SKSurrogate.deps``).

The base install no longer carries xgboost / lightgbm / rgf / SALib / eli5 /
skrebate / matplotlib / imbalanced-learn / category_encoders: they are extras,
resolved lazily. These tests pin down the two guarantees that make that safe:

1. The error a user sees names the missing package *and* the extra that
   installs it (``pip install "sksurrogate[xgboost]"``).
2. Only a genuinely missing module is translated — a package that is installed
   but broken still surfaces its own error.

They also pin the catalog's availability annotation, which the web UI uses to
disable rows whose backend is absent.
"""

from __future__ import annotations

import importlib

import pytest

from SKSurrogate import deps
from SKSurrogate.deps import (
    EXTRA_FOR_MODULE,
    MissingOptionalDependency,
    extra_for,
    is_available,
    require,
)


# --------------------------------------------------------------------------
# mapping / probing
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "module,extra",
    [
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("rgf.sklearn", "rgf"),
        ("SALib.sample", "sensitivity"),
        ("eli5.sklearn", "explain"),
        ("skrebate", "explain"),
        ("imblearn.pipeline", "imbalance"),
        ("category_encoders.one_hot", "categorical"),
        ("matplotlib.pyplot", "plots"),
        ("dask.distributed", "dask"),
        ("distributed", "dask"),
    ],
)
def test_extra_for_maps_module_to_extra(module, extra):
    assert extra_for(module) == extra


def test_extra_for_unknown_module_is_none():
    assert extra_for("os") is None
    assert extra_for("pandas") is None


def test_every_mapped_extra_exists_in_metadata():
    """The mapping must not name an extra that setup.py does not define.

    Parsed from ``setup.py`` rather than from installed metadata, because the
    tests can run from a source tree whose distribution predates the extras.
    """
    import pathlib
    import re

    setup_py = pathlib.Path(__file__).resolve().parents[1] / "setup.py"
    text = setup_py.read_text(encoding="utf-8")
    block = text.split("extras_require=", 1)[1].split("python_requires=", 1)[0]
    declared = set(re.findall(r'^\s*"([a-z0-9\-]+)":', block, re.MULTILINE))
    declared |= {"all"}  # built from the computed union, so not a literal key
    assert set(EXTRA_FOR_MODULE.values()) <= declared


def test_is_available_true_for_core_module():
    assert is_available("sklearn") is True
    assert is_available("numpy") is True


def test_is_available_false_for_missing_module():
    assert is_available("sksurrogate_not_a_real_module") is False


def test_is_available_false_for_missing_parent():
    """find_spec raises for a missing parent; the probe must not."""
    assert is_available("sksurrogate_not_a_real_module.submodule") is False


# --------------------------------------------------------------------------
# require()
# --------------------------------------------------------------------------
def test_require_returns_imported_module():
    assert require("json").dumps({"a": 1}) == '{"a": 1}'


def test_require_names_the_extra(tmp_path, monkeypatch):
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name.startswith("xgboost"):
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(deps.importlib, "import_module", fake_import)

    with pytest.raises(MissingOptionalDependency) as excinfo:
        require("xgboost", purpose="the XGBoost estimator")

    message = str(excinfo.value)
    assert "xgboost" in message
    assert "the XGBoost estimator" in message
    assert 'pip install "sksurrogate[xgboost]"' in message
    assert 'pip install "sksurrogate[all]"' in message
    # Still an ImportError, so existing handlers keep working.
    assert isinstance(excinfo.value, ImportError)


def test_require_translates_submodule_failure_to_its_extra(monkeypatch):
    """``import dask.distributed`` fails naming ``distributed``; that is dask's extra."""
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "dask.distributed":
            raise ModuleNotFoundError("No module named 'distributed'", name="distributed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(deps.importlib, "import_module", fake_import)
    with pytest.raises(MissingOptionalDependency) as excinfo:
        require("dask.distributed", purpose="the Dask execution backend")
    assert 'sksurrogate[dask]' in str(excinfo.value)


def test_require_propagates_broken_install(monkeypatch):
    """An installed-but-broken package must not be reported as 'not installed'."""

    def fake_import(name, *args, **kwargs):
        raise ImportError("cannot import name 'X' from 'y' (broken build)")

    monkeypatch.setattr(deps.importlib, "import_module", fake_import)
    with pytest.raises(ImportError) as excinfo:
        require("broken_pkg")
    assert not isinstance(excinfo.value, MissingOptionalDependency)


# --------------------------------------------------------------------------
# integration: the library surfaces the same message
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "module,attribute",
    [
        ("SALib.sample.saltelli", "sample"),
        ("SALib.sample.morris", "sample"),
        ("SALib.sample.latin", "sample"),
        ("SALib.analyze.sobol", "analyze"),
        ("SALib.analyze.morris", "analyze"),
        ("SALib.analyze.delta", "analyze"),
    ],
)
def test_sensapprx_lazy_salib_imports_resolve(module, attribute):
    """SensAprx resolves SALib *submodules*, not attributes of the package.

    ``from SALib.sample import saltelli`` imports ``SALib.sample.saltelli``;
    ``require("SALib.sample").saltelli`` does not exist. This pins the shorter
    path so a refactor cannot reintroduce an AttributeError at fit time.
    """
    pytest.importorskip("SALib")
    imported = require(module, purpose="sensitivity analysis")
    assert hasattr(imported, attribute)


def test_aml_get_class_names_the_extra(monkeypatch):
    from SKSurrogate.aml import AML

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name.startswith("lightgbm"):
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(deps.importlib, "import_module", fake_import)
    with pytest.raises(MissingOptionalDependency) as excinfo:
        AML._get_class("lightgbm.LGBMClassifier")
    assert 'pip install "sksurrogate[lightgbm]"' in str(excinfo.value)


def test_aml_types_names_the_extra(monkeypatch):
    """``AML.types()`` resolves catalog FQNs; a missing backend is explained."""
    from SKSurrogate.aml import AML

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name.startswith("rgf"):
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(deps.importlib, "import_module", fake_import)

    aml = AML(config={"sklearn.ensemble.RandomForestClassifier": {"n_estimators": 10}})
    aml.config["rgf.sklearn.RGFClassifier"] = {"max_leaf": 50}
    with pytest.raises(MissingOptionalDependency) as excinfo:
        aml.types()
    assert 'sksurrogate[rgf]' in str(excinfo.value)


# --------------------------------------------------------------------------
# catalog annotation
# --------------------------------------------------------------------------
def test_catalog_annotation_marks_core_available():
    from api.experiment_catalog import annotate_availability

    rows = {row["fqn"]: row for row in annotate_availability()}
    lr = rows["sklearn.linear_model.LogisticRegression"]
    assert lr["available"] is True
    assert lr["install_extra"] is None


def test_catalog_annotation_flags_missing_backend(monkeypatch):
    import api.experiment_catalog as catalog

    real = catalog.is_available
    monkeypatch.setattr(
        catalog, "is_available", lambda module: False if module == "rgf" else real(module)
    )
    rows = {row["fqn"]: row for row in catalog.annotate_availability()}
    assert rows["rgf.sklearn.RGFClassifier"]["available"] is False
    assert rows["rgf.sklearn.RGFClassifier"]["install_extra"] == "rgf"
    assert rows["sklearn.linear_model.LogisticRegression"]["available"] is True


def test_catalog_source_rows_are_not_mutated():
    """annotate_availability copies; the module-level table stays plain."""
    from api.experiment_catalog import EXPERIMENT_COMPONENTS, annotate_availability

    annotate_availability()
    assert all("available" not in row for row in EXPERIMENT_COMPONENTS)
