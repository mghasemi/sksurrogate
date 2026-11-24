import os
import re

from setuptools import setup


def _read_version():
    init_py = os.path.join(os.path.dirname(__file__), "SKSurrogate", "__init__.py")
    with open(init_py, encoding="utf-8") as f:
        match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', f.read(), re.MULTILINE)
    if match is None:
        raise RuntimeError("Cannot determine version: __version__ missing in SKSurrogate/__init__.py")
    return match.group(1)


Description = (
    "scikit-learn-native surrogate tuner for hyperparameter and AutoML pipeline search; "
    "a drop-in replacement for BayesSearchCV, with a small core install and opt-in extras."
)

# --------------------------------------------------------------------------
# Dependency tiers
# --------------------------------------------------------------------------
# CORE is exactly what ``import SKSurrogate`` needs. Verified with
# ``python -X importtime -c "import SKSurrogate"``: the only third-party
# modules imported at module scope anywhere in SKSurrogate/ are numpy, scipy,
# pandas, joblib, scikit-learn and peewee. Everything else the library can use
# (gradient-boosting backends, SALib, eli5, skrebate, matplotlib, dask,
# category_encoders, imbalanced-learn) is imported lazily inside functions, so
# it belongs in an extra.
#
# Rule when adding a dependency: a module-scope import goes in CORE; a lazy
# import goes in an extra below *and* its call site is wrapped with
# ``SKSurrogate.deps.require`` so that a missing package raises an error naming
# the extra that provides it.
_CORE = [
    "numpy>=1.22",
    "scipy>=1.10",
    "pandas",
    "scikit-learn>=1.5",
    "joblib>=1.2",
    "peewee",
    "tqdm>=4.66.3",
]

# Estimator backends. Catalog rows such as ``lightgbm.LGBMClassifier`` are
# resolved through importlib when a search space is built, so these are optional
# until a user actually names one of them.
_LIGHTGBM = ["lightgbm>=4.6"]
_XGBOOST = ["xgboost>=2.1"]
# rgf-python ships wheels only for manylinux1/macOS x86_64 and win_amd64 (last
# release 3.12.0, 2022). On Linux/aarch64 and Apple Silicon pip falls back to
# the sdist, which needs a C++ toolchain — keep it an explicit opt-in.
_RGF = ["rgf-python>=3.12.0"]

_ESTIMATORS = _LIGHTGBM + _XGBOOST + _RGF + ["category-encoders"]
_SENSITIVITY = ["SALib"]
_EXPLAIN = ["eli5", "skrebate"]
_IMBALANCE = ["imbalanced-learn"]
_PLOTS = ["matplotlib"]
_DASK = ["dask[distributed]>=2024.1"]
_API = ["fastapi>0.109", "uvicorn[standard]>=0.11.7", "python-multipart>=0.0.31"]
_BENCHMARK = ["openml>=0.15,<0.16", "optuna", "scikit-posthocs>=0.9"]

setup(
    name="SKSurrogate",
    # Single source of truth: SKSurrogate.__version__ (kept equal to the
    # GitHub release tag on every upload).
    version=_read_version(),
    author="Mehdi Ghasemi",
    author_email="mehdi.ghasemi@gmail.com",
    packages=["SKSurrogate", "sksurrogate_mcp"],
    # The MCP server validates ``api_request`` against this generated snapshot;
    # without it in the wheel the escape hatch degrades to an error payload.
    # Pinned by the install-smoke CI job (reads it via importlib.resources).
    package_data={"sksurrogate_mcp": ["openapi_ops.json"]},
    url="https://github.com/mghasemi/sksurrogate",
    license="MIT",
    # Ship the single authoritative licence file explicitly (setuptools also
    # auto-detects LICENSE, but naming it keeps the sdist deterministic).
    license_files=["LICENSE"],
    description=Description,
    long_description=open("./readme.rst").read(),
    long_description_content_type="text/x-rst",
    keywords=[
        "hyperparameter optimization",
        "surrogate",
        "scikit-learn",
        "AutoML",
        "Bayesian optimization",
        "trust region",
        "evolutionary optimization",
        "drift detection",
        "Auto Machine Learning",
        "Optimization",
        "Surrogate Optimization",
        "Machine Learning",
        "Regression",
        "Random Search",
    ],
    project_urls={
        "Source": "https://github.com/mghasemi/sksurrogate",
        "Documentation": "https://sksurrogate.readthedocs.io/",
        "Zenodo (software)": "https://doi.org/10.5281/zenodo.23631823",
        "Zenodo (benchmark data)": "https://doi.org/10.5281/zenodo.23629024",
    },
    install_requires=list(_CORE),
    extras_require={
        # ---- estimator backends ------------------------------------------
        "lightgbm": list(_LIGHTGBM),
        "xgboost": list(_XGBOOST),
        "rgf": list(_RGF),
        "gbm": _LIGHTGBM + _XGBOOST,
        "estimators": list(_ESTIMATORS),
        # ---- feature support ---------------------------------------------
        "sensitivity": list(_SENSITIVITY),
        "explain": list(_EXPLAIN),
        "imbalance": list(_IMBALANCE),
        "categorical": ["category-encoders"],
        "plots": list(_PLOTS),
        # ---- serving / operations ----------------------------------------
        # The API's optional distributed execution backend (api/jobs.py and
        # SKSurrogate.execution). The control plane degrades gracefully to the
        # local backend when it is absent.
        "dask": list(_DASK),
        "api": list(_API),
        # The web UI is a static bundle served by the API, so this extra only
        # needs the API stack.
        "ui": list(_API),
        # ---- studies -----------------------------------------------------
        "benchmark": list(_BENCHMARK),
        # Wall-clock AutoML arm (benchmarks/automl_arm.py). Separate extra: these
        # are large and irrelevant to the tuner itself.
        "automl": ["autogluon.tabular", "flaml[automl]"],
        # Time-series/forecasting support (SKSurrogate.temporal, forecast_metrics,
        # forecast_intervals) needs no package beyond the core requirements —
        # numpy/pandas/scikit-learn already are dependencies. The extra exists so
        # `pip install ".[forecasting]"` is a valid, documented install and so a
        # future optional adapter (e.g. a statsmodels/exogenous bridge) has a
        # home without changing what the base install pulls in. `import sksurrogate`
        # works identically with and without it.
        "forecasting": [],
        # ---- agent integration ---------------------------------------------
        # Agent-facing MCP server (sksurrogate_mcp): an external wrapper over
        # the HTTP control plane — zero core dependencies, stdio transport.
        # Verified against MCP Python SDK 2.x (FastMCP was renamed MCPServer in
        # 2.0; the floor keeps a single import path). The <3 bound is
        # deliberate: the server imports mcp.server.mcpserver.MCPServer, so a
        # future major rename would need a code change, not a silent upgrade.
        "mcp": ["httpx>=0.27", "mcp>=2.3,<3"],
        # ---- developer / CI ----------------------------------------------
        "test": ["pytest", "pygments", "hypothesis>=6.100"],
        # Everything CI and the Docker image run, minus the deliberately
        # enormous `automl` arm (AutoGluon + PyTorch).
        "all": (
            _ESTIMATORS
            + _SENSITIVITY
            + _EXPLAIN
            + _IMBALANCE
            + _PLOTS
            + _DASK
            + _API
            + _BENCHMARK
        ),
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
    ],
    entry_points={
        "console_scripts": [
            "sksurrogate-batch-predict=SKSurrogate.inference:main",
            # MCP server for agents (stdio transport); needs the "mcp" extra.
            "sksurrogate-mcp=sksurrogate_mcp.server:main",
        ],
    },
)
