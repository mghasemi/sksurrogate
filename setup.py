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
    "Lightweight, maintained, scikit-learn-native surrogate tuner for hyperparameter and "
    "AutoML pipeline search; a drop-in replacement for BayesSearchCV."
)

setup(
    name="SKSurrogate",
    # Single source of truth: SKSurrogate.__version__ (kept equal to the
    # GitHub release tag on every upload).
    version=_read_version(),
    author="Mehdi Ghasemi",
    author_email="mehdi.ghasemi@gmail.com",
    packages=["SKSurrogate"],
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
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.10",
        "pandas",
        "matplotlib",
        "scikit-learn>=1.5",
        "lightgbm>=4.6",
        "xgboost>=2.1",
        "rgf-python>=3.12.0",
        "SALib",
        "imbalanced-learn",
        "joblib>=1.2",
        "eli5",
        "skrebate",
        "tqdm>=4.66.3",
        "peewee",
        "category_encoders",
    ],
    extras_require={
        "test": ["pytest", "pygments"],
        # The API's optional distributed execution backend (api/jobs.py and
        # SKSurrogate.execution). The control plane degrades gracefully to the
        # local backend when it is absent.
        "dask": ["dask[distributed]>=2024.1"],
        "benchmark": ["openml>=0.15,<0.16", "optuna", "scikit-posthocs>=0.9"],
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
        ],
    },
)
