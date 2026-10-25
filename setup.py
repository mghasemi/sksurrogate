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
        "benchmark": ["openml>=0.15,<0.16", "optuna"],
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
