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


Description = "A python package that implements or wraps a few machine learning tools Scikit-Learn style."

setup(
    name="SKSurrogate",
    # Single source of truth: SKSurrogate.__version__ (kept equal to the
    # GitHub release tag on every upload).
    version=_read_version(),
    author="Mehdi Ghasemi",
    author_email="mehdi.ghasemi@gmail.com",
    packages=["SKSurrogate"],
    url="https://github.com/mghasemi/sksurrogate.git",
    license="MIT",
    description=Description,
    long_description=open("./readme.rst").read(),
    long_description_content_type="text/x-rst",
    keywords=[
        "Auto Machine Learning",
        "AutoML",
        "Optimization",
        "Surrogate Optimization",
        "Numerical",
        "Machine Learning",
        "Regression",
        "Random Search",
    ],
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
