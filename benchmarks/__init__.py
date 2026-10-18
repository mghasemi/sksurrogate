"""Reproducible benchmark suite for SKSurrogate (see DESIGN.md).

This package is tooling, not part of the runtime library: it compares
SKSurrogate's surrogate-guided search against baselines under equal
evaluation budgets. Installing ``pip install sksurrogate[benchmark]``
pulls the optional dependencies (openml, optuna) needed here.
"""
