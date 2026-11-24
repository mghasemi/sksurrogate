"""Paired statistics for benchmark comparisons — re-export of the core.

This module used to carry its own copy of the paired statistics. It is now a
thin re-export of :mod:`SKSurrogate.compare`, which is the single implementation
used by both the control plane (``api.routers.comparison``) and this study code,
so the product and the benchmark study can no longer drift apart. The function
bodies here were verified AST-identical to the core before the copy was removed.

The statistical contract has not moved and has not weakened. It lives in the
module docstring of :mod:`SKSurrogate.compare`, and every exported function
carries it: paired-only inputs (the unit of inference is the per-pair
difference, never the marginal distributions), percentile bootstrap intervals
that are *not* Bayesian posterior intervals, Holm step-down across the family of
method-versus-control comparisons, Cliff's delta as an *ordinal* effect size
rather than a mean difference, and the pairwise-NaN policy. Read it before
interpreting a report row: the numbers are only as meaningful as the alignment
of the arrays that produced them.

Depends only on NumPy and SciPy (through the core module), never on
scikit-learn.
"""

from __future__ import annotations

from SKSurrogate.compare import (
    bootstrap_ci,
    bootstrap_paired_ci,
    cliffs_delta,
    holm_correction,
    paired_deltas,
    paired_report_row,
    wilcoxon_holm,
    wilcoxon_pvalues,
)

__all__ = [
    "bootstrap_ci",
    "bootstrap_paired_ci",
    "cliffs_delta",
    "holm_correction",
    "paired_deltas",
    "paired_report_row",
    "wilcoxon_holm",
    "wilcoxon_pvalues",
]
