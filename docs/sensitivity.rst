==============================
Sensitivity Analysis
==============================
This module could be easily taken off of this library. But one could take advantage of sensitivity
analysis to reduce the complexity of expensive models. Moreover, observations in practice shows
significant gains in performance by employing data preprocessing based on sensitivity analysis.

Sensitivity analysis is defined as the study of how the uncertainty in the output of a model can be
apportioned to sources of uncertainty in inputs.

Given a model :math:`y=f(x_1,\dots,x_n)`, the following are standard sensitivity measures quantifying
sensitivity of the model with respect to :math:`x_1,\dots,x_n`:

Morris
===========================================
The **Morris** method facilitates a global sensitivity analysis by making a number of local changes at
different points of the possible range of input values. The following quantities are usually
measured regarding Morris method:

.. math::
    \mu_i=\int\frac{\partial f}{\partial x_i}dx_1\cdots dx_n,\\
    \mu^*_i=\int|\frac{\partial f}{\partial x_i}|dx_1\cdots dx_n,\\
    \sigma_i=Var(\frac{\partial f}{\partial x_i}).

Generally, :math:`\mu^*` is used to detect input factors with an important overall influence
on the output. :math:`\sigma` is used to detect factors involved in interaction with other factors
or whose effect is non-linear.

Sobol
===========================================
The **Sobol** method (aka variance-based sensitivity analysis) works by decomposing the variance of
the output of the model into fractions which can be attributed to inputs or sets of inputs.
The first-order indices are defined as:

.. math::
    S_i=\frac{D_i(y)}{Var(y)},\quad S_{ij}=\frac{D_{ij}(y)}{Var(y)},\dots

where

.. math::
    D_i(y)=Var_{x_i}(E_{x_{-i}}(y|x_i)), ~
    D_{ij}(y)=Var_{x_{ij}}(E_{x_{-ij}}(y|x_i,x_j))-(D_i(y)+D_j(y)),\dots,

and the total-effect index:

.. math::
    S_{T_i}=\frac{E_{x_{-i}}(Var_{x_i}(y|x_{-i}))}{Var(y)}=
    1-\frac{Var_{x_{-i}}(E_{x_i}(y|x_{-i}))}{Var(y)}.

Moment-Independent :math:`\delta` Index
===========================================
Let :math:`g_Y(y)` be the distribution of the values of :math:`y` and denote by
:math:`g_{Y|x_i}(y)` the distribution of values of :math:`y` when the value of :math:`x_i` is fixed.
Let :math:`s(x_i)=\int|g_Y(y)-g_{Y|x_i}(y)|dy`, then the delta index of :math:`x_i` is defined as:

.. math::
    \delta_i=\frac{1}{2}\int s(x_i)g_{x_i}dx_i,

where :math:`g_{x_i}(x_i)` is the distribution of the values of :math:`x_i`.

.. note::
    The class `SensAprx` acts as a scikit-learn wrapper as a transformer based on the sensitivity
    analysis library `SALib <https://salib.readthedocs.io/en/latest/index.html>`_.

    It accepts a scikit-learn compatible regressor at initiation, fits the regressor on the
    :math:`X, y` arguments of `SensAprx.fit` and performs sensitivity analysis on the regressor.

        + The type of analysis can be determined at initiation by choosing `method`
          among `['sobol', 'morris', 'delta-mmnt']` (default: 'sobol').

        + After calling the `fit` method, coefficients are stored in `SensAprx.weights_`.

        + After calling `SensAprx.fit` by calling `SensAprx.transform(X)` selects the top `n`
          features where `n` is given at initiation through `n_features_to_select`.

        + It is easier to do sensitivity analysis on functions using SALib's ui, but if one
          prefers using scikit-learn's wrapper, then the function should be modified to resemble
          a scikit-learn regressor which simply ignores training data.

**Interpretation and implementation limits.**

The API fits the sensitivity surrogate to the selected training partition,
then evaluates it on method-generated samples over independent rectangular
feature ranges based on each observed minimum and maximum plus a margin of 0.2
in each feature's units. These indices
describe the fitted surrogate's response to input perturbations under that
design; they are not Pearson feature-target associations, model-score
attributions, or causal effects. Correlated features, and numeric columns that
are identifiers rather than measurements, can produce unrealistic combinations
in the generated design, so interpret the results cautiously and consider the
method assumptions and generated ranges before using rankings to narrow a
search.

**Numeric features only.** The sampled design and its surrogate regressor are
numeric procedures, and both Pearson correlation and the delta index are defined
through means and variances, so non-numeric columns are excluded rather than
silently coerced. A registered partition that mixes types — the Telco churn CSV
carries ``Surname``, ``Geography`` and ``Gender`` as text — is projected onto its
numeric columns; the API reports the exclusion (``excluded_columns`` /
``excluded_feature_names`` in the result, and a note above the ranking in the
web UI) and answers HTTP 400 when a partition has no numeric feature column at
all. Rows with a non-finite value in the numeric block are dropped as a complete
case and the count is reported as ``rows_dropped``. To analyse a text column,
encode it (for example one-hot) and re-register the partition: an ordinal code
would be perturbed as if it were a quantity, which the index cannot justify.

**SALib versions and the delta estimator.** The delta moment-independent index
changed shape in SALib 1.6: ``delta.analyze`` now returns ``delta_balanced``,
``delta_raw`` and ``delta_step`` instead of a single ``delta``. ``SensAprx``
resolves whichever key the installed version provides and records it as
``estimator`` in the API result (``delta`` on SALib ≤ 1.5, ``delta_balanced`` on
≥ 1.6) — the three are different statistics, so compare runs from different
environments by that field rather than by method name alone.


================================================
Eliminate features based on Pearson correlation
================================================
The Pearson correlation is widely used to eliminate some of highly correlated features to reduce the number
of features. Surprisingly, there is no `scikit-learn` compatible code implementing feature selection according
to a given correlation threshold (at the time publishing this library).

Although, it sound like an easy task to do, it is not clear how to select a minimal set of features with low
correlation and which ones can be safely excluded.

`SKSurrogate.sensapprx.CorrelationThreshold` implements the following algorithm to select a minimal set of features
with correlation below a given threshold:

.. note::
    **Input:** the set of all variables :math:`V`, a positive threshold :math:`t`

    **Output** a subset :math:`W\subseteq V` where :math:`\forall a,b\in W\quad |corr(a,b)|<t`

    set :math:`P:=` the set of all pairs :math:`(a, b)\in V\times V` where :math:`|corr(a, b)|\ge t`

    set :math:`W:=\{v\in V:\forall x\in V |corr(v,x)|<t\}`

    while :math:`P\neq\emptyset` do:
        form the undirected graph :math:`G=G(V\setminus W, P)`
        find a node :math:`w\in V\setminus W` with highest degree
        update :math:`W:=W\cup\{w\}`
        remove all pairs from :math:`P` involving :math:`w`

    return :math:`W`

The above procedure selects those features which has high (positive or negative) correlation with higher number of
other features and omits the other features. Repeats this process until no more pair with high correlation remains.

Heatmaps
========

``SKSurrogate.mltrace.mltrack.heatmap`` renders any :class:`pandas.DataFrame`
as a labelled matrix (``imshow`` with row/column ticks), which is how the
tracker visualises feature weights and correlation structure. The web control
plane exposes the same two data sets as JSON so they can be drawn in the
browser without a Python process:

``GET /api/sensitivity/{task}/correlation-matrix``
    Pearson correlation matrix of the stored ``train`` partition, built as
    ``pandas.DataFrame(X, columns=feature_columns).corr()``.

``GET /api/sensitivity/{task}/weights-heatmap``
    Feature weights, one column per weight type. Prefers the weights persisted
    by a sensitivity run in the tracker (``source: "stored"``); when none are
    available it computes ``pearson`` (correlation with the target) and
    ``variance`` from the stored train partition instead and reports
    ``source: "computed"``.

Both endpoints return ``{"labels": [...], "columns": [...], "values": [[...]]}``
and accept ``train_partition`` and ``max_features`` (default ``60``) query
parameters. Requests for wider datasets are rejected with HTTP ``400`` so that
an oversized matrix is never shipped to the client; prune the feature set first
or raise the limit.

These analyses complement Explain's permutation importance and partial
dependence. Pearson feature correlation describes observed pairwise association;
target correlation is a marginal data association; Sobol, Morris, and delta
indices describe surrogate responses over generated inputs; permutation
importance measures a score drop on a selected reference partition. Their
quantities and input contracts differ, so numerical disagreement is expected
and raw values are not a common ranking or causal evidence. The UI links these
pages to encourage interpretation side by side without conflating the measures.