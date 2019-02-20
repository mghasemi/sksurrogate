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
Morris method facilitates a global sensitivity analysis by making a number of local changes at
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
Sobol method (aka variance-based sensitivity analysis) works by decomposing the variance of
the output of the model into fractions which can br attributed to inputs or sets of inputs.
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