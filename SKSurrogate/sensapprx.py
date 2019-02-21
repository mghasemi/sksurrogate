"""
Sensitivity Analysis
===========================

Sensitivity analysis of a dataset based on a fit, sklearn style.
The core functionality is provided by `SALib <https://github.com/SALib/SALib>`_ .
"""

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except:
    BaseEstimator = type('BaseEstimator', (object,), dict())
    TransformerMixin = type('TransformerMixin', (object,), dict())


class SensAprx(BaseEstimator, TransformerMixin):
    """
    Transform data to select the most secretive factors according to a regressor that fits the data.

    :param n_features_to_select: `int` number of top features to be selected
    :param regressor: a sklearn style regressor to fit the data for sensitivity analysis
    :param method: `str` the sensitivity analysis method; defalt 'sobol', other options are 'morris' and 'delta-mmnt'
    :param margin: domain margine, default: .2
    :param num_smpl: number of samples to perform the analysis, default: 1000
    :param num_levels: number of levels for morris analysis, default: 6
    :param grid_jump: grid jump for morris analysis, default: 1
    :param num_resmpl: number of resamples for moment independent analysis, default: 10
    :param reduce: whether to reduce the data points to uniques and calculate the averages of the target or not, default: False
    :param domain: pre-calculated unique points, if none, and reduce is `True` then unique points will be found
    :param probs: pre-calculated values associated to `domain` points
    """

    def __init__(self, n_features_to_select=10, regressor=None, method='sobol', margin=.2, num_smpl=600, num_levels=6,
                 grid_jump=1, num_resmpl=10, reduce=False, domain=None, probs=None):
        self.n_features_to_select = n_features_to_select
        self.regressor = regressor
        self.method = method
        self.margin = margin
        self.num_smpl = num_smpl
        self.num_levels = num_levels
        self.grid_jump = grid_jump
        self.num_resmpl = num_resmpl
        self.reduce = reduce
        self.domain = domain
        self.probs = probs
        self.weights_ = None

    def _avg_fucn(self, X, y):
        from numpy import unique, concatenate, array
        if self.reduce:
            X_ = unique(X, axis=0)
            self.domain = []
            self.probs = []
            data_space = concatenate((X, y.reshape(y.shape[0], 1)), axis=1)
            for row in X_:
                X_temp = data_space
                for idx in range(row.shape[0]):
                    X_temp = X_temp[X_temp[:, idx] == row[idx]]
                y_p = sum(X_temp[:, -1]) / float(len(X_temp))
                self.domain.append(row)
                self.probs.append(y_p)
            self.domain = array(self.domain)
            self.probs = array(self.probs)
        else:
            self.domain = X
            self.probs = y

    def fit(self, X, y):
        """
        Fits the regressor to the data `(X, y)` and performs a sensitivity analysis on the result of the regression.

        :param X: Training data
        :param y: Target values
        :return: `self`
        """
        from numpy import argpartition
        N = len(X[0])
        if (self.domain is None) or (self.probs is None):
            self._avg_fucn(X, y)
        if self.regressor is None:
            from sklearn.svm import SVR
            self.regressor = SVR()
        self.regressor.fit(self.domain, self.probs)
        bounds = [[min(self.domain[:, idx]) - self.margin, max(self.domain[:, idx]) + self.margin] for idx in range(N)]
        problem = dict(num_vars=N, names=['x%d' % (idx) for idx in range(N)], bounds=bounds)
        Res = []
        if self.method == 'sobol':
            from SALib.sample import saltelli
            from SALib.analyze import sobol
            param_values = saltelli.sample(problem, self.num_smpl)
            Y = self.regressor.predict(param_values)
            Res = sobol.analyze(problem, Y)['ST']
            self.weights_ = Res
        elif self.method == 'morris':
            from SALib.sample import morris as mrs
            from SALib.analyze import morris
            param_values = mrs.sample(problem, self.num_smpl, num_levels=self.num_levels, grid_jump=self.grid_jump)
            Y = self.regressor.predict(param_values)
            Res = morris.analyze(problem, param_values, Y, num_levels=self.num_levels, grid_jump=self.grid_jump)[
                'mu_star']
            self.weights_ = Res
        elif self.method == 'delta-mmnt':
            from SALib.sample import latin
            from SALib.analyze import delta
            param_values = latin.sample(problem, self.num_smpl)
            Y = self.regressor.predict(param_values)
            Res = delta.analyze(problem, param_values, Y, num_resamples=self.num_resmpl)['delta']
            self.weights_ = Res
        self.top_features_ = argpartition(Res, -self.n_features_to_select)[-self.n_features_to_select:]
        return self

    def transform(self, X):
        return X[:, self.top_features_[:self.n_features_to_select]]

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params and returns a transformed version of X.

        :param X: numpy array of shape [n_samples, n_features]; Training set.
        :param y: numpy array of shape [n_samples]; Target values.
        :return: Transformed array
        """
        self.fit(X, y)
        return self.transform(X)
