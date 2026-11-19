"""Named single-surrogate families for the pipeline search.

Reflects the Experiments workflow: *one* surrogate per experiment, applied to
every pipeline the search evaluates. :func:`build_surrogate` constructs the
one-entry surrogate list for a named estimator (GPR / KRR / SVR) together with
its iteration budget, sampling strategy and optimizer, so the library and the
API share a single source of truth for the estimator grids.
"""

from .structsearch import BoxSample, CompactSample, SphereSample

#: Estimators available for a single-surrogate search.
SURROGATE_ESTIMATORS = ("gpr", "krr", "svr")

#: Samplers the surrogate search can hand to ``SurrogateRandomCV``.
SURROGATE_SAMPLERS = {
    "BoxSample": BoxSample,
    "CompactSample": CompactSample,
    "SphereSample": SphereSample,
}


def _gpr_regressor():
    from numpy import logspace
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, Sum, WhiteKernel
    from sklearn.model_selection import RandomizedSearchCV

    return RandomizedSearchCV(
        GaussianProcessRegressor(),
        param_distributions={
            "alpha": logspace(-8, 1, 30),
            "kernel": [
                Sum(Matern(length_scale=l, nu=p), WhiteKernel(noise_level=q))
                for l in logspace(-3, 3, 30)
                for p in [0.5, 1.5, 2.5]
                for q in logspace(-3, 1.8, 30)
            ],
        },
        n_iter=35,
        cv=3,
    )


def _krr_regressor():
    from numpy import logspace
    from sklearn.gaussian_process.kernels import ExpSineSquared, Matern, Sum
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import RandomizedSearchCV

    return RandomizedSearchCV(
        KernelRidge(),
        param_distributions={
            "alpha": [1e0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
            "kernel": [
                Sum(Matern(), ExpSineSquared(l, p))
                for l in logspace(-2, 2, 20)
                for p in logspace(0, 2, 20)
            ],
        },
        n_iter=30,
        cv=2,
    )


def _svr_regressor():
    from numpy import logspace
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.svm import SVR

    return RandomizedSearchCV(
        SVR(),
        param_distributions={
            "gamma": logspace(-5, 2.0, 20),
            "tol": logspace(-6, 1.0, 20),
            "epsilon": logspace(-6, 1.0, 20),
        },
        n_iter=40,
        cv=2,
    )


_REGRESSOR_BUILDERS = {
    "gpr": _gpr_regressor,
    "krr": _krr_regressor,
    "svr": _svr_regressor,
}


def build_surrogate(estimator, itrs=30, sampling="BoxSample", solver="L-BFGS-B"):
    """Build the single surrogate configured for a pipeline search.

    :param estimator: one of :data:`SURROGATE_ESTIMATORS` (``gpr``, ``krr`` or
        ``svr``); the regressor's own hyperparameters are tuned by an inner
        ``RandomizedSearchCV`` over its grid.
    :param itrs: number of iterations for the surrogate search (default 30).
    :param sampling: one of :data:`SURROGATE_SAMPLERS` (``BoxSample``,
        ``CompactSample`` or ``SphereSample``; default ``BoxSample``).
    :param solver: ``scipy.optimize`` solver (default ``L-BFGS-B``).
    :return: a one-entry list of ``(regressor, itrs, sampling, solver)`` tuples
        consumed by ``AML.optimize_pipeline``, which runs one
        ``SurrogateRandomCV`` pass on every evaluated pipeline.
    """
    if estimator not in SURROGATE_ESTIMATORS:
        raise ValueError(
            "unknown surrogate estimator: %s (choose from %s)"
            % (estimator, ", ".join(SURROGATE_ESTIMATORS))
        )
    if sampling not in SURROGATE_SAMPLERS:
        raise ValueError(
            "unknown surrogate sampling: %s (choose from %s)"
            % (sampling, ", ".join(SURROGATE_SAMPLERS))
        )
    return [
        (_REGRESSOR_BUILDERS[estimator](), int(itrs), SURROGATE_SAMPLERS[sampling], solver)
    ]
