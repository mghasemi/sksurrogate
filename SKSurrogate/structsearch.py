r"""
Surrogate Random Search
====================================

This module provides basic functionality to optimize an expensive black-box function
based on *Surrogate Random Search*.
The Structured Random Search (SRS) method attempts to approximate an optimal solution to the following

    minimize :math:`f(x)`

    subject to
        :math:`g_i(x)\ge0~`   :math:`~i=1,\dots, m,`

where arbitrary evaluations of :math:`f` is not a viable option.
The original random search itself is guarantee to converge to a local solution, but the convergence is
usually very slow and most information about :math:`f` is dismissed except for the best candidate. SRS
tries to use all information acquired about :math:`f` so far during the iterations. At :math:`i^{th}`
round of iteration SRS replaces :math:`f` by a surrogate :math:`\hat{f}_i` that enjoys many nice
analytical properties which make its optimization an easier task to overcome. Then by solving the above
optimization problem with :math:`f` replaced by :math:`\hat{f}` one gets a more informed candidate
:math:`x_i` for the next iteration. If a certain number of iterations do not result in a better candidate,
the method returns to random sampling to collect more information about :math:`f`. The surrogate
function :math:`\hat{f}_i` can be found in many ways such as (non)linear regression, Gaussian
process regression, etc. and  `SurrogateSearch` do not have a preference. But by default it uses a polynomial
regression of degree 3 if no regressor is provided. Any regressor following the architecture of `scikit-learn`
is acceptable. Note that regressors usually require a minimum number of data points to function properly.

There are various ways for sampling a random point in feasible space which affects the performance of SRS.
`SurrogateSearch` implements two methods: `BoxSample` and `SphereSample`. One can choose whether to shrink the
volume of the box or sphere that the sample is selected from too.
"""


class BaseSample(object):
    """
    This is the base class for various sampling methods.

    :param init_radius: **optional** (default=2.); positive real number indicating the initial radius of the local
        search ball.
    :param contraction: **optional** (default=.0); the contraction factor which must be a positive real less than 1.
    :param ineq: **optional**; a list of functions whose positivity region will be the acceptable condition.
    :param bounds: **optional**; the list of (ordered) tuples determining the bound of each component.
    """

    def __init__(self, **kwargs):
        import random

        self.init_radius = kwargs.get("init_radius", 2.0)
        self.contraction = kwargs.get("contraction", 0.9)
        if self.contraction >= 1:
            raise "`contraction` factor must be between 0. and 1."
        self.ineqs = kwargs.pop("ineq", [])
        self.bounds = kwargs.pop("bounds", None)
        self.rng = kwargs.get("rng", random)

    def check_constraints(self, point):
        """
        Checks constraints on the sample if provided

        :param point: the candidate to be checked
        :return: `boolean` True or False for if all constraints hold or not.
        """

        n = len(point)
        for cns in self.ineqs:
            if cns(point) < 0.0:
                return False
        if self.bounds is not None:
            for i in range(n):
                if (point[i] < self.bounds[i][0]) or (point[i] > self.bounds[i][1]):
                    return False
        return True

    def sample(self, centre, cntrctn=1.0):
        raise NotImplementedError("The method `sample` is not implemented")


class CompactSample(BaseSample):
    """
    Generates samples uniformly out of a box.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cntrctn = None

    def sample(self, centre, cntrctn=1.0):
        """
        Samples a point out of a box centered at `centre`

        :param centre: `numpy.array` the center of the box
        :param cntrctn: `float` customized contraction factor
        :return: `numpy.array` a new sample
        """
        from numpy import array

        self.cntrctn = cntrctn
        flag = False
        n = len(centre)
        candid = []
        while not flag:
            candid = array(
                [self.rng.uniform(self.bounds[_][0], self.bounds[_][1]) for _ in range(n)]
            )
            flag = self.check_constraints(candid)
        return candid


class BoxSample(BaseSample):
    """
    Generates samples out of a box around a given center.

    :param init_radius: `float` the initial half-length of the edges of the sampling box; default: 2.
    :param contraction: `float` the contraction factor for repeated sampling.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bounds = kwargs.pop("bounds", None)

    def sample(self, centre, cntrctn=1.0):
        """
        Samples a point out of a box centered at `centre`

        :param centre: `numpy.array` the center of the box
        :param cntrctn: `float` customized contraction factor
        :return: `numpy.array` a new sample
        """
        from numpy import array

        flag = False
        n = len(centre)
        candid = []
        radius = self.init_radius * cntrctn
        while not flag:
            candid = array([self.rng.uniform(-radius, radius) for _ in range(n)]) + centre
            flag = self.check_constraints(candid)
            radius = radius * self.contraction
        return candid


class SphereSample(BaseSample):
    """
    Generates samples out of an sphere around a given center.

    :param init_radius: `float` the initial radius of the sampling sphere; default: 2.
    :param contraction: `float` the contraction factor for repeated sampling.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bounds = kwargs.pop("bounds", None)

    def sample(self, centre, cntrctn=1.0):
        """
        Samples a point out of an sphere centered at `centre`

        :param centre: `numpy.array` the center of the sphere
        :param cntrctn: `float` customized contraction factor
        :return: `numpy.array` a new sample
        """
        from numpy import empty, sqrt

        flag = False
        n = len(centre)
        candid = []
        radius = self.init_radius * cntrctn
        while not flag:
            rng = list(range(n))
            self.rng.shuffle(rng)
            candid = empty(n)
            for idx in range(n):
                remaining = radius ** 2 - sum(candid[:idx] ** 2)
                r = sqrt(max(remaining, 0.0))
                candid[idx] = self.rng.uniform(-r, r)
            candid = candid[rng] + centre
            flag = self.check_constraints(candid)
            radius = radius * self.contraction
        return candid


class SurrogateSearch(object):
    """
    An implementation of the Surrogate Random Search (SRS).

    :param objective: a `callable`, the function to be minimized
    :param ineq: a list of callables which represent the constraints (default: [])
    :param task_name: `str` a name to refer to the optimization task, store & restore previously acquired
        (default: 'optim_task')
    :param bounds: a list of tuples of real numbers representing the bounds on each variable; default: None
    :param max_iter: `int` the maximum number of iterations (default: 50)
    :param radius: `float` the initial radius of sampling region (default: 2.)
    :param contraction: `float` the rate of radius contraction (default: .9)
    :param sampling: the sampling method either `BoxSample` or `SphereSample` (default `SphereSample`)
    :param search_sphere: `boolean` whether to fit the surrogate function on a neighbourhood of current candidate or
        over all sampled points (default: False)
    :param deg: `int` degree of polynomial regressor if one chooses to fit polynomial surrogates (default: 3)
    :param min_evals: `int` minimum number of samples before fitting a surrogate (default will be calculated as if the
        surrogate is a polynomial of degree 3)
    :param regressor: a regressor (scikit-learn style) to find a surrogate
    :param scipy_solver: `str` the scipy solver ('COBYLA' or 'SLSQP') to solve the local optimization problem at each
        iteration (default: 'COBYLA')
    :param max_itr_no_prog: `int` maximum number of iterations with no progress (default: infinity)
    :param Continue: `boolean` continues the progress from where it has been interrupted (default: False)
    :param warm_start: `boolean` use data from the previous attempts, but starts from the first iteration
        (default: False)
    :param cpu_limit: Optional cap on the parallelism consumed by any internal worker pool.
    :param verbose: `boolean` whether to report the progress on commandline or not (default: False)
    """

    def __init__(self, objective, **kwargs):
        from numpy import inf
        from scipy.special import binom
        import random

        self.objective = objective
        self.random_state = kwargs.pop("random_state", None)
        self.rng = random.Random(self.random_state)
        self.ineqs = kwargs.pop("ineq", [])
        self.bounds = kwargs.pop("bounds", None)
        self.MaxIter = kwargs.pop("max_iter", 50)
        self.time_limit = kwargs.pop("time_limit", None)
        self.max_evals = kwargs.pop("max_evals", None)
        self.cpu_limit = kwargs.pop("cpu_limit", None)
        self.memory_limit = kwargs.pop("memory_limit", None)
        self.prune_threshold = kwargs.pop("prune_threshold", None)
        self.min_partial_folds = kwargs.pop("min_partial_folds", 1)
        self.radius = kwargs.pop("radius", 2.0)
        self.contraction = kwargs.pop("contraction", 0.9)
        sampling = kwargs.pop("sampling", SphereSample)
        self.Sampling = sampling(
            init_radius=self.radius,
            contraction=self.contraction,
            ineq=self.ineqs,
            bounds=self.bounds,
            rng=self.rng,
        )
        self.iteration = 0
        self.verbose = kwargs.pop("verbose", False)
        self.MaxIterNoProg = kwargs.pop("max_itr_no_prog", inf)
        self.NumIterNoProg = 0
        self.x0 = kwargs.pop("x0", None)
        self.evaluated = []
        self.NumFailedLocOptim = 0
        # init data storing protocol
        self.TaskName = kwargs.pop("task_name", "optim_task")
        self.Continue = kwargs.pop("Continue", False)
        self.warm_start = kwargs.pop("warm_start", False)
        self.optimizer = kwargs.get("optimizer", "scipy")
        self.__load()
        if self.x0 is not None:
            val = self.objective(self.x0)
            self.evaluated.append([self.x0, val])
            self.current = self.x0
            self.current_val = val
            n_ = len(self.x0)
        else:
            n_ = 1
        self.search_sphere = kwargs.pop("search_sphere", False)
        poly_deg = kwargs.pop("deg", 3)
        self.MinEvals = kwargs.pop("min_evals", binom(n_ + poly_deg, n_))
        self.regressor = kwargs.pop("regressor", None)
        self.scipy_solver = kwargs.get("scipy_solver", "COBYLA")
        self.termination_reason = None
        self.summary_ = {}
        if self.regressor is None:
            from .NpyProximation import HilbertRegressor

            self.regressor = HilbertRegressor(deg=3)
            new_n_evals = binom(n_ + 3, n_)
            if self.MinEvals < new_n_evals:
                self.MinEvals = new_n_evals
        self.other = kwargs

    def __save(self):
        """
        Logs state of the optimization progress at each iteration
        :return: None
        """
        from pickle import dumps

        fl = open(self.TaskName + ".pkl", "wb")
        data2store = dict(
            iteration=self.iteration,
            evaluated=self.evaluated,
            current=self.current,
            current_val=self.current_val,
        )
        fl.write(dumps(data2store))
        fl.close()

    def __load(self):
        """
        Loads previous information saved, if any
        :return: None
        """
        from pickle import loads

        restored_data = {}
        try:
            fl = open(self.TaskName + ".pkl", "rb")
            restored_data = loads(fl.read())
            fl.close()
        except FileNotFoundError:
            pass
        expected_dimension = len(self.bounds) if self.bounds is not None else None
        restored_current = restored_data.get("current")
        if expected_dimension is not None and restored_current is not None:
            if len(restored_current) != expected_dimension:
                restored_data = {}
        if expected_dimension is not None and restored_data.get("evaluated"):
            restored_data["evaluated"] = [
                item
                for item in restored_data["evaluated"]
                if len(item[0]) == expected_dimension
            ]
        if self.Continue:
            if "iteration" in restored_data:
                self.iteration = restored_data["iteration"]
            if "evaluated" in restored_data:
                self.evaluated = restored_data["evaluated"]
            if "current" in restored_data:
                self.current = restored_data["current"]
                self.x0 = None
            if "current_val" in restored_data:
                self.current_val = restored_data["current_val"]
        if self.warm_start:
            if "evaluated" in restored_data:
                self.evaluated = restored_data["evaluated"]

    def __round_optim(self, obj, x0, cns):
        """
        Solves the iteration round's local optimization
        :param obj: the approximate objective
        :param x0: initial point of the optimization, i.e., the round's sampled point
        :param cns: constraints
        :return: result of the optimization
        """
        res = None
        if self.optimizer == "scipy":
            from scipy.optimize import minimize

            if (len(cns) > 0) and (
                    self.scipy_solver not in ["COBYLA", "SLSQP", "L-BFGS-B"]
            ):
                raise ValueError(
                    "%s does not support arbitrary constraints" % self.scipy_solver
                )
            res = minimize(
                obj, x0, method=self.scipy_solver, constraints=cns, bounds=self.bounds
            )
        return res

    def __optim_param(self):
        """
        Fetches and sets the Optimization's parameters
        :return: None
        """
        self.optimizer = self.other.pop("optimizer", "scipy")
        self.scipy_solver = self.other.pop("scipy_solver", "COBYLA")

    def __iterate(self, x0):
        """
        Repeat a round of optimization.
        :param x0: init point for the iteration
        :return: None
        """
        from numpy import sqrt, array, isfinite

        close_points = []
        X = []
        y = []
        n_close_points = 0
        n_ = len(x0)
        for p in self.evaluated:
            if not isfinite(p[1]):
                continue
            if self.search_sphere:
                cur = p[0] - x0
                distance = sqrt(sum([t ** 2 for t in cur]))
                if distance <= self.radius:
                    close_points.append(tuple(list(p[0]) + [p[1]]))
                    X.append(p[0])
                    y.append(p[1])
                    n_close_points += 1
            else:
                close_points.append(tuple(list(p[0]) + [p[1]]))
                X.append(p[0])
                y.append(p[1])
                n_close_points += 1

        if n_close_points >= self.MinEvals:
            if self.verbose > 1:
                print(
                    "Found enough number of points (%d) for approximation."
                    % n_close_points
                )
            self.regressor.fit(array(X), array(y))
            apprx = lambda x: self.regressor.predict(array([x]))[0]
            cns = []
            if self.search_sphere:
                r = self.radius * (self.contraction ** int(self.NumFailedLocOptim / 2))
                cns_f = lambda x, x_=x0, r_=r: r_ ** 2 - sum(
                    [(x[i] - x_[i]) ** 2 for i in range(n_)]
                )
                cns = [{"type": "ineq", "fun": cns_f}]
            for inq in self.ineqs:
                cns.append({"type": "ineq", "fun": inq})
            min_res = self.__round_optim(apprx, x0, cns)
            x_0 = min_res.x
            if min_res.success:
                self.NumFailedLocOptim = 0
                try:
                    val = self.objective(x_0)
                    self.evaluated.append([x_0, val])
                    if self.verbose > 1:
                        print(
                            """Minimum point of the approximation: %s;\nThe objective's value: %f;\nThe surrogate\\
                             value %f"""
                            % (str(x_0), val, min_res.fun)
                        )
                    if val < self.current_val:
                        self.NumIterNoProg = 0
                        if self.verbose > 0:
                            print("NEW CANDIDATE")
                        self.current = x_0
                        self.current_val = val
                    else:
                        self.NumIterNoProg += 1
                except ValueError:
                    self.NumIterNoProg += 1
                    self.NumFailedLocOptim += 1
                    if self.verbose > 1:
                        print("""Error in function evaluation.""")
                    return
            else:
                self.NumFailedLocOptim += 1
                if self.verbose > 1:
                    print("The local optimization was not successful.")
        else:
            val = self.objective(x0)
            self.evaluated.append([x0, val])
            if self.verbose > 1:
                print(
                    """New point sampled: %s;\nThe objective's value: %f"""
                    % (str(x0), val)
                )
            if val < self.current_val:
                self.NumIterNoProg = 0
                if self.verbose > 0:
                    print("NEW CANDIDATE")
                self.current = x0
                self.current_val = val
            else:
                self.NumIterNoProg += 1
        if self.verbose > 1:
            print("==========" * 8)
        self.iteration += 1

    def __call__(self, *args, **kwargs):
        """
        Runs the structured random search.
        :return: the best minimum point and value
        """
        pbar = None
        # try:
        #    from tqdm import tqdm
        # except ImportError:
        #    tqdm = None
        from math import log
        from time import monotonic

        started = monotonic()
        self.termination_reason = None
        self.summary_ = {}
        # if tqdm is not None:
        if self.verbose > 0:
            # pbar = tqdm(total=self.MaxIter)
            # pbar.update(self.iteration)
            print("Iteration {m} / {n}".format(m=self.iteration, n=self.MaxIter))
        self.__optim_param()
        while self.iteration <= self.MaxIter:
            if self.time_limit is not None and monotonic() - started >= self.time_limit:
                self.termination_reason = "time_limit"
                break
            if self.max_evals is not None and len(self.evaluated) >= self.max_evals:
                self.termination_reason = "max_evals"
                break
            if self.verbose > 1:
                print("Iteration # %d" % self.iteration)
                print("----------" * 8)
            new_point = self.Sampling.sample(
                self.current,
                cntrctn=self.contraction
                        ** (self.NumFailedLocOptim + log(self.iteration + 1)),
            )
            self.__iterate(new_point)
            if self.NumIterNoProg > self.MaxIterNoProg:
                if self.verbose > 1:
                    print("No progress in %d iterations." % self.NumIterNoProg)
                self.termination_reason = "max_iter_no_progress"
                break
            # if tqdm is not None:
            if self.verbose > 0:
                print("End of iteration.")
                # pbar.update(1)  # update the progressbar
            self.__save()  # save the progress
        if self.termination_reason is None:
            self.termination_reason = "max_iter" if self.iteration >= self.MaxIter else "completed"
        elapsed = monotonic() - started
        self.summary_ = {
            "termination_reason": self.termination_reason,
            "status": "terminated" if self.termination_reason != "completed" else "completed",
            "budget": {
                "time_limit": self.time_limit,
                "max_evals": self.max_evals,
                "cpu_limit": self.cpu_limit,
                "memory_limit": self.memory_limit,
                "elapsed_seconds": elapsed,
                "iterations": self.iteration,
                "evaluations": len(self.evaluated),
            },
            "current_value": getattr(self, "current_val", None),
        }
        return self.current, self.current_val

    def progress(self):
        """
        Generates `matplotlib` plots that represent distributions of each variable and the progress in minimization.

        :return: objective's process plot, variables' distributions
        """
        from matplotlib import pyplot as plt
        from pickle import load
        from pandas import DataFrame

        fl = open(self.TaskName + ".pkl", "rb")
        arr = load(fl)
        fl.close()
        evals_dict = {"obj": [_[1] for _ in arr["evaluated"]]}
        n_ = len(arr["evaluated"][0][0])
        for i in range(n_):
            evals_dict["v%d" % i] = [_[0][i] for _ in arr["evaluated"]]
        df = DataFrame(evals_dict)
        figr = df.plot("obj", kind="kde").get_figure()
        b = evals_dict["obj"]
        z = [min(b[:idx]) for idx in range(1, len(b))]
        _, ax = plt.subplots(1, 1)
        ax.set_xlabel("# Evaluations")
        ax.set_ylabel("Objective Value")
        ax.set_title("Objective's Progress")
        ax.grid(True)
        ax.plot(z)
        return ax, figr


########################################################################################################################
# Scikit-Learn Search CV
class Real(object):
    """
    The range of possible values for a real variable;
    `a` is the minimum and `b` is the maximum.
    Defaults are + and - infinity.

    :param a: the lower bound for the (closed) interval defined by instance (accepting '-numpy.inf')
    :param b: the upper bound for the (closed) interval defined by instance (accepting 'numpy.inf')
    """

    VType = "real"

    def __init__(self, a=None, b=None, **kwargs):
        from numpy import inf

        self.lower = a if a is not None else -inf
        self.upper = b if b is not None else inf
        self.scale = kwargs.pop("scale", "linear")
        self.bound_tuple = (self.lower, self.upper)
        self.extra = kwargs


class Integer(object):
    """
    The range of possible values for an integer variable;
    `a` is the minimum and `b` is the maximum.
    Defaults are + and - infinity.

    :param a: the lower bound for the integer interval defined by instance (accepting '-numpy.inf')
    :param b: the upper bound for the integer interval defined by instance (accepting 'numpy.inf')
    """

    VType = "integer"

    def __init__(self, a=None, b=None, **kwargs):
        from numpy import inf

        self.lower = int(a) - 0.49 if a is not None else -inf
        self.upper = int(b) + 0.49 if b is not None else inf
        self.scale = kwargs.pop("scale", "linear")
        self.bound_tuple = (self.lower, self.upper)
        self.extra = kwargs


class Categorical(object):
    """
    A list of possible values fr the search algorithm to choose from.

    :param items: A list of possible values for a parameter
    """

    VType = "categorical"

    def __init__(self, items, **kwargs):
        self.items = items
        self.lower = -0.49
        self.upper = len(self.items) - 0.51
        self.bound_tuple = (self.lower, self.upper)
        self.extra = kwargs


class HDReal(object):
    """
    An `n` dimensional box of real numbers corresponding to the classification groups (e.g. `class_weight`).
    `a` is the list of lower bounds and `b` is the list of upper bounds.

    :param a: a tuple of lower bounds for each dimension
    :param b: a tuple of upper bounds for each dimension
    """

    VType = "hdreal"

    def __init__(self, a, b, **kwargs):
        self.n = len(a)
        if len(b) != self.n:
            raise IndexError("`a` and `b` must be of the same dimension.")
        self.lower = a
        self.upper = b
        self.bound_tuple = [(self.lower[_], self.upper[_]) for _ in range(self.n)]
        self.bound_tuple = tuple(self.bound_tuple)
        self.extra = kwargs


try:
    from sklearn.model_selection._search import BaseSearchCV
    from joblib import Parallel, delayed
except ModuleNotFoundError:
    BaseSearchCV = type("BaseSearchCV", (object,), dict())
    Parallel = type("Parallel", (object,), dict())
    delayed = type("delayed", (object,), dict())


class SurrogateRandomCV(BaseSearchCV):
    """
    Surrogate Random Search optimization over hyperparameters.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by `max_iter`.

    :param estimator: estimator object. An object of that type is instantiated for each search point. This object is
        assumed to implement the scikit-learn estimator api. Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
    :param params: dict Dictionary with parameters names (string) as keys and domains as lists of parameter ranges
        to try. Domains are either lists of categorical (string) values or 2 element lists specifying a min and max
        for integer or float parameters
    :param scoring: string, callable or `None`, default=None
        A string (see model evaluation documentation) or a scorer callable
        object / function with signature ``scorer(estimator, X, y)``. If
        ``None``, the ``score`` method of the estimator is used.
    :param max_iter: int, default=50
        Number of parameter settings that are sampled. max_iter trades
        off runtime vs quality of the solution. Consider increasing
        ``n_points`` if you want to try more parameter settings in parallel.
    :param min_evals: int, default=25; Number of random evaluations before employing an approximation for the
        response surface.
    :param n_jobs: int, default=-1; number of processes to run in parallel
    :param fit_params: dict, optional; Parameters to pass to the fit method.
    :param pre_dispatch: int, or string, optional;
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an explosion of
        memory consumption when more jobs get dispatched than CPUs can process.
        This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    :param cv: int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 3-fold cross validation,

          - integer, to specify the number of folds in a `(Stratified)KFold`,

          - An object to be used as a cross-validation generator.

          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
    :param refit: boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this RandomizedSearchCV instance after fitting.
    :param verbose: int, default=0
        Prints internal information about the progress of each iteration.
    """

    def __init__(
            self,
            estimator,
            params,
            scoring=None,
            fit_params=None,
            n_jobs=-1,
            refit=True,
            cv=None,
            verbose=0,
            pre_dispatch="2*n_jobs",
            error_score="raise",
            return_train_score=True,
            max_iter=50,
            min_evals=25,
            regressor=None,
            sampling=CompactSample,
            radius=None,
            contraction=0.95,
            search_sphere=False,
            optimizer="scipy",
            scipy_solver="SLSQP",
            task_name="optim_task",
            warm_start=True,
            Continue=False,
            max_itr_no_prog=10000,
            ineqs=(),
            init=None,
            time_limit=None,
            max_evals=None,
            random_state=None,
            cpu_limit=None,
            memory_limit=None,
            prune_threshold=None,
            min_partial_folds=1,
            forbidden=(),
            conditional=(),
    ):
        super(SurrogateRandomCV, self).__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.params = params
        self.fit_params = fit_params
        self.params_list = list(params.keys())
        self.max_iter = max_iter
        self.min_evals = min_evals
        self.radius = radius
        self.contraction = contraction
        self.regressor = regressor
        self.sampling = sampling
        self.search_sphere = search_sphere
        self.optimizer = optimizer
        self.scipy_solver = scipy_solver
        self.task_name = task_name
        self.warm_start = warm_start
        self.Continue = Continue
        self.max_itr_no_prog = max_itr_no_prog
        self.time_limit = time_limit
        self.max_evals = max_evals
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.prune_threshold = prune_threshold
        self.min_partial_folds = min_partial_folds
        self.forbidden = tuple(forbidden)
        self.conditional = dict(conditional)
        self.random_state = random_state
        self.bounds = []
        if self.cpu_limit is not None:
            limit = int(self.cpu_limit)
            if limit < 1:
                raise ValueError("cpu_limit must be a positive integer")
            if self.n_jobs == -1 or self.n_jobs > limit:
                self.n_jobs = limit
        self.ineqs = ineqs
        self.init = init if init is not None else {}
        self.OPTIM = None
        self.scorer_ = {}
        self.x0 = ()
        self.best_estimator_ = None
        self.best_estimator_score = 0.0
        self.best_score_ = 0.
        self.evaluation_history_ = []
        self.cv_results_ = {}
        self.termination_reason = None
        self.summary_ = {}
        self.failure_summary_ = {}

    def _validate_search_space(self):
        """Validate parameter definitions before starting model evaluation."""
        from sklearn.base import clone
        from math import isfinite

        estimator_params = self.estimator.get_params(deep=True)
        representative = {}
        for name in self.params_list:
            if name not in estimator_params:
                raise ValueError(
                    "Unknown search parameter %r for %s"
                    % (name, self.estimator.__class__.__name__)
                )
            parameter = self.params[name]
            value_type = getattr(parameter, "VType", None)
            if value_type == "categorical":
                if not parameter.items:
                    raise ValueError("Search parameter %r has no categorical values" % name)
                representative[name] = parameter.items[0]
                continue
            if value_type == "hdreal":
                if parameter.n < 1:
                    raise ValueError("Search parameter %r must contain at least one value" % name)
                representative[name] = {
                    index: bounds[0] for index, bounds in enumerate(parameter.bound_tuple)
                }
                continue
            if value_type not in {"integer", "real"}:
                raise ValueError("Unsupported search space type for %r: %r" % (name, value_type))
            lower = parameter.lower
            upper = parameter.upper
            if lower > upper:
                raise ValueError("Search parameter %r has a lower bound above its upper bound" % name)
            scale = getattr(parameter, "scale", "linear")
            if scale not in {"linear", "log"}:
                raise ValueError("Unsupported scale for search parameter %r: %r" % (name, scale))
            if scale == "log" and (lower <= 0 or not isfinite(lower) or not isfinite(upper)):
                raise ValueError("Log-scaled search parameter %r must have finite positive bounds" % name)
            representative[name] = lower

        try:
            clone(self.estimator).set_params(**representative)
        except Exception as exc:
            raise ValueError(
                "Invalid estimator search parameters: %s: %s"
                % (exc.__class__.__name__, exc)
            ) from exc

    @staticmethod
    def _encoded_bounds(parameter):
        from math import log

        if getattr(parameter, "scale", "linear") == "log":
            return log(parameter.lower), log(parameter.upper)
        return parameter.lower, parameter.upper

    @staticmethod
    def _encode_value(parameter, value):
        from math import log

        if getattr(parameter, "scale", "linear") == "log":
            return log(value)
        return value

    @staticmethod
    def _decode_value(parameter, value):
        from math import exp

        if getattr(parameter, "scale", "linear") == "log":
            return exp(value)
        return value

    def _forbidden_reason(self, parameters):
        for rule in self.forbidden:
            if callable(rule):
                matched = rule(parameters)
            elif isinstance(rule, dict):
                matched = all(parameters.get(name) == value for name, value in rule.items())
            else:
                raise ValueError("Forbidden rules must be callables or parameter mappings")
            if matched:
                return getattr(rule, "__name__", None) or "forbidden parameter combination"
        return None

    @staticmethod
    def _condition_matches(rule, parameters):
        if callable(rule):
            return bool(rule(parameters))
        if isinstance(rule, dict):
            return all(parameters.get(name) == value for name, value in rule.items())
        raise ValueError("Conditional rules must be callables or parameter mappings")

    def _apply_conditional_parameters(self, parameters):
        active = parameters.copy()
        for name, rule in self.conditional.items():
            if name in active and not self._condition_matches(rule, parameters):
                active.pop(name)
        return active

    def _zero_feature_reason(self, parameters, X, y):
        """Return a reason when a candidate pipeline removes every feature."""
        import numpy as np
        from sklearn.base import clone

        pipeline = clone(self.estimator)
        if not hasattr(pipeline, "steps") or len(pipeline.steps) < 2:
            return None
        pipeline.set_params(**parameters)
        features = X
        for _, transformer in pipeline.steps[:-1]:
            try:
                if hasattr(transformer, "fit_transform"):
                    features = transformer.fit_transform(features, y)
                else:
                    transformer.fit(features, y)
                    features = transformer.transform(features)
            except Exception as exc:
                return (
                    "pipeline preprocessing failed before evaluation: %s: %s"
                    % (exc.__class__.__name__, exc)
                )
            if np.asarray(features).ndim > 1 and np.asarray(features).shape[1] == 0:
                return "pipeline preprocessing produced zero features"
        return None

    def _validate_pipeline_structure(self):
        """Reject malformed pipelines before constructing the evaluation loop."""
        steps = getattr(self.estimator, "steps", None)
        if steps is None:
            return
        if not steps:
            raise ValueError("Pipeline must contain at least one step")

        names = []
        for index, step in enumerate(steps):
            if not isinstance(step, tuple) or len(step) != 2:
                raise ValueError("Pipeline step %d must be a (name, estimator) tuple" % index)
            name, transformer = step
            if not isinstance(name, str) or not name or name.startswith("__"):
                raise ValueError("Pipeline step %d has an invalid name %r" % (index, name))
            if name in names:
                raise ValueError("Pipeline step names must be unique: %r" % name)
            names.append(name)
            if index < len(steps) - 1:
                if transformer is None or not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                    raise ValueError(
                        "Pipeline intermediate step %r must implement fit and transform"
                        % name
                    )
            elif transformer is not None and not hasattr(transformer, "fit"):
                raise ValueError("Pipeline final step %r must implement fit" % name)

    def fit(self, X, y=None, groups=None, **fit_params):
        """
        Run fit with all sets of parameters.

        :param X: array-like, `shape = [n_samples, n_features]`
            Training vector, where n_samples is the number of samples and n_features is the number of features.
        :param y: array-like, `shape = [n_samples]` or `[n_samples, n_output]`, optional;
            Target relative to X for classification or regression; None for unsupervised learning.
        :param groups: array-like, with shape `(n_samples,)`, optional;
            Group labels for the samples used while splitting the dataset into train/test set.
        :param fit_params: dict of `string -> object`;
            Parameters passed to the fit method of the estimator
        :return: `self`
        """
        from random import uniform
        from numpy import array, unique, sqrt
        from sklearn.base import clone, is_classifier
        from sklearn.metrics import check_scoring
        from sklearn.model_selection import check_cv
        from sklearn.model_selection._validation import _fit_and_score

        self._validate_search_space()
        self._validate_pipeline_structure()
        # from lightgbm.sklearn import LightGBMError
        radius_list = []
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
        target_classes = []
        if y is not None:
            target_classes = unique(y)
        # if type(self.cv) is int:
        #    cv = ShuffleSplit(n_splits=self.cv, test_size=.25)
        # else:
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        x0_ = []
        self.bounds = list(self.bounds)
        for param in self.params_list:
            param_num_range = self.params[param]
            if param_num_range.VType != "hdreal":
                lower, upper = self._encoded_bounds(param_num_range)
                radius_list.append(
                    (upper - lower) / 2.0
                )
                if param in self.init:
                    if param_num_range.VType == "categorical":
                        x0_.append(param_num_range.items.index(self.init[param]))
                    else:
                        x0_.append(self._encode_value(param_num_range, self.init[param]))
                else:
                    x0_.append(uniform(lower, upper))
                if param_num_range.VType == "categorical":
                    self.bounds.append(param_num_range.bound_tuple)
                else:
                    self.bounds.append((lower, upper))
            else:
                for i in range(param_num_range.n):
                    radius_list.append(
                        (
                                param_num_range.bound_tuple[i][1]
                                - param_num_range.bound_tuple[i][0]
                        )
                        / 2.0
                    )
                    if (param in self.init) and (i in self.init[param]):
                        x0_.append(self.init[param][i])
                    else:
                        x0_.append(
                            uniform(
                                param_num_range.bound_tuple[i][0],
                                param_num_range.bound_tuple[i][1],
                            )
                        )
                self.bounds = self.bounds + list(param_num_range.bound_tuple)
        if self.radius is None:
            rds = 0.0
            for r in radius_list:
                rds += r ** 2
            self.radius = sqrt(rds)
        self.x0 = array(x0_)
        self.bounds = tuple(self.bounds)
        cv_dat = list(cv.split(X, y, groups))

        def obj(x):
            from time import perf_counter

            started = perf_counter()
            cand_params = {}
            _idx = 0
            for _param in self.params_list:
                _param_num_range = self.params[_param]
                if _param_num_range.VType != "hdreal":
                    if _param_num_range.VType == "integer":
                        cand_params[_param] = int(round(self._decode_value(_param_num_range, x[_idx])))
                    elif _param_num_range.VType == "categorical":
                        category_index = min(
                            max(int(round(x[_idx])), 0), len(_param_num_range.items) - 1
                        )
                        cand_params[_param] = _param_num_range.items[category_index]
                    else:
                        cand_params[_param] = self._decode_value(_param_num_range, x[_idx])
                    _idx += 1
                else:
                    _cls_dict = {}
                    for i_ in range(_param_num_range.n):
                        _cls_dict[target_classes[i_]] = x[_idx]
                        _idx += 1
                    cand_params[_param] = _cls_dict
            # cl = clone(self.estimator)
            # cl.set_params(**cand_params)
            score = 0
            n_test = 0

            cand_params = self._apply_conditional_parameters(cand_params)
            forbidden_reason = self._forbidden_reason(cand_params)
            if forbidden_reason is not None:
                self.evaluation_history_.append(
                    {
                        "params": cand_params.copy(),
                        "score": None,
                        "status": "invalid",
                        "duration": perf_counter() - started,
                        "fold_duration": 0.0,
                        "error": forbidden_reason,
                        "partial_folds": 0,
                        "failure_records": [
                            {
                                "fold": None,
                                "type": "ForbiddenParameterCombination",
                                "message": forbidden_reason,
                                "traceback": None,
                                "parameters": cand_params.copy(),
                            }
                        ],
                    }
                )
                return float("inf")
            zero_feature_reason = self._zero_feature_reason(cand_params, X, y)
            if zero_feature_reason is not None:
                self.evaluation_history_.append(
                    {
                        "params": cand_params.copy(),
                        "score": None,
                        "status": "invalid",
                        "duration": perf_counter() - started,
                        "fold_duration": 0.0,
                        "error": zero_feature_reason,
                        "partial_folds": 0,
                        "failure_records": [
                            {
                                "fold": None,
                                "type": "ZeroFeaturePipeline",
                                "message": zero_feature_reason,
                                "traceback": None,
                                "parameters": cand_params.copy(),
                            }
                        ],
                    }
                )
                return float("inf")

            def parallel_fit_score(
                cl, cand_params_, X, y, scorer, train, test, fold, verbose, fit_params_, error_score
            ):
                fold_started = perf_counter()
                cl.set_params(**cand_params_)
                try:
                    _score = _fit_and_score(
                        estimator=cl,
                        X=X,
                        y=y,
                        scorer=scorer,  #
                        train=train,
                        test=test,
                        verbose=verbose,  #
                        parameters=cand_params_,
                        fit_params=fit_params_,  #
                        score_params={},
                        error_score=error_score,  #
                    )
                    if _score.get("fit_error") is not None:
                        fit_error = _score["fit_error"]
                        return {
                            "score": None,
                            "error": repr(fit_error),
                            "duration": perf_counter() - fold_started,
                            "failure_type": "FitError",
                            "failure_message": str(fit_error),
                            "failure_traceback": str(fit_error),
                            "fold": fold,
                        }
                    return {
                        "score": _score["test_scores"],
                        "error": None,
                        "duration": perf_counter() - fold_started,
                        "failure_type": None,
                        "failure_message": None,
                        "failure_traceback": None,
                        "fold": fold,
                    }
                except Exception as exc:
                    import traceback

                    if self.verbose > 1:
                        print("Model evaluation error")
                    return {
                        "score": None,
                        "error": repr(exc),
                        "duration": perf_counter() - fold_started,
                        "failure_type": exc.__class__.__name__,
                        "failure_message": str(exc),
                        "failure_traceback": traceback.format_exc(),
                        "fold": fold,
                    }

            try:
                scores = Parallel(n_jobs=self.n_jobs,
                                  verbose=self.verbose,
                                  pre_dispatch=self.pre_dispatch
                                  )(delayed(parallel_fit_score)(clone(self.estimator),
                                                                cand_params_=cand_params,
                                                                X=X,
                                                                y=y,
                                                                scorer=self.scorer_,
                                                                train=train,
                                                                test=test,
                                                                fold=fold,
                                                                verbose=self.verbose,
                                                                fit_params_=self.fit_params,
                                                                error_score=self.error_score
                                                                )
                                    for fold, (train, test) in enumerate(cv_dat))
            except Exception as exc:
                scores = ({
                    "score": None,
                    "error": repr(exc),
                    "duration": 0.0,
                    "failure_type": exc.__class__.__name__,
                    "failure_message": str(exc),
                    "failure_traceback": repr(exc),
                    "fold": None,
                },)
            errors = [result["error"] for result in scores if result["error"] is not None]
            durations = [result["duration"] for result in scores]
            failure_records = [
                {
                    "fold": result.get("fold"),
                    "type": result.get("failure_type"),
                    "message": result.get("failure_message"),
                    "traceback": result.get("failure_traceback"),
                    "parameters": cand_params.copy(),
                }
                for result in scores
                if result.get("error") is not None
            ]
            for result in scores:
                sc = result["score"]
                if sc is not None:
                    score += sc
                    n_test += 1
            score = score / float(max(n_test, 1))
            if (
                    self.prune_threshold is not None
                    and n_test >= self.min_partial_folds
                    and score <= self.prune_threshold
            ):
                self.evaluation_history_.append(
                    {
                        "params": cand_params.copy(),
                        "score": None,
                        "status": "pruned",
                        "duration": perf_counter() - started,
                        "fold_duration": sum(durations),
                        "error": "; ".join(errors) if errors else None,
                        "partial_folds": n_test,
                        "failure_records": failure_records,
                    }
                )
                return float("inf")
            self.evaluation_history_.append(
                {
                    "params": cand_params.copy(),
                    "score": -score if n_test else None,
                    "status": "complete" if n_test else "failed",
                    "duration": perf_counter() - started,
                    "fold_duration": sum(durations),
                    "error": "; ".join(errors) if errors else None,
                    "partial_folds": n_test,
                    "failure_records": failure_records,
                }
            )
            return -score

        self.OPTIM = SurrogateSearch(
            obj,
            x0=self.x0,
            max_iter=self.max_iter,
            min_evals=self.min_evals,
            ineqs=self.ineqs,
            bounds=self.bounds,
            verbose=self.verbose,
            radius=self.radius,
            regressor=self.regressor,
            sampling=self.sampling,
            search_sphere=self.search_sphere,
            contraction=self.contraction,
            max_itr_no_prog=self.max_itr_no_prog,
            time_limit=self.time_limit,
            max_evals=self.max_evals,
            random_state=self.random_state,
            optimizer=self.optimizer,
            scipy_solver=self.scipy_solver,
            task_name=self.task_name,
            warm_start=self.warm_start,
            Continue=self.Continue,
            cpu_limit=self.cpu_limit,
            memory_limit=self.memory_limit,
            prune_threshold=self.prune_threshold,
            min_partial_folds=self.min_partial_folds,
        )
        x, scr = self.OPTIM()
        best_params_ = {}
        idx = 0
        for param in self.params_list:
            param_num_range = self.params[param]
            if param_num_range.VType != "hdreal":
                if param_num_range.VType == "integer":
                    best_params_[param] = int(round(x[idx]))
                elif param_num_range.VType == "categorical":
                    category_index = min(
                        max(int(round(x[idx])), 0), len(param_num_range.items) - 1
                    )
                    best_params_[param] = param_num_range.items[category_index]
                else:
                    best_params_[param] = self._decode_value(param_num_range, x[idx])
                idx += 1
            else:
                cls_dict = {}
                for i in range(param_num_range.n):
                    cls_dict[target_classes[i]] = x[idx]
                    idx += 1
                best_params_[param] = cls_dict
        self.best_estimator_ = clone(self.estimator).set_params(**best_params_)
        self.best_estimator_score = scr
        self.best_score_ = scr
        self.termination_reason = getattr(self.OPTIM, "termination_reason", None)
        self.summary_ = getattr(self.OPTIM, "summary_", {})
        self.cv_results_ = {
            "params": [item["params"] for item in self.evaluation_history_],
            "mean_test_score": [item["score"] for item in self.evaluation_history_],
            "status": [item["status"] for item in self.evaluation_history_],
            "duration": [item["duration"] for item in self.evaluation_history_],
            "error": [item["error"] for item in self.evaluation_history_],
            "failure_records": [item.get("failure_records", []) for item in self.evaluation_history_],
        }
        estimator_name = self.estimator.__class__.__name__
        failed_trials = [item for item in self.evaluation_history_ if item["status"] == "failed"]
        self.failure_summary_ = {
            "by_estimator": {
                estimator_name: {
                    "trials": len(self.evaluation_history_),
                    "failed_trials": len(failed_trials),
                    "failure_rate": len(failed_trials) / float(max(len(self.evaluation_history_), 1)),
                }
            },
            "failure_types": {
                failure_type: sum(
                    1
                    for item in failed_trials
                    for record in item.get("failure_records", [])
                    if record.get("type") == failure_type
                )
                for failure_type in sorted(
                    {
                        record.get("type")
                        for item in failed_trials
                        for record in item.get("failure_records", [])
                        if record.get("type")
                    }
                )
            },
            "invalid_trials": sum(
                1 for item in self.evaluation_history_ if item["status"] == "invalid"
            ),
        }
        return self

    def pareto_frontier(self, score_key="score", cost_key="duration"):
        """Return trials not dominated on score and evaluation cost.

        Scores are maximized and costs are minimized. Failed trials and trials
        missing either metric are excluded. The returned records are copies and
        remain safe to inspect after subsequent searches.
        """
        valid = [
            item for item in self.evaluation_history_
            if item.get("status") == "complete"
            and item.get(score_key) is not None
            and item.get(cost_key) is not None
        ]
        frontier = []
        for candidate in valid:
            dominated = any(
                other is not candidate
                and other[score_key] >= candidate[score_key]
                and other[cost_key] <= candidate[cost_key]
                and (
                    other[score_key] > candidate[score_key]
                    or other[cost_key] < candidate[cost_key]
                )
                for other in valid
            )
            if not dominated:
                frontier.append(candidate.copy())
        return frontier
