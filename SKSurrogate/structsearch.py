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
process regression, etc. and  `SurrogateSearch` do not have a preference. But by default (since v0.2) it fits an
un-tuned RBF Gaussian process (`sklearn.gaussian_process.GaussianProcessRegressor`), chosen over the legacy cubic
Hilbert polynomial on the evidence of the equal-budget benchmark suite — see `benchmarks/DESIGN.md` sec. 12 and raw
records DOI 10.5281/zenodo.23629024; the Hilbert surface remains available via
`regressor=HilbertRegressor(deg=...)`. Any regressor following the architecture of `scikit-learn`
is acceptable. Note that regressors usually require a minimum number of data points to function properly.

There are various ways for sampling a random point in feasible space which affects the performance of SRS.
`SurrogateSearch` implements two methods: `BoxSample` and `SphereSample`. One can choose whether to shrink the
volume of the box or sphere that the sample is selected from too.
"""

import logging

from ._logutil import emit, get_logger

logger = get_logger(__name__)


#: ``scipy.optimize.minimize`` methods that honour nonlinear constraints.
#:
#: Everything else (notably ``L-BFGS-B``, ``TNC``, ``BFGS``, ``Powell``,
#: ``Nelder-Mead``, ``CG``) accepts a ``constraints=`` argument and then **ignores
#: it** after a warning — the returned point need not be feasible at all. Passing
#: constraints to such a method is a silent correctness bug, so it is refused here
#: rather than left to SciPy's warning.
SCIPY_CONSTRAINED_SOLVERS = frozenset({"COBYLA", "SLSQP", "trust-constr", "COBYQA"})


def check_scipy_solver_supports_constraints(solver, has_constraints, context="SurrogateSearch"):
    """Refuse a solver that would silently drop inequality constraints.

    :param solver: the ``scipy.optimize.minimize`` method name.
    :param has_constraints: whether the call will pass any constraints at all.
    :param context: the configuration site named in the error, e.g. ``scipy_solver``.
    :raises ValueError: when constraints are requested on a bounds-only method.
    """
    if not has_constraints or solver in SCIPY_CONSTRAINED_SOLVERS:
        return
    raise ValueError(
        "%s(%r) cannot handle inequality constraints: scipy.optimize.minimize accepts "
        "the constraints argument for this method and then ignores it, so the returned "
        "point need not be feasible. Use one of %s, or remove the constraints."
        % (
            context,
            solver,
            ", ".join(sorted(SCIPY_CONSTRAINED_SOLVERS)),
        )
    )



class SearchStopped(Exception):
    """Raised inside a search when its ``stop_check`` asks it to stop.

    Deliberately **not** a ``ValueError``: the optimization loop treats a
    ``ValueError`` raised by the objective as a failed candidate and carries on
    (see the surrogate-exploit branch in ``SurrogateSearch.__iterate``), while
    a stop request must unwind the whole search. Callers that own a checkpoint
    catch this, persist state, and offer a resume.
    """


class BaseSample(object):
    """
    This is the base class for various sampling methods.

    :param init_radius: **optional** (default=2.); positive real number indicating the initial radius of the local
        search ball.
    :param contraction: **optional** (default=.0); the contraction factor which must be a positive real less than 1.
    :param ineq: **optional**; a list of functions whose positivity region will be the acceptable condition.
    :param bounds: **optional**; the list of (ordered) tuples determining the bound of each component.
    :param rng: **optional**; a ``numpy.random.Generator``. A private generator is created when it is
        omitted, so a sampler never draws from the global ``random``/``numpy.random`` state.
    """

    def __init__(self, **kwargs):
        from numpy.random import default_rng

        self.init_radius = kwargs.get("init_radius", 2.0)
        self.contraction = kwargs.get("contraction", 0.9)
        if self.contraction >= 1:
            raise ValueError("`contraction` factor must be between 0. and 1.")
        self.ineqs = kwargs.pop("ineq", [])
        self.bounds = kwargs.pop("bounds", None)
        self.rng = kwargs.get("rng") or default_rng()

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


def _resolve_continue_flag(legacy, canonical):
    if canonical is not None and legacy is not False and legacy != canonical:
        raise ValueError("Continue and continue_ cannot specify conflicting values")
    return legacy if canonical is None else canonical


#: Random streams a search derives from one ``random_state``. Kept separate so that
#: changing how the starting point is drawn cannot silently reroute the sampling
#: path (and vice versa) of an otherwise identical experiment.
RNG_STREAM_NAMES = ("init_candidate", "sampling")


def make_rng_streams(random_state, names=RNG_STREAM_NAMES):
    """Derive independent ``numpy.random.Generator`` streams from ``random_state``.

    :param random_state: ``None`` (fresh entropy), an ``int``, a
        ``numpy.random.Generator`` or a ``numpy.random.RandomState`` — the same
        values scikit-learn accepts on its estimators.
    :param names: stream names; the last one receives the caller's generator when a
        ``Generator`` is supplied, so an external generator keeps driving the search.
    :return: ``dict`` mapping each name to a ``numpy.random.Generator``.

    The streams are children of one :class:`numpy.random.SeedSequence`, so a single
    seed reproduces all of them, and no stream can perturb another. Nothing here
    touches the global ``random`` / ``numpy.random`` state.
    """
    from numbers import Integral

    from numpy.random import Generator, RandomState, SeedSequence, default_rng

    names = tuple(names)
    caller_generator = None
    if isinstance(random_state, Generator):
        caller_generator = random_state
        seed_sequence = SeedSequence(int(caller_generator.integers(0, 2 ** 63 - 1)))
    elif isinstance(random_state, RandomState):
        seed_sequence = SeedSequence(int(random_state.randint(0, 2 ** 31 - 1)))
    elif random_state is None:
        seed_sequence = SeedSequence()
    elif isinstance(random_state, Integral):
        seed_sequence = SeedSequence(int(random_state))
    else:
        raise TypeError(
            "random_state must be None, an int, a numpy.random.Generator or a "
            "numpy.random.RandomState; got %r" % (random_state,)
        )
    streams = {
        name: default_rng(child)
        for name, child in zip(names, seed_sequence.spawn(len(names)))
    }
    if caller_generator is not None:
        streams[names[-1]] = caller_generator
    return streams


def resolve_sampling(sampling):
    """Return a sampler class from a class or one of ``"box"``, ``"sphere"``, ``"compact"``."""
    if isinstance(sampling, str):
        registry = {"box": BoxSample, "sphere": SphereSample, "compact": CompactSample}
        try:
            return registry[sampling.lower()]
        except KeyError:
            raise ValueError("sampling must be one of %s or a sampler class" % sorted(registry))
    return sampling


from .scoring import check_scoring_direction


class SurrogateSearch(object):
    """
    An implementation of the Surrogate Random Search (SRS).

    :param objective: a `callable`, the function to be minimized
    :param ineq: a list of callables which represent the constraints (default: [])
    :param task_name: `str` optional name used to checkpoint and resume this search. When it is
        omitted (default) nothing is written to disk, so a library user's working directory stays
        untouched; pass ``checkpoint_dir`` to choose where the checkpoint lives.
    :param checkpoint_dir: `str` optional directory for the checkpoint written when ``task_name`` is given.
    :param rng: optional ``numpy.random.Generator`` used for sampling. When omitted, one is derived from
        ``random_state`` together with the starting-point stream (see :func:`make_rng_streams`).
    :param surrogate_fit_error: ``"degrade"`` (default) records a surrogate fit/prediction/optimizer
        failure and falls back to exploration for that round; ``"raise"`` aborts the search on the
        first such failure (the pre-0.6 behaviour).
    :param bounds: a list of tuples of real numbers representing the bounds on each variable; default: None
    :param max_iter: `int` the evaluation budget: the number of objective evaluations to spend,
        counting the initial candidate (default: 50). Rounds are not evaluations - a round whose
        surrogate fit or acquisition fails spends none and is charged to the no-progress counter.
    :param radius: `float` the initial radius of sampling region (default: 2.)
    :param contraction: `float` the rate of radius contraction (default: .9)
    :param sampling: the sampling method either `BoxSample` or `SphereSample` (default `SphereSample`)
    :param search_sphere: `boolean` whether to fit the surrogate function on a neighbourhood of current candidate or
        over all sampled points (default: False)
    :param deg: `int` degree used to compute the default minimum number of evaluations (default: 3); if you pass your own
        polynomial surface, e.g. `regressor=HilbertRegressor(deg=...)`, that object's degree governs instead
    :param min_evals: `int` minimum number of samples before fitting a surrogate (default will be calculated as if the
        surrogate is a polynomial of degree 3)
    :param regressor: a regressor (scikit-learn style) to find a surrogate; default since v0.2 is an un-tuned RBF
        `GaussianProcessRegressor` — pass e.g. `HilbertRegressor(deg=3)` for the legacy cubic polynomial surface
    :param scipy_solver: `str` the scipy solver used for the local optimization problem at each
        iteration (default: 'COBYLA'). Constraints are honoured only by 'COBYLA', 'SLSQP',
        'trust-constr' and 'COBYQA'; bounds-only methods such as 'L-BFGS-B' are rejected when
        constraints are present, because SciPy accepts the argument and then ignores it.
    :param max_itr_no_prog: `int` maximum number of iterations with no progress (default: infinity)
    :param continue_: `boolean` continues progress from where it was interrupted (default: False)
    :param Continue: legacy spelling for `continue_`; retained for compatibility
    :param warm_start: `boolean` use data from the previous attempts, but starts from the first iteration
        (default: False)
    :param cpu_limit: Optional cap on the parallelism consumed by any internal worker pool.
    :param verbose: `boolean` whether to report the progress on commandline or not (default: False)
    """

    def __init__(self, objective, **kwargs):
        from numpy import inf
        from scipy.special import binom

        self.objective = objective
        self.random_state = kwargs.pop("random_state", None)
        # A caller that already derived streams passes the sampling one in; otherwise
        # derive both here so a standalone SurrogateSearch is reproducible too.
        self.rng = kwargs.pop("rng", None) or make_rng_streams(self.random_state)["sampling"]
        # Both spellings are accepted: ``ineq`` is the documented name on the sampler
        # and ``ineqs`` is what SurrogateRandomCV forwards, which used to be dropped
        # silently here (no constraints ever reached the sampler or the optimizer).
        if "ineqs" in kwargs:
            self.ineqs = kwargs.pop("ineqs")
        else:
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
        sampling = resolve_sampling(kwargs.pop("sampling", SphereSample))
        self.Sampling = sampling(
            init_radius=self.radius,
            contraction=self.contraction,
            ineq=self.ineqs,
            bounds=self.bounds,
            rng=self.rng,
        )
        self.iteration = 0
        self.n_rounds = 0
        self.verbose = kwargs.pop("verbose", False)
        self.MaxIterNoProg = kwargs.pop("max_itr_no_prog", inf)
        self.NumIterNoProg = 0
        self.x0 = kwargs.pop("x0", None)
        self.x0_pending = self.x0 is not None
        self.evaluated = []
        self.sampling_sources_ = []
        self._next_eval_phase = None
        self.NumFailedLocOptim = 0
        # init data storing protocol
        self.TaskName = kwargs.pop("task_name", None)
        self.checkpoint_dir = kwargs.pop("checkpoint_dir", None)
        #: ``"degrade"`` (default) records a surrogate failure, falls back to
        #: exploration for that round and keeps going; ``"raise"`` restores the
        #: pre-0.6 behaviour of aborting the search on the first failure.
        self.surrogate_fit_error = kwargs.pop("surrogate_fit_error", "degrade")
        if self.surrogate_fit_error not in ("degrade", "raise"):
            raise ValueError(
                "surrogate_fit_error must be 'degrade' or 'raise'; got %r"
                % (self.surrogate_fit_error,)
            )
        legacy_continue = kwargs.pop("Continue", False)
        canonical_continue = kwargs.pop("continue_", None)
        self.Continue = _resolve_continue_flag(legacy_continue, canonical_continue)
        self.warm_start = kwargs.pop("warm_start", False)
        self.optimizer = kwargs.get("optimizer", "scipy")
        self.__load()
        if self.x0 is not None:
            n_ = len(self.x0)
        else:
            n_ = 1
        # The starting point is evaluated by ``__call__``, not here: one place spends
        # the evaluation budget, and constructing a search never calls the objective.
        if self.evaluated:
            self.current = self.evaluated[-1][0]
            self.current_val = self.evaluated[-1][1]
        else:
            self.current = None
            self.current_val = inf
        self.search_sphere = kwargs.pop("search_sphere", False)
        poly_deg = kwargs.pop("deg", 3)
        requested_min_evals = kwargs.pop("min_evals", None)
        self.min_evals_source = "user" if requested_min_evals is not None else "default"
        self.MinEvals = (
            requested_min_evals
            if requested_min_evals is not None
            else binom(n_ + poly_deg, n_)
        )
        requested_regressor = kwargs.pop("regressor", None)
        self.regressor = requested_regressor
        regressor_source = "user" if requested_regressor is not None else "default"
        self.scipy_solver = kwargs.get("scipy_solver", "COBYLA")
        # Fail at configuration time rather than after a long search: an unsupported
        # solver used to be accepted and then silently ignore the constraints.
        if self.optimizer == "scipy":
            check_scipy_solver_supports_constraints(
                self.scipy_solver, bool(self.ineqs) or bool(self.search_sphere)
            )
        self.termination_reason = None
        self.summary_ = {}
        if self.regressor is None:
            # Default response surface since v0.2: stock, un-tuned RBF Gaussian process.
            # Evidence: equal-budget benchmark suite (benchmarks/DESIGN.md sec. 12; raw
            # records DOI 10.5281/zenodo.23629024) -- top mean CV on all six datasets at
            # both B=30 and B=60 vs the legacy cubic Hilbert default (seed-level wins
            # 20W/17T/5L and 25W/13T/4L). The MinEvals combinatorial floor below is kept
            # identical to v0.1 so surrogate firing matches that benchmark configuration;
            # pass regressor=HilbertRegressor(deg=<deg>) for the legacy surface.
            from sklearn.gaussian_process import GaussianProcessRegressor

            self.regressor = GaussianProcessRegressor()
            new_n_evals = binom(n_ + 3, n_)
            # The floor is a *default*: an explicitly requested ``min_evals`` is honoured
            # exactly, because silently raising it made "fire the surface after k points"
            # mean something else than the caller asked for.
            if self.min_evals_source == "default" and self.MinEvals < new_n_evals:
                self.MinEvals = new_n_evals
        regressor_type = type(self.regressor)
        self.surrogate_info_ = {
            "regressor": "%s.%s" % (regressor_type.__module__, regressor_type.__qualname__),
            "source": regressor_source,
            "min_evals": int(self.MinEvals),
            "min_evals_source": self.min_evals_source,
            "fit_error_policy": self.surrogate_fit_error,
            "automatic_fallback": None,
            "fit_attempts": 0,
            "acquisition_attempts": 0,
            "fit_failures": 0,
            "prediction_failures": 0,
            "local_optimization_failures": 0,
            "fit_history": [],
        }
        self.surrogate_fit_history_ = self.surrogate_info_["fit_history"]
        self.other = kwargs

    def __checkpoint_path(self):
        """Path of this run's checkpoint, or ``None`` when persistence is off."""
        from os.path import join

        if not self.TaskName:
            return None
        if self.checkpoint_dir:
            return join(self.checkpoint_dir, self.TaskName + ".pkl")
        return self.TaskName + ".pkl"

    def __save(self):
        """
        Logs state of the optimization progress at each iteration
        :return: None
        """
        import os
        from pickle import dumps
        from tempfile import mkstemp

        path = self.__checkpoint_path()
        if path is None:
            # No task name: a library user's working directory stays untouched.
            return
        data2store = dict(
            iteration=self.iteration,
            n_rounds=self.n_rounds,
            evaluated=self.evaluated,
            current=self.current,
            current_val=self.current_val,
            sampling_sources=self.sampling_sources_,
        )
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        # Atomic replace: a crash or a parallel reader must never see a half-written
        # checkpoint (the previous version truncated the real file in place).
        handle, temp_path = mkstemp(dir=directory, prefix=".checkpoint-", suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(dumps(data2store))
            os.replace(temp_path, path)
        except BaseException:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def __load(self):
        """
        Loads previous information saved, if any
        :return: None
        """
        from pickle import loads

        restored_data = {}
        path = self.__checkpoint_path()
        if path is not None:
            try:
                fl = open(path, "rb")
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
        # Sampling provenance is only restored when it aligns one-to-one with the
        # restored evaluations; checkpoints written before this key existed carry
        # no provenance, and inventing tags for them would be a false record.
        restored_sources = restored_data.get("sampling_sources", [])
        if isinstance(restored_sources, list) and len(restored_sources) == len(self.evaluated):
            self.sampling_sources_ = list(restored_sources)
        else:
            self.sampling_sources_ = []

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

            # A method that cannot handle constraints silently ignores them, so a
            # constrained round must never be handed to one (L-BFGS-B used to be on
            # the allowlist and returned infeasible points).
            check_scipy_solver_supports_constraints(
                self.scipy_solver, len(cns) > 0, context="scipy_solver"
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

    def __record_evaluation(self, point, phase, value):
        """Append one objective evaluation and update the incumbent.

        The single place where an evaluation enters the run: it keeps the four
        counters consistent (``evaluated`` and ``sampling_sources_`` grow together,
        which the checkpoint format and the audit bundle both rely on) and it owns
        the no-progress bookkeeping, so no path can evaluate without being counted.
        """
        self.evaluated.append([point, value])
        self.sampling_sources_.append(phase)
        if self.current is None:
            # First incumbent: neither progress nor a failure to improve. Without this
            # branch a run whose every evaluation is +inf (all folds failing) would
            # keep ``current`` at None and crash while sampling around it.
            self.current = point
            self.current_val = value
            return
        if value < self.current_val:
            self.NumIterNoProg = 0
            if self.verbose > 0:
                emit(logging.INFO, "NEW CANDIDATE")
            self.current = point
            self.current_val = value
        else:
            self.NumIterNoProg += 1

    def __evaluate(self, point, phase):
        """Evaluate ``point`` as the round's candidate."""
        self._next_eval_phase = phase
        value = self.objective(point)
        self.__record_evaluation(point, phase, value)

    def __iterate(self, x0):
        """
        Runs one round of the search: explore, or fit the surface and exploit it.

        :param x0: the round's sampled point
        :return: ``str`` the phase that was evaluated, or ``"acquisition_failed"``
            when the round produced no evaluation (surrogate fit, prediction or the
            local optimizer failed and the failure policy is not ``"raise"``).
        """
        from numpy import sqrt, array, isfinite

        X = []
        y = []
        n_ = len(x0)
        for point, value in self.evaluated:
            if not isfinite(value):
                continue
            if self.search_sphere:
                distance = sqrt(sum([(point[i] - x0[i]) ** 2 for i in range(n_)]))
                if distance > self.radius:
                    continue
            X.append(point)
            y.append(value)
        n_close_points = len(X)

        if n_close_points < self.MinEvals:
            if self.verbose > 1:
                emit(
                    logging.DEBUG,
                    "Not enough points for approximation (%d < %d); exploring."
                    % (n_close_points, self.MinEvals),
                )
            self.__evaluate(x0, "explore")
            self.iteration += 1
            return "explore"

        if self.verbose > 1:
            emit(
                logging.DEBUG,
                "Found enough number of points (%d) for approximation."
                % n_close_points,
            )
        fit_record = {
            "iteration": int(self.iteration),
            "training_points": int(n_close_points),
            "fit_status": "started",
            "optimization_status": "not_run",
        }
        self.surrogate_fit_history_.append(fit_record)
        self.surrogate_info_["fit_attempts"] += 1

        def __degrade(stage, exc):
            """Record a failed round and fall back to plain exploration."""
            self.surrogate_info_["acquisition_attempts"] += 1
            self.NumFailedLocOptim += 1
            if self.surrogate_fit_error == "raise":
                raise exc
            emit(
                logging.WARNING,
                "Surrogate %s failed at round %d (%s: %s); exploring instead "
                "(surrogate_fit_error='degrade')."
                % (stage, self.iteration, type(exc).__name__, exc),
            )
            # Same phase as a plain exploration round: the degradation is recorded
            # in surrogate_fit_history_, and the public phase vocabulary stays the
            # documented ``init`` / ``explore`` / ``surrogate_exploit``.
            self.__evaluate(x0, "explore")
            self.iteration += 1
            return "explore"

        try:
            self.regressor.fit(array(X), array(y))
        except Exception as exc:
            fit_record.update(
                fit_status="failed",
                failure_stage="fit",
                failure_type=type(exc).__name__,
            )
            self.surrogate_info_["fit_failures"] += 1
            return __degrade("fit", exc)
        fit_record["fit_status"] = "complete"
        # In-sample fidelity of the surrogate itself. Records how well the response
        # surface explains the observed scores so far; a new key only, existing
        # fit_record keys are left untouched.
        surrogate_scorer = getattr(self.regressor, "score", None)
        fit_score = None
        if callable(surrogate_scorer):
            try:
                candidate_fit_score = float(surrogate_scorer(array(X), array(y)))
            except Exception:
                candidate_fit_score = None
            if candidate_fit_score is not None and isfinite(candidate_fit_score):
                fit_score = candidate_fit_score
        fit_record["fit_score"] = fit_score

        def apprx(x):
            try:
                prediction = array(self.regressor.predict(array([x])))
                if prediction.size != 1 or not isfinite(float(prediction.reshape(-1)[0])):
                    raise ValueError(
                        "Surrogate predictions must contain exactly one finite value"
                    )
            except Exception as exc:
                fit_record.update(
                    status="failed",
                    failure_stage="predict",
                    failure_type=type(exc).__name__,
                )
                self.surrogate_info_["prediction_failures"] += 1
                emit(
                    logging.ERROR,
                    "Surrogate prediction failed at round %d: %s: %s"
                    % (self.iteration, type(exc).__name__, exc),
                )
                raise
            return float(prediction.reshape(-1)[0])

        cns = []
        if self.search_sphere:
            r = self.radius * (self.contraction ** int(self.NumFailedLocOptim / 2))
            cns_f = lambda x, x_=x0, r_=r: r_ ** 2 - sum(
                [(x[i] - x_[i]) ** 2 for i in range(n_)]
            )
            cns = [{"type": "ineq", "fun": cns_f}]
        for inq in self.ineqs:
            cns.append({"type": "ineq", "fun": inq})
        self.surrogate_info_["acquisition_attempts"] += 1
        try:
            min_res = self.__round_optim(apprx, x0, cns)
        except Exception as exc:
            if "failure_stage" not in fit_record:
                fit_record.update(
                    status="failed",
                    failure_stage="local_optimization",
                    failure_type=type(exc).__name__,
                )
                self.surrogate_info_["local_optimization_failures"] += 1
            return __degrade("local optimization", exc)

        if not min_res.success:
            # A round that produced no evaluation still advances the run: it is
            # charged to the no-progress counter, so a surface that keeps failing
            # can never consume the budget silently.
            fit_record["optimization_status"] = "failed"
            fit_record["optimizer_status"] = getattr(min_res, "status", None)
            self.surrogate_info_["local_optimization_failures"] += 1
            self.NumFailedLocOptim += 1
            self.NumIterNoProg += 1
            if self.verbose > 1:
                emit(logging.DEBUG, "The local optimization was not successful.")
            self.iteration += 1
            return "acquisition_failed"

        fit_record["optimization_status"] = "success"
        self.NumFailedLocOptim = 0
        x_0 = min_res.x
        try:
            self._next_eval_phase = "surrogate_exploit"
            val = self.objective(x_0)
        except ValueError as exc:
            # A candidate the objective refuses (domain error) is not an evaluation;
            # the round is charged to the no-progress counter instead.
            self.NumIterNoProg += 1
            self.NumFailedLocOptim += 1
            if self.surrogate_fit_error == "raise":
                raise
            if self.verbose > 1:
                emit(logging.DEBUG, "Error in function evaluation: %s" % (exc,))
            self.iteration += 1
            return "acquisition_failed"
        self.__record_evaluation(x_0, "surrogate_exploit", val)
        if self.verbose > 1:
            emit(
                logging.DEBUG,
                """Minimum point of the approximation: %s;\nThe objective's value: %f;\nThe surrogate
                 value %f"""
                % (str(x_0), val, min_res.fun),
            )
        if self.verbose > 1:
            emit(logging.DEBUG, "==========" * 8)
        self.iteration += 1
        return "surrogate_exploit"

    def __budget(self):
        """Effective evaluation budget: ``min(n_iter, max_evals)``.

        ``n_iter`` is the number of candidate evaluations the caller asked for and
        ``max_evals`` a hard cap, so a cap below the request binds first and a cap
        above it does not silently extend the run.
        """
        if self.max_evals is None:
            return int(self.MaxIter)
        return int(min(int(self.MaxIter), int(self.max_evals)))

    def __summarise(self, started):
        from time import monotonic

        elapsed = monotonic() - started
        self.n_rounds = int(self.iteration)
        self.summary_ = {
            "termination_reason": self.termination_reason,
            "status": "completed" if self.termination_reason in ("n_iter", "max_evals") else "terminated",
            "budget": {
                "time_limit": self.time_limit,
                "n_iter": int(self.MaxIter),
                "max_evals": self.max_evals,
                "cpu_limit": self.cpu_limit,
                "memory_limit": self.memory_limit,
                "elapsed_seconds": elapsed,
                "rounds": int(self.iteration),
                "evaluations": len(self.evaluated),
                "acquisition_calls": int(self.surrogate_info_.get("acquisition_attempts", 0)),
                "surrogate_fits": int(self.surrogate_info_.get("fit_attempts", 0)),
            },
            "current_value": getattr(self, "current_val", None),
            "surrogate": self.surrogate_info_,
        }
        fit_failures = int(self.surrogate_info_.get("fit_failures", 0))
        prediction_failures = int(self.surrogate_info_.get("prediction_failures", 0))
        local_optimization_failures = int(
            self.surrogate_info_.get("local_optimization_failures", 0)
        )
        if fit_failures or prediction_failures or local_optimization_failures:
            # One aggregated line instead of one per event: the per-event detail
            # already lives in surrogate_fit_history_, so warning on every failure
            # would flood a long run without adding information.
            emit(
                logging.WARNING,
                "Surrogate diagnostics: %d fit failure(s), %d prediction failure(s), "
                "%d local-optimization failure(s) across %d fit attempt(s)."
                % (
                    fit_failures,
                    prediction_failures,
                    local_optimization_failures,
                    int(self.surrogate_info_.get("fit_attempts", 0)),
                ),
            )

    def __call__(self, *args, **kwargs):
        """
        Runs the structured random search.
        :return: the best minimum point and value
        """
        from math import log
        from time import monotonic

        started = monotonic()
        self.termination_reason = None
        self.summary_ = {}
        self.__optim_param()
        budget = self.__budget()
        if self.verbose > 0:
            # Kept as "Iteration m / n" for console compatibility; ``m`` is the
            # number of evaluations already spent and ``n`` the evaluation budget.
            emit(logging.INFO, "Iteration {m} / {n}".format(m=len(self.evaluated), n=budget))
        # Rounds are not evaluations: a round whose acquisition fails spends none, so
        # a permanently failing surface would spin forever without this ceiling.
        max_rounds = 2 * budget + 10
        try:
            if self.x0_pending and not self.evaluated:
                self.x0_pending = False
                self.__evaluate(self.x0, "init")
            while len(self.evaluated) < budget:
                if self.time_limit is not None and monotonic() - started >= self.time_limit:
                    self.termination_reason = "time_limit"
                    break
                if self.iteration >= max_rounds:
                    self.termination_reason = "acquisition_failed"
                    break
                if self.verbose > 1:
                    emit(logging.DEBUG, "Round # %d" % self.iteration)
                    emit(logging.DEBUG, "----------" * 8)
                new_point = self.Sampling.sample(
                    self.current,
                    cntrctn=self.contraction
                    ** (self.NumFailedLocOptim + log(self.iteration + 1)),
                )
                self.__iterate(new_point)
                if self.verbose > 0:
                    emit(logging.INFO, "End of iteration.")
                if self.NumIterNoProg > self.MaxIterNoProg:
                    emit(
                        logging.WARNING,
                        "Search stopped after %d rounds without improvement."
                        % self.NumIterNoProg,
                    )
                    self.termination_reason = "max_iter_no_progress"
                    break
                self.__save()
            if self.termination_reason is None:
                # Documented precedence: when the budget runs out *and* the clock is
                # already past the limit at that moment, time was the binding
                # constraint (a zero or very small time_limit must report itself
                # rather than being masked by an equally exhausted evaluation budget).
                if self.time_limit is not None and monotonic() - started >= self.time_limit:
                    self.termination_reason = "time_limit"
                else:
                    budget_is_max_evals = (
                        self.max_evals is not None
                        and int(self.max_evals) <= int(self.MaxIter)
                    )
                    self.termination_reason = (
                        "max_evals" if budget_is_max_evals else "n_iter"
                    )
        except SearchStopped:
            # A pause/cancel is not a failure: record where it landed, then let the
            # caller persist and offer a resume.
            self.termination_reason = "stopped"
            self.__summarise(started)
            raise
        self.__summarise(started)
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


def as_space(value):
    """Coerce shorthand into a search-space object.

    ``(a, b)`` tuples become ``Integer`` (both ints) or ``Real``; any other list
    or tuple becomes ``Categorical``. Existing space objects pass through.
    """
    from numpy import floating, integer

    if getattr(value, "VType", None) is not None:
        return value
    if isinstance(value, tuple) and len(value) == 2 and all(
        isinstance(v, (int, float, integer, floating)) and not isinstance(v, bool)
        for v in value
    ):
        if all(isinstance(v, (int, integer)) for v in value):
            return Integer(*value)
        return Real(*value)
    if isinstance(value, (list, tuple, set)):
        return Categorical(list(value))
    return value


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
    :param param_distributions: dict Scikit-learn style name for ``params``; either may be given.
    :param n_iter: int Scikit-learn style name for ``max_iter``; either may be given.
    :param params: dict Dictionary with parameters names (string) as keys and domains as lists of parameter ranges
        to try. Domains are either lists of categorical (string) values or 2 element lists specifying a min and max
        for integer or float parameters
    :param scoring: string, scorer object or `None`, default=None
        A scorer name (see model evaluation documentation) or a
        ``make_scorer``/``get_scorer`` object. If ``None``, the ``score``
        method of the estimator is used. A *bare* callable is rejected: it
        carries no direction, and the search would minimize its negation —
        maximizing a loss. Wrap it: ``make_scorer(fn, greater_is_better=False)``.
    :param max_iter: int, default=50
        **Number of candidate evaluations** (scikit-learn's ``n_iter``). The initial
        candidate counts toward it; with ``max_evals`` unset, a search records exactly
        ``n_iter`` evaluations. ``n_iter`` is an accepted alias.
    :param min_evals: int, default=25; Number of random evaluations before employing an approximation for the
        response surface.
    :param regressor: scikit-learn-style regressor used to fit the surrogate response surface;
        default since v0.2 is an un-tuned RBF `GaussianProcessRegressor` (see
        `benchmarks/DESIGN.md` sec. 12), pass e.g. `HilbertRegressor(deg=3)` for the legacy cubic polynomial.
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
    :param random_state: int, ``numpy.random.Generator``, ``numpy.random.RandomState`` or None, default=None
        Seeds two independent streams - the starting point (``init_candidate``) and the
        sampling path (``sampling``) - derived from one :class:`numpy.random.SeedSequence`.
        Same seed, same data, same budget and same library versions reproduce the same
        candidate sequence, in one process and across processes. Fitting never touches
        the global ``random``/``numpy.random`` state. The *estimator's* randomness is not
        seeded by the search: pass ``estimator__random_state`` if it is stochastic.
    :param task_name: str, optional; enables checkpoint/resume. When omitted (default)
        nothing is written to disk.
    :param checkpoint_dir: str, optional; directory for that checkpoint.
    :param surrogate_fit_error: ``"degrade"`` (default) or ``"raise"``; see
        :class:`SurrogateSearch`.
    :param callbacks: callable or iterable of callables, default=None
        Called with one event dictionary per evaluated candidate: the trial record
        plus a zero-based ``trial`` index and the optimizer's ``phase``
        (``init`` / ``explore`` / ``surrogate_exploit``, or ``None`` when unknown).
        A callback that raises is logged and skipped; it never aborts the search.
        See :doc:`telemetry` for the event contract and ``to_audit_dict`` for the
        end-of-run bundle.
    """

    def __init__(
            self,
            estimator,
            params=None,
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
            task_name=None,
            checkpoint_dir=None,
            surrogate_fit_error="degrade",
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
            continue_=None,
            param_distributions=None,
            n_iter=None,
            callbacks=None,
            stop_check=None,
    ):
        resolved_continue = _resolve_continue_flag(Continue, continue_)
        if param_distributions is not None and params is not None and params != param_distributions:
            raise ValueError("params and param_distributions cannot both be given")
        space = param_distributions if param_distributions is not None else params
        if space is None:
            raise ValueError("param_distributions (or its alias params) is required")
        if n_iter is not None and max_iter != 50 and max_iter != n_iter:
            raise ValueError("max_iter and n_iter cannot specify conflicting values")
        self.param_space_ = {name: as_space(v) for name, v in space.items()}
        self.param_distributions = param_distributions
        self.n_iter = n_iter
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
        self.params_list = list(space.keys())
        self.max_iter = max_iter
        self.max_iter_ = n_iter if n_iter is not None else max_iter
        self.min_evals = min_evals
        self.radius = radius
        self.contraction = contraction
        self.regressor = regressor
        self.sampling = sampling
        self.search_sphere = search_sphere
        self.optimizer = optimizer
        self.scipy_solver = scipy_solver
        self.task_name = task_name
        self.checkpoint_dir = checkpoint_dir
        self.surrogate_fit_error = surrogate_fit_error
        self.warm_start = warm_start
        self.Continue = resolved_continue
        self.max_itr_no_prog = max_itr_no_prog
        self.time_limit = time_limit
        self.max_evals = max_evals
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.prune_threshold = prune_threshold
        self.min_partial_folds = min_partial_folds
        self.forbidden = tuple(forbidden)
        self.conditional = conditional if isinstance(conditional, dict) else dict(conditional)
        self.random_state = random_state
        self.callbacks = callbacks
        #: Optional ``Callable[[], bool]``; returns True when the caller wants
        #: the search to stop at the next safe point (a candidate boundary).
        self.stop_check = stop_check
        self.bounds = []
        if self.cpu_limit is not None:
            limit = int(self.cpu_limit)
            if limit < 1:
                raise ValueError("cpu_limit must be a positive integer")
            if self.n_jobs == -1 or self.n_jobs > limit:
                self.n_jobs = limit
        self.ineqs = ineqs
        # A bounds-only method would drop those constraints, so reject it here rather
        # than letting the first surrogate round hand SciPy an argument it ignores.
        check_scipy_solver_supports_constraints(
            self.scipy_solver,
            self.optimizer == "scipy" and (bool(self.ineqs) or bool(search_sphere)),
            context="SurrogateRandomCV(scipy_solver=...)",
        )
        self.init = init if init is not None else {}
        self.OPTIM = None
        self.scorer_ = {}
        self.x0 = ()
        self.best_estimator_ = None
        self.best_estimator_score = 0.0
        self.best_score_ = float("-inf")
        self.evaluation_history_ = []
        self.cv_results_ = {}
        self.termination_reason = None
        self.summary_ = {}
        self.failure_summary_ = {}
        self.surrogate_info_ = {}

    @property
    def continue_(self):
        return self.Continue

    @continue_.setter
    def continue_(self, value):
        self.Continue = value

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
            parameter = self.param_space_[name]
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
                if hasattr(parameter, "rvs"):
                    raise ValueError(
                        "Search parameter %r is a scipy.stats distribution, which this "
                        "search does not sample from. Pass an explicit list, a "
                        "(low, high) tuple, or Integer/Real/Categorical." % (name,)
                    )
                raise ValueError(
                    "Unsupported search space type for %r: %r"
                    % (name, getattr(parameter, "VType", type(parameter).__name__))
                )
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

    @staticmethod
    def _rule_value_matches(actual, expected):
        """Compare a candidate's parameter value against a rule's expected value.

        A scalar is compared with equality; a list/tuple/set of values matches
        when the actual value is any one of them (the ``depends_on`` contract:
        "allowed values").
        """
        if isinstance(expected, (list, tuple, set)):
            return any(actual == item for item in expected)
        return actual == expected

    def _forbidden_reason(self, parameters):
        for rule in self.forbidden:
            if callable(rule):
                matched = rule(parameters)
            elif isinstance(rule, dict):
                matched = all(
                    self._rule_value_matches(parameters.get(name), value)
                    for name, value in rule.items()
                )
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
            return all(
                SurrogateRandomCV._rule_value_matches(parameters.get(name), value)
                for name, value in rule.items()
            )
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
            if hasattr(transformer, "fit_resample") and not hasattr(transformer, "transform"):
                # A sampler (e.g. imblearn's SMOTE) resamples rows and has no
                # ``transform``; under ``imblearn.pipeline.Pipeline`` it is routed
                # through ``fit_resample`` and cannot remove features. Treating it as
                # broken preprocessing rejected every sampler-first pipeline.
                continue
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
                # An intermediate step must be usable in a pipeline regardless of
                # whether it builds on sklearn (``transform``) or imblearn
                # semantics (``fit_resample`` for samplers such as SMOTE). The
                # pipeline object itself is an ``imblearn.pipeline.Pipeline``
                # whenever samplers are present (see ``aml.optimize_pipeline``),
                # so a raw sampler is valid mid-pipeline even though it has no
                # ``transform``. Without this, a legitimate ``[SMOTE, estimator]``
                # candidate was falsely rejected and aborted the whole search.
                if transformer is None or not hasattr(transformer, "fit") or (
                    not hasattr(transformer, "transform")
                    and not hasattr(transformer, "fit_resample")
                ):
                    raise ValueError(
                        "Pipeline intermediate step %r must implement fit and transform"
                        % name
                    )
            elif transformer is not None and not hasattr(transformer, "fit"):
                raise ValueError("Pipeline final step %r must implement fit" % name)

    def _format_cv_results(self, test_scores, ranks):
        """Assemble ``cv_results_`` in scikit-learn's column layout.

        The ``param_*``, ``mean_test_score``, ``std_test_score``,
        ``split<i>_test_score``, ``rank_test_score`` and ``mean_fit_time`` columns
        follow the ``BaseSearchCV`` contract, so tooling that reads search results
        works against this class. The remaining columns (``loss``, ``status``,
        ``error``, ``duration``, ``failure_records``) are sksurrogate-specific and
        report the trial-level diagnostics the search records.
        """
        from numpy import nan, nanstd

        history = self.evaluation_history_
        n_trials = len(history)
        split_columns = {}
        for position, item in enumerate(history):
            for fold, fold_score in enumerate(item.get("fold_scores") or []):
                key = "split%d_test_score" % fold
                column = split_columns.setdefault(key, [nan] * n_trials)
                column[position] = fold_score
        std_test_score = []
        mean_fit_time = []
        for item in history:
            fold_scores = [value for value in (item.get("fold_scores") or []) if value is not None]
            std_test_score.append(
                float(nanstd(fold_scores)) if len(fold_scores) > 1 else nan
            )
            fold_times = [
                value for value in (item.get("fold_fit_times") or []) if value is not None
            ]
            mean_fit_time.append(
                float(sum(fold_times)) / len(fold_times) if fold_times else nan
            )
        results = {
            "params": [item["params"] for item in history],
            "mean_test_score": test_scores,
            "std_test_score": std_test_score,
            "rank_test_score": ranks,
            "mean_fit_time": mean_fit_time,
            "loss": [item["score"] for item in history],
            "status": [item["status"] for item in history],
            "duration": [item["duration"] for item in history],
            "error": [item["error"] for item in history],
            "failure_records": [item.get("failure_records", []) for item in history],
        }
        for name in self.params_list:
            results["param_%s" % name] = [item["params"].get(name) for item in history]
        results.update(split_columns)
        return results

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
        from numpy import array, unique, sqrt
        from sklearn.base import clone, is_classifier
        from sklearn.metrics import check_scoring
        from sklearn.model_selection import check_cv
        from sklearn.model_selection._validation import _fit_and_score

        self._validate_search_space()
        self._validate_pipeline_structure()
        if isinstance(self.scoring, dict):
            raise ValueError(
                "multi-metric scoring (scoring=dict) is not supported by "
                "SurrogateRandomCV: pass one scorer name or make_scorer object. A "
                "multi-metric search needs a `refit=<metric>` contract that this "
                "searcher does not implement, and silently scoring one metric while "
                "reporting another is worse than refusing."
            )
        # ``fit``'s keyword arguments (``sample_weight`` and friends) were accepted and
        # then dropped: every fold and the refit received only ``self.fit_params``.
        # Merge once, use everywhere, and keep the merged mapping for introspection.
        self.fit_params_ = dict(self.fit_params or {})
        self.fit_params_.update(fit_params or {})
        callbacks = self.callbacks
        if callbacks is None:
            callbacks = []
        elif callable(callbacks):
            callbacks = [callbacks]
        else:
            callbacks = list(callbacks)
        # from lightgbm.sklearn import LightGBMError
        radius_list = []
        # A bare callable has no sign: negating it would turn a loss into a score
        # to maximize. Fail here, before any fit, with guidance.
        # Resolved once into fitted attributes: assigning back onto ``self.scoring``
        # made ``get_params`` differ before and after ``fit`` and broke ``clone``.
        self.scoring_ = check_scoring_direction(self.scoring)
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring_)
        # Two independent streams: the starting point and the sampling path.
        self.rng_streams_ = make_rng_streams(self.random_state)
        init_rng = self.rng_streams_["init_candidate"]
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
            param_num_range = self.param_space_[param]
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
                    x0_.append(init_rng.uniform(lower, upper))
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
                            init_rng.uniform(
                                param_num_range.bound_tuple[i][0],
                                param_num_range.bound_tuple[i][1],
                            )
                        )
                self.bounds = self.bounds + list(param_num_range.bound_tuple)
        # ``radius`` is an __init__ parameter: resolving it must not overwrite it,
        # or ``get_params`` changes between two ``clone`` calls around a fit.
        radius = self.radius
        if radius is None:
            rds = 0.0
            for r in radius_list:
                rds += r ** 2
            radius = sqrt(rds)
        self.radius_ = radius
        if int(self.min_evals) > int(self.max_iter_):
            from warnings import warn

            warn(
                "min_evals=%d exceeds the evaluation budget n_iter=%d: the response "
                "surface can never fire, so this search is a pure random-search "
                "ablation. Set n_iter > min_evals for a surrogate-assisted search."
                % (int(self.min_evals), int(self.max_iter_)),
                UserWarning,
                stacklevel=2,
            )
            self._surface_never_fires = True
        else:
            self._surface_never_fires = False
        self.x0 = array(x0_)
        self.bounds = tuple(self.bounds)
        cv_dat = list(cv.split(X, y, groups))

        def _notify_trial(record):
            """Deliver one per-trial event to every registered callback.

            ``trial`` is the record's zero-based position in
            ``evaluation_history_``; ``phase`` is the optimizer's sampling tag
            for this objective call (``None`` when unavailable). A callback that
            raises is logged and skipped so a broken callback can never abort a
            search.
            """
            event = dict(record)
            event["trial"] = len(self.evaluation_history_) - 1
            event["phase"] = getattr(self.OPTIM, "_next_eval_phase", None)
            for callback in callbacks:
                try:
                    callback(event)
                except Exception:
                    logger.error("SurrogateRandomCV trial callback failed", exc_info=True)

        def obj(x):
            from time import perf_counter

            if self.stop_check is not None and self.stop_check():
                raise SearchStopped("Search stopped by request")
            started = perf_counter()
            cand_params = {}
            _idx = 0
            for _param in self.params_list:
                _param_num_range = self.param_space_[_param]
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
                record = {
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
                self.evaluation_history_.append(record)
                _notify_trial(record)
                return float("inf")
            zero_feature_reason = self._zero_feature_reason(cand_params, X, y)
            if zero_feature_reason is not None:
                record = {
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
                self.evaluation_history_.append(record)
                _notify_trial(record)
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
                        return_times=True,
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
                        "fit_time": _score.get("fit_time"),
                        "failure_type": None,
                        "failure_message": None,
                        "failure_traceback": None,
                        "fold": fold,
                    }
                except Exception as exc:
                    import traceback

                    if self.verbose > 1:
                        emit(logging.DEBUG, "Model evaluation error")
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
                                                                fit_params_=self.fit_params_,
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
                record = {
                    "params": cand_params.copy(),
                    "score": None,
                    "status": "pruned",
                    "duration": perf_counter() - started,
                    "fold_duration": sum(durations),
                    "error": "; ".join(errors) if errors else None,
                    "partial_folds": n_test,
                    "failure_records": failure_records,
                }
                self.evaluation_history_.append(record)
                _notify_trial(record)
                return float("inf")
            record = {
                "params": cand_params.copy(),
                "score": -score if n_test else None,
                "status": "complete" if n_test else "failed",
                "duration": perf_counter() - started,
                "fold_duration": sum(durations),
                "error": "; ".join(errors) if errors else None,
                "partial_folds": n_test,
                "failure_records": failure_records,
                # Per-fold detail: cv_results_ mirrors scikit-learn's split columns.
                "fold_scores": [result["score"] for result in scores],
                "fold_fit_times": [result.get("fit_time") for result in scores],
            }
            self.evaluation_history_.append(record)
            _notify_trial(record)
            if n_test:
                return -score
            # No fold produced a finite score (every fit/scoring raised — e.g. a
            # searched parameter compiling to a value the estimator rejects, like
            # rgf's reg_depth=0). Report +inf, not -0.0, so the minimizing search
            # and the best-estimator fallback below never treat a failed trial as
            # a zero-error optimum: for a regression loss (+RMSE) 0.0 is the
            # global minimum and would make the search converge to an infeasible
            # region, crashing the refit (see the forbidden/zero-feature returns).
            return float("inf")

        self.OPTIM = SurrogateSearch(
            obj,
            x0=self.x0,
            max_iter=self.max_iter_,
            min_evals=self.min_evals,
            ineqs=self.ineqs,
            bounds=self.bounds,
            verbose=self.verbose,
            radius=self.radius_,
            rng=self.rng_streams_["sampling"],
            surrogate_fit_error=self.surrogate_fit_error,
            checkpoint_dir=self.checkpoint_dir,
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
        self.surrogate_info_ = self.OPTIM.surrogate_info_
        # A surface that cannot fire inside the budget is a documented, reported state
        # (it is a legitimate random-search ablation), not something to infer silently.
        self.surrogate_info_["surface_never_fires"] = bool(self._surface_never_fires)
        x, scr = self.OPTIM()
        self.n_evaluations_ = len(self.evaluation_history_)
        self.n_rounds_ = int(getattr(self.OPTIM, "n_rounds", 0) or 0)
        self.n_surrogate_fits_ = int(self.surrogate_info_.get("fit_attempts", 0))
        best_params_ = {}
        idx = 0
        for param in self.params_list:
            param_num_range = self.param_space_[param]
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
        # Mirror the trial-level conditioning so set_params/refit use exactly the
        # parameters CV actually scored; conditionally-inactive parameters were
        # dropped from every trial, so decoding them back would refit a model the
        # reported score never saw.
        best_params_ = self._apply_conditional_parameters(best_params_)
        # The optimizer's solution can land inside a region every trial rejected
        # (a forbidden combination, or a candidate whose folds all failed).
        # Refitting those settings either raises (e.g. an invalid
        # solver/regularization combination) or silently returns a banned
        # model, so fall back to the best trial that was actually completed;
        # when no trial was, skip the refit entirely.
        from numpy import isfinite

        feasible = [
            item
            for item in self.evaluation_history_
            if item.get("score") is not None and isfinite(item["score"])
        ]
        self.termination_reason = getattr(self.OPTIM, "termination_reason", None)
        self.summary_ = getattr(self.OPTIM, "summary_", {})
        losses = [item["score"] for item in self.evaluation_history_]
        test_scores = [-v if v is not None and isfinite(v) else float("nan") for v in losses]
        ranked = sorted(
            (i for i, v in enumerate(test_scores) if isfinite(v)), key=lambda i: -test_scores[i]
        )
        ranks = [len(ranked) + 1] * len(test_scores)
        for position, index in enumerate(ranked, start=1):
            ranks[index] = position
        self.best_index_ = ranked[0] if ranked else None
        self.cv_results_ = self._format_cv_results(test_scores, ranks)
        if not feasible:
            self.best_params_ = {}
            self.best_estimator_score = float("inf")
            self.best_score_ = float("-inf")
            if self.refit is not False:
                # Nothing was feasible. Expose an *unfitted* clone so callers can still
                # inspect the searched pipeline's structure, and flag it: a placeholder
                # must never be mistaken for a fitted model (``best_score_`` is -inf
                # and ``best_estimator_is_placeholder`` is True).
                self.best_estimator_ = clone(self.estimator)
                self.best_estimator_is_placeholder = True
        else:
            if not isfinite(scr) or self._forbidden_reason(best_params_) is not None:
                best_trial = min(feasible, key=lambda item: item["score"])
                best_params_ = {
                    name: value
                    for name, value in best_trial["params"].items()
                    if name in self.params_list
                }
                scr = best_trial["score"]
            self.best_params_ = dict(best_params_)
            # best_estimator_score is the minimized loss; best_score_ follows the
            # scikit-learn convention (higher is better).
            self.best_estimator_score = scr
            self.best_score_ = -scr
            self.best_estimator_is_placeholder = False
            refit_kwargs = dict(self.fit_params_)
            if callable(self.refit):
                # scikit-learn's callable refit receives ``cv_results_`` and returns
                # the estimator to store; it was previously treated as ``True``.
                self.best_estimator_ = self.refit(self.cv_results_)
            elif self.refit is not False:
                # Refit the best estimator on the full dataset so it can be used for
                # prediction/scoring (mirrors sklearn's BaseSearchCV refit behavior;
                # uses the same fit_params as the CV folds above).
                best_estimator = clone(self.estimator).set_params(**best_params_)
                best_estimator.fit(X, y, **refit_kwargs)
                self.best_estimator_ = best_estimator
            elif hasattr(self, "best_estimator_"):
                # ``refit=False`` means "do not fit a final model": no attribute, as
                # in scikit-learn, so ``predict`` raises instead of running unfitted.
                del self.best_estimator_
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

    @staticmethod
    def _json_safe(value):
        """Coerce ``value`` into JSON-native data (dict/list/str/float/None)."""
        from math import isfinite

        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return value if isfinite(value) else None
        if isinstance(value, dict):
            return {
                str(key): SurrogateRandomCV._json_safe(item) for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [SurrogateRandomCV._json_safe(item) for item in value]
        # numpy scalars and other 0-d wrappers expose ``item``; fall back to str.
        converter = getattr(value, "item", None)
        if callable(converter):
            try:
                return SurrogateRandomCV._json_safe(converter())
            except Exception:
                pass
        return str(value)

    def _encoded_search_space(self):
        """Per-parameter search-space encoding used by :meth:`to_audit_dict`."""
        space = {}
        for name in self.params_list:
            parameter = self.param_space_[name]
            value_type = getattr(parameter, "VType", None)
            if value_type == "categorical":
                space[name] = {
                    "type": "categorical",
                    "items": self._json_safe(list(getattr(parameter, "items", []))),
                }
            elif value_type == "hdreal":
                space[name] = {
                    "type": "hdreal",
                    "bounds": self._json_safe(list(parameter.bound_tuple)),
                }
            else:
                lower, upper = self._encoded_bounds(parameter)
                space[name] = {
                    "type": value_type,
                    "bounds": [float(lower), float(upper)],
                    "scale": getattr(parameter, "scale", "linear"),
                }
        return space

    def to_audit_dict(self):
        """Return a JSON-safe audit bundle describing the search and its trials.

        The bundle is the contract between this MIT-licensed core and any
        consumer (control plane, notebook, archive): it carries the estimator and
        its parameters, the encoded search space, the scoring/CV summary, the
        budget and termination state, every trial record with its zero-based
        ``trial`` index, the surrogate diagnostics, and the sampling provenance.

        ``json.dumps(search.to_audit_dict())`` succeeds without a custom
        encoder; values that are not JSON-native (estimator objects, numpy
        scalars, splitters) are coerced to their string form or ``None``. Works
        before ``fit`` as well, with empty evaluation lists.
        """
        estimator_type = self.estimator.__class__
        summary = self.summary_ or {}
        budget = summary.get("budget")
        if not isinstance(budget, dict):
            budget = {}
        evaluations = []
        for index, item in enumerate(self.evaluation_history_):
            record = dict(item)
            record["trial"] = index
            evaluations.append(self._json_safe(record))
        optimizer = getattr(self, "OPTIM", None)
        sampling_sources = getattr(optimizer, "sampling_sources_", None)
        return {
            "schema_version": 1,
            "estimator": {
                "class": "%s.%s" % (estimator_type.__module__, estimator_type.__qualname__),
                "params": self._json_safe(self.get_params(deep=False)),
            },
            "search_space": self._encoded_search_space(),
            "scoring": self._json_safe(self.scoring),
            "cv": self._json_safe(self.cv),
            "n_iter": self._json_safe(self.n_iter),
            "max_iter": self._json_safe(self.max_iter),
            "random_state": self._json_safe(self.random_state),
            "budget": self._json_safe(budget),
            "termination": {
                "termination_reason": self._json_safe(summary.get("termination_reason")),
                "status": self._json_safe(summary.get("status")),
            },
            "evaluations": evaluations,
            "surrogate": self._json_safe(self.surrogate_info_),
            "sampling_sources": self._json_safe(list(sampling_sources or [])),
        }
