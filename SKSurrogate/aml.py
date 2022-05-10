"""
Optimized Pipeline Detector
===============================

"""
try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ModuleNotFoundError:
    BaseEstimator = type("BaseEstimator", (object,), dict())
    TransformerMixin = type("TransformerMixin", (object,), dict())


class StackingEstimator(BaseEstimator, TransformerMixin):
    """
    Meta-transformer for adding predictions and/or class probabilities as synthetic feature(s).

    :param estimator: object with fit, predict, and predict_proba methods. The estimator to generate
        synthetic features from.
    :param res: True (default), stacks the final result of estimator
    :param probs: True (default), stacks probabilities calculated by estimator
    :param decision: True (default), stacks the result of decision function of the estimator
    """

    def __init__(self, estimator, res=True, probs=True, decision=True):

        self.estimator = estimator
        self.res = res
        self.probs = probs
        self.decision = decision

    def set_params(self, **params):
        self.estimator.set_params(**params)
        return self

    def fit(self, X, y=None, **fit_params):
        """
        Fit the StackingEstimator meta-transformer.

        :param X: array-like of shape (n_samples, n_features). The training input samples.
        :param y: array-like, shape (n_samples,). The target values (integers that correspond to classes
            in classification, real numbers in regression).
        :param fit_params: Other estimator-specific parameters.
        :return: self, object. Returns a copy of the estimator
        """
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        """
        Transform data by adding two synthetic feature(s).

        :param X: numpy ndarray, {n_samples, n_components}. New data, where n_samples is the number of
            samples and n_components is the number of components.
        :return: X_transformed: array-like, shape (n_samples, n_features + 1) or
            (n_samples, n_features + 1 + n_classes) for classifier with predict_proba attribute;
            The transformed feature set.
        """
        import numpy as np
        from sklearn.utils import check_array

        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if self.probs and hasattr(self.estimator, "predict_proba"):
            X_transformed = np.hstack((X, (self.estimator.predict_proba(X))))

        # add class decision_function as a synthetic feature
        if self.decision and hasattr(self.estimator, "decision_function"):
            X_transformed = np.hstack(
                (
                    X_transformed,
                    np.reshape(self.estimator.decision_function(X), (-1, 1)),
                )
            )

        # add class prediction as a synthetic feature
        if self.res:
            X_transformed = np.hstack(
                (X_transformed, np.reshape(self.estimator.predict(X), (-1, 1)))
            )

        return X_transformed


class Words(object):
    """
    This class takes a set as alphabet and generates words of a given length accordingly.
    A `Words` instant accepts the following parameters:

    :param letters: is a set of letters (symbols) to make up the words
    :param last: a subset of `letters` that are allowed to appear at the end of a word
    :param first: a set of words that can only appear at the beginning of a word
    :param repeat: whether consecutive occurrence of a letter is allowed
    """

    def __init__(self, letters, last=None, first=None, repeat=False):
        self.letters = letters
        self.last = last
        self.first = first
        self.words = []
        self.repeat = repeat

    def _check_cons_repeat(self, o):
        lng = len(o)
        for i in range(1, lng):
            if self.repeat:
                return True
            if o[i - 1] == o[i]:
                return False
        return True

    def _check_mid_first(self, o):
        if (self.first is None) or (self.first == []):
            return True
        n_ = len(o)
        for i in range(1, n_):
            if o[i] in self.first:
                return False
        return True

    def Generate(self, l):
        """
        Generates the set of legitimate words of length `l`

        :param l: int, the length of words
        :return: set of all legitimate words of length `l`
        """
        from itertools import product

        words = []
        for o in product(self.letters, repeat=l):
            if self.last is not None:
                if o[-1] in self.last:
                    if self._check_cons_repeat(o) and self._check_mid_first(o):
                        words.append(o)
            else:
                if self._check_cons_repeat(o) and self._check_mid_first(o):
                    words.append(o)
        return words


try:
    from .structsearch import Real, Integer, Categorical, HDReal
except ModuleNotFoundError:
    Real = lambda a, b: None
    Integer = lambda a, b: None
    Categorical = lambda a: None
    HDReal = lambda a, b: None
except ImportError:
    Real = lambda a, b: None
    Integer = lambda a, b: None
    Categorical = lambda a: None
    HDReal = lambda a, b: None

default_config = {
    # Classifiers
    "sklearn.naive_bayes.BernoulliNB": {
        "alpha": Real(10.0e-5, 100.0),
        "fit_prior": Categorical([True, False]),
    },
    "sklearn.naive_bayes.GaussianNB": {"var_smoothing": Real(1.0e-9, 2.0e-1)},
    "sklearn.tree.DecisionTreeClassifier": {
        "criterion": Categorical(["gini", "entropy"]),
        "splitter": Categorical(["best", "random"]),
        "min_samples_split": Integer(2, 10),
        "min_samples_leaf": Integer(1, 10),
        "class_weight": HDReal((1.0e-5, 1.0e-5), (20.0, 20.0)),
    },
    "sklearn.linear_model.LogisticRegression": {
        "penalty": Categorical(["l1", "l2"]),
        "C": Real(1.0e-6, 10.0),
        "class_weight": HDReal((1.0e-5, 1.0e-5), (20.0, 20.0))
        # 'dual': Categorical([True, False])
    },
    "sklearn.svm.SVC": {
        "C": Real(1e-6, 20.0),
        "gamma": Real(1e-6, 10.0),
        "tol": Real(1e-6, 10.0),
        "class_weight": HDReal((1.0e-5, 1.0e-5), (20.0, 20.0)),
    },
    "sklearn.ensemble.RandomForestClassifier": {
        "n_estimators": Integer(10, 200),
        "criterion": Categorical(["gini", "entropy"]),
        "min_samples_split": Integer(2, 10),
        "min_samples_leaf": Integer(1, 10),
        "class_weight": HDReal((1.0e-5, 1.0e-5), (20.0, 20.0)),
    },
    "sklearn.ensemble.GradientBoostingClassifier": {
        "loss": Categorical(["deviance", "exponential"]),
        "learning_rate": Real(1.0e-6, 1.0 - 1.0e-6),
        "n_estimators": Integer(10, 500),
        "subsample": Real(1.0e-6, 1.0),
        "criterion": Categorical(["friedman_mse", "mse", "mae"]),
        # "min_samples_split": Integer(2, 20),
        # "min_samples_leaf": Integer(1, 20),
        # "min_weight_fraction_leaf": Real(0., .5),
        # "max_depth": Integer(2, 20),
        # "min_impurity_decrease": Real(0., 10.),
        # "min_impurity_split": Real(0., 10.),
        # "max_features": Categorical(['auto', 'sqrt', 'log2', None]),
        "tol": Real(1.0e-6, 0.1),
    },
    "lightgbm.LGBMClassifier": {
        "boosting_type": Categorical(["gbdt", "dart", "goss", "rf"]),
        "num_leaves": Integer(2, 100),
        "learning_rate": Real(1.0e-7, 1.0 - 1.0e-6),
        "n_estimators": Integer(5, 250),
        "min_split_gain": Real(0.0, 1.0),
        # "min_child_weight": Real(1.e-6, 1.),
        # "min_child_samples": Integer(5, 200),
        "subsample": Real(1.0e-6, 1.0),
        # "subsample_freq": Integer(0, 10),
        # "colsample_bytree": Real(1.e-5, 1.),
        # "class_weight": HDReal((1.e-5, 1.e-5), (20., 20.)),
        # "reg_alpha": Real(0., 10.),
        # "reg_lambda": Real(0., 10.),
        "importance_type": Categorical(["split", "gain"]),
    },
    "xgboost.XGBClassifier": {
        "max_depth": Integer(2, 20),
        "learning_rate": Real(1.0e-5, 1.0),
        "n_estimators": Integer(50, 400),
        "objective": Categorical(
            ["binary:logistic", "binary:logitraw", "binary:hinge"]
        ),
        "booster": Categorical(["gbtree", "gblinear", "dart"]),
        "gamma": Real(0.0, 100.0),
        # 'min_child_weight': Real(0., 100.),
        # 'max_delta_step': Real(0., 100.),
        "subsample": Real(1.0e-6, 1.0),
        # 'colsample_bytree': Real(1.e-6, 1.),
        # 'colsample_bylevel': Real(1.e-6, 1.),
        # 'reg_alpha': Real(0., 10.),
        # 'reg_lambda': Real(0., 10.),
        # 'scale_pos_weight': Real(1.e-5, 1.e3),
        # 'base_score': Real(1.e-5, .9999)
    },
    # 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis': {
    #    'tol': Real(1.e-5, 2.)
    # },
    # Feature Selectors
    "sklearn.feature_selection.VarianceThreshold": {"threshold": Real(0.0, 0.3)},
    "sklearn.decomposition.PCA": {
        # 'svd_solver': Categorical(['randomized']),
        "iterated_power": Integer(1, 11),
        "n_components": Integer(2, 30),
    },
    "sklearn.decomposition.TruncatedSVD": {
        "n_components": Integer(2, 30),
        "algorithm": Categorical(["randomized", "arpack"]),
    },
    # Preprocesssors
    "sklearn.preprocessing.StandardScaler": {
        "with_mean": Categorical([True, False]),
        "with_std": Categorical([True, False]),
    },
    "sklearn.preprocessing.Normalizer": {"norm": Categorical(["l1", "l2", "max"])},
    # Transformers
    # 'sklearn.preprocessing.PowerTransformer': {
    #    'method': Categorical(['yeo-johnson', 'box-cox']),
    #    'standardize': Categorical([True, False])
    # },
    "sklearn.kernel_approximation.Nystroem": {
        "kernel": Categorical(["rbf", "poly", "sigmoid"]),
        "gamma": Real(1.0e-2, 10.0),
        "n_components": Integer(10, 100),
    },
    "sklearn.kernel_approximation.RBFSampler": {
        "gamma": Real(1.0e-3, 10.0),
        "n_components": Integer(10, 100),
    },
    "sklearn.kernel_approximation.AdditiveChi2Sampler": {"sample_steps": Integer(1, 3)},
    "sklearn.kernel_approximation.SkewedChi2Sampler": {
        "skewedness": Real(0.1, 5.0),
        "n_components": Integer(10, 100),
    },
    # Sensitivity Analysis
    "sksurrogate.SensAprx": {
        "n_features_to_select": Integer(2, 30),
        "method": Categorical(["sobol", "morris", "delta-mmnt"]),
        # 'regressor': Categorical([None, regressor])
    },
    # Manifold Learning
    # 'sklearn.manifold.Isomap': {
    #    'n_neighbors': Integer(2, 12),
    #    'n_components': Integer(1, 10),
    # },
    # 'sklearn.manifold.LocallyLinearEmbedding': {
    #    'n_neighbors': Integer(2, 12),
    #    'n_components': Integer(1, 10),
    #    'reg ': Real(1.e-5, .2),
    #    'method': Categorical(['standard', 'ltsa'])
    # },
    # 'sklearn.manifold.MDS': {
    #    'n_components': Integer(1, 10),
    #    'n_init': Integer(2, 8),
    # },
    # 'sklearn.manifold.SpectralEmbedding': {
    #    'n_components': Integer(1, 10),
    #    'affinity': Categorical(['nearest_neighbors', 'rbf', 'precomputed'])
    # },
    # 'sklearn.manifold.TSNE': {
    #    'n_components': Integer(1, 10),
    #    'perplexity': Real(5., 50.),
    #    'early_exaggeration': Real(5., 25.),
    #    'learning_rate ': Real(10., 500.)
    # },
    # Sampler
    "imblearn.over_sampling.SMOTE": {"k_neighbors": Integer(3, 10)},
}


class AML(object):
    """
    A class that accepts a nested dictionary with machine learning libraries as its keys
    and a dictionary of their parameters and their ranges as value of each key and finds
    an optimum combination based on training data.

    :param config: A dictionary whose keys are scikit-learn-style objects (as strings) and its corresponding
        values are dictionaries of the parameters and their acceptable ranges/values
    :param length: default=5; Maximum number of objects in generated pipelines
    :param scoring: default='accuracy'; The scoring method to be optimized. Must follow the sklearn scoring signature
    :param cat_cols: default=None; The list of indices of categorical columns
    :param surrogates: default=None; A list of 4-tuples determining surrogates. The first entity of each pair is
        a scikit-learn regressor and the
        2nd entity is the number of iterations that this surrogate needs to be estimated and optimized.
        The 3rd is the sampling strategy and the
        4th is the `scipy.optimize` solver
    :param min_random_evals: default=15; Number of randomly sampled initial values for hyper parameters
    :param cv: default=`ShuffleSplit(n_splits=3, test_size=.25); The cross validation method
    :param check_point: default='./'; The path where the optimization results will be stored
    :param stack_res: default=True; `StackingEstimator`s `res`
    :param stack_probs: default=True; `StackingEstimator`s `probs`
    :param stack_decision: default=True; `StackingEstimator`s `decision`
    :param verbose: default=1; Level of output details
    :param n_jobs: int, default=-1; number of processes to run in parallel
    """

    def __init__(
            self,
            config=None,
            length=5,
            scoring="accuracy",
            cat_cols=None,
            surrogates=None,
            min_random_evals=15,
            cv=None,
            check_point="./",
            stack_res=True,
            stack_probs=True,
            stack_decision=True,
            verbose=1,
            n_jobs=-1
    ):
        from collections import OrderedDict

        if config is None:
            self.config = default_config
        else:
            self.config = config
        self.config_types = {}
        self.couldBlast = []
        self.couldBfirst = []
        self.feature_selector = set()
        self.known_feature_selectors = {
            "sklearn.decomposition.FactorAnalysis",
            "sklearn.decomposition.FastICA",
            "sklearn.decomposition.IncrementalPCA",
            "sklearn.decomposition.KernelPCA",
            "sklearn.decomposition.LatentDirichletAllocation",
            "sklearn.decomposition.MiniBatchDictionaryLearning",
            "sklearn.decomposition.MiniBatchSparsePCA",
            "sklearn.decomposition.NMF",
            "sklearn.decomposition.PCA",
            "sklearn.decomposition.SparsePCA",
            "sklearn.decomposition.TruncatedSVD",
            "sklearn.feature_selection.VarianceThreshold",
            "sklearn.manifold.LocallyLinearEmbedding",
            "sklearn.manifold.Isomap",
            "sklearn.manifold.MDS",
            "sklearn.manifold.SpectralEmbedding",
            "sklearn.manifold.TSNE",
            "sksurrogate.SensAprx",
            "skrebate.ReliefF",
            "skrebate.SURF",
            "skrebate.SURFstar",
            "skrebate.MultiSURF",
            "skrebate.MultiSURFstar",
            "skrebate.TuRF",
        }
        self.letters = self.config_types.keys()
        self.length = length
        self.scoring = scoring
        self.cat_cols = cat_cols
        self.surrogates = surrogates
        self.check_point = check_point
        self.min_random_evals = min_random_evals
        self.stack_res = stack_res
        self.stack_probs = stack_probs
        self.stack_decision = stack_decision
        self.verbose = verbose
        self.num_features = 2
        self.n_jobs = n_jobs
        # TBD: check cv
        if cv is None:
            from sklearn.model_selection import ShuffleSplit

            self.cv = ShuffleSplit(n_splits=3, test_size=0.25)
        else:
            self.cv = cv
        self.types()
        self.words = Words(self.letters, last=self.couldBlast, first=self.couldBfirst)
        self.models = OrderedDict([])
        self.best_estimator_ = None
        self.best_estimator_score = 0.0

    def types(self):
        """
        Recognizes the type of each estimator to determine legitimate placement of each

        :return: None
        """
        from importlib import import_module
        from sklearn.feature_selection import SelectorMixin

        for alg in self.config:
            if alg != "sklearn.pipeline.FeatureUnion":
                detail = alg.split(".")
                module_str = ".".join(detail[:-1])
                module = import_module(module_str)
                clss = module.__getattribute__(detail[-1])
                mdl = clss()
                if hasattr(mdl, "_estimator_type"):
                    self.config_types[alg] = mdl._estimator_type
                    if mdl._estimator_type in ["regressor", "classifier"]:
                        self.couldBlast.append(alg)
                    elif mdl._estimator_type == "sampler":
                        self.couldBfirst.append(alg)
                else:
                    self.config_types[alg] = "transformer"
                if (
                        issubclass(clss, SelectorMixin)
                        or alg in self.known_feature_selectors
                ):
                    self.feature_selector.add(alg)
            else:
                self.config_types[alg] = "transformer"

    def _validate_sequence(self, seq):
        """
        Validates the structure of the building sequence

        :param seq: a sequence of (genes) estimators
        :return: True or False
        """
        if self.config_types[seq[-1]] not in ["regressor", "classifier"]:
            return False
        n = len(seq) - 1
        if seq[n - 1] == "sklearn.pipeline.FeatureUnion":
            return False
        flag = False
        cnt = 0
        for idx in range(n):
            gene = seq[idx]
            if gene == "sklearn.pipeline.FeatureUnion":
                flag = True
            elif (
                    (gene in self.feature_selector)
                    or (self.config_types[gene] in ["regressor", "classifier"])
            ) and flag:
                cnt += 1
            elif flag and not (
                    (gene in self.feature_selector)
                    or (self.config_types[gene] in ["regressor", "classifier"])
            ):
                if cnt == 0:
                    return False
                else:
                    flag = False
                    cnt = 0
        return True

    def add_surrogate(self, estimator, itrs, sampling=None, optim="L-BFGS-B"):
        """
        Adding a regressor for surrogate optimization procedure.

        :param estimator: A scikit-learn style regressor
        :param itrs: Number of iterations the `estimator` needs to be fitted and optimized
        :param sampling: default= BoxSample; The sampling strategy (`CompactSample`, `BoxSample` or `SphereSample`)
        :param optim: default='L-BFGS-B';`scipy.optimize` solver
        :return: None
        """
        if self.surrogates is None:
            self.surrogates = []
        if sampling is None:
            from .structsearch import BoxSample

            sampling = BoxSample
        self.surrogates.append((estimator, itrs, sampling, optim))

    def _cast(self, n, X, y):
        """
        Evaluates and optimizes all legitimate combinations of length `n`

        :param n: The length of pipelines
        :param X: Training data
        :param y: Observed values
        :return: None
        """
        from .structsearch import BoxSample, CompactSample

        if self.couldBfirst == []:
            from sklearn.pipeline import Pipeline
        else:
            from imblearn.pipeline import Pipeline
        from sklearn.model_selection import RandomizedSearchCV

        if self.surrogates is None:
            from numpy import logspace
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.kernel_ridge import KernelRidge
            from sklearn.gaussian_process.kernels import (
                Matern,
                Sum,
                ExpSineSquared,
                WhiteKernel,
            )

            param_grid_gpr = {
                "alpha": logspace(-8, 1, 20),
                "kernel": [
                    Sum(Matern(length_scale=l, nu=p), WhiteKernel(noise_level=q))
                    for l in logspace(-3, 3, 20)
                    for p in [0.5, 1.5, 2.5]
                    for q in logspace(-3, 1.5, 20)
                ],
            }
            GPR = RandomizedSearchCV(
                GaussianProcessRegressor(),
                param_distributions=param_grid_gpr,
                n_iter=20,
                cv=2,
            )
            param_grid_krr = {
                "alpha": logspace(-4, 0, 10),
                "kernel": [
                    Sum(Matern(), ExpSineSquared(l, p))
                    for l in logspace(-2, 2, 20)
                    for p in logspace(0, 2, 20)
                ],
            }
            KRR = RandomizedSearchCV(
                KernelRidge(), param_distributions=param_grid_krr, n_iter=30, cv=2
            )
            self.surrogates = [
                (KRR, 35, CompactSample, "L-BFGS-B"),
                (GPR, 50, BoxSample, "L-BFGS-B"),
            ]
            self.min_random_evals = 10
        Pop = []
        candidates = self.words.Generate(n)
        for cnddt in candidates:
            if self._validate_sequence(cnddt):
                Pop.append(cnddt)
        for seq in Pop:
            if not self._validate_sequence(seq):
                continue
            best_mdl, best_scr = self.optimize_pipeline(seq, X, y)
            self.models[seq] = (best_mdl, best_scr)
            if self.verbose > 0:
                print("score:%f" % best_scr)
                print(best_mdl)

    def fit(self, X, y):
        """
        Generates and optimizes all legitimate pipelines. The best pipeline can be retrieved from `self.best_estimator_`

        :param X: Training data
        :param y: Corresponding observations
        :return: `self`
        """
        _X, _y = X, y
        if self.cat_cols is not None:
            from category_encoders.one_hot import OneHotEncoder

            enc = OneHotEncoder(
                cols=self.cat_cols, return_df=False, handle_unknown="ignore"
            )
            enc.fit(X)
            _X = enc.transform(X)

        X_, y_ = _X, _y
        self.num_features = len(X_[0])
        for l in range(1, self.length + 1):
            self._cast(l, X_, y_)
        self.best_estimator_ = list(self.get_top(1).items())[0][1][0]
        self.best_estimator_score = list(self.get_top(1).items())[0][1][1]
        return self

    @staticmethod
    def _get_class(library):
        """
        Takes a string that refers to a class in an accessible module and returns the associated class

        :param library: string referring to a class
        :return: the actual class
        """
        from importlib import import_module

        detail = library.split(".")
        module_str = ".".join(detail[:-1])
        module = import_module(module_str)
        clss = module.__getattribute__(detail[-1])
        return clss

    def eoa_fit(self, X, y, **kwargs):
        """
        Applies evolutionary optimization methods to find an optimum pipeline

        :param X: Training data
        :param y: Corresponding observations
        :param kwargs: `EOA` parameters
        :return: `self`
        """
        from .structsearch import BoxSample, CompactSample
        from .eoa import EOA

        _X, _y = X, y
        if self.cat_cols is not None:
            from category_encoders.one_hot import OneHotEncoder

            enc = OneHotEncoder(
                cols=self.cat_cols, return_df=False, handle_unknown="ignore"
            )
            enc.fit(X)
            _X = enc.transform(X)
        X_, y_ = _X, _y
        self.num_features = len(X_[0])
        Pop = []
        for l in range(1, self.length + 1):
            candidates = self.words.Generate(l)
            for cnddt in candidates:
                if self._validate_sequence(cnddt):
                    Pop.append(cnddt)

        def _eval(ppl):
            if self.couldBfirst == []:
                from sklearn.pipeline import Pipeline
            else:
                from imblearn.pipeline import Pipeline
            from sklearn.model_selection import RandomizedSearchCV

            if self.surrogates is None:
                from numpy import logspace
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.kernel_ridge import KernelRidge
                from sklearn.gaussian_process.kernels import (
                    Matern,
                    Sum,
                    ExpSineSquared,
                    WhiteKernel,
                )

                param_grid_gpr = {
                    "alpha": logspace(-8, 1, 20),
                    "kernel": [
                        Sum(Matern(length_scale=l_, nu=p), WhiteKernel(noise_level=q))
                        for l_ in logspace(-3, 3, 20)
                        for p in [0.5, 1.5, 2.5]
                        for q in logspace(-3, 1.5, 20)
                    ],
                }
                GPR = RandomizedSearchCV(
                    GaussianProcessRegressor(),
                    param_distributions=param_grid_gpr,
                    n_iter=20,
                    cv=2,
                )
                param_grid_krr = {
                    "alpha": logspace(-4, 0, 10),
                    "kernel": [
                        Sum(Matern(), ExpSineSquared(l_, p))
                        for l_ in logspace(-2, 2, 20)
                        for p in logspace(0, 2, 20)
                    ],
                }
                KRR = RandomizedSearchCV(
                    KernelRidge(), param_distributions=param_grid_krr, n_iter=30, cv=2
                )
                self.surrogates = [
                    (KRR, 35, CompactSample, "L-BFGS-B"),
                    (GPR, 50, BoxSample, "L-BFGS-B"),
                ]
                self.min_random_evals = 10
            from collections import OrderedDict

            fitted = OrderedDict([])
            for seq in ppl:
                best_mdl, best_scr = self.optimize_pipeline(seq, X_, y_)
                if seq not in self.models:
                    self.models[seq] = (best_mdl, best_scr)
                if self.verbose > 0:
                    print("score:%f" % best_scr)
                    print(best_mdl)
                fitted[seq] = -best_scr
            return fitted

        num_parents = kwargs.pop("num_parents", 30)
        mutation_prob = kwargs.pop("mutation_prob", 0.1)
        _eoa = EOA(
            population=Pop,
            fitness=_eval,
            num_parents=num_parents,
            mutation_prob=mutation_prob,
            term_genes=self.couldBlast,
            init_genes=self.couldBfirst,
            check_point=self.check_point,
            **kwargs
        )
        _eoa()
        self.best_estimator_ = list(self.get_top(1).items())[0][1][0]
        return self

    def get_top(self, num=5):
        """
        Finds the top `n` pipelines

        :param num: Number of pipelines to be returned
        :return: An OrderedDict of top models
        """
        from collections import OrderedDict

        return OrderedDict(sorted(self.models.items(), key=lambda x: x[1][1])[:num])

    def optimize_pipeline(self, seq, X, y):
        """
        Constructs and optimizes a pipeline according to the steps passed through `seq` which is a tuple of
        estimators and transformers.

        :param seq: the tuple of steps of the pipeline to be optimized
        :param X: numpy array of training features
        :param y: numpy array of training values
        :return: the optimized pipeline and its score
        """
        from .structsearch import SurrogateRandomCV

        if self.couldBfirst == []:
            from sklearn.pipeline import Pipeline
        else:
            from imblearn.pipeline import Pipeline
        OPTIM = None
        n = len(seq)
        idx = 0
        ent_idx = 0
        steps = []
        config = {}
        task_name = self.check_point + "_".join(seq)
        while ent_idx < n:
            est = seq[ent_idx]
            clss = self._get_class(est)
            pre = "stp_%d" % idx
            if (
                    self.config_types[est] in ["regressor", "classifier"]
                    and ent_idx < n - 1
            ):
                mdl = clss()
                steps.append(
                    (
                        pre,
                        StackingEstimator(
                            mdl,
                            res=self.stack_res,
                            probs=self.stack_probs,
                            decision=self.stack_decision,
                        ),
                    )
                )
                ent_idx += 1
            elif est == "sklearn.pipeline.FeatureUnion":
                self.config[est] = dict()
                int_idx = 1
                int_steps = []
                next_est = seq[ent_idx + int_idx]
                while (
                        (self.config_types[next_est] in ["regressor", "classifier"])
                        or (next_est in self.known_feature_selectors)
                ) and (ent_idx + int_idx < n - 1):
                    int_pre = "int_%d" % int_idx
                    if next_est in self.known_feature_selectors:
                        int_mdl = self._get_class(next_est)()
                        # set the parameter's dictionary
                        for kw in self.config[next_est]:
                            self.config[est][int_pre + "__" + kw] = self.config[
                                next_est
                            ][kw]
                    else:
                        from eli5.sklearn import PermutationImportance
                        from sklearn.feature_selection import SelectFromModel
                        from numpy import inf

                        int_est = self._get_class(next_est)()
                        int_mdl = SelectFromModel(
                            PermutationImportance(int_est, scoring=self.scoring, cv=3),
                            threshold=-inf,
                        )
                        self.config[est][int_pre + "__" + "max_features"] = Integer(
                            1, self.num_features
                        )
                        for kw in self.config[next_est]:
                            self.config[est][
                                int_pre + "__" + "estimator__estimator__" + kw
                                ] = self.config[next_est][kw]
                    int_steps.append((int_pre, int_mdl))
                    int_idx += 1
                    next_est = seq[ent_idx + int_idx]
                if int_steps != []:
                    mdl = clss(int_steps)
                    steps.append((pre, mdl))
                ent_idx += int_idx
            else:
                mdl = clss()
                steps.append((pre, mdl))
                ent_idx += 1
            for kw in self.config[est]:
                config[pre + "__" + kw] = self.config[est][kw]
            idx += 1
        ppln = Pipeline(steps)
        if self.verbose > 0:
            print("=" * 90)
            print(seq)
            print("-" * 90)
        for srgt in self.surrogates:
            OPTIM = SurrogateRandomCV(
                ppln,
                params=config,
                max_iter=srgt[1],
                min_evals=self.min_random_evals,
                scoring=self.scoring,
                cv=self.cv,
                verbose=max(self.verbose - 1, 0),
                sampling=srgt[2],
                regressor=srgt[0],
                scipy_solver=srgt[3],
                task_name=task_name,
                Continue=True,
                warm_start=True,
                n_jobs=self.n_jobs
            )
            OPTIM.fit(X, y)
        return OPTIM.best_estimator_, OPTIM.best_estimator_score
