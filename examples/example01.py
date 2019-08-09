# Find an optimum classification pipeline

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import Matern, Sum, ExpSineSquared
from SKSurrogate import *

param_grid_krr = {
    "alpha": np.logspace(-4, 0, 5),
    "kernel": [
        Sum(Matern(), ExpSineSquared(l, p))
        for l in np.logspace(-2, 2, 10)
        for p in np.logspace(0, 2, 10)
    ],
}
regressor = RandomizedSearchCV(
    KernelRidge(), param_distributions=param_grid_krr, n_iter=5, cv=2
)

config = {
    # Classifiers
    "sklearn.naive_bayes.GaussianNB": {"var_smoothing": Real(1.0e-9, 2.0e-1)},
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
    "lightgbm.LGBMClassifier": {
        "boosting_type": Categorical(["gbdt", "dart", "goss", "rf"]),
        "num_leaves": Integer(2, 100),
        "learning_rate": Real(1.0e-7, 1.0 - 1.0e-6),  # prior='uniform'),
        "n_estimators": Integer(5, 250),
        "min_split_gain": Real(0.0, 1.0),  # prior='uniform'),
        "subsample": Real(1.0e-6, 1.0),  # prior='uniform'),
        "importance_type": Categorical(["split", "gain"]),
    },
    # Preprocesssors
    "sklearn.preprocessing.StandardScaler": {
        "with_mean": Categorical([True, False]),
        "with_std": Categorical([True, False]),
    },
    "skrebate.ReliefF": {
        "n_features_to_select": Integer(2, 10),
        "n_neighbors": Integer(2, 10),
    },
    # Sensitivity Analysis
    "SKSurrogate.sensapprx.SensAprx": {
        "n_features_to_select": Integer(2, 20),
        "method": Categorical(["sobol", "morris", "delta-mmnt"]),
        "regressor": Categorical([None, regressor]),
    },
}
import warnings

warnings.filterwarnings("ignore", category=Warning)


genetic_data = pd.read_csv(
    "https://github.com/EpistasisLab/scikit-rebate/raw/master/data/"
    "GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz",
    sep="\t",
    compression="gzip",
)
X, y = genetic_data.drop("class", axis=1).values, genetic_data["class"].values

A = AML(config=config, length=3, check_point="./", verbose=2)
A.eoa_fit(X, y, max_generation=10, num_parents=10)
print(A.get_top(5))
