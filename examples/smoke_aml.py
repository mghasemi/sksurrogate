"""Bounded AML end-to-end smoke for CI (runs in well under a minute).

Exercises the public AML API on the bundled airfoil dataset with the cheapest
estimator pair and a single EOA generation. The full user demo lives in
examples/example02.py; that search intentionally runs several generations over
KernelRidge kernels and takes tens of minutes on shared CI hardware.
"""

import tempfile

import pandas as pd

from sklearn.linear_model import Ridge

from SKSurrogate import AML, Categorical
from SKSurrogate.structsearch import BoxSample

df = pd.read_csv(
    "examples/data/airfoil_self_noise.dat",
    sep="\t",
    names=["Frequency", "Angle", "length", "velocity", "thickness", "level"],
)
# A small slice keeps the smoke to seconds even on 2-core CI runners; the full
# search cost belongs to the user-facing demo, not to CI.
X = df.drop("level", axis=1).values[:300]
y = df["level"].values[:300]

config = {
    "sklearn.linear_model.LinearRegression": {},
    "sklearn.preprocessing.StandardScaler": {
        "with_mean": Categorical([True, False]),
        "with_std": Categorical([True, False]),
    },
}
# Isolated checkpoint dir: EOA auto-resumes from <check_point>/EOA.eoa whenever
# it exists, and a checkpoint written by a different config would poison the run.
check_point = tempfile.mkdtemp(prefix="sksurrogate-smoke-") + "/"
# Cheap explicit surrogate: the default implicit pair (GPR/KRR over large kernel
# grids) spends tens of CPU-minutes per candidate; a cheap regressor with few
# iterations keeps this smoke within CI budgets while still exercising the
# surrogate-guided search path.
A = AML(
    config=config,
    length=2,
    verbose=0,
    scoring="neg_mean_squared_error",
    check_point=check_point,
    cv=2,
    n_jobs=1,
    surrogates=[(Ridge(), 5, BoxSample, "L-BFGS-B")],
    min_random_evals=3,
)
A.eoa_fit(X, y, max_generations=1, num_parents=2)
top = A.get_top(1)
assert top is not None
print("AML smoke OK:", top)
