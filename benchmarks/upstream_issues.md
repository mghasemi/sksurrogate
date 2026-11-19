# Upstream defect reports — `SKSurrogate` v0.2.1

Three findings from building the pipeline-space benchmark (`DESIGN-cspace.md`). Each is
reproducible from the public API, each is independent of the benchmark's results, and each
changes what a user's search actually does. Drafted here so they can be filed verbatim; the
benchmark itself works around all three **without patching the library** (`DESIGN.md` v1.4
principle) and measures their cost.

Line references are to `SKSurrogate/structsearch.py` at `0.2.1`.

---

## Issue 1 — `MinEvals` scales as `C(p+3, p)`, so the surrogate silently never engages in moderate dimensions

**Severity: high (silent, no warning, no error).**

`SurrogateSearch.__init__` raises the engagement threshold whenever it installs its own default
regressor:

```python
# structsearch.py:278-287
self.regressor = kwargs.pop("regressor", None)
...
if self.regressor is None:
    self.regressor = GaussianProcessRegressor()
    new_n_evals = binom(n_ + 3, n_)
    if self.MinEvals < new_n_evals:
        self.MinEvals = new_n_evals
```

With $p$ search parameters the floor is $\binom{p+3}{p}$:

| $p$ | floor |
|---:|---:|
| 3 | 20 |
| 5 | 56 |
| 8 | 165 |
| **11** | **364** |
| 15 | 816 |

At $p = 11$ (a pipeline space: 3 structural choices + 8 estimator knobs) no realistic budget
reaches the floor, so `SurrogateSearch` degenerates to pure random sampling **while still being
reported as surrogate-guided**. Nothing in `fit` warns; `evaluation_history_` looks normal.
A user comparing "SKSurrogate" against Optuna would be comparing random search against TPE.

**Reproducer** (no datasets needed; verified 2026-10-26 against 0.2.1, output below):

```python
import numpy as np
from math import comb
from sklearn.ensemble import RandomForestClassifier
from SKSurrogate import Categorical, Integer, SurrogateRandomCV

# 11 searchable coordinates on one estimator (RandomForest exposes plenty)
params = {
    "n_estimators": Integer(10, 200), "max_depth": Integer(2, 20),
    "min_samples_split": Integer(2, 20), "min_samples_leaf": Integer(1, 10),
    "max_features": Categorical(["sqrt", "log2", None]),
    "max_leaf_nodes": Integer(10, 100),
    "min_impurity_decrease": Categorical([0.0, 1e-4, 1e-3]),
    "criterion": Categorical(["gini", "entropy"]),
    "bootstrap": Categorical([True, False]),
    "ccp_alpha": Categorical([0.0, 1e-4]),
    "max_samples": Categorical([0.5, 0.8, None]),
}
print(len(params), comb(len(params) + 3, len(params)))        # 11 364

X = np.random.RandomState(0).rand(200, 4)
y = np.random.RandomState(1).randint(0, 2, 200)
search = SurrogateRandomCV(RandomForestClassifier(n_estimators=10), params,
                           max_evals=40, max_iter=160, cv=3, n_jobs=1,
                           random_state=0, refit=False, task_name="issue1_probe")
search.fit(X, y)
print(search.OPTIM.MinEvals, len(search.OPTIM.evaluated))     # 364.0 41
```

Observed:

```
11 364
364.0 41
```

`MinEvals = 364` while the run evaluates `41` points: the fit branch of `__iterate` is
unreachable for the entire run, with no warning and no error. Setting `max_evals=400` would
reach it; any AutoML-style budget will not.

**Suggested fix (either):**
1. Do not clamp a user-supplied `min_evals`; treat the combinatorial value as a *default* only
   when `min_evals` was not passed. The constructor cannot currently distinguish "user passed
   the default" from "user passed 25" because `min_evals=25` is the signature default.
2. Or record and expose engagement: `search.n_surrogate_fits_` / `search.engaged_`, and emit a
   `UserWarning` when `max_evals < MinEvals` ("the surrogate will never engage").

The second is the minimum required to make the failure visible; the first is the fix users want.

---

## Issue 2 — `'passthrough'` / `None` pipeline steps are rejected as preprocessing failures

**Severity: high for pipeline search (the space cannot express "no transformer").**

sklearn's `Pipeline` treats `'passthrough'` and `None` as "identity": the step is skipped.
`SurrogateRandomCV._zero_feature_reason` probes each intermediate step directly:

```python
# structsearch.py:1002-1026
for _, transformer in pipeline.steps[:-1]:
    if hasattr(transformer, "fit_transform"):
        features = transformer.fit_transform(features, y)
    else:
        transformer.fit(features, y)
        features = transformer.transform(features)
```

A string sentinel has neither method, so the `except` branch returns
`"pipeline preprocessing failed before evaluation: AttributeError: 'str' object has no attribute 'fit'"`
and the candidate is recorded `status="invalid"` with a score of `inf`. `_validate_pipeline_structure`
additionally rejects a `None` step at construction time.

Net effect: searching over *whether* a transformer is present — the most common pipeline-search
axis — is impossible. Workaround is `FunctionTransformer()` per absent step, which changes the
pipeline object and requires the user to know this internal probe exists.

**Reproducer** (verified 2026-10-26, output below):

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from SKSurrogate import Categorical, SurrogateRandomCV

pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
pipe.set_params(scale="passthrough")          # legal sklearn: step is skipped

search = SurrogateRandomCV(pipe, {"clf__C": Categorical([0.1, 1.0])}, cv=3, n_jobs=1, refit=False)
print(repr(search._zero_feature_reason({}, X, y)))
```

Observed:

```
"pipeline preprocessing failed before evaluation: AttributeError: 'str' object has no attribute 'fit'"
```

Any candidate that sets a step to `'passthrough'` (or `None`) is therefore scored `inf` and
recorded `status="invalid"`.

**Suggested fix:** skip sentinels explicitly before probing:

```python
if transformer is None or transformer == "passthrough":
    continue
```

and accept them in `_validate_pipeline_structure`. Also treat `"drop"` as unsupported-with-reason
rather than as a preprocessing failure.

---

## Issue 3 — `Categorical` parameters are encoded ordinally, so no metric exists on the axis

**Severity: high for heterogeneous spaces; by design (documented) for homogeneous ones.**

```python
# structsearch.py:1108-1111 — bounds per categorical
self.bounds.append(param_num_range.bound_tuple)      # (0, len(items) - 1)
# structsearch.py:1153-1157 — decode
category_index = min(max(int(round(x[_idx])), 0), len(_param_num_range.items) - 1)
cand_params[_param] = _param_num_range.items[category_index]
```

Each categorical becomes an integer coordinate on the real line. For `criterion in {gini, entropy}`
that is harmless. For `clf in {logreg, rf, et, hgb, svc, knn}` the ordering is arbitrary:

* the trust-region step treats `rf -> et` as a small perturbation and `logreg -> knn` as a large
  one, neither of which is meaningful;
* a GP fitted on the ordinal axis extrapolates across families (measured: predicting at
  $x = 3.5$ with 4 categories returns a value **above** every observed response, instead of
  interpolating between categories 3 and 0);
* the local optimiser proposes fractional coordinates that the decode step rounds, so the
  surface it optimises differs from the surface it is scored on.

**Suggested fix:** encode categoricals with a geometric embedding before fitting the surface
(one-hot, or a circular/simplex embedding for differentiability), and decode by nearest level.
A per-parameter `encoding=` option (`"ordinal" | "onehot" | "circular"`) would keep the current
behaviour available for ordered grids. Related: a `mixed_encoding` regressor contract so users
can supply their own without forking the local optimiser.

**Benchmark-side workaround used today:** `benchmarks/pipelines.py::MixedCategoricalSurface`,
injected through the public `regressor=` kwarg, maps each categorical coordinate to
$(\cos 2\pi j/k, \sin 2\pi j/k)$ — equidistant categories, continuous in the raw coordinate, and
differentiable for SLSQP/COBYLA.

---

## Cross-cutting note — no engagement telemetry

Neither `SurrogateSearch` nor `SurrogateRandomCV` exposes *how many times the surrogate was
fitted* or *at which evaluation it first fired*. Both are needed to audit a run: without them a
result that says "surrogate search found X" cannot be distinguished from random search.
Suggest `search.n_surrogate_fits_` and `search.engaged_at_` (evaluation index or `None`), which
is exactly what the benchmark currently reconstructs with a wrapper regressor
(`benchmarks/cspace_methods.py::_EngagementProbe`).
