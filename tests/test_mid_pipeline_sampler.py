"""Mid-pipeline samplers are valid in imblearn pipelines (regression).

The failure mode: a search-space including ``imblearn.over_sampling.SMOTE``
made the EOA build a legitimate ``[SMOTE, estimator]`` pipeline. The structure
validator required *every* intermediate step to implement ``transform`` (a
sklearn rule), but a sampler implements ``fit_resample`` instead — so the whole
experiment job aborted with
``Pipeline intermediate step 'stp_0' must implement fit and transform``.

These tests pin that samplers are accepted as intermediate steps: they are valid
under ``imblearn.pipeline.Pipeline`` (which routes samplers through
``fit_resample``), so pipelines containing them must validate and fit.
"""

import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from SKSurrogate.structsearch import Real, SurrogateRandomCV


def _frame():
    rng = np.random.RandomState(0)
    X = rng.rand(90, 6)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.75).astype(int)
    return X, y


def test_mid_pipeline_sampler_is_accepted_and_fits():
    X, y = _frame()
    estimator = Pipeline(
        [
            ("stp_0", SMOTE()),
            ("stp_1", LogisticRegression(max_iter=400)),
        ]
    )
    opt = SurrogateRandomCV(
        estimator,
        param_distributions={"stp_1__C": Real(1e-3, 3.0)},
        n_iter=3,
        cv=2,
        n_jobs=1,
        random_state=0,
        scoring="accuracy",
    )
    # Regression: this used to raise
    # "Pipeline intermediate step 'stp_0' must implement fit and transform".
    opt.fit(X, y)
    # The surrogate can leave the score non-finite; what matters is the
    # pipeline was accepted and a best estimator was produced.
    assert opt.best_estimator_ is not None


def test_sampler_pipeline_validate_sequence_keeps_valid_seq():
    """The EOA-level filter keeps a valid sampler-first pipeline (not dropped)."""
    from SKSurrogate.aml import AML

    config = {
        "sklearn.linear_model.LogisticRegression": {"C": Real(1e-3, 3.0)},
        "imblearn.over_sampling.SMOTE": {},
    }
    aml = AML(config=config, length=2, n_jobs=1)
    aml.types()
    seq = ["imblearn.over_sampling.SMOTE", "sklearn.linear_model.LogisticRegression"]
    # The EOA filter must keep it (it is a valid pipeline), not drop it early.
    assert aml._validate_sequence(seq) is True
