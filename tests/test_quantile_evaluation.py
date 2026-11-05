import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import make_scorer, mean_pinball_loss

from SKSurrogate import mltrack


class TestQuantileEvaluation(unittest.TestCase):
    def test_nested_cv_honors_quantile_pinball_scorer(self):
        x = np.arange(90, dtype=float)
        target = 1.5 * x + np.sin(x / 4.0) + np.where(x % 3 == 0, 3.0, -1.0)
        train = pd.DataFrame({"feature": x[:72], "target": target[:72]})
        validation = pd.DataFrame({"feature": x[72:], "target": target[72:]})
        estimator = QuantileRegressor(quantile=0.8, alpha=0.0)
        scorer = make_scorer(
            mean_pinball_loss,
            alpha=0.8,
            greater_is_better=False,
        )

        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("quantile-evaluation", db_name=database.name)
            tracker.RegisterData(train, "target", partition="train")
            tracker.RegisterData(validation, "target", partition="validation")
            try:
                results = tracker.nested_cv_evaluation(
                    estimator,
                    inner_cv=2,
                    outer_cv=3,
                    train_partition="train",
                    validation_partition="validation",
                    scoring=scorer,
                )
                final_model = QuantileRegressor(quantile=0.8, alpha=0.0).fit(
                    train[["feature"]].to_numpy(),
                    train["target"].to_numpy(),
                )
                expected_validation_score = scorer(
                    final_model,
                    validation[["feature"]].to_numpy(),
                    validation["target"].to_numpy(),
                )
            finally:
                tracker.close()

        for fold in results["fold_metrics"]:
            expected_score = -mean_pinball_loss(
                fold["outer_truth"],
                fold["outer_predictions"],
                alpha=0.8,
            )
            self.assertAlmostEqual(fold["outer_score"], expected_score)
        self.assertAlmostEqual(
            results["final_validation_score"],
            expected_validation_score,
        )


if __name__ == "__main__":
    unittest.main()
