import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import get_scorer, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from SKSurrogate import build_cv, cv_to_spec, mltrack


class TestTimeSeriesEvaluation(unittest.TestCase):
    def _series(self, start, stop):
        time = np.arange(start, stop, dtype=float)
        target = 0.1 * time + np.sin(time / 3.0)
        lag = np.concatenate(([target[0]], target[:-1]))
        return pd.DataFrame({"time": time, "lag_1": lag, "target": target})

    def test_timeseries_splitter_spec_preserves_order_gap_and_parameters(self):
        splitter = TimeSeriesSplit(n_splits=4, test_size=10, gap=2)
        restored = build_cv(cv_to_spec(splitter))
        X = np.arange(100).reshape(-1, 1)

        self.assertEqual(cv_to_spec(restored), cv_to_spec(splitter))
        for train_indices, test_indices in restored.split(X):
            self.assertLess(train_indices[-1] + 2, test_indices[0])
            self.assertLess(train_indices[-1], test_indices[0])
            self.assertTrue(np.all(np.diff(train_indices) == 1))
            self.assertTrue(np.all(np.diff(test_indices) == 1))

    def test_nested_cv_honors_sklearn_scorer_for_time_ordered_regression(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            tracker = mltrack("time-series-evaluation", db_name=database.name)
            train = self._series(0, 100)
            validation = self._series(100, 115)
            tracker.RegisterData(train, "target", partition="train")
            tracker.RegisterData(validation, "target", partition="validation")
            splitter = TimeSeriesSplit(n_splits=4, test_size=10, gap=2)
            scorer = get_scorer("neg_mean_absolute_error")

            try:
                results = tracker.nested_cv_evaluation(
                    LinearRegression(),
                    inner_cv=TimeSeriesSplit(n_splits=3, test_size=5, gap=2),
                    outer_cv=splitter,
                    train_partition="train",
                    validation_partition="validation",
                    scoring=scorer,
                )
                final_model = LinearRegression().fit(
                    train[["time", "lag_1"]].to_numpy(),
                    train["target"].to_numpy(),
                )
                expected_validation_score = scorer(
                    final_model,
                    validation[["time", "lag_1"]].to_numpy(),
                    validation["target"].to_numpy(),
                )
            finally:
                tracker.close()

        self.assertEqual(len(results["fold_metrics"]), 4)
        for fold in results["fold_metrics"]:
            expected_score = -mean_absolute_error(
                fold["outer_truth"], fold["outer_predictions"]
            )
            self.assertAlmostEqual(fold["outer_score"], expected_score)
            self.assertLessEqual(fold["outer_score"], 0.0)
            self.assertTrue(all(score <= 0.0 for score in fold["inner_scores"]))
        self.assertAlmostEqual(
            results["final_validation_score"], expected_validation_score
        )
        self.assertEqual(
            results["fold_metrics"][0]["outer_truth"],
            train["target"].iloc[60:70].tolist(),
        )
        self.assertEqual(
            results["fold_metrics"][-1]["outer_truth"],
            train["target"].iloc[90:100].tolist(),
        )


if __name__ == "__main__":
    unittest.main()
