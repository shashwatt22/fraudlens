"""Synthetic checks for train-only anomaly features and empirical percentiles."""

import numpy as np
import pandas as pd
import pytest

from src.behavior import (ANOMALY_FEATURES, MISSING_FLAGS, AnomalyPreprocessor, anomaly_percentiles,
                          anomaly_scores, fit_isolation_forest)


def test_anomaly_imputation_uses_training_medians_and_excludes_metadata():
    train = pd.DataFrame({column: [1, 3, 5] for column in ANOMALY_FEATURES})
    train["prior_mean_amount"] = [1, 3, np.nan]
    train["is_new_product"] = np.nan
    train = train.assign(TransactionID=[10, 11, 12], isFraud=[0, 1, 1], entity_key="proxy", TransactionDT=42)
    preprocessor = AnomalyPreprocessor().fit(train)
    expected = [*ANOMALY_FEATURES, *(f"{column}__missing" for column in MISSING_FLAGS)]
    assert preprocessor.get_feature_names_out() == expected
    assert set(expected).isdisjoint({"TransactionID", "isFraud", "entity_key", "TransactionDT"})

    held_out = pd.DataFrame({column: [np.nan, 1000] for column in ANOMALY_FEATURES})
    matrix = preprocessor.transform(held_out)
    assert matrix.dtype == np.float32 and np.isfinite(matrix).all()
    assert matrix[0, expected.index("log_amount")] == 3
    assert matrix[0, expected.index("prior_mean_amount")] == 2
    assert matrix[0, expected.index("is_new_product")] == 0
    assert matrix[0, expected.index("prior_mean_amount__missing")] == 1
    assert matrix[1, expected.index("prior_mean_amount__missing")] == 0
    assert preprocessor.imputer_.statistics_[expected.index("prior_mean_amount")] == 2


def test_isolation_scores_direction_finiteness_and_batch_independence():
    rng = np.random.default_rng(42)
    training = pd.DataFrame(rng.normal(size=(500, len(ANOMALY_FEATURES))), columns=ANOMALY_FEATURES)
    training["isFraud"] = 1  # The fitted model does not require a label or both classes.
    prep, model, fit_rows = fit_isolation_forest(training, threads=1)
    assert fit_rows == 500
    probe = pd.DataFrame([[0] * len(ANOMALY_FEATURES), [50] * len(ANOMALY_FEATURES)], columns=ANOMALY_FEATURES)
    scores = anomaly_scores(model, prep, probe)
    assert np.isfinite(scores).all() and scores[1] > scores[0]
    np.testing.assert_array_equal(scores, -model.score_samples(prep.transform(probe)))
    reference = np.sort(anomaly_scores(model, prep, training))
    joint = anomaly_percentiles(reference, scores)
    alone = anomaly_percentiles(reference, anomaly_scores(model, prep, probe.iloc[[1]]))
    assert joint[1] == alone[0] and joint[1] >= joint[0]


def test_empirical_midrank_keeps_ties_and_bounds():
    reference = np.array([0.1, 0.2, 0.2, 0.8])
    scores = np.array([0.0, 0.1, 0.15, 0.2, 0.5, 0.8, 1.0])
    np.testing.assert_array_equal(anomaly_percentiles(reference, scores), [0, 0.125, 0.25, 0.5, 0.75, 0.875, 1])
    assert anomaly_percentiles(np.ones(6), np.array([1]))[0] == 0.5
    with pytest.raises(ValueError, match="nonempty, sorted, finite"):
        anomaly_percentiles(reference[::-1], scores)
    with pytest.raises(ValueError, match="finite scores"):
        anomaly_percentiles(reference, [np.nan])
