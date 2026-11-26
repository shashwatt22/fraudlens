"""Development and leakage checks using entirely synthetic rows."""

import numpy as np
import pandas as pd
import pytest

from src.v2.advanced import entity_codes, prior_statistics
from src.v2.challengers import DenseFeatures, _encode_frequency, _frequency_maps
from src.v2.data import FirstSeen, walk_forward
from src.v2.experiment import MatrixView, review_budgets
from src.v2.explanation import tree_contributions
from src.v2.final import _calibrate, evaluate, policy_boundaries, prepare


def test_walk_forward_preserves_equal_timestamps_and_leaves_policy_rows():
    times = np.repeat(np.arange(20), 3)
    folds = walk_forward(times)
    assert all(times[f.train_end - 1] < times[f.train_end] for f in folds)
    assert folds[0].validation_end == folds[1].train_end
    assert folds[-1].validation_end < len(times)


def test_first_seen_codes_and_unseen_validation_handling():
    vocabulary = FirstSeen()
    train = vocabulary.encode(pd.Series(["b", "a", "b", None]))
    validation = vocabulary.encode(pd.Series(["c", "a"]))
    np.testing.assert_allclose(train, [0, 1, 0, np.nan])
    np.testing.assert_allclose(validation, [2, 1])
    matrix = np.column_stack((np.r_[train, validation], np.arange(6)))
    view = MatrixView(matrix, {"category": 0, "number": 1}, {"category"}).fit(4)
    np.testing.assert_allclose(view.read(4, 6), [[np.nan, 4], [1, 5]])


def test_strict_statistics_have_expected_counts_variance_and_inclusive_lower_bound():
    times = np.array([0, 120, 120, 3600, 3600, 3700, 86400])
    groups = np.array([0, 0, 0, 0, 1, -1, 0])
    amounts = np.array([4., 6., 8., 10., 100., 5., 20.])
    result = prior_statistics(times, groups, amounts)
    assert result.loc[1, "count"] == result.loc[2, "count"] == 1
    assert result.loc[3, "count_1h"] == 3
    assert result.loc[3, "mean"] == 6
    np.testing.assert_allclose(result.loc[3, "std"], np.sqrt(8/3))
    assert result.loc[6, "count_24h"] == 4
    assert result.loc[4, "count"] == 0
    assert result.loc[5].isna().all()
    pd.testing.assert_frame_equal(result.iloc[:3], prior_statistics(times[:3], groups[:3], amounts[:3]))


def test_entity_key_cold_start_and_frequency_are_training_fitted():
    np.testing.assert_array_equal(entity_codes([1, 1, 1, 2], [2, np.nan, 2, 2]), [0, -1, 0, 1])
    train = np.array([[0., 1.], [1., 2.], [0., 0.], [1., 3.]], dtype=np.float32)
    validation = np.array([[2., 4.], [0., 3.], [np.nan, 1.]], dtype=np.float32)
    frequency = _frequency_maps(train, [0])
    transformed = _encode_frequency(validation.copy(), frequency)
    np.testing.assert_allclose(transformed[:, 0], [0., np.log(3), np.nan])
    fitted = DenseFeatures(2, [0]).fit(train, np.array([0, 1, 0, 1]))
    assert np.isfinite(fitted.transform(validation)).all()


def test_review_budgets_keep_score_ties_in_source_order():
    labels = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    budgets = review_budgets(labels, np.ones(10))
    assert budgets["10"] == {"reviewed": 1, "precision": 1., "recall": 1.}


def test_policy_blocks_do_not_split_ties_and_keep_later_evaluation_disjoint():
    times = np.repeat(np.arange(10), 4)
    boundaries = policy_boundaries(times)
    assert boundaries == [12, 20]
    assert all(times[cut - 1] < times[cut] for cut in boundaries)
    labels = np.tile([0, 0, 1, 0], 10)
    scores = np.tile([.08, .23, .56, .05], 10)
    _, policy = _calibrate(scores, labels, times)
    assert policy["windows"] == [12, 8, 20]
    assert policy["policy_evaluation"]["confusion_matrix"] == [[15, 0], [0, 5]]


def test_retrospective_gate_refuses_to_open_before_frozen_choices(tmp_path):
    with pytest.raises(ValueError, match="Finish development first"):
        prepare(tmp_path, 1)
    with pytest.raises(ValueError, match="unopened, frozen"):
        evaluate(tmp_path, None, None, 1)


def test_native_categorical_tree_contributions_sum_to_raw_margin():
    import xgboost as xgb

    matrix = np.column_stack((np.arange(32) % 2, np.arange(32))).astype(np.float32)
    view = MatrixView(matrix, {"category": 0, "number": 1}, {"category"}).fit(len(matrix))
    training = xgb.DMatrix(matrix, label=(matrix[:, 0] > 0), feature_types=view.types,
                           feature_names=list(view.features), nthread=1, enable_categorical=True)
    model = xgb.train({"objective": "binary:logistic", "tree_method": "hist", "nthread": 1}, training, num_boost_round=5)
    contributions, margin, error, names, _ = tree_contributions("rich", model, {}, view, (2, 8), 1)
    np.testing.assert_allclose(contributions.sum(axis=1), margin, atol=1e-5)
    assert names == ["category", "number"] and error < 1e-5
