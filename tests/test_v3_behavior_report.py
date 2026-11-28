import numpy as np
from sklearn.metrics import average_precision_score

from src.v3.behavior_report import RankedAP, compare_bootstrap, entity_blocks


def test_weighted_ap_matches_sklearn_with_score_ties_and_zero_counts():
    labels = np.array([0, 1, 1, 0, 1, 0])
    scores = np.array([.9, .9, .8, .6, .6, .1])
    blocks = np.array([0, 1, 1, 2, 3, 0])
    weights = np.array([2, 0, 1, 3])
    expected = average_precision_score(labels, scores, sample_weight=weights[blocks])
    np.testing.assert_allclose(RankedAP(labels, scores, blocks)(weights), expected, atol=1e-15)


def test_bootstrap_reuses_entity_blocks_across_folds_and_keeps_missing_rows_separate():
    blocks, known, missing = entity_blocks([20, 10, -1, 20, -1])
    np.testing.assert_array_equal(blocks, [1, 0, 2, 1, 3])
    assert (known, missing) == (2, 2)

    labels = np.array([0, 1, 1, 1])
    scores = np.array([.1, .4, .6, .8])
    folds = [(labels, scores, scores, np.array(keys)) for keys in ([2, 3, 4, -1], [4, 3, 6, -1])]
    report = compare_bootstrap(folds, replicates=30)
    assert report["entity_blocks"] == 4 and report["missing_key_singleton_blocks"] == 2
    assert report["mean_ap_delta_ci95"] == {"lower": 0., "upper": 0.}
