"""DBSCAN preparation and unsupervised selection on synthetic rows."""

import numpy as np
import pandas as pd
import pytest

from src.clustering import (FEATURES, ClusteringPreprocessor, choose_configuration, cluster_metrics,
                            run_clustering, sample_positions)


def test_training_preprocessor_ignores_metadata_and_handles_missing_history():
    train = pd.DataFrame({column: [1., 3., 10.] for column in FEATURES})
    train.loc[0, "amount_to_prior_mean"] = np.nan
    train = train.assign(TransactionID=[1, 2, 3], isFraud=[0, 0, 1], entity_key="synthetic", classifier_score=.9)
    original = train.copy()
    preprocessor = ClusteringPreprocessor().fit(train)
    feature_only = train[list(FEATURES)]
    np.testing.assert_array_equal(preprocessor.transform(train), preprocessor.transform(feature_only))
    expected_median = np.median(np.log1p([3., 10.]))
    assert preprocessor.imputer_.statistics_[2] == pytest.approx(expected_median)
    transformed = preprocessor.transform(train)
    assert np.isfinite(transformed).all() and transformed.shape == (3, 6)
    assert transformed[0, -1] > transformed[1, -1]
    np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=1e-12)
    higher = feature_only * 1_000
    preprocessor.transform(higher)
    assert preprocessor.imputer_.statistics_[2] == pytest.approx(expected_median)
    pd.testing.assert_frame_equal(original, train)


def test_sampling_is_reproducible_bounded_and_chronological():
    a = sample_positions(100, 10)
    np.testing.assert_array_equal(a, sample_positions(100, 10))
    assert len(a) == 10 and np.all(np.diff(a) > 0)
    assert not np.array_equal(a, sample_positions(100, 10, seed=1))
    np.testing.assert_array_equal(sample_positions(3, 10), [0, 1, 2])


def test_cluster_summary_noise_single_cluster_and_all_noise():
    matrix = np.array([[0.], [.1], [5.], [5.1], [100.]])
    normal = cluster_metrics(matrix, np.array([0, 0, 1, 1, -1]))
    assert normal["noise"] == 1 and normal["noise_fraction"] == .2
    assert normal["clustered"] == 4 and normal["silhouette"] > .90
    for labels, count in (([0, 0, 0, 0, -1], 1), ([-1] * 5, 0)):
        report = cluster_metrics(matrix, np.asarray(labels))
        assert report["clusters"] == count and report["silhouette"] is None


def test_configuration_excludes_excess_noise_and_handles_undefined_silhouette():
    reports = [{"status": "completed", "silhouette": .9, "noise_fraction": .7, "largest_cluster_share": .5},
               {"status": "completed", "silhouette": .4, "noise_fraction": .1, "largest_cluster_share": .7},
               {"status": "completed", "silhouette": None, "noise_fraction": .05, "largest_cluster_share": 1},
               {"status": "skipped_dense"}]
    assert choose_configuration(reports) == 1
    assert choose_configuration(reports[2:]) == 0
    with pytest.raises(ValueError, match="neighborhood budget"):
        choose_configuration(reports[-1:])


def test_cluster_fit_and_pca_alignment_are_invariant_to_label_changes():
    array = np.repeat([1., 20.], 24)
    frame = pd.DataFrame({column: array for column in FEATURES}).assign(
        TransactionAmt=array, TransactionID=np.arange(48), isFraud=np.arange(48) % 2,
    )
    run = run_clustering(frame, 48, 1)
    changed = run_clustering(frame.assign(isFraud=0), 48, 1)
    assert run.selected["clusters"] == 2
    pd.testing.assert_frame_equal(run.assignments, changed.assignments)
    sample = frame.set_index("TransactionID").loc[run.assignments["TransactionID"]]
    coords = run.pca.transform(run.preprocessor.transform(sample))
    np.testing.assert_allclose(run.assignments[["pca_1", "pca_2"]], coords, atol=1e-10)
    assert changed.profiles["fraud_rows"].sum() == 0 and run.profiles["fraud_rows"].sum() == 24
