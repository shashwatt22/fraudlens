import numpy as np
from scipy.sparse import issparse

from src.v2.challengers import DenseFeatures
from src.v3 import logistic_features
from src.v3.logistic_features import ChunkedDenseFeatures, solver_matrix


def test_chunked_preprocessing_preserves_v2_representation_and_handles_missing_and_unseen(monkeypatch):
    monkeypatch.setattr(logistic_features, "ROW_BATCH", 31)
    rng = np.random.default_rng(42)
    training = rng.normal(size=(253, 25)).astype(np.float32)
    training[::3, 1] = np.nan
    training[:, 17] = rng.integers(0, 5, len(training))
    training[::7, 17] = np.nan
    training[:, 19] = np.nan
    labels = (training[:, 0] + training[:, 4] > 0).astype(np.uint8)
    v2 = DenseFeatures(18, [17], True).fit(training, labels)
    chunked = ChunkedDenseFeatures(18, [17], True).fit(training, labels)

    validation = np.full((4, 25), np.nan, dtype=np.float32)
    validation[:, 17] = [999, 0, 4, np.nan]
    validation[1, 2] = 2000
    np.testing.assert_array_equal(chunked.selected, v2.selected)
    np.testing.assert_array_equal(chunked.imputer.statistics_, v2.imputer.statistics_)
    np.testing.assert_allclose(chunked.transform(validation).toarray(), v2.transform(validation).toarray(), atol=1e-6)
    assert max(chunked.one_hot.categories_[0][:-1]) == 4
    assert len(chunked.frequencies[17]) == 5


def test_solver_matrix_matches_chunked_v2_values_and_stays_csr(monkeypatch):
    monkeypatch.setattr(logistic_features, "ROW_BATCH", 11)
    rng = np.random.default_rng(7)
    training = rng.integers(4, size=(67, 4)).astype(np.float32)
    training[::7, 2] = np.nan
    labels = rng.integers(2, size=len(training))
    transform = ChunkedDenseFeatures(6, [0, 2], True).fit(training, labels)
    expected = transform.transform(training)
    actual = solver_matrix(transform, training)
    assert issparse(actual) and actual.format == "csr"
    assert actual.dtype == np.float64 and actual.shape == expected.shape
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())
