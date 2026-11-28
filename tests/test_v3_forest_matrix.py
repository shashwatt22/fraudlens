import numpy as np

from src.v2.challengers import DenseFeatures
from src.v3 import forest_experiment
from src.v3.forest_experiment import training_matrix
from src.v3.logistic_features import ChunkedDenseFeatures


def test_chunked_dense_forest_matrix_matches_the_v2_representation(monkeypatch):
    monkeypatch.setattr(forest_experiment, "ROW_BATCH", 29)
    rng = np.random.default_rng(42)
    values = rng.normal(size=(211, 26)).astype(np.float32)
    values[:, 6] = rng.integers(0, 10, len(values))
    values[::3, 6] = np.nan
    values[::5, 23] = np.nan
    values[:, 25] = np.nan
    labels = (values[:, 1] < .1).astype(np.uint8)
    original = DenseFeatures(20, [6], False).fit(values, labels)
    adapted = ChunkedDenseFeatures(20, [6], False).fit(values, labels)
    np.testing.assert_array_equal(adapted.selected, original.selected)
    result = training_matrix(adapted, values)
    assert result.dtype == np.float32 and result.flags.c_contiguous
    np.testing.assert_allclose(result, original.transform(values), atol=1e-6, rtol=0)
