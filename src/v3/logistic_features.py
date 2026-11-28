"""Memory-bounded fitting of the existing V2 Logistic Regression representation."""

import warnings

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.v2.challengers import DenseFeatures, _encode_frequency, _frequency_maps


COLUMN_BATCH = 16
ROW_BATCH = 4096


class ChunkedDenseFeatures(DenseFeatures):
    """Fit the V2 transform in small batches and retain its inference code."""

    def fit(self, matrix, labels):
        n_features = matrix.shape[1]
        self.frequencies = _frequency_maps(matrix, self.categorical)
        medians = np.empty(n_features, dtype=np.float64)
        ranks = np.empty(2 * n_features, dtype=np.float64)

        for lo in range(0, n_features, COLUMN_BATCH):
            hi = min(lo + COLUMN_BATCH, n_features)
            encoded = _encode_frequency(matrix[:, lo:hi].copy(order="C"), {
                index - lo: counts for index, counts in self.frequencies.items() if lo <= index < hi
            })
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            values = imputer.fit_transform(encoded)
            medians[lo:hi] = imputer.statistics_
            full = np.column_stack((values, np.isnan(encoded).astype(np.float32)))
            with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
                warnings.simplefilter("ignore", UserWarning)
                scores, _ = f_classif(full, labels)
            ranks[lo:hi], ranks[n_features + lo:n_features + hi] = np.split(scores, 2)

        # With no missingness indicators inside SimpleImputer, its only fitted values are these medians.
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(medians.reshape(1, -1))
        self.selected = np.argsort(-np.nan_to_num(ranks, nan=-np.inf), kind="stable")[:self.top_k]
        self.scaler = StandardScaler()
        for lo in range(0, len(matrix), ROW_BATCH):
            original = matrix[lo:min(lo + ROW_BATCH, len(matrix))]
            self.scaler.partial_fit(self._transform_raw(original)[:, self.selected])
        if self.sparse_categories:
            self.one_hot = OneHotEncoder(dtype=np.float32, min_frequency=7, max_categories=96,
                                         handle_unknown="infrequent_if_exist").fit(matrix[:, self.categorical])
        return self


def solver_matrix(transform, training):
    """Build one CSR input for L-BFGS without also holding all transformed batches."""
    rows = len(training)
    upper_bound = rows * (len(transform.selected) + len(transform.categorical))
    # L-BFGS expects float64. Keeping only that copy saves a second whole sparse matrix.
    storage_bytes = upper_bound * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize) + (rows + 1) * 4
    if upper_bound >= np.iinfo(np.int32).max or storage_bytes > 1_500_000_000:
        raise MemoryError("The single CSR solver matrix would exceed the experiment's allocation budget.")
    data = np.empty(upper_bound, dtype=np.float64)
    indices = np.empty(upper_bound, dtype=np.int32)
    indptr = np.empty(rows + 1, dtype=np.int32)
    indptr[0] = 0
    nz = 0
    for lo in range(0, rows, ROW_BATCH):
        hi = min(lo + ROW_BATCH, rows)
        batch = transform.transform(training[lo:hi])
        data[nz:nz + batch.nnz] = batch.data
        indices[nz:nz + batch.nnz] = batch.indices
        indptr[lo + 1:hi + 1] = nz + batch.indptr[1:]
        nz += batch.nnz
    return csr_matrix((data[:nz], indices[:nz], indptr), shape=(rows, batch.shape[1]), copy=False)
