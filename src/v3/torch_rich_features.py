"""Keep all approved V2 values and gradient-varying missing flags for the MLP."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data_loader import ID, TARGET, TIME
from src.features import ENTITY
from src.v2.challengers import DenseFeatures, _encode_frequency, _frequency_maps
from .behavior import FEATURES as REJECTED
from .logistic_features import COLUMN_BATCH, ROW_BATCH


def validate_features(names):
    forbidden = {name.casefold() for name in (ID, TARGET, TIME, ENTITY, *REJECTED)}
    if len(set(names)) != len(names) or any(name.casefold() in forbidden for name in names):
        raise ValueError("The input manifest contains duplicated or prohibited features.")


class RichNeuralFeatures(DenseFeatures):
    """Reuse V2 frequency encoding, median fill and scaling without a supervised cap."""

    def __init__(self, categorical):
        super().__init__(None, categorical, False)

    def fit(self, matrix):
        width = matrix.shape[1]
        self.frequencies = _frequency_maps(matrix, self.categorical)
        medians = np.empty(width, dtype=np.float64)
        varying = np.zeros(width, dtype=bool)
        for lo in range(0, width, COLUMN_BATCH):
            hi = min(lo + COLUMN_BATCH, width)
            encoded = _encode_frequency(matrix[:, lo:hi].copy(order="C"), {
                index - lo: counts for index, counts in self.frequencies.items() if lo <= index < hi
            })
            missing = np.count_nonzero(np.isnan(encoded), axis=0)
            varying[lo:hi] = (missing > 0) & (missing < len(matrix))
            medians[lo:hi] = SimpleImputer(strategy="median", keep_empty_features=True).fit(encoded).statistics_

        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(medians.reshape(1, -1))
        self.selected = np.r_[np.arange(width), width + np.flatnonzero(varying)]
        self.scaler = StandardScaler()
        for lo in range(0, len(matrix), ROW_BATCH):
            self.scaler.partial_fit(self._transform_raw(matrix[lo:lo + ROW_BATCH])[:, self.selected])
        return self


def transform_block(preprocessor, source):
    """Materialize a single fold block in bounded batches, with one float32 copy."""
    width = len(preprocessor.selected)
    size = len(source) * width * np.dtype(np.float32).itemsize
    if size > 1_500_000_000:
        raise MemoryError("The transformed block exceeds its 1.5 GB allocation budget.")
    result = np.empty((len(source), width), dtype=np.float32)
    for lo in range(0, len(source), ROW_BATCH):
        hi = min(lo + ROW_BATCH, len(source))
        result[lo:hi] = preprocessor.transform(source[lo:hi])
    return result


def input_inventory(names, categories, control, rich):
    width = len(names)
    old, new = set(control.selected.tolist()), set(rich.selected.tolist())
    return {
        "control_values": [name for i, name in enumerate(names) if i in old],
        "control_missing_indicators": [name for i, name in enumerate(names) if width + i in old],
        "control_omitted_values": [name for i, name in enumerate(names) if i not in old],
        "control_omitted_missing_indicators": [name for i, name in enumerate(names) if width + i not in old],
        "challenger_numeric_values": [name for i, name in enumerate(names) if i not in categories],
        "challenger_frequency_values": [name for i, name in enumerate(names) if i in categories],
        "challenger_missing_indicators": [name for i, name in enumerate(names) if width + i in new],
        "constant_missing_indicators_excluded": [name for i, name in enumerate(names) if width + i not in new],
    }
