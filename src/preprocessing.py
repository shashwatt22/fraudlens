"""Training-fitted preprocessing for the classical models and ablation."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .features import CATEGORICAL_BASIC, HISTORICAL_COLUMNS, NUMERIC_BASIC


def numeric_columns(include_history: bool) -> list[str]:
    return [*NUMERIC_BASIC, *(HISTORICAL_COLUMNS if include_history else [])]


class FraudPreprocessor(TransformerMixin, BaseEstimator):
    """Median-impute and scale numbers, then append missing flags and capped one-hot values."""

    def __init__(self, include_history: bool = True, max_categories: int = 32):
        self.include_history = include_history
        self.max_categories = max_categories

    def _categories(self, frame: pd.DataFrame) -> pd.DataFrame:
        mapped = {}
        for column, levels in self.category_levels_.items():
            values = frame[column].astype("string")
            kept = "value=" + values.where(values.isin(levels))
            mapped[column] = kept.fillna("other").mask(values.isna(), "missing")
        return pd.DataFrame(mapped, index=frame.index)

    def _numbers(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.numeric_columns_].to_numpy(dtype=np.float32, na_value=np.nan)

    def fit(self, frame: pd.DataFrame, y=None):
        if self.max_categories < 3:
            raise ValueError("max_categories needs room for at least one value, missing, and other.")

        self.numeric_columns_ = numeric_columns(self.include_history)
        self.feature_names_in_ = np.asarray([*self.numeric_columns_, *CATEGORICAL_BASIC], dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        numbers = self._numbers(frame)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler_ = StandardScaler().fit(self.imputer_.fit_transform(numbers))

        self.category_levels_ = {}
        for column in CATEGORICAL_BASIC:
            counts = frame[column].astype("string").value_counts().sort_index()
            most_common = counts.sort_values(ascending=False, kind="stable")
            self.category_levels_[column] = tuple(most_common.head(self.max_categories - 2).index)

        categories = [[*(f"value={value}" for value in levels), "missing", "other"]
                      for levels in self.category_levels_.values()]
        self.encoder_ = OneHotEncoder(categories=categories, dtype=np.float32, sparse_output=True)
        self.encoder_.fit(self._categories(frame))
        return self

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        check_is_fitted(self, "encoder_")
        numbers = self._numbers(frame)
        scaled = self.scaler_.transform(self.imputer_.transform(numbers))
        block = np.column_stack((scaled, np.isnan(numbers).astype(np.float32))).astype(np.float32)
        rows, columns = block.shape

        # XGBoost hist treats absent sparse entries as missing. Keep every numeric zero explicit.
        numeric = csr_matrix((block.ravel(), np.tile(np.arange(columns, dtype=np.int32), rows),
                              np.arange(0, (rows + 1) * columns, columns, dtype=np.int32)),
                             shape=block.shape)
        return hstack((numeric, self.encoder_.transform(self._categories(frame))), format="csr")

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "encoder_")
        return np.asarray([
            *self.numeric_columns_,
            *(f"{column}__missing" for column in self.numeric_columns_),
            *self.encoder_.get_feature_names_out(CATEGORICAL_BASIC),
        ], dtype=object)
