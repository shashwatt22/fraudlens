"""Training-only preprocessing checks using synthetic rows."""

import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix_csr

from src.features import CATEGORICAL_BASIC, HISTORICAL_COLUMNS, NUMERIC_BASIC
from src.preprocessing import FraudPreprocessor


def synthetic_frame() -> pd.DataFrame:
    frame = pd.DataFrame({column: ["W", "W", "X", None] for column in CATEGORICAL_BASIC})
    for column in [*NUMERIC_BASIC, *HISTORICAL_COLUMNS]:
        frame[column] = [1.0, 3.0, 5.0, np.nan]
    frame["card1"] = pd.Series([100, 100, 999, None], dtype="Int32")
    frame["prior_mean_amount"] = np.nan
    return frame.assign(isFraud=[0, 1, 0, 1], TransactionID=[11, 22, 33, 44],
                        TransactionDT=[1, 2, 3, 4], entity_key="synthetic")


def test_feature_selection_excludes_target_metadata_and_basic_has_no_history_flags():
    frame = synthetic_frame()
    basic = FraudPreprocessor(include_history=False).fit(frame)
    behavior = FraudPreprocessor().fit(frame)
    assert list(basic.feature_names_in_) == [*NUMERIC_BASIC, *CATEGORICAL_BASIC]
    assert set(behavior.feature_names_in_) - set(basic.feature_names_in_) == set(HISTORICAL_COLUMNS)
    assert not {"TransactionID", "TransactionDT", "entity_key", "isFraud"}.intersection(behavior.feature_names_in_)
    extra = set(behavior.get_feature_names_out()) - set(basic.get_feature_names_out())
    assert extra == {*HISTORICAL_COLUMNS, *(f"{name}__missing" for name in HISTORICAL_COLUMNS)}
    np.testing.assert_array_equal(basic.transform(frame.drop(columns=HISTORICAL_COLUMNS)).toarray(),
                                  basic.transform(frame).toarray())


def test_train_fitted_median_scaling_and_unknown_missing_categories():
    train = synthetic_frame()
    preprocessor = FraudPreprocessor(max_categories=4).fit(train)
    names = preprocessor.get_feature_names_out().tolist()
    holdout = train.iloc[:2].copy()
    holdout["ProductCD"] = ["never-observed", None]
    holdout["card1"] = pd.Series([8888, None], dtype="Int32")
    holdout["TransactionAmt"] = [200.0, np.nan]
    holdout["prior_mean_amount"] = [4.0, np.nan]
    encoded = preprocessor.transform(holdout)

    assert isspmatrix_csr(encoded) and encoded.dtype == np.float32
    assert encoded.shape[1] == preprocessor.transform(train).shape[1] == len(names)
    assert np.isfinite(encoded.data).all()
    assert preprocessor.category_levels_["ProductCD"] == ("W", "X")
    assert preprocessor.imputer_.statistics_[names.index("TransactionAmt")] == 3
    assert preprocessor.scaler_.mean_[names.index("TransactionAmt")] == 3
    assert preprocessor.imputer_.statistics_[names.index("prior_mean_amount")] == 0
    assert encoded[1, names.index("TransactionAmt")] == 0
    assert names.index("TransactionAmt") in encoded[1].indices  # Explicit numeric zero for XGBoost.
    for name, row in [("TransactionAmt__missing", 1), ("prior_mean_amount__missing", 1),
                      ("ProductCD_other", 0), ("ProductCD_missing", 1),
                      ("card1_other", 0), ("card1_missing", 1)]:
        assert encoded[row, names.index(name)] == 1


def test_category_cap_groups_rare_and_unseen_values_without_merging_missing():
    frame = synthetic_frame()
    frame["ProductCD"] = ["W", "W", "rare", None]
    preprocessor = FraudPreprocessor(max_categories=3).fit(frame)
    names = preprocessor.get_feature_names_out().tolist()
    assert preprocessor.category_levels_["ProductCD"] == ("W",)
    assert [name for name in names if name.startswith("ProductCD_")] == [
        "ProductCD_value=W", "ProductCD_missing", "ProductCD_other"
    ]
    matrix = preprocessor.transform(frame)
    assert matrix[2, names.index("ProductCD_other")] == 1
    assert matrix[3, names.index("ProductCD_missing")] == 1
