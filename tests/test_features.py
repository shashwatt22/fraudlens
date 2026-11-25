"""Hand-computed data history checks. All fixtures are synthetic."""

import numpy as np
import pandas as pd
import pytest

from src.features import ENTITY, HISTORICAL_COLUMNS, KEY_COLUMNS, add_behavioral_features


def synthetic_transactions(rows):
    frame = pd.DataFrame(rows, columns=["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD"])
    frame = frame.assign(card1=100, card2=200, addr1=300, addr2=80, isFraud=0)
    return frame.astype({column: "Int32" for column in KEY_COLUMNS})


@pytest.fixture
def transactions():
    return synthetic_transactions([
        (101, 1_000, 10, "W"),
        (102, 1_000, 30, "W"),
        (103, 4_600, 50, "C"),
        (104, 4_600, 70, "T"),
        (105, 4_601, 90, "W"),
        (106, 87_400, 120, "C"),
        (107, 87_401, 180, "C"),
        (108, 4_600, 100, "C"),
    ])


def test_same_timestamp_rows_see_only_earlier_timestamp_groups(transactions):
    transactions.loc[7, "card1"] = 999  # Interleaved independent entity.
    shuffled = transactions.sample(frac=1, random_state=42)
    actual = add_behavioral_features(shuffled)
    assert actual["TransactionID"].tolist() == shuffled["TransactionID"].tolist()
    actual = actual.set_index("TransactionID")

    assert actual.loc[[101, 102], "prior_txn_count"].tolist() == [0, 0]
    assert actual.loc[[101, 102], "is_new_product"].tolist() == [1, 1]
    assert actual.loc[[103, 104], "prior_txn_count"].tolist() == [2, 2]
    assert actual.loc[[103, 104], "prior_mean_amount"].tolist() == [20, 20]
    assert actual.loc[[103, 104], "amount_to_prior_mean"].tolist() == [2.5, 3.5]
    assert actual.loc[[103, 104], "seconds_since_prior_txn"].tolist() == [3_600, 3_600]
    assert actual.loc[105, "prior_txn_count"] == 4
    assert actual.loc[105, "prior_mean_amount"] == 40
    assert actual.loc[105, "is_new_product"] == 0
    assert actual.loc[108, "prior_txn_count"] == 0


def test_window_bounds_include_exact_lower_bound_and_exclude_current_time(transactions):
    transactions.loc[7, "card1"] = 999
    actual = add_behavioral_features(transactions).set_index("TransactionID")
    assert actual.loc[103, "prior_count_1h"] == 2
    assert actual.loc[105, "prior_count_1h"] == 2
    assert actual.loc[106, "prior_count_24h"] == 5
    assert actual.loc[107, "prior_count_24h"] == 4
    assert actual.loc[107, "prior_count_1h"] == 1


def test_future_transactions_and_labels_cannot_change_earlier_features(transactions):
    prefix = transactions.iloc[:4].copy()
    expected = add_behavioral_features(prefix)[HISTORICAL_COLUMNS]
    changed = transactions.copy()
    changed.loc[changed["TransactionDT"] > 4_600, "TransactionAmt"] = 2_000_000
    changed.loc[changed["TransactionDT"] > 4_600, "ProductCD"] = "T"
    changed["isFraud"] = 1
    added = synthetic_transactions([(109, 94_000, 3_000_000, "W")])
    extended = pd.concat([added, changed], ignore_index=True)
    result = add_behavioral_features(extended).set_index("TransactionID")
    pd.testing.assert_frame_equal(
        result.loc[prefix["TransactionID"], HISTORICAL_COLUMNS].reset_index(drop=True), expected
    )


def test_missing_key_rows_do_not_share_history():
    frame = synthetic_transactions([(1, 10, 20, "W"), (2, 20, 30, "W")])
    frame["addr2"] = pd.NA
    actual = add_behavioral_features(frame)
    assert actual[ENTITY].isna().all()
    assert actual[["prior_txn_count", "prior_count_1h", "prior_count_24h"]].eq(0).all().all()
    assert actual[["prior_mean_amount", "amount_to_prior_mean", "seconds_since_prior_txn",
                   "is_new_product"]].isna().all().all()
    assert actual.loc[0, "log_amount"] == pytest.approx(np.log1p(20))


def test_cold_start_missing_amount_and_product_and_zero_denominator():
    frame = synthetic_transactions([
        (1, 10, None, None), (2, 20, 0, "W"), (3, 30, 10, None), (4, 40, None, "W"),
    ])
    actual = add_behavioral_features(frame)
    assert actual["prior_txn_count"].tolist() == [0, 1, 2, 3]
    assert actual["seconds_since_prior_txn"].tolist() == [pd.NA, 10, 10, 10]
    assert actual["is_new_product"].tolist() == [pd.NA, 1, pd.NA, 0]
    assert actual["amount_to_prior_mean"].isna().all()
    assert actual["log_amount"].isna().tolist() == [True, False, False, True]
    assert actual["prior_mean_amount"].isna().tolist() == [True, True, False, False]
    assert actual["prior_mean_amount"].iloc[2:].tolist() == [0, 5]
