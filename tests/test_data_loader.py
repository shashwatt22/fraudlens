"""Schema and left-join checks with tiny synthetic CSVs and frames."""

import pandas as pd
import pytest

from src.data_loader import IDENTITY_DTYPES, TRANSACTION_DTYPES, load_inputs, merge_identity


def write_synthetic_inputs(directory, **overrides):
    row = dict.fromkeys(TRANSACTION_DTYPES, 1)
    row.update(TransactionID=102, isFraud=0, TransactionDT=100, TransactionAmt=10.5,
               ProductCD="W", card4="visa", card6="debit", P_emaildomain="example.com",
               R_emaildomain=None, addr1=None, unused="ignored")
    row.update(overrides)
    pd.DataFrame([row]).to_csv(directory / "train_transaction.csv", index=False)
    pd.DataFrame([[102, "desktop", None]], columns=list(IDENTITY_DTYPES)).to_csv(
        directory / "train_identity.csv", index=False
    )


def test_load_selects_columns_and_records_missing_values(tmp_path):
    write_synthetic_inputs(tmp_path)
    transactions, identities, sources = load_inputs(tmp_path)
    assert list(transactions.columns) == list(TRANSACTION_DTYPES)
    assert list(identities.columns) == list(IDENTITY_DTYPES)
    assert str(transactions["TransactionAmt"].dtype) == "float32"
    assert str(transactions["card2"].dtype) == "Int16"
    assert str(transactions["ProductCD"].dtype) == "category"
    assert str(transactions["isFraud"].dtype) == "uint8"
    assert sources["transactions"]["csv_columns"] == len(TRANSACTION_DTYPES) + 1
    assert sources["transactions"]["missing"]["addr1"] == {"count": 1, "fraction": 1.0}
    assert sources["identities"]["missing"]["DeviceInfo"]["count"] == 1


@pytest.mark.parametrize(("overrides", "diagnostic"), [
    ({"isFraud": 256}, "binary"),
    ({"isFraud": None}, "binary"),
    ({"TransactionDT": -1}, "nonnegative elapsed seconds"),
    ({"TransactionDT": None}, "nonnegative elapsed seconds"),
    ({"TransactionAmt": float("inf")}, "finite and nonnegative"),
    ({"TransactionID": None}, "nonmissing, unique"),
])
def test_load_rejects_invalid_core_values(tmp_path, overrides, diagnostic):
    write_synthetic_inputs(tmp_path, **overrides)
    with pytest.raises(ValueError, match=diagnostic):
        load_inputs(tmp_path)


def test_load_reports_missing_schema_column(tmp_path):
    write_synthetic_inputs(tmp_path)
    path = tmp_path / "train_identity.csv"
    pd.read_csv(path).drop(columns=["DeviceInfo"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns: DeviceInfo"):
        load_inputs(tmp_path)


def test_load_rejects_duplicate_source_ids(tmp_path):
    write_synthetic_inputs(tmp_path)
    path = tmp_path / "train_identity.csv"
    raw = pd.read_csv(path)
    pd.concat([raw, raw]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="train_identity.csv.*nonmissing, unique"):
        load_inputs(tmp_path)


def test_left_join_preserves_order_and_distinguishes_absent_identity():
    transactions = pd.DataFrame({"TransactionID": [3, 1, 2], "isFraud": [0, 0, 1]})
    identities = pd.DataFrame({"TransactionID": [1, 3, 4], "DeviceType": ["mobile", None, "desktop"]})
    result = merge_identity(transactions, identities)
    assert result["TransactionID"].tolist() == [3, 1, 2]
    assert result["isFraud"].tolist() == [0, 0, 1]
    assert result["DeviceType"].isna().tolist() == [True, False, True]
    assert result["identity_missing"].tolist() == [0, 0, 1]


@pytest.mark.parametrize("duplicate_side", ["transaction", "identity"])
def test_left_join_rejects_duplicates_on_either_side(duplicate_side):
    unique = pd.DataFrame({"TransactionID": [1]})
    duplicated = pd.DataFrame({"TransactionID": [1, 1]})
    left, right = (duplicated, unique) if duplicate_side == "transaction" else (unique, duplicated)
    with pytest.raises(pd.errors.MergeError, match="one-to-one"):
        merge_identity(left, right)
