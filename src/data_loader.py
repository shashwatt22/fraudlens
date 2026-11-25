"""Load the selected IEEE-CIS columns and validate the transaction grain."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ID = "TransactionID"
TIME = "TransactionDT"
TARGET = "isFraud"
AMOUNT = "TransactionAmt"
PRODUCT = "ProductCD"
IDENTITY_MISSING = "identity_missing"

TRANSACTION_DTYPES = {
    ID: "Int32",
    TARGET: "Int64",
    TIME: "Int32",
    AMOUNT: "float32",
    PRODUCT: "category",
    "card1": "Int32",
    "card2": "Int16",
    "card3": "Int16",
    "card4": "category",
    "card5": "Int16",
    "card6": "category",
    "addr1": "Int16",
    "addr2": "Int16",
    "P_emaildomain": "category",
    "R_emaildomain": "category",
}
IDENTITY_DTYPES = {ID: "Int32", "DeviceType": "category", "DeviceInfo": "category"}


def missing_summary(frame: pd.DataFrame) -> dict:
    """Report missing counts and fractions for every loaded column."""
    return {
        name: {"count": int(count), "fraction": float(count / len(frame))}
        for name, count in frame.isna().sum().items()
    }


def _read_selected(path: Path, dtypes: dict) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Put the IEEE-CIS training CSVs in the data directory.")

    with path.open(encoding="utf-8-sig", newline="") as file:
        header = next(csv.reader(file), [])

    missing = sorted(dtypes.keys() - set(header))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")
    if len(set(header)) != len(header):
        raise ValueError(f"{path.name} has duplicate column names.")

    frame = pd.read_csv(path, usecols=list(dtypes), dtype=dtypes)
    if frame.empty:
        raise ValueError(f"{path.name} contains no data rows.")
    if frame[ID].isna().any() or not frame[ID].is_unique:
        raise ValueError(f"{path.name} requires nonmissing, unique {ID} values.")

    frame[ID] = frame[ID].astype(np.int32)
    info = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "rows": len(frame),
        "csv_columns": len(header),
        "loaded_columns": len(frame.columns),
        "missing": missing_summary(frame),
    }
    return frame, info


def load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read training sources with explicit dtypes and validate required values."""
    transactions, transaction_info = _read_selected(
        data_dir / "train_transaction.csv", TRANSACTION_DTYPES
    )
    identities, identity_info = _read_selected(data_dir / "train_identity.csv", IDENTITY_DTYPES)

    if not transactions[TARGET].isin([0, 1]).all():
        raise ValueError(f"{TARGET} must be nonmissing and binary (0 or 1).")
    if transactions[TIME].isna().any() or transactions[TIME].lt(0).any():
        raise ValueError(f"{TIME} must contain nonnegative elapsed seconds without missing values.")
    observed_amounts = transactions[AMOUNT].dropna()
    if not np.isfinite(observed_amounts).all() or observed_amounts.lt(0).any():
        raise ValueError(f"Observed {AMOUNT} values must be finite and nonnegative.")

    transactions[TARGET] = transactions[TARGET].astype(np.uint8)
    transactions[TIME] = transactions[TIME].astype(np.int32)
    transaction_info["fraud_rows"] = int(transactions[TARGET].sum())
    transaction_info["fraud_rate"] = float(transactions[TARGET].mean())
    identity_info["unmatched_ids"] = int((~identities[ID].isin(transactions[ID])).sum())
    return transactions, identities, {"transactions": transaction_info, "identities": identity_info}


def merge_identity(transactions: pd.DataFrame, identities: pd.DataFrame) -> pd.DataFrame:
    """Attach at most one identity row to every transaction, preserving left order."""
    if transactions[ID].isna().any() or identities[ID].isna().any():
        raise ValueError(f"Cannot merge missing {ID} values.")

    merged = transactions.merge(identities, on=ID, how="left", sort=False,
                                validate="one_to_one", indicator=True)
    if len(merged) != len(transactions):
        raise ValueError("The identity join changed the transaction row count.")
    merged[IDENTITY_MISSING] = merged.pop("_merge").eq("left_only").astype(np.uint8)
    return merged
