"""Summarize observed data without fitting transformations."""

import pandas as pd

from .data_loader import AMOUNT, IDENTITY_MISSING, PRODUCT, TARGET, TIME, missing_summary
from .features import ENTITY


def describe_partition(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "fraud_rows": int(frame[TARGET].sum()),
        "fraud_rate": float(frame[TARGET].mean()),
        "first_second": int(frame[TIME].min()),
        "last_second": int(frame[TIME].max()),
    }


def describe_data(frame: pd.DataFrame, partitions: dict, sources: dict) -> dict:
    amounts = frame[AMOUNT].dropna()
    amount_summary = None
    if not amounts.empty:
        amount_summary = {
            "observations": len(amounts),
            "minimum": float(amounts.min()),
            "mean": float(amounts.mean()),
            "median": float(amounts.median()),
            "p95": float(amounts.quantile(0.95)),
            "maximum": float(amounts.max()),
        }

    return {
        "sources": sources,
        "processed": {
            **describe_partition(frame),
            "columns": len(frame.columns),
            "distinct_timestamps": int(frame[TIME].nunique()),
            "identity_matches": int(frame[IDENTITY_MISSING].eq(0).sum()),
            "transactions_without_identity": int(frame[IDENTITY_MISSING].sum()),
            "transactions_without_entity_key": int(frame[ENTITY].isna().sum()),
            "transactions_with_zero_prior_count": int(frame["prior_txn_count"].eq(0).sum()),
            "observed_entity_keys": int(frame[ENTITY].nunique()),
            "transaction_amount": amount_summary,
            "product_counts": {
                str(product): int(count)
                for product, count in frame[PRODUCT].value_counts().items()
                if count
            },
            "missing": missing_summary(frame),
        },
        "partitions": {name: describe_partition(partition) for name, partition in partitions.items()},
    }
