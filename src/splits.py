"""Chronological prefixes and timestamp-safe partition boundaries."""

import numpy as np
import pandas as pd

from .data_loader import ID, TARGET, TIME


def chronological_prefix(transactions: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Sort by time and ID, extending a limited prefix through its final timestamp."""
    ordered = transactions.sort_values([TIME, ID], ignore_index=True)
    if limit is None or limit >= len(ordered):
        return ordered
    if limit < 1:
        raise ValueError("The transaction limit must be positive.")
    times = ordered[TIME].to_numpy()
    stop = times.searchsorted(times[limit - 1], side="right")
    return ordered.iloc[:stop].copy()


def chronological_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cut near 60% and 80% at distinct time boundaries, with both classes per split."""
    times = frame[TIME].to_numpy()
    if np.any(times[1:] < times[:-1]):
        raise ValueError(f"Sort rows by {TIME} before partitioning.")
    boundaries = np.flatnonzero(times[1:] != times[:-1]) + 1
    if len(boundaries) < 2:
        raise ValueError("Three distinct timestamps are required. Use a larger chronological slice.")

    train_options = boundaries[:-1]
    train_stop = int(train_options[np.abs(train_options - 0.6 * len(frame)).argmin()])
    validation_options = boundaries[boundaries > train_stop]
    validation_stop = int(validation_options[np.abs(validation_options - 0.8 * len(frame)).argmin()])
    partitions = {
        "train": frame.iloc[:train_stop],
        "validation": frame.iloc[train_stop:validation_stop],
        "test": frame.iloc[validation_stop:],
    }
    for name, partition in partitions.items():
        if set(partition[TARGET].unique()) != {0, 1}:
            raise ValueError(f"The {name} partition needs both label classes. Use a larger chronological slice.")
    return partitions
