"""Check timestamp-safe boundaries with a tiny synthetic chronology."""

import pandas as pd
import pytest

from src.splits import chronological_prefix, chronological_split


@pytest.fixture
def ordered():
    times = [1] * 2 + [2] * 2 + [3] * 2 + [4] * 4 + [5] * 3 + [6] * 2
    return pd.DataFrame({"TransactionID": range(1, 16), "TransactionDT": times,
                         "isFraud": [index % 2 for index in range(15)]})


def test_quick_prefix_orders_rows_and_includes_the_whole_final_timestamp(ordered):
    shuffled = ordered.sample(frac=1, random_state=42)
    actual = chronological_prefix(shuffled, 9)
    pd.testing.assert_frame_equal(actual, ordered.iloc[:10])
    pd.testing.assert_frame_equal(chronological_prefix(shuffled, None), ordered)


def test_split_moves_boundaries_to_distinct_timestamps(ordered):
    splits = chronological_split(ordered)
    assert [len(frame) for frame in splits.values()] == [10, 3, 2]
    assert splits["train"]["TransactionDT"].max() < splits["validation"]["TransactionDT"].min()
    assert splits["validation"]["TransactionDT"].max() < splits["test"]["TransactionDT"].min()
    pd.testing.assert_frame_equal(pd.concat(splits.values()), ordered)


def test_split_rejects_a_partition_with_one_class(ordered):
    ordered.loc[ordered["TransactionDT"] == 5, "isFraud"] = 0
    with pytest.raises(ValueError, match="validation partition needs both label classes"):
        chronological_split(ordered)


def test_split_rejects_unsorted_rows_and_too_few_distinct_times(ordered):
    with pytest.raises(ValueError, match="Sort rows"):
        chronological_split(ordered.iloc[::-1])
    with pytest.raises(ValueError, match="Three distinct timestamps"):
        chronological_split(ordered.iloc[:4])
