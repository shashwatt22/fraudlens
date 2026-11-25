"""Compute behavior from earlier observations of an approximate entity."""

import numpy as np
import pandas as pd

from .data_loader import AMOUNT, ID, IDENTITY_MISSING, PRODUCT, TARGET, TIME

ENTITY = "entity_key"
KEY_COLUMNS = ["card1", "card2", "addr1", "addr2"]
WINDOWS = {"prior_count_1h": 3_600, "prior_count_24h": 86_400}
HISTORICAL_COLUMNS = [
    "prior_txn_count",
    "prior_mean_amount",
    "amount_to_prior_mean",
    "seconds_since_prior_txn",
    *WINDOWS,
    "is_new_product",
]
NUMERIC_BASIC = [AMOUNT, "log_amount", IDENTITY_MISSING]
CATEGORICAL_BASIC = [
    PRODUCT, "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo",
]
METADATA_COLUMNS = [ID, TIME, ENTITY]


def feature_schema(frame: pd.DataFrame) -> dict:
    """Separate predictive inputs from labels and metadata for downstream use."""
    return {
        "label": TARGET,
        "metadata": METADATA_COLUMNS,
        "basic_numeric": NUMERIC_BASIC,
        "basic_categorical": CATEGORICAL_BASIC,
        "historical_numeric": HISTORICAL_COLUMNS,
        "entity_key_components": KEY_COLUMNS,
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
    }


def _window_counts(times: np.ndarray, prior: np.ndarray, starts: np.ndarray) -> dict:
    """Search sorted distinct times within each entity, including the lower bound."""
    counts = {column: np.zeros(len(times), dtype=np.int32) for column in WINDOWS}
    for start, end in zip(starts, np.r_[starts[1:], len(times)]):
        if end - start == 1:
            continue
        group_times = times[start:end]
        group_prior = prior[start:end]
        for column, seconds in WINDOWS.items():
            first = np.searchsorted(group_times, group_times - seconds, side="left")
            counts[column][start:end] = group_prior - group_prior[first]
    return counts


def _prior_by_timestamp(known: pd.DataFrame) -> pd.DataFrame:
    # Collapse equal times first. Every row at a time gets the same history.
    buckets = known.groupby([ENTITY, TIME], sort=True, observed=True).agg(
        txn_count=(TIME, "size"),
        amount_count=(AMOUNT, "count"),
        amount_total=(AMOUNT, "sum"),
    )
    cumulative = buckets.groupby(level=ENTITY, sort=False).cumsum()
    prior = cumulative.groupby(level=ENTITY, sort=False).shift(fill_value=0)
    prior_mean = prior["amount_total"].div(prior["amount_count"].where(prior["amount_count"] > 0))

    times = buckets.index.get_level_values(TIME).to_numpy(dtype=np.int64)
    entities = np.asarray(buckets.index.codes[0])
    starts = np.flatnonzero(np.r_[True, entities[1:] != entities[:-1]]) if len(times) else np.array([], dtype=int)
    gap = np.diff(times, prepend=0).astype(np.float64)
    gap[starts] = np.nan
    return pd.DataFrame(
        {
            "prior_txn_count": prior["txn_count"],
            "prior_mean_amount": prior_mean,
            "seconds_since_prior_txn": gap,
            **_window_counts(times, prior["txn_count"].to_numpy(), starts),
        },
        index=buckets.index,
    )


def add_behavioral_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Preserve row order and compute history independently of the fraud label."""
    frame = transactions.reset_index(drop=True).copy()
    key_parts = frame[KEY_COLUMNS].astype("string")
    frame[ENTITY] = key_parts[KEY_COLUMNS[0]].str.cat(key_parts[KEY_COLUMNS[1:]], sep="|")
    frame["log_amount"] = np.log1p(frame[AMOUNT]).astype(np.float32)

    # An incomplete key stays missing and never contributes to a shared history.
    known = frame.loc[frame[ENTITY].notna(), [ENTITY, TIME, AMOUNT, PRODUCT]].copy()
    known[AMOUNT] = known[AMOUNT].astype(np.float64)
    buckets = _prior_by_timestamp(known)
    history = buckets.reindex(pd.MultiIndex.from_frame(known[[ENTITY, TIME]]))
    history.index = known.index
    history = history.reindex(frame.index)

    frame["prior_txn_count"] = history["prior_txn_count"].fillna(0).astype(np.int32)
    mean = history["prior_mean_amount"]
    frame["prior_mean_amount"] = mean.astype(np.float32)
    frame["amount_to_prior_mean"] = frame[AMOUNT].div(mean.where(mean > 0)).astype(np.float32)
    frame["seconds_since_prior_txn"] = history["seconds_since_prior_txn"].astype("Int32")
    for column in WINDOWS:
        frame[column] = history[column].fillna(0).astype(np.int32)

    # A product is new exactly when its first observed time equals the current time.
    product_rows = known.dropna(subset=[PRODUCT])
    first_seen = product_rows.groupby([ENTITY, PRODUCT], sort=False, observed=True)[TIME].transform("min")
    novelty = product_rows[TIME].eq(first_seen).astype("Int8")
    frame["is_new_product"] = novelty.reindex(frame.index)
    return frame
