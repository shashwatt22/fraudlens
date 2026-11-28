"""Strictly prior seven-day and D-offset statistics for the existing D1 anchor."""

import numpy as np
import pandas as pd

from src.v2.advanced import DAY, entity_codes


WEEK = 7 * DAY
PREFIX = "card_address_anchor"
OFFSETS = ("D4", "D10", "D15")
INPUTS = ("TransactionAmt", *(f"day_minus_{name}" for name in OFFSETS))
FEATURES = (
    f"{PREFIX}_prior_count_7d", f"{PREFIX}_prior_amount_mean_7d",
    f"{PREFIX}_prior_amount_std_7d", f"{PREFIX}_amount_deviation_7d",
    *(name for offset in OFFSETS for name in (f"{PREFIX}_prior_day_minus_{offset}_mean",
                                              f"{PREFIX}_prior_day_minus_{offset}_std",
                                              f"{PREFIX}_day_minus_{offset}_deviation")),
)


def anchor_keys(matrix, names):
    columns = {name: i for i, name in enumerate(names)}
    return entity_codes(matrix[:, columns["card1"]], matrix[:, columns["addr1"]],
                        np.round(matrix[:, columns["day_minus_D1"]]))


def _moments(values, starts, ids, first, left):
    """Compute observed-value moments for full prior and seven-day windows."""
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.).astype(np.float64, copy=False)
    # Reset cumulative sums at entity boundaries so other entities cannot affect a prefix.
    buckets = np.column_stack([np.add.reduceat(term, starts, axis=0)
                               for term in (finite.astype(np.int64), safe, safe ** 2)])
    cumulative = pd.DataFrame(buckets).groupby(ids, sort=False).cumsum().to_numpy()
    previous = np.empty_like(cumulative)
    previous[0] = 0
    previous[1:] = cumulative[:-1]
    previous[first] = 0
    prior = previous.copy()
    count = values.shape[1]
    amount = [0, count, 2 * count]
    prior[:, amount] -= previous[left][:, amount]
    observed, total, squares = np.split(prior, 3, axis=1)
    mean = np.divide(total, observed, out=np.full_like(total, np.nan), where=observed > 0)
    variance = np.divide(squares, observed, out=np.full_like(total, np.nan), where=observed > 0) - mean ** 2
    std = np.sqrt(np.maximum(variance, 0.))
    std[observed == 1] = 0.
    return mean, std


def behavioral_block(times, keys, measures):
    """Return only the 13 requested additions, in source row order and float32."""
    times = np.asarray(times, dtype=np.int64)
    keys = np.asarray(keys)
    measures = np.asarray(measures, dtype=np.float32)
    if keys.shape != times.shape or measures.shape != (len(times), len(INPUTS)):
        raise ValueError("Expected aligned times, entity codes and four measurements.")
    result = np.full((len(times), len(FEATURES)), np.nan, dtype=np.float32)
    known = np.flatnonzero(keys >= 0)
    if not len(known):
        return result

    order = known[np.lexsort((times[known], keys[known]))]
    time, ids = times[order], keys[order].astype(np.int64)
    new_entity = np.r_[True, ids[1:] != ids[:-1]]
    new_bucket = new_entity | np.r_[True, time[1:] != time[:-1]]
    starts = np.flatnonzero(new_bucket)
    bucket_rows = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(order)]))
    first = np.flatnonzero(new_entity[starts])
    lower_bound = np.maximum.accumulate(np.where(new_entity[starts], np.arange(len(starts)), 0))
    duration = int(time.max()) - int(time.min())
    timeline = ids[starts] * (duration + WEEK + 1) + time[starts] - int(time.min())
    left = np.maximum(np.searchsorted(timeline, timeline - WEEK, side="left"), lower_bound)
    result[order, 0] = (starts - starts[left])[bucket_rows]

    ordered_values = measures[order]
    mean, std = _moments(ordered_values, starts, ids[starts], first, left)
    for i in range(len(INPUTS)):
        row_mean = mean[bucket_rows, i]
        current = ordered_values[:, i]
        delta = np.where(np.isfinite(current), current - row_mean, np.nan)
        result[order, 1 + 3 * i:4 + 3 * i] = np.column_stack((row_mean, std[bucket_rows, i], delta))
    return result
