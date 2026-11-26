"""Row-local transforms and strict, label-free historical feature families."""

import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import pandas as pd

from src.data_loader import ID, TIME

from .data import sha256, write_json

DAY = 86_400


def entity_codes(*columns):
    values = np.column_stack(columns)
    observed = np.isfinite(values).all(axis=1)
    codes = np.full(len(values), -1, dtype=np.int32)
    keys = pd.MultiIndex.from_arrays(values[observed].T)
    codes[observed] = pd.factorize(keys, sort=False)[0]
    return codes


def prior_statistics(times, groups, amounts):
    """Return history through [start, t), excluding the whole current second."""
    times, groups, amounts = map(np.asarray, (times, groups, amounts))
    known = groups >= 0
    if not known.any():
        return pd.DataFrame({name: np.full(len(times), np.nan, dtype=np.float32)
                             for name in ("count", "mean", "std", "gap", "count_1h", "count_24h")})
    current = pd.DataFrame({"group": groups[known], "time": times[known], "amount": amounts[known].astype(np.float64)})
    current["square"] = current.amount ** 2
    agg = current.groupby(["group", "time"], sort=True).agg(count=("amount", "size"), observed=("amount", "count"),
                                                           total=("amount", "sum"), square=("square", "sum"))
    cumulative = agg.groupby(level="group").cumsum()
    previous = cumulative.groupby(level="group").shift(fill_value=0)
    valid = previous.observed.where(previous.observed > 0)
    mean = previous.total / valid
    variance = (previous.square / valid - mean ** 2).clip(lower=0)
    bucket_groups = agg.index.get_level_values("group").to_numpy(dtype=np.int64)
    time = agg.index.get_level_values("time").to_numpy(dtype=np.int64)
    new_group = np.r_[True, bucket_groups[1:] != bucket_groups[:-1]]
    starts = np.maximum.accumulate(np.where(new_group, np.arange(len(time)), 0))
    keys = bucket_groups * (int(time.max()) + DAY + 1) + time
    count = previous["count"].to_numpy()
    result = pd.DataFrame({"count": count, "mean": mean, "std": np.sqrt(variance),
                           "gap": np.where(new_group, np.nan, np.diff(time, prepend=0))}, index=agg.index)
    for name, width in (("count_1h", 3_600), ("count_24h", DAY)):
        left = np.maximum(np.searchsorted(keys, keys - width, side="left"), starts)
        result[name] = count - count[left]
    result = result.reindex(pd.MultiIndex.from_frame(current[["group", "time"]]))
    result.index = np.flatnonzero(known)
    return result.reindex(np.arange(len(times))).astype(np.float32)


def append_history(output, prefix, times, key, amounts):
    statistics = prior_statistics(times, key, amounts)
    for name in statistics:
        output[f"{prefix}_prior_{name}"] = statistics[name]
    mean = statistics["mean"].to_numpy()
    deviation = amounts - mean
    output[f"{prefix}_amount_deviation"] = deviation
    output[f"{prefix}_amount_ratio"] = np.divide(amounts, mean, out=np.full_like(mean, np.nan), where=mean > 0)


def advanced_features(data, names, metadata, raw):
    inputs = {name: data[:, col] for col, name in enumerate(names)}
    times = metadata[TIME].to_numpy(dtype=np.int64)
    amount = inputs["TransactionAmt"]
    days = times // DAY
    output = {"amount_log": np.log1p(amount), "amount_fraction": np.remainder(amount, 1),
              "day_sin": np.sin(2*np.pi*(times % DAY)/DAY), "day_cos": np.cos(2*np.pi*(times % DAY)/DAY),
              "week_sin": np.sin(2*np.pi*(days % 7)/7), "week_cos": np.cos(2*np.pi*(days % 7)/7)}
    for name in ("D1", "D4", "D10", "D11", "D15"):
        output[f"day_minus_{name}"] = days - inputs[name]
    for left, right in (("D15", "D1"), ("D10", "D1"), ("D15", "D10"), ("C1", "C2")):
        output[f"{left}_minus_{right}"] = inputs[left] - inputs[right]
    for prefix in ("C", "D", "V", "id_"):
        columns = [values for name, values in inputs.items() if re.fullmatch(prefix + r"\d+", name)]
        output[f"observed_{prefix.rstrip('_')}"] = sum(np.isfinite(col).astype(np.uint16) for col in columns)

    card = inputs["card1"]
    address = inputs["addr1"]
    append_history(output, "card_address", times, entity_codes(card, address), amount)
    d1_anchor = np.round(output["day_minus_D1"])
    append_history(output, "card_address_anchor", times, entity_codes(card, address, d1_anchor), amount)
    append_history(output, "card_address_email", times, entity_codes(card, address, inputs["P_emaildomain"]), amount)
    for name in ("ProductCD", "P_emaildomain", "R_emaildomain", "DeviceInfo"):
        groups = entity_codes(inputs[name])
        output[f"{name}_past_frequency"] = prior_statistics(times, groups, amount)["count"]

    tx = pd.read_csv(raw / "train_transaction.csv", usecols=[ID, "P_emaildomain", "R_emaildomain"], nrows=len(times))
    if not np.array_equal(tx[ID].to_numpy(), metadata[ID].to_numpy()):
        raise ValueError("Email rows are out of order.")
    purchaser = tx["P_emaildomain"].astype("string").str.lower()
    recipient = tx["R_emaildomain"].astype("string").str.lower()
    same_provider = purchaser.eq(recipient).astype("Float32")
    output["email_equal"] = same_provider
    for prefix, series in (("p", purchaser), ("r", recipient)):
        for provider, pattern in (("google", r"^(?:gmail|googlemail)\."), ("yahoo", r"^(?:yahoo|ymail|rocketmail)\."),
                                  ("microsoft", r"^(?:outlook|hotmail|live|msn)\.")):
            output[f"{prefix}_email_{provider}"] = series.str.contains(pattern, regex=True).astype("Float32")
    identity = pd.read_csv(raw / "train_identity.csv", usecols=[ID, "DeviceInfo"]).set_index(ID)
    device = identity.reindex(metadata[ID].to_numpy())["DeviceInfo"].astype("string").str.lower()
    for family, pattern in (("apple", "iphone|ipad|mac"), ("windows", "windows"), ("samsung", "samsung|^sm-|^sgh-"),
                            ("huawei", "huawei|honor")):
        output[f"device_{family}"] = device.str.contains(pattern, regex=True).astype("Float32").to_numpy(dtype=np.float32)
    return pd.DataFrame(output).astype(np.float32)


def build_advanced(output, data, cache, metadata, raw):
    manifest = output / "advanced.json"
    key = {"features_sha256": sha256(output / "features.json"), "implementation_sha256": sha256(Path(__file__))}
    if manifest.is_file():
        report = json.loads(manifest.read_bytes())
        if any(report[k] != v for k, v in key.items()):
            raise ValueError("Advanced cache code or source differs; use a fresh artifact root.")
        return report
    started = perf_counter()
    frame = advanced_features(data, cache["features"], metadata, raw)
    rankers = cache["features"] + list(frame)
    rankers_cache = np.lib.format.open_memmap(output / "features_advanced.npy", mode="w+", dtype=np.float32,
                                             shape=(len(frame), len(rankers)), fortran_order=True)
    rankers_cache[:, :data.shape[1]] = data
    rankers_cache[:, data.shape[1]:] = frame.to_numpy(dtype=np.float32)
    rankers_cache.flush()
    report = {**key, "features": rankers, "added_features": list(frame), "seconds": perf_counter() - started}
    write_json(manifest, report)
    return report
