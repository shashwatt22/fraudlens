"""Cache development-side covariates without exposing the retrospective labels."""

import csv
from dataclasses import dataclass
from hashlib import file_digest
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.data_loader import ID, TARGET, TIME
from src.features import CATEGORICAL_BASIC, HISTORICAL_COLUMNS
from src.training_data import read_phase1_manifest

CHUNK_ROWS = 16_384
METADATA = (ID, TIME, TARGET)
IDENTITY_NUMERIC = tuple(f"id_{i:02d}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 18, 19, 20, 21, 22, 24, 25, 26, 32))
TRANSACTION_NUMERIC = {"TransactionAmt", "dist1", "dist2", *(f"C{i}" for i in range(1, 15)),
                       *(f"D{i}" for i in range(1, 16)), *(f"V{i}" for i in range(1, 340))}
FRACTIONS = (.45, .60, .75, .90)
CACHE_SCHEMA = 1


@dataclass(frozen=True)
class Fold:
    train_end: int
    validation_end: int


def walk_forward(times, fractions=FRACTIONS):
    """Advance each cut past its whole tied timestamp group."""
    times = np.asarray(times)
    if len(times) < 10 or np.any(times[1:] < times[:-1]):
        raise ValueError("Walk-forward folds require a nontrivial sorted development set.")
    cuts = [int(np.searchsorted(times, times[int(len(times) * fraction) - 1], side="right"))
            for fraction in fractions]
    if not 0 < cuts[0] < cuts[1] < cuts[2] < cuts[3] < len(times):
        raise ValueError("Timestamp groups leave an empty fold or decision-policy block.")
    return [Fold(start, end) for start, end in zip(cuts, cuts[1:])]


def sha256(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def write_json(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def header(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        fields = next(csv.reader(file))
    if len(fields) != len(set(fields)):
        raise ValueError(f"CSV has repeated columns: {path.name}")
    return fields


def development_metadata(v1):
    manifest, digest = read_phase1_manifest(v1)
    parts = [pd.read_parquet(v1 / manifest["files"][name], columns=list(METADATA))
             for name in ("train", "validation")]
    frame = pd.concat(parts, ignore_index=True)
    first_final = manifest["partitions"]["test"]["first_second"]
    if frame[ID].duplicated().any() or frame[TIME].max() >= first_final:
        raise ValueError("Development rows overlap the original final-test time range.")
    return manifest, digest, frame


class FirstSeen:
    """Allocate category codes monotonically in encounter order across chunks."""

    def __init__(self):
        self.codes = {}

    def encode(self, values):
        inverse, distinct = pd.factorize(values, sort=False)
        lookup = np.fromiter((self.codes.setdefault(value, len(self.codes)) for value in distinct),
                             dtype=np.float32, count=len(distinct))
        result = np.full(len(values), np.nan, dtype=np.float32)
        valid = inverse >= 0
        result[valid] = lookup[inverse[valid]]
        return result


def build_cache(raw, v1, output, provenance, *, retrospective_metadata=None):
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    v1_manifest, v1_digest, frame = development_metadata(v1)
    development_rows = len(frame)
    parts = ["train", "validation"]
    fields = list(METADATA)
    if retrospective_metadata is not None:
        if not provenance.get("decision_freeze_sha256"):
            raise ValueError("A frozen V2 decision is required for retrospective covariates.")
        fields = [ID, TIME]
        if list(retrospective_metadata) != fields or len(retrospective_metadata) != v1_manifest["partitions"]["test"]["rows"]:
            raise ValueError("Retrospective metadata must contain only ID and time for all final rows.")
        frame = pd.concat([frame[fields], retrospective_metadata], ignore_index=True)
        if frame[ID].duplicated().any() or not frame[TIME].is_monotonic_increasing:
            raise ValueError("Retrospective transaction order is inconsistent with development.")
        parts.append("test")
    tx_path = raw / "train_transaction.csv"
    id_path = raw / "train_identity.csv"
    metadata = output / "features.json"
    hashes = {p.name: sha256(p) for p in (tx_path, id_path)}
    expected = {"schema": CACHE_SCHEMA, "v1_manifest_sha256": v1_digest, "raw_sha256": hashes}
    if retrospective_metadata is not None:
        expected["decision_freeze_sha256"] = provenance["decision_freeze_sha256"]
    if metadata.exists():
        previous = json.loads(metadata.read_bytes())
        if any(previous.get(key) != value for key, value in expected.items()):
            raise ValueError("Feature cache belongs to a different input snapshot. Use a fresh artifact root.")
        return previous

    transaction_columns = [name for name in header(tx_path) if name not in METADATA]
    identity_columns = [name for name in header(id_path) if name != ID]
    if len(transaction_columns) != 391 or len(identity_columns) != 40:
        raise ValueError("Unexpected source columns in the IEEE-CIS training files.")
    numeric = TRANSACTION_NUMERIC | set(IDENTITY_NUMERIC) | {"identity_missing", *HISTORICAL_COLUMNS}
    features = transaction_columns + identity_columns + ["identity_missing", *HISTORICAL_COLUMNS]
    category = [name for name in features if name not in numeric]
    encoders = {name: FirstSeen() for name in category}
    types = {name: "float32" if name in numeric else "str" for name in features}
    types.update({ID: "int32", TIME: "int32", TARGET: "uint8"})
    identities = pd.read_csv(id_path, dtype={name: types[name] for name in [ID, *identity_columns]}).set_index(ID)
    if not identities.index.is_unique:
        raise ValueError("Identity IDs require a one-to-zero-or-one match.")
    # Future identity rows are removed before category allocation or feature summaries.
    identities = identities.loc[identities.index.isin(frame[ID])]
    history = pd.concat([pd.read_parquet(v1 / v1_manifest["files"][name], columns=list(HISTORICAL_COLUMNS))
                         for name in parts], ignore_index=True)
    matrix = np.lib.format.open_memmap(output / "features.npy", mode="w+", dtype=np.float32,
                                       shape=(len(frame), len(features)), fortran_order=True)
    matched = np.zeros(len(frame), dtype=bool)
    stop = 0
    tx_fields = [*fields, *transaction_columns]
    for chunk in pd.read_csv(tx_path, usecols=tx_fields, dtype={name: types[name] for name in tx_fields},
                             nrows=len(frame), chunksize=CHUNK_ROWS):
        start, stop = stop, stop + len(chunk)
        reference = frame.iloc[start:stop]
        if any(not np.array_equal(chunk[name].to_numpy(), reference[name].to_numpy()) for name in fields):
            raise ValueError("Raw chronological prefix differs from the V1 development partitions.")
        identity = identities.reindex(chunk[ID].to_numpy())
        matched[start:stop] = chunk[ID].isin(identities.index).to_numpy()
        for col, name in enumerate(features):
            if name == "identity_missing":
                values = ~matched[start:stop]
            elif name in HISTORICAL_COLUMNS:
                values = history[name].iloc[start:stop]
            else:
                values = (chunk if name in transaction_columns else identity)[name]
            matrix[start:stop, col] = (encoders[name].encode(values) if name in encoders
                                       else np.asarray(values, dtype=np.float32))
    if stop != len(frame):
        raise ValueError("The raw development prefix has too few rows.")
    matrix.flush()
    frame.to_parquet(output / "development_metadata.parquet", index=False)
    folds = walk_forward(frame[TIME].iloc[:development_rows].to_numpy())
    policy_end = folds[-1].validation_end
    scans = {name: _feature_audit(matrix[:, i], folds, name in encoders) for i, name in enumerate(features)}
    information = {**expected, "features": features, "categorical": category, "rows": len(frame),
                   "folds": [dict(train_end=f.train_end, validation_end=f.validation_end) for f in folds],
                   "policy_rows": development_rows - policy_end, "time_min": int(frame[TIME].min()),
                   "time_max": int(frame[TIME].max()), "holdout_start": v1_manifest["partitions"]["test"]["first_second"],
                   "identity_matches": int(matched.sum()), "groups": feature_sets(features, identity_columns),
                   "feature_audit": scans, "data_seconds": perf_counter() - started, "provenance": provenance}
    write_json(metadata, information)
    return information


def _feature_audit(column, folds, categorical):
    summary = []
    for start, end in ((0, folds[0].train_end), *((f.train_end, f.validation_end) for f in folds)):
        values = column[start:end]
        valid = values[np.isfinite(values)]
        report = {"missing_rate": 1 - len(valid) / len(values)}
        if len(valid):
            report.update({"distinct": int(np.unique(valid).size)} if categorical else
                          dict(zip(("p05", "median", "p95"), map(float, np.quantile(valid, (.05, .5, .95))))))
        summary.append(report)
    return summary


def feature_sets(features, identity):
    raw = [name for name in features if name not in HISTORICAL_COLUMNS]
    return {"rich_raw": raw, "rich_history": features,
            "basic_native": [name for name in features if name in {"TransactionAmt", "identity_missing", *CATEGORICAL_BASIC}],
            "no_anonymous": [name for name in features if not name.startswith(("C", "D", "V", "id_"))
                             or name in {"DeviceInfo", "DeviceType"}],
            "no_vesta": [name for name in features if not name.startswith("V")],
            "no_identity": [name for name in features if name not in identity and name != "identity_missing"]}
