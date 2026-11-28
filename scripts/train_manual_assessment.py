"""Fit the separate manual assessment model on the verified development prefix."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
import xgboost as xgb

from src.dashboard_v3 import manual_model as manual
from src.dashboard_v3.store import SCORE_SHA, V1_SHA
from src.evaluate import select_threshold

NAMES = ("fit", "threshold_selection", "development_verification")
STOPS = (0, 354_324, 425_188, 472_432)
EXPECTED = ((86400, 8745772, 11988), (8745798, 10766686, 2957), (10766698, 12192842, 1654))
PARAMETERS = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "max_depth": 4,
              "eta": .06, "min_child_weight": 15, "lambda": 4, "seed": 42, "nthread": 6}
ROUNDS = 160


def digest(file):
    with file.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def metrics(y, probability, threshold):
    flags = probability.astype(np.float64) >= threshold
    return {"average_precision": average_precision_score(y, probability), "roc_auc": roc_auc_score(y, probability),
            "precision": precision_score(y, flags, zero_division=0), "recall": recall_score(y, flags),
            "f1": f1_score(y, flags), "accuracy": accuracy_score(y, flags), "brier": brier_score_loss(y, probability),
            "log_loss": log_loss(y, probability), "flagged": int(flags.sum()),
            "confusion_matrix": confusion_matrix(y, flags, labels=[0, 1]).tolist()}


def run(raw, v1, v3, output):
    started = perf_counter()
    if output.exists():
        raise FileExistsError("Choose a fresh output directory.")
    raw_manifest = v3 / "step7c-retrospective" / "covariates" / "features.json"
    source = v3 / "step7c-retrospective" / "score_manifest.json"
    if digest(source) != SCORE_SHA or digest(v1 / "full" / "manifest.json") != V1_SHA:
        raise ValueError("The frozen lineage files do not match.")
    scores = json.loads(source.read_bytes())
    advanced = raw_manifest.with_name("advanced.json")
    if digest(advanced) != scores["feature_manifest_sha256"]:
        raise ValueError("The development feature manifest lineage is invalid.")
    if digest(raw_manifest) != json.loads(advanced.read_bytes())["features_sha256"]:
        raise ValueError("The raw feature manifest lineage is invalid.")
    if digest(raw) != json.loads(raw_manifest.read_bytes())["raw_sha256"]["train_transaction.csv"]:
        raise ValueError("The transaction source differs from the verified benchmark.")
    plan = v3 / "step7a-refit" / "refit_plan.json"
    if digest(plan) != scores["frozen_hashes"]["refit_plan.json"]:
        raise ValueError("The verified V3 plan differs.")
    parts = json.loads(plan.read_bytes())["partitions"]
    if parts["pre_policy"]["end"] != STOPS[2] or parts["policy"]["end"] != STOPS[3]:
        raise ValueError("The development boundaries differ from the verified plan.")
    cols = (manual.AMOUNT, *manual.FIELDS, "TransactionDT", "isFraud")
    raw_frame = pd.read_csv(raw, usecols=cols, nrows=STOPS[-1],
                            dtype={manual.AMOUNT: "float32", "TransactionDT": "int32", "isFraud": "uint8"})
    times = raw_frame["TransactionDT"]
    labels = raw_frame["isFraud"].to_numpy()
    if (len(raw_frame) != STOPS[-1] or not times.is_monotonic_increasing or not np.isin(labels, [0, 1]).all()
            or not raw_frame[manual.AMOUNT].between(manual.MIN_AMOUNT, manual.MAX_AMOUNT).all()):
        raise ValueError("The development prefix does not match its expected schema.")
    for lo, hi, (first, last, fraud) in zip(STOPS[:-1], STOPS[1:], EXPECTED, strict=True):
        if (times.iloc[lo], times.iloc[hi-1], labels[lo:hi].sum()) != (first, last, fraud):
            raise ValueError("A chronological window does not match the verified development population.")

    frame = manual.normalize(raw_frame)
    categories = manual.fit_categories(frame.iloc[:STOPS[1]])
    values, features, _ = manual.encode(frame, categories)
    train = xgb.DMatrix(values[:STOPS[1]], label=labels[:STOPS[1]], feature_names=features, nthread=6)
    train_started = perf_counter()
    booster = xgb.train(PARAMETERS, train, num_boost_round=ROUNDS)
    fit_seconds = perf_counter() - train_started
    outputs = {}
    for index in (1, 2):
        name, lo, hi = NAMES[index], STOPS[index], STOPS[index + 1]
        first, last, fraud = EXPECTED[index]
        matrix = xgb.DMatrix(values[lo:hi], feature_names=features, nthread=6)
        probability = booster.predict(matrix)
        if index == 1:
            threshold = float(select_threshold(labels[lo:hi], probability))
        outputs[name] = {"rows": hi-lo, "fraud_rows": fraud, "first_second": first, "last_second": last,
                         "selected": metrics(labels[lo:hi], probability, threshold),
                         "at_0_5": metrics(labels[lo:hi], probability, .5)}

    output.mkdir(parents=True, exist_ok=False)
    model_file = output / "model.ubj"
    booster.save_model(model_file)
    report = {
        "schema": manual.SCHEMA, "model": manual.MODEL, "scope": "chronological development only",
        "inputs": list(manual.LABELS), "categories": categories, "threshold": threshold, "rounds": ROUNDS,
        "parameters": PARAMETERS, "model_sha256": digest(model_file), "model_bytes": model_file.stat().st_size,
        "fit": dict(zip(("rows", "fraud_rows", "first_second", "last_second"), (STOPS[1], 11988, 86400, 8745772), strict=True)),
        "raw_missing": raw_frame[list(manual.FIELDS)].isna().sum().to_dict(),
        "metrics": outputs, "fit_seconds": fit_seconds, "wall_seconds": perf_counter()-started,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    candidate = manual.AssessmentModel(output)
    candidate.assess(49.95, "visa", "debit", "google")
    reloaded = candidate.booster.predict(matrix)
    np.testing.assert_array_equal(reloaded, probability)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("raw", "v1", "v3", "output"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    run(**vars(parser.parse_args()))
