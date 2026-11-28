"""Train one optional-context manual model and compare it on the two V3 policy halves."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.train_manual_assessment import PARAMETERS, ROUNDS, STOPS, digest, metrics
from src.dashboard_v3 import manual_enhanced as enhanced
from src.dashboard_v3 import manual_model as manual
from src.evaluate import select_threshold

PLAN_SHA = "078f0f073c1fb27630f7bbd77a5e990d34ff44aecf447aef40eed07d1653d0c6"
CONTROL_SHA = "55ec2b3e9248cdb667dbbd6e59b752d595f7d340e795ffc59c20a0fad34632c2"
ID, TIME, TARGET = "TransactionID", "TransactionDT", "isFraud"
IDENTITY_ROWS = 120_464
POLICY = ("earlier_policy", "later_policy")
RULE = {"min_ap_gain": .01, "max_auc_loss": .005, "min_f1_gain": 0., "max_workload_gain": .02}


def comparison(later, live, candidate):
    delta = {key: candidate[key] - later[key] for key in ("average_precision", "roc_auc", "f1", "review_rate")}
    live_delta = {key: candidate[key] - live[key] for key in ("average_precision", "roc_auc", "f1", "review_rate")}
    checks = {
        "average_precision": delta["average_precision"] >= RULE["min_ap_gain"],
        "roc_auc": delta["roc_auc"] >= -RULE["max_auc_loss"],
        "f1": min(delta["f1"], live_delta["f1"]) >= RULE["min_f1_gain"],
        "review_rate": max(delta["review_rate"], live_delta["review_rate"]) <= RULE["max_workload_gain"],
    }
    return {"criteria": RULE, "passed": checks, "promote": all(checks.values()),
            "delta_vs_earlier_tuned_control": delta, "delta_vs_live_control": live_delta}


def development(raw, development_root, v3):
    plan = v3 / "step7a-refit" / "refit_plan.json"
    if digest(plan) != PLAN_SHA:
        raise ValueError("The V3 development plan hash differs.")
    plan = json.loads(plan.read_bytes())
    cols = [ID, TIME, TARGET]
    metadata_file = development_root / "development_metadata.parquet"
    if digest(metadata_file) != plan["inputs"][metadata_file.name]:
        raise ValueError("The ordered development metadata hash differs.")
    metadata = pd.read_parquet(metadata_file, columns=cols)
    if len(metadata) != STOPS[-1]:
        raise ValueError("Unexpected development population.")
    tx = pd.read_csv(raw / "train_transaction.csv", usecols=[*cols, manual.AMOUNT, *manual.FIELDS, "R_emaildomain"],
                     nrows=len(metadata), dtype={ID: "int32", TIME: "int32", TARGET: "uint8", manual.AMOUNT: "float32"})
    pd.testing.assert_frame_equal(tx[cols], metadata[cols], check_dtype=False)
    dev_identity = pd.read_csv(raw / "train_identity.csv", usecols=[ID, "DeviceType"],
                               nrows=IDENTITY_ROWS, dtype={ID: "int32"})
    if (len(dev_identity) != IDENTITY_ROWS or not dev_identity[ID].is_unique
            or not dev_identity[ID].is_monotonic_increasing or not dev_identity[ID].isin(tx[ID]).all()):
        raise ValueError("The expected identity prefix does not align to development transactions.")
    merged = tx.merge(dev_identity, how="left", on=ID, validate="one_to_one", sort=False)
    pd.testing.assert_frame_equal(merged[cols], metadata[cols], check_dtype=False)
    return merged, plan["partitions"]


def measure(y, scores, threshold):
    out = metrics(y, scores, threshold)
    out.update(threshold=threshold, rows=len(y), review_rate=out["flagged"] / len(y))
    return out


def match_control(manifest, data):
    expected = manifest["metrics"]["development_verification"]["selected"]
    for key, value in expected.items():
        if not np.allclose(value, data[key], rtol=0, atol=2e-12):
            raise ValueError(f"Existing control did not reproduce its verified metric: {key}")


def run(raw, development_root, v3, output):
    started = perf_counter()
    if output.exists():
        raise FileExistsError("Choose a new output directory.")
    control_dir = v3 / "manual-assessment"
    if digest(control_dir / "manifest.json") != CONTROL_SHA:
        raise ValueError("The existing manual control manifest hash differs.")
    control = manual.AssessmentModel(control_dir)
    if control.manifest["fit"]["rows"] != STOPS[1] or control.booster.num_boosted_rounds() != ROUNDS:
        raise ValueError("The control used a different fit configuration.")
    frame, partitions = development(raw, development_root, v3)
    labels = frame[TARGET].to_numpy()
    rich = enhanced.normalize(frame)
    categories = manual.fit_categories(rich.iloc[:STOPS[1]], enhanced.FIELDS)
    values, names, _ = manual.encode(rich, categories, enhanced.FIELDS)
    baseline_values, baseline_names, _ = manual.encode(rich, control.categories)
    matrix = xgb.DMatrix(values[:STOPS[1]], label=labels[:STOPS[1]], feature_names=names, nthread=6)
    fit_started = perf_counter()
    challenger = xgb.train(PARAMETERS, matrix, num_boost_round=ROUNDS)
    fit_seconds = perf_counter() - fit_started

    window_results = {}
    origin = partitions[POLICY[0]]["start"]
    baseline_matrix = xgb.DMatrix(baseline_values[origin:], feature_names=baseline_names, nthread=6)
    control_scores = control.booster.predict(baseline_matrix)
    match_control(control.manifest, measure(labels[origin:], control_scores, control.threshold))
    for scope in POLICY:
        part = partitions[scope]
        lo, hi = part["start"], part["end"]
        y = labels[lo:hi]
        if len(y) != part["rows"] or int(y.sum()) != part["fraud_rows"]:
            raise ValueError(f"Unexpected population for {scope}.")
        source = xgb.DMatrix(values[lo:hi], feature_names=names, nthread=6)
        baseline = control_scores[lo-origin:hi-origin]
        changed = challenger.predict(source)
        if scope == POLICY[0]:
            matched_threshold = float(select_threshold(y, baseline))
            threshold = float(select_threshold(y, changed))
        window_results[scope] = {
            "population": part,
            "original_control": measure(y, baseline, control.threshold),
            "matched_control": measure(y, baseline, matched_threshold),
            "challenger": measure(y, changed, threshold),
            "unknown_counts": {col: int(rich[col].iloc[lo:hi].eq(manual.UNKNOWN).sum()) for col in enhanced.EXTRA_FIELDS},
        }

    later = window_results[POLICY[-1]]
    decision = comparison(later["matched_control"], later["original_control"], later["challenger"])
    output.mkdir(parents=True, exist_ok=False)
    model_file = output / "model.ubj"
    challenger.save_model(model_file)
    restored = xgb.Booster(params={"nthread": 1})
    restored.load_model(model_file)
    np.testing.assert_array_equal(changed, restored.predict(source))

    report = {"schema": manual.SCHEMA, "model": enhanced.MODEL, "scope": "chronological development only",
              "inputs": list(enhanced.LABELS), "categories": categories, "threshold": threshold, "rounds": ROUNDS,
              "parameters": PARAMETERS, "fit": control.manifest["fit"], "windows": window_results,
              "decision": decision, "model_sha256": digest(model_file), "model_bytes": model_file.stat().st_size,
              "fit_seconds": fit_seconds, "wall_seconds": perf_counter() - started}
    (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("raw", "development_root", "v3", "output"):
        parser.add_argument(f"--{flag.replace('_', '-')}", required=True, type=Path)
    run(**vars(parser.parse_args()))
