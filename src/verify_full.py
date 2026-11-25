"""Audit saved full-run lineage, row alignment and reloaded model scores."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn import config_context
from threadpoolctl import threadpool_limits
import torch

from .behavior import anomaly_percentiles, anomaly_scores
from .dashboard.artifacts import FOLDERS, DashboardRun
from .dashboard.logic import demo_cases, outcome
from .data_loader import ID, TARGET, TIME
from .evaluate import positive_scores
from .lab_benchmark import EVAL_LIMIT, TRAIN_LIMIT, stratified_positions
from .neural import load_checkpoint, make_loader, score_batches
from .phase3 import Phase3Store
from .phase6 import NETWORKS, _dense, _predict
from .pipeline import FULL_ROWS, _hash, _inventory, _separation
from .risk_engine import composite_risk, require_alignment
from .scratch_checks import TorchReference
from .scratch_neural import ScratchNetwork
from .train import THREADS, _write_json


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _verify_lineage(run):
    run.require_complete()
    mode = run.mode_dir
    release = json.loads((mode / "holdout_release.json").read_bytes())
    freeze = mode / "protocol_freeze.json"
    _require(release["phase1_manifest_sha256"] == run.digests[1], "Phase 1 differs from the holdout release.")
    _require(release["protocol_freeze_sha256"] == _hash(freeze), "The recorded protocol freeze has changed.")
    _require(release["separation"] == _separation(run.root), "Development and holdout separation has changed.")
    quick = json.loads(freeze.read_bytes())["prior_quick_artifacts"]
    _require(quick == _inventory(run.root / "quick"), "A quick development artifact has changed.")
    pretest_hashes = release["frozen_artifacts"]
    folders = {"phase2": 2, "phase2b": "2B", "phase3": 3, "phase6": 6}
    checked_files = 0
    for phase, manifest in run.manifests.items():
        folder = mode / FOLDERS[phase]
        immutable = (run.path(1, "eda").parent if phase == 1 else folder / manifest["run_directory"])
        _require(_hash(immutable / "manifest.json") == _hash(folder / "manifest.json"),
                 f"Phase {phase} immutable manifest disagrees with its latest pointer.")
    for folder, phase in folders.items():
        path = mode / folder / run.manifests[phase]["run_directory"]
        for name, expected in pretest_hashes[folder].items():
            _require(_hash(path / name) == expected, f"Frozen {folder}/{name} has changed.")
            checked_files += 1
    _require(_hash(mode / "phase5" / "manifest.json") == pretest_hashes["phase5_manifest_sha256"],
             "The frozen Phase 5 run has changed.")
    return {"separation": release["separation"], "quick_files_preserved": len(quick),
            "frozen_phase_files_checked": checked_files, "phase_manifests_checked": len(run.manifests)}


def _verify_alignment(run):
    fields = [ID, TIME, TARGET]
    partitions = {}
    for name in ("train", "validation", "test"):
        frame = pd.read_parquet(run.path(1, name), columns=fields)
        partitions[name] = frame
        expected = run.partitions[name]
        _require(len(frame) == expected["rows"] and frame[TARGET].sum() == expected["fraud_rows"]
                 and frame[TIME].is_monotonic_increasing and frame[ID].is_unique
                 and (frame[TIME].iloc[0], frame[TIME].iloc[-1]) == (expected["first_second"], expected["last_second"]),
                 f"Phase 1 {name} does not match its manifest.")
        limit = TRAIN_LIMIT if name == "train" else EVAL_LIMIT
        sample = frame.iloc[stratified_positions(frame[TARGET].to_numpy(), limit)]
        require_alignment(sample, pd.read_parquet(run.path(6, f"{name}_sample"), columns=fields))
        require_alignment(frame, pd.read_parquet(run.path(3, f"{name}_anomaly"), columns=[ID]), fields=(ID,))
        if name != "train":
            for phase, key, reference in ((2, f"{name}_predictions", frame), ("2B", f"{name}_predictions", frame),
                                          (3, f"{name}_risk", frame), (6, f"{name}_predictions", sample)):
                require_alignment(reference, pd.read_parquet(run.path(phase, key), columns=fields))
    joined = pd.concat(partitions.values(), ignore_index=True)
    _require(len(joined) == FULL_ROWS and joined[ID].is_unique and joined[TIME].is_monotonic_increasing,
             "The combined Phase 1 partitions do not cover the expected unique source rows.")
    return {"aligned_partition_rows": {name: len(frame) for name, frame in partitions.items()},
            "phase6_samples_reconstructed": True}


def _predictions(path, ids):
    table = pd.read_parquet(path, filters=[(ID, "in", ids.tolist())]).set_index(ID)
    return table.loc[ids]


def _max_error(actual, expected, tolerance):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    _require(np.isfinite(error) and error <= tolerance, f"Reloaded score error {error} exceeds {tolerance}.")
    return error


def _verify_reload(run, threads):
    test_path = run.path(1, "test")
    all_ids = pd.read_parquet(test_path, columns=[ID])[ID].to_numpy()
    ids = all_ids[np.linspace(0, len(all_ids) - 1, 512, dtype=int)]
    frame = _predictions(test_path, ids).reset_index()
    saved = _predictions(run.path(2, "test_predictions"), ids)
    errors = {}
    for feature_set in ("basic", "behavioral"):
        prep = joblib.load(run.path(2, "preprocessors", feature_set))
        matrix = prep.transform(frame)
        for name, info in run.manifests[2]["models"].items():
            if info["feature_set"] == feature_set:
                model = joblib.load(run.path(2, "classifiers", name))
                errors[name] = _max_error(positive_scores(model, matrix), saved[f"{name}_score"], 1e-6)
    neural = run.manifests["2B"]
    model = load_checkpoint(run.path("2B", "checkpoint"), neural["architecture"]["input_dim"])
    batches = make_loader(matrix, frame[TARGET].to_numpy(), neural["training_config"]["batch_size"], training=False)
    loss = torch.nn.BCEWithLogitsLoss()
    _, scores = score_batches(model, batches, loss)
    neural_scores = _predictions(run.path("2B", "test_predictions"), ids)["mlp_score"]
    errors["mlp"] = _max_error(scores, neural_scores, 2e-6)

    anomaly_model = joblib.load(run.path(3, "anomaly_model"))
    anomaly_prep = joblib.load(run.path(3, "anomaly_preprocessor"))
    reference = np.load(run.path(3, "anomaly_reference"), allow_pickle=False)
    risks = _predictions(run.path(3, "test_risk"), ids)
    raw_anomaly = anomaly_scores(anomaly_model, anomaly_prep, frame)
    errors["anomaly"] = _max_error(raw_anomaly, risks["anomaly_score"], 1e-12)
    percentile = anomaly_percentiles(reference, raw_anomaly)
    errors["anomaly_percentile"] = _max_error(percentile, risks["anomaly_percentile"], 1e-12)
    errors["composite"] = _max_error(composite_risk(saved["xgboost_score"], percentile), risks["risk_score"], 1e-12)

    subset_ids = pd.read_parquet(run.path(6, "test_sample"), columns=[ID])[ID].to_numpy()[::8]
    subset = _predictions(test_path, subset_ids).reset_index()
    phase6 = run.manifests[6]
    sparse = joblib.load(run.path(6, "preprocessor")).transform(subset)
    dense, _ = _dense(sparse)
    subset_scores = _predictions(run.path(6, "test_predictions"), subset_ids)
    lab_errors = {}
    with config_context(working_memory=64):
        for name, info in phase6["models"].items():
            checkpoint = run.path(6, "preprocessor").parent / info["file"]
            if name in NETWORKS:
                model = ScratchNetwork.load(checkpoint)
                if name == "torch_relu32_reference":
                    model = TorchReference(model)
            else:
                model = joblib.load(checkpoint)
            scores, _ = _predict(name, model, sparse, dense, info["threshold"])
            lab_errors[name] = _max_error(scores, subset_scores[f"{name}_score"], 2e-6)
    return {"primary_rows": len(ids), "primary_max_score_errors": errors,
            "matched_rows": len(subset_ids), "matched_max_score_errors": lab_errors, "threads": threads}


def _verify_demo(run, threads):
    store = Phase3Store(run.root, "full", threads)
    table = store.read_risk("test")
    cases = demo_cases(table)
    _require(len(cases) == 6, "The final evaluation is missing a requested demonstration case.")
    verified = {}
    for label, transaction_id in cases.items():
        assessment = store.assess("test", transaction_id)
        row = table.loc[table[ID].eq(transaction_id)].iloc[0]
        verified[label] = {**assessment, "observed_label": int(row[TARGET]),
                           "outcome": outcome(row[TARGET], assessment["classifier_decision"])}
    path = run.mode_dir / "demo_cases.json"
    _write_json(path, {"mode": "full", "partition": "test", "phase3_manifest_sha256": run.digests[3], "cases": verified})
    checks = {key: max(value["shap"]["checks"][key] for value in verified.values())
              for key in next(iter(verified.values()))["shap"]["checks"]}
    return {"cases_checked": len(verified), "maximum_local_errors": checks,
            "global_rows": run.manifests[3]["shap"]["global_rows"], "global": run.manifests[3]["shap"]["checks"]}


def run_verification(root: Path, threads: int = THREADS) -> Path:
    started = perf_counter()
    run = DashboardRun(root.resolve(), "full")
    torch.set_num_threads(threads)
    checks = {}
    with threadpool_limits(limits=threads):
        for name, check, arguments in (("lineage", _verify_lineage, (run,)), ("alignment", _verify_alignment, (run,)),
                                       ("saved_model_reload", _verify_reload, (run, threads)),
                                       ("shap_and_demo", _verify_demo, (run, threads))):
            checks[name] = check(*arguments)
            print(f"Verified {name.replace('_', ' ')}.", flush=True)
    path = run.mode_dir / "verification.json"
    _write_json(path, {"mode": "full", **checks, "runtime_seconds": perf_counter() - started})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--threads", type=int, default=THREADS)
    args = parser.parse_args()
    print(run_verification(args.artifact_root, args.threads))


if __name__ == "__main__":
    main()
