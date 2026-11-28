"""Score the approved V3 checkpoints before opening the reused holdout labels."""

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch

from src.data_loader import AMOUNT, ID, PRODUCT, TARGET, TIME
from src.evaluate import evaluate_scores
from src.features import HISTORICAL_COLUMNS, KEY_COLUMNS, add_behavioral_features
from src.training_data import read_phase1_manifest
from src.v2.advanced import build_advanced
from src.v2.data import CHUNK_ROWS, build_cache, sha256, write_json
from src.v2.experiment import _peak_memory, review_budgets
from .behavior import anchor_keys
from .decision_policy import MODELS, PRIMARY, freeze_file, losses, operating_metrics, read, reliability
from .diagnostics import annotated_metrics, development_inputs, seen_groups
from .final_refit_report import scorer


SCOPE = "reused retrospective holdout"
ROWS = 118_108
THREADS = 6
STEP7B_DECISION = "138b12eac5773e4bbd79eec064f4dc3d5cafc9386cda2742673f4b4463579edf"
STEP7B_MODELS = "a11e926442fabcf6f1166e61b4600b7d191d2d2b5dc3b495b875133336c5b988"


@contextmanager
def inference_memory():
    """Sample Windows process working set during one model's inference."""
    reading = {"sampled_working_set_peak_bytes": None}
    if os.name != "nt":
        yield reading
        return
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    handle = ctypes.c_void_p(kernel.GetCurrentProcess())
    finished = Event()

    def sample():
        counters = (ctypes.c_size_t * 10)()
        counters[0] = ctypes.sizeof(counters)
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), ctypes.sizeof(counters)):
            reading["sampled_working_set_peak_bytes"] = max(reading["sampled_working_set_peak_bytes"] or 0, int(counters[2]))

    def monitor():
        while not finished.wait(.025):
            sample()

    sample()
    worker = Thread(target=monitor, daemon=True)
    worker.start()
    try:
        yield reading
    finally:
        finished.set()
        worker.join()
        sample()


def frozen_inputs(refit, policy):
    parts = {name: policy / name for name in ("decision_freeze.json", "model_freeze.json", "policy_metrics.json")}
    hashes = read(policy / "artifact_hashes.json")
    result = {name: sha256(file) for name, file in parts.items()}
    if (result["decision_freeze.json"] != STEP7B_DECISION or result["model_freeze.json"] != STEP7B_MODELS
            or any(hashes.get(name) != digest for name, digest in result.items())):
        raise ValueError("Step 7B frozen manifest or metrics hash mismatch.")
    decision, models = (read(parts[key]) for key in ("decision_freeze.json", "model_freeze.json"))
    policy_metrics = read(parts["policy_metrics.json"])
    if (decision["model_freeze_sha256"] != STEP7B_MODELS or policy_metrics["decision_sha256"] != STEP7B_DECISION
            or policy_metrics["model_sha256"] != STEP7B_MODELS or decision["primary"] != PRIMARY or models["primary"] != PRIMARY
            or set(models["models"]) != set(MODELS) or set(decision["thresholds"]) != set(MODELS)
            or decision["calibrator"] != {"method": "identity"}
            or not np.isclose(decision["thresholds"][PRIMARY]["selected_f1"], .1938110143, atol=5e-11, rtol=0)):
        raise ValueError("The recorded primary classifier, calibration or threshold changed.")
    result["refit_plan.json"] = sha256(refit / "refit_plan.json")
    if result["refit_plan.json"] != models["step7a_plan_sha256"]:
        raise ValueError("Step 7A refit plan mismatch.")
    for name, row in models["models"].items():
        checks = {**row["artifact_sha256"], row["score_file"]: row["score_sha256"],
                  f"{name}/fit.json": row["fit_manifest_sha256"]}
        for file, expected in checks.items():
            digest = sha256(refit / file)
            if digest != expected:
                raise ValueError(f"Checkpoint or policy score changed: {file}")
            result[file] = digest
    return decision, models, result


def validate_alignment(reference, observed):
    if not observed[[ID, TIME]].reset_index(drop=True).equals(reference[[ID, TIME]].reset_index(drop=True)):
        raise ValueError("Holdout transaction IDs or timestamps do not match in source order.")


def test_path(v1, development):
    manifest, digest = read_phase1_manifest(v1)
    cache = read(development / "features.json")
    if digest != cache["v1_manifest_sha256"] or manifest["partitions"]["test"]["rows"] != ROWS:
        raise ValueError("The original V1 holdout differs from the development lineage.")
    return manifest, v1 / manifest["files"]["test"]


def verify_prior_history(v1, manifest, cache, matrix):
    columns = [ID, TIME, AMOUNT, PRODUCT, *KEY_COLUMNS]
    timeline = pd.concat([pd.read_parquet(v1 / manifest["files"][name], columns=columns)
                          for name in ("train", "validation", "test")], ignore_index=True)
    rebuilt = add_behavioral_features(timeline)
    for name in HISTORICAL_COLUMNS:
        expected = matrix[:, cache["features"].index(name)]
        np.testing.assert_array_equal(rebuilt[name].to_numpy(dtype=np.float32, na_value=np.nan), expected,
                                      err_msg=f"Archived causal feature mismatch: {name}")


def score_holdout(raw, v1, development, refit, policy, out):
    if (out / "label_access.json").exists() or (out / "score_manifest.json").exists():
        raise FileExistsError("Use the existing scored run or a separate new output directory.")
    decision, models, before = frozen_inputs(refit, policy)
    start_time = perf_counter()
    source = development / "development_metadata.parquet"
    if sha256(source) != models["development_metadata_sha256"]:
        raise ValueError("Development metadata hash changed.")
    plan = read(refit / "refit_plan.json")
    if any(sha256(development / file) != digest for file, digest in plan["inputs"].items()):
        raise ValueError("Refit development input files changed.")
    dev, times, _, _ = development_inputs(development)
    manifest, path = test_path(v1, development)
    metadata = pd.read_parquet(path, columns=[ID, TIME])
    start, stop = len(times), len(times) + len(metadata)
    expected_range = manifest["partitions"]["test"]
    if (len(metadata) != ROWS or metadata[ID].duplicated().any() or not metadata[TIME].is_monotonic_increasing
            or metadata[TIME].iloc[0] != expected_range["first_second"] or metadata[TIME].iloc[-1] != expected_range["last_second"]
            or times[-1] >= metadata[TIME].iloc[0]):
        raise ValueError("Unexpected holdout boundaries.")
    features = out / "covariates"
    cache = build_cache(raw, v1, features, {"decision_freeze_sha256": STEP7B_DECISION}, retrospective_metadata=metadata)
    if any(cache[k] != dev[0][k] for k in ("features", "categorical", "raw_sha256")):
        raise ValueError("The retrospective covariate source differs from development.")
    matrix = np.load(features / "features.npy", mmap_mode="r", allow_pickle=False)
    timeline = pd.read_parquet(features / "development_metadata.parquet", columns=[ID, TIME])
    validate_alignment(metadata, timeline.iloc[start:])
    verify_prior_history(v1, manifest, cache, matrix)
    advanced = build_advanced(features, matrix, cache, timeline, raw)
    rich = np.load(features / "features_advanced.npy", mmap_mode="r", allow_pickle=False)
    if rich.shape != (stop, 497) or advanced["features"] != dev[1]["features"]:
        raise ValueError("The original V2 feature set changed.")
    for earlier, extended in zip(dev[2:], (matrix, rich), strict=True):
        for lo in range(0, start, CHUNK_ROWS):
            hi = min(lo+CHUNK_ROWS, start)
            if not np.array_equal(earlier[lo:hi], extended[lo:hi], equal_nan=True):
                raise ValueError("Appending holdout covariates altered a development feature.")
    del dev, matrix
    features_seconds = perf_counter() - start_time
    write_json(out / "feature_audit.json", {"unchanged_development_rows": start, "holdout_rows": ROWS, "columns": 497,
                                           "v1_history_recomputed_and_matched": HISTORICAL_COLUMNS,
                                           "advanced_columns": advanced["added_features"], "feature_seconds": features_seconds})
    print(f"Causal features verified through {stop:,} rows. Scoring four frozen checkpoints.", flush=True)

    torch.set_num_threads(THREADS)
    predictions = metadata.copy()
    timings = {}
    with threadpool_limits(limits=THREADS):
        for name in MODELS:
            model = models["models"][name]
            location = refit / model["model_file"]
            tic = perf_counter()
            predict = scorer(name, location.parent, location.name, rich, advanced["features"], cache["categorical"], model["fit_rows"])
            with inference_memory() as memory:
                prepared = perf_counter()
                values = predict(start, stop)
                infer_seconds = perf_counter() - prepared
            if values.shape != (ROWS,) or not np.all((values >= 0) & (values <= 1)):
                raise ValueError(f"Invalid holdout scores for {name}.")
            predictions[f"{name}_score"] = values
            predictions[f"{name}_decision"] = values.astype(np.float64) >= decision["thresholds"][name]["selected_f1"]
            timings[name] = {"inference_seconds": infer_seconds, "load_and_prepare_seconds": prepared-tic, **memory}
            del predict
            print(f"{name}: {infer_seconds:.2f} s inference.", flush=True)
    after = frozen_inputs(refit, policy)[2]
    if after != before:
        raise ValueError("Frozen files changed during scoring.")
    temporary = out / "predictions.tmp.parquet"
    temporary_predictions = out / "predictions.parquet"
    predictions.to_parquet(temporary, index=False)
    temporary.replace(temporary_predictions)
    freeze_file(out / "score_manifest.json", {"scope": SCOPE, "rows": ROWS, "features_seconds": features_seconds,
                "timings": timings, "score_stage_peak_working_set_bytes": _peak_memory(), "frozen_hashes": before,
                "predictions_sha256": sha256(temporary_predictions), "feature_audit_sha256": sha256(out / "feature_audit.json"),
                "v1_manifest_sha256": cache["v1_manifest_sha256"], "feature_manifest_sha256": sha256(features / "advanced.json"),
                "matrix_sha256": sha256(features / "features_advanced.npy"), "evaluation_code_sha256": sha256(Path(__file__))})


def read_archives(v1, v2, truth):
    phase = read(v1 / "phase2" / "manifest.json")
    folder = v1 / "phase2" / phase["run_directory"]
    files = phase["files"]
    v1_predictions = folder / files["test_predictions"]
    v1_metrics = folder / files["test_metrics"]
    v1_selection = folder / files["selection"]
    v2_predictions = v2 / "retrospective" / "champion_scores.parquet"
    v2_metrics = v2 / "retrospective_results.json"
    old, newer = (pd.read_parquet(path) for path in (v1_predictions, v2_predictions))
    validate_alignment(truth, old)
    validate_alignment(truth, newer)
    np.testing.assert_array_equal(old[TARGET], truth[TARGET])
    v2_record = read(v2_metrics)
    if (v2_record["scope"] != SCOPE or v2_record["rows"] != len(truth) or v2_record["fraud_rows"] != int(truth[TARGET].sum())
            or v2_record["champion"] != "ensemble_weighted"):
        raise ValueError("The archived V2 population or scorer differs.")
    v2_champion = v2_record["models"]["ensemble_weighted"]
    np.testing.assert_array_equal(newer["raw_score"], newer["policy_score"])
    return {
        "v1_xgboost": (old["xgboost_score"].to_numpy(), old["xgboost_decision"].to_numpy(),
                       read(v1_selection)["thresholds"]["xgboost"], read(v1_metrics)["models"]["xgboost"]),
        "v2_ensemble": (newer["raw_score"].to_numpy(), newer["decision"].to_numpy(), v2_champion["threshold"], v2_champion["metrics"]),
    }, {f"archive_{file.name}": sha256(file) for file in (v1_predictions, v1_metrics, v1_selection, v2_predictions, v2_metrics)}


def model_metrics(y, scores, cutoff):
    if not np.all((scores >= 0) & (scores <= 1)):
        raise ValueError("Expected finite probabilities or original scores in [0, 1].")
    ranking, curves = evaluate_scores(y, scores, scores >= .5)
    brier, log_loss = losses(y, scores).mean(axis=1)
    return {"average_precision": ranking["average_precision"], "roc_auc": ranking["roc_auc"],
            "brier": float(brier), "log_loss": float(log_loss), "review_budgets": review_budgets(y, scores),
            "at_0_5": operating_metrics(y, scores, scores, .5), "frozen": operating_metrics(y, scores, scores, cutoff)}, {
                **curves, "reliability": reliability(y, scores)}


def label_gate(out, score_manifest):
    if not score_manifest.is_file():
        raise ValueError("A verified score manifest must exist before opening labels.")
    with (out / "label_access.json").open("x", encoding="utf-8", newline="\n") as receipt:
        json.dump({"opened_at_utc": datetime.now(timezone.utc).isoformat(), "scope": SCOPE,
                   "score_manifest_sha256": sha256(score_manifest), "decision_sha256": STEP7B_DECISION}, receipt, indent=2)


def evaluate(v1, development, refit, policy, out):
    decision, models, before = frozen_inputs(refit, policy)
    scored = out / "score_manifest.json"
    score_manifest = read(scored)
    matrix_path = out / "covariates" / "features_advanced.npy"
    if (score_manifest["rows"] != ROWS or score_manifest["frozen_hashes"] != before
            or score_manifest["evaluation_code_sha256"] != sha256(Path(__file__))
            or score_manifest["predictions_sha256"] != sha256(out / "predictions.parquet")
            or score_manifest["matrix_sha256"] != sha256(matrix_path)
            or score_manifest["feature_manifest_sha256"] != sha256(out / "covariates" / "advanced.json")
            or score_manifest["feature_audit_sha256"] != sha256(out / "feature_audit.json")):
        raise ValueError("The saved score or feature artifacts differ from the pre-label manifest.")
    manifest, path = test_path(v1, development)
    predictions = pd.read_parquet(out / "predictions.parquet")
    validate_alignment(pd.read_parquet(path, columns=[ID, TIME]), predictions)
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    names = read(out / "covariates" / "advanced.json")["features"]
    keys = anchor_keys(matrix, names)
    start = len(keys)-ROWS
    if any(not np.all((predictions[f"{name}_score"] >= 0) & (predictions[f"{name}_score"] <= 1)) for name in MODELS):
        raise ValueError("Invalid saved scores before opening labels.")
    label_gate(out, scored)
    truth = pd.read_parquet(path, columns=[ID, TIME, TARGET])
    validate_alignment(truth, predictions)
    y = truth[TARGET].to_numpy(dtype=np.uint8)
    if not np.isin(y, (0, 1)).all() or y.sum() != 4064:
        raise ValueError("Unexpected holdout fraud population.")
    archives, archive_hashes = read_archives(v1, development, truth)
    results, curves = {}, {}
    for name in MODELS:
        score = predictions[f"{name}_score"].to_numpy(dtype=np.float64)
        cutoff = decision["thresholds"][name]["selected_f1"]
        np.testing.assert_array_equal(score >= cutoff, predictions[f"{name}_decision"])
        results[name], curves[name] = model_metrics(y, score, cutoff)
        fit = models["models"][name]["fit_rows"]
        groups = seen_groups(keys, np.arange(fit), start, len(keys))
        results[name]["entities"] = {group: {"at_0_5": annotated_metrics(y[mask], score[mask], .5),
                                             "frozen": annotated_metrics(y[mask], score[mask], cutoff)}
                                     for group, mask in groups.items()}
        results[name]["fit_rows"] = fit
    for name, (score, recorded, cutoff, metrics) in archives.items():
        score = score.astype(np.float64)
        np.testing.assert_array_equal(score >= cutoff, recorded)
        results[name], curves[name] = model_metrics(y, score, cutoff)
        for key in ("average_precision", "roc_auc"):
            np.testing.assert_allclose(results[name][key], metrics[key], rtol=0, atol=1e-12)
        if results[name]["frozen"]["confusion_matrix"] != metrics["confusion_matrix"]:
            raise ValueError(f"Archived confusion matrix changed for {name}.")
    if frozen_inputs(refit, policy)[2] != before:
        raise ValueError("Frozen checkpoint or policy changed during evaluation.")
    write_json(out / "curves.json", curves)
    write_json(out / "results.json", {"schema_version": 1, "scope": SCOPE, "primary": PRIMARY, "calibration": "identity",
                "population": {"rows": ROWS, "fraud_rows": int(y.sum()), "first_second": int(truth[TIME].iloc[0]),
                               "last_second": int(truth[TIME].iloc[-1]), "previously_evaluated_in": ["V1", "V2"]},
                "ranking_scores": "original", "probability_scores": "original with identity calibration for V3 XGBoost",
                "models": results, "resources": {key: score_manifest[key] for key in
                                                   ("features_seconds", "timings", "score_stage_peak_working_set_bytes")},
                "lineage": {"score_manifest_sha256": sha256(scored), "curves_sha256": sha256(out / "curves.json"),
                            "label_access_sha256": sha256(out / "label_access.json"), "frozen_hashes": before, "archives": archive_hashes,
                            "v1_manifest_sha256": score_manifest["v1_manifest_sha256"], "holdout_partition": manifest["files"]["test"]}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("score", "evaluate"))
    for name in ("raw", "v1", "development", "refit", "policy", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    inputs = (args.raw, args.v1, args.development, args.refit, args.policy)
    if any(out.is_relative_to(path.resolve()) for path in inputs):
        raise ValueError("Use a separate output directory.")
    if args.stage == "score":
        out.mkdir(parents=True, exist_ok=True)
        score_holdout(*inputs, out)
    else:
        evaluate(*inputs[1:], out)
    print(f"{args.stage} completed in {out}.")


if __name__ == "__main__":
    main()
