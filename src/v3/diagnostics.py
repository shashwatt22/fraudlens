"""Analyze saved V2 development predictions and the frozen XGBoost checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data_loader import TARGET, TIME
from src.evaluate import evaluate_scores
from src.features import HISTORICAL_COLUMNS, KEY_COLUMNS
from src.lab_benchmark import stratified_positions
from src.v2.advanced import entity_codes
from src.v2.challengers import SUBSET, TRAIN_LIMIT
from src.v2.data import sha256, walk_forward, write_json
from src.v2.final import _cache_matrices, _implementation_hashes, _score, policy_boundaries


MODELS = ("logistic_l2", "random_forest_depth", "xgb_advanced_depth8", "torch_mlp_regularized")
XGB = "xgb_advanced_depth8"
PROXIES = {
    "v1_entity": tuple(KEY_COLUMNS),
    "card_address": ("card1", "addr1"),
    "card_address_anchor": ("card1", "addr1", "round(day_minus_D1)"),
    "card_address_email": ("card1", "addr1", "P_emaildomain"),
}


def metrics_at_cutoff(labels, scores, cutoff):
    """The evaluator accepts one binary decision per row, not a cutoff scalar."""
    rows, _ = evaluate_scores(labels, scores, scores >= cutoff)
    return rows


def seen_groups(keys, fitted_positions, start, stop):
    """Compare complete validation keys to keys in the actual fitted rows."""
    val = keys[start:stop]
    fitted = keys[fitted_positions]
    known = val >= 0
    seen = known & np.isin(val, fitted[fitted >= 0])
    return {"seen_in_fit": seen, "unseen_in_fit": known & ~seen, "missing_key": ~known}


def annotated_metrics(labels, scores, cutoff):
    rows = len(labels)
    fraud_rows = int(labels.sum())
    return {"rows": rows, "fraud_rows": fraud_rows, "fraud_rate": fraud_rows / rows if rows else None,
            **(metrics_at_cutoff(labels, scores, cutoff) if rows else {})}


def development_inputs(root):
    """Accept only the development cache and reduce metadata to time and label."""
    features = root / "features.json"
    cache = json.loads(features.read_bytes())
    if "decision_freeze_sha256" in cache or cache.get("time_max", 0) >= cache["holdout_start"]:
        raise ValueError("Use the V2 development cache root, not the retrospective cache.")
    matrices = _cache_matrices(root)
    _, advanced, raw, rich = matrices
    if (advanced["features_sha256"] != sha256(features)
            or raw.shape != (cache["rows"], len(cache["features"]))
            or rich.shape != (cache["rows"], len(advanced["features"]))):
        raise ValueError("Development cache manifests and matrices are inconsistent.")
    metadata = pd.read_parquet(root / "development_metadata.parquet", columns=[TIME, TARGET])
    times = metadata[TIME].to_numpy(dtype=np.int64)
    labels = metadata[TARGET].to_numpy(dtype=np.uint8)
    if (len(times) != cache["rows"] or times.max() >= cache["holdout_start"]
            or times.max() != cache["time_max"] or not np.isin(labels, (0, 1)).all()):
        raise ValueError("Unexpected development metadata population.")
    folds = walk_forward(times)
    if [(f.train_end, f.validation_end) for f in folds] != [(f["train_end"], f["validation_end"]) for f in cache["folds"]]:
        raise ValueError("Saved folds differ from the timestamp-aligned development folds.")
    return matrices, times, labels, folds


def proxy_codes(matrices):
    _, advanced, _, matrix = matrices
    columns = {name: index for index, name in enumerate(advanced["features"])}

    def column(name):
        if name == "round(day_minus_D1)":
            return np.round(matrix[:, columns["day_minus_D1"]])
        return matrix[:, columns[name]]

    return {name: entity_codes(*(column(field) for field in fields)) for name, fields in PROXIES.items()}


def fit_population(name, train_end, times, labels, expected):
    sample_end = train_end
    if name == "torch_mlp_regularized":
        edge = times[int(.90 * train_end) - 1]
        sample_end = int(np.searchsorted(times[:train_end], edge, side="right"))
    if name in SUBSET:
        positions = stratified_positions(labels[:sample_end], TRAIN_LIMIT)
    else:
        positions = np.arange(train_end)
    if len(positions) != expected:
        raise ValueError(f"Reconstructed fit count does not match the checkpoint manifest for {name}.")
    return positions, train_end - sample_end


def summarize_folds(root, times, labels, folds, keys):
    results = {}
    for name in MODELS:
        directory = root / "challengers" / name
        results[name] = []
        for index, fold in enumerate(folds, 1):
            start, stop = fold.train_end, fold.validation_end
            record = json.loads((directory / f"fold{index}.json").read_bytes())
            scores = np.load(directory / f"scores{index}.npy", allow_pickle=False)
            if ((start, stop) != (record["train_end"], record["validation_end"])
                    or scores.shape != (stop - start,) or not np.isfinite(scores).all()
                    or record["candidate"]["name"] != name or not (directory / f"model{index}.joblib").is_file()):
                raise ValueError(f"Incomplete or misaligned fold {index} for {name}.")
            positions, early_stop_rows = fit_population(name, start, times, labels, record["fit_rows"])
            y = labels[start:stop]
            cutoff = record["threshold"]
            population = annotated_metrics(y, scores, cutoff)
            np.testing.assert_allclose(population["average_precision"], record["metrics"]["average_precision"], atol=1e-12)
            segments = {
                proxy: {group: annotated_metrics(y[mask], scores[mask], cutoff)
                        for group, mask in seen_groups(codes, positions, start, stop).items()}
                for proxy, codes in keys.items()
            }
            results[name].append({"fold": index, "train_prefix_rows": start, "fit_rows": len(positions),
                                  "fit_fraud_rows": int(labels[positions].sum()), "early_stopping_rows": early_stop_rows,
                                  "best_epoch": record.get("learning_history", {}).get("best_epoch"),
                                  "threshold": cutoff, "validation": population, "entities": segments})
    return results


def compare_xgb_policy(root, matrices, times, labels, fold, threads):
    freeze = json.loads((root / "decision_freeze.json").read_bytes())
    start = fold.validation_end
    if (freeze["implementation_sha256"] != _implementation_hashes() or freeze["train_end"] != fold.train_end
            or freeze["development_manifest_sha256"] != sha256(root / "features.json")
            or freeze["advanced_manifest_sha256"] != sha256(root / "advanced.json")):
        raise ValueError("The development scorer does not match the original freeze record.")
    artifact = freeze["models"][XGB]
    checkpoint = root / "challengers" / XGB / "model3.joblib"
    if checkpoint.relative_to(root).as_posix() != artifact["file"] or sha256(checkpoint) != artifact["sha256"]:
        raise ValueError("Unexpected fold-3 XGBoost checkpoint.")
    policy = freeze["policy"][XGB]
    if policy["calibrator"] != "identity":
        raise ValueError("The recorded XGBoost policy is expected to use its raw score.")
    first, second = policy_boundaries(times[start:])
    if [first, second - first, len(times) - start - second] != policy["windows"]:
        raise ValueError("Development policy boundaries differ from the frozen windows.")
    expected = np.load(checkpoint.parent / "scores3.npy", allow_pickle=False)
    with threadpool_limits(limits=threads):
        reloaded = _score(XGB, checkpoint, matrices, fold.train_end, start, fold.train_end, times, threads)
        np.testing.assert_allclose(reloaded, expected, atol=2e-6, rtol=1e-5)
        policy_scores = _score(XGB, checkpoint, matrices, start, len(times), fold.train_end, times, threads)
    validation = annotated_metrics(labels[fold.train_end:start], expected, .5)
    policy_labels = labels[start:]
    if policy_scores.shape != policy_labels.shape or not np.isfinite(policy_scores).all():
        raise ValueError("Expected finite scores covering exactly the development policy block.")
    cutoff = policy["threshold"]
    windows = {"whole_block": (0, len(policy_labels)), "calibrator_fit": (0, first),
               "threshold_selection": (first, second), "policy_evaluation": (second, len(policy_labels))}
    blocks = {name: annotated_metrics(policy_labels[lo:hi], policy_scores[lo:hi], cutoff)
              for name, (lo, hi) in windows.items()}
    np.testing.assert_allclose(blocks["policy_evaluation"]["average_precision"],
                               policy["policy_evaluation"]["average_precision"], atol=1e-12)
    return {"fold_3_validation": validation, "policy_blocks": blocks, "policy_cutoff": cutoff,
            "max_checkpoint_reload_error": float(np.max(np.abs(reloaded - expected)))}


def run(development, output, threads):
    development, output = development.resolve(), output.resolve()
    if output.is_relative_to(development) or "retrospective" in {p.lower() for p in development.parts}:
        raise ValueError("Keep diagnostic output separate from the V2 development inputs.")
    matrices, times, labels, folds = development_inputs(development)
    results = summarize_folds(development, times, labels, folds, proxy_codes(matrices))
    policy = compare_xgb_policy(development, matrices, times, labels, folds[-1], threads)
    report = {"scope": "V2 development predictions and frozen fold-3 XGBoost inference",
              "development_rows": len(labels), "proxy_definitions": PROXIES,
              "features": {"identity_indicator": "identity_missing", "inherited_history": HISTORICAL_COLUMNS,
                           "advanced_additions": matrices[1]["added_features"]},
              "models": results, "xgboost_policy_comparison": policy}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(f"Saved the development-only diagnostics to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    run(**vars(args))


if __name__ == "__main__":
    main()
