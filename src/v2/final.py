"""Freeze a development-only policy, then evaluate the reused holdout once."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from threadpoolctl import threadpool_limits
import xgboost as xgb

from src.data_loader import ID, TARGET, TIME
from src.evaluate import evaluate_scores, positive_scores, select_threshold
from src.training_data import read_phase1_manifest

from .advanced import build_advanced
from .challengers import BOOSTING, SUBSET, _encode_frequency
from .data import build_cache, sha256, write_json
from .experiment import CHUNK_ROWS, MatrixView, TRIALS, review_budgets
from .validation_report import can_promote, records, summarize

REFERENCES = {"rich", "v1", "history", "unweighted", "basic_native", "no_vesta", "no_anonymous", "no_identity"}
REQUIRED = {*REFERENCES, *BOOSTING, *SUBSET}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _implementation_hashes():
    package = Path(__file__).resolve().parent
    return {name: sha256(package / name) for name in ("data.py", "advanced.py", "experiment.py", "challengers.py", "final.py")}


def policy_boundaries(times):
    boundaries = [int(np.searchsorted(times, times[int(len(times) * q) - 1], side="right")) for q in (.25, .50)]
    if not 0 < boundaries[0] < boundaries[1] < len(times):
        raise ValueError("Calibration and policy selection require distinct chronological blocks.")
    return boundaries


def _calibrate(scores, labels, times, *, margin=False, dummy=False):
    first, second = policy_boundaries(times)
    base = scores if margin else logit(np.clip(scores.astype(np.float64), 1e-7, 1-1e-7))
    alternatives = {"identity": None}
    if not dummy:
        sigmoid = LogisticRegression(random_state=42).fit(base[:first, None], labels[:first])
        if sigmoid.coef_.item() > 0:
            alternatives["sigmoid"] = sigmoid
        alternatives["isotonic"] = IsotonicRegression(out_of_bounds="clip").fit(scores[:first], labels[:first])
    transformed = {name: _apply(name, model, scores, margin) for name, model in alternatives.items()}
    selection = {key: float(brier_score_loss(labels[first:second], value[first:second])) for key, value in transformed.items()}
    chosen = min(selection, key=selection.get)
    adjusted = transformed[chosen]
    threshold = .5 if dummy else select_threshold(labels[first:second], adjusted[first:second])
    decision = adjusted[second:] >= threshold
    metrics, _ = evaluate_scores(labels[second:], adjusted[second:], decision)
    calibration_check = {name: {"brier": float(brier_score_loss(labels[second:], values[second:])),
                                "reliability": _reliability(labels[second:], values[second:])}
                         for name, values in transformed.items()}
    return ((chosen, alternatives[chosen], margin), {"calibrator": chosen, "threshold": threshold,
             "windows": [first, second-first, len(times)-second], "selection_brier": selection,
             "policy_evaluation": metrics, "calibration_evaluation": calibration_check,
             "alternatives": _operating_points(labels[first:second], adjusted[first:second]),
             "review_budgets": review_budgets(labels[second:], adjusted[second:])})


def _apply(name, model, scores, margin):
    if name == "identity":
        return expit(scores) if margin else scores
    if name == "sigmoid":
        values = scores if margin else logit(np.clip(scores.astype(np.float64), 1e-7, 1-1e-7))
        return model.predict_proba(values[:, None])[:, 1]
    return model.predict(scores)


def _reliability(labels, scores):
    bins = np.minimum(9, (scores * 10).astype(int))
    return [{"rows": int(mask.sum()), "mean_score": float(scores[mask].mean()), "observed_fraud": float(labels[mask].mean())}
            for i in range(10) if (mask := bins == i).any()]


def _operating_points(labels, scores):
    precision, recall, cutoffs = precision_recall_curve(labels, scores)
    result = {}
    for name, constrained, target, objective in (("high_precision", precision[:-1], .50, recall[:-1]),
                                                ("high_recall", recall[:-1], .70, precision[:-1]),
                                                ("precision_at_least_85", precision[:-1], .85, recall[:-1]),
                                                ("recall_at_least_85", recall[:-1], .85, precision[:-1])):
        eligible = np.flatnonzero(constrained >= target)
        if len(eligible):
            index = eligible[np.argmax(objective[eligible])]
            result[name] = {"threshold": float(cutoffs[index]), "precision": float(precision[index]),
                            "recall": float(recall[index]), "reviewed": int((scores >= cutoffs[index]).sum())}
        else:
            result[name] = None
    return result


def _segment_metrics(labels, scores, threshold, had_history):
    scopes = {"had_prior_proxy_history": np.flatnonzero(had_history),
              "cold_start": np.flatnonzero(~had_history)}
    scopes.update({f"chronological_third_{i}": rows for i, rows in enumerate(np.array_split(np.arange(len(scores)), 3), 1)})
    segments = {}
    for name, positions in scopes.items():
        if len(positions):
            y, s = labels[positions], scores[positions]
            metrics, _ = evaluate_scores(y, s, s >= threshold)
            segments[name] = {"rows": len(y), "fraud_rows": int(y.sum()), "metrics": metrics}
    return segments


def _finalists(found):
    retained = {"rich", "history", "unweighted", "xgb_advanced", "xgb_advanced_unweighted"}
    categories = {}
    for name in REQUIRED - REFERENCES:
        details = (BOOSTING | SUBSET)[name]
        family = details.family if details.family != "numpy" else f"numpy_{bool(details.options['hidden'])}"
        ap = found[name][1]["metrics"]["average_precision"]["mean"]
        if family not in categories or (ap, name) > categories[family]:
            categories[family] = (ap, name)
    retained.update(name for _, name in categories.values())
    return sorted(retained)


def _cache_matrices(root):
    manifest = json.loads((root / "features.json").read_bytes())
    advanced = json.loads((root / "advanced.json").read_bytes())
    return (manifest, advanced, np.load(root / "features.npy", mmap_mode="r"),
            np.load(root / "features_advanced.npy", mmap_mode="r"))


def _views(name, matrices, train_end, times):
    cache, advanced, data, rich = matrices
    pool = data if name in TRIALS else rich
    all_features = cache["features"] if name in TRIALS else advanced["features"]
    names = cache["groups"][TRIALS[name].features] if name in TRIALS else all_features
    details = (BOOSTING | SUBSET).get(name)
    if details and details.family == "lgbm":
        train_end = int(np.searchsorted(times[:train_end], times[int(.90 * train_end) - 1], side="right"))
    return MatrixView(pool, {name: all_features.index(name) for name in names}, cache["categorical"]).fit(train_end)


def _score(name, model_path, matrices, start, stop, train_end, times, threads):
    view = _views(name, matrices, train_end, times)
    if name in TRIALS:
        model = xgb.Booster(params={"nthread": threads})
        model.load_model(model_path)
        family = "xgb"
        payload = {}
    else:
        family = (BOOSTING | SUBSET)[name].family
        payload = joblib.load(model_path)
        model = payload["model"]
    outputs = []
    batch = 2048 if name in SUBSET else CHUNK_ROWS
    for low in range(start, stop, batch):
        high = min(low+batch, stop)
        if family == "xgb":
            value = model.inplace_predict(view.read(low, high), validate_features=False)
        elif family == "lgbm":
            value = model.predict(view.read(low, high), num_threads=threads)
        elif family == "catboost":
            matrix = _encode_frequency(view.data[low:high, view.columns], payload["frequency_maps"])
            value = positive_scores(model, matrix)
        elif family == "dummy":
            value = positive_scores(model, np.zeros((high-low, 1)))
        else:
            matrix = payload["transform"].transform(view.data[low:high, view.columns])
            if family == "torch":
                import torch
                torch.set_num_threads(threads)
                with torch.inference_mode():
                    value = torch.sigmoid(model(torch.from_numpy(matrix)).squeeze(1)).numpy()
            elif family == "numpy":
                value = model.predict_scores(matrix)
            elif family == "svm":
                value = model.decision_function(matrix)
            else:
                value = positive_scores(model, matrix)
        outputs.append(value)
    return np.concatenate(outputs)


def prepare(root, threads):
    frozen = root / "decision_freeze.json"
    if frozen.exists():
        raise FileExistsError("The final V2 policy already exists. Use the recorded freeze.")
    found = records(root)
    pending = REQUIRED - set(found)
    patterns_file = root / "patterns.json"
    if pending or not patterns_file.is_file() or len(json.loads(patterns_file.read_bytes())["folds"]) != 3:
        raise ValueError(f"Finish development first. Pending trials: {', '.join(sorted(pending)) or 'none'}. "
                         f"Patterns report exists: {patterns_file.is_file()}.")
    summarize(root)
    selection = json.loads((root / "development_selection.json").read_bytes())
    champion = selection["champion_individual"]
    names = set(_finalists(found)) | {champion}
    blends = {}
    best_fold = found[champion][1]["folds"][2]["metrics"]["average_precision"]
    for blend in selection["blends"]:
        name = f"ensemble_{blend['name']}"
        blends[name] = blend
        names.update(blend["members"])
    matrices = _cache_matrices(root)
    frame = pd.read_parquet(root / "development_metadata.parquet")
    time = frame[TIME].to_numpy(dtype=np.int64)
    last = matrices[0]["folds"][-1]
    start, stop, train_end = last["validation_end"], len(frame), last["train_end"]
    policy_y = frame[TARGET].iloc[start:].to_numpy(dtype=np.uint8)
    models, reports, calibrators, raw = {}, {}, {}, {}
    for name in sorted(names):
        path = found[name][0].parent
        model = path / ("model3.ubj" if name in TRIALS else "model3.joblib")
        models[name] = {"file": model.relative_to(root).as_posix(), "sha256": sha256(model)}
        expected = np.load(path / "scores3.npy", allow_pickle=False)
        reloaded = _score(name, model, matrices, train_end, start, train_end, time, threads)
        np.testing.assert_allclose(reloaded, expected, atol=2e-6, rtol=1e-5,
                                   err_msg=f"Saved model score mismatch for {name}")
        models[name]["validation_reload_max_error"] = float(np.max(np.abs(reloaded - expected)))
        raw[name] = _score(name, model, matrices, start, stop, train_end, time, threads)
        result = _calibrate(raw[name], policy_y, time[start:], margin=name.startswith("svm"), dummy=name == "dummy")
        calibrators[name], reports[name] = result
        print(f"Frozen policy inputs for {name}: {reports[name]['calibrator']}", flush=True)
    for name, blend in blends.items():
        score = sum(raw[member] * weight for member, weight in blend["members"].items())
        calibrators[name], reports[name] = _calibrate(score, policy_y, time[start:])
    acceptable = [name for name, blend in blends.items()
                  if blend["folds"][2]["metrics"]["average_precision"] > best_fold
                  and can_promote(blend, found[champion][1])
                  and reports[name]["policy_evaluation"]["average_precision"] > reports[champion]["policy_evaluation"]["average_precision"]]
    chosen = max(acceptable, key=lambda key: reports[key]["policy_evaluation"]["average_precision"]) if acceptable else champion
    joblib.dump(calibrators, root / "calibrators.joblib", compress=1)
    write_json(frozen, {"frozen_at_utc": _now(), "population": "reused retrospective holdout", "champion": chosen,
                        "individual": champion, "blends": {name: value["members"] for name, value in blends.items()},
                        "models": models, "policy": reports, "train_end": train_end,
                        "calibrators_sha256": sha256(root / "calibrators.joblib"),
                        "development_selection_sha256": sha256(root / "development_selection.json"),
                        "development_manifest_sha256": sha256(root / "features.json"),
                        "advanced_manifest_sha256": sha256(root / "advanced.json"),
                        "implementation_sha256": _implementation_hashes(),
                        "retrospective_labels_used": False})


def evaluate(root, raw, v1, threads):
    freeze = root / "decision_freeze.json"
    receipt = root / "retrospective_access.json"
    if not freeze.is_file() or receipt.exists():
        raise ValueError("Retrospective evaluation requires an unopened, frozen development decision.")
    policy = json.loads(freeze.read_bytes())
    if policy["implementation_sha256"] != _implementation_hashes():
        raise ValueError("Frozen scoring or preprocessing implementation changed.")
    for name, key in (("calibrators.joblib", "calibrators_sha256"), ("features.json", "development_manifest_sha256"),
                      ("advanced.json", "advanced_manifest_sha256"), ("development_selection.json", "development_selection_sha256")):
        if sha256(root / name) != policy[key]:
            raise ValueError(f"Frozen input changed: {name}")
    for name, model in policy["models"].items():
        if sha256(root / model["file"]) != model["sha256"]:
            raise ValueError(f"Frozen model changed: {name}")
    manifest, digest = read_phase1_manifest(v1)
    reference = json.loads((root / "features.json").read_bytes())
    if digest != reference["v1_manifest_sha256"]:
        raise ValueError("V1 snapshot differs from the frozen development run.")
    input_file = v1 / manifest["files"]["test"]
    metadata = pd.read_parquet(input_file, columns=[ID, TIME])
    holdout = root / "retrospective"
    covariates = build_cache(raw, v1, holdout, {"decision_freeze_sha256": sha256(freeze)}, retrospective_metadata=metadata)
    if covariates["raw_sha256"] != reference["raw_sha256"]:
        raise ValueError("Raw source fingerprints changed before retrospective scoring.")
    source = np.load(holdout / "features.npy", mmap_mode="r")
    timeline = pd.read_parquet(holdout / "development_metadata.parquet")
    build_advanced(holdout, source, covariates, timeline, raw)
    matrices = _cache_matrices(holdout)
    time = timeline[TIME].to_numpy(dtype=np.int64)
    start, stop = reference["rows"], len(timeline)
    if not np.array_equal(timeline[ID].iloc[start:].to_numpy(), metadata[ID].to_numpy()):
        raise ValueError("Retrospective order changed before scoring.")
    development = _cache_matrices(root)
    for prefix, extended in zip(development[2:], matrices[2:], strict=True):
        for low in range(0, start, CHUNK_ROWS):
            high = min(low + CHUNK_ROWS, start)
            if not np.array_equal(prefix[low:high], extended[low:high], equal_nan=True):
                raise ValueError("Appending retrospective covariates changed a development feature.")
    del development
    scores, inferences = {}, {}
    for name, model in policy["models"].items():
        tic = perf_counter()
        scores[name] = _score(name, root / model["file"], matrices, start, stop, policy["train_end"], time, threads)
        inferences[name] = perf_counter() - tic
    for name, blend in policy["blends"].items():
        scores[name] = sum(scores[member] * weight for member, weight in blend.items())
        inferences[name] = sum(inferences[member] for member in blend)
    if any(not np.isfinite(values).all() or len(values) != len(metadata) for values in scores.values()):
        raise ValueError("Incomplete model output before opening labels.")
    calibrators = joblib.load(root / "calibrators.joblib")
    adjusted = {name: _apply(calibration, model, scores[name], margin)
                for name, (calibration, model, margin) in calibrators.items()}
    champion = policy["champion"]
    metadata.assign(raw_score=scores[champion], policy_score=adjusted[champion],
                    decision=adjusted[champion] >= policy["policy"][champion]["threshold"]).to_parquet(
                        holdout / "champion_scores.parquet", index=False)
    with receipt.open("x", encoding="utf-8") as marker:
        json.dump({"opened_at_utc": _now(), "decision_freeze_sha256": sha256(freeze), "holdout_rows": len(metadata)}, marker)
    labels = pd.read_parquet(input_file, columns=[TARGET])[TARGET].to_numpy(dtype=np.uint8)
    report = {}
    for name, values in adjusted.items():
        threshold = policy["policy"][name]["threshold"]
        metrics, curves = evaluate_scores(labels, values, values >= threshold)
        report[name] = {"metrics": metrics, "threshold": threshold, "budgets": review_budgets(labels, values),
                        "brier": float(brier_score_loss(labels, values)), "inference_seconds": inferences[name],
                        "raw_roc_auc": float(roc_auc_score(labels, scores[name])),
                        "raw_average_precision": float(average_precision_score(labels, scores[name]))}
        write_json(holdout / f"{name}_curves.json", curves)
    count_index = covariates["features"].index("prior_txn_count")
    segments = _segment_metrics(labels, adjusted[champion], policy["policy"][champion]["threshold"],
                                source[start:stop, count_index] > 0)
    write_json(root / "retrospective_results.json", {"scope": "reused retrospective holdout", "rows": len(labels),
                                                   "fraud_rows": int(labels.sum()), "champion": champion,
                                                   "models": report, "champion_segments": segments})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v2"))
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--v1", type=Path)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        if args.stage == "prepare":
            prepare(args.output.resolve(), args.threads)
        else:
            if args.raw is None or args.v1 is None or args.output.resolve().is_relative_to(args.v1.resolve()):
                parser.error("Use --raw, --v1 and a separate V2 output root for evaluation.")
            evaluate(args.output.resolve(), args.raw.resolve(), args.v1.resolve(), args.threads)


if __name__ == "__main__":
    main()
