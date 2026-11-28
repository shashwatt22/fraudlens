"""Freeze Step 7B policy on early development rows, then verify the later rows."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.data_loader import TARGET, TIME
from src.evaluate import select_threshold
from src.v2.data import sha256, write_json
from src.v2.experiment import review_budgets
from .diagnostics import annotated_metrics


MODELS = ("logistic", "forest", "xgboost", "torch")
PRIMARY = "xgboost"
METHODS = ("identity", "sigmoid", "isotonic")
PROTOCOL = {"seed": 42, "bootstrap_draws": 2000, "block_seconds": 86400,
            "confidence": .95, "precision_target": .90, "recall_target": .70}
EPSILON = 1e-15


def read(path):
    return json.loads(path.read_bytes())


def freeze_file(path, payload):
    """Allow reruns of identical decisions and refuse to replace a saved freeze."""
    if path.exists():
        if read(path) != payload:
            raise ValueError(f"Refusing to change frozen file {path.name}.")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def model_manifest(development, refit):
    plan_file = refit / "refit_plan.json"
    plan = read(plan_file)
    summary = read(refit / "experiment_summary.json")
    digest = sha256(plan_file)
    source = development / "development_metadata.parquet"
    if sha256(source) != plan["inputs"][source.name] or summary["plan_sha256"] != digest:
        raise ValueError("Step 7A development lineage differs from the saved plan.")
    features = read(development / "features.json")
    if ("decision_freeze_sha256" in features or plan["rows"] != features["rows"]
            or features["time_max"] >= features["holdout_start"]
            or plan["partitions"]["policy"]["last_second"] != features["time_max"]):
        raise ValueError("Only the V2 pre-holdout development inputs are permitted.")
    parts = plan["partitions"]
    early, later, policy = (parts[name] for name in ("earlier_policy", "later_policy", "policy"))
    cut = plan["original_policy_calibration_end"]
    if (early["rows"] != later["rows"] or early["rows"] != 23_622 or early["start"] != policy["start"]
            or early["end"] != later["start"] or later["end"] != policy["end"] or early["last_second"] >= later["first_second"]
            or parts["pre_policy"]["last_second"] >= early["first_second"] or not early["start"] < cut < early["end"]):
        raise ValueError("Invalid policy boundaries.")
    models = {}
    for name in MODELS:
        folder = refit / name
        fit = read(folder / "fit.json")
        expected = summary["models"][name]
        predictions = refit / "policy_scores" / f"{name}.npy"
        if (fit != expected["fit"] or fit["freeze_sha256"] != digest or fit["name"] != name
                or any(sha256(folder / file) != checksum for file, checksum in fit["artifacts"].items())
                or sha256(predictions) != expected["scores_sha256"]):
            raise ValueError(f"Step 7A artifacts failed verification for {name}.")
        models[name] = {"fit_manifest_sha256": sha256(folder / "fit.json"), "model_file": f"{name}/{fit['model_file']}",
                        "artifact_sha256": {f"{name}/{key}": value for key, value in fit["artifacts"].items()},
                        "score_file": f"policy_scores/{name}.npy", "score_sha256": expected["scores_sha256"],
                        "fit_rows": expected["gradient_training_rows"], "recipe": plan["recipes"][name]}
    return {"scope": "previously inspected development policy block", "primary": PRIMARY, "models": models,
            "step7a_plan_sha256": digest, "development_metadata_sha256": sha256(source),
            "partitions": {name: parts[name] for name in ("policy", "earlier_policy", "later_policy")}, "internal_cut": cut}


def load_window(development, refit, manifest, name):
    part, policy = (manifest["partitions"][key] for key in (name, "policy"))
    frame = pd.read_parquet(development / "development_metadata.parquet", columns=[TIME, TARGET], engine="pyarrow",
                            filters=[(TIME, ">=", part["first_second"]), (TIME, "<=", part["last_second"])])
    times, labels = frame[TIME].to_numpy(), frame[TARGET].to_numpy()
    if (len(labels) != part["rows"] or len(labels) != part["end"] - part["start"] or not np.isin(labels, (0, 1)).all()
            or labels.sum() != part["fraud_rows"] or times[0] != part["first_second"] or times[-1] != part["last_second"]
            or np.any(np.diff(times) < 0)):
        raise ValueError(f"Unexpected population or timestamps in {name}.")
    lo, hi = part["start"] - policy["start"], part["end"] - policy["start"]
    result = {}
    for key, model in manifest["models"].items():
        scores = np.load(refit / model["score_file"], mmap_mode="r", allow_pickle=False)
        values = np.asarray(scores[lo:hi], dtype=np.float64)
        if scores.shape != (policy["rows"],) or values.shape != labels.shape or not np.all((values >= 0) & (values <= 1)):
            raise ValueError(f"Invalid scores for {key} in {name}.")
        result[key] = values
    return times, labels, result


def fit_calibrator(method, scores, labels):
    model = {"method": method}
    if method == "sigmoid":
        estimator = LogisticRegression(random_state=42).fit(logit(np.clip(scores, EPSILON, 1-EPSILON))[:, None], labels)
        model.update(slope=float(estimator.coef_.item()), intercept=float(estimator.intercept_.item()))
    elif method == "isotonic":
        estimator = IsotonicRegression(out_of_bounds="clip").fit(scores, labels)
        model.update(x=estimator.X_thresholds_.tolist(), y=estimator.y_thresholds_.tolist())
    return model


def calibrate(model, scores):
    if model["method"] == "sigmoid":
        return expit(model["slope"] * logit(np.clip(scores, EPSILON, 1-EPSILON)) + model["intercept"])
    if model["method"] == "isotonic":
        return np.interp(scores, model["x"], model["y"])
    if model["method"] != "identity":
        raise ValueError("Unknown calibration method.")
    return scores


def losses(labels, probabilities):
    p = np.clip(probabilities, EPSILON, 1-EPSILON)
    return np.vstack(((probabilities - labels) ** 2, -np.where(labels == 1, np.log(p), np.log1p(-p))))


def reliability(labels, scores):
    bins = np.minimum((scores * 10).astype(int), 9)
    return [{"count": int(mask.sum()), "forecast": float(scores[mask].mean()), "observed": float(labels[mask].mean())}
            for index in range(10) if (mask := bins == index).any()]


def compare_calibration(times, labels, raw, cut):
    if not 0 < cut < len(labels) or times[cut-1] >= times[cut]:
        raise ValueError("Calibration comparison must use disjoint timestamp groups.")
    y, base = labels[cut:], raw[cut:]
    _, blocks = np.unique(times[cut:] // PROTOCOL["block_seconds"], return_inverse=True)
    counts = np.bincount(blocks)
    rng = np.random.default_rng(PROTOCOL["seed"])
    samples = rng.integers(len(counts), size=(PROTOCOL["bootstrap_draws"], len(counts)))
    comparison = {}
    for method in METHODS:
        model = fit_calibrator(method, raw[:cut], labels[:cut])
        score = calibrate(model, base)
        values = losses(y, score)
        difference = values - losses(y, base)
        sums = np.stack([np.bincount(blocks, weights=row) for row in difference])
        deltas = sums[:, samples].sum(axis=2) / counts[samples].sum(axis=1)
        intervals = np.quantile(deltas, [.025, .975], axis=1).T
        eligible = bool(method != "identity" and len(counts) >= 2 and model.get("slope", 1) > 0 and np.all(intervals[:, 1] < 0))
        comparison[method] = {"brier": float(values[0].mean()), "log_loss": float(values[1].mean()),
                              "delta_ci": dict(zip(("brier", "log_loss"), intervals.tolist())), "eligible": eligible,
                              "reliability": reliability(y, score)}
    return {"fit_rows": cut, "comparison_rows": len(y), "day_blocks": len(counts),
            "fit_last_second": int(times[cut-1]), "comparison_first_second": int(times[cut]), "methods": comparison}


def alternatives(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    result = {}
    for name, constraint, target, objective in (
        ("high_precision", precision[:-1], PROTOCOL["precision_target"], recall[:-1]),
        ("high_recall", recall[:-1], PROTOCOL["recall_target"], precision[:-1]),
    ):
        eligible = np.flatnonzero(constraint >= target)
        best = max(eligible, key=lambda i: (objective[i], constraint[i], thresholds[i])) if len(eligible) else None
        result[name] = float(thresholds[best]) if best is not None else None
    return result


def develop(development, refit, out):
    models = model_manifest(development, refit)
    freeze_file(out / "model_freeze.json", models)
    times, y, scores = load_window(development, refit, models, "earlier_policy")
    comparison = compare_calibration(times, y, scores[PRIMARY], models["internal_cut"] - models["partitions"]["policy"]["start"])
    choices = comparison["methods"]
    justified = [method for method in METHODS if choices[method]["eligible"]]
    method = min(justified, key=lambda m: (choices[m]["brier"], choices[m]["log_loss"])) if justified else "identity"
    calibrator = fit_calibrator(method, scores[PRIMARY], y)
    policies = {}
    for name, raw in scores.items():
        values = calibrate(calibrator, raw) if name == PRIMARY else raw
        policies[name] = {"selected_f1": select_threshold(y, values),
                          **(alternatives(y, values) if name == PRIMARY else {})}
    freeze_file(out / "decision_freeze.json", {"model_freeze_sha256": sha256(out / "model_freeze.json"),
                "implementation_sha256": sha256(Path(__file__)), "primary": PRIMARY, "protocol": PROTOCOL,
                "comparison": comparison, "calibrator": calibrator, "thresholds": policies})


def operating_metrics(labels, raw, decision_values, cutoff):
    metrics = annotated_metrics(labels, decision_values, cutoff)
    reviewed = int(np.count_nonzero(decision_values >= cutoff))
    return {"threshold": cutoff, "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"],
            "accuracy": metrics["accuracy"], "confusion_matrix": metrics["confusion_matrix"],
            "reviewed": reviewed, "review_rate": reviewed / len(raw)}


def window_metrics(y, scores, decision):
    output = {}
    for name, raw in scores.items():
        probabilities = calibrate(decision["calibrator"], raw) if name == PRIMARY else raw
        raw_losses, probability_losses = losses(y, raw).mean(axis=1), losses(y, probabilities).mean(axis=1)
        precision, recall, _ = precision_recall_curve(y, raw)
        policies = {key: operating_metrics(y, raw, probabilities, cutoff) for key, cutoff in
                    {"fixed_0.5": .5, **decision["thresholds"][name]}.items() if cutoff is not None}
        if name == PRIMARY:
            policies["original_score_0.5"] = operating_metrics(y, raw, raw, .5)
        output[name] = {"ranking": {"ap": float(average_precision_score(y, raw)), "roc_auc": float(roc_auc_score(y, raw)),
                                    "review_budgets": review_budgets(y, raw), "precision": precision.tolist(), "recall": recall.tolist()},
                        "probability": {"raw_brier": float(raw_losses[0]), "raw_log_loss": float(raw_losses[1]),
                                        "brier": float(probability_losses[0]), "log_loss": float(probability_losses[1]),
                                        "raw_reliability": reliability(y, raw), "reliability": reliability(y, probabilities)},
                        "operating_points": policies}
    return output


def verify(development, refit, out):
    path = out / "decision_freeze.json"
    if not path.is_file() or not (out / "model_freeze.json").is_file():
        raise ValueError("Freeze the early policy before evaluating the later window.")
    decision = read(path)
    models = read(out / "model_freeze.json")
    if (decision["model_freeze_sha256"] != sha256(out / "model_freeze.json")
            or decision["implementation_sha256"] != sha256(Path(__file__)) or models != model_manifest(development, refit)):
        raise ValueError("Frozen policy lineage no longer matches the files.")
    windows = {}
    for name in ("earlier_policy", "later_policy"):
        _, labels, scores = load_window(development, refit, models, name)
        windows[name] = window_metrics(labels, scores, decision)
    summary = {"decision_sha256": sha256(path), "model_sha256": sha256(out / "model_freeze.json"),
               "partitions": models["partitions"], "windows": windows}
    write_json(out / "policy_metrics.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("develop", "verify"))
    for key in ("development", "refit", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    root, refit, out = (path.resolve() for path in (args.development, args.refit, args.output))
    if out.is_relative_to(root) or out.is_relative_to(refit) or any(
            "retrospective" in {part.casefold() for part in path.parts} for path in (root, refit, out)):
        raise ValueError("Use development inputs and a separate policy output path.")
    out.mkdir(parents=True, exist_ok=True)
    (develop if args.stage == "develop" else verify)(root, refit, out)
    print(f"Policy {args.stage} completed in {out}.")


if __name__ == "__main__":
    main()
