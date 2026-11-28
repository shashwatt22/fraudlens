"""Create the public aggregate export from verified offline reports only."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


BUNDLE_SHA = "9812d81f3538f231bd607395ac9592f31c9d78d85f8eb8ba38be8ce1241dc7b9"
V1_SHA = "302e1755897bc4ad3ddda4baaa0f5651976202e67ed6e38034e1707a95616a6b"
SCORE_SHA = "c9044671236453f20d0b874092483604db7f115e65be07537680128b34f4e2d4"
MODELS = ("xgboost", "logistic", "forest", "torch", "v1_xgboost", "v2_ensemble")
SCALARS = ("name", "average_precision", "roc_auc", "brier", "log_loss")
POINT = ("threshold", "precision", "recall", "f1", "accuracy", "confusion_matrix", "reviewed", "review_rate")
EXPERIMENTS = (
    ("step2-behavior", "Seven-day and D-offset features", "control", "treatment", "Rejected"),
    ("step3a-logistic", "Full-data Logistic Regression", "sampled", "full", "Selected"),
    ("step4a-forest", "Full-data Random Forest", "sampled", "full", "Selected"),
    ("step5a-depth", "XGBoost depth 8 to 10", "control", "treatment", "Selected"),
    ("step6a-torch", "Full-data 224-input MLP", "sampled", "full", "Selected"),
    ("step6b-rich", "Richer MLP representation", "control", "challenger", "Rejected"),
)


def read(path, expected=None):
    content = path.read_bytes()
    if expected and sha256(content).hexdigest() != expected:
        raise ValueError(f"{path.name} failed its frozen digest check.")
    return json.loads(content)


def select(row, keys):
    return {key: row[key] for key in keys}


def curve(source, x, y):
    size = len(source[x])
    indexes = sorted({round(i * (size - 1) / 100) for i in range(101)})
    return {axis: [round(source[axis][i], 5) for i in indexes] for axis in (x, y)}


def model_snapshot(row):
    charts = row["curves"]
    result = select(row, SCALARS)
    result.update({cutoff: select(row[cutoff], POINT) for cutoff in ("frozen", "at_0_5")})
    result["review_budgets"] = {key: select(metrics, ("reviewed", "recall"))
                                for key, metrics in row["review_budgets"].items()}
    result["curves"] = {
        "roc": curve(charts["roc"], "false_positive_rate", "true_positive_rate"),
        "precision_recall": curve(charts["precision_recall"], "recall", "precision"),
        "reliability": [select(point, ("count", "forecast", "observed")) for point in charts["reliability"]
                        if point["count"] >= 100],
    }
    return result


def behavioral_summary(v2):
    rows = []
    for item in read(v2 / "patterns.json")["folds"]:
        for name, family in item["families"].items():
            cluster = family["clusters"]["selected"]
            anomaly = next(result for result in family["anomaly"] if result["tree_samples"] == 768)
            rows.append({"fold": item["fold"], "representation": name, "anomaly_ap": anomaly["average_precision"],
                         "silhouette": cluster["silhouette"], "noise_fraction": cluster["noise_fraction"]})
    return rows


def build(v3, v1, v2):
    retrospect = v3 / "step7c-retrospective"
    saved = read(retrospect / "dashboard.json", BUNDLE_SHA)
    scores = read(retrospect / "score_manifest.json", SCORE_SHA)
    source = read(v1 / "manifest.json", V1_SHA)
    fit = v3 / "step7a-refit"
    neural = read(fit / "torch" / "fit.json", scores["frozen_hashes"]["torch/fit.json"])
    plan = read(fit / "refit_plan.json", scores["frozen_hashes"]["refit_plan.json"])
    experiments = []
    for folder, name, left, right, selection in EXPERIMENTS:
        metrics = read(v3 / folder / "experiment_summary.json")["means"]
        experiments.append({"name": name, "baseline_ap": metrics[left]["average_precision"],
                            "candidate_ap": metrics[right]["average_precision"], "selection": selection})
    entity = saved["models"]["xgboost"]["entities"]
    return {
        "schema_version": 1, "population": select(saved["population"], ("rows", "fraud_rows", "previously_evaluated_in")),
        "scope": saved["scope"], "calibration": saved["calibration"], "primary": saved["primary"],
        "dataset": {"transactions": source["source_rows"]["transactions"], "identities": source["source_rows"]["identities"],
                    "pre_policy": plan["partitions"]["pre_policy"]["rows"], "policy": plan["partitions"]["policy"]["rows"]},
        "models": {name: model_snapshot(saved["models"][name]) for name in MODELS},
        "entity_cohorts": {name: select(row["frozen"], ("rows", "fraud_rate", "average_precision", "recall"))
                           for name, row in entity.items()},
        "experiments": experiments,
        "neural": {"architecture": [neural["input_dim"], *plan["recipes"]["torch"]["candidate"]["options"]["widths"], 1],
                   "parameters": neural["parameters"], "best_epoch": neural["best_epoch"],
                   "epochs": [select(row, ("epoch", "train_loss", "internal_validation_loss", "internal_validation_ap"))
                              for row in neural["history"]]},
        "pattern_folds": behavioral_summary(v2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("v3", "v1", "v2", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    args = parser.parse_args()
    payload = build(args.v3, args.v1, args.v2)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
