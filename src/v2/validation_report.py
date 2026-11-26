"""Read measured development trials, calculate Pareto results and test OOF blends."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data_loader import TARGET
from src.evaluate import evaluate_scores, select_threshold

from .data import write_json
from .experiment import review_budgets


def records(output):
    result = {}
    files = [*output.glob("trials/*/summary.json"), *output.glob("challengers/*/summary.json")]
    plan_path = output / "features.json"
    plan = json.loads(plan_path.read_bytes())["folds"] if plan_path.is_file() else None
    for path in sorted(files, key=lambda f: ((f.parent / "fold3.json").stat().st_mtime_ns, f.parent.name)):
        report = json.loads(path.read_bytes())
        if len(report["folds"]) != 3:
            continue
        if plan and any(any(fold.get(key) != expected[key] for key in ("train_end", "validation_end"))
                        for fold, expected in zip(report["folds"], plan, strict=True)):
            raise ValueError(f"Validation fold populations differ for {path.parent.name}.")
        result[path.parent.name] = (path, report)
    return result


def can_promote(candidate, current):
    metrics = candidate["metrics"]
    present = current["metrics"]
    gains = sum(x["metrics"]["average_precision"] > y["metrics"]["average_precision"]
                for x, y in zip(candidate["folds"], current["folds"], strict=True))
    return (metrics["average_precision"]["mean"] > present["average_precision"]["mean"] and gains >= 2
            and metrics["roc_auc"]["mean"] >= present["roc_auc"]["mean"] - .01)


def _group(name):
    if name.startswith("lgbm"):
        return "lgbm"
    if name.startswith("catboost"):
        return "catboost"
    if name.startswith("torch"):
        return "torch"
    return "xgb"


def _mean(report, metric):
    return report["metrics"][metric]["mean"]


def _is_ensemble_component(name):
    return name in {"rich", "history", "v1", "unweighted"} or name.startswith(("xgb_", "lgbm", "catboost", "torch"))


def _fold_arrays(found, root, frame):
    plan = json.loads((root / "features.json").read_bytes())["folds"]
    groups = {}
    for name, (_, report) in found.items():
        if not _is_ensemble_component(name):
            continue
        scores = report["folds"][:2]
        ap = float(np.mean([fold["metrics"]["average_precision"] for fold in scores]))
        group = _group(name)
        if group not in groups or ap > groups[group][1]:
            groups[group] = name, ap
    if len(groups) < 2:
        return [], [], []
    names = [name for name, _ in groups.values()]
    data, truth = [], []
    for index, fold in enumerate(plan, 1):
        start, end = fold["train_end"], fold["validation_end"]
        values = [np.load(found[name][0].parent / f"scores{index}.npy", allow_pickle=False) for name in names]
        if any(item.shape != (end-start,) or not np.isfinite(item).all() for item in values):
            raise ValueError(f"Misaligned validation scores in fold {index}.")
        truth.append(frame[TARGET].iloc[start:end].to_numpy(dtype=np.uint8))
        data.append(np.column_stack(values))
    return names, data, truth


def _weights(data, truth):
    count = data[0].shape[1]

    def objective(weights):
        return float(np.mean([average_precision_score(y, x @ weights) for x, y in zip(data[:2], truth[:2], strict=True)]))

    weights = np.full(count, 1 / count)
    best = objective(weights)
    for _ in range(2):
        for column in range(count):
            for share in (.1, .25, .5):
                candidate = weights * (1-share)
                candidate[column] += share
                performance = objective(candidate)
                if performance > best:
                    weights, best = candidate, performance
    return weights


def _blend_report(name, weights, names, data, truth):
    scores = [x @ weights for x in data]
    decision = select_threshold(np.concatenate(truth[:2]), np.concatenate(scores[:2]))
    folds = []
    for i, (y, score) in enumerate(zip(truth, scores, strict=True), 1):
        metrics, _ = evaluate_scores(y, score, score >= decision)
        folds.append({"fold": i, "metrics": metrics, "budgets": review_budgets(y, score),
                      "used_to_select_weights": i < 3})
    result = {"name": name, "members": dict(zip(names, map(float, weights), strict=True)), "threshold": decision,
              "folds": folds, "metrics": {name: {"mean": float(np.mean([f["metrics"][name] for f in folds]))}
                                         for name in ("average_precision", "roc_auc", "f1")}}
    return result


def summarize(root):
    found = records(root)
    if "v1" not in found:
        raise ValueError("Complete the V1 chronological reference before publishing selection summaries.")
    champion = found["v1"][1]
    accepted = ["v1"]
    for name, (_, report) in found.items():
        if name != "v1" and can_promote(report, champion):
            accepted.append(name)
            champion = report
    means = {name: (_mean(record, "average_precision"), _mean(record, "roc_auc")) for name, (_, record) in found.items()}
    pareto = [name for name, pair in means.items() if not any(other != pair and all(b >= a for a, b in zip(pair, other))
                                                             for other in means.values())]
    frame = pd.read_parquet(root / "development_metadata.parquet", columns=[TARGET])
    names, data, truth = _fold_arrays(found, root, frame)
    blends = []
    if len(names) >= 2:
        for label, weights in (("mean", np.full(len(names), 1/len(names))), ("weighted", _weights(data, truth))):
            blends.append(_blend_report(label, weights, names, data, truth))
    selection = {"scope": "development only", "champion_individual": accepted[-1], "progression": accepted,
                 "pareto": pareto, "completed_trials": list(found), "blends": blends,
                 "ensemble_promotion_pending": bool(blends)}
    report = root / "development_selection.json"
    write_json(report, selection)
    print(f"{len(found)} completed trials. Accepted individual progression: {' -> '.join(accepted)}")
    for blend in blends:
        print(f"{blend['name']} blend: mean AP {blend['metrics']['average_precision']['mean']:.5f}; "
              f"untuned fold 3 AP {blend['folds'][2]['metrics']['average_precision']:.5f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/v2"))
    summarize(parser.parse_args().output)


if __name__ == "__main__":
    main()
