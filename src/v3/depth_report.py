"""Report the paired depth-8 and depth-10 XGBoost development comparison."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.v2.data import sha256, write_json
from .behavior import anchor_keys
from .behavior_experiment import load_inputs
from .behavior_report import ARMS, METRICS, compare_bootstrap, gib
from .diagnostics import XGB


ACCURACY_METRICS = (*METRICS, "accuracy")
LABELS = {"control": "Depth 8", "treatment": "Depth 10"}


def checked_records(root, out, matrices, labels, plan, completed):
    _, advanced, _, data = matrices
    keys = anchor_keys(data[:plan[-1].validation_end], advanced["features"])
    records, pairs = [], []
    for index in completed:
        folder = out / f"fold{index}"
        record = json.loads((folder / "result.json").read_bytes())
        control = root / "challengers" / XGB / f"scores{index}.npy"
        challenger = folder / "depth10_scores.npy"
        fold = plan[index - 1]
        start, stop = fold.train_end, fold.validation_end
        if (record["fold"] != index or record["fit_rows"] != start or record["validation_end"] != stop
                or record["input_digests"][control.name] != sha256(control)
                or record["scores_sha256"] != sha256(challenger)
                or record["model_sha256"] != sha256(folder / "depth10.ubj")):
            raise ValueError(f"Fold {index} artifacts differ from the recorded completion.")
        records.append(record)
        pairs.append((labels[start:stop], np.load(control, allow_pickle=False),
                      np.load(challenger, allow_pickle=False), keys[start:stop]))
    return records, pairs


def summary(records, pairs, status):
    metrics, gaps, entity, resources = {}, {}, {}, {}
    checks = None
    uncertainty = None
    gains = 0
    delta = {}
    if records:
        for arm in ARMS:
            values = {name: float(np.mean([row[arm][name] for row in records])) for name in ACCURACY_METRICS}
            values.update({f"recall_at_{q}pct": float(np.mean([row["review_budgets"][arm][str(q)]["recall"]
                                                              for row in records])) for q in (5, 10)})
            metrics[arm] = values
            train = {key: float(np.mean([row["train"][arm][key] for row in records]))
                     for key in ("average_precision", "roc_auc")}
            gaps[arm] = {**train, **{f"gap_{key}": train[key] - values[key] for key in train}}
            entity[arm] = {group: float(np.mean([row["entities"][group][arm]["average_precision"] for row in records]))
                           for group in records[0]["entities"]}
            key = "archived_control" if arm == "control" else "challenger_resources"
            resources[arm] = {name: float(np.mean([row[key][name] for row in records]))
                              for name in records[0][key] if all(row[key][name] is not None for row in records)}
        delta = {name: metrics["treatment"][name] - metrics["control"][name] for name in metrics["control"]}
        gains = sum(row["treatment"]["average_precision"] > row["control"]["average_precision"] for row in records)
        checks = {"all_folds_completed": len(records) == 3, "mean_ap_gain_at_least_0.01": delta["average_precision"] >= .01,
                  "two_or_more_folds_improved": gains >= 2, "mean_auc_loss_at_most_0.01": delta["roc_auc"] >= -.01}
        uncertainty = compare_bootstrap(pairs)
    return {"scope": "V2 chronological development folds", "folds": records, "status": status, "means": metrics,
            "deltas": delta, "train_means_and_gaps": gaps, "entity_means": entity, "mean_resources": resources,
            "improved_ap_folds": gains, "promotion_checks": checks, "recommend_promotion": bool(checks and all(checks.values())),
            "bootstrap": uncertainty, "challenger_total_fit_seconds": sum(row["challenger_resources"]["fit_seconds"] for row in records),
            "challenger_peak_bytes": max((row["challenger_resources"]["peak_working_set_bytes"] or 0 for row in records), default=0)}


def markdown(report):
    rows, status = report["folds"], report["status"]
    recommendation = "Promote depth 10." if report["recommend_promotion"] else "Keep depth 8."
    lines = ["# FraudLens V3 Step 5A XGBoost capacity experiment", "", f"**{recommendation}**"]
    if len(rows) < 3:
        lines += ["", f"Completed {len(rows)} of the three folds. {status['constraint'] or 'The run ended early.'}"]
    if rows:
        delta = report["deltas"]
        lines += ["", f"Mean AP changed by {delta['average_precision']:+.5f} and ROC-AUC by {delta['roc_auc']:+.5f}. "
                  f"AP improved in {report['improved_ap_folds']} of the {len(rows)} completed folds. "
                  "Promotion requires an AP gain of at least 0.01, gains in at least two folds and an AUC drop no larger than 0.01."]
    lines += ["", "Both arms used the original 497 V2 predictors, the same three chronological train and validation populations, "
              "320 boosting rounds, learning rate 0.055, L2 regularization 4, seed 42, and six CPU threads. "
              "The challenger changes only maximum tree depth from 8 to 10. The saved depth-8 control predictions were verified by checkpoint reload. "
              "All metrics use development data. The later policy block and the retrospective holdout are outside the comparison.", "",
              "## Validation performance", "",
              "Precision, recall, F1 and accuracy use a cutoff of 0.5. Review budgets are ranked within each fold and rounded up to whole rows. "
              "Means weight folds equally.", "",
              "| Fold | Fit rows | Model | AP | ROC-AUC | Precision | Recall | F1 | Accuracy | Recall at 5% | Recall at 10% |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        for arm in ARMS:
            scores = [row[arm][key] for key in ACCURACY_METRICS]
            scores.extend(row["review_budgets"][arm][str(q)]["recall"] for q in (5, 10))
            lines.append(f"| {row['fold']} | {row['fit_rows']:,} | {LABELS[arm]} | " + " | ".join(f"{value:.5f}" for value in scores) + " |")
    if rows:
        for arm in ARMS:
            scores = report["means"][arm]
            keys = (*ACCURACY_METRICS, "recall_at_5pct", "recall_at_10pct")
            lines.append(f"| Mean | — | {LABELS[arm]} | " + " | ".join(f"{scores[key]:.5f}" for key in keys) + " |")
    lines += ["", "## Training and validation gaps", "",
              "Training metrics are in-sample and measure fit, including overfitting. Each gap subtracts the later validation score from the training score.", "",
              "| Fold | Model | Train AP | Validation AP | AP gap | Train ROC-AUC | Validation ROC-AUC | ROC-AUC gap |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        for arm in ARMS:
            vals = []
            for key in ("average_precision", "roc_auc"):
                train, valid = row["train"][arm][key], row[arm][key]
                vals.extend((train, valid, train - valid))
            lines.append(f"| {row['fold']} | {LABELS[arm]} | " + " | ".join(f"{value:.5f}" for value in vals) + " |")
    if rows:
        for arm in ARMS:
            vals = []
            for key in ("average_precision", "roc_auc"):
                vals.extend((report["train_means_and_gaps"][arm][key], report["means"][arm][key], report["train_means_and_gaps"][arm][f"gap_{key}"]))
            lines.append(f"| Mean | {LABELS[arm]} | " + " | ".join(f"{value:.5f}" for value in vals) + " |")
    lines += ["", "## Entity results", "",
              "A complete card–address–D1 key is seen when it occurs in that fold's training prefix. Missing keys form a separate stratum.", "",
              "| Fold | Entity stratum | Rows | Fraud rate | Depth-8 AP | Depth-10 AP |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        for group, arms in row["entities"].items():
            a, b = (arms[arm] for arm in ARMS)
            lines.append(f"| {row['fold']} | {group.replace('_', ' ')} | {a['rows']:,} | {a['fraud_rate']:.2%} | "
                         f"{a['average_precision']:.5f} | {b['average_precision']:.5f} |")
    if rows:
        for group in report["entity_means"]["control"]:
            a, b = (report["entity_means"][arm][group] for arm in ARMS)
            lines.append(f"| Mean | {group.replace('_', ' ')} | — | — | {a:.5f} | {b:.5f} |")
        boot = report["bootstrap"]
        lines += ["", "## Paired uncertainty", "",
                  f"The {boot['valid_replicates']:,} valid resamples use matching entity-block weights for both scorers and every fold. "
                  f"There are {boot['entity_blocks']:,} complete entity blocks and {boot['missing_key_singleton_blocks']:,} missing-key singleton blocks. "
                  "Percentile intervals are conditional on the fitted checkpoints and observed development periods. They do not account for alternate seeds or future time periods.", "",
                  "| Fold | AP change | 95% interval |", "| --- | ---: | --- |"]
        for row, ci in zip(rows, boot["fold_ap_delta_ci95"], strict=True):
            diff = row["treatment"]["average_precision"] - row["control"]["average_precision"]
            lines.append(f"| {row['fold']} | {diff:+.5f} | [{ci['lower']:+.5f}, {ci['upper']:+.5f}] |")
        ci = boot["mean_ap_delta_ci95"]
        lines.append(f"| Mean | {report['deltas']['average_precision']:+.5f} | [{ci['lower']:+.5f}, {ci['upper']:+.5f}] |")
    lines += ["", "## Runtime and checkpoints", "",
              "Depth-8 timings and process peaks come from its archived run. Depth-10 resources come from separate fold workers. "
              "Inference is validation scoring. The reported stored sizes are compressed Joblib for depth 8 and native UBJSON for depth 10.", "",
              "| Fold | Model | Training (s) | Inference (s) | Peak GiB | Checkpoint MiB |", "| --- | --- | ---: | ---: | ---: | ---: |"]

    def print_resource(fold, arm, stats):
        lines.append(f"| {fold} | {LABELS[arm]} | {stats['fit_seconds']:.2f} | {stats['inference_seconds']:.2f} | "
                     f"{gib(stats.get('peak_working_set_bytes'))} | {stats['checkpoint_bytes'] / (1 << 20):.2f} |")

    for row in rows:
        print_resource(row["fold"], "control", row["archived_control"])
        print_resource(row["fold"], "treatment", row["challenger_resources"])
    if rows:
        for arm in ARMS:
            print_resource("Mean", arm, report["mean_resources"][arm])
    lines += ["", f"The challenger fit time was {report['challenger_total_fit_seconds']:.2f} s total. "
              f"The training command took {status['wall_seconds']:.2f} s wall time, including verification and checkpointing.", "",
              "## Reproduce or resume", "",
              "The runner verifies existing fold hashes before skipping completed work. Choose a fresh output directory for an independent run.", "",
              "```powershell", r"Set-Location E:\fraudlens-v3", r"$python = 'E:\fraudlens\.venv\Scripts\python.exe'",
              r"$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'", r"$dev = 'E:\fraudlens-v2\artifacts\v2'",
              r"$out = 'artifacts\v3\step5a-depth'", r"& $python -B -m src.v3.depth_experiment --development $dev --output $out",
              r"& $python -B -m src.v3.depth_report --development $dev --output $out --markdown 'docs\V3_STEP5A.md'", "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    root, out, matrices, _, labels, plan = load_inputs(args.development, args.output)
    status = json.loads((out / "run_status.json").read_bytes())
    records, pairs = checked_records(root, out, matrices, labels, plan, status["verified_folds"])
    report = summary(records, pairs, status)
    write_json(out / "experiment_summary.json", report)
    if args.markdown:
        args.markdown.write_text(markdown(report), encoding="utf-8", newline="\n")
    print(f"Reported {len(records)} completed folds. Promote depth 10: {report['recommend_promotion']}.")


if __name__ == "__main__":
    main()
