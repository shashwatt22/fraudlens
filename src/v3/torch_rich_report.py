"""Write Step 6B metrics, paired bootstrap and learning curves from saved scores."""

import argparse
import json
from pathlib import Path
import re

import numpy as np

from src.v2.data import sha256, write_json
from .behavior import anchor_keys
from .behavior_report import compare_bootstrap
from .diagnostics import development_inputs
from .torch_report import COLUMNS, cells, plt, validation
from .torch_rich import experiment_args, paths


ARMS = ("control", "challenger")


def collect(root, control, out):
    status = json.loads((out / "run_status.json").read_bytes())
    matrices, _, labels, plan = development_inputs(root)
    keys = anchor_keys(matrices[3][:plan[-1].validation_end], matrices[1]["features"])
    rows, paired = [], []
    for i in status["verified_folds"]:
        folder, old = out / f"fold{i}", control / f"fold{i}"
        row = json.loads((folder / "result.json").read_bytes())
        manifest = json.loads((old / "result.json").read_bytes())
        if (any(sha256(p / name) != digest for p, record in ((folder, row), (old, manifest))
                for name, digest in record["artifact_sha256"].items())
                or sha256(old / "result.json") != row["provenance"]["control_result_sha256"]
                or row["best_epoch"] != max(row["history"], key=lambda epoch: epoch["internal_validation_ap"])["epoch"]):
            raise ValueError(f"Fold {i} artifact or best-epoch mismatch.")
        start, end = plan[i - 1].train_end, plan[i - 1].validation_end
        paired.append((labels[start:end], np.load(old / "external_scores.npy", allow_pickle=False),
                      np.load(folder / "external_scores.npy", allow_pickle=False), keys[start:end]))
        rows.append(row)
    means = {arm: {key: float(np.mean([validation(row, arm)[key] for row in rows])) for key in COLUMNS} for arm in ARMS} if rows else {}
    delta = {key: means["challenger"][key] - means["control"][key] for key in COLUMNS} if rows else {}
    improved = sum(row["external"]["challenger"]["average_precision"] > row["external"]["control"]["average_precision"] for row in rows)
    checks = {"all_folds_completed": len(rows) == 3, "mean_ap_gain_at_least_0.01": bool(rows and delta["average_precision"] >= .01),
              "at_least_two_folds_improved": improved >= 2, "mean_auc_loss_at_most_0.01": bool(rows and delta["roc_auc"] >= -.01)}
    return {"folds": rows, "status": status, "means": means, "deltas": delta, "improved_folds": improved, "promotion_checks": checks,
            "recommend_promotion": all(checks.values()), "bootstrap": compare_bootstrap(paired) if paired else None}


def chart(rows, path):
    # Use the Step 6A plotting layout with both models' internal loss curves.
    figure, axes = plt.subplots(len(rows), 2, squeeze=False, figsize=(10, 3.05 * len(rows)), layout="constrained")
    for row, (left, right) in zip(rows, axes, strict=True):
        for history, arm, color, style, best in (
            (row["control_history"], "224 inputs", "#586174", "--", row["control_best_epoch"]),
            (row["history"], f"{row['input_dim']} inputs", "#06718e", "-", row["best_epoch"]),
        ):
            epochs = [item["epoch"] for item in history]
            left.plot(epochs, [item["train_loss"] for item in history], linestyle=style, color=color, label=f"{arm} training")
            left.plot(epochs, [item["internal_validation_loss"] for item in history], linestyle=style, color=color, marker=".",
                      label=f"{arm} internal")
            right.plot(epochs, [item["internal_validation_ap"] for item in history], linestyle=style, color=color, marker=".", label=arm)
            right.axvline(best, linestyle=":", color=color, alpha=.65)
        for axis, ylabel in ((left, "Binary cross-entropy"), (right, "Average precision")):
            axis.set(title=f"Fold {row['provenance']['fold']}", xlabel="Epoch", ylabel=ylabel)
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    figure.suptitle("FraudLens V3 Step 6B · Training and internal early stopping")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def markdown(report, figure):
    folds, status, means, delta = (report[name] for name in ("folds", "status", "means", "deltas"))
    lines = ["# FraudLens V3 Step 6B richer PyTorch inputs", ""]
    if len(folds) < 3:
        lines += [f"Completed {len(folds)} of three folds. {status['constraint'] or 'The run ended early.'}", ""]
    if folds:
        decision = "Promote the richer MLP." if report["recommend_promotion"] else "Keep the 224-input Step 6A MLP."
        lines += [f"**{decision}** Mean external AP changed from {means['control']['average_precision']:.5f} to {means['challenger']['average_precision']:.5f} "
                  f"({delta['average_precision']:+.5f}), with {report['improved_folds']} improved folds. Mean ROC-AUC changed by {delta['roc_auc']:+.5f}. "
                  "The rule requires an AP gain of at least 0.01, improvement in at least two folds, and an AUC decline of at most 0.01. "
                  "Depth-10 XGBoost remains the overall development champion.", ""]
    lines += ["## Representation and protocol", "",
              "Step 6A starts from 497 V2 candidate columns. It replaces categories with training log-counts, imputes missing values with training medians, "
              "and appends one missing indicator per candidate. A supervised ANOVA F-score ranks those 994 columns and keeps the best 224, which are standardized. "
              "A value can be excluded even when its missing flag survives. All three controls excluded the values of `TransactionAmt`, `card1`, `D1`, `D4` and `prior_txn_count` alongside many anonymized Vesta fields.", "",
              "The challenger retains all 460 approved numerical values and 37 single-column categorical frequency encodings. It adds only missing indicators that vary on the gradient rows. "
              "It retains the existing missing and unseen-category handling and uses no supervised feature ranking. Labels, transaction IDs, raw timestamps, entity keys and the rejected Step 2 columns are excluded. "
              "The per-fold JSON lists every retained and omitted control value and flag by name.", "",
              "Only the first layer's input width changes in the architecture. Both networks use 192, 96 and 48 hidden units with BatchNorm, GELU and dropout 0.18, followed by a single logit. "
              "Both use AdamW, unweighted binary cross-entropy, cosine scheduling, seed 42 and six CPU threads. Preprocessing uses gradient-training rows only. "
              "Gradient, internal early-stopping and external validation populations exactly match Step 6A. The internal AP selects checkpoints.", "",
              "| Fold | Old → new width | Control values / flags | Omitted control values / Vesta values | New flags | Old → new parameters | Gradient / internal / external rows | Best epoch / total, control → challenger |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in folds:
        p, inv = row["provenance"], row["input_inventory"]
        omitted = inv["control_omitted_values"]
        vesta = sum(re.fullmatch(r"V\d+", name) is not None for name in omitted)
        lines.append(f"| {p['fold']} | {row['control_input_dim']} → {row['input_dim']} | {len(inv['control_values'])} / {len(inv['control_missing_indicators'])} | "
                     f"{len(omitted)} / {vesta} | {len(inv['challenger_missing_indicators'])} | {row['control_parameters']:,} → {row['parameters']:,} | "
                     f"{p['gradient_rows']:,} / {p['early_stopping_rows']:,} / {p['validation_end']-p['train_end']:,} | "
                     f"{row['control_best_epoch']} / {len(row['control_history'])} → {row['best_epoch']} / {len(row['history'])} |")
    lines += ["", "## External validation", "", "Decision metrics use 0.5. Review budgets rank the highest-scored rows within the fold. Means weight folds equally.", "",
              "| Fold | Arm | AP | ROC-AUC | Precision | Recall | F1 | Accuracy | Recall at 5% | Recall at 10% |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in folds:
        for arm in ARMS:
            lines.append(f"| {row['provenance']['fold']} | {arm} | {cells(validation(row, arm), COLUMNS)} |")
    for arm in means:
        lines.append(f"| Mean | {arm} | {cells(means[arm], COLUMNS)} |")
    if folds:
        lines.append(f"| Mean change | challenger − control | {cells(delta, COLUMNS)} |")
        boot = report["bootstrap"]
        lines += ["", "## Paired entity-block uncertainty", "",
                  f"The {boot['valid_replicates']:,} paired bootstrap resamples use shared block weights for the two score sets. "
                  "Complete card–address–D1 keys are shared across folds. Incomplete keys are singleton blocks. The intervals describe these fitted models on observed development periods.", "",
                  "| Fold | AP change | 95% interval |", "| --- | ---: | --- |"]
        for row, ci in zip(folds, boot["fold_ap_delta_ci95"], strict=True):
            change = row["external"]["challenger"]["average_precision"] - row["external"]["control"]["average_precision"]
            lines.append(f"| {row['provenance']['fold']} | {change:+.5f} | [{ci['lower']:+.5f}, {ci['upper']:+.5f}] |")
        ci = boot["mean_ap_delta_ci95"]
        lines.append(f"| Mean | {delta['average_precision']:+.5f} | [{ci['lower']:+.5f}, {ci['upper']:+.5f}] |")
        lines += ["", "## Learning curves and overfitting", "", f"![Training and internal validation losses and internal average precision]({figure})", "",
                  "| Fold | Arm | Train AP | External AP | AP gap | Train ROC-AUC | External ROC-AUC |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in folds:
            for arm in ARMS:
                tr, ex = row["training"][arm], row["external"][arm]
                a, b = tr["average_precision"], ex["average_precision"]
                lines.append(f"| {row['provenance']['fold']} | {arm} | {a:.5f} | {b:.5f} | {a-b:.5f} | {tr['roc_auc']:.5f} | {ex['roc_auc']:.5f} |")
        for arm in ARMS:
            a = np.mean([row["training"][arm]["average_precision"] for row in folds])
            b = means[arm]["average_precision"]
            auc = np.mean([row["training"][arm]["roc_auc"] for row in folds])
            lines.append(f"| Mean | {arm} | {a:.5f} | {b:.5f} | {a-b:.5f} | {auc:.5f} | {means[arm]['roc_auc']:.5f} |")
        lines += ["", "The in-sample versus later external gap includes temporal differences. The plotted early-stop curves use the same timestamp-aligned internal populations. Epoch values are in the JSON.", "",
                  "## Seen and unseen entity proxies", "", "Seen keys appeared among the gradient rows. Both models use the same groups and prevalence.", "",
                  "| Fold | Entity stratum | Rows | Fraud rate | Control AP | Challenger AP |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for row in folds:
            for group, pair in row["entities"].items():
                a, b = (pair[arm] for arm in ARMS)
                lines.append(f"| {row['provenance']['fold']} | {group.replace('_',' ')} | {a['rows']:,} | {a['fraud_rate']:.2%} | "
                             f"{a['average_precision']:.5f} | {b['average_precision']:.5f} |")
        for group in folds[0]["entities"]:
            a, b = (np.mean([row["entities"][group][arm]["average_precision"] for row in folds]) for arm in ARMS)
            lines.append(f"| Mean | {group.replace('_',' ')} | — | — | {a:.5f} | {b:.5f} |")
    lines += ["", "## Resources and verification", "",
              "Training time includes preprocessing and internal scoring. Working set is measured per worker. Checkpoint sizes include the fitted preprocessor.", "",
              "| Fold | Arm | Training (s) | External inference (s) | Peak GiB | Checkpoint KiB |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in folds:
        for arm, values in (("control (Step 6A)", row["control_resources"]), ("challenger", row)):
            lines.append(f"| {row['provenance']['fold']} | {arm} | {values['training_seconds']:.2f} | {values['inference_seconds']:.2f} | "
                         f"{values['peak_working_set_bytes']/(1<<30):.2f} | {values['checkpoint_bytes']/1024:.2f} |")
    if folds:
        total = sum(row["training_seconds"] for row in folds)
        peak = max(row["peak_working_set_bytes"] for row in folds) / (1 << 30)
        error = max(row[k] for row in folds for k in ("control_replay_max_difference", "challenger_replay_max_difference"))
        lines += ["", f"Challenger training totaled {total:.2f} s, with {status['wall_seconds']:.2f} s command wall time and a maximum {peak:.2f} GiB worker peak. "
                  f"Maximum replay error across both arms' external scores was `{error:.3g}`."]
    lines += ["", "## Reproduce or resume", "", "The runner retains one epoch checkpoint per fold with the latest optimizer state and best weights. "
              "It reuses verified completed folds and limits training to 45 minutes and a 7.5 GB worker guard.", "", "```powershell",
              r"Set-Location E:\fraudlens-v3", r"$python = 'E:\fraudlens\.venv\Scripts\python.exe'",
              r"$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'", r"$dev = 'E:\fraudlens-v2\artifacts\v2'",
              r"$control = 'artifacts\v3\step6a-torch'", r"$out = 'artifacts\v3\step6b-rich'",
              r"& $python -B -m src.v3.torch_rich --development $dev --control $control --output $out",
              r"& $python -B -m src.v3.torch_rich_report --development $dev --control $control --output $out --markdown 'docs\V3_STEP6B.md'",
              "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    experiment_args(parser)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    root, control, out = paths(args)
    report = collect(root, control, out)
    image = args.markdown.with_name("v3_step6b_learning_curves.png")
    if report["folds"]:
        chart(report["folds"], image)
    write_json(out / "experiment_summary.json", report)
    args.markdown.write_text(markdown(report, image.name), encoding="utf-8", newline="\n")
    print(f"Reported {len(report['folds'])} folds. Promote richer input: {report['recommend_promotion']}.")


if __name__ == "__main__":
    main()
