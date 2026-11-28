"""Verify and report completed V3 full-data MLP folds without fitting models."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.v2.data import sha256, write_json
from .logistic_experiment import METRICS
from .torch_full import CUTOFF, CANDIDATE, EPOCHS, PATIENCE, BATCH, SEED, THREADS


ARMS = ("sampled", "full")
COLUMNS = (*METRICS, "recall_at_5pct", "recall_at_10pct")


def validation(row, arm):
    return {**{metric: row["external"][arm][metric] for metric in METRICS},
            **{f"recall_at_{q}pct": row["review_budgets"][arm][str(q)]["recall"] for q in (5, 10)}}


def summarize(out):
    status = json.loads((out / "run_status.json").read_bytes())
    folds = []
    for i in status["verified_folds"]:
        directory = out / f"fold{i}"
        row = json.loads((directory / "result.json").read_bytes())
        if (row["provenance"]["fold"] != i
                or any(sha256(directory / name) != checksum for name, checksum in row["artifact_sha256"].items())
                or row["best_epoch"] != max(row["history"], key=lambda record: record["internal_validation_ap"])["epoch"]):
            raise ValueError(f"Fold {i} checkpoint or selected epoch does not match the manifest.")
        folds.append(row)
    means = {arm: {metric: float(np.mean([validation(row, arm)[metric] for row in folds])) for metric in COLUMNS}
             for arm in ARMS} if folds else {}
    return {"scope": "V2 chronological development folds", "cutoff": CUTOFF, "folds": folds, "status": status, "means": means,
            "deltas": {metric: means["full"][metric] - means["sampled"][metric] for metric in COLUMNS} if folds else {},
            "total_full_training_seconds": sum(row["training_seconds"] for row in folds),
            "max_peak_working_set_bytes": max((row["peak_working_set_bytes"] or 0 for row in folds), default=0)}


def plot_curves(folds, path):
    figure, axes = plt.subplots(len(folds), 2, figsize=(10, 3.1 * len(folds)), squeeze=False, layout="constrained")
    for row, (loss, ap) in zip(folds, axes, strict=True):
        fold = row["provenance"]["fold"]
        full, sampled = row["history"], row["archived"]["history"]["epochs"]
        for records, field, caption, color, style in (
            (full, "train_loss", "Full training", "#075e79", "-"),
            (full, "internal_validation_loss", "Full early-stop", "#b24114", "-"),
            (sampled, "train_loss", "Sampled training", "#515b6a", "--"),
        ):
            loss.plot([r["epoch"] for r in records], [r[field] for r in records], label=caption, color=color, linestyle=style, marker=".")
        for records, caption, color, style in ((full, "Full", "#075e79", "-"), (sampled, "Sampled", "#515b6a", "--")):
            ap.plot([r["epoch"] for r in records], [r["internal_validation_ap"] for r in records],
                    label=caption, color=color, linestyle=style, marker=".")
        for axis, label in ((loss, "Binary cross-entropy"), (ap, "Average precision")):
            axis.set(title=f"Fold {fold}", xlabel="Epoch", ylabel=label)
            axis.axvline(row["best_epoch"], color="#075e79", linestyle=":", linewidth=1, label="Full best epoch")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    figure.suptitle("FraudLens V3 PyTorch learning curves · chronological development folds")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def cells(row, keys):
    return " | ".join(f"{row[key]:.5f}" if row[key] is not None else "—" for key in keys)


def markdown(report, curve_name):
    folds, means, delta = (report[key] for key in ("folds", "means", "deltas"))
    status = report["status"]
    lines = ["# FraudLens V3 Step 6A full-data PyTorch MLP", ""]
    if len(folds) != 3:
        lines.extend((f"Completed {len(folds)} of three folds. {status['constraint'] or 'Run ended early.'}", ""))
    if folds:
        lines.extend((f"Mean external validation AP moved from **{means['sampled']['average_precision']:.5f}** to "
                      f"**{means['full']['average_precision']:.5f}** ({delta['average_precision']:+.5f}, "
                      f"{delta['average_precision']/means['sampled']['average_precision']:+.2%} relative).", ""))
    lines.extend(("## Frozen design", "",
                  "The experiment uses the original 497-column V2 feature cache, excluding the archived Step 2 feature block. The full model refits the original frequency encoding, median imputation, missing indicators, ANOVA selection and scaling on gradient rows only. "
                  "The selected 224 input columns are materialized in float32 batches. The final chronological tenth of each training prefix, adjusted to keep timestamps together, selects the checkpoint. The next chronological external fold is evaluated separately.", "",
                  f"The architecture is `224 → 192 → 96 → 48 → 1`, with Linear → BatchNorm1d → GELU → Dropout(0.18) for every hidden layer and a final linear logit. "
                  f"It uses unweighted BCEWithLogitsLoss and AdamW (initial learning rate {CANDIDATE.options['rate']}, weight decay {CANDIDATE.options['weight_decay']}), "
                  f"cosine annealing over {EPOCHS} epochs, shuffled batches up to {BATCH}, seed {SEED}, six CPU threads, and AP patience {PATIENCE}. "
                  "Selection uses the first epoch attaining the highest internal AP. Full-data epochs contain more optimizer steps. Internal scoring is batched and logs evaluation-mode BCE in addition to AP. "
                  "The original V2 checkpoints saved training loss and internal AP each epoch, with no archived internal loss curve.", "",
                  "| Fold | Sampled gradient rows | Full gradient rows | Internal early-stop rows | External rows | Inputs | Parameters | Best epoch sampled/full | Epochs sampled/full |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"))
    for row in folds:
        src, archive = row["provenance"], row["archived"]
        old = archive["history"]
        lines.append(f"| {src['fold']} | {archive['gradient_rows']:,} | {src['gradient_rows']:,} | {src['early_stopping_rows']:,} | "
                     f"{src['validation_end']-src['train_end']:,} | {row['input_dim']} | {row['parameters']:,} | "
                     f"{old['best_epoch']}/{row['best_epoch']} | {len(old['epochs'])}/{len(row['history'])} |")
    lines.extend(("", "## External validation", "", "Means weight each fold equally. Threshold-based metrics use 0.5. The two review budgets rank the same rows with source-order tie-breaking.", "",
                  "| Fold | Arm | AP | ROC-AUC | Precision | Recall | F1 | Accuracy | Recall at 5% | Recall at 10% |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"))
    for row in folds:
        for arm in ARMS:
            lines.append(f"| {row['provenance']['fold']} | {arm} | {cells(validation(row, arm), COLUMNS)} |")
    for arm in means:
        lines.append(f"| Mean | {arm} | {cells(means[arm], COLUMNS)} |")
    if folds:
        lines.append(f"| Mean change | full − sampled | {cells(delta, COLUMNS)} |")
    lines.extend(("", "## Training versus external validation", "", "Training metrics score each model on its own gradient-training population at its selected epoch. Those populations differ, and the gap includes temporal shift.", "",
                  "| Fold | Arm | Train AP | External AP | Train ROC-AUC | External ROC-AUC |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"))
    for row in folds:
        for arm in ARMS:
            tr, val = row["train"][arm], row["external"][arm]
            lines.append(f"| {row['provenance']['fold']} | {arm} | {tr['average_precision']:.5f} | {val['average_precision']:.5f} | "
                         f"{tr['roc_auc']:.5f} | {val['roc_auc']:.5f} |")
    for arm in means:
        train_ap, train_auc = (float(np.mean([row['train'][arm][metric] for row in folds])) for metric in ("average_precision", "roc_auc"))
        lines.append(f"| Mean | {arm} | {train_ap:.5f} | {means[arm]['average_precision']:.5f} | {train_auc:.5f} | {means[arm]['roc_auc']:.5f} |")
    if folds:
        lines.extend(("", "## Learning curves", "", f"![Training loss, internal validation loss and internal AP by epoch]({curve_name})", "",
                      "Epoch-level values, learning rates and elapsed times are in the saved JSON."))
    lines.extend(("", "## Seen and unseen entity proxies", "", "A complete card–address–D1 key is seen if it occurred before the gradient cut. The table applies that same definition to both models on identical rows. The JSON also reports each model’s metrics using its actual fitted keys, including the smaller sampled key set.", "",
                  "| Fold | Group | External rows | Fraud prevalence | Sampled AP | Full AP |", "| --- | --- | ---: | ---: | ---: | ---: |"))
    for row in folds:
        for name, pair in row["entity_common_full_fit"].items():
            a, b = (pair[arm] for arm in ARMS)
            lines.append(f"| {row['provenance']['fold']} | {name.replace('_',' ')} | {a['rows']:,} | {a['fraud_rate']:.2%} | "
                         f"{a['average_precision']:.5f} | {b['average_precision']:.5f} |")
    for name in (folds[0]["entity_common_full_fit"] if folds else ()):
        a, b = (np.mean([row["entity_common_full_fit"][name][arm]["average_precision"] for row in folds]) for arm in ARMS)
        lines.append(f"| Mean | {name.replace('_',' ')} | — | — | {a:.5f} | {b:.5f} |")
    lines.extend(("", "## Resources and verification", "", "Training includes preprocessing, gradient updates and internal scoring. Peak memory is process working set. Checkpoint size includes the model and fitted transform. The resumable epoch state also saves optimizer and RNG state separately.", "",
                  "| Fold | Arm | Training (s) | External inference (s) | Peak GiB | Checkpoint KiB |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"))
    def resources(row):
        return (f"{row.get('training_seconds', row.get('fit_seconds')):.2f} | {row['inference_seconds']:.2f} | "
                f"{row['peak_working_set_bytes']/(1<<30):.2f} | {row['checkpoint_bytes']/1024:.2f}")
    for row in folds:
        for arm, measured in (("sampled (archived)", row["archived"]), ("full", row)):
            lines.append(f"| {row['provenance']['fold']} | {arm} | {resources(measured)} |")
    if folds:
        lines.extend(("", f"Full-data training took {report['total_full_training_seconds']:.2f} s. The supervised command took {status['wall_seconds']:.2f} s. "
                      f"Maximum full worker peak was {report['max_peak_working_set_bytes']/(1<<30):.2f} GiB. "
                      f"The largest reload difference across full external scores was `{max(row['reload_max_score_difference'] for row in folds):.3g}` "
                      f"and across archived control scores was `{max(row['control_max_score_difference'] for row in folds):.3g}`. "
                      f"The sampled preprocessing parity probe differed by at most `{max(row['control_preprocessing_max_difference'] for row in folds):.3g}`."))
    lines.extend(("", "## Reproduce or resume", "", "The runner fits one fold at a time within a 45-minute wall-clock budget with a 7.5 GB worker memory guard. It verifies complete folds, and an interrupted fold resumes from its last recorded epoch. Use a fresh output directory for an independent run.", "",
                  "```powershell", r"Set-Location E:\fraudlens-v3", r"$python = 'E:\fraudlens\.venv\Scripts\python.exe'",
                  r"$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'", r"$dev = 'E:\fraudlens-v2\artifacts\v2'",
                  r"$out = 'artifacts\v3\step6a-torch'", r"& $python -B -m src.v3.torch_full --development $dev --output $out",
                  r"& $python -B -m src.v3.torch_report --output $out --markdown 'docs\V3_STEP6A.md'",
                  r"& $python -B -m pytest -q tests/test_v3_torch_full.py", "```", ""))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.output)
    write_json(args.output / "experiment_summary.json", report)
    figure = args.markdown.with_name("v3_step6a_learning_curves.png")
    if report["folds"]:
        plot_curves(report["folds"], figure)
    args.markdown.write_text(markdown(report, figure.name), encoding="utf-8", newline="\n")
    print(f"Reported {len(report['folds'])} completed neural-network folds.")


if __name__ == "__main__":
    main()
