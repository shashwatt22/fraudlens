"""Summarize already completed full-data Logistic Regression checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.v2.data import sha256, write_json
from .logistic_experiment import CUTOFF, METRICS, THREADS


ARMS = ("sampled", "full")
COLUMNS = (*METRICS, "recall_at_5pct", "recall_at_10pct")


def read_results(out):
    status = json.loads((out / "run_status.json").read_bytes())
    folds = []
    for index in (1, 2, 3):
        folder = out / f"fold{index}"
        if not (folder / "result.json").is_file():
            continue
        item = json.loads((folder / "result.json").read_bytes())
        if (item["fold"] != index or item["fit_rows"] != item["train_end"] or item["cutoff"] != CUTOFF
                or item["full_model_sha256"] != sha256(folder / "logistic_full.joblib")
                or item["full_scores_sha256"] != sha256(folder / "full_scores.npy")):
            raise ValueError(f"Fold {index} checkpoint is inconsistent.")
        folds.append(item)
    if [item["fold"] for item in folds] != status["verified_folds"]:
        raise ValueError("Worker status and completed fold checkpoints differ.")
    return folds, status


def arm_metrics(fold, arm):
    return {**{name: fold[arm][name] for name in METRICS},
            **{f"recall_at_{q}pct": fold["review_budgets"][arm][str(q)]["recall"] for q in (5, 10)}}


def summary(folds, status):
    means = {arm: {metric: float(np.mean([arm_metrics(fold, arm)[metric] for fold in folds]))
                   for metric in COLUMNS} for arm in ARMS} if folds else {}
    delta = {name: means["full"][name] - means["sampled"][name] for name in COLUMNS} if folds else {}
    return {"scope": "V2 expanding development folds", "folds": folds, "status": status,
            "means": means, "deltas": delta,
            "relative_ap_change": delta["average_precision"] / means["sampled"]["average_precision"] if folds else None,
            "training_seconds": sum(fold["training_seconds"] for fold in folds),
            "solver_seconds": sum(fold["solver_seconds"] for fold in folds),
            "fold_worker_seconds": sum(fold["worker_seconds"] for fold in folds),
            "max_peak_working_set_bytes": max((fold["peak_working_set_bytes"] or 0 for fold in folds), default=0)}


def gib(value):
    return f"{value / (1 << 30):.2f}" if value else "—"


def numbers(values):
    return " | ".join(f"{values[name]:.5f}" for name in COLUMNS)


def markdown(report):
    folds, status = report["folds"], report["status"]
    sections = ["# FraudLens V3 Step 3A full-data Logistic Regression", ""]
    if len(folds) != 3:
        sections += [f"Completed {len(folds)} of the three prescribed folds. Constraint: {status['constraint']}.", ""]
    if folds:
        means = report["means"]
        sections += [f"Mean AP on the completed folds changed from **{means['sampled']['average_precision']:.5f}** to "
                     f"**{means['full']['average_precision']:.5f}**, a change of {report['deltas']['average_precision']:+.5f} "
                     f"({report['relative_ap_change']:+.2%} relative).", ""]
    sections += ["The input is the verified 497-column V2 feature matrix. This step excludes the 13 Step 2 features, "
                 "which remain in the separate completed experiment. The three validation windows contain 70,865, 70,865 and 70,864 rows. "
                 "The sampled and full models score identical validation rows and share the original evaluation cutoff of 0.5. "
                 "Review budgets rank the same rows, using source order to break ties and rounding up to the nearest row. "
                 "Summary metrics equally weight the folds.", "",
                 "## Preprocessing and solver", "",
                 "The archived `logistic_l2` and the full-data fit both use unweighted L2 (`C=0.08`), L-BFGS, an intercept, "
                 f"tolerance `1e-4` and a maximum of 1,000 iterations. The experiment limits native libraries to {THREADS} CPU threads where supported. "
                 "The same `C` preserves the configured regularization. With more rows, the penalty is smaller relative to the summed data loss.", "",
                 "Each full training prefix refits categorical frequencies, median imputation, ANOVA ranking of 192 numeric or missingness inputs, "
                 "scaling and the original one-hot encoder (`min_frequency=7`, `max_categories=96`). "
                 "Preprocessing values can change because they are learned again on the larger training set. "
                 "The implementation fits imputation and ranking in column batches, accumulates scaling moments by row batch, and builds one float64 CSR matrix for L-BFGS. "
                 "Validation applies the fitted transform without altering it. No raw, policy-block, or retrospective scores enter fitting or evaluation.", "",
                 "## Paired validation results", "",
                 "| Fold | Model | Fit rows | AP | ROC-AUC | Precision | Recall | F1 | Accuracy | Recall at 5% | Recall at 10% |",
                 "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for fold in folds:
        for arm in ARMS:
            rows = fold["sampled_fit_rows"] if arm == "sampled" else fold["fit_rows"]
            sections.append(f"| {fold['fold']} | {arm} | {rows:,} | {numbers(arm_metrics(fold, arm))} |")
    if folds:
        for arm in ARMS:
            sections.append(f"| Mean | {arm} | — | {numbers(report['means'][arm])} |")
        sections.append(f"| Mean change | full − sampled | — | {numbers(report['deltas'])} |")
    sections += ["", "## Convergence and resources", "",
                 "Training time includes preprocessing, sparse matrix construction and the solver. Peak memory is the fold worker's Windows process working set. "
                 "The supervisor runs one worker at a time under a 30-minute wall-clock budget, with a 7.5 GB memory guard.", "",
                 "| Fold | Fit fraud rows | CSR shape | Training (s) | Solver (s) | Validation (s) | Peak GiB | L-BFGS iterations | Converged |",
                 "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for fold in folds:
        rows, columns = fold['csr_shape']
        sections.append(f"| {fold['fold']} | {fold['fit_fraud_rows']:,} | {rows:,} × {columns:,} | "
                        f"{fold['training_seconds']:.2f} | {fold['solver_seconds']:.2f} | {fold['inference_seconds']:.2f} | "
                        f"{gib(fold['peak_working_set_bytes'])} | {fold['iterations']} | {'Yes' if fold['converged'] else 'No'} |")
    if folds:
        parity = max(fold["old_preprocessing_parity"]["maximum_difference"] for fold in folds)
        sections += ["", f"Total measured training time was {report['training_seconds']:.2f} s. The fold workers took {report['fold_worker_seconds']:.2f} s "
                     "including control verification and checkpointing. Each sampled control score array matched its saved model's predictions exactly. "
                     f"Fitting the chunked preprocessing on each original 16,000-row sample reproduced the saved feature choices and transformation with maximum probe difference `{parity:.3g}`. "
                     "Reloading each full model and preprocessor reproduced its validation probe scores exactly."]
    sections += ["", "## Reproduce or resume", "",
                 "Each completed fold has an independent fitted preprocessor, model, score array, and metric manifest. Intact folds are reused when the same command resumes. "
                 "The artifact directory stays local and is ignored by Git. A new output path gives a clean reproduction.", "",
                 "```powershell", r"Set-Location E:\fraudlens-v3", r"$python = 'E:\fraudlens\.venv\Scripts\python.exe'",
                 r"$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'", "$env:PYTHONIOENCODING = 'utf-8'",
                 r"$out = 'artifacts\v3\step3a-logistic'",
                 r"& $python -B -m src.v3.logistic_experiment --development 'E:\fraudlens-v2\artifacts\v2' --output $out",
                 r"& $python -B -m src.v3.logistic_report --output $out --markdown 'docs\V3_STEP3A.md'", "```", ""]
    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    report = summary(*read_results(args.output))
    write_json(args.output / "experiment_summary.json", report)
    if args.markdown:
        args.markdown.write_text(markdown(report), encoding="utf-8", newline="\n")
    print(f"Summarized {len(report['folds'])} completed folds.")


if __name__ == "__main__":
    main()
