"""Audit saved V3 Random Forest folds and summarize the paired comparison."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from src.v2.data import sha256, write_json
from .diagnostics import development_inputs
from .forest_experiment import NAME, SCORE_ATOL, score_partition
from .logistic_experiment import CUTOFF, THREADS
from .logistic_report import ARMS, COLUMNS, arm_metrics, gib


def verified_folds(root, out, status):
    matrices, _, _, plan = development_inputs(root)
    folds = []
    with threadpool_limits(limits=1):
        for index in status["verified_folds"]:
            folder = out / f"fold{index}"
            record = json.loads((folder / "result.json").read_bytes())
            model = folder / "forest_full.joblib"
            scores = folder / "full_scores.npy"
            control = root / "challengers" / NAME / f"scores{index}.npy"
            fold = plan[index - 1]
            if (record["fold"] != index or record["fit_rows"] != fold.train_end or record["validation_end"] != fold.validation_end
                    or sha256(model) != record["full_model_sha256"] or sha256(scores) != record["full_scores_sha256"]
                    or sha256(control) != record["control_scores_sha256"] or model.stat().st_size != record["checkpoint_bytes"]):
                raise ValueError(f"Checkpoint, control or row boundaries differ in fold {index}.")
            payload = joblib.load(model)
            if (len(payload["model"].estimators_) != 140 or payload["train_rows"] != fold.train_end
                    or payload["model"].get_params() != record["settings"]):
                raise ValueError(f"Invalid restored forest in fold {index}.")
            expected = np.load(scores, allow_pickle=False)
            actual = score_partition(payload["model"], payload["transform"], matrices[3][fold.train_end:fold.validation_end])
            np.testing.assert_allclose(actual, expected, rtol=0, atol=SCORE_ATOL)
            record["full_reload_max_score_difference"] = float(np.max(np.abs(actual - expected)))
            del payload
            folds.append(record)
            print(f"Fold {index}: verified all {len(actual):,} reloaded validation scores.", flush=True)
    return folds


def summarize(folds, status):
    def mean(values):
        return float(np.mean(values)) if all(value is not None for value in values) else None

    means = {arm: {metric: mean([arm_metrics(fold, arm)[metric] for fold in folds]) for metric in COLUMNS}
             for arm in ARMS} if folds else {}
    delta = {metric: (means["full"][metric] - means["sampled"][metric]
                      if None not in (means["full"][metric], means["sampled"][metric]) else None)
             for metric in COLUMNS} if folds else {}
    resource_means = {"sampled": {}, "full": {}}
    if folds:
        for name, archive_name in (("training_seconds", "fit_seconds"), ("inference_seconds", "inference_seconds"),
                                   ("peak_working_set_bytes", "process_peak_working_set"),
                                   ("checkpoint_bytes", "checkpoint_bytes")):
            resource_means["sampled"][name] = mean([fold["archived"][archive_name] for fold in folds])
            resource_means["full"][name] = mean([fold[name] for fold in folds])
        resource_means["sampled"]["fit_rows"] = mean([fold["archived"]["fit_rows"] for fold in folds])
        resource_means["full"]["fit_rows"] = mean([fold["fit_rows"] for fold in folds])
    return {"scope": "V2 chronological development folds", "folds": folds, "status": status,
            "means": means, "deltas": delta, "resource_means": resource_means,
            "relative_ap_change": delta["average_precision"] / means["sampled"]["average_precision"] if folds else None,
            "training_seconds": sum(fold["training_seconds"] for fold in folds),
            "tree_fit_seconds": sum(fold["forest_fit_seconds"] for fold in folds),
            "worker_seconds": sum(fold["worker_seconds"] for fold in folds),
            "max_peak_working_set_bytes": max((fold["peak_working_set_bytes"] or 0 for fold in folds), default=0)}


def numbers(values):
    return " | ".join("—" if values[name] is None else f"{values[name]:.5f}" for name in COLUMNS)


def mib(size):
    return f"{size / (1 << 20):.2f}"


def markdown(report):
    folds, status, means = (report[name] for name in ("folds", "status", "means"))
    lines = ["# FraudLens V3 Step 4A full-data Random Forest", ""]
    if len(folds) < 3:
        lines += [f"Completed {len(folds)} of three folds. {status['constraint'] or 'Training stopped before all folds completed.'}", ""]
    if folds:
        lines += [f"Mean AP on the completed folds changed from **{means['sampled']['average_precision']:.5f}** to "
                  f"**{means['full']['average_precision']:.5f}**, a change of {report['deltas']['average_precision']:+.5f} "
                  f"({report['relative_ap_change']:+.2%} relative).", ""]
    lines += ["Step 3A’s full-data Logistic Regression stays the project baseline. This experiment compares two Random Forest training populations. "
              "The source has 497 V2 predictors. The rejected Step 2 features remain in their archived experiment. "
              "Both Random Forest models score identical chronological validation rows. Development contains all data used by this comparison.", "",
              "Both arms use 140 bootstrap trees, Gini impurity, depth 22, leaf minimum 3, square-root feature sampling, seed 42 and no class weights. "
              f"The full-data run retained all classifier settings, including {THREADS} tree workers, with one native library thread per worker to avoid nested parallelism.", "",
              "For each fold, the full training prefix refits category frequencies, median missing-value fills, ANOVA rankings and scaling. "
              "The transformation selects 180 numeric and missingness inputs and delivers the original dense float32 representation. "
              "Column and row batches bound preprocessing temporaries. Validation uses the fitted transform. The original 16,000-row sample was recreated "
              "solely for preprocessing parity checks. The original fitted classifier and scores supply the comparator.", "",
              "## Paired validation metrics", "",
              f"Decisions use cutoff `{CUTOFF}`. Review budgets use the highest-scored 5% or 10% of each fold, rounded up with source order breaking ties. "
              "Means give equal weight to each completed fold.", "",
              "| Fold | Arm | Actual training rows | AP | ROC-AUC | Precision | Recall | F1 | Accuracy | Recall at 5% | Recall at 10% |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for fold in folds:
        for arm in ARMS:
            rows = fold['archived']['fit_rows'] if arm == 'sampled' else fold['fit_rows']
            lines.append(f"| {fold['fold']} | {arm} | {rows:,} | {numbers(arm_metrics(fold, arm))} |")
    if folds:
        for arm in ARMS:
            resource = report["resource_means"][arm]
            lines.append(f"| Mean | {arm} | {resource['fit_rows']:,.0f} | {numbers(means[arm])} |")
        lines += [f"| Mean change | full − sampled | — | {numbers(report['deltas'])} |", "",
                  "| Fold | Change in AP | Relative AP change |", "| --- | ---: | ---: |"]
        for fold in folds:
            before, after = (fold[arm]["average_precision"] for arm in ARMS)
            lines.append(f"| {fold['fold']} | {after - before:+.5f} | {(after - before) / before:+.2%} |")
    lines += ["", "## Runtime, memory and model size", "",
              "Training includes preprocessing and the tree fits. Model sizes are actual compressed Joblib files including the fitted preprocessing. "
              "Working set is the peak for each process. Archived V2 process peaks can include earlier trials in that process.", "",
              "| Fold | Arm | Training (s) | Validation scoring (s) | Peak GiB | Checkpoint MiB |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for fold in folds:
        archive = fold["archived"]
        lines.append(f"| {fold['fold']} | sampled (archived) | {archive['fit_seconds']:.2f} | {archive['inference_seconds']:.2f} | "
                     f"{gib(archive['process_peak_working_set'])} | {mib(archive['checkpoint_bytes'])} |")
        lines.append(f"| {fold['fold']} | full | {fold['training_seconds']:.2f} | {fold['inference_seconds']:.2f} | "
                     f"{gib(fold['peak_working_set_bytes'])} | "
                     f"{mib(fold['checkpoint_bytes'])} |")
    if folds:
        for arm in ARMS:
            stats = report["resource_means"][arm]
            lines.append(f"| Mean | {arm} | {stats['training_seconds']:.2f} | {stats['inference_seconds']:.2f} | "
                         f"{gib(stats['peak_working_set_bytes'])} | {mib(stats['checkpoint_bytes'])} |")
        old_error = max(fold["archived"]["max_replayed_score_difference"] for fold in folds)
        fit_error = max(fold["archived"]["preprocessing_probe_max_difference"] for fold in folds)
        replay_error = max(fold["full_reload_max_score_difference"] for fold in folds)
        lines += ["", f"Full training took {report['training_seconds']:.2f} s total, including {report['tree_fit_seconds']:.2f} s in tree fitting. "
                  f"The largest working set was {gib(report['max_peak_working_set_bytes'])} GiB. The supervised run took {status['wall_seconds']:.2f} s "
                  "including archived-score checks, inference, and checkpointing.", "",
                  f"The sampled classifier replay matched all stored scores within `{old_error:.3g}`. Its preprocessing probe matched within `{fit_error:.3g}`. "
                  f"Reloading the full checkpoints reproduced the complete validation predictions within `{replay_error:.3g}`."]
    lines += ["", "## Reproduce or resume", "",
              "The runner fits one fold at a time with a 30-minute wall-clock budget and a 7.5 GB working-set guard. "
              "It verifies and reuses completed folds at the same output path. Use a fresh output path for a new independent run.", "",
              "```powershell", r"Set-Location E:\fraudlens-v3", r"$python = 'E:\fraudlens\.venv\Scripts\python.exe'",
              r"$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'", "$env:OMP_NUM_THREADS = '1'",
              "$env:OPENBLAS_NUM_THREADS = '1'", "$env:MKL_NUM_THREADS = '1'",
              r"$development = 'E:\fraudlens-v2\artifacts\v2'", r"$out = 'artifacts\v3\step4a-forest'",
              r"& $python -B -m src.v3.forest_experiment --development $development --output $out",
              r"& $python -B -m src.v3.forest_report --development $development --output $out --markdown 'docs\V3_STEP4A.md'",
              "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    root, out = args.development.resolve(), args.output.resolve()
    if out.is_relative_to(root) or "retrospective" in {part.lower() for part in root.parts}:
        raise ValueError("Use V2 development inputs and a separate output directory.")
    status = json.loads((out / "run_status.json").read_bytes())
    report = summarize(verified_folds(root, out, status), status)
    write_json(out / "experiment_summary.json", report)
    if args.markdown:
        args.markdown.write_text(markdown(report), encoding="utf-8", newline="\n")
    print(f"Reported {len(report['folds'])} completed Random Forest folds.")


if __name__ == "__main__":
    main()
