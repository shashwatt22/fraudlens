"""Summarize and resample paired development scores from the completed V3 experiment."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.v2.data import sha256, write_json
from src.train import SEED
from .behavior import FEATURES, anchor_keys
from .behavior_experiment import ROUNDS, THREADS, THRESHOLD, load_inputs
from .diagnostics import XGB


BOOTSTRAPS = 1000
METRICS = ("average_precision", "roc_auc", "precision", "recall", "f1")
ARMS = ("control", "treatment")


class RankedAP:
    """Cache the score order and tie boundaries for weighted average precision."""

    def __init__(self, labels, scores, blocks):
        order = np.argsort(-scores, kind="stable")
        sorted_scores = scores[order]
        self.ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores) - 1]
        self.blocks = blocks[order]
        self.labels = labels[order]

    def __call__(self, weights):
        weight = weights[self.blocks]
        positives = np.cumsum(weight * self.labels)[self.ends]
        if positives[-1] == 0:
            return np.nan
        totals = np.cumsum(weight)[self.ends]
        precision = np.divide(positives, totals, out=np.zeros(len(totals)), where=totals > 0)
        return float((np.diff(positives, prepend=0) * precision).sum() / positives[-1])


def entity_blocks(keys):
    """Complete keys form shared blocks. Each incomplete row gets its own block."""
    keys = np.asarray(keys)
    missing = keys < 0
    _, known = np.unique(keys[~missing], return_inverse=True)
    count = int(known.max(initial=-1)) + 1
    blocks = np.empty(len(keys), dtype=np.int64)
    blocks[~missing] = known
    blocks[missing] = count + np.arange(np.count_nonzero(missing))
    return blocks, count, int(np.count_nonzero(missing))


def compare_bootstrap(folds, replicates=BOOTSTRAPS, seed=42):
    """Draw one set of entity multiplicities per replicate, shared across both arms and folds."""
    started = perf_counter()
    blocks, known, missing = entity_blocks(np.concatenate([key for _, _, _, key in folds]))
    count = known + missing
    readers = []
    start = 0
    for labels, control, treatment, _ in folds:
        stop = start + len(labels)
        readers.append([RankedAP(labels, scores, blocks[start:stop]) for scores in (control, treatment)])
        start = stop
    rng = np.random.default_rng(seed)
    results = np.empty((replicates, len(folds)))
    for i in range(replicates):
        weight = np.bincount(rng.integers(count, size=count), minlength=count)
        results[i] = [treatment(weight) - control(weight) for control, treatment in readers]
    finite = results[np.isfinite(results).all(axis=1)]
    if len(finite) == 0:
        raise ValueError("No bootstrap replicate included both arms with defined AP in every fold.")

    def interval(values):
        low, high = np.quantile(values, (.025, .975))
        return {"lower": float(low), "upper": float(high)}

    return {"replicates": replicates, "valid_replicates": len(finite), "seed": seed, "entity_blocks": known,
            "missing_key_singleton_blocks": missing, "seconds": perf_counter() - started,
            "mean_ap_delta_ci95": interval(finite.mean(axis=1)),
            "fold_ap_delta_ci95": [interval(finite[:, i]) for i in range(len(folds))]}


def read_completed(root, out, matrices, labels, folds):
    _, advanced, _, data = matrices
    keys = anchor_keys(data[:folds[-1].validation_end], advanced["features"])
    reports, inputs = [], []
    for index, fold in enumerate(folds, 1):
        folder = out / f"fold{index}"
        item = json.loads((folder / "result.json").read_bytes())
        control = root / "challengers" / XGB / f"scores{index}.npy"
        archived = json.loads((control.parent / f"fold{index}.json").read_bytes())
        treatment = folder / "treatment_scores.npy"
        model = folder / "treatment.ubj"
        start, stop = fold.train_end, fold.validation_end
        if (item["fold"] != index or item["train_end"] != start or item["validation_end"] != stop
                or item["feature_manifest_sha256"] != sha256(out / "feature_block.json")
                or item["control_scores_sha256"] != sha256(control)
                or item["treatment_scores_sha256"] != sha256(treatment)
                or item["treatment_model_sha256"] != sha256(model)):
            raise ValueError(f"Fold {index} is incomplete or differs from its completed records.")
        inputs.append((labels[start:stop], np.load(control, allow_pickle=False),
                       np.load(treatment, allow_pickle=False), keys[start:stop]))
        item["control_archived_inference_seconds"] = archived["inference_seconds"]
        item["control_archived_peak_working_set_bytes"] = archived["process_peak_working_set"]
        reports.append(item)
    return reports, inputs


def summarize(feature, folds, inputs):
    means = {arm: {name: float(np.mean([fold[arm][name] for fold in folds])) for name in METRICS} for arm in ARMS}
    for arm in ARMS:
        for budget in (5, 10):
            means[arm][f"recall_at_{budget}pct"] = float(np.mean([f["review_budgets"][arm][str(budget)]["recall"] for f in folds]))
    delta = {name: means["treatment"][name] - means["control"][name] for name in means["control"]}
    better_folds = sum(fold["treatment"]["average_precision"] > fold["control"]["average_precision"] for fold in folds)
    checks = {"mean_ap_gain_at_least_0.01": delta["average_precision"] >= .01,
              "at_least_two_ap_gains": better_folds >= 2, "mean_auc_loss_at_most_0.01": delta["roc_auc"] >= -.01}
    importance = {name: {"splits": int(sum(fold["importance"]["weight"][name] for fold in folds)),
                         "total_gain": sum(fold["importance"]["total_gain"][name] for fold in folds)} for name in FEATURES}
    return {"folds": folds, "means": means, "deltas": delta, "feature_block": feature,
            "improved_ap_folds": better_folds, "promotion_checks": checks, "recommend_promotion": all(checks.values()),
            "uncertainty": compare_bootstrap(inputs), "importance_across_folds": importance,
            "mean_new_feature_gain_share": float(np.mean([f["importance"]["share_of_total_gain"] for f in folds])),
            "fit_total_seconds": sum(fold["fit_seconds"] for fold in folds),
            "inference_total_seconds": sum(fold["inference_seconds"] for fold in folds),
            "fold_worker_total_seconds": sum(fold["wall_seconds"] for fold in folds),
            "max_fit_and_score_peak_working_set_bytes": max(f["treatment_fit_and_score_peak_working_set_bytes"] or 0 for f in folds),
            "max_fold_worker_peak_working_set_bytes": max(f["worker_peak_working_set_bytes"] or 0 for f in folds)}


def markdown(report):
    folds, means, delta = (report[key] for key in ("folds", "means", "deltas"))
    boot = report["uncertainty"]
    feature = report["feature_block"]
    recommendation = "Promote the treatment." if report["recommend_promotion"] else "Keep the V2 control."
    lines = ["# FraudLens V3 Step 2 behavioral feature experiment", "",
             f"**{recommendation}** Mean AP changed by {delta['average_precision']:+.5f}, with gains in "
             f"{report['improved_ap_folds']} of three folds. Mean ROC-AUC changed by {delta['roc_auc']:+.5f}. "
             "The prespecified promotion rule requires a mean AP gain of at least 0.01, AP gains in at least two folds, and an AUC decline of at most 0.01.",
             "", "The read-only V2 control was verified by reloading every checkpoint and matching every saved development score exactly. "
             f"The sole treatment adds {len(FEATURES)} numeric predictors to the original 497 columns and uses the same {ROUNDS}-round, depth-8 XGBoost fit, "
             f"seed {SEED}, {THREADS} CPU threads, training-only category handling and three chronological folds. "
             "The training prefixes contain 212,594, 283,459 and 354,324 rows. Validation contains 70,865, 70,865 and 70,864 rows. The experiment ends after fold 3's validation block. "
             "The later decision-policy block and retrospective holdout were outside the experiment.",
             "", "The entity remains `card1`, `addr1` and `round(day_minus_D1)`. The amount window includes `t − 604800` and excludes all of timestamp `t`. "
             "D-offset summaries use all earlier timestamps for that entity. Counts include rows with missing amounts. Means and population standard deviations use finite values only, with standard deviation zero for a single value. "
             "No history yields a zero seven-day count and missing moments. An incomplete key yields all missing values. A deviation needs both a current value and a prior mean.",
             "", "## Rankings and decisions", "",
             f"Decisions use the original cutoff `{THRESHOLD}`. Review budgets use the top-scored 5% and 10% of each fold, rounded up with original order breaking ties. "
             "The mean equally weights the three folds.", "",
             "| Fold | Arm | AP | ROC-AUC | Precision | Recall | F1 | Recall at 5% | Recall at 10% |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for fold in folds:
        for arm in ARMS:
            row = fold[arm]
            budgets = fold["review_budgets"][arm]
            values = [*[row[m] for m in METRICS], budgets["5"]["recall"], budgets["10"]["recall"]]
            lines.append(f"| {fold['fold']} | {arm} | " + " | ".join(f"{x:.5f}" for x in values) + " |")
    for arm in ARMS:
        row = means[arm]
        values = (row[name] for name in (*METRICS, "recall_at_5pct", "recall_at_10pct"))
        lines.append(f"| Mean | {arm} | " + " | ".join(f"{x:.5f}" for x in values) + " |")
    lines += ["", "## Paired entity-block uncertainty", "",
              f"The {boot['valid_replicates']:,} valid bootstrap replicates used the same resampled block weights for both scorers and all folds. "
              f"There were {boot['entity_blocks']:,} complete entity blocks and {boot['missing_key_singleton_blocks']:,} incomplete-key singleton blocks. "
              "Intervals describe uncertainty from these development observations with fixed model fits. They do not measure variation across new time periods or refits.", "",
              "| Fold | Treatment minus control AP | 95% percentile interval |", "| --- | ---: | --- |"]
    for fold, interval in zip(folds, boot["fold_ap_delta_ci95"], strict=True):
        gain = fold['treatment']['average_precision'] - fold['control']['average_precision']
        lines.append(f"| {fold['fold']} | {gain:+.5f} | [{interval['lower']:+.5f}, {interval['upper']:+.5f}] |")
    interval = boot["mean_ap_delta_ci95"]
    lines.append(f"| Mean | {delta['average_precision']:+.5f} | [{interval['lower']:+.5f}, {interval['upper']:+.5f}] |")
    lines += ["", "## Entity and history strata", "",
              "Seen and unseen use complete keys and membership in that fold's training prefix. Prior history uses the unchanged V2 strictly prior count, so unseen keys can gain history during validation. Groups across the two definitions overlap. The table shows row counts, prevalence and AP. The JSON includes AUC and all cutoff metrics by group.", "",
              "| Fold | Group | Rows | Fraud rate | Control AP | Treatment AP |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for fold in folds:
        for group, arms in fold["groups"].items():
            before, after = (arms[arm] for arm in ARMS)
            lines.append(f"| {fold['fold']} | {group.replace('_', ' ')} | {before['rows']:,} | {before['fraud_rate']:.2%} | "
                         f"{before['average_precision']:.5f} | {after['average_precision']:.5f} |")
    lines += ["", "## New feature importance", "",
              f"New columns account for {report['mean_new_feature_gain_share']:.2%} of total training split gain on average. Split importance describes how the fitted models used the columns. Validation metrics measure whether adding them improved ranking.", "",
              "| New feature | Splits across folds | Total split gain |", "| --- | ---: | ---: |"]
    order = sorted(report["importance_across_folds"].items(), key=lambda item: item[1]["total_gain"], reverse=True)
    lines.extend(f"| `{name}` | {value['splits']:,} | {value['total_gain']:,.1f} |" for name, value in order)
    lines += ["", "## Runtime and memory", "",
              f"The {feature['rows']:,}-row float32 feature block took {feature['seconds']:.2f} s to build. "
              "Existing V2 columns and the added block are combined only in the current batch. Windows peak working set includes mapped pages and process overhead. A separate six-thread worker handles each fold.", "",
              "| Fold | Treatment fit (s) | Treatment inference (s) | Fit and score peak (GiB) | Worker peak (GiB) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for fold in folds:
        lines.append(f"| {fold['fold']} | {fold['fit_seconds']:.2f} | {fold['inference_seconds']:.2f} | "
                     f"{gib(fold['treatment_fit_and_score_peak_working_set_bytes'])} | {gib(fold['worker_peak_working_set_bytes'])} |")
    lines.append(f"| Total time / maximum memory | {report['fit_total_seconds']:.2f} | {report['inference_total_seconds']:.2f} | "
                 f"{gib(report['max_fit_and_score_peak_working_set_bytes'])} | {gib(report['max_fold_worker_peak_working_set_bytes'])} |")
    archived_fit = sum(f["control_archived_fit_seconds"] for f in folds)
    archived_score = sum(f["control_archived_inference_seconds"] for f in folds)
    archived_peak = max(f["control_archived_peak_working_set_bytes"] or 0 for f in folds)
    lines += ["", f"The three fold workers took {report['fold_worker_total_seconds']:.2f} s in total, including checkpointing and control verification. "
              f"The paired bootstrap took {boot['seconds']:.2f} s.", "",
              f"The archived V2 control run recorded {archived_fit:.2f} s of fitting, {archived_score:.2f} s of scoring and a {gib(archived_peak)} GiB process high-water mark. "
              "These measurements came from the original V2 run and its process scope.", "", "## Reproduce or resume", "",
              "This repository keeps the cache, model checkpoints, validation predictions and reports under `artifacts/v3/step2-behavior`. "
              "Rerunning an intact completed fold verifies and reuses its saved files. A fresh output directory starts a new treatment run.", "",
              "```powershell", r"Set-Location E:\fraudlens-v3", r"$python = 'E:\fraudlens\.venv\Scripts\python.exe'",
              r"$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'", "$env:PYTHONIOENCODING = 'utf-8'",
              "$env:OMP_NUM_THREADS = '6'",
              "$env:OPENBLAS_NUM_THREADS = '6'", "$env:MKL_NUM_THREADS = '6'",
              r"$dev = 'E:\fraudlens-v2\artifacts\v2'", r"$out = 'artifacts\v3\step2-behavior'",
              "foreach ($fold in 1..3) {", "    & $python -B -m src.v3.behavior_experiment --development $dev --output $out --fold $fold",
              "    if ($LASTEXITCODE -ne 0) { throw ('Failed fold ' + $fold) }", "}",
              r"& $python -B -m src.v3.behavior_report --development $dev --output $out --markdown 'docs\V3_STEP2.md'", "```", ""]
    return "\n".join(lines)


def gib(value):
    return f"{value / (1 << 30):.2f}" if value else "—"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    root, out, matrices, _, labels, plan = load_inputs(args.development, args.output)
    folds, inputs = read_completed(root, out, matrices, labels, plan)
    feature = json.loads((out / "feature_block.json").read_bytes())
    report = summarize(feature, folds, inputs)
    write_json(out / "experiment_summary.json", report)
    if args.markdown:
        if args.markdown.resolve().is_relative_to(root):
            raise ValueError("Keep the Markdown report separate from the V2 development inputs.")
        args.markdown.write_text(markdown(report), encoding="utf-8", newline="\n")
    print(f"Mean AP {report['means']['control']['average_precision']:.5f} to {report['means']['treatment']['average_precision']:.5f}. "
          f"Recommend promotion: {report['recommend_promotion']}.")


if __name__ == "__main__":
    main()
