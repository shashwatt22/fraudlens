"""Verify frozen refits and score the reserved development policy block."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import torch
import xgboost as xgb

from src.v2.data import sha256, write_json
from src.v2.experiment import MatrixView, _peak_memory, review_budgets
from .behavior import anchor_keys
from .depth_experiment import predict
from .diagnostics import annotated_metrics, development_inputs, seen_groups
from .final_refit import SELECTIONS, add_paths, freeze, safe_paths
from .forest_experiment import score_partition
from .torch_full import CUTOFF, SCORE_ATOL, THREADS, network, save_atomic, score_source


LABELS = {"logistic": "Logistic Regression", "forest": "Random Forest", "xgboost": "XGBoost depth 10 refit",
          "torch": "PyTorch MLP (224 inputs)", "xgboost_fold3": "XGBoost depth 10 fold 3"}
COLUMNS = ("average_precision", "roc_auc", "precision", "recall", "f1", "accuracy")
WINDOWS = ("policy", "earlier_policy", "later_policy")


def scorer(name, folder, file, source, names, categorical, end):
    """Reload a saved checkpoint and bind the matching inference preprocessing."""
    if name.startswith("xgboost"):
        model = xgb.Booster(params={"nthread": THREADS})
        model.load_model(folder / file)
        if model.feature_names != names or model.num_boosted_rounds() != 320:
            raise ValueError("XGBoost checkpoint input columns or round count differ from the refit plan.")
        view = MatrixView(source, dict(zip(names, range(len(names)))), categorical).fit(end)
        return lambda lo, hi: predict(model, view, lo, hi)

    prep = joblib.load(folder / "preprocessor.joblib")
    if name == "torch":
        saved = torch.load(folder / file, map_location="cpu", weights_only=True)
        model = network(saved["input_dim"])
        model.load_state_dict(saved["state_dict"])
        return lambda lo, hi: score_source(model, source[lo:hi], prep)
    model = joblib.load(folder / file)
    return lambda lo, hi: score_partition(model, prep, source[lo:hi])


def policy_metrics(labels, scores, groups):
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Expected one finite score per policy row.")
    return {"overall": annotated_metrics(labels, scores, CUTOFF), "review_budgets": review_budgets(labels, scores),
            "entities": {name: annotated_metrics(labels[mask], scores[mask], CUTOFF) for name, mask in groups.items()}}


def score_all(root, prior, out):
    status = json.loads((out / "run_status.json").read_bytes())
    if status["completed"] != list(SELECTIONS) or status["constraint"] is not None:
        raise ValueError("All four model checkpoints must be frozen before scoring any policy rows.")
    plan = freeze(root, prior, out)
    digest = sha256(out / "refit_plan.json")
    matrices, _, labels, _ = development_inputs(root)
    cache, advanced, _, source = matrices
    names, parts = advanced["features"], plan["partitions"]
    start, stop = parts["policy"]["start"], parts["policy"]["end"]
    all_models = {}
    for name in SELECTIONS:
        folder = out / name
        record = json.loads((folder / "fit.json").read_bytes())
        if (record["name"] != name or record["freeze_sha256"] != digest
                or any(sha256(folder / file) != checksum for file, checksum in record["artifacts"].items())):
            raise ValueError(f"Invalid frozen checkpoint for {name}.")
        all_models[name] = (folder, record, parts["torch_gradient"]["end"] if name == "torch" else start)

    old = plan["selected"]["xgboost"]
    old_folder = prior / old["path"]
    old_record = json.loads((old_folder / "result.json").read_bytes())
    previous_end = plan["fold3"]["train_end"]
    if old_record["fit_rows"] != previous_end or old_record["validation_end"] != start:
        raise ValueError("The verified depth-10 fold-3 model differs from the refit comparison plan.")
    all_models["xgboost_fold3"] = (old_folder, {"model_file": "depth10.ubj", **old_record["challenger_resources"]}, previous_end)

    keys = anchor_keys(source, names)
    scores_dir = out / "policy_scores"
    scores_dir.mkdir(exist_ok=True)
    report = {}
    torch.set_num_threads(THREADS)
    with threadpool_limits(limits=THREADS):
        for name, (folder, record, fit_end) in all_models.items():
            infer = perf_counter()
            make_scorer = lambda: scorer(name, folder, record["model_file"], source, names, cache["categorical"], fit_end)
            predict_once = make_scorer()
            if name == "xgboost_fold3":
                expected = np.load(old_folder / "depth10_scores.npy", allow_pickle=False)
                np.testing.assert_array_equal(predict_once(fit_end, start), expected)
            timed = perf_counter()
            values = predict_once(start, stop)
            seconds = perf_counter() - timed
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {name} policy scores.")
            dest = scores_dir / f"{name}.npy"
            save_atomic(values, dest, lambda scores, path: np.save(path, scores, allow_pickle=False))
            del predict_once
            replay = make_scorer()(start, stop)
            stored = np.load(dest, allow_pickle=False)
            tolerance = SCORE_ATOL if name == "torch" else 2e-15
            np.testing.assert_allclose(replay, stored, rtol=0, atol=tolerance)
            windows = {}
            for window in WINDOWS:
                lo, hi = parts[window]["start"], parts[window]["end"]
                groups = seen_groups(keys, np.arange(fit_end), lo, hi)
                windows[window] = policy_metrics(labels[lo:hi], stored[lo-start:hi-start], groups)
            report[name] = {"fit": record, "gradient_training_rows": fit_end, "policy_score_seconds": seconds,
                            "policy_score_and_replay_seconds": perf_counter() - infer, "reload_max_error": float(np.max(np.abs(replay - stored))),
                            "scores_sha256": sha256(dest), "windows": windows}
            print(f"{LABELS[name]} policy AP {windows['policy']['overall']['average_precision']:.5f}.", flush=True)
    return {"plan": plan, "plan_sha256": digest, "status": status, "models": report, "cutoff": CUTOFF,
            "policy_score_process_peak_working_set_bytes": _peak_memory()}


def fmt(value):
    return f"{value:.5f}" if value is not None else "—"


def markdown(report):
    plan, models = report["plan"], report["models"]
    lines = ["# FraudLens V3 Step 7A final refit", "",
             "The selected Step 3A Logistic Regression, Step 4A Random Forest, Step 5A depth-10 XGBoost, and Step 6A 224-input MLP were refit. "
             "Step 2 and Step 6B inputs are excluded. All four checkpoints were saved and verified before policy scoring. The reserved block had been inspected in earlier work, so these are development diagnostics.", "",
             "## Chronological populations", "",
             "The times below are inclusive endpoints in `TransactionDT` seconds. Each boundary keeps equal timestamps together.", "",
             "| Population | Rows | Fraud rows | First second | Last second |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, part in plan["partitions"].items():
        lines.append(f"| {name.replace('_', ' ')} | {part['rows']:,} | {part['fraud_rows']:,} | {part['first_second']:,} | {part['last_second']:,} |")
    lines += ["", "The earlier policy portion combines the two earlier V2 policy quarters. The later portion is the original verification half. No policy observations were used for fitting, model selection, threshold selection or calibration in this step.", "",
              "## Frozen model configurations", "",
              "| Model | Gradient or estimator rows | Preprocessing and estimator |", "| --- | ---: | --- |",
              f"| Logistic Regression | {models['logistic']['gradient_training_rows']:,} | V2 log-frequency, median fill, missing flags and scaling. ANOVA top 192 plus bounded one-hot encoding. Unweighted L2 (`C=0.08`), L-BFGS, tolerance `1e-4`, cap 1,000 iterations. Converged in {models['logistic']['fit']['iterations']} iterations: {models['logistic']['fit']['converged']}. |",
              f"| Random Forest | {models['forest']['gradient_training_rows']:,} | V2 log-frequency, median fill, missing flags and scaling. ANOVA top 180. Unweighted, 140 bootstrap trees, depth 22, leaf minimum 3, Gini, square-root feature sampling, seed 42. |",
              f"| XGBoost refit | {models['xgboost']['gradient_training_rows']:,} | Original 497 V2 columns, missing values and native categoricals. Depth 10, 320 rounds, learning rate 0.055, L2 4, max bin 128, row sampling 0.85, column sampling 0.8, no class weighting, seed 42. |",
              f"| PyTorch MLP | {models['torch']['gradient_training_rows']:,} | Gradient-only V2 transform with ANOVA top 224. 224 → 192 → 96 → 48 → 1, BatchNorm, GELU, dropout 0.18, unweighted BCE, AdamW (0.0007, decay 0.002), cosine schedule, seed 42. Selected epoch {models['torch']['fit']['best_epoch']} of {len(models['torch']['fit']['history'])} by internal AP. |",
              "", "## Reserved policy metrics", "", "Decision metrics use the fixed 0.5 threshold. Budgets rank within each window and keep source order for tied scores.", "",
              "| Window | Model | AP | ROC-AUC | Precision | Recall | F1 | Accuracy | Recall at 1% | Recall at 5% | Recall at 10% |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for window in WINDOWS:
        for name, record in models.items():
            values = record["windows"][window]
            metrics = [values["overall"][key] for key in COLUMNS] + [values["review_budgets"][str(q)]["recall"] for q in (1, 5, 10)]
            lines.append(f"| {window.replace('_',' ')} | {LABELS[name]} | " + " | ".join(map(fmt, metrics)) + " |")
    lines += ["", "## Depth-10 recency comparison", "",
              f"The existing fold-3 checkpoint fit {models['xgboost_fold3']['gradient_training_rows']:,} rows. The refit adds {models['xgboost']['gradient_training_rows']-models['xgboost_fold3']['gradient_training_rows']:,} more recent pre-policy rows. "
              "The original external fold-3 scores matched exactly before using this checkpoint on the same policy rows as the new model.", "",
              "| Window | AP change | ROC-AUC change |", "| --- | ---: | ---: |"]
    for window in WINDOWS:
        old, new = (models[key]["windows"][window]["overall"] for key in ("xgboost_fold3", "xgboost"))
        lines.append(f"| {window.replace('_',' ')} | {new['average_precision']-old['average_precision']:+.5f} | {new['roc_auc']-old['roc_auc']:+.5f} |")
    lines += ["", "## Entity proxies", "", "A complete card–address–D1 key is seen if it appeared among the model's fitting rows. Missing keys form a separate group. PyTorch and fold-3 XGBoost have smaller fit prefixes, so their group memberships differ. The JSON includes all metrics for each stratum.", "",
              "| Window | Model | Seen rows | Seen AP | Unseen rows | Unseen AP | Missing-key AP |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for window in WINDOWS:
        for name, record in models.items():
            group = record["windows"][window]["entities"]
            a, b, m = (group[key] for key in ("seen_in_fit", "unseen_in_fit", "missing_key"))
            lines.append(f"| {window.replace('_',' ')} | {LABELS[name]} | {a['rows']:,} | {fmt(a['average_precision'])} | "
                         f"{b['rows']:,} | {fmt(b['average_precision'])} | {fmt(m['average_precision'])} |")
    lines += ["", "## Runtime and checkpoint verification", "", "Fit time includes preprocessing. Peak fit memory is per worker. Checkpoint size includes fitted preprocessing where applicable. All policy predictions were scored again from reloaded files and compared to the stored arrays.", "",
              "| Model | Fit seconds | Policy inference seconds | Fit peak GiB | Checkpoint MiB | Maximum reload error |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, record in models.items():
        fit = record["fit"]
        lines.append(f"| {LABELS[name]} | {fit['fit_seconds']:.2f} | {record['policy_score_seconds']:.2f} | "
                     f"{fit['peak_working_set_bytes']/(1<<30):.2f} | {fit['checkpoint_bytes']/(1<<20):.3f} | {record['reload_max_error']:.3g} |")
    lines += ["", f"Refit command wall time was {report['status']['wall_seconds']:.2f} s. Policy scoring process peak was {report['policy_score_process_peak_working_set_bytes']/(1<<30):.2f} GiB. Fold-3 fit resources are from the original run.", "",
              "## Reproduce or resume", "", "The refit runner verifies and reuses saved checkpoints, and the scoring command requires all four checkpoints. The prediction arrays align by source order to the `policy.start` and `policy.end` offsets in the plan. A fresh output path gives an independent refit.", "", "```powershell",
              r"Set-Location E:\fraudlens-v3", r"$python = 'E:\fraudlens\.venv\Scripts\python.exe'",
              r"$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'", r"$dev = 'E:\fraudlens-v2\artifacts\v2'",
              r"$prior = 'artifacts\v3'", r"$out = 'artifacts\v3\step7a-refit'",
              r"& $python -B -m src.v3.final_refit --development $dev --prior $prior --output $out",
              r"& $python -B -m src.v3.final_refit_report --development $dev --prior $prior --output $out --markdown 'docs\V3_STEP7A.md'",
              r"& $python -B -m pytest -q tests/test_v3_final_refit.py", "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_paths(parser)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    root, prior, out = safe_paths(args.development, args.prior, args.output)
    result = score_all(root, prior, out)
    write_json(out / "experiment_summary.json", result)
    args.markdown.write_text(markdown(result), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
