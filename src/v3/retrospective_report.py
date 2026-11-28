"""Render the frozen retrospective results without reopening any labels."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.v2.data import sha256, write_json
from .decision_policy import read
from .retrospective import SCOPE


NAMES = {"logistic": "V3 Logistic Regression", "forest": "V3 Random Forest", "xgboost": "V3 depth-10 XGBoost",
         "torch": "V3 PyTorch MLP", "v1_xgboost": "V1 XGBoost", "v2_ensemble": "V2 weighted ensemble"}
COLORS = {"logistic": "#2b6da8", "forest": "#76539a", "xgboost": "#00726a", "torch": "#a35825",
          "v1_xgboost": "#5c6470", "v2_ensemble": "#a13759"}
METRICS = ("precision", "recall", "f1", "accuracy")


def chart_points(curve, x_key, y_key, limit=401):
    """Preserve endpoints and omit per-score thresholds from display curves."""
    x, y = curve[x_key], curve[y_key]
    count = min(len(x), limit)
    indexes = [round(i * (len(x) - 1) / (count - 1)) for i in range(count)] if count > 1 else range(count)
    return {x_key: [x[i] for i in indexes], y_key: [y[i] for i in indexes]}


def dashboard_bundle(result, curves):
    """Package aggregate metrics and lightweight display curves only."""
    return {
        "schema_version": 1,
        **{key: result[key] for key in ("scope", "population", "primary", "calibration", "ranking_scores", "probability_scores", "resources")},
        "display_curves": "ROC and precision–recall series reduced to at most 401 points each. Metrics use all rows.",
        "models": {name: {"name": NAMES[name], **row,
                          "curves": {
                              "roc": chart_points(curves[name]["roc"], "false_positive_rate", "true_positive_rate"),
                              "precision_recall": chart_points(curves[name]["precision_recall"], "recall", "precision"),
                              "reliability": curves[name]["reliability"],
                          }} for name, row in result["models"].items()},
    }


def plot(result, curves, output):
    paths = []
    population = result["population"]
    for kind, filename, xlabel, ylabel in (
        ("roc", "v3_step7c_roc.png", "False positive rate", "True positive rate"),
        ("precision_recall", "v3_step7c_precision_recall.png", "Recall", "Precision"),
        ("reliability", "v3_step7c_calibration.png", "Mean predicted score", "Observed fraud rate"),
    ):
        fig, axis = plt.subplots(figsize=(8.8, 5.3), layout="constrained")
        for model in NAMES:
            row = curves[model][kind]
            model_metrics = result["models"][model]
            if kind == "roc":
                x, y = row["false_positive_rate"], row["true_positive_rate"]
                score = model_metrics["roc_auc"]
            elif kind == "precision_recall":
                x, y = row["recall"][::-1], row["precision"][::-1]
                score = model_metrics["average_precision"]
            else:
                x, y = [p["forecast"] for p in row], [p["observed"] for p in row]
                score = model_metrics["brier"]
            label = f"{NAMES[model]} ({score:.4f})"
            style = "--" if model.startswith(("v1_", "v2_")) else "-"
            options = {"label": label, "color": COLORS[model], "linestyle": style, "linewidth": 1.9 if model == "xgboost" else 1.3}
            if kind == "reliability":
                axis.plot(x, y, marker="o", markersize=3, **options)
            else:
                axis.step(x, y, where="pre", **options)
        if kind == "precision_recall":
            axis.axhline(population["fraud_rows"]/population["rows"], color="#8592a3", linestyle=":", label="Fraud prevalence")
            selected = result["models"]["xgboost"]["frozen"]
            axis.scatter(selected["recall"], selected["precision"], color=COLORS["xgboost"], edgecolors="white", zorder=4,
                         label="V3 primary frozen threshold")
        else:
            axis.plot([0, 1], [0, 1], color="#8592a3", linestyle=":", label="Chance" if kind == "roc" else "Perfect calibration")
        caption = {"roc": "ROC (AUC)", "precision_recall": "Precision–recall (AP)", "reliability": "Calibration (Brier score)"}[kind]
        axis.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.01), title=f"{caption} · Reused retrospective holdout")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc="lower right" if kind == "roc" else "best")
        path = output / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def comma(value):
    return f"{value:,}"


def metrics(row, fields):
    return " | ".join(f"{row[key]:.5f}" if row[key] is not None else "—" for key in fields)


def markdown(report, plots):
    pop = report["population"]
    models = report["models"]
    lines = ["# FraudLens V3 Step 7C retrospective evaluation", "",
             f"This **reused retrospective holdout** contains {comma(pop['rows'])} transactions and {comma(pop['fraud_rows'])} frauds "
             f"({pop['fraud_rows']/pop['rows']:.3%}) from transaction seconds {comma(pop['first_second'])} to {comma(pop['last_second'])}. "
             "V1 and V2 previously evaluated these observations. The results describe historical performance on this fixed population.", "",
             "Depth-10 XGBoost is the frozen V3 primary classifier. The Step 7A checkpoints and Step 7B thresholds were reused, including XGBoost identity calibration. "
             "No classifier, scaling, encoding, calibration or threshold fitting took place. All four V3 models scored identical rows before the label-access receipt was created.", "",
             "## Original-score rankings and probability losses", "",
             "AP, ROC-AUC and review budgets use original ranking scores. Recall at each budget reviews the highest scores, rounds up to whole rows and breaks ties in source order. "
             "Brier and log loss use the original probability outputs or score, which also equals the policy value under identity calibration. V2's weighted ensemble score remains uncalibrated.", "",
             "| Model | AP | ROC-AUC | Brier | Log loss | Recall at 1% | Recall at 5% | Recall at 10% |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, label in NAMES.items():
        row = models[name]
        recall = " | ".join(f"{row['review_budgets'][str(q)]['recall']:.5f}" for q in (1, 5, 10))
        lines.append(f"| {label} | {metrics(row, ('average_precision','roc_auc','brier','log_loss'))} | {recall} |")
    lines += ["", "## Default and frozen operating points", "", "Confusion matrices give `[TN, FP]` and `[FN, TP]` rows. "
              "The V1 and V2 frozen results use their respective archived cutoffs. Their transaction IDs, timestamps, and frozen decisions were checked against the archived predictions.", "",
              "| Model | Point | Threshold | Precision | Recall | F1 | Accuracy | Confusion matrix | Reviews | Reviews / 1,000 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for name, label in NAMES.items():
        for point, title in (("at_0_5", "Default"), ("frozen", "Frozen")):
            row = models[name][point]
            lines.append(f"| {label} | {title} | {row['threshold']:.10g} | {metrics(row, METRICS)} | "
                         f"`{row['confusion_matrix']}` | {comma(row['reviewed'])} | {row['review_rate']*1000:.1f} |")
    v3 = models["xgboost"]
    lines += ["", "## Primary model compared with earlier versions", "",
              "| Archive | V3 AP change | V3 ROC-AUC change | V3 frozen F1 change |", "| --- | ---: | ---: | ---: |"]
    for name in ("v1_xgboost", "v2_ensemble"):
        old = models[name]
        lines.append(f"| {NAMES[name]} | {v3['average_precision']-old['average_precision']:+.5f} | {v3['roc_auc']-old['roc_auc']:+.5f} | "
                     f"{v3['frozen']['f1']-old['frozen']['f1']:+.5f} |")
    lines += ["", "## Entity-key performance", "",
              "Seen card–address–D1 proxy keys occurred in the relevant V3 model's fit prefix. Unseen keys are complete keys absent from that prefix. "
              "PyTorch uses its gradient-fit cutoff. Missing keys form a separate group. The JSON includes metrics for each group at both operating points.", "",
              "| Model | Entity group | Rows | Fraud prevalence | AP | ROC-AUC | Frozen precision | Frozen recall | Frozen F1 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in ("logistic", "forest", "xgboost", "torch"):
        for group, points in models[name]["entities"].items():
            row = points["frozen"]
            lines.append(f"| {NAMES[name]} | {group.replace('_',' ')} | {comma(row['rows'])} | {row['fraud_rate']:.2%} | "
                         f"{metrics(row, ('average_precision','roc_auc','precision','recall','f1'))} |")
    lines += ["", "## Curves", ""]
    lines.extend(f"[{title}]({path.name})" for title, path in zip(("ROC", "Precision–recall", "Calibration"), plots, strict=True))
    resources = report["resources"]
    lines += ["", "## Inference and integrity", "",
              "| Model | Load and prepare (s) | Inference (s) | Sampled process working set peak (GiB) |", "| --- | ---: | ---: | ---: |"]
    for name, timing in resources["timings"].items():
        peak = timing["sampled_working_set_peak_bytes"]
        lines.append(f"| {NAMES[name]} | {timing['load_and_prepare_seconds']:.2f} | {timing['inference_seconds']:.2f} | "
                     f"{f'{peak/(1<<30):.2f}' if peak else '—'} |")
    lines += ["", f"Feature build and audit took {resources['features_seconds']:.2f} s. The scoring-stage process peak, including feature preparation, "
              f"was {resources['score_stage_peak_working_set_bytes']/(1<<30):.2f} GiB. Per-model measurements are sampled process working sets, including memory-mapped pages.", "",
              "The seven inherited causal history columns were independently reconstructed. Appending holdout covariates left the 472,432 development rows unchanged in both feature matrices. "
              "The V2 builder calculates histories using strictly earlier timestamp groups. Forecasts and the feature matrix were hashed before opening labels. "
              "The four checkpoint, policy and manifest hashes matched before and after scoring and evaluation.", "",
              "## Reproduce or render", "",
              "Use a fresh directory for a separate retrospective evaluation. `evaluate` requires precomputed scores and creates an exclusive label-access receipt. "
              "Regenerate plots or the report from the saved metrics and curves without reopening labels. The local per-row predictions and mapped features remain under the ignored artifact directory. "
              "Use `dashboard.json` for the aggregate metrics and display curves. `results.json` records full scalar precision and evaluation lineage.", "", "```powershell",
              r"Set-Location E:\fraudlens-v3", r"$python = 'E:\fraudlens\.venv\Scripts\python.exe'", r"$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'",
              "$env:OMP_NUM_THREADS = '6'", "$env:OPENBLAS_NUM_THREADS = '6'", "$env:MKL_NUM_THREADS = '6'",
              r"$out = 'artifacts\v3\step7c-retrospective-reproduction'",
              r"$args7c = @('--raw','E:\fraudlens\data\raw','--v1','E:\fraudlens\artifacts\full',",
              r"    '--development','E:\fraudlens-v2\artifacts\v2','--refit','artifacts\v3\step7a-refit',",
              r"    '--policy','artifacts\v3\step7b-policy','--output',$out)",
              r"& $python -B -m src.v3.retrospective score @args7c", r"& $python -B -m src.v3.retrospective evaluate @args7c",
              r"& $python -B -m src.v3.retrospective_report --output $out --markdown (Join-Path $out 'V3_STEP7C.md')",
              r"& $python -B -m pytest -q tests/test_v3_retrospective.py", "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    result_file, curve_file = out / "results.json", out / "curves.json"
    result = read(result_file)
    source = {"curves_sha256": curve_file, "label_access_sha256": out / "label_access.json", "score_manifest_sha256": out / "score_manifest.json"}
    if result["scope"] != SCOPE or any(sha256(file) != result["lineage"][key] for key, file in source.items()):
        raise ValueError("The retrospective evaluation files do not match the results bundle.")
    curves = read(curve_file)
    plots = plot(result, curves, args.markdown.parent)
    args.markdown.write_text(markdown(result, plots), encoding="utf-8", newline="\n")
    scores = read(out / "score_manifest.json")
    files = (result_file, *source.values(), out / "feature_audit.json", out / "predictions.parquet", *plots, args.markdown)
    if sha256(out / "predictions.parquet") != scores["predictions_sha256"]:
        raise ValueError("The local prediction file no longer matches its manifest.")
    bundle = out / "dashboard.json"
    write_json(bundle, dashboard_bundle(result, curves))
    write_json(out / "artifact_hashes.json", {f.name: sha256(f) for f in (*files, bundle)})


if __name__ == "__main__":
    main()
