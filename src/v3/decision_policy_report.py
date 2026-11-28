"""Render the frozen Step 7B diagnostics without fitting or choosing policies."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.v2.data import sha256, write_json
from .decision_policy import MODELS, PRIMARY, read


NAMES = {"xgboost": "Depth-10 XGBoost", "logistic": "Logistic Regression", "forest": "Random Forest", "torch": "PyTorch MLP"}
WINDOWS = {"earlier_policy": "Earlier", "later_policy": "Later"}
STYLES = {"identity": "#4b5563", "sigmoid": "#057c85", "isotonic": "#a24316"}


def plot_calibration(summary, decision, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), layout="constrained")
    for axis in axes:
        axis.plot([0, 1], [0, 1], linestyle="--", color="#8592a3", label="Perfect calibration")
        axis.set(xlabel="Mean predicted probability", ylabel="Observed fraud rate", xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=.2)

    def line(axis, bins, label, color):
        axis.plot([p["forecast"] for p in bins], [p["observed"] for p in bins], marker="o", label=label, color=color)

    axes[0].set_title("Earlier internal comparison")
    for name, row in decision["comparison"]["methods"].items():
        line(axes[0], row["reliability"], name.capitalize(), STYLES[name])
    for axis, (window, label) in zip(axes[1:], WINDOWS.items(), strict=True):
        axis.set_title(f"{label} policy portion")
        prob = summary["windows"][window][PRIMARY]["probability"]
        method = decision["calibrator"]["method"]
        if method != "identity":
            line(axis, prob["raw_reliability"], "Original scores", STYLES["identity"])
        line(axis, prob["reliability"], f"Frozen {method}", STYLES[method])
    for axis in axes:
        axis.legend(fontsize=8, loc="upper left")
    fig.suptitle("Depth-10 XGBoost calibration · development diagnostics")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pr(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    colors = {"xgboost": "#076c78", "logistic": "#985229", "forest": "#655088", "torch": "#2960a6"}
    for axis, (window, label) in zip(axes, WINDOWS.items(), strict=True):
        rows = summary["windows"][window]
        part = summary["partitions"][window]
        for name in (PRIMARY, "logistic", "forest", "torch"):
            rank = rows[name]["ranking"]
            axis.step(rank["recall"][::-1], rank["precision"][::-1], where="pre", color=colors[name],
                      label=f"{NAMES[name]} ({rank['ap']:.3f})")
        primary = rows[PRIMARY]["operating_points"]["selected_f1"]
        axis.scatter(primary["recall"], primary["precision"], color=colors[PRIMARY], edgecolor="white", s=60,
                     label="XGBoost frozen F1 cutoff", zorder=3)
        axis.axhline(part["fraud_rows"] / part["rows"], color="#8592a3", linestyle=":", label="Fraud prevalence")
        axis.set(title=f"{label} policy portion", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("Precision–recall from original classifier scores")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fmt(value):
    return "—" if value is None else f"{value:.5f}"


def markdown(summary, decision, calibration_plot, pr_plot):
    comparison = decision["comparison"]
    method = decision["calibrator"]["method"]
    lines = ["# FraudLens V3 Step 7B final decision policy", "",
             f"**Depth-10 XGBoost remains primary.** Its frozen calibration method is **{method}**. "
             "The four classifiers and their Step 7A predictions were reused. All policy numbers describe a previously inspected development block.", "",
             "## Chronological partitions", "", "| Portion | Rows | Fraud rows | First second | Last second |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, label in WINDOWS.items():
        part = summary["partitions"][name]
        lines.append(f"| {label} | {part['rows']:,} | {part['fraud_rows']:,} | {part['first_second']:,} | {part['last_second']:,} |")
    lines += ["", f"Calibration comparison fits on the first {comparison['fit_rows']:,} earlier rows and compares on the next {comparison['comparison_rows']:,}. "
              f"The internal cut is between seconds {comparison['fit_last_second']:,} and {comparison['comparison_first_second']:,}. "
              "The `develop` command writes the immutable calibration and threshold manifest before `verify` loads the later labels or computes its metrics.", "",
              "## Calibration selection", "",
              f"The rule preserves identity unless both 95% paired day-block bootstrap intervals for loss differences lie below zero. The comparison uses "
              f"{comparison['day_blocks']} elapsed-day blocks and {decision['protocol']['bootstrap_draws']:,} resamples. A sigmoid also needs a positive slope. "
              "When several methods pass, the lowest Brier score wins, with log loss breaking ties. The selected method is then fitted on the whole earlier portion. Identity has no fitted parameters. "
              "A negative loss difference favors the calibration method.", "",
              "| Method | Brier | Log loss | Brier difference, 95% interval | Log loss difference, 95% interval | Eligible |", "| --- | ---: | ---: | --- | --- | --- |"]
    for name, values in comparison["methods"].items():
        intervals = [f"[{lo:+.6f}, {hi:+.6f}]" for lo, hi in (values["delta_ci"][key] for key in ("brier", "log_loss"))]
        eligible = "Reference" if name == "identity" else ("Yes" if values["eligible"] else "No")
        lines.append(f"| {name} | {values['brier']:.6f} | {values['log_loss']:.6f} | {intervals[0]} | {intervals[1]} | {eligible} |")
    reason = ("Neither alternative cleared both loss criteria, so identity was retained."
              if method == "identity" else f"{method.capitalize()} cleared both loss criteria and had the lowest Brier score among qualifying methods.")
    lines += ["", reason, "", f"![XGBoost reliability curves]({calibration_plot})", "",
              "## Ranking and probability metrics", "",
              "AP, ROC-AUC, precision–recall curves and review budgets always use original scores. The budgets rank within each portion, break ties in source order, and round up to whole reviews. "
              "Brier and log loss evaluate the displayed probability values. Only XGBoost was eligible for calibration. Other models retain their original probability outputs.", "",
              "| Portion | Model | AP | ROC-AUC | Original Brier | Policy Brier | Original log loss | Policy log loss | Recall at 1% | Recall at 5% | Recall at 10% |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for window, label in WINDOWS.items():
        for model in MODELS:
            row = summary["windows"][window][model]
            rank, p = row["ranking"], row["probability"]
            values = [rank["ap"], rank["roc_auc"], *(p[key] for key in ("raw_brier", "brier", "raw_log_loss", "log_loss")),
                      *(rank["review_budgets"][str(q)]["recall"] for q in (1, 5, 10))]
            lines.append(f"| {label} | {NAMES[model]} | " + " | ".join(map(fmt, values)) + " |")
    lines += ["", f"![Precision–recall curves for earlier and later policy portions]({pr_plot})", "",
              "## Frozen thresholds and workload", "",
              "The primary and model-specific thresholds maximize earlier-portion F1, breaking ties at the highest cutoff. "
              "XGBoost's alternatives maximize recall at 90% precision and precision at 70% recall on the earlier portion. "
              "The default 0.5 applies to policy probabilities. When XGBoost uses a transformation, the original-score 0.5 comparison is shown separately. "
              "Confusion matrices show `[TN, FP]` and `[FN, TP]` rows. Reviews per 1,000 transaction rows describe observed workload, which may change on future data.", "",
              "| Portion | Model | Policy | Threshold | Precision | Recall | F1 | Accuracy | Confusion matrix | Reviews | Reviews per 1,000 |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for window, label in WINDOWS.items():
        for model in MODELS:
            points = summary["windows"][window][model]["operating_points"]
            for key, point in points.items():
                if key == "original_score_0.5" and method == "identity":
                    continue
                values = [f"{point['threshold']:.10g}", *(fmt(point[k]) for k in ("precision", "recall", "f1", "accuracy")),
                          f"`{point['confusion_matrix']}`", f"{point['reviewed']:,}", f"{point['review_rate']*1000:.1f}"]
                lines.append(f"| {label} | {NAMES[model]} | {key.replace('_', ' ')} | " + " | ".join(values) + " |")
    lines += ["", "## Reproduce and verify", "",
              "The local policy artifact directory holds verified model and prediction hashes, the decision freeze, and detailed metrics. "
              "The calibrator parameters, if any, are stored in the decision freeze. No classifier checkpoints were changed.", "", "```powershell",
              r"Set-Location E:\fraudlens-v3", r"$python = 'E:\fraudlens\.venv\Scripts\python.exe'", r"$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'",
              r"$dev = 'E:\fraudlens-v2\artifacts\v2'", r"$refit = 'artifacts\v3\step7a-refit'", r"$out = 'artifacts\v3\step7b-policy'",
              r"& $python -B -m src.v3.decision_policy develop --development $dev --refit $refit --output $out",
              r"& $python -B -m src.v3.decision_policy verify --development $dev --refit $refit --output $out",
              r"& $python -B -m src.v3.decision_policy_report --output $out --markdown 'docs\V3_STEP7B.md'",
              r"& $python -B -m pytest -q tests/test_v3_decision_policy.py", "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    out, report = args.output, args.markdown
    freeze, models, metrics = (out / name for name in ("decision_freeze.json", "model_freeze.json", "policy_metrics.json"))
    data, decision = read(metrics), read(freeze)
    if data["decision_sha256"] != sha256(freeze) or data["model_sha256"] != sha256(models):
        raise ValueError("Cannot report metrics generated from a different freeze.")
    calibration, pr = (report.with_name(name) for name in ("v3_step7b_calibration.png", "v3_step7b_precision_recall.png"))
    plot_calibration(data, decision, calibration)
    plot_pr(data, pr)
    report.write_text(markdown(data, decision, calibration.name, pr.name), encoding="utf-8", newline="\n")
    write_json(out / "artifact_hashes.json", {path.name: sha256(path) for path in (freeze, models, metrics, report, calibration, pr)})


if __name__ == "__main__":
    main()
