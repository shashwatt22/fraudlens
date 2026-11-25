"""Saved model comparisons and exploratory validation-only thresholds."""

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .logic import CORE_MODELS, MODEL_NAMES, TARGET, threshold_metrics
from .syllabus import render_syllabus
from .ui import AMBER, BLUE, SLATE, TEAL, chart_layout, decimal, hero, percent, plot

METRICS = {"average_precision": "Average precision", "roc_auc": "ROC-AUC", "precision": "Precision",
           "recall": "Recall", "specificity": "Specificity", "f1": "F1", "accuracy": "Accuracy"}
RANKINGS = {"classifier": "XGBoost classifier", "anomaly": "Isolation Forest", "composite": "Fixed 80:20 composite"}


def _comparison(run, partition, metrics):
    validation = metrics if partition == "validation" else run.metrics("validation")
    best = max(CORE_MODELS, key=lambda key: validation[key]["average_precision"])
    champion = run.manifests[2]["champion"]
    left, right = st.columns(2)
    left.metric("Highest validation AP", MODEL_NAMES[best], help=f"AP {validation[best]['average_precision']:.5f}", border=True)
    right.metric("Deployed classical classifier", MODEL_NAMES[champion], border=True)
    st.caption(f"Leading benchmark validation AP {validation[best]['average_precision']:.5f} · "
               f"Deployed classifier validation AP {validation[champion]['average_precision']:.5f}")
    train_times = {key: run.manifests[2]["models"][key]["fit_seconds"] for key in CORE_MODELS if key != "mlp"}
    train_times["mlp"] = run.manifests["2B"]["training_summary"]["training_seconds"]
    table = pd.DataFrame([{"Model": MODEL_NAMES[key], **{label: metrics[key][metric] for metric, label in METRICS.items()},
                           "Training seconds": train_times[key]} for key in CORE_MODELS])
    st.subheader(f"{run.partition_name(partition)} model comparison")
    st.dataframe(table, hide_index=True, column_config={name: st.column_config.NumberColumn(format="%.5f")
                 for name in METRICS.values()} | {"Training seconds": st.column_config.NumberColumn(format="%.3f")})
    ordered = table.sort_values("Average precision")
    colors = [TEAL if item == MODEL_NAMES[best] else BLUE for item in ordered["Model"]]
    fig = go.Figure(go.Bar(y=ordered["Model"], x=ordered["Average precision"], orientation="h", marker_color=colors,
                            text=ordered["Average precision"].map(lambda value: f"{value:.5f}"), textposition="auto"))
    plot(chart_layout(fig, height=300, x="Average precision"), "model_comparison_ap")
    st.caption("Select a column heading to sort. Dummy precision is undefined when it raises no alerts. "
               "MLP training time includes epoch validation.")


def _curve(curve, x, y, baseline, color, label, auc):
    xs, ys = np.asarray(curve[x]), np.asarray(curve[y])
    indices = np.linspace(0, len(xs) - 1, min(len(xs), 650), dtype=int)
    fig = go.Figure(go.Scatter(x=xs[indices], y=ys[indices], name=label, mode="lines", line=dict(color=color, width=2.8)))
    fig.add_trace(go.Scatter(x=[0, 1], y=baseline, name="Reference", mode="lines", line=dict(color=SLATE, dash="dot")))
    fig.add_annotation(x=.98, y=.05, xref="paper", yref="paper", xanchor="right", text=auc,
                       showarrow=False, bgcolor="#192b42", borderpad=5)
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return fig


def _evaluation(run, partition, metrics):
    selected = st.selectbox("Evaluation model", CORE_MODELS, index=3, format_func=MODEL_NAMES.get, key="eval_model")
    phase = "2B" if selected == "mlp" else 2
    frozen = (run.manifests["2B"]["selection"]["threshold"] if phase == "2B"
              else run.manifests[2]["models"][selected]["threshold"])
    original = metrics[selected]
    live = dict(original)
    live["flagged"] = original["confusion_matrix"][0][1] + original["confusion_matrix"][1][1]
    st.caption("Frozen decision: majority rule" if frozen is None else f"Frozen threshold {frozen:.10f}")
    if partition == "validation":
        threshold = st.slider("Exploratory validation threshold", 0.0, 1.0, float(frozen if frozen is not None else .5),
                              step=.001, format="%.3f", key=f"threshold_{run.mode}_{selected}")
        preds = run.table(phase, "validation_predictions", [TARGET, f"{selected}_score"])
        live = threshold_metrics(preds[TARGET], preds[f"{selected}_score"], threshold)
        st.caption("The slider recalculates validation decisions from saved scores. Saved thresholds and decisions stay frozen.")
    else:
        st.info(f"{run.partition_name(partition)} metrics are frozen. "
                + ("This quick slice is a development diagnostic." if run.mode == "quick" else "Threshold exploration uses validation."))
    columns = st.columns(5)
    for col, name in zip(columns[:4], ("precision", "recall", "f1", "specificity"), strict=True):
        col.metric(METRICS[name], percent(live[name]), border=True)
    columns[4].metric("Flagged transactions", f"{live['flagged']:,}", border=True)

    left, right = st.columns([1, 1.65], gap="medium")
    with left, st.container(border=True):
        st.subheader("Confusion matrix")
        z = live["confusion_matrix"]
        fig = go.Figure(go.Heatmap(z=z, x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"],
                                   text=[[f"{n:,}" for n in row] for row in z], texttemplate="%{text}",
                                   colorscale=[[0, "#172337"], [1, "#277d88"]], showscale=False,
                                   hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z:,} rows<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        plot(chart_layout(fig, x="Predicted class", y="Observed class"), "evaluation_matrix")
    with right, st.container(border=True):
        st.subheader("Ranking curves")
        st.caption(f"{run.partition_name(partition)} · saved curve coordinates sampled for display")
        curves = run.report(phase, f"{partition}_curves")["models"][selected]
        if not curves["roc"] or not curves["precision_recall"]:
            st.info("ROC and precision–recall curves are unavailable for this partition.")
        else:
            roc = _curve(curves["roc"], "false_positive_rate", "true_positive_rate", [0, 1], BLUE,
                         MODEL_NAMES[selected], f"AUC {original['roc_auc']:.5f}")
            prevalence = run.partitions[partition]["fraud_rate"]
            pr = _curve(curves["precision_recall"], "recall", "precision", [prevalence] * 2, TEAL,
                        MODEL_NAMES[selected], f"AP {original['average_precision']:.5f}")
            first, second = st.columns(2)
            with first:
                plot(chart_layout(roc, x="False-positive rate", y="True-positive rate"), "evaluation_roc")
            with second:
                plot(chart_layout(pr, x="Recall", y="Precision"), "evaluation_pr")
    with st.expander("Metric definitions"):
        st.markdown("- **Precision** is the share of flagged transactions labeled fraud.\n"
                    "- **Recall** is the share of fraud cases flagged.\n"
                    "- **Specificity** is the share of legitimate transactions left unflagged.\n"
                    "- **F1** is the harmonic mean of precision and recall. **Accuracy** is the share of correct decisions.\n"
                    "- **AP** summarizes the precision–recall curve. **ROC-AUC** summarizes ranking across false-positive rates.")


def _deep_learning(run):
    manifest = run.manifests["2B"]
    arch = manifest["architecture"]
    config = manifest["training_config"]
    summary = manifest["training_summary"]
    history = pd.DataFrame(run.report("2B", "history")["epochs"])
    st.subheader("PyTorch multilayer perceptron")
    st.caption("Training and validation · saved CPU benchmark")
    hidden = arch["hidden_widths"]
    layer_names = [("Input", f"{arch['input_dim']} features"), *[(f"Hidden {n}", f"{width} units · BatchNorm · ReLU · Dropout {arch['dropout']:g}")
                   for n, width in enumerate(hidden, 1)], ("Output", "1 logit · sigmoid for scoring")]
    cards = "".join(f'<div class="fl-layer"><strong>{escape(name)}</strong><span>{escape(detail)}</span></div>'
                    for name, detail in layer_names)
    st.html(f'<div class="fl-architecture" aria-label="Neural architecture">{cards}</div>')
    widgets = st.columns(4)
    for widget, (label, value) in zip(widgets, (("Trainable parameters", f"{arch['trainable_parameters']:,}"),
                                               ("Epochs completed", summary["epochs_completed"]),
                                               ("Best checkpoint", f"Epoch {summary['best_epoch']}"),
                                               ("Training time", f"{summary['training_seconds']:.2f} s")), strict=True):
        widget.metric(label, value, border=True)
    st.caption(f"{config['optimizer']} · {config['loss']} · positive class weight {config['positive_class_weight']:.4f} · "
               f"learning rate {config['learning_rate']:g} · weight decay {config['weight_decay']:g} · batch size {config['batch_size']}")
    reason = (f"Early stopped after {summary['epochs_since_best']} epochs without validation AP improvement."
              if summary["stop_reason"] == "validation_ap_patience" else "Training reached the configured epoch limit.")
    st.write(reason)
    first, second = st.columns(2, gap="medium")
    with first, st.container(border=True):
        st.markdown("**Weighted loss**")
        fig = go.Figure()
        for name, label, color in (("train_loss", "Training", TEAL), ("validation_loss", "Validation", AMBER)):
            fig.add_scatter(x=history["epoch"], y=history[name], name=label, mode="lines+markers", line=dict(color=color))
        plot(chart_layout(fig, height=275, x="Epoch", y="Weighted loss"), "mlp_loss")
    with second, st.container(border=True):
        st.markdown("**Validation average precision**")
        best = history.loc[history["epoch"].eq(summary["best_epoch"])].iloc[0]
        fig = go.Figure(go.Scatter(x=history["epoch"], y=history["validation_ap"], mode="lines+markers",
                                    line=dict(color=TEAL), name="Validation AP"))
        fig.add_scatter(x=[best["epoch"]], y=[best["validation_ap"]], mode="markers", name="Best checkpoint",
                        marker=dict(color=AMBER, size=12, symbol="diamond"))
        plot(chart_layout(fig, height=275, x="Epoch", y="Average precision"), "mlp_ap")
    low = history.loc[history["validation_loss"].idxmin()]
    st.caption(f"Validation loss reached its minimum at epoch {int(low['epoch'])}. Validation AP peaked at epoch "
               f"{int(best['epoch'])}. The checkpoint follows ranking AP, which can improve even as weighted loss rises.")


def _ablation(run, partition):
    ablation = run.report(2, "ablation")
    report = ablation[partition]
    st.subheader(f"XGBoost behavioral ablation · {run.partition_name(partition)}")
    st.write("Both models use the same chronological rows and settings. The history variant adds seven strictly earlier behavior features.")
    basic = run.manifests[2]["features"]["basic"]
    behavior = run.manifests[2]["features"]["behavioral"]
    st.caption(f"Basic {basic['raw_dimension']} raw features / {basic['transformed_dimension']} encoded columns · "
               f"With history {behavior['raw_dimension']} raw features / {behavior['transformed_dimension']} encoded columns")
    with st.expander("Exact feature sets"):
        st.write("**Basic inputs**", ", ".join(basic["raw_features"]))
        st.write("**Historical additions**", ", ".join(name for name in behavior["raw_features"] if name not in basic["raw_features"]))
    rows = []
    for key, value in report["basic"].items():
        delta = report["delta_behavioral_minus_basic"][key]
        rows.append({"Metric": METRICS[key], "Basic": value, "With history": report["behavioral"][key],
                     "Absolute change": delta, "Relative change": delta / value if value and delta is not None else None})
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        "Basic": st.column_config.NumberColumn(format="%.5f"), "With history": st.column_config.NumberColumn(format="%.5f"),
        "Absolute change": st.column_config.NumberColumn(format="%+.5f"),
        "Relative change": st.column_config.NumberColumn(format="percent"),
    })
    keys = ("average_precision", "roc_auc")
    fig = go.Figure([go.Bar(x=[METRICS[key] for key in keys], y=[report[field][key] for key in keys], name=name,
                          marker_color=color) for field, name, color in (("basic", "Basic", BLUE), ("behavioral", "With history", TEAL))])
    fig.update_layout(barmode="group")
    plot(chart_layout(fig, y="Metric value"), "ablation_compare")
    val = ablation["validation"]["delta_behavioral_minus_basic"]["average_precision"]
    test = ablation["test"]["delta_behavioral_minus_basic"]["average_precision"]
    if val > 0 and test < 0:
        st.info(f"History improved validation AP by {val:.5f} and weakened {run.partition_name('test').lower()} AP by {-test:.5f}.")
    else:
        st.info(f"History changed validation AP by {val:+.5f} and {run.partition_name('test').lower()} AP by {test:+.5f}.")


def _risk_ranking(run, partition):
    has_test = "test_rankings" in run.manifests[3]["files"]
    show_test = partition == "test" and has_test
    report = run.report(3, "test_rankings" if show_test else "rankings")
    partition_name = run.partition_name("test" if show_test else "validation")
    rankings = report["rankings"]
    st.subheader(f"{partition_name} risk-ranking experiment")
    st.markdown("**Risk score = 100 × (0.8 × classifier score + 0.2 × anomaly percentile)**")
    summary, budgets = [], []
    for name, values in rankings.items():
        summary.append({"Ranking": RANKINGS[name], "Average precision": values["average_precision"], "ROC-AUC": values["roc_auc"]})
        for share, budget in values["top_review"].items():
            budgets.append({"Ranking": RANKINGS[name], "Budget": f"Top {share}%", "Reviewed": budget["reviewed"],
                            "Fraud found": budget["fraud_found"], "Precision": budget["precision"], "Recall": budget["recall"]})
    one, two = st.columns([1, 1.6], gap="medium")
    with one:
        st.dataframe(pd.DataFrame(summary), hide_index=True, column_config={key: st.column_config.NumberColumn(format="%.5f")
                     for key in ("Average precision", "ROC-AUC")})
    with two:
        st.dataframe(pd.DataFrame(budgets), hide_index=True, column_config={key: st.column_config.NumberColumn(format="percent")
                     for key in ("Precision", "Recall")})
    gap = rankings["composite"]["average_precision"] - rankings["classifier"]["average_precision"]
    result = (f"The fixed blend underperformed classifier-only ranking by {-gap:.5f} AP" if gap < 0
              else f"The fixed blend changed AP by {gap:+.5f} versus classifier-only ranking")
    st.info(f"{result} on these {partition_name.lower()} rows. "
            "The blend is an experimental decision-support signal.")
    st.caption(f"{report['rows']:,} {partition_name.lower()} rows · {report['tie_rule']}")


def render(run, partition):
    hero("Model Lab", "Compare saved benchmarks, explore validation thresholds and inspect the behavioral experiments.", run.mode)
    st.caption(f"Selected evaluation partition · {run.partition_name(partition)}")
    metrics = run.metrics(partition)
    first, second, third, fourth, fifth = st.tabs(["Model comparison", "Evaluation lab", "Deep learning", "Behavioral ablation", "Risk ranking"])
    with first:
        _comparison(run, partition, metrics)
    with second:
        _evaluation(run, partition, metrics)
    with third:
        _deep_learning(run)
        render_syllabus(run, partition)
    with fourth:
        _ablation(run, partition)
    with fifth:
        _risk_ranking(run, partition)
