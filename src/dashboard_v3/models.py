"""Frozen retrospective comparisons and separate development diagnostics."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import store
from .store import DataUnavailable, MODELS, NAMES, THRESHOLD
from .ui import COLORS, NAVY, TEAL, cards, chart, hero, pct, policy_strip


def _comparison(pack):
    st.subheader("Final V3 model comparison")
    st.caption("Reused retrospective holdout · metrics share the same observations · accuracy is sensitive to fraud prevalence.")
    point = st.segmented_control("Decision threshold for comparison", ["Frozen thresholds", "0.5 for all models"],
                                 default="Frozen thresholds", key="v3_operating_point")
    key = "at_0_5" if point == "0.5 for all models" else "frozen"
    rows = []
    for name in MODELS:
        model = pack["models"][name]
        decision = model[key]
        rows.append({"Model": NAMES[name], "AP": model["average_precision"], "ROC-AUC": model["roc_auc"],
                     "Precision": decision["precision"], "Recall": decision["recall"], "F1": decision["f1"],
                     "Accuracy": decision["accuracy"], "Brier": model["brier"], "Log loss": model["log_loss"],
                     "Threshold": decision["threshold"], "Reviews / 1,000": decision["review_rate"] * 1000})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        name: st.column_config.NumberColumn(format="%.5f")
        for name in ("AP", "ROC-AUC", "Precision", "Recall", "F1", "Accuracy", "Brier", "Log loss", "Threshold")
    } | {"Reviews / 1,000": st.column_config.NumberColumn(format="%.1f")})
    st.caption("AP and ROC-AUC use original scores. Brier and log loss use saved probability outputs.")
    return key


def _curves(pack, point):
    labels = ("Precision–recall", "ROC", "Calibration", "Confusion matrix")
    selected = st.radio("Chart view", labels, horizontal=True, key="v3_chart")
    if selected == "Confusion matrix":
        model = st.selectbox("Model for confusion matrix", MODELS, format_func=NAMES.__getitem__, key="v3_confusion_model")
        nums = pack["models"][model][point]
        matrix = nums["confusion_matrix"]
        with st.container(border=True):
            st.subheader(f"{NAMES[model]} · confusion matrix")
            st.caption(f"Reused retrospective holdout · cutoff {nums['threshold']:.10f} · rows are observed labels")
            fig = go.Figure(go.Heatmap(z=matrix, x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"],
                            text=[[f"{count:,}" for count in row] for row in matrix], texttemplate="%{text}",
                            textfont={"size": 17}, colorscale=[[0, "#ebf4f1"], [1, "#218073"]], showscale=False,
                            hovertemplate="Observed %{y}<br>Predicted %{x}<br>%{text} transactions<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            chart(fig, "v3_confusion", x="Model prediction", y="Observed label", height=315)
        return

    with st.container(border=True):
        st.subheader(selected)
        st.caption("Reused retrospective holdout · aggregate chart coordinates")
        figure = go.Figure()
        for name in MODELS:
            model = pack["models"][name]
            if selected == "Precision–recall":
                curve = model["curves"]["precision_recall"]
                x, y, suffix = curve["recall"], curve["precision"], f"AP {model['average_precision']:.3f}"
                x_label, y_label = "Recall", "Precision"
            elif selected == "ROC":
                curve = model["curves"]["roc"]
                x, y, suffix = curve["false_positive_rate"], curve["true_positive_rate"], f"AUC {model['roc_auc']:.3f}"
                x_label, y_label = "False positive rate", "True positive rate"
            else:
                curve = model["curves"]["reliability"]
                x, y, suffix = [p["forecast"] for p in curve], [p["observed"] for p in curve], f"Brier {model['brier']:.3f}"
                x_label, y_label = "Mean forecast", "Observed fraud rate"
            figure.add_scatter(x=x, y=y, name=f"{NAMES[name]} · {suffix}",
                               mode="lines+markers" if selected == "Calibration" else "lines",
                               line=dict(color=COLORS[name], width=3 if name == "xgboost" else 2),
                               marker={"size": 6}, hovertemplate="%{x:.1%} / %{y:.1%}<extra>%{fullData.name}</extra>")
        if selected != "Precision–recall":
            figure.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal" if selected == "Calibration" else "Chance",
                               line={"dash": "dot", "color": "#8196A7"})
        else:
            calibration = pack["models"]["xgboost"]["frozen"]
            figure.add_scatter(x=[calibration["recall"]], y=[calibration["precision"]], mode="markers",
                               name="Primary frozen policy", marker={"color": TEAL, "size": 9})
            pop = pack["population"]
            figure.add_hline(y=pop["fraud_rows"]/pop["rows"], line_dash="dot", line_color="#8196A7",
                             annotation_text="Fraud prevalence", annotation_position="bottom right")
        figure.update_xaxes(range=[0, 1.025])
        figure.update_yaxes(range=[0, 1.025])
        chart(figure, f"v3_curves_{selected}", x=x_label, y=y_label, percent_x=True, percent_y=True, height=390)


def _versions(pack):
    st.subheader("V1 → V2 → V3")
    ids = ("v1_xgboost", "v2_ensemble", "xgboost")
    metrics = pack["models"]
    delta = metrics["xgboost"]["average_precision"] - metrics["v2_ensemble"]["average_precision"]
    st.caption(f"Same reused holdout · V3 improved AP by {delta:+.5f} over V2. Each version uses its archived frozen threshold for F1.")
    names = [NAMES[item] if item != "xgboost" else "V3 depth-10 XGBoost" for item in ids]
    with st.container(border=True):
        figure = go.Figure()
        for metric, label, color in (("average_precision", "AP", TEAL), ("roc_auc", "ROC-AUC", NAVY), ("f1", "Frozen F1", "#9D7248")):
            values = [metrics[name]["frozen"][metric] if metric == "f1" else metrics[name][metric] for name in ids]
            figure.add_bar(x=names, y=values, name=label, marker_color=color, text=[f"{value:.3f}" for value in values],
                           textposition="outside")
        figure.update_layout(barmode="group")
        figure.update_yaxes(range=[0, 1.05])
        chart(figure, "v3_versions", y="Metric value", height=315)


def _entities(pack):
    with st.expander("Entity-key performance and review budgets"):
        st.caption("Seen keys occurred in the relevant model’s fit prefix. PyTorch uses its gradient-training prefix.")
        entities, budgets = [], []
        for name in MODELS:
            model = pack["models"][name]
            budgets.append({"Model": NAMES[name], **{f"Recall at {q}%": model["review_budgets"][str(q)]["recall"] for q in (1, 5, 10)}})
            for group, metrics in model["entities"].items():
                row = metrics["frozen"]
                entities.append({"Model": NAMES[name], "Group": group.replace("_", " "), "Rows": row["rows"],
                                 "AP": row["average_precision"], "Fraud prevalence": row["fraud_rate"], "Frozen recall": row["recall"]})
        st.dataframe(pd.DataFrame(budgets), hide_index=True, width="stretch", column_config={
            f"Recall at {q}%": st.column_config.NumberColumn(format="percent") for q in (1, 5, 10)})
        st.dataframe(pd.DataFrame(entities), hide_index=True, width="stretch", column_config={
            col: st.column_config.NumberColumn(format="percent") for col in ("Fraud prevalence", "Frozen recall")})


def _neural(paths):
    st.subheader("Neural network")
    try:
        digest = store.score_manifest(paths)["frozen_hashes"]["torch/fit.json"]
        fit = store.report(paths.refit / "torch" / "fit.json", digest)
        plan = store.policy_plan(paths)
        shape = [fit["input_dim"], *plan["recipes"]["torch"]["candidate"]["options"]["widths"], 1]
    except (DataUnavailable, KeyError) as error:
        st.info(f"Frozen MLP training history unavailable. {error}")
        return
    cards([
        ("Neural architecture", " → ".join(map(str, shape)), "Dense layers · BatchNorm · GELU · dropout"),
        ("Trainable parameters", f"{fit['parameters']:,}", "Refitted 224-input model"),
        ("Best saved epoch", f"{fit['best_epoch']} of {len(fit['history'])}", "Selected by internal development AP"),
    ])
    st.caption("Learning curves use the chronological internal validation set for early stopping.")
    history = pd.DataFrame(fit["history"])
    first, second = st.columns(2)
    for column, series, yaxis in ((first, {"train_loss": "Gradient training", "internal_validation_loss": "Internal validation"}, "BCE loss"),
                                  (second, {"internal_validation_ap": "Internal validation"}, "Average precision")):
        with column, st.container(border=True):
            fig = go.Figure()
            for index, (key, label) in enumerate(series.items()):
                fig.add_scatter(x=history["epoch"], y=history[key], name=label, mode="lines+markers",
                                line={"color": (TEAL, NAVY)[index], "width": 2.5})
            fig.add_vline(x=fit["best_epoch"], line_color="#9D7248", line_dash="dot")
            fig.update_xaxes(dtick=1)
            chart(fig, f"v3_mlp_{yaxis}", x="Epoch", y=yaxis, height=275)


def _experiments(paths):
    st.subheader("Development experiments")
    st.caption("Mean AP over three chronological development folds.")
    design = (
        ("step2-behavior", "Seven-day and D-offset features", "control", "treatment", "Rejected · retain original V2 features"),
        ("step3a-logistic", "Full-data Logistic Regression", "sampled", "full", "Selected"),
        ("step4a-forest", "Full-data Random Forest", "sampled", "full", "Selected"),
        ("step5a-depth", "XGBoost depth 8 → 10", "control", "treatment", "Selected"),
        ("step6a-torch", "Full-data 224-input MLP", "sampled", "full", "Selected"),
        ("step6b-rich", "Richer MLP representation", "control", "challenger", "Rejected · retained 224 inputs"),
    )
    rows = []
    for folder, label, first, second, decision in design:
        try:
            means = store.report(paths.v3 / folder / "experiment_summary.json")["means"]
            a, b = (means[key]["average_precision"] for key in (first, second))
        except (DataUnavailable, KeyError):
            a, b = None, None
        rows.append({"Experiment": label, "Baseline AP": a, "Challenger AP": b,
                     "Δ AP": b - a if b is not None else None, "Outcome": decision})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        col: st.column_config.NumberColumn(format="%.5f") for col in ("Baseline AP", "Challenger AP", "Δ AP")})


def _sandbox(paths):
    st.subheader("Threshold sandbox")
    st.caption("Exploratory · later development policy block · any setting here leaves the frozen decision policy intact.")
    if not st.toggle("Open development threshold sandbox", key="v3_sandbox_open"):
        return
    try:
        fixed = store.development_preview(paths, THRESHOLD)
        cutoff = st.slider("Exploratory threshold", 0.0, 1.0, round(THRESHOLD, 3), .001, key="v3_sandbox_cutoff")
        preview = store.development_preview(paths, cutoff)
    except (DataUnavailable, OSError, ValueError) as error:
        st.info(f"Saved development policy data unavailable. {error}")
        return
    cards([
        ("Verification rows", f"{preview['rows']:,}", "Previously inspected development observations"),
        ("Explorer reviews", f"{preview['reviewed']:,}", f"Frozen cutoff would review {fixed['reviewed']:,}"),
        ("Explorer precision", pct(preview["precision"]), f"Frozen precision {pct(fixed['precision'])}"),
        ("Explorer recall", pct(preview["recall"]), f"Frozen recall {pct(fixed['recall'])}"),
    ])


def render(paths, pack, private, issue):
    hero("Model Intelligence", "Compare the frozen models and review the development evidence behind their selection.")
    policy_strip()
    view = st.segmented_control("Model intelligence view", ("Retrospective results", "Development evidence"),
                                default="Retrospective results", key="v3_models_view")
    if view == "Development evidence":
        _neural(paths)
        _experiments(paths)
        _sandbox(paths)
        return
    if pack is None:
        st.warning(f"V3 retrospective metrics unavailable. {issue}")
        return
    point = _comparison(pack)
    _curves(pack, point)
    _versions(pack)
    _entities(pack)
