"""Standalone public FraudLens evaluation demo. Uses a fixed aggregate resource."""

from hashlib import sha256
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.dashboard_v3.ui import COLORS, STYLE, cards, chart, hero, pct, policy_strip

RESULTS_SHA256 = "47db18c288e526e366f500ff39cf5b54943192b6cc76e29ccece6ad909157553"
V3_MODELS = ("xgboost", "logistic", "forest", "torch")
HISTORY = ("v1_xgboost", "v2_ensemble", "xgboost")
NAMES = {"xgboost": "V3 XGBoost", "logistic": "V3 Logistic Regression", "forest": "V3 Random Forest",
         "torch": "V3 PyTorch MLP", "v1_xgboost": "V1 XGBoost", "v2_ensemble": "V2 weighted ensemble"}
SCOPE = "Reused retrospective holdout"
@st.cache_data(show_spinner=False, max_entries=1)
def results():
    content = Path(__file__).with_name("results.json").read_bytes()
    if sha256(content).hexdigest() != RESULTS_SHA256:
        raise ValueError("Unexpected aggregate results digest.")
    data = json.loads(content)
    if data["schema_version"] != 1 or data["scope"] != SCOPE.lower() or data["primary"] != "xgboost":
        raise ValueError("Unexpected aggregate results schema.")
    return data


def overview(data):
    primary, pop = data["models"]["xgboost"], data["population"]
    frozen = primary["frozen"]
    cards([
        ("Transactions studied", f"{data['dataset']['transactions']:,}", "IEEE-CIS benchmark"),
        ("Historical average precision", f"{primary['average_precision']:.5f}", "Final V3 research model"),
        ("Historical ROC-AUC", f"{primary['roc_auc']:.5f}", "Final V3 research model"),
        ("Historical F1", f"{frozen['f1']:.5f}", f"{frozen['reviewed']:,} flagged at frozen threshold"),
    ])
    st.caption(f"Reused retrospective holdout · {pop['rows']:,} transactions · {pop['fraud_rows']:,} labeled fraud.")


def public_check(data):
    hero("Check Transaction & Alerts", "The local app assesses transactions and records alerts.", "Public demo")
    st.info("Transaction assessment and real email are unavailable here. Use the local classroom app for a genuine score.")
    with st.container(border=True):
        st.subheader("How the local check works")
        st.write("Enter an amount, card network, card type and purchaser email provider category. Device type and recipient email provider category are optional. The local model returns a risk score, a review decision and the factors that influenced it.")
        st.caption("Dry-run alerts stay in the local session. Real delivery requires an authorized recipient and explicit activation.")

def _model_table(data):
    view = st.segmented_control("Operating point", ("Frozen thresholds", "0.5 for all models"), default="Frozen thresholds", key="fl_operating")
    key = "at_0_5" if view == "0.5 for all models" else "frozen"
    rows = []
    for name in V3_MODELS:
        item = data["models"][name]
        point = item[key]
        rows.append({"Model": NAMES[name], "AP": item["average_precision"], "ROC-AUC": item["roc_auc"],
                     "F1": point["f1"], "Precision": point["precision"], "Recall": point["recall"],
                     "Accuracy": point["accuracy"], "Brier": item["brier"], "Log loss": item["log_loss"]})
    st.dataframe(rows, hide_index=True, width="stretch", column_config={
        name: st.column_config.NumberColumn(format="%.5f") for name in rows[0] if name != "Model"})
    return key


def _model_plot(data, point):
    selected = st.radio("Performance chart", ("Confusion matrix", "Precision–recall", "ROC", "Calibration"), horizontal=True)
    with st.container(border=True):
        figure = go.Figure()
        if selected == "Confusion matrix":
            model = st.selectbox("Classifier", V3_MODELS, format_func=NAMES.__getitem__)
            stats = data["models"][model][point]
            st.caption(f"Threshold {stats['threshold']:.10f} · rows are observed labels.")
            axis = ["Legitimate", "Fraud"]
            figure.add_heatmap(x=axis, y=axis, z=stats["confusion_matrix"], texttemplate="%{z:,}",
                               colorscale=[[0, "#ebf4f1"], [1, "#218073"]], showscale=False)
            figure.update_yaxes(autorange="reversed")
            chart(figure, "public_confusion", x="Model decision", y="Observed class", height=305)
            return
        spec = {"Precision–recall": ("precision_recall", "recall", "precision", "Recall", "Precision"),
                "ROC": ("roc", "false_positive_rate", "true_positive_rate", "False positive rate", "True positive rate"),
                "Calibration": ("reliability", None, None, "Mean forecast", "Observed fraud share")}
        name, x_key, y_key, x_title, y_title = spec[selected]
        for model in V3_MODELS:
            curve = data["models"][model]["curves"][name]
            if name == "reliability":
                x, y = [p["forecast"] for p in curve], [p["observed"] for p in curve]
            else:
                x, y = curve[x_key], curve[y_key]
            figure.add_scatter(x=x, y=y, mode="lines+markers" if name == "reliability" else "lines",
                               name=NAMES[model], line={"color": COLORS[model], "width": 3 if model == "xgboost" else 2})
        if name == "precision_recall":
            population = data["population"]
            figure.add_hline(y=population["fraud_rows"] / population["rows"], line_dash="dot", line_color="#8196A7")
        else:
            figure.add_scatter(x=[0, 1], y=[0, 1], name="Reference", line={"dash": "dot", "color": "#8196A7"})
        figure.update_xaxes(range=[0, 1.02])
        figure.update_yaxes(range=[0, 1.02])
        chart(figure, "public_performance", x=x_title, y=y_title, percent_x=True, percent_y=True, height=365)
        st.caption("Display curves contain rounded aggregate coordinates. Reported scalar metrics use the complete evaluation population.")


def _development(data):
    st.subheader("Development experiments")
    st.caption("Mean AP over three chronological development folds.")
    st.dataframe([{"Experiment": row["name"], "Baseline AP": row["baseline_ap"], "Candidate AP": row["candidate_ap"],
                   "Decision": row["selection"]} for row in data["experiments"]], hide_index=True, width="stretch",
                 column_config={key: st.column_config.NumberColumn(format="%.5f") for key in ("Baseline AP", "Candidate AP")})
    neural = data["neural"]
    st.subheader("Refitted neural network")
    cards([("Architecture", " → ".join(map(str, neural["architecture"])), "Dense network"),
           ("Trainable parameters", f"{neural['parameters']:,}", "Final neural configuration"),
           ("Selected epoch", f"{neural['best_epoch']} of {len(neural['epochs'])}", "Chronological internal validation")])
    plot = st.radio("Learning curves", ("Internal validation AP", "Training and validation loss"), horizontal=True)
    fields = (("internal_validation_ap", "Internal validation AP"),) if plot.endswith("AP") else (
        ("train_loss", "Training loss"), ("internal_validation_loss", "Internal validation loss"))
    figure = go.Figure()
    for i, (field, name) in enumerate(fields):
        figure.add_scatter(x=[row["epoch"] for row in neural["epochs"]], y=[row[field] for row in neural["epochs"]],
                           mode="lines+markers", name=name, line={"color": (COLORS["xgboost"], COLORS["logistic"])[i]})
    figure.add_vline(x=neural["best_epoch"], line_dash="dot")
    chart(figure, "public_learning", x="Epoch", y=plot, height=297)


def research_results(data):
    st.subheader("Historical research results")
    st.caption("Four frozen V3 models on the same retrospective holdout.")
    point = _model_table(data)
    left, right = st.columns([1.12, 1], gap="large")
    with left:
        _model_plot(data, point)
    with right:
        st.markdown("**V1 to V3 performance**")
        models = data["models"]
        diff = models["xgboost"]["average_precision"] - models["v1_xgboost"]["average_precision"]
        st.caption(f"{diff:+.5f} AP from V1 to V3 on the same rows.")
        figure = go.Figure()
        for key, label, color in (("average_precision", "Average precision", COLORS["xgboost"]),
                                  ("roc_auc", "ROC-AUC", COLORS["logistic"])):
            values = [models[name][key] for name in HISTORY]
            figure.add_bar(x=["V1", "V2", "V3"], y=values, text=[f"{value:.3f}" for value in values],
                           textposition="outside", name=label, marker_color=color)
        figure.update_layout(barmode="group")
        figure.update_yaxes(range=[0, 1.08])
        chart(figure, "public_versions", y="Metric value", height=318)

def behavior(data):
    st.subheader("V3 performance by proxy history")
    st.caption(f"{SCOPE} · XGBoost cohorts defined against the model-fit prefix.")
    cohorts = data["entity_cohorts"]
    group = {"seen_in_fit": "Seen in fit", "unseen_in_fit": "Unseen in fit", "missing_key": "Missing key"}
    figure = go.Figure(go.Bar(x=[group[name] for name in cohorts], y=[row["average_precision"] for row in cohorts.values()],
                              marker_color=COLORS["xgboost"], text=[f"{row['rows']:,} rows" for row in cohorts.values()],
                              hovertemplate="%{x}<br>AP %{y:.3f}<br>%{text}<extra></extra>"))
    chart(figure, "public_cohorts", y="Average precision", height=290)
    st.caption("Fraud prevalence varies by cohort. This affects comparisons of average precision.")
    with st.expander("V2 Isolation Forest and DBSCAN"):
        st.caption("Earlier development folds. Anomaly AP uses validation rows. DBSCAN metrics describe a training sample.")
        st.dataframe([{"Fold": row["fold"], "Inputs": row["representation"], "Anomaly AP": row["anomaly_ap"],
                       "DBSCAN silhouette": row["silhouette"], "DBSCAN noise share": row["noise_fraction"]}
                      for row in data["pattern_folds"]], hide_index=True, width="stretch")
    st.caption("The local workbench also includes row-level PCA, Apriori and TreeSHAP exploration.")


def public_dashboard(data):
    hero("Dashboard", "Verified fraud detection research and model performance.", "Public demo")
    overview(data)
    research_results(data)
    advanced = st.expander("Research methods, neural learning and behavioral analysis", on_change="rerun")
    if advanced.open:
        with advanced:
            _development(data)
            behavior(data)


PAGES = {"Dashboard": public_dashboard, "Check Transaction & Alerts": public_check}

def main():
    st.set_page_config(page_title="FraudLens | Public Research Demo", layout="wide", initial_sidebar_state="auto")
    st.html(STYLE)
    with st.sidebar:
        st.html('<div class="fl-brand"><span class="fl-mark" aria-hidden="true">FL</span>'
                '<div><strong>FraudLens</strong><small>Public research demo</small></div></div>')
        page = st.radio("Public navigation", list(PAGES), key="public_nav", label_visibility="collapsed")
        st.divider()
        st.html('<div class="fl-side-line"><strong>Aggregate evaluation</strong>IEEE-CIS Fraud Detection<br>FraudLens V3</div>')
    try:
        data = results()
    except (KeyError, ValueError, OSError):
        st.error("The verified aggregate bundle is unavailable. Check the deployment package.")
        return
    PAGES[page](data)
    st.caption("Academic research prototype · aggregate results only")
    st.markdown("[IEEE-CIS Fraud Detection dataset attribution](https://www.kaggle.com/competitions/ieee-fraud-detection/data)")


if __name__ == "__main__":
    main()
