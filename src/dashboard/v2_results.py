"""Optional V2 views backed by a compatible, frozen retrospective run."""

from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import ID, TIME

from .ui import BLUE, RED, TEAL, chart_layout, plot


def _load(path):
    return json.loads(path.read_bytes())


def compatible_root(run, partition):
    if run.mode != "full" or partition != "test":
        return None
    root = Path(os.environ.get("FRAUDLENS_V2_ARTIFACTS", Path(__file__).resolve().parents[2] / "artifacts" / "v2"))
    if not (root / "retrospective_results.json").is_file():
        return None
    try:
        decision = (root / "decision_freeze.json").read_bytes()
        frozen = json.loads(decision)
        receipt = _load(root / "retrospective_access.json")
        meta = _load(root / "features.json")
        results = _load(root / "retrospective_results.json")
        if (results["scope"] != "reused retrospective holdout" or results["champion"] != frozen["champion"]
                or results["rows"] != receipt["holdout_rows"] or meta["v1_manifest_sha256"] != run.digests[1]
                or sha256(decision).hexdigest() != receipt["decision_freeze_sha256"]
                or receipt["holdout_rows"] != run.partitions["test"]["rows"]):
            raise ValueError("The V2 run belongs to a different V1 snapshot or freeze record.")
    except (OSError, ValueError, KeyError) as error:
        st.warning(f"V2 retrospective results unavailable. {error}")
        return None
    return root


def model_lab(run, partition):
    root = compatible_root(run, partition)
    if root is None:
        return
    final = _load(root / "retrospective_results.json")
    freeze = _load(root / "decision_freeze.json")
    development = _load(root / "development_selection.json")
    winner = final["champion"]
    st.divider()
    st.subheader("V2 retrospective evaluation")
    st.caption(f"Frozen V2 champion · {winner} · {final['rows']:,} matched rows · Historical holdout reused from V1")
    before = run.metrics("test")["xgboost"]
    after = final["models"][winner]["metrics"]
    metrics = ("average_precision", "roc_auc", "precision", "recall", "f1", "accuracy", "specificity")
    rows = [(field.replace("_", " ").title(), before[field], after[field]) for field in metrics]
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "V1 XGBoost", "V2 frozen champion"]), hide_index=True)
    groups = {}
    history = []
    for name in development["completed_trials"]:
        for directory in ("trials", "challengers"):
            path = root / directory / name / "summary.json"
            if path.is_file():
                results = _load(path)["metrics"]
                result = results["average_precision"]
                history.append({"Candidate": name, "Mean AP": result["mean"], "AP standard deviation": result["std"],
                                "Mean ROC-AUC": results["roc_auc"]["mean"], "Accepted": name in development["progression"]})
                groups[name] = (result["mean"], result["std"])
                break
    accepted = [name for name in development["progression"] if name in groups]
    graph = go.Figure(go.Scatter(x=accepted, y=[groups[name][0] for name in accepted],
                                 error_y={"type": "data", "array": [groups[name][1] for name in accepted]},
                                 mode="lines+markers", line={"color": TEAL, "width": 3}))
    plot(chart_layout(graph, x="Accepted individual model", y="Mean temporal validation AP"), "v2_progression")
    with st.expander("All completed V2 candidates"):
        st.dataframe(pd.DataFrame(history), hide_index=True)
    curves = _load(root / "retrospective" / f"{winner}_curves.json")
    precision_recall = curves["precision_recall"]
    coordinates = np.linspace(0, len(precision_recall["recall"]) - 1, min(2_000, len(precision_recall["recall"])), dtype=int)
    pr = go.Figure(go.Scatter(x=np.take(precision_recall["recall"], coordinates), y=np.take(precision_recall["precision"], coordinates),
                              mode="lines", line={"color": BLUE}))
    plot(chart_layout(pr, x="Recall", y="Precision"), "v2_final_pr")
    calibration = freeze["policy"][winner]
    st.caption(f"Frozen policy: {calibration['calibrator']} · cutoff {calibration['threshold']:.5f} · "
               f"final Brier score {final['models'][winner]['brier']:.5f}")
    diagram = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal", line={"dash": "dot", "color": BLUE}))
    for name, row in calibration["calibration_evaluation"].items():
        bins = row["reliability"]
        diagram.add_scatter(x=[point["mean_score"] for point in bins], y=[point["observed_fraud"] for point in bins],
                            mode="lines+markers", name=f"{name}, Brier {row['brier']:.4f}")
    plot(chart_layout(diagram, x="Mean validation score per bin", y="Observed fraud fraction"), "v2_reliability")


@st.cache_data(max_entries=2)
def _scores(root, freeze_digest):
    return pd.read_parquet(root / "retrospective" / "champion_scores.parquet")


def review_queue(run, partition, navigate):
    root = compatible_root(run, partition)
    if root is None:
        return
    decision = _load(root / "decision_freeze.json")
    release = _load(root / "retrospective_access.json")
    top = _scores(root, release["decision_freeze_sha256"]).sort_values(["policy_score", ID], ascending=[False, True]).head(15)
    st.subheader("V2 review ranking")
    st.caption(f"{decision['champion']} · Frozen retrospective policy · Sorted by V2 policy score")
    st.dataframe(top[[ID, TIME, "policy_score", "decision"]].rename(columns={"policy_score": "V2 policy score",
                                                                             "decision": "V2 flagged"}), hide_index=True)
    transaction = st.selectbox("Select an entry from the V2 ranking", top[ID].to_list(), key="v2_review_id")
    st.button("Open V2 case in Investigator", key="v2_review_open", on_click=navigate,
              args=("full", "test", int(transaction), True))


@st.cache_resource(max_entries=1)
def _explainer(root, freeze_digest):
    from src.v2.explanation import V2Explainer
    return V2Explainer(root)


def investigation(run, partition, transaction):
    root = compatible_root(run, partition)
    if root is None:
        return
    release = _load(root / "retrospective_access.json")
    table = _scores(root, release["decision_freeze_sha256"])
    match = table.loc[table[ID] == transaction]
    if len(match) != 1:
        st.warning("V2 has no aligned assessment for this transaction.")
        return
    case = match.iloc[0]
    st.subheader("V2 assessment for this transaction")
    st.metric("V2 policy score", f"{case['policy_score']:.4f}")
    st.write("Frozen V2 decision: **Flagged**" if case["decision"] else "Frozen V2 decision: **Cleared**")
    if st.button("Explain the V2 scorer", key=f"v2_explain_{transaction}"):
        try:
            report = _explainer(root, release["decision_freeze_sha256"]).explain(transaction)
        except (ValueError, OSError, KeyError) as error:
            st.warning(str(error))
            return
        scope = "Ensemble component" if report["component_only"] else "Tree scorer"
        st.caption(f"{scope} · {report['model']} · Signed contributions to the raw margin before calibration")
        values = list(reversed(report["contributions"]))
        graph = go.Figure(go.Bar(x=[n for _, n in values], y=[label for label, _ in values], orientation="h",
                                 marker_color=[RED if amount > 0 else BLUE for _, amount in values]))
        plot(chart_layout(graph, x="Raw-margin contribution"), "v2_local_contributions")
        st.caption(f"Base value {report['base_value']:.5f} · Reconstructed margin {report['raw_margin']:.5f}")


def patterns(run, partition):
    root = compatible_root(run, partition)
    if root is None:
        return
    report = _load(root / "patterns.json")
    st.divider()
    st.subheader("V2 pattern experiments")
    st.caption("Three expanding development folds. Anomaly AP uses validation rows. Clustering stability uses training samples.")
    table, rules = [], []
    for trial in report["folds"]:
        for name, values in trial["families"].items():
            cluster = values["clusters"]
            table.append({"Fold": trial["fold"], "Representation": name,
                          "Anomaly AP (256)": values["anomaly"][0]["average_precision"],
                          "Anomaly AP (768)": values["anomaly"][1]["average_precision"],
                          "DBSCAN silhouette": cluster["selected"]["silhouette"],
                          "DBSCAN noise fraction": cluster["selected"]["noise_fraction"],
                          "DBSCAN resample ARI": cluster["purged_10_percent_ari"]})
        for search in trial["rules"]:
            for rule in search["rules"]:
                valid = rule["validation"]
                rules.append({"Fold": trial["fold"], "Search": search["mode"], "Rule": rule["label"],
                              "Validation matches": valid["matches"], "Validation lift": valid["lift"]})
    st.dataframe(pd.DataFrame(table), hide_index=True)
    with st.expander("V2 Apriori validation results"):
        st.dataframe(pd.DataFrame(rules), hide_index=True)
