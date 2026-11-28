"""Archived pattern studies and on-demand global SHAP for the V3 primary model."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.artifacts import ArtifactError, load_run
from . import explanations, store
from .store import DataUnavailable
from .ui import AMBER, NAVY, TEAL, cards, chart, hbar, hero, pct

GROUP_COLORS = (TEAL, "#496D9A", "#7860A5", "#AE6A33", "#9C4A70")


def _archive(paths):
    run = load_run(paths.legacy, "full")
    if run.digests[1] != store.V1_SHA:
        raise DataUnavailable("The archived study’s V1 lineage differs from the V3 source.")
    return run


def _anomaly(run, _private):
    st.subheader("Isolation Forest")
    anomaly = run.manifests[3]["anomaly"]
    st.caption("V1 Phase 3 · Isolation Forest fitted to a bounded training sample and benchmarked against its training reference.")
    cards([
        ("Fit sample", f"{anomaly['fit_rows']:,}", "Training only"),
        ("Training reference", f"{anomaly['reference_rows']:,}", "Percentile reference"),
        ("Isolation trees", f"{anomaly['settings']['n_estimators']:,}", "Archived unsupervised model"),
    ])
    sample = st.radio("Compare training against", ("V1 validation", "Reused retrospective holdout"), horizontal=True, key="v3_anomaly_sample")
    part = "validation" if sample == "V1 validation" else "test"
    training = run.table(3, "train_anomaly", ["anomaly_percentile"])
    compared = run.table(3, f"{part}_anomaly", ["anomaly_percentile"])
    st.caption(f"Training reference compared with {sample.lower()} · {len(compared):,} comparison rows")
    with st.container(border=True):
        figure = go.Figure()
        bins = np.linspace(0, 1, 26)
        for frame, label, color in ((training, "V1 training reference", NAVY), (compared, sample, TEAL)):
            counts, _ = np.histogram(frame["anomaly_percentile"].dropna(), bins=bins)
            figure.add_bar(x=(bins[1:] + bins[:-1]) / 2, y=counts / counts.sum(), name=label,
                           marker_color=color, opacity=.65, width=.036)
        figure.update_layout(barmode="overlay")
        chart(figure, "v3_isolation", x="Training-reference percentile", y="Share of rows",
              percent_x=True, percent_y=True)
    st.caption("Anomaly percentiles measure unusual behavior relative to the training reference. They use a separate score from the V3 classifier.")


def _clusters(run, private):
    st.subheader("DBSCAN & PCA")
    report = run.report(5, "clustering")
    profile = run.table(5, "profiles")
    selected = report["selected"]
    st.caption(f"V1 Phase 5 · {report['sample_rows']:,} sampled from {report['training_population_rows']:,} training transactions.")
    cards([
        ("Clusters", selected["clusters"], "Inlier groups"),
        ("Noise share", pct(selected["noise_fraction"]), f"{selected['noise']:,} sample rows"),
        ("Inlier silhouette", f"{selected['silhouette']:.3f}" if selected["silhouette"] is not None else "—", "Selection diagnostic"),
        ("PCA variance", pct(sum(report["pca_explained_variance_ratio"])), "Two display components"),
    ])
    names = {int(group): "Noise" if group == -1 else f"Cluster {group}" for group in profile["cluster"]}
    if private:
        st.info("Individual PCA points are hidden in aggregate-only view.")
        fig = hbar([names[int(i)] for i in profile["cluster"]], profile["rows"])
        chart(fig, "v3_cluster_aggregate", x="Sample rows")
    else:
        samples = run.table(5, "clusters", ["cluster", "pca_1", "pca_2"])
        if len(samples) != report["sample_rows"] or profile["rows"].sum() != len(samples):
            raise DataUnavailable("The cluster report and coordinates disagree.")
        shown = st.multiselect("Show groups", list(names), format_func=names.__getitem__, key="v3_cluster_groups",
                               placeholder="All groups")
        fig = go.Figure()
        for group in shown or names:
            subset = samples.loc[samples["cluster"] == group]
            fig.add_scattergl(x=subset["pca_1"], y=subset["pca_2"], name=names[group], mode="markers",
                              marker={"size": 5, "color": "#8899a9" if group == -1 else GROUP_COLORS[group % len(GROUP_COLORS)], "opacity": .55},
                              hovertemplate="PC 1 %{x:.2f}<br>PC 2 %{y:.2f}<extra>%{fullData.name}</extra>")
        first, second = report["pca_explained_variance_ratio"]
        with st.container(border=True):
            chart(fig, "v3_pca", x=f"PC 1 · {first:.1%} variance", y=f"PC 2 · {second:.1%} variance", height=380)
    st.caption("DBSCAN groups were formed in standardized input space. The PCA projection is a visualization of that sample.")
    st.dataframe(profile[["cluster", "rows", "fraud_rows", "fraud_rate"]].assign(cluster=profile["cluster"].map(names))
                 .rename(columns={"cluster": "Group", "rows": "Rows", "fraud_rows": "Labeled fraud", "fraud_rate": "Fraud share"}),
                 hide_index=True, width="stretch", column_config={"Fraud share": st.column_config.NumberColumn(format="percent")})


def _apriori(run, _private):
    st.subheader("Apriori association rules")
    report = run.report(5, "association")
    st.caption(f"V1 Phase 5 · discovered in {report['training_baseline']['rows']:,} sampled training rows and checked on "
               f"{report['validation_baseline']['rows']:,} validation rows.")
    query = st.text_input("Search rule conditions", placeholder="For example, amount", key="v3_rule_search")
    rules = [row for row in report["rules"] if query.casefold().strip() in row["label"].casefold()]
    if not rules:
        st.info("No saved rules match the search.")
        return
    rows = [{"Rule → fraud": rule["label"], "Training lift": rule["train"]["lift"], "Validation lift": rule["validation"]["lift"],
             "Validation matches": rule["validation"]["matches"], "Validation confidence": rule["validation"]["confidence"]} for rule in rules]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Validation confidence": st.column_config.NumberColumn(format="percent")})
    choice = st.selectbox("Inspect a saved rule", list(range(len(rules))), format_func=lambda i: rules[i]["label"], key="v3_rule")
    st.write(rules[choice]["interpretation"])
    st.caption("Lift compares fraud frequency among matching rows with the partition baseline. The rules describe associations in the saved populations.")


def _archived_shap(run, _private):
    st.subheader("Archived global SHAP")
    summary = run.report(3, "global_shap_importance")
    st.caption(f"V1 Phase 3 · XGBoost · {summary['rows']:,} sampled validation observations.")
    top = list(reversed(summary["grouped"][:12]))
    chart(hbar([row["display_name"] for row in top], [row["mean_absolute_shap"] for row in top], NAVY),
          "v3_v1_shap", x="Mean absolute grouped SHAP · log-odds", height=375)


def _v2_patterns(paths):
    with st.expander("Archived V2 development patterns"):
        if not st.toggle("Load V2 pattern summary", key="v3_v2_patterns"):
            return
        try:
            source = store.report(paths.v2 / "patterns.json")
            st.caption("V2 · three expanding development folds. Isolation Forest AP is measured on validation; DBSCAN profiles use training samples.")
            rows, rules = [], []
            for fold in source["folds"]:
                for family, values in fold["families"].items():
                    cluster = values["clusters"]
                    for anomaly in values["anomaly"]:
                        rows.append({"Fold": fold["fold"], "Representation": family, "Tree sample": anomaly["tree_samples"],
                                     "Isolation Forest AP": anomaly["average_precision"], "DBSCAN silhouette": cluster["selected"]["silhouette"]})
                for mode in fold["rules"]:
                    for rule in mode["rules"]:
                        rules.append({"Fold": fold["fold"], "Search": mode["mode"], "Rule": rule["label"],
                                      "Validation matches": rule["validation"]["matches"], "Validation lift": rule["validation"]["lift"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            if rules:
                st.dataframe(pd.DataFrame(rules), hide_index=True, width="stretch")
        except (DataUnavailable, OSError, ValueError, KeyError) as error:
            st.info(f"V2 pattern archive unavailable. {error}")


def _global(paths, pack, private, issue):
    st.subheader("V3 global SHAP")
    st.caption("Refitted depth-10 XGBoost · 24 evenly spaced rows from the reused retrospective holdout · exact TreeSHAP.")
    st.caption("This bounded sample describes average attribution magnitude. It can vary with the sample and feature correlations.")
    if pack is None:
        st.warning(f"Frozen V3 results unavailable. {issue}")
        return
    if st.button("Calculate global TreeSHAP", key="v3_global_button", type="primary"):
        st.session_state["v3_global_ready"] = True
    if not st.session_state.get("v3_global_ready"):
        return
    try:
        with st.spinner("Attributing the bounded holdout sample…"):
            result = explanations.global_summary(paths, store.queue(paths, pack))
    except (DataUnavailable, ValueError, OSError, RuntimeError) as error:
        st.warning(f"Global TreeSHAP unavailable. {error}")
        return
    names = [explanations.FEATURE_LABELS.get(name, name) for name in reversed(result["names"])]
    chart(hbar(names, list(reversed(result["values"]))), "v3_global_shap", x="Mean absolute contribution · log-odds", height=415)
    st.caption(f"{result['samples']} rows explained · Maximum score reconstruction error {result['score_error']:.2g}")


def render(paths, pack, private, issue):
    hero("Behavioral Analytics", "Explore archived patterns and inspect the features behind the final primary model.", "V3 & research archive")
    sections = {"Isolation Forest": _anomaly, "DBSCAN & PCA": _clusters, "Apriori rules": _apriori, "V1 global SHAP": _archived_shap}
    choice = st.radio("Behavioral analysis", ("V3 global SHAP", *sections), horizontal=True, key="v3_behavior_view")
    if choice == "V3 global SHAP":
        _global(paths, pack, private, issue)
    else:
        try:
            sections[choice](_archive(paths), private)
        except (ArtifactError, DataUnavailable, OSError, ValueError, KeyError) as error:
            st.info(f"Archived analysis unavailable. {error}")
    _v2_patterns(paths)
