"""Render the saved Phase 5 experiments without running their algorithms."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .artifacts import ArtifactError
from .ui import AMBER, BLUE, TEAL, chart_layout, decimal, percent, plot

NOISE = "#8c99ac"
CLUSTER_COLORS = (TEAL, BLUE, AMBER, "#a3a0ef", "#5ccadf", "#e996bb", "#acd58a")


def _label(cluster: int) -> str:
    return "Noise" if cluster == -1 else f"Cluster {cluster}"


def _color(cluster: int) -> str:
    return NOISE if cluster == -1 else CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]


def cluster_view(run):
    report = run.report(5, "clustering")
    samples = run.table(5, "clusters", ["cluster", "pca_1", "pca_2"])
    profiles = run.table(5, "profiles")
    if len(samples) != report["sample_rows"] or profiles["rows"].sum() != len(samples) or report["partition"] != "train":
        raise ArtifactError(5, "Cluster assignments, profiles and training metadata do not agree.")
    selected = report["selected"]
    st.subheader("DBSCAN training-sample clusters")
    st.caption(f"{len(samples):,} of {report['training_population_rows']:,} training transactions · uniform sample, seed {report['seed']}")
    first, second, third, fourth = st.columns(4, gap="medium")
    first.metric("Clusters", selected["clusters"], border=True)
    second.metric("Clustered rows", f"{selected['clustered']:,}", border=True)
    third.metric("Noise rows", f"{selected['noise']:,}", help=f"{selected['noise_fraction']:.2%} of the sample", border=True)
    fourth.metric("Inlier silhouette", decimal(selected["silhouette"], 3), border=True)
    st.write(f"Selected **ε {selected['eps']:g}** and **min_samples {selected['min_samples']}**. "
             "Epsilon sets the neighborhood radius in scaled feature space. Min samples sets the count needed for a core point. "
             "Noise observations could not be reached from a core point.")

    clusters = sorted(int(value) for value in profiles["cluster"])
    with st.container(border=True):
        st.subheader("PCA cluster map")
        kept = st.multiselect("Show groups", clusters, format_func=_label, placeholder="All groups", key="cluster_filter")
        fig = go.Figure()
        for cluster in kept or clusters:
            group = samples.loc[samples["cluster"].eq(cluster)]
            fig.add_scattergl(x=group["pca_1"], y=group["pca_2"], mode="markers", name=_label(cluster),
                              marker={"color": _color(cluster), "size": 5, "opacity": .58},
                              hovertemplate="PC 1 %{x:.2f}<br>PC 2 %{y:.2f}<extra>%{fullData.name}</extra>")
        first_pc, second_pc = report["pca_explained_variance_ratio"]
        plot(chart_layout(fig, height=390, x=f"PC 1 · {percent(first_pc, 1)} variance",
                          y=f"PC 2 · {percent(second_pc, 1)} variance"), "dbscan_pca")
        st.caption(report["clustering_space"] + " Cluster labels describe this sample. New transactions have no DBSCAN assignment.")

    names = profiles["cluster"].map(_label)
    colors = profiles["cluster"].map(_color)
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        st.subheader("Cluster size")
        chart = go.Figure(go.Bar(x=names, y=profiles["rows"], marker_color=colors, text=profiles["rows"], textposition="auto"))
        plot(chart_layout(chart, y="Sample rows"), "dbscan_size")
    with right, st.container(border=True):
        st.subheader("Observed fraud by group")
        chart = go.Figure(go.Bar(x=names, y=profiles["fraud_rate"], marker_color=colors, customdata=profiles["fraud_rows"],
                                hovertemplate="%{x}<br>%{y:.2%} labeled fraud · %{customdata} rows<extra></extra>"))
        chart.update_yaxes(tickformat=".1%")
        plot(chart_layout(chart, y="Observed fraud share"), "dbscan_fraud")
    st.caption("Fraud labels were used after clustering to describe each group.")

    st.subheader("Behavioral profiles")
    display = profiles.assign(cluster=names).rename(columns={
        "cluster": "Group", "rows": "Rows", "fraud_rows": "Fraud rows", "fraud_rate": "Fraud rate",
        "median_amount": "Median amount", "median_prior_count": "Median previous count",
        "median_amount_ratio": "Median amount ratio", "median_gap_seconds": "Median gap (s)",
        "median_prior_24h": "Median 24-hour count", "cold_start_share": "Cold starts",
    })
    st.dataframe(display, hide_index=True, column_config={
        name: st.column_config.NumberColumn(format="percent") for name in ("Fraud rate", "Cold starts")
    } | {name: st.column_config.NumberColumn(format="%.2f") for name in ("Median amount", "Median amount ratio")})
    with st.expander("Training features and configuration comparison"):
        st.write(report["preprocessing"])
        st.write(" · ".join(report["raw_features"]))
        st.caption(report["selection"])
        grid = pd.DataFrame(report["comparisons"]).rename(columns={"eps": "Epsilon", "min_samples": "Min samples",
                             "noise_fraction": "Noise share", "clusters": "Clusters", "silhouette": "Inlier silhouette",
                             "status": "Status", "neighbor_links": "Neighbor links"})
        shown = [name for name in ("Epsilon", "Min samples", "Clusters", "Noise share", "Inlier silhouette", "Neighbor links", "Status")
                 if name in grid]
        st.dataframe(grid[shown], hide_index=True, column_config={"Noise share": st.column_config.NumberColumn(format="percent")})


def association_view(run):
    report = run.report(5, "association")
    if report["discovery_partition"] != "train" or report["evaluation_partition"] != "validation":
        raise ArtifactError(5, "Association discovery and evaluation partitions are incompatible.")
    rules, conditions, settings = report["rules"], report["conditions"], report["settings"]
    train, validation = report["training_baseline"], report["validation_baseline"]
    st.subheader("Apriori fraud association rules")
    st.caption("Rules were discovered on a natural-prevalence training sample and evaluated unchanged on validation.")
    first, second, third = st.columns(3, gap="medium")
    first.metric("Fraud-associated rules", len(rules), border=True)
    second.metric("Training baseline fraud", percent(train["fraud_rate"]), help=f"{train['rows']:,} sampled rows", border=True)
    third.metric("Validation baseline fraud", percent(validation["fraud_rate"]), help=f"{validation['rows']:,} rows", border=True)
    st.caption(f"Training sample {train['rows']:,} · Validation {validation['rows']:,} · "
               f"Minimum training support {settings['min_support']:.2%}, confidence {settings['min_confidence']:.1%}, "
               f"{settings['min_matches']:,} antecedent matches and {settings['min_fraud_matches']:,} matching fraud rows.")
    with st.expander("Frozen condition definitions and metric guide"):
        st.write(f"Numeric cutoffs were fitted on {report['conditions_fitted_on_rows']:,} training rows. "
                 "History-dependent conditions require an observed value and earlier activity.")
        definitions = [{"Condition": item["label"], "Rule": item["basis"],
                        "Cutoff": ("Unavailable" if item["cutoff"] is None else
                                   f"{'≥' if item['operation'] == 'ge' else '='} {float(item['cutoff']):.7g}")}
                       for item in conditions]
        st.dataframe(pd.DataFrame(definitions), hide_index=True)
        st.write("**Support** is the share of all rows matching the conditions and fraud. **Confidence** is the share of "
                 "matching rows labeled fraud. **Lift** divides confidence by the fraud prevalence of the same partition.")
    if not rules:
        st.info("No fraud rules met the fixed training support, confidence and occurrence requirements.")
        return

    query = st.text_input("Search antecedent conditions", placeholder="For example, amount", key="apriori_search")
    matching = [rule for rule in rules if query.strip().casefold() in rule["label"].casefold()]
    if not matching:
        st.info("No mined rules match this search.")
        return
    rows = []
    for rule in matching:
        record = {"Antecedent → fraud": rule["label"]}
        for key, name in (("train", "Train"), ("validation", "Validation")):
            record[f"{name} matches"] = rule[key]["matches"]
            record[f"{name} fraud"] = rule[key]["fraud_matches"]
            record[f"{name} lift"] = rule[key]["lift"]
        rows.append(record)
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={name: st.column_config.NumberColumn(format="%.3f")
                 for name in ("Train lift", "Validation lift")})
    selected = st.selectbox("Inspect a rule", range(len(matching)), format_func=lambda index: matching[index]["label"],
                            key="apriori_rule")
    rule = matching[selected]
    details = []
    for key, name in (("train", "Training sample"), ("validation", "Validation")):
        stats = rule[key]
        details.append({"Partition": name, "Support": stats["support"], "Confidence": stats["confidence"], "Lift": stats["lift"],
                        "Matches": stats["matches"], "Fraud matches": stats["fraud_matches"], "Fraud baseline": stats["fraud_rate"]})
    with st.container(border=True):
        st.markdown(f"**{rule['label']} → fraud**")
        st.write(rule["interpretation"])
        if rule["validation"]["lift"] is not None and rule["validation"]["lift"] < 1 <= rule["train"]["lift"]:
            st.caption("Fraud occurred more often than baseline among training matches and less often than baseline among validation matches.")
        st.dataframe(pd.DataFrame(details), hide_index=True, column_config={
            name: st.column_config.NumberColumn(format="percent") for name in ("Support", "Confidence", "Fraud baseline")
        } | {"Lift": st.column_config.NumberColumn(format="%.3f")})
        st.caption("These associations describe observed rows and do not establish causes or change classifier predictions.")
