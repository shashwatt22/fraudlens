"""Portfolio metrics and a ranked, transaction-level review queue."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .investigator import select_case
from .logic import CORE_MODELS, ID, MODEL_NAMES, RANK_COLUMNS, TIME, parse_id, review_rows
from .ui import AMBER, BLUE, PRIORITY_COLORS, RED, SLATE, TEAL, chart_layout, hero, histogram, percent, plot
from .v2_results import review_queue as render_v2


def _activity(run, risk, partition):
    selected = run.partition_name(partition)
    all_rows = pd.concat([run.table(1, split, [TIME, "TransactionAmt"]) for split in run.partitions], ignore_index=True)
    elapsed = all_rows[TIME].to_numpy() / 86400
    counts, edges = np.histogram(elapsed, bins=48)
    activity = go.Figure(go.Scatter(x=(edges[:-1] + edges[1:]) / 2, y=counts, mode="lines", fill="tozeroy",
                                    line=dict(color=TEAL, width=2.6), fillcolor="rgba(67,201,189,.13)",
                                    hovertemplate="Elapsed day %{x:.1f}<br>%{y:,} transactions<extra></extra>"))
    left, right = st.columns([1.48, 1], gap="medium")
    with left, st.container(border=True):
        st.subheader("Transaction activity")
        st.caption("All chronological partitions · relative elapsed time")
        plot(chart_layout(activity, x="Elapsed days from dataset origin", y="Transactions"), "overview_activity")
    with right, st.container(border=True):
        st.subheader("Review priorities")
        st.caption(f"{selected} · illustrative priority bands")
        labels = list(PRIORITY_COLORS)
        distribution = risk["review_priority"].value_counts().reindex(labels, fill_value=0)
        priority = go.Figure(go.Bar(x=labels, y=distribution.to_list(), marker_color=list(PRIORITY_COLORS.values()),
                                    text=[f"{n:,}" for n in distribution], textposition="auto"))
        plot(chart_layout(priority, y="Transactions"), "overview_priority")

    left, right = st.columns(2, gap="medium")
    sizes = list(run.partitions.values())
    fraud = sum(item["fraud_rows"] for item in sizes)
    total = len(all_rows)
    with left, st.container(border=True):
        st.subheader("Observed labels")
        st.caption("Fraud and legitimate transactions across the processed run")
        donut = go.Figure(go.Pie(labels=["Legitimate", "Fraud"], values=[total - fraud, fraud], hole=.72,
                                 marker_colors=[BLUE, RED], sort=False, textinfo="none",
                                 hovertemplate="%{label}<br>%{value:,} · %{percent}<extra></extra>"))
        donut.add_annotation(text=percent(fraud / total), x=.5, y=.5, showarrow=False, font=dict(size=22))
        plot(chart_layout(donut), "overview_labels")
    with right, st.container(border=True):
        st.subheader("Fraud prevalence by partition")
        names = [run.partition_name(split) for split in run.partitions]
        rates = [item["fraud_rate"] for item in sizes]
        bars = go.Figure(go.Bar(x=names, y=rates, marker_color=[BLUE, TEAL, SLATE],
                                 text=[percent(rate) for rate in rates], textposition="auto"))
        bars.update_yaxes(tickformat=".1%")
        plot(chart_layout(bars, y="Observed fraud share"), "overview_prevalence")

    st.subheader("Amount and score distributions")
    amounts = all_rows["TransactionAmt"].dropna().to_numpy()
    amount_plot = None
    if amounts.size:
        cutoff = float(np.quantile(amounts, .99))
        amount_plot = histogram(amounts[amounts <= cutoff], color=BLUE)
        amount_scope = f"All partitions · up to the 99th percentile ({cutoff:,.2f})"
    else:
        amount_scope = "All partitions"
    for column, title, subtitle, figure, axis, key in zip(
        st.columns(3, gap="medium"), ("Transaction amount", "Classifier scores", "Composite risk"),
        (amount_scope, f"{selected} · uncalibrated scores", f"{selected} · 0–100 review score"),
        (amount_plot, histogram(risk["classifier_score"], BLUE, bounds=(0, 1)),
         histogram(risk["risk_score"], AMBER, bounds=(0, 100))),
        ("Amount", "Classifier score", "Composite risk"), ("amount", "score", "risk"), strict=True,
    ):
        with column, st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(subtitle)
            plot(chart_layout(figure, height=220, x=axis) if figure is not None else None, f"overview_{key}")


def _queue(run, risk, partition):
    st.subheader(f"{run.partition_name(partition)} review queue")
    st.caption("Classifier score is the default ranking because it had the highest average precision in the validation ranking experiment.")
    first, second, third, fourth = st.columns([1.2, 1.15, 1.35, .72], gap="medium")
    ranking = first.selectbox("Rank by", tuple(RANK_COLUMNS), key="queue_rank")
    query = second.text_input("Search transaction ID", placeholder="Enter an exact ID", key="queue_search")
    priorities = third.multiselect("Review priority", tuple(PRIORITY_COLORS), placeholder="All priorities", key="queue_priority")
    limit = fourth.selectbox("Rows shown", (15, 30, 60, 100), key="queue_limit")
    show_truth = st.toggle("Show observed labels for evaluation", key="queue_truth")
    if query.strip() and parse_id(query) is None:
        st.info("Enter a positive numeric transaction ID.")
        return
    view = review_rows(risk, ranking, priorities, query, limit)
    if view.empty:
        st.info("No transactions match these filters in the selected partition.")
        return
    columns = [ID, "TransactionAmt", "classifier_score", "classifier_decision", "anomaly_percentile",
               "risk_score", "review_priority"]
    export = view[columns].copy()
    export["classifier_decision"] = export["classifier_decision"].map({0: "Unflagged", 1: "Flagged"})
    if show_truth:
        export["Observed label"] = view["isFraud"].map({0: "Legitimate", 1: "Fraud"})
    st.dataframe(export, hide_index=True, height=min(555, 38 + 35 * len(export)), column_config={
        ID: st.column_config.NumberColumn("Transaction ID", format="%d"),
        "TransactionAmt": st.column_config.NumberColumn("Amount", format="%.2f"),
        "classifier_score": st.column_config.NumberColumn("Classifier score", format="%.4f"),
        "classifier_decision": "Frozen decision", "review_priority": "Review priority",
        "anomaly_percentile": st.column_config.NumberColumn("Anomaly percentile", format="percent"),
        "risk_score": st.column_config.NumberColumn("Composite risk / 100", format="%.2f"),
    })
    choose, open_column, download = st.columns([1.2, 1, 1], vertical_alignment="bottom")
    selected = choose.selectbox("Transaction to investigate", view[ID].to_list(), key="queue_case")
    open_column.button("Open in Investigator", type="primary", key="open_case",
                       on_click=select_case, args=(run.mode, partition, selected, True))
    download.download_button("Download shown rows", data=export.to_csv(index=False), mime="text/csv",
                             file_name=f"fraudlens-{run.mode}-{partition}-review.csv", on_click="ignore")
    row_label = "row" if len(export) == 1 else "rows"
    st.caption(f"Showing {len(export):,} {row_label}. The frozen classifier decision and illustrative review priority are separate fields.")


def render(run, partition):
    hero("Overview & Review Queue", "Explore chronological fraud activity, compare risk signals and open a transaction for investigation.", run.mode)
    risk = run.risk(partition)
    rankings = run.report(3, "rankings")["rankings"]
    sizes = run.partitions
    total = sum(item["rows"] for item in sizes.values())
    fraud = sum(item["fraud_rows"] for item in sizes.values())
    metrics = st.columns(4, gap="medium")
    metrics[0].metric("Processed transactions", f"{total:,}", border=True)
    metrics[1].metric("Observed fraud prevalence", percent(fraud / total), border=True)
    core_count = sum(key in run.manifests[2]["models"] for key in CORE_MODELS if key != "mlp")
    metrics[2].metric("Core models · neural benchmarks", f"{core_count} · 1", border=True)
    metrics[3].metric("Deployed classifier · validation AP", f"{rankings['classifier']['average_precision']:.5f}", border=True)
    counts = "  ·  ".join(f"{run.partition_name(split)} {item['rows']:,}" for split, item in sizes.items())
    st.caption(f"{counts}  ·  Deployed model {MODEL_NAMES[run.manifests[2]['champion']]}")
    st.caption(f"{run.partition_name(partition)} · {int(risk['classifier_decision'].sum()):,} flagged by the frozen classifier")
    overview, queue = st.tabs(["Portfolio overview", "Review queue"])
    with overview:
        _activity(run, risk, partition)
    with queue:
        _queue(run, risk, partition)
        render_v2(run, partition, select_case)
