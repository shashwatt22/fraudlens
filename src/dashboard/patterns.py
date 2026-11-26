"""Anomaly, clustering, association mining, and model-attribution views."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .artifacts import ArtifactError, generation_commands
from .discovery import association_view, cluster_view
from .ui import BLUE, TEAL, chart_layout, hero, histogram, plot
from .v2_results import patterns as render_v2


def _anomaly(run, partition):
    forest = run.manifests[3]["anomaly"]
    settings = forest["settings"]
    label = run.partition_name(partition)
    summary = st.columns(3)
    summary[0].metric("Training reference rows", f"{forest['reference_rows']:,}", border=True)
    summary[1].metric("Isolation trees", f"{settings['n_estimators']:,}", border=True)
    summary[2].metric("Numeric inputs and missing flags", len(forest["transformed_features"]), border=True)
    st.write("Isolation Forest finds observations that are easier to isolate with random splits. "
             "The percentile compares each score with a fixed training reference and gives ties the same midrank. "
             "An unusually high percentile can occur on a legitimate transaction.")
    training = run.table(3, "train_anomaly", ["anomaly_score", "anomaly_percentile"])
    evaluation = run.table(3, f"{partition}_anomaly", ["anomaly_score", "anomaly_percentile"])
    for column, title, key, bounds in zip(st.columns(2, gap="medium"), ("Raw anomaly scores", "Anomaly percentiles"),
                                          ("anomaly_score", "anomaly_percentile"), (None, (0, 1)), strict=True):
        with column, st.container(border=True):
            st.subheader(title)
            st.caption(f"Training reference and {label.lower()} · normalized histogram share")
            combined = pd.concat([training[key], evaluation[key]]).to_numpy(dtype=float, na_value=np.nan)
            finite = combined[np.isfinite(combined)]
            span = bounds or ((float(finite.min()), float(finite.max())) if finite.size else None)
            fig = go.Figure()
            for values, name, color in ((training[key], "Training reference", BLUE), (evaluation[key], label, TEAL)):
                hist = histogram(values, color, bounds=span)
                if hist is not None:
                    trace = hist.data[0]
                    trace.y = trace.y / trace.y.sum()
                    trace.update(name=name, opacity=.63, hovertemplate="%{x:.4f}<br>%{y:.2%} of rows<extra>%{fullData.name}</extra>")
                    fig.add_trace(trace)
            fig.update_layout(barmode="overlay")
            fig.update_yaxes(tickformat=".0%")
            plot(chart_layout(fig, height=275, x=title, y="Share of rows") if fig.data else None, f"pattern_{key}")


def _explainability(run):
    importance = run.report(3, "global_shap_importance")
    with st.container(border=True):
        st.subheader("Global grouped SHAP importance")
        st.caption(f"{importance['rows']:,} sampled validation rows · mean absolute signed group sum · log-odds units")
        top = pd.DataFrame(importance["grouped"]).head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=top["mean_absolute_shap"], y=top["display_name"], orientation="h", marker_color=TEAL,
                               hovertemplate="%{y}<br>%{x:.6f} log-odds<extra></extra>"))
        plot(chart_layout(fig, height=370, x="Mean absolute grouped SHAP"), "pattern_shap")
        st.caption("These attributions describe the saved classifier on the sampled rows. Correlated features can share attribution.")


def _discovery(run, view):
    try:
        run.require_patterns()
        view(run)
    except (ArtifactError, KeyError, TypeError, ValueError) as error:
        st.warning(f"Phase 5 results are unavailable for this mode. {error}")
        st.code(generation_commands(run.mode, 5), language="powershell")


def render(run, partition):
    hero("Behavioral Intelligence & Patterns", "Explore unusual transactions, training-sample clusters, and fraud associations evaluated on validation.", run.mode)
    anomaly, clustering, association, explainability = st.tabs(
        ["Anomaly Intelligence", "Behavioral Clustering", "Association Rules", "Global Explainability"]
    )
    with anomaly:
        _anomaly(run, partition)
    with clustering:
        _discovery(run, cluster_view)
    with association:
        _discovery(run, association_view)
    with explainability:
        _explainability(run)
    render_v2(run, partition)
