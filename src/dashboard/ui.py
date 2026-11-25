"""Shared layout and Plotly styling for the local dashboard."""

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

TEAL, BLUE, AMBER, RED, SLATE = "#43c9bd", "#74a9fa", "#f4bb63", "#ff7684", "#9aaeca"
PRIORITY_COLORS = {"Low": TEAL, "Medium": AMBER, "High": RED}

STYLE = """
<style>
:root { --fl-line: #253650; --fl-muted: #a4b5cc; --fl-surface: #111c2d; }
.stApp { background-color: #0b1220; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 1450px; padding: 3.65rem 2.65rem 3rem; }
[data-testid="stSidebar"] { border-right: 1px solid var(--fl-line); }
[data-testid="stSidebarContent"] { padding-top: .3rem; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .4rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: .45rem .55rem; border: 1px solid transparent; border-radius: .5rem; align-items: center;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: #192b42; border-color: #2d465f; }
[data-testid="stSidebar"] [role="radiogroup"] label p { font-size: .88rem; }
[data-testid="stMetric"] { background: var(--fl-surface); border-radius: .62rem; }
[data-testid="stMetricLabel"] p { color: #afc0d8; font-size: .83rem; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; letter-spacing: -.035em; font-size: clamp(1.3rem, 2.3vw, 2.0rem); }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--fl-line); }
[data-testid="stCaptionContainer"] { color: var(--fl-muted); }
.fl-brand { display: flex; align-items: center; gap: .7rem; margin: .5rem 0 .55rem; }
.fl-logo { width: 2.25rem; height: 2.25rem; flex: 0 0 auto; border-radius: .55rem; display: grid; place-items: center;
    background: #153844; color: #6ee2d9; border: 1px solid #24535d; font-size: .83rem; font-weight: 750; }
.fl-name {font-size: 1.36rem; font-weight: 770; letter-spacing: -.045em; }
.fl-brand-subtitle { color: var(--fl-muted); font-size: .85rem; line-height: 1.48; max-width: 17rem; margin-bottom: 1rem; }
.fl-hero { display: flex; justify-content: space-between; align-items: start; gap: 1rem; margin: 0 0 .6rem; }
.fl-hero h1 { margin: 0; padding: 0; font-size: clamp(1.9rem, 3.5vw, 3.1rem); letter-spacing: -.055em; line-height: 1.14; }
.fl-hero p { margin: .58rem 0 0; color: var(--fl-muted); line-height: 1.52; max-width: 62ch; }
.fl-mode { white-space: nowrap; border: 1px solid #28565e; color: #83e2d9; background: #14343d;
    padding: .33rem .62rem; border-radius: .38rem; font-size: .72rem; font-weight: 720; letter-spacing: .065em; }
.fl-architecture { display: grid; grid-template-columns: repeat(auto-fit,minmax(135px,1fr)); gap: .7rem; margin: .25rem 0; }
.fl-layer { background: var(--fl-surface); border: 1px solid var(--fl-line); border-top: 2px solid #43c9bd;
    border-radius: .55rem; padding: .75rem .9rem; }
.fl-layer strong { display: block; font-size: 1.15rem; }
.fl-layer span { color: var(--fl-muted); font-size: .8rem; }
@media (max-width: 900px) { [data-testid="stMainBlockContainer"] { padding-left: 1.3rem; padding-right: 1.3rem; } }
@media (max-width: 800px) {
    [data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: .9rem; }
    [data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 195px; min-width: min(195px, 100%);
    }
}
@media (max-width: 540px) { .fl-hero { flex-direction: column; } }
</style>
"""


def hero(title: str, description: str, mode: str) -> None:
    st.html(f'<section class="fl-hero"><div><h1>{escape(title)}</h1><p>{escape(description)}</p></div>'
            f'<span class="fl-mode">{escape(mode.upper())} MODE</span></section>')


def percent(value, digits=2) -> str:
    return "Unavailable" if pd.isna(value) else f"{value:.{digits}%}"


def decimal(value, digits=4) -> str:
    return "Unavailable" if pd.isna(value) else f"{value:,.{digits}f}"


def count(value) -> str:
    return "Unavailable" if pd.isna(value) else f"{int(value):,}"


def chart_layout(fig: go.Figure, height=260, x="", y="") -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=12, r=14, t=14, b=12), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#dce7f7", size=12),
                      legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0),
                      hoverlabel=dict(bgcolor="#18293f", font_color="#f2f6ff"))
    fig.update_xaxes(title_text=x, gridcolor="#24344b", zerolinecolor="#536680", automargin=True)
    fig.update_yaxes(title_text=y, gridcolor="#24344b", zerolinecolor="#536680", automargin=True)
    return fig


def histogram(values, color=TEAL, bins=32, bounds=None) -> go.Figure | None:
    numeric = np.asarray(values, dtype=np.float64)
    numeric = numeric[np.isfinite(numeric)]
    if not numeric.size:
        return None
    counts, edges = np.histogram(numeric, bins=bins, range=bounds)
    return go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges) * .88,
                            marker_color=color, hovertemplate="%{x:.3g}<br>%{y:,} transactions<extra></extra>"))


def plot(fig, key: str) -> None:
    if fig is None:
        st.info("No measured values are available for this chart.")
    else:
        st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False, "scrollZoom": False}, key=key)
