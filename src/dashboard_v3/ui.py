"""Visual tokens and responsive components for the V3 workbench."""

from html import escape

import plotly.graph_objects as go
import streamlit as st

TEAL, NAVY, AMBER, RED = "#087F70", "#163450", "#97671D", "#AD4E42"
COLORS = {"xgboost": TEAL, "logistic": "#496D9A", "forest": "#7860A5", "torch": "#AE6A33",
          "v1_xgboost": "#718292", "v2_ensemble": "#9C4A70"}
STYLE = """
<style>
:root { --fl-ink:#122840; --fl-muted:#53677b; --fl-edge:#dee7ef; --fl-accent:#087f70; }
.stApp {background:#f4f7fa;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stMainBlockContainer"] {max-width: 1430px; padding:3.1rem 2.5rem 3.5rem;}
section[data-testid="stSidebar"] {background:#102337; border-right:1px solid #294257; color:#e6eef7;}
[data-testid="stSidebarContent"] {padding: .4rem .25rem;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:.34rem;}
[data-testid="stSidebar"] [role="radiogroup"] label {margin:0; padding:.58rem .55rem; border-radius:.5rem; border:1px solid transparent;}
[data-testid="stSidebar"] [role="radiogroup"] label p {color:#e6eef7; font-weight:550; font-size:.91rem;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#1b3b4b; border-color:#2a5860;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#1b3b4b;}
[data-testid="stSidebar"] hr {border-color:#345066;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p, [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {color:#c1d1e0;}
[data-testid="stSidebarCollapseButton"] button {color:#d9e9f3;}
[data-testid="stSidebarCollapsedControl"] button {color:#163450;}
.fl-brand {display:flex; align-items:center; gap:.7rem; color:#f5fbff; margin:.2rem 0 1.45rem;}
.fl-mark {display:grid; place-items:center; width:2.55rem; height:2.55rem; border-radius:.72rem;
          background:#d0ebe4; color:#12483f; font-size:.78rem; font-weight:800; letter-spacing:-.02em;}
.fl-brand strong {display:block; font-size:1.17rem; letter-spacing:-.025em; line-height:1.17;}
.fl-brand small {display:block; margin-top:.14rem; color:#a9c0d1; font-size:.75rem;}
.fl-side-line {color:#bfd2e3; font-size:.83rem; line-height:1.55;}
.fl-side-line strong {display:block; color:#e7f2fb; font-size:.85rem;}
.fl-breadcrumb {font-size:.76rem; font-weight:700; color:#597184; letter-spacing:.075em; text-transform:uppercase; margin:0 0 .65rem;}
.fl-hero {position:relative; display:flex; align-items:center; justify-content:space-between; gap:1.2rem;
          border-radius:1.04rem; padding:1.85rem 2rem; margin:0 0 .2rem; color:#fff; background:#112940; border:1px solid #244459;}
.fl-hero h1 {color:#fff; font-size:clamp(1.95rem,4.1vw,3.15rem); letter-spacing:-.057em; line-height:1.08; margin:0; padding:0;}
.fl-hero p {color:#c9d9e7; line-height:1.55; margin:.62rem 0 0; max-width:63ch; font-size:.98rem;}
.fl-hero-tag {align-self:flex-start; border:1px solid #4c707e; border-radius:2rem; white-space:nowrap; color:#c8ede5;
              background:#1a3a4a; font-size:.77rem; padding:.36rem .75rem; font-weight:650;}
.fl-card-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(min(165px,100%),1fr)); gap:.7rem;}
.fl-card {background:#fff; border:1px solid var(--fl-edge); border-radius:.78rem; padding:.9rem 1rem;}
.fl-card-title {font-size:.81rem; color:var(--fl-muted); font-weight:650;}
.fl-card-value {font-size:clamp(1.38rem,2.5vw,2rem); color:var(--fl-ink); font-weight:750; letter-spacing:-.05em; font-variant-numeric:tabular-nums; line-height:1.22; margin:.25rem 0;}
.fl-card-detail {font-size:.75rem; line-height:1.35; color:var(--fl-muted);}
.fl-card:first-child {border-top:3px solid #128877; padding-top:calc(.9rem - 2px);}
.fl-policy {display:flex; flex-wrap:wrap; gap:.32rem 1.1rem; padding:.68rem .86rem; color:#30576e; font-size:.82rem; line-height:1.5;
            background:#e9f2f6; border:1px solid #d6e8ef; border-radius:.7rem;}
.fl-policy strong {color:#15374e;}
.fl-queue, .fl-foot {font-size:.78rem; color:var(--fl-muted); line-height:1.5;}
[data-testid="stVerticalBlockBorderWrapper"] {background:#fff; border-color:var(--fl-edge); border-radius:.8rem;}
[data-testid="stMainBlockContainer"] h2 {letter-spacing:-.037em; color:var(--fl-ink); padding:.1rem 0 .18rem;}
[data-testid="stMainBlockContainer"] h3 {letter-spacing:-.025em; color:var(--fl-ink); padding:.15rem 0 .25rem;}
[data-testid="stCaptionContainer"] p {color:var(--fl-muted);}
[data-testid="stDataFrame"] {background:#fff; border-radius:.55rem;}
[data-testid="stTabs"] [role="tablist"] {gap:.55rem;}
[data-testid="stTabs"] [role="tab"] {white-space:nowrap;}
@media (max-width: 1000px) { [data-testid="stMainBlockContainer"] {padding:3rem 1.5rem 2.5rem;} }
@media (max-width: 760px) {
  [data-testid="stMainBlockContainer"] {padding:2.85rem .9rem 2rem;}
  .fl-hero {align-items:flex-start; flex-direction:column; padding:1.24rem 1rem; gap:.75rem;}
  .fl-hero p {font-size:.91rem;}
  [data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] {flex-wrap:wrap; gap:.8rem;}
  [data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {min-width:min(215px,100%); flex:1 1 215px;}
}
</style>
"""


def hero(title, description, period="V3 retrospective"):
    st.html(f'<div class="fl-breadcrumb">Fraud intelligence / {escape(title)}</div><section class="fl-hero">'
            f'<div><h1>{escape(title)}</h1><p>{escape(description)}</p></div>'
            f'<span class="fl-hero-tag">{escape(period)}</span></section>')


def cards(items):
    rows = []
    for label, value, detail in items:
        rows.append(f'<div class="fl-card"><div class="fl-card-title">{escape(str(label))}</div>'
                    f'<div class="fl-card-value">{escape(str(value))}</div>'
                    f'<div class="fl-card-detail">{escape(str(detail))}</div></div>')
    st.html('<div class="fl-card-grid">' + "".join(rows) + '</div>')


def policy_strip():
    st.html('<div class="fl-policy"><strong>Frozen decision policy</strong><span>Depth-10 XGBoost</span>'
            '<span>Threshold <b>0.1938110143</b></span><span>Identity calibration</span></div>')


def chart(figure, key, *, x="", y="", height=340, percent_x=False, percent_y=False):
    figure.update_layout(height=height, margin=dict(l=12, r=12, t=12, b=12), paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#38536c", size=12),
                         legend=dict(orientation="h", y=1.03, yanchor="bottom", x=0),
                         hoverlabel=dict(bgcolor="#102337", font_color="white"), autosize=True)
    figure.update_xaxes(title=x, gridcolor="#e2eaf1", zerolinecolor="#bdcbd8", automargin=True,
                        tickformat=".0%" if percent_x else None)
    figure.update_yaxes(title=y, gridcolor="#e2eaf1", zerolinecolor="#bdcbd8", automargin=True,
                        tickformat=".0%" if percent_y else None)
    st.plotly_chart(figure, width="stretch", theme=None, config={"displayModeBar": False, "scrollZoom": False}, key=key)


def hbar(names, values, color=TEAL):
    return go.Figure(go.Bar(x=list(values), y=list(names), orientation="h", marker_color=color,
                            hovertemplate="%{y}<br>%{x:.4f}<extra></extra>"))


def pct(value, digits=2):
    return "—" if value is None else f"{value:.{digits}%}"
