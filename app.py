"""Launch the two-page local FraudLens workbench."""

from html import escape
from pathlib import Path

import streamlit as st

from public_demo import app as public_app
from src.dashboard_v3 import alert_ui, behavior, manual_enhanced, manual_model, manual_ui, models
from src.dashboard_v3.store import DataUnavailable, Paths, bundle, report
from src.dashboard_v3.ui import STYLE, cards, hero, pct

PAGES = ("Dashboard", "Check Transaction & Alerts")
PRODUCT_STYLE = """
<style>
[data-testid="stMainBlockContainer"] {max-width: 1680px; padding:2.05rem 2.5rem 3rem;}
.fl-breadcrumb {display:none;}
.fl-hero {padding:1.2rem 1.65rem; border-left:4px solid #51bca8;}
.fl-hero h1 {font-size:clamp(1.9rem,4vw,2.85rem); line-height:1.16;}
.fl-hero p {margin:.32rem 0 0; font-size:1rem;}
[data-testid="stCaptionContainer"] {color:#4e6379 !important; opacity:1 !important;}
[data-testid="stCaptionContainer"] p, .stCaptionContainer p {color:#4e6379 !important; font-size:.85rem;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:#c4d4e3 !important;}
.st-key-fl_device [role="radiogroup"], .st-key-fl_recipient [role="radiogroup"] {flex-wrap:wrap; row-gap:.25rem;}
@media (max-width:760px) {
  [data-testid="stMainBlockContainer"] {padding:2.6rem .9rem 2rem;}
  .fl-hero {padding:1rem 1.1rem;}
  .fl-hero p {font-size:.95rem;}
}
</style>
"""


def _activity():
    manual, alerts = manual_ui.session(), alert_ui.session()
    alerts.refresh()
    cards([
        ("Manual assessments", f"{manual.total:,}", f"{manual.flagged:,} flagged this session"),
        ("Notifications", f"{alerts.created:,}", f"{alerts.counts['Dry run']:,} dry-run records"),
        ("Emails accepted", f"{alerts.counts['SMTP accepted']:,}", f"{alerts.counts['Failed']:,} failed · {alerts.counts['Sending']:,} sending"),
    ])
    if manual.recent:
        record = manual.recent[-1]
        latest = "Flagged" if record.result.flagged else "Not flagged"
        st.caption(f"Latest manual assessment · {record.reference} · {latest} · {record.created:%H:%M:%S} UTC")


def _manual_metrics(paths):
    try:
        directory, enhanced = manual_enhanced.selected(paths.v3)
        data = report(directory / "manifest.json")
        if enhanced and data["model"] == manual_enhanced.MODEL and data["decision"]["promote"]:
            window = data["windows"]["later_policy"]["population"]
            row = data["windows"]["later_policy"]["challenger"]
            description = "Manual model with optional context"
        elif not enhanced and data["model"] == manual_model.MODEL:
            window = data["metrics"]["development_verification"]
            row = window["selected"]
            description = "Four-input manual model"
        else:
            raise ValueError("Invalid manual model manifest.")
    except (DataUnavailable, ValueError, KeyError, TypeError):
        st.caption("Manual model metrics are unavailable in this workspace.")
        return
    st.subheader("Manual model validation")
    st.caption(f"{description} · {window['rows']:,} later development rows · review threshold {pct(data['threshold'], 3)}.")
    cards([
        ("Average precision", f"{row['average_precision']:.5f}", "Manual model ranking"),
        ("ROC-AUC", f"{row['roc_auc']:.5f}", "Manual model ranking"),
        ("Precision", pct(row["precision"]), "Share of flags labeled fraud"),
        ("Recall", pct(row["recall"]), "Share of labeled fraud flagged"),
    ])
    st.caption("The manual model uses fewer inputs and a different evaluation period. A flag calls for review.")


def _academic(paths, pack):
    panel = st.expander("Research methods, learning curves and supplementary analysis", on_change="rerun")
    if panel.open:
        with panel:
            models._neural(paths)
            models._experiments(paths)
            if pack is not None:
                models._entities(pack)
            behavior._v2_patterns(paths)
            st.caption("The archived research workbench retains historical replay, case-level SHAP, clustering, anomaly detection and association rules.")


def dashboard(paths, private):
    hero("Dashboard", "Fraud detection results and recent transaction activity.", "Public demo" if private else "Local workspace")
    try:
        overview = public_app.results()
        pack = overview if private else bundle(paths)
    except (DataUnavailable, ValueError, OSError, KeyError):
        st.warning("Verified historical metrics are unavailable.")
        pack = None
    if pack is not None:
        public_app.overview(overview)
    if not private:
        st.subheader("Recent activity")
        _activity()
    if pack is not None:
        public_app.research_results(pack)
    if not private:
        _manual_metrics(paths)
        _academic(paths, pack)
    st.caption("FraudLens is an academic project based on the IEEE-CIS benchmark. The local checker uses a separate model designed around manually supplied inputs.")


def main():
    st.set_page_config(page_title="FraudLens | Transaction Screening", layout="wide", initial_sidebar_state="auto")
    st.html(STYLE + PRODUCT_STYLE)
    paths = Paths.from_env(Path(__file__).resolve().parent)
    private = alert_ui.public_demo() or st.query_params.get("aggregate") == "1"
    if private:
        alert_ui.privacy(force_dry_run=True)
    with st.sidebar:
        st.html('<div class="fl-brand"><span class="fl-mark" aria-hidden="true">FL</span>'
                '<div><strong>FraudLens</strong><small>Transaction screening</small></div></div>')
        page = st.radio("Navigation", PAGES, key="fl_product_nav", label_visibility="collapsed")
        st.divider()
        st.html(f'<div class="fl-side-line"><strong>{escape("Public demo" if private else "Local workspace")}</strong>'
                'IEEE-CIS research benchmark</div>')
    if page == PAGES[0]:
        dashboard(paths, private)
    elif private:
        public_app.public_check(None)
    else:
        hero("Check Transaction & Alerts", "Assess a transaction and manage its follow-up.", "Manual assessment")
        manual_ui.render(paths)


if __name__ == "__main__":
    main()
