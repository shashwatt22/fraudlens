"""Launch the local FraudLens V3 Streamlit workbench."""

from pathlib import Path

import streamlit as st

from src.dashboard_v3 import alert_ui, behavior, cases, models, replay, replay_ui
from src.dashboard_v3.store import DataUnavailable, Paths, bundle
from src.dashboard_v3.ui import STYLE


PAGES = {
    "Command Center": cases.command_center,
    "Live Monitor": replay_ui.render,
    "Alert Center": alert_ui.render,
    "Transaction Investigator": cases.investigator,
    "Model Intelligence": models.render,
    "Behavioral Analytics": behavior.render,
}


def main():
    st.set_page_config(page_title="FraudLens | Fraud Intelligence", page_icon="🛡️", layout="wide",
                       initial_sidebar_state="auto")
    st.html(STYLE)
    paths = Paths.from_env(Path(__file__).resolve().parents[2])
    public = alert_ui.public_demo()
    with st.sidebar:
        st.html('<div class="fl-brand"><span class="fl-mark" aria-hidden="true">FL</span>'
                '<div><strong>FraudLens</strong><small>Fraud intelligence · V3</small></div></div>')
        page = st.radio("Workspace navigation", PAGES.keys(), key="v3_page", label_visibility="collapsed")
        st.divider()
        st.html('<div class="fl-side-line"><strong>Local research workspace</strong>'
                'IEEE-CIS Fraud Detection<br>Frozen V3 evaluation</div>')
        if public:
            st.caption("Public demo · aggregate view")
            private = True
        else:
            private = st.toggle("Hide local case details", value=st.query_params.get("aggregate") == "1", key="v3_private",
                                help="Use aggregate-only view when sharing the screen or capturing screenshots.")
    if private:
        alert_ui.privacy(force_dry_run=public)
    if page != "Live Monitor" or private:
        replay.pause_session()
    pack = None
    issue = None
    try:
        pack = bundle(paths)
    except DataUnavailable as error:
        issue = str(error)

    PAGES[page](paths, pack, private, issue)
    st.html('<div class="fl-foot">FraudLens V3 · Academic research prototype · IEEE-CIS / Vesta via Kaggle</div>')


if __name__ == "__main__":
    main()
