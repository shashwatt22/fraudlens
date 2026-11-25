"""Run the local, read-only FraudLens application with Streamlit."""

import os
from pathlib import Path

import streamlit as st

from src.dashboard import investigator, model_lab, overview, patterns
from src.dashboard.artifacts import ArtifactError, available_modes, generation_commands, load_run
from src.dashboard.ui import STYLE

PAGES = {"Overview & Review Queue": overview.render, "Transaction Investigator": investigator.render,
         "Model Lab": model_lab.render, "Behavioral Intelligence & Patterns": patterns.render}


def main() -> None:
    st.set_page_config(page_title="FraudLens", layout="wide", initial_sidebar_state="auto")
    st.html(STYLE)
    root = Path(os.environ.get("FRAUDLENS_ARTIFACTS", Path(__file__).parent / "artifacts")).resolve()
    test_name = {"quick": "Quick test", "full": "Final test", "smoke": "Test"}
    with st.sidebar:
        st.html('<div class="fl-brand"><span class="fl-logo" aria-hidden="true">FL</span>'
                '<span class="fl-name">FraudLens</span></div>'
                '<p class="fl-brand-subtitle">Financial Fraud &amp; Behavioral Risk Intelligence</p>')
        page = st.radio("Navigation", tuple(PAGES), key="navigation", label_visibility="collapsed")
        st.divider()
        mode = st.selectbox("Dataset mode", available_modes(root), format_func=str.capitalize, key="dataset_mode")
        partition = st.radio("Evaluation partition", ("validation", "test"), key="evaluation_partition",
                             format_func=lambda item: test_name[mode] if item == "test" else item.capitalize())
        st.divider()
        st.caption("IEEE-CIS research dataset · Educational prototype")

    try:
        run = load_run(root, mode)
        PAGES[page](run, partition)
    except (ArtifactError, KeyError, TypeError, ValueError) as error:
        phase = error.phase if isinstance(error, ArtifactError) else 3
        st.error(f"Dashboard data unavailable. {error}")
        st.markdown("Use a matching set of offline runs. From the repository, generate the missing stage and any missing prerequisites.")
        st.code(generation_commands(mode, phase), language="powershell")


main()
