"""Command Center, local review queue, and transaction investigation."""

from math import ceil, isfinite

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import AMOUNT, ID, PRODUCT, TIME
from . import explanations, store
from .store import DECISION, SCORE, DataUnavailable
from .ui import RED, TEAL, cards, chart, hbar, hero, pct, policy_strip


def navigate(tx_id=None, replay_score=None):
    st.session_state["v3_case_input"] = "" if tx_id is None else str(tx_id)
    st.session_state["v3_active_case"] = tx_id
    st.session_state["v3_page"] = "Transaction Investigator"
    st.session_state["v3_replay_case"] = (tx_id, replay_score) if replay_score is not None else None


def _dataset(paths, pack):
    population = pack["population"]
    try:
        source, fit = store.source_manifest(paths), store.policy_plan(paths)
    except DataUnavailable as error:
        st.info(f"Dataset breakdown unavailable. {error}")
        return
    if (fit["rows"] + population["rows"] != source["processed_rows"]
            or source["partitions"]["test"]["fraud_rows"] != population["fraud_rows"]):
        st.warning("The dataset and V3 evaluation manifests do not agree.")
        return
    st.subheader("Dataset & chronology")
    total_frauds = sum(part["fraud_rows"] for part in source["partitions"].values())
    cards([
        ("IEEE-CIS transactions", f"{source['source_rows']['transactions']:,}", f"{total_frauds:,} labeled fraud across all partitions"),
        ("Identity records", f"{source['source_rows']['identities']:,}", "Optional identity table"),
        ("Pre-policy population", f"{fit['partitions']['pre_policy']['rows']:,}", "Estimator fit prefix; MLP reserves internal validation"),
        ("Decision-policy block", f"{fit['partitions']['policy']['rows']:,}", "Development and verification"),
    ])
    with st.expander("Chronological partitions"):
        rows = []
        for label, window in (("Pre-policy population", fit["partitions"]["pre_policy"]),
                              ("Policy development", fit["partitions"]["earlier_policy"]),
                              ("Policy verification", fit["partitions"]["later_policy"]),
                              ("Reused retrospective holdout", source["partitions"]["test"])):
            rows.append({"Population": label, "Rows": window["rows"], "Fraud rows": window["fraud_rows"],
                         "First second": window["first_second"], "Last second": window["last_second"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("TransactionDT is elapsed seconds from the dataset reference, with equal timestamps kept together.")


def _reset_page():
    st.session_state["v3_queue_page"] = 1


def _queue(paths, pack, private):
    st.subheader("Transaction review queue")
    st.caption("Local retrospective records · ranked by the frozen primary model")
    if private:
        st.info("Local case details are hidden. Turn off the sidebar privacy switch to open the review queue.")
        policy = pack["models"]["xgboost"]["frozen"]
        _, false_alarms = policy["confusion_matrix"][0]
        missed, detected = policy["confusion_matrix"][1]
        cards([("Frauds flagged", f"{detected:,}", "Observed fraud above the frozen threshold"),
               ("False alarms", f"{false_alarms:,}", "Observed legitimate above the frozen threshold"),
               ("Frauds below threshold", f"{missed:,}", "Observed fraud outside the review queue")])
        return
    try:
        frame = store.queue(paths, pack)
    except DataUnavailable as error:
        st.warning(f"Local review queue unavailable. {error}")
        return

    with st.container(border=True):
        query = st.text_input("Search transaction ID", placeholder="Exact ID across the holdout",
                              key="v3_queue_query", on_change=_reset_page)
        cols = st.columns(4)
        searching = bool(query.strip())
        decision = cols[0].selectbox("Decision", ("Flagged", "All decisions", "Below threshold"),
                                     key="v3_queue_decision", disabled=searching, on_change=_reset_page)
        products = sorted(map(str, frame[PRODUCT].dropna().unique()))
        product = cols[1].selectbox("Product", ["All products", *products],
                                    key="v3_queue_product", disabled=searching, on_change=_reset_page)
        minimum = cols[2].number_input("Minimum risk", 0.0, 1.0, 0.0, .05, format="%.2f",
                                       key="v3_queue_min", disabled=searching, on_change=_reset_page)
        order = cols[3].selectbox("Sort by", ("Highest risk", "Largest amount", "Newest", "Oldest"),
                                  key="v3_queue_order", disabled=searching, on_change=_reset_page)
        try:
            matched = store.filter_queue(frame, query=query, decision=decision, product=product, minimum=minimum, order=order)
        except ValueError as error:
            st.warning(str(error))
            return
        if matched.empty:
            st.info("No transactions match these criteria.")
            return
        if searching:
            st.caption("Exact-ID search covers all decisions and products.")
        size = 12
        number = ceil(len(matched) / size)
        page = 1
        if number > 1:
            st.session_state.setdefault("v3_queue_page", 1)
            if st.session_state["v3_queue_page"] > number:
                _reset_page()
            page = st.number_input("Queue page", 1, number, step=1, key="v3_queue_page")
        shown = matched.iloc[(page - 1) * size:page * size]
        st.caption(f"{len(matched):,} matching · showing {(page - 1) * size + 1:,}–{min(page * size, len(matched)):,}")
        display = shown[[ID, TIME, AMOUNT, PRODUCT, SCORE]].rename(columns={
            ID: "Transaction ID", TIME: "Elapsed seconds", AMOUNT: "Amount", PRODUCT: "Product", SCORE: "Risk score",
        }).assign(Decision=np.where(shown[DECISION], "Flagged", "Below threshold"))
        st.dataframe(display, hide_index=True, width="stretch", height=min(460, 37 + 36 * len(display)), column_config={
            "Transaction ID": st.column_config.NumberColumn(format="%d"),
            "Elapsed seconds": st.column_config.NumberColumn(format="%d"),
            "Amount": st.column_config.NumberColumn(format="%.2f"),
            "Risk score": st.column_config.NumberColumn(format="percent"),
        })
        picker, action = st.columns([3, 2], vertical_alignment="bottom")
        selected = picker.selectbox("Transaction to investigate", shown[ID].astype(int).to_list(), key="v3_queue_pick")
        action.button("Open investigation", type="primary", on_click=navigate, args=(selected,), width="stretch")


def command_center(paths, pack, private, issue):
    hero("Command Center", "Triage high-risk transactions and track the final model’s performance on a fixed historical window.")
    policy_strip()
    if pack is None:
        st.warning(f"V3 metrics and review queue unavailable. {issue}")
        return
    primary = pack["models"]["xgboost"]
    policy, population = primary["frozen"], pack["population"]
    cards([
        ("Average precision", f"{primary['average_precision']:.5f}", "Original score ranking"),
        ("ROC-AUC", f"{primary['roc_auc']:.5f}", "Original score ranking"),
        ("Frozen-policy F1", f"{policy['f1']:.5f}", f"{pct(policy['precision'])} precision · {pct(policy['recall'])} recall"),
        ("Flagged for review", f"{policy['reviewed']:,}", f"{policy['review_rate']*1000:.1f} per 1,000 transactions"),
    ])
    st.caption(f"Reused retrospective holdout · {population['rows']:,} transactions · "
               f"{population['fraud_rows']:,} labeled fraud ({pct(population['fraud_rows']/population['rows'])}) · "
               "Previously evaluated in V1 and V2")
    _queue(paths, pack, private)
    _dataset(paths, pack)


def _select_case():
    st.session_state.pop("v3_replay_case", None)
    st.session_state["v3_active_case"] = st.session_state["v3_case_input"]


def _history(paths, position):
    st.subheader("Behavioral context")
    st.caption("Card–address–D1 entity proxy · history strictly before the current timestamp group")
    try:
        missing, values = explanations.history(paths, position)
    except (DataUnavailable, OSError, ValueError) as error:
        st.warning(f"Behavioral inputs unavailable. {error}")
        return
    def show(value, digits=0):
        return f"{value:,.{digits}f}" if isfinite(value) else "Unavailable"

    if missing:
        st.info("This transaction lacks one or more components of the entity proxy.")
    elif values["Prior transactions"] == 0:
        st.info("The complete entity proxy has no previous transactions in the available chronology.")
    cards([(name, show(values[name], 0), "Earlier transactions") for name in ("Prior transactions", "Prior hour", "Prior 24 hours")])
    detail = [{"Measure": name, "Value": show(value, 0 if name.startswith("Seconds") else 2)}
              for name, value in values.items() if name not in ("Prior transactions", "Prior hour", "Prior 24 hours")]
    st.dataframe(pd.DataFrame(detail), hide_index=True, width="stretch")


def _explain(paths, position, value, tx_id):
    st.subheader("Why the model scored it this way")
    st.caption("Exact TreeSHAP · refitted V3 depth-10 XGBoost · contribution to raw log-odds")
    if st.button("Calculate TreeSHAP", type="primary", key="v3_local_explain"):
        st.session_state["v3_explained_case"] = tx_id
    if st.session_state.get("v3_explained_case") != tx_id:
        return
    try:
        with st.spinner("Calculating exact tree contributions…"):
            result = explanations.local(paths, position, value)
    except (DataUnavailable, OSError, ValueError, RuntimeError) as error:
        st.warning(f"TreeSHAP unavailable. {error}")
        return
    names = [explanations.FEATURE_LABELS.get(name, name) for name in reversed(result["names"])]
    values = list(reversed(result["values"]))
    fig = hbar(names, values, [RED if score > 0 else TEAL for score in values])
    chart(fig, "v3_local_shap", x="Signed contribution to model log-odds", height=400)
    st.caption(f"Base {result['base']:.4f} · Remaining features {result['remaining']:+.4f} · Raw margin {result['margin']:.4f} · "
               f"Score reconstruction error {result['score_error']:.2g}")
    st.caption("Positive contributions raise model risk. Anonymized IEEE-CIS fields keep their original names.")


def investigator(paths, pack, private, issue):
    hero("Transaction Investigator", "Inspect a transaction’s frozen decision, prior behavior and model explanation.")
    policy_strip()
    if private:
        st.info("Local case details are hidden. Turn off the sidebar privacy switch to investigate a transaction.")
        with st.container(border=True):
            st.subheader("Investigation workflow")
            st.markdown("1. Open a flagged case from the review queue or look up its transaction ID.\n"
                        "2. Inspect the frozen score and behavior before that timestamp.\n"
                        "3. Calculate TreeSHAP and reveal the research label when needed.")
        return
    if pack is None:
        st.warning(f"V3 transaction investigation unavailable. {issue}")
        return
    with st.form("v3_case_form", border=True):
        st.session_state.setdefault("v3_case_input", str(st.session_state.get("v3_active_case") or ""))
        st.text_input("Transaction ID", key="v3_case_input", placeholder="Enter a holdout transaction ID")
        st.form_submit_button("Investigate transaction", on_click=_select_case)
    active = st.session_state.get("v3_active_case")
    if active in (None, ""):
        st.info("Open a case from the Command Center or enter an exact ID.")
        return
    try:
        tx_id = store.transaction_id(str(active))
        frame = store.queue(paths, pack)
    except (DataUnavailable, ValueError) as error:
        st.warning(str(error))
        return
    hits = np.flatnonzero(frame[ID].to_numpy() == tx_id)
    if not len(hits):
        st.info("That transaction is outside the saved retrospective holdout.")
        return
    position = int(hits[0])
    row = frame.iloc[position]
    score, timestamp = float(row[SCORE]), int(row[TIME])
    st.subheader(f"Transaction {tx_id}")
    replay_case = st.session_state.get("v3_replay_case")
    if replay_case is not None and replay_case[0] == tx_id:
        if abs(score - replay_case[1]) > 2e-6:
            st.warning("Replay inference and the frozen evaluation score disagree for this transaction.")
        else:
            st.caption("Opened from Live Monitor · replay inference matches the frozen case score.")
    cards([
        ("Frozen V3 risk score", pct(score, 4), "Identity-calibrated output"),
        ("Model decision", "Flagged" if row[DECISION] else "Below threshold", "Frozen threshold 0.1938110143"),
        ("Transaction amount", f"{row[AMOUNT]:,.2f}" if pd.notna(row[AMOUNT]) else "Unavailable",
         f"Product {row[PRODUCT]}" if pd.notna(row[PRODUCT]) else "Product unavailable"),
        ("Elapsed seconds", f"{timestamp:,}", "Identity record missing" if row["identity_missing"] else "Identity record available"),
    ])
    with st.container(border=True):
        st.markdown("**Observed fraud label · research dataset**")
        st.caption("The observed label records the dataset outcome. The model decision shows which cases the model would flag.")
        if st.checkbox("Reveal observed label", key=f"v3_label_{tx_id}"):
            try:
                label = store.observed_label(paths, tx_id, timestamp)
                st.write("Observed label: **Fraud**" if label else "Observed label: **Legitimate**")
            except (DataUnavailable, OSError, ValueError) as error:
                st.warning(str(error))
    _history(paths, position)
    _explain(paths, position, score, tx_id)
