"""Transaction assessment, explicit label reveal, and on-demand TreeSHAP."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .artifacts import ArtifactError, explain_case, generation_commands
from .logic import ID, demo_cases, outcome, parse_id
from .ui import BLUE, PRIORITY_COLORS, RED, chart_layout, count, decimal, hero, percent, plot


def _input_key(mode, partition):
    return f"case_id_input_{mode}_{partition}"


def select_case(mode, partition, transaction_id, navigate=False):
    key = (mode, partition, int(transaction_id))
    ids = dict(st.session_state.get("case_ids", {}))
    ids[(mode, partition)] = key[2]
    st.session_state["case_ids"] = ids
    st.session_state[_input_key(mode, partition)] = str(key[2])
    st.session_state["revealed_case"] = None
    st.session_state["explained_case"] = None
    if navigate:
        st.session_state["navigation"] = "Transaction Investigator"


def _show_assessment(run, partition, row):
    cols = st.columns([1, 1, 1, 1.35], gap="medium")
    threshold = run.manifests[3]["classifier"]["threshold"]
    with cols[0]:
        st.metric("Transaction ID", str(int(row[ID])), border=True)
        st.metric("Transaction amount", decimal(row["TransactionAmt"], 2), border=True)
    with cols[1]:
        st.metric("Classifier score", decimal(row["classifier_score"]), border=True)
        st.metric("Frozen decision", "Flagged" if row["classifier_decision"] else "Unflagged", border=True)
    with cols[2]:
        st.metric("Anomaly percentile", percent(row["anomaly_percentile"]), border=True)
        st.metric("Frozen threshold", decimal(threshold), border=True)
    priority = row["review_priority"]
    with cols[3], st.container(border=True):
        st.markdown("**Composite risk**")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=row["risk_score"], number=dict(suffix=" / 100", font=dict(size=26)),
            gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 40, 75, 100]), borderwidth=0,
                       bar=dict(color=PRIORITY_COLORS[priority]), steps=[dict(range=[0, 40], color="#15333d"),
                       dict(range=[40, 75], color="#44371f"), dict(range=[75, 100], color="#45272f")]),
        ))
        plot(chart_layout(gauge, height=118), "case_risk_gauge")
        st.caption(f"{priority} review priority · illustrative band")

    raw = run.table(1, partition, [ID, "ProductCD", "DeviceType", "DeviceInfo"], int(row[ID]))
    if len(raw) != 1:
        raise ArtifactError(1, "Transaction metadata is unavailable in the matching Phase 1 partition.")
    raw = raw.iloc[0]
    info = st.columns([1.35, 1, 1, 1.45], gap="medium")
    for column, label, value in zip(info, ("Elapsed timestamp", "Product category", "Device type", "Identity record"),
                                    (f"{int(row['TransactionDT']):,} s", raw["ProductCD"], raw["DeviceType"],
                                     "No match" if row["identity_missing"] else "Matched"), strict=True):
        column.caption(label)
        column.text("Unavailable" if pd.isna(value) else str(value))
    if pd.notna(raw["DeviceInfo"]):
        st.caption("Device information")
        st.text(str(raw["DeviceInfo"]))
    st.caption("Time is relative to the dataset origin. The frozen classifier score and composite risk are uncalibrated decision-support signals.")


def _show_history(row):
    st.subheader("Historical behavior")
    if row["prior_txn_count"] == 0:
        st.info("Cold start. No strictly earlier transactions were found for this approximate entity.")
    for widgets, details in (
        (st.columns(4), [("Prior transactions", count(row["prior_txn_count"])), ("Prior mean amount", decimal(row["prior_mean_amount"], 2)),
                         ("Current / prior mean", "Unavailable" if pd.isna(row["amount_to_prior_mean"]) else f"{row['amount_to_prior_mean']:.2f}×"),
                         ("Seconds since prior", count(row["seconds_since_prior_txn"]))]),
        (st.columns([1, 1, 2]), [("Prior 1-hour count", count(row["prior_count_1h"])),
                                ("Prior 24-hour count", count(row["prior_count_24h"])),
                                ("Previously observed product", "Unavailable" if pd.isna(row["is_new_product"])
                                 else ("No" if row["is_new_product"] else "Yes"))]),
    ):
        for widget, (label, value) in zip(widgets, details, strict=True):
            widget.metric(label, value, border=True)
    st.caption("History uses a card-and-address proxy and strictly earlier timestamps. A proxy can merge or fragment histories.")


def _show_shap(run, partition, case):
    st.subheader("Classifier explanation")
    if st.button("Explain classifier score", key="explain_case", type="primary"):
        st.session_state["explained_case"] = case
    if st.session_state.get("explained_case") != case:
        st.caption("Compute exact TreeSHAP for the selected transaction.")
        return
    try:
        with st.spinner("Computing TreeSHAP for this transaction"):
            data = explain_case(run.root, run.mode, run.digests[3], partition, case[2])
    except ArtifactError as error:
        st.warning(str(error))
        st.code(generation_commands(run.mode), language="powershell")
        return
    groups = pd.DataFrame(data["grouped_contributions"])
    visible = groups.head(12).sort_values("contribution")
    values = visible["contribution"]
    chart = go.Figure(go.Bar(x=values, y=visible["display_name"], orientation="h",
                             marker_color=[RED if value > 0 else BLUE for value in values],
                             customdata=visible["feature"],
                             hovertemplate="%{y}<br>%{x:+.5f} log-odds<extra></extra>"))
    with st.container(border=True):
        st.caption("Largest signed contributions · red raises the classifier margin · blue lowers it")
        plot(chart_layout(chart, height=370, x="Contribution to raw model margin (log-odds)"), "case_shap")
        cols = st.columns(3)
        for col, label, field in zip(cols, ("Base model output", "Reconstructed margin", "Reconstructed score"),
                                     ("base_value", "raw_margin", "reconstructed_score"), strict=True):
            col.metric(label, decimal(data[field], 6), border=True)
    upward = groups.loc[groups["contribution"].gt(0)].nlargest(1, "contribution")
    downward = groups.loc[groups["contribution"].lt(0)].nsmallest(1, "contribution")
    if not upward.empty:
        top = upward.iloc[0]
        st.write(f"{top['display_name']} raised the model margin most among the groups shown ({top['contribution']:+.4f} log-odds).")
    if not downward.empty:
        bottom = downward.iloc[0]
        st.write(f"{bottom['display_name']} lowered the model margin most ({bottom['contribution']:+.4f} log-odds).")
    st.caption("TreeSHAP describes the saved classifier’s output. A contribution is a model attribution, with no causal interpretation.")
    with st.expander("All transformed-feature contributions"):
        transformed = pd.DataFrame(data["transformed_contributions"])
        transformed = transformed.iloc[transformed["contribution"].abs().sort_values(ascending=False).index]
        st.dataframe(transformed[["name", "display_name", "contribution"]], hide_index=True, height=315,
                     column_config={"name": "Transformed feature", "display_name": "Original feature",
                                    "contribution": st.column_config.NumberColumn("Log-odds contribution", format="%+.6f")})


def render(run, partition):
    hero("Transaction Investigator", "Inspect a saved assessment, trace earlier behavior and explain the deployed classifier’s score.", run.mode)
    risk = run.risk(partition)
    partition_name = run.partition_name(partition)
    demo = demo_cases(risk)
    ids = st.session_state.get("case_ids", {})
    selected = ids.get((run.mode, partition), demo.get("Behavioral context", int(risk[ID].iloc[0])))
    key = _input_key(run.mode, partition)
    st.session_state.setdefault(key, str(selected))
    left, right = st.columns([1, 1], gap="medium")
    with left, st.form(f"case_form_{run.mode}_{partition}", border=True):
        entered = st.text_input(f"{partition_name} transaction ID", key=key)
        submitted = st.form_submit_button("Load transaction", type="primary")
    with right, st.container(border=True):
        label = st.selectbox("Curated demonstration example", tuple(demo), key=f"demo_{run.mode}_{partition}")
        st.button("Load example", key="load_example", on_click=select_case, args=(run.mode, partition, demo[label]))
    st.caption("These curated examples illustrate observed outcomes, unusual activity and available history. Use Model Lab for performance estimates.")
    if submitted:
        number = parse_id(entered)
        if number is None or not risk[ID].eq(number).any():
            st.warning(f"That transaction ID was not found in {partition_name}. Enter a valid ID from the selected partition.")
        else:
            selected = number
            st.session_state["case_ids"] = {**ids, (run.mode, partition): number}
            st.session_state["revealed_case"] = None
            st.session_state["explained_case"] = None
    matches = risk.loc[risk[ID].eq(selected)]
    if matches.empty:
        st.info(f"Select a transaction from the {partition_name} partition.")
        return
    row = matches.iloc[0]
    case = (run.mode, partition, selected)
    _show_assessment(run, partition, row)
    _show_history(row)
    _show_shap(run, partition, case)
    st.subheader("Observed outcome")
    if st.button("Reveal Ground Truth", key="reveal_truth"):
        st.session_state["revealed_case"] = case
    if st.session_state.get("revealed_case") == case:
        observed = "Fraud" if row["isFraud"] else "Legitimate"
        st.info(f"Observed label: {observed} · {outcome(row['isFraud'], row['classifier_decision'])}")
    else:
        st.caption("The observed label is hidden until you reveal it.")
