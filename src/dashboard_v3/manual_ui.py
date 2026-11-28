"""Transaction form, verified assessment, and bounded session history."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import streamlit as st

from . import alert_mail, alert_ui, manual_enhanced, manual_model
from .ui import RED, TEAL, cards, chart, hbar, pct

SESSION = "fl_manual"
EXAMPLES = {
    "Debit · $49.95": (49.95, "visa", "debit", "google"),
    "Credit · $450": (450.0, "mastercard", "credit", "unknown"),
    "Credit · $899.99": (899.99, "discover", "credit", "unknown"),
}
WIDGETS = ("fl_amount", "fl_network", "fl_card_type", "fl_provider")
EXTRA_WIDGETS = ("fl_device", "fl_recipient")
MODEL_ERRORS = (OSError, ValueError, KeyError, RuntimeError)


@dataclass(frozen=True)
class Record:
    reference: str
    created: datetime
    result: manual_model.AssessmentScore


class AssessmentSession:
    def __init__(self):
        self.recent = deque(maxlen=25)
        self.total = self.flagged = 0

    def save(self, result):
        record = Record(f"MA-{uuid4().hex[:16].upper()}", datetime.now(timezone.utc), result)
        self.recent.append(record)
        self.total += 1
        self.flagged += int(result.flagged)
        return record


def session():
    if SESSION not in st.session_state:
        st.session_state[SESSION] = AssessmentSession()
    return st.session_state[SESSION]


@st.cache_resource(show_spinner=False, max_entries=1)
def _load(directory, signature, enhanced):
    engine = manual_model.AssessmentModel(directory, manual_enhanced if enhanced else None)
    if enhanced and not engine.manifest["decision"]["promote"]:
        raise ValueError("The optional-context model has not passed its promotion checks.")
    return engine


def _example(name, enhanced):
    for key, value in zip(WIDGETS, EXAMPLES[name], strict=True):
        st.session_state[key] = value
    if enhanced:
        for key in EXTRA_WIDGETS:
            st.session_state[key] = manual_model.UNKNOWN


def _form(enhanced):
    for key, value in zip(WIDGETS, (None, "unknown", "unknown", "unknown"), strict=True):
        st.session_state.setdefault(key, value)
    if enhanced:
        for key in EXTRA_WIDGETS:
            st.session_state.setdefault(key, manual_model.UNKNOWN)
    st.subheader("Transaction details")
    st.caption("Enter a USD amount and choose known details. The form never asks for a card number or email address.")
    with st.expander("Use a prefilled example"):
        for column, name in zip(st.columns(3), EXAMPLES, strict=True):
            column.button(name, key=f"fl_example_{name}", on_click=_example, args=(name, enhanced), width="stretch")
    with st.form("fl_transaction", border=True):
        amount = st.number_input("Amount (USD)", min_value=0.0, step=.01, format="%.2f", placeholder="Enter amount", key=WIDGETS[0])
        columns = st.columns(3)
        values = [column.selectbox(label, list(opts), format_func=opts.__getitem__, key=key)
                  for column, label, opts, key in zip(columns, manual_model.LABELS[1:], manual_model.OPTIONS, WIDGETS[1:], strict=True)]
        if enhanced:
            with st.expander("Optional context"):
                st.caption("Add these details if known. The form accepts provider categories only.")
                extras = st.columns(2)
                for column, label, choices, key in zip(extras, manual_enhanced.EXTRA_LABELS,
                                                        manual_enhanced.EXTRA_OPTIONS, EXTRA_WIDGETS, strict=True):
                    value = column.radio(label, list(choices), horizontal=True, index=None,
                                         format_func=choices.__getitem__, key=key)
                    values.append(value or manual_model.UNKNOWN)
        submitted = st.form_submit_button("Assess Transaction", type="primary", width="stretch")
    return submitted, (amount, *values)


def _result(record):
    outcome = record.result
    with st.container(border=True):
        st.subheader("Assessment result")
        cards([("Model risk score", pct(outcome.score), "Manual assessment model"),
               ("Decision", "Flagged" if outcome.flagged else "Not flagged", f"Review threshold {pct(outcome.threshold, 3)}")])
        if outcome.flagged:
            st.warning("Recommended action: Hold for manual review and verify the purchase through your normal process.")
        else:
            st.success("Recommended action: Continue standard checks. Escalate if you have other reasons for concern.")
        st.caption(f"{record.reference} · {record.created:%d %b %Y, %H:%M:%S} UTC")
        with st.expander("Why this score?", expanded=True):
            values = list(outcome.impacts)[::-1]
            bars = hbar(outcome.labels[::-1], values, [RED if value > 0 else TEAL for value in values])
            chart(bars, "fl_manual_shap", x="Contribution to model score (log-odds)", height=215)
            st.caption("Exact TreeSHAP. Positive contributions raise the model score. These are model signals for review.")


def _history(state):
    st.subheader("Recent assessments")
    if not state.recent:
        st.caption("Submitted checks will appear here during this session.")
        return
    st.dataframe(pd.DataFrame([{"Reference": row.reference, "Assessed (UTC)": row.created.strftime("%Y-%m-%d %H:%M:%S"),
                                "Risk score": row.result.score, "Decision": "Flagged" if row.result.flagged else "Not flagged"}
                              for row in reversed(state.recent)]), hide_index=True, width="stretch",
                 column_config={"Risk score": st.column_config.NumberColumn(format="percent")})


def render(paths):
    state = session()
    alerts = alert_ui.session()
    directory, enhanced = manual_enhanced.selected(paths.v3)
    try:
        engine = _load(directory, manual_model.signature(directory), enhanced)
    except MODEL_ERRORS:
        st.warning("Manual assessment is unavailable. Run the local model setup in the project guide.")
        return
    if enhanced:
        st.caption("Four core inputs and two optional fields · scores support human review.")
    else:
        st.caption("Four-input screening model · scores support human review.")
    form, result = st.columns([1, 1], gap="large")
    with form:
        submitted, values = _form(enhanced)
        if submitted:
            try:
                assessed = engine.assess(*values)
            except ValueError as error:
                st.error(str(error))
            except (OSError, RuntimeError):
                st.error("The model could not complete this assessment.")
            else:
                record = state.save(assessed)
                config, _ = alert_mail.configuration() if alerts.mode == "Real delivery" else (None, None)
                if not alert_ui.delivery_hidden():
                    alerts.collect_assessment(state.total, record.reference, assessed, engine.name, config=config)
    with result:
        if state.recent:
            _result(state.recent[-1])
        else:
            with st.container(border=True):
                st.subheader("Assessment result")
                st.info("Enter details or choose an example, then select Assess Transaction.")
                st.caption("Your result will show the score, review decision and what influenced it.")
    _history(state)
    alert_ui.delivery_panel(alerts, context={"threshold": engine.threshold, "model": engine.name, "source": "Manual assessment"})
