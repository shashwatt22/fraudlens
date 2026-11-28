"""Alert Center and the handoff from a committed replay batch."""

from html import escape
import os

import pandas as pd
import streamlit as st

from . import alert_mail, alerts
from .ui import cards, hero, policy_strip

STATE_KEY = "v3_alerts"
CONFIRM_KEY = "v3_alert_confirm"
PUBLIC_KEY = "v3_public_demo"


def public_demo():
    if os.environ.get("FRAUDLENS_PUBLIC_DEMO") == "1" or st.query_params.get("demo") == "1":
        st.session_state[PUBLIC_KEY] = True
    return st.session_state.get(PUBLIC_KEY, False)


def delivery_hidden():
    return public_demo() or st.session_state.get("v3_private", False)


def session():
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = alerts.AlertSession()
    return st.session_state[STATE_KEY]


def privacy(force_dry_run=False):
    if (state := st.session_state.get(STATE_KEY)) is not None and (force_dry_run or state.mode == alerts.REAL):
        state.dry_run()
        st.session_state.pop(CONFIRM_KEY, None)


def bind_replay(replay_id):
    state = session()
    if state.replay_id != replay_id:
        state.bind(replay_id)
        st.session_state.pop(CONFIRM_KEY, None)


def accept_events(events):
    if not events or delivery_hidden():
        return
    state = session()
    config, _ = alert_mail.configuration() if state.mode == alerts.REAL else (None, None)
    state.collect(events, config=config)


def _test(context=None):
    if delivery_hidden():
        return
    state = session()
    config, _ = alert_mail.configuration() if state.mode == alerts.REAL else (None, None)
    state.test(config=config, **(context or {}))


def _mode(mode):
    if delivery_hidden():
        return
    state = session()
    if mode == alerts.DISABLED:
        state.disable()
    else:
        state.dry_run()
    st.session_state.pop(CONFIRM_KEY, None)


def _activate():
    if delivery_hidden():
        return
    config, _ = alert_mail.configuration()
    if st.session_state.get(CONFIRM_KEY):
        session().activate(config)


def _stats(state):
    cards([
        ("Notification mode", state.mode, "Current session"),
        ("Notifications logged", f"{state.created:,}", f"{state.counts[alerts.DRY_RUN]:,} dry-run · {state.counts[alerts.DISABLED]:,} disabled"),
        ("SMTP accepted", f"{state.counts['SMTP accepted']:,}", f"{state.counts['Failed']:,} failed · {state.counts['Sending']:,} sending"),
        ("Real attempt quota", f"{state.attempts} / {alerts.SESSION_LIMIT}", f"{alerts.MIN_INTERVAL}s minimum between attempts"),
    ])


def _history(state):
    st.subheader("Notification history")
    st.caption(f"UTC · newest first · latest {alerts.HISTORY_LIMIT} notifications retained.")
    if not state.history:
        st.info("A flagged case or an email test will appear here.")
        return
    st.dataframe(pd.DataFrame([
        {"Created (UTC)": row.created.strftime("%Y-%m-%d %H:%M:%S"),
         "Updated (UTC)": row.updated.strftime("%Y-%m-%d %H:%M:%S") if row.updated else "—",
         "Case reference": row.notice.reference, "Kind": "Test" if row.notice.is_test else row.notice.source,
         "Risk score": row.notice.score, "Mode": row.mode, "Status": row.status, "Detail": row.detail}
        for row in reversed(state.history)
    ]), hide_index=True, width="stretch", column_config={
        "Risk score": st.column_config.NumberColumn(format="percent"),
    })
    latest = state.history[-1]
    st.caption(f"Latest notification · {latest.status} · {latest.detail}")


def _demo_preview():
    st.info("Dry run only. Delivery controls and local notification history are hidden in this view.")
    with st.container(border=True):
        st.subheader("Email preview")
        st.caption("Illustrative example · synthetic case reference")
        title = escape(alert_mail.MODEL)
        status = escape(alert_mail.REVIEW)
        st.html(f'<div class="fl-policy"><strong>Historical replay review alert</strong>'
                f'<span>Case FL-DEMO</span><span>{title}</span><span>{status}</span></div>')
        st.write("Replay events from the local workspace populate notification history automatically.")


def render(paths, pack, private, issue):
    hero("Alert Center", "Review notification outcomes from historical replay and control local email delivery.", "Historical replay")
    policy_strip()
    state = session()
    if private:
        privacy(force_dry_run=True)
        _demo_preview()
        return
    delivery_panel(state)


def delivery_panel(state, context=None):
    state.refresh()
    with st.container(border=True):
        st.subheader("Email alerts" if context else "Delivery controls")
        description = ("Flags create a notification. Dry run records it locally."
                       if context else "Dry run is the default. A replay reset removes real-delivery authorization and preserves the session quota.")
        st.caption(description)
        columns = st.columns(3)
        columns[0].button("Dry run", key="v3_alert_dry", on_click=_mode, args=(alerts.DRY_RUN,), width="stretch",
                          disabled=state.mode == alerts.DRY_RUN)
        columns[1].button("Disable alerts", key="v3_alert_disable", on_click=_mode, args=(alerts.DISABLED,), width="stretch",
                          disabled=state.mode == alerts.DISABLED)
        columns[2].button("Send Test Email", key="v3_alert_test", on_click=_test, args=(context,), width="stretch",
                          disabled=state.mode == alerts.DISABLED)
        config, setup = alert_mail.configuration()
        if state.mode == alerts.REAL and config is None:
            state.dry_run()
        if config is None:
            st.caption(f"Authorized recipient: unavailable. {setup}" if context else f"Real delivery unavailable. {setup}")
        elif state.mode == alerts.REAL:
            st.success("Real delivery is active for the configured, authorized recipient.")
        else:
            if context:
                st.caption("Authorized recipient configured on this server.")
            agreed = st.checkbox("I authorize delivery to the recipient configured on this server.", key=CONFIRM_KEY)
            st.button("Activate real delivery", key="v3_alert_activate", type="primary", on_click=_activate,
                      disabled=not agreed or state.attempts >= alerts.SESSION_LIMIT)
        st.caption("Test emails use the current mode. Tests and automatic alerts share the real-send quota and rate limit.")

    @st.fragment(run_every=1.0 if state.pending else None)
    def monitor():
        was_pending = bool(state.pending)
        state.refresh()
        if was_pending and not state.pending:
            st.rerun()
        if context:
            st.caption(f"{state.mode} · {state.attempts}/{alerts.SESSION_LIMIT} real attempts · at least {alerts.MIN_INTERVAL}s apart")
        else:
            _stats(state)
        _history(state)

    monitor()
