"""Live Monitor for local historical replay."""

from functools import partial
from time import monotonic

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import alert_ui, cases, replay
from .store import THRESHOLD, DataUnavailable
from .ui import RED, TEAL, cards, chart, hero, pct, policy_strip

STATE = replay.REPLAY_KEY
ERROR = "v3_replay_error"
ACTION = "v3_replay_action"
HANDOFF = "v3_replay_handoff"


def _control(action):
    state = st.session_state[STATE]
    st.session_state.pop(ERROR, None)
    if action == "reset":
        state.reset()
        alert_ui.bind_replay(state.replay_id)
        st.session_state.pop(ACTION, None)
        st.session_state.pop("v3_replay_pick", None)
    elif action == "pause":
        state.pause()
    elif action == "step":
        st.session_state[ACTION] = "step"
    else:
        state.start(monotonic())


def _open_event():
    state = st.session_state[STATE]
    state.pause()
    selected = st.session_state.get("v3_replay_pick")
    event = next((event for event in state.review if event.event_id == selected), None)
    if event is None:
        st.session_state[HANDOFF] = True
        st.session_state[ERROR] = "That event has left the recent queue. Choose a retained flag."
        return
    cases.navigate(event.transaction_id, replay_score=event.score)
    st.session_state[HANDOFF] = True


def _controls(state, private):
    with st.container(border=True):
        st.subheader("Replay controls")
        speed = st.select_slider("Replay speed · transactions per second", options=replay.SPEEDS, value=5,
                                 key="v3_replay_speed", disabled=private)
        columns = st.columns(5)
        for col, label, action, unavailable in zip(
            columns,
            ("Start", "Pause", "Resume", "Step one", "Reset"),
            ("start", "pause", "resume", "step", "reset"),
            (state.status != "Ready", state.status != "Running", state.status != "Paused",
             state.status not in ("Ready", "Paused"), state.status == "Ready"), strict=True,
        ):
            col.button(label, width="stretch", key=f"v3_replay_{action}", disabled=private or unavailable,
                       on_click=_control, args=(action,), type="primary" if action in ("start", "resume") else "secondary")
    return speed


def _stats(state, speed):
    cards([
        ("Replay status", state.status, f"Speed setting · {speed} transactions/s"),
        ("Processed transactions", f"{state.processed:,}", f"of {state.total:,} historical transactions"),
        ("Flagged transactions", f"{state.flagged:,}", f"{pct(state.flagged / state.processed) if state.processed else '—'} of processed"),
    ])
    st.progress(state.processed / state.total, text=f"Chronological progress · {state.processed:,} / {state.total:,}")


def _table(frame):
    st.dataframe(pd.DataFrame(frame), hide_index=True, width="stretch", height=min(360, 39 + 35 * len(frame)), column_config={
        "Replay row": st.column_config.NumberColumn(format="%d"), "Transaction ID": st.column_config.NumberColumn(format="%d"),
        "Elapsed second": st.column_config.NumberColumn(format="%d"), "Amount": st.column_config.NumberColumn(format="%.2f"),
        "Risk score": st.column_config.NumberColumn(format="percent"),
    })


def _feed(state):
    st.subheader("Transaction feed")
    st.caption(f"Most recent {replay.FEED_LIMIT} assessed records · newest first")
    if not state.recent:
        st.info("Start the replay or step through one transaction.")
        return
    _table([{"Replay row": row.sequence, "Transaction ID": row.transaction.tx_id, "Elapsed second": row.transaction.elapsed_second,
             "Amount": row.transaction.amount, "Risk score": row.score, "Decision": "Flagged" if row.flagged else "Below threshold"}
            for row in reversed(state.recent)])


def _queue(state):
    st.subheader("Review queue")
    st.caption(f"Most recent {replay.QUEUE_LIMIT} flagged records · newest first")
    if not state.review:
        st.info("Flagged transactions will appear here when their scores reach the frozen cutoff.")
        return
    events = {event.event_id: event for event in reversed(state.review)}
    _table([{"Replay row": event.sequence, "Transaction ID": event.transaction_id, "Risk score": event.score}
            for event in events.values()])
    if st.session_state.get("v3_replay_pick") not in events:
        st.session_state.pop("v3_replay_pick", None)
    st.selectbox("Flagged transaction", list(events), key="v3_replay_pick",
                 format_func=lambda key: f"Transaction {events[key].transaction_id} · row {events[key].sequence}")
    st.button("Investigate flagged transaction", type="primary", key="v3_replay_investigate",
              on_click=_open_event)


def _risk_chart(state):
    rows = list(state.recent)
    fig = go.Figure()
    fig.add_scatter(x=[row.sequence for row in rows], y=[row.score for row in rows], mode="lines+markers",
                    line={"color": TEAL, "width": 2}, marker={"color": [RED if row.flagged else TEAL for row in rows], "size": 6},
                    hovertemplate="Replay row %{x}<br>Risk %{y:.2%}<extra></extra>", showlegend=False)
    fig.add_hline(y=THRESHOLD, line_dash="dot", line_color=RED, annotation_text="Frozen cutoff",
                  annotation_position="top left")
    fig.update_yaxes(range=[0, min(1, max(THRESHOLD, *(row.score for row in rows)) * 1.12)])
    with st.container(border=True):
        st.subheader("Recent risk scores")
        chart(fig, "v3_replay_risk", x="Replay sequence", y="Risk score", percent_y=True, height=245)


def render(paths, pack, private, issue):
    hero("Live Monitor", "Replay historical transactions in order and send flagged cases to the investigator.", "Historical replay")
    policy_strip()
    st.caption("Reused retrospective holdout · scores computed during replay from verified prior-history features. "
               "Elapsed seconds are dataset timestamps. Observed labels are revealed only in the investigator.")
    # A fragment button requests a full rerun. A normal sidebar change may arrive first.
    st.session_state.pop(HANDOFF, None)
    if pack is None:
        st.warning(f"Replay unavailable. {issue}")
        return
    if STATE not in st.session_state:
        st.session_state[STATE] = replay.Replay(pack["population"]["rows"])
    state = st.session_state[STATE]
    if not private:
        alert_ui.bind_replay(state.replay_id)
    if private:
        state.pause()
    speed = _controls(state, private)
    if not private:
        st.caption(f"Alert Center mode · {alert_ui.session().mode}. Flagged cases create notification records.")

    @st.fragment(run_every=1.05 if state.status == "Running" and not private else None)
    def monitor():
        if st.session_state.pop(HANDOFF, False):
            st.rerun()
        previous = state.status
        if not private:
            try:
                score = partial(replay.score_batch, paths)
                if st.session_state.pop(ACTION, None) == "step":
                    alert_ui.accept_events(state.step(score))
                elif state.status == "Running":
                    alert_ui.accept_events(state.tick(monotonic(), speed, score))
            except (DataUnavailable, OSError, RuntimeError, ValueError) as error:
                state.pause()
                st.session_state[ERROR] = str(error)
        if previous != state.status:
            st.rerun()
        _stats(state, speed)
        if issue := st.session_state.get(ERROR):
            st.warning(f"Replay stopped. {issue}")
        if private:
            st.info("Local transaction feed and review details are hidden. Use the sidebar privacy switch to enable local replay.")
            return
        feed, queue = st.columns([1.35, 1], gap="medium")
        with feed, st.container(border=True):
            _feed(state)
        with queue, st.container(border=True):
            _queue(state)
        if state.recent:
            _risk_chart(state)

    monitor()
