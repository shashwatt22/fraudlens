"""Session replay, event idempotency, and frozen inference regressions."""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.data_loader import ID, TIME
from src.dashboard_v3 import replay, store
from src.dashboard_v3.replay import Replay, Transaction

ROOT = Path(__file__).resolve().parents[1]


def sample(start, stop):
    scores = (.01, store.THRESHOLD, .95, np.nextafter(store.THRESHOLD, 0), .7)
    return [(Transaction(100 + i, 500 + i // 2, 1.), scores[i % len(scores)]) for i in range(start, stop)]


def test_start_pause_resume_and_reset():
    state = Replay(10)
    assert state.tick(0, 5, sample) == ()
    state.start(1.0)
    assert state.tick(.8, 5, sample) == ()
    events = state.tick(1.0, 5, sample)
    assert state.status == "Running" and state.processed == 5 and state.flagged == 3
    assert [event.sequence for event in events] == [2, 3, 5]
    state.pause()
    assert state.status == "Paused" and state.tick(50.0, 5, sample) == ()
    state.start(55.0)
    tail = state.tick(55.0, 20, sample)
    assert [event.sequence for event in tail] == [7, 8, 10]
    assert state.status == "Complete" and state.processed == 10
    assert state.tick(56.0, 20, sample) == ()
    state.reset()
    assert (state.status, state.processed, state.flagged, list(state.recent), list(state.review)) == ("Ready", 0, 0, [], [])
    assert state.replay_id != events[0].replay_id


def test_ticks_and_early_reruns_do_not_duplicate_review_events():
    state = Replay(200)
    state.start(10.0)
    fresh = state.tick(10.0, 50, sample)
    assert len(fresh) == 30 and len({event.event_id for event in fresh}) == 30
    assert fresh[0].score == fresh[0].threshold == store.THRESHOLD
    assert not hasattr(fresh[0], "label")
    assert state.tick(10.0, 100, sample) == ()
    assert state.tick(10.9, 100, sample) == ()
    more = state.tick(11.0, 100, sample)
    assert len(more) == 60 and not {event.event_id for event in more} & {event.event_id for event in fresh}
    assert state.processed == 150 and state.flagged == 90
    assert len(state.recent) == replay.FEED_LIMIT and len(state.review) == replay.QUEUE_LIMIT
    assert state.recent[0].sequence == 121 and state.recent[-1].sequence == 150
    assert all(event.replay_id == state.replay_id for event in state.review)


def test_step_includes_equal_timestamps_and_uses_inclusive_frozen_threshold():
    state = Replay(4)
    assert state.step(sample) == ()
    boundary, = state.step(sample)
    assert boundary.transaction_id == 101
    assert state.recent[0].transaction.elapsed_second == state.recent[1].transaction.elapsed_second
    assert state.status == "Paused" and state.processed == 2
    state.step(sample)
    assert state.step(sample) == ()  # One representable float below the cutoff.
    assert state.status == "Complete" and state.flagged == 2


def test_invalid_result_does_not_advance_cursor_or_publish_events():
    state = Replay(10)
    state.start(1.)

    def invalid(start, stop):
        rows = sample(start, stop)
        rows[-1] = rows[-1][0], float("nan")
        return rows

    with pytest.raises(store.DataUnavailable, match="could not be verified"):
        state.tick(1., 5, invalid)
    assert state.processed == state.flagged == 0 and not state.review
    assert len(state.tick(1., 5, sample)) == 3


def test_real_replay_inference_matches_archived_holdout_rows():
    paths = store.Paths.from_env(ROOT)
    if not (paths.retrospective / "predictions.parquet").is_file():
        pytest.skip("Local V3 release artifacts are required.")
    archived = store.queue(paths, store.bundle(paths))
    for start in (0, 58_990, store.ROWS - 100):
        expected = archived.iloc[start:start + 100]
        computed = replay.score_batch(paths, start, start + 100)
        assert [row.tx_id for row, _ in computed] == expected[ID].tolist()
        assert [row.elapsed_second for row, _ in computed] == expected[TIME].tolist()
        scores = np.array([score for _, score in computed], dtype=np.float64)
        np.testing.assert_allclose(scores, expected[store.SCORE], atol=2e-7, rtol=0)
        np.testing.assert_array_equal(scores >= store.THRESHOLD, expected[store.DECISION])


def test_monitor_step_and_investigation_navigation(monkeypatch):
    monkeypatch.setattr(replay, "score_batch", lambda paths, start, stop: sample(start, stop))
    from src.dashboard_v3 import cases
    cases_seen = []

    def capture(paths, pack, private, issue):
        import streamlit as st
        cases_seen.append(st.session_state["v3_replay_case"])
        st.title("Investigation handoff")

    monkeypatch.setattr(cases, "investigator", capture)
    monkeypatch.setattr(store, "bundle", lambda paths: {"population": {"rows": 10}})
    app = AppTest.from_file(str(ROOT / "archive" / "apps" / "app_research.py"), default_timeout=15)
    app.session_state["v3_page"] = "Live Monitor"
    app.run()
    assert not app.exception
    app.button(key="v3_replay_step").click().run()
    app.button(key="v3_replay_step").click().run()
    assert app.session_state[replay.REPLAY_KEY].processed == 2
    app.run()
    assert app.session_state[replay.REPLAY_KEY].processed == 2
    app.button(key="v3_replay_investigate").click().run()
    assert not app.exception
    assert app.session_state["v3_page"] == "Transaction Investigator"
    assert cases_seen == [(101, store.THRESHOLD)]
