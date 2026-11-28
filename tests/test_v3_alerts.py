"""Email privacy, delivery policy, and replay-to-Alert Center regressions."""

from concurrent.futures import Future
from pathlib import Path
import smtplib
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard_v3 import alert_mail, alert_ui, alerts, replay, store

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings():
    # Reserved example domain only. All delivery tests replace the network transport.
    return {"FRAUDLENS_SMTP_ENABLED": "1", "FRAUDLENS_SMTP_HOST": "smtp.example.invalid",
            "FRAUDLENS_SMTP_PORT": "465", "FRAUDLENS_SMTP_TLS": "ssl",
            "FRAUDLENS_SMTP_FROM": "sender@example.invalid", "FRAUDLENS_SMTP_TO": "recipient@example.invalid",
            "FRAUDLENS_SMTP_USERNAME": "test-user", "FRAUDLENS_SMTP_PASSWORD": "test-value"}


def event(sequence=1, replay_id="run-a", score=.79):
    return replay.AlertEvent(f"{replay_id}:{sequence}", replay_id, sequence, 987654321, 123456789, score)


def resolved(value):
    task = Future()
    task.set_result(value)
    return task


def test_requires_server_opt_in_secure_transport_and_exactly_one_mailbox(settings):
    assert alert_mail.configuration({})[0] is None
    config, error = alert_mail.configuration(settings)
    assert config.tls == "ssl" and error is None
    for field, invalid in (("TO", "one@example.invalid,two@example.invalid"),
                           ("TO", "name@example.invalid\r\nBcc:a@example.invalid"),
                           ("TLS", "cleartext"), ("PORT", "0"), ("PASSWORD", "")):
        config, error = alert_mail.configuration({**settings, f"FRAUDLENS_SMTP_{field}": invalid})
        assert config is None and error == alert_mail.INVALID_CONFIG


def test_replay_to_private_multipart_message(settings):
    config, _ = alert_mail.configuration(settings)
    state = alerts.AlertSession()
    state.bind("run-a")
    state.collect([event(score=store.THRESHOLD)])
    record = state.history[-1]
    email = alert_mail.message(config, record.notice)
    assert email.get_content_type() == "multipart/alternative"
    assert [part.get_content_type() for part in email.iter_parts()] == ["text/plain", "text/html"]
    for token in ("FraudLens V3", "Risk score", "0.1938110143", "Pending analyst review", record.notice.reference):
        assert token in email.get_body(preferencelist=("plain",)).get_content()
        assert token in email.get_body(preferencelist=("html",)).get_content()
    for secret in ("987654321", "123456789", "run-a", "test-value"):
        assert secret not in email.as_string()
    assert "recipient@example.invalid" not in repr(config)


def test_dry_run_disabled_dedup_and_bounded_history(monkeypatch):
    transport = Mock(side_effect=AssertionError("No SMTP submission allowed"))
    monkeypatch.setattr(alert_mail, "submit", transport)
    state = alerts.AlertSession()
    state.bind("run-a")
    state.collect([event(), event(), event(2, score=.01)])
    state.test()
    assert state.created == 2 and state.counts["Dry run"] == 2
    state.disable()
    state.collect([event(3), event(3)])
    assert state.counts["Disabled"] == 1
    state.dry_run()
    state.collect([event(i) for i in range(4, 200)])
    assert len(state.history) == alerts.HISTORY_LIMIT
    assert state.created == 199 and transport.call_count == 0


def test_rate_limit_quota_dedup_and_reset_reactivation(monkeypatch, settings):
    config, _ = alert_mail.configuration(settings)
    transport = Mock(side_effect=lambda *args: resolved(None))
    monkeypatch.setattr(alert_mail, "submit", transport)
    state = alerts.AlertSession()
    assert not state.activate(None) and state.activate(config)
    state.bind("run-a")
    state.test(config=config, now=1)
    state.collect([event(), event()], config=config, now=2)
    assert state.counts["Rate limited"] == 1 and state.attempts == 1
    state.collect([event(2)], config=config, now=31)
    assert state.counts["SMTP accepted"] == 2 and transport.call_count == 2
    state.bind("run-b")
    assert state.mode == alerts.DRY_RUN and state.attempts == 2
    state.collect([event(50), event(replay_id="run-b")], config=config, now=70)
    assert state.counts["Dry run"] == 1
    state.activate(config)
    for now in (70, 100, 130, 160):
        state.test(config=config, now=now)
    assert state.attempts == alerts.SESSION_LIMIT and state.counts["Session limit"] == 1
    assert transport.call_count == 5


def test_failure_status_is_isolated_and_unconfirmed_sends_are_not_retried(monkeypatch, settings):
    config, _ = alert_mail.configuration(settings)
    failed = Future()
    transport = Mock(return_value=failed)
    monkeypatch.setattr(alert_mail, "submit", transport)
    state = alerts.AlertSession()
    state.activate(config)
    state.bind("run-a")
    state.collect([event()], config=config, now=1)
    assert state.counts["Sending"] == 1
    failed.set_exception(RuntimeError("secret diagnostic"))
    state.refresh()
    state.collect([event()], config=config, now=35)
    record = state.history[-1]
    assert record.status == "Failed" and record.updated is not None
    assert "secret" not in record.detail and transport.call_count == 1


@pytest.mark.parametrize("tls", ["ssl", "starttls"])
def test_secure_smtp_success_and_redacted_failure(monkeypatch, settings, tls):
    settings["FRAUDLENS_SMTP_TLS"] = tls
    config, _ = alert_mail.configuration(settings)
    relay = Mock()
    relay.send_message.return_value = {}
    connection = Mock(return_value=relay)
    monkeypatch.setattr(smtplib, "SMTP_SSL" if tls == "ssl" else "SMTP", connection)
    notice = alert_mail.MailNotice("TEST-EXAMPLE", None, store.THRESHOLD, True)
    assert alert_mail.deliver(config, notice) is None
    relay.login.assert_called_once_with(config.username, config.password)
    assert relay.send_message.call_args.kwargs["to_addrs"] == [config.recipient]
    relay.close.assert_called_once()
    if tls == "ssl":
        assert connection.call_args.kwargs["context"].check_hostname
        relay.starttls.assert_not_called()
    else:
        assert relay.ehlo.call_count == 2 and relay.starttls.call_args.kwargs["context"].check_hostname
        assert [call[0] for call in relay.method_calls][:4] == ["ehlo", "starttls", "ehlo", "login"]
    relay.send_message.side_effect = smtplib.SMTPRecipientsRefused({config.recipient: (550, b"private details")})
    assert alert_mail.deliver(config, notice) == "Sender or recipient declined by the SMTP server."
    relay.login.side_effect = smtplib.SMTPAuthenticationError(535, b"private details")
    assert alert_mail.deliver(config, notice) == "SMTP authentication failed."


def test_public_demo_hides_controls_even_when_server_configured(monkeypatch, settings):
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("FRAUDLENS_PUBLIC_DEMO", "1")
    guard = Mock(side_effect=AssertionError("Public demo must not read SMTP configuration"))
    monkeypatch.setattr(alert_mail, "configuration", guard)
    app = AppTest.from_file(str(ROOT / "archive" / "apps" / "app_research.py"), default_timeout=15)
    app.session_state["v3_page"] = "Alert Center"
    app.run()
    assert not app.exception
    assert not app.button and not app.checkbox and not app.toggle
    assert app.session_state[alert_ui.STATE_KEY].mode == alerts.DRY_RUN
    guard.assert_not_called()


def test_integration_replay_notifications_and_explicit_real_mode(monkeypatch, settings):
    monkeypatch.delenv("FRAUDLENS_PUBLIC_DEMO", raising=False)
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(store, "bundle", lambda paths: {"population": {"rows": 10}})
    monkeypatch.setattr(replay, "score_batch", lambda paths, start, stop:
                        [(replay.Transaction(i, i, 1.), .75) for i in range(start, stop)])
    submit = Mock(side_effect=lambda *args: resolved(None))
    monkeypatch.setattr(alert_mail, "submit", submit)
    app = AppTest.from_file(str(ROOT / "archive" / "apps" / "app_research.py"), default_timeout=15)
    app.session_state["v3_page"] = "Live Monitor"
    app.run()
    app.button(key="v3_replay_step").click().run()
    app.run()
    session = app.session_state[alert_ui.STATE_KEY]
    assert session.counts["Dry run"] == 1 and not submit.called
    app.radio(key="v3_page").set_value("Alert Center").run()
    assert app.button(key="v3_alert_activate").disabled
    app.checkbox(key=alert_ui.CONFIRM_KEY).check().run()
    app.button(key="v3_alert_activate").click().run()
    assert session.mode == alerts.REAL and not submit.called
    app.button(key="v3_alert_test").click().run()
    assert submit.call_count == 1 and session.counts["SMTP accepted"] == 1
    app.radio(key="v3_page").set_value("Live Monitor").run()
    app.button(key="v3_replay_reset").click().run()
    assert session.mode == alerts.DRY_RUN and session.attempts == 1
    app.button(key="v3_replay_step").click().run()
    assert session.counts["Dry run"] == 2 and submit.call_count == 1
    assert not app.exception
