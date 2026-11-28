"""Manual scoring, model explanation, history, and email integration regressions."""

from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from scipy.special import expit
from streamlit.testing.v1 import AppTest

from src.dashboard_v3 import alert_mail, alert_ui, manual_model as model, manual_ui, store
from src.dashboard_v3.alerts import AlertSession

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fitted():
    path = store.Paths.from_env(ROOT).v3 / "manual-assessment"
    if not (path / "model.ubj").is_file():
        pytest.skip("Local manual assessment checkpoint is required.")
    return model.AssessmentModel(path)


@pytest.mark.parametrize("bad", [0, -1, None, float("inf"), float("nan"), 49.951, 40000, True])
def test_rejects_invalid_amount(bad):
    with pytest.raises(ValueError):
        model.validate(bad, "visa", "debit", "google")


def test_supported_categories_and_training_only_vocabulary():
    raw = pd.DataFrame({model.AMOUNT: [42.998, 10., 22.], "card4": ["visa", None, "mastercard"],
                        "card6": ["debit", "debit or credit", "credit"],
                        "P_emaildomain": ["gmail.com", "anonymous.com", "example.invalid"]})
    normalized = model.normalize(raw)
    assert normalized["card6"].tolist() == ["debit", "unknown", "credit"]
    assert normalized["P_emaildomain"].tolist() == ["google", "unknown", "other"]
    assert normalized[model.AMOUNT].iloc[0] == 43
    categories = model.fit_categories(normalized.iloc[:2])
    assert "mastercard" not in categories["card4"]
    encoded, names, _ = model.encode(normalized.iloc[2:], categories)
    assert encoded[0, 0] == 22 and not encoded[0, 1:].any() and len(names) == encoded.shape[1]
    with pytest.raises(ValueError):
        model.validate(90, "visa", "debit", "someone@example.invalid")


def test_actual_inference_shap_and_development_metrics(fitted):
    small = fitted.assess(*manual_ui.EXAMPLES["Debit · $49.95"])
    large = fitted.assess(*manual_ui.EXAMPLES["Credit · $450"])
    assert small.score == pytest.approx(.0348826125, abs=2e-7) and not small.flagged
    assert large.score == pytest.approx(.0711045936, abs=2e-7) and large.flagged
    assert sum(large.impacts) + large.base == pytest.approx(large.margin, abs=1e-4)
    assert expit(large.margin) == pytest.approx(large.score, abs=2e-6)
    data = fitted.manifest
    assert data["threshold"] == pytest.approx(.0573460199)
    assert data["metrics"]["development_verification"]["selected"]["confusion_matrix"] == [[40527, 5063], [1013, 641]]


def test_alerts_share_delivery_limits_and_exclude_form_details(monkeypatch, fitted):
    def accepted(*args):
        task = Future()
        task.set_result(None)
        return task

    transport = Mock(side_effect=accepted)
    monkeypatch.setattr(alert_mail, "submit", transport)
    alerts, history = AlertSession(), manual_ui.AssessmentSession()
    result = fitted.assess(*manual_ui.EXAMPLES["Credit · $450"])
    record = history.save(result)
    config = alert_mail.SMTPConfig("smtp.example.invalid", 465, "ssl", "from@example.invalid",
                                   "to@example.invalid", "test", "test")
    alerts.collect_assessment(history.total, record.reference, result, model.MODEL)
    assert alerts.counts["Dry run"] == 1 and not transport.called
    alerts.activate(config)
    alerts.collect_assessment(history.total, record.reference, result, model.MODEL, config=config, now=1)
    assert not transport.called
    record = history.save(result)
    alerts.collect_assessment(history.total, record.reference, result, model.MODEL, config=config, now=2)
    assert alerts.counts["SMTP accepted"] == 1 and transport.call_count == 1
    notice = transport.call_args.args[1]
    email = alert_mail.message(config, notice)
    body = email.get_body(preferencelist=("plain",)).get_content()
    assert model.MODEL in body and record.reference in body and "450" not in body
    alerts.test(config=config, now=3)
    assert alerts.counts["Rate limited"] == 1


def test_two_pages_and_public_session_has_no_form_or_email(monkeypatch):
    monkeypatch.setenv("FRAUDLENS_PUBLIC_DEMO", "1")
    guard = Mock(side_effect=AssertionError("Public mode must not read mail settings."))
    monkeypatch.setattr(alert_mail, "configuration", guard)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    assert app.radio(key="fl_product_nav").options == ["Dashboard", "Check Transaction & Alerts"]
    app.radio(key="fl_product_nav").set_value("Check Transaction & Alerts").run()
    assert not app.exception and not app.button and not app.number_input and not app.checkbox
    guard.assert_not_called()


def test_form_assessment_and_recent_history(monkeypatch, fitted):
    monkeypatch.setenv("FRAUDLENS_PUBLIC_DEMO", "0")
    monkeypatch.setenv("FRAUDLENS_SMTP_ENABLED", "0")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15)
    app.session_state["fl_product_nav"] = "Check Transaction & Alerts"
    app.run()
    assert not app.exception
    example = "Credit · $899.99"
    app.button(key=f"fl_example_{example}").click().run()
    assert app.number_input(key="fl_amount").value == 899.99
    next(button for button in app.button if button.label == "Assess Transaction").click().run()
    assert app.session_state[manual_ui.SESSION].total == 1
    assert app.session_state[manual_ui.SESSION].flagged == 1
    assert app.session_state[alert_ui.STATE_KEY].counts["Dry run"] == 1
    app.run()
    assert app.session_state[manual_ui.SESSION].total == 1
    assert app.session_state[alert_ui.STATE_KEY].created == 1
    assert not app.exception
