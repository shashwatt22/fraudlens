"""Optional feature parity, promotion policy, and active-model email integration."""

from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from scipy.special import expit
from streamlit.testing.v1 import AppTest

from scripts.compare_manual_challenger import comparison
from src.dashboard_v3 import alert_mail, alert_ui, manual_enhanced as enhanced, manual_model as manual, manual_ui, store

ROOT = Path(__file__).resolve().parents[1]


def test_optional_context_represents_unknown_values_without_inventing_data():
    raw = pd.DataFrame({manual.AMOUNT: [25., 25.], "card4": ["visa", "visa"],
                        "card6": ["debit", "debit"], "P_emaildomain": ["gmail.com", "gmail.com"],
                        "DeviceType": [None, "mobile"], "R_emaildomain": ["anonymous.com", "hotmail.com"]})
    normalized = enhanced.normalize(raw)
    assert normalized.iloc[0].to_dict() == enhanced.validate(25., "visa", "debit", "google")
    assert normalized.iloc[1].to_dict() == enhanced.validate(25., "visa", "debit", "google", "mobile", "microsoft")
    with pytest.raises(ValueError, match="optional context"):
        enhanced.validate(25., "visa", "debit", "google", "mobile", "customer@example.invalid")


def test_promotion_checks_ranking_f1_and_workload():
    base = dict(average_precision=.09, roc_auc=.70, f1=.17, review_rate=.08)
    delivered = dict(average_precision=.09, roc_auc=.70, f1=.19, review_rate=.10)
    better = dict(average_precision=.15, roc_auc=.75, f1=.23, review_rate=.04)
    assert comparison(base, delivered, better)["promote"]
    assert not comparison(base, delivered, {**better, "f1": .18})["promote"]
    assert not comparison(base, delivered, {**better, "review_rate": .15})["promote"]


@pytest.fixture
def promoted():
    location = store.Paths.from_env(ROOT).v3 / "manual-challenger"
    if not (location / "model.ubj").is_file():
        pytest.skip("Local optional-context checkpoint is required.")
    return manual.AssessmentModel(location, enhanced)


def test_active_checkpoint_scores_with_or_without_optional_fields(promoted):
    inputs = manual_ui.EXAMPLES["Credit · $899.99"]
    omitted = promoted.assess(*inputs)
    explicit = promoted.assess(*inputs, "unknown", "unknown")
    known = promoted.assess(*inputs, "desktop", "microsoft")
    assert omitted == explicit and omitted.flagged
    assert not known.flagged
    assert known.labels == enhanced.LABELS
    assert known.base + sum(known.impacts) == pytest.approx(known.margin, abs=1e-4)
    assert expit(known.margin) == pytest.approx(known.score, abs=2e-6)
    assert promoted.manifest["decision"]["promote"]


def test_promoted_form_alert_uses_current_model_cutoff_and_reference(monkeypatch, promoted):
    monkeypatch.setenv("FRAUDLENS_PUBLIC_DEMO", "0")
    config = alert_mail.SMTPConfig("smtp.example.invalid", 465, "ssl", "from@example.invalid",
                                   "to@example.invalid", "test", "test")
    monkeypatch.setattr(alert_mail, "configuration", lambda: (config, None))
    accepted = Future()
    accepted.set_result(None)
    transport = Mock(return_value=accepted)
    monkeypatch.setattr(alert_mail, "submit", transport)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15)
    app.session_state["fl_product_nav"] = "Check Transaction & Alerts"
    app.run()
    app.checkbox(key=alert_ui.CONFIRM_KEY).check().run()
    app.button(key="v3_alert_activate").click().run()
    app.button(key="fl_example_Credit · $899.99").click().run()
    next(button for button in app.button if button.label == "Assess Transaction").click().run()
    app.run()
    assert not app.exception and transport.call_count == 1
    notice = transport.call_args.args[1]
    assert notice.model == enhanced.MODEL
    assert notice.threshold == promoted.threshold
    assert notice.reference == app.session_state[manual_ui.SESSION].recent[-1].reference
    assert notice.score >= notice.threshold
    assert app.session_state[alert_ui.STATE_KEY].counts["SMTP accepted"] == 1
    assert any("Latest notification · SMTP accepted" in item.value for item in app.caption)
