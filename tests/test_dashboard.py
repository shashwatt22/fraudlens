"""Dashboard utilities and real-artifact Streamlit interaction smoke checks."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard.artifacts import ArtifactError, DashboardRun, generation_commands
from src.dashboard.logic import demo_cases, outcome, parse_id, review_rows, threshold_metrics
from src.dashboard.ui import histogram

ROOT = Path(__file__).resolve().parents[1]
V1_ARTIFACTS = Path(os.environ.get("FRAUDLENS_ARTIFACTS", ROOT / "artifacts"))
RUN_DIR = V1_ARTIFACTS / "quick"


def _app():
    app = AppTest.from_file(str(ROOT / "archive" / "apps" / "app_legacy.py"), default_timeout=40)
    app.session_state["dataset_mode"] = "quick"
    return app


def _clean(app):
    assert not app.exception, [str(error.value) for error in app.exception]
    assert not app.error, [str(error.value) for error in app.error]


def _metric(app, label):
    return next(item.value for item in app.metric if item.label == label)


@pytest.fixture
def quick_artifacts(monkeypatch):
    if not (RUN_DIR / "phase3" / "manifest.json").is_file():
        pytest.skip("Real quick artifacts are excluded from Git. Generate them to run dashboard integration checks.")
    monkeypatch.setenv("FRAUDLENS_ARTIFACTS", str(RUN_DIR.parent))
    return DashboardRun(RUN_DIR.parent, "quick")


def test_threshold_with_no_positive_predictions_and_invalid_inputs():
    result = threshold_metrics([1, 0, 1], [.8, .3, .7], 1.0)
    assert result["confusion_matrix"] == [[1, 0], [2, 0]]
    assert result["precision"] is None and result["specificity"] == 1 and result["flagged"] == 0
    assert threshold_metrics([1, 0], [.8, .4], .8)["flagged"] == 1
    with pytest.raises(ValueError):
        threshold_metrics([0], [np.nan], .5)


def test_queue_filters_and_curated_outcomes_are_deterministic():
    frame = pd.DataFrame({"TransactionID": [12, 9, 8, 15, 7], "TransactionDT": [2, 1, 1, 2, 1],
                          "classifier_score": [.8, .1, .8, .9, .7], "classifier_decision": [1, 0, 1, 1, 0],
                          "anomaly_score": [.8, .1, .8, .9, .9], "risk_score": [50, 10, 53, 90, 70],
                          "review_priority": ["Medium", "Low", "Medium", "High", "Medium"],
                          "prior_mean_amount": [10, None, 13, 14, 5], "prior_txn_count": [1, 0, 2, 3, 4],
                          "isFraud": [0, 0, 1, 1, 1]})
    assert review_rows(frame, "Classifier score", ["Medium"], "", 3)["TransactionID"].tolist() == [8, 12, 7]
    assert review_rows(frame, "Composite risk", [], "12", 2)["TransactionID"].tolist() == [12]
    assert review_rows(frame, "Composite risk", ["High"], "12", 2).empty
    assert demo_cases(frame, preferred=12) == {"Correctly detected fraud": 8, "Correctly cleared legitimate": 9,
                                              "False positive": 12, "Missed fraud": 7, "Behavioral context": 12,
                                              "High anomaly": 7}
    assert outcome(1, 0) == "False negative"
    assert parse_id(" 12 ") == 12 and parse_id("-5") is None and parse_id("9999999999") is None


def test_histogram_accepts_constant_and_missing_inputs():
    figure = histogram([.2, .2, np.nan], bounds=(.2, .2))
    assert sum(figure.data[0].y) == 2
    assert histogram([np.nan]) is None


def test_app_missing_artifacts_shows_exact_recovery_command(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUDLENS_ARTIFACTS", str(tmp_path))
    app = _app().run()
    assert not app.exception and "Phase 1" in app.error[0].value
    assert "-m src.phase1 --mode quick" in app.code[0].value
    assert "-m src.train_mlp --mode quick" in generation_commands("quick", "2B")


def test_artifact_loader_rejects_incompatible_manifest(tmp_path):
    path = tmp_path / "quick"
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"phase": 1, "schema_version": 999, "mode": "quick"}), encoding="utf-8")
    with pytest.raises(ArtifactError, match="schema version"):
        DashboardRun(tmp_path, "quick")


def test_complete_full_artifacts_are_the_default_and_quick_is_selectable(quick_artifacts):
    root = RUN_DIR.parent
    if not (root / "full" / "pipeline_summary.json").is_file():
        pytest.skip("Generate the full artifacts to run this dashboard integration check.")
    app = AppTest.from_file(str(ROOT / "archive" / "apps" / "app_legacy.py"), default_timeout=40).run()
    _clean(app)
    assert app.selectbox(key="dataset_mode").value == "full"
    assert _metric(app, "Processed transactions") == "590,540"
    app.selectbox(key="dataset_mode").set_value("quick").run()
    _clean(app)
    assert _metric(app, "Processed transactions") == "100,000"


def test_app_queue_case_handoff_reveal_shap_and_session_state(quick_artifacts):
    app = _app().run()
    _clean(app)
    assert _metric(app, "Processed transactions") == "100,000"
    assert app.selectbox(key="queue_rank").value == "Classifier score"
    assert "Observed label" not in app.dataframe[0].value.columns
    app.text_input(key="queue_search").set_value("999999999").run()
    _clean(app)
    assert any("No transactions match" in item.value for item in app.info)
    app.radio(key="evaluation_partition").set_value("test").run()
    example = quick_artifacts.report(3, "example")
    transaction_id = example["TransactionID"]
    app.text_input(key="queue_search").set_value(str(transaction_id)).run()
    _clean(app)
    assert app.selectbox(key="queue_case").value == transaction_id
    app.button(key="open_case").click().run()
    _clean(app)
    assert app.radio(key="navigation").value == "Transaction Investigator"
    assert _metric(app, "Transaction ID") == str(transaction_id)
    assert not any("Observed label:" in item.value for item in app.info)
    app.button(key="explain_case").click().run()
    _clean(app)
    assert _metric(app, "Reconstructed score") == f"{example['classifier_score']:.6f}"
    app.button(key="reveal_truth").click().run()
    assert any("Observed label:" in item.value for item in app.info)
    app.radio(key="navigation").set_value("Overview & Review Queue").run()
    app.radio(key="navigation").set_value("Transaction Investigator").run()
    assert app.radio(key="evaluation_partition").value == "test"
    assert _metric(app, "Transaction ID") == str(transaction_id)
    app.text_input(key="case_id_input_quick_test").set_value("invalid")
    next(button for button in app.button if button.label == "Load transaction").click().run()
    assert "not found" in app.warning[0].value
    assert _metric(app, "Transaction ID") == str(transaction_id)


def test_app_model_lab_threshold_and_behavioral_page(quick_artifacts):
    app = _app().run()
    app.radio(key="navigation").set_value("Model Lab").run()
    _clean(app)
    assert _metric(app, "Highest validation AP") == "PyTorch MLP"
    assert _metric(app, "Deployed classical classifier") == "XGBoost with history"
    assert _metric(app, "Best checkpoint") == "Epoch 23"
    assert [item.label for item in app.tabs] == ["Model comparison", "Evaluation lab", "Deep learning",
                                                "Behavioral ablation", "Risk ranking"]
    app.slider(key="threshold_quick_xgboost").set_value(1.0).run()
    _clean(app)
    assert _metric(app, "Flagged transactions") == "0"
    assert _metric(app, "Precision") == "Unavailable"
    app.slider(key="threshold_quick_xgboost").set_value(0.0).run()
    assert _metric(app, "Flagged transactions") == "20,000"
    app.radio(key="evaluation_partition").set_value("test").run()
    _clean(app)
    assert not app.slider and _metric(app, "Flagged transactions") == "936"
    app.radio(key="navigation").set_value("Behavioral Intelligence & Patterns").run()
    _clean(app)
    assert _metric(app, "Training reference rows") == "60,000"
    assert any(item.value == "Global grouped SHAP importance" for item in app.subheader)


def test_app_investigator_displays_real_cold_start_without_revealing_truth(quick_artifacts):
    risk = quick_artifacts.risk("validation")
    cold = risk.loc[risk["prior_txn_count"].eq(0) & risk["prior_mean_amount"].isna()]
    key = str(int(cold["TransactionID"].iloc[0]))
    app = _app().run()
    app.radio(key="navigation").set_value("Transaction Investigator").run()
    app.text_input(key="case_id_input_quick_validation").set_value(key)
    next(button for button in app.button if button.label == "Load transaction").click().run()
    _clean(app)
    assert _metric(app, "Transaction ID") == key
    assert _metric(app, "Prior mean amount") == "Unavailable"
    assert any("Cold start" in notice.value for notice in app.info)
    assert not any("Observed label:" in notice.value for notice in app.info)
