"""Fixed risk policy and ID-aligned ranking on synthetic rows."""

import numpy as np
import pandas as pd
import pytest

from src.risk_engine import (CONTEXT_COLUMNS, assess_transaction, build_risk_table, composite_risk,
                             review_priority, validation_rankings)


def test_composite_weights_and_priority_boundaries():
    np.testing.assert_allclose(composite_risk([0, 1, 0.5, 1], [0, 0, 0.75, 1]), [0, 80, 55, 100])
    assert review_priority([0, 39.999, 40, 74.999, 75, 100]).tolist() == [
        "Low", "Low", "Medium", "Medium", "High", "High"
    ]


@pytest.mark.parametrize("classifier,anomaly", [([np.nan], [0]), ([1], [np.inf]), ([-0.1], [0]),
                                                ([0], [1.01]), ([0, 1], [1])])
def test_invalid_composite_inputs(classifier, anomaly):
    with pytest.raises(ValueError):
        composite_risk(classifier, anomaly)


def test_risk_table_checks_alignment_and_exposes_unavailable_history():
    frame = pd.DataFrame({column: [0, 1] for column in CONTEXT_COLUMNS}).assign(
        TransactionID=[9, 7], TransactionDT=[1, 2], isFraud=[0, 1], prior_mean_amount=[pd.NA, 5],
        amount_to_prior_mean=[pd.NA, 2], seconds_since_prior_txn=[pd.NA, 7], is_new_product=[pd.NA, 1],
        identity_missing=[1, 0],
    )
    classifier = frame[["TransactionID", "TransactionDT", "isFraud"]].assign(
        classifier_score=[0.2, 0.9], classifier_decision=[0, 1]
    )
    anomaly = frame[["TransactionID"]].assign(anomaly_score=[0.3, 0.5], anomaly_percentile=[0.1, 0.8])
    with pytest.raises(ValueError, match="row order"):
        build_risk_table(frame, classifier.iloc[::-1], anomaly, 0.8)
    with pytest.raises(ValueError, match="row order"):
        build_risk_table(frame, classifier, anomaly.iloc[::-1], 0.8)
    with pytest.raises(ValueError, match="frozen classification threshold"):
        build_risk_table(frame, classifier, anomaly, 0.95)
    result = build_risk_table(frame, classifier, anomaly, 0.8)
    assert result["TransactionID"].tolist() == [9, 7]
    assert result["review_priority"].tolist() == ["Low", "High"]
    assessment = assess_transaction(result.iloc[0])
    assert "isFraud" not in assessment
    assert assessment["behavioral_context"]["history_status"] == "cold_start"
    assert assessment["behavioral_context"]["prior_mean_amount"] is None
    assert assessment["behavioral_context"]["is_new_product"] is None
    assert not assessment["behavioral_context"]["identity_available"]
    assert assessment["shap"]["status"] == "available_on_demand"


def test_review_budgets_round_up_and_break_score_ties_by_id():
    table = pd.DataFrame({"TransactionID": range(101, 0, -1), "isFraud": [0] * 99 + [1, 1],
                          "classifier_score": 0.5, "anomaly_score": 0.5, "risk_score": 50.0})
    report = validation_rankings(table)
    for result in report.values():
        assert result["roc_auc"] == 0.5 and result["average_precision"] == pytest.approx(2 / 101)
        assert result["top_review"]["1"] == {"reviewed": 2, "fraud_found": 2, "precision": 1, "recall": 1}
        assert result["top_review"]["5"]["reviewed"] == 6
        assert result["top_review"]["10"]["reviewed"] == 11
