"""Hand-check rule arithmetic and frozen condition definitions."""

import pandas as pd
import pytest

from src.association import MiningSettings, encode_basket, evaluate_rules, fit_conditions, mine_rules, rule_metrics


def test_hand_calculated_support_confidence_and_lift():
    basket = pd.DataFrame({"amount": [True, True, True, False, False], "fraud": [True, True, False, True, False]})
    result = rule_metrics(basket, ("amount",))
    assert result["matches"] == 3 and result["fraud_matches"] == 2
    assert result["support"] == .4 and result["confidence"] == pytest.approx(2 / 3)
    assert result["lift"] == pytest.approx((2 / 3) / (3 / 5))


def test_zero_occurrence_and_zero_baseline_are_explicit():
    basket = pd.DataFrame({"condition": [False, False], "fraud": [False, False]})
    result = rule_metrics(basket, ("condition",))
    assert result["support"] == 0 and result["confidence"] is None and result["lift"] is None
    result = rule_metrics(basket.assign(condition=[True, False]), ("condition",))
    assert result["confidence"] == 0 and result["lift"] is None
    assert rule_metrics(basket.iloc[:0], ("condition",))["support"] is None
    assert mine_rules(basket) == ([], 0)


def test_frozen_cutoffs_and_missing_history_require_observation():
    train = pd.DataFrame({"TransactionAmt": [10, 20, 100], "prior_txn_count": [1, 2, 0],
                          "amount_to_prior_mean": [1., 3., None], "prior_count_24h": [0, 4, 0],
                          "is_new_product": [0, 1, 1], "identity_missing": [0, 1, 1],
                          "seconds_since_prior_txn": [100., 300., None], "isFraud": [0, 1, 0]})
    conditions = fit_conditions(train)
    by_key = {condition.key: condition for condition in conditions}
    assert by_key["high_amount"].cutoff == pytest.approx(84)
    assert by_key["large_vs_history"].cutoff == pytest.approx(2.8)
    assert by_key["recent_activity"].cutoff == pytest.approx(3)
    assert by_key["long_gap"].cutoff == pytest.approx(250)
    held = train.assign(TransactionAmt=[85, 80, 1000], amount_to_prior_mean=[None, 100, 100],
                        seconds_since_prior_txn=[None, 50, 1000])
    basket = encode_basket(held, conditions)
    assert basket["high_amount"].tolist() == [True, False, True]
    assert basket["large_vs_history"].tolist() == [False, True, False]
    assert basket["long_gap"].tolist() == [False, False, False]
    assert basket["new_product"].tolist() == [False, True, False]
    assert basket["no_prior_history"].tolist() == [False, False, True]


def test_mlxtend_mining_retains_fraud_consequent_and_reevaluates_identical_antecedents():
    from src.association import Condition

    train = pd.DataFrame({"high": [True, True, True, False, False, False], "fraud": [True, True, False, False, False, False]})
    settings = MiningSettings(min_support=.1, min_confidence=.5, min_matches=2, min_fraud_matches=2)
    rules, itemsets = mine_rules(train, settings)
    assert itemsets == 3 and len(rules) == 1
    assert rules[0]["antecedents"] == ("high",) and rules[0]["consequent"] == "fraud"
    validation = pd.DataFrame({"high": [True, True, False, False], "fraud": [False, False, True, False]})
    condition = Condition("high", "High amount", "TransactionAmt", "ge", 100, "Synthetic cutoff")
    evaluated, = evaluate_rules(rules, validation, (condition,))
    assert evaluated["train"]["confidence"] == pytest.approx(2 / 3)
    assert evaluated["validation"]["matches"] == 2 and evaluated["validation"]["support"] == 0
    assert evaluated["validation"]["confidence"] == 0 and evaluated["validation"]["lift"] == 0
