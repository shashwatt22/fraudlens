"""Hand-check metrics, threshold selection and positional prediction alignment."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import evaluate_scores, prediction_table, select_threshold


def test_f1_threshold_uses_validation_arrays_and_highest_threshold_breaks_ties():
    validation_labels = np.array([1, 0, 1, 0, 0])
    validation_scores = np.array([0.8, 0.7, 0.6, 0.3, 0.2])
    threshold = select_threshold(validation_labels, validation_scores)
    assert threshold == 0.6
    assert select_threshold(np.array([1, 0, 0, 1]), np.array([0.9, 0.8, 0.7, 0.6])) == 0.9

    test_labels = np.array([0, 1, 0, 1])
    test_scores = np.array([0.9, 0.8, 0.4, 0.3])
    assert select_threshold(test_labels, test_scores) != threshold  # A distinct optimum would leak.
    test_metrics, _ = evaluate_scores(test_labels, test_scores, test_scores >= threshold)
    assert test_metrics["confusion_matrix"] == [[1, 1], [1, 1]]


def test_evaluation_metrics_and_curves_match_hand_calculation():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.6, 0.9])
    metrics, curves = evaluate_scores(labels, scores, scores >= 0.6)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert [metrics[key] for key in ("accuracy", "precision", "recall", "specificity", "f1",
                                    "roc_auc", "average_precision")] == pytest.approx(
        [0.75, 2 / 3, 1, 0.5, 0.8, 0.75, 5 / 6]
    )
    assert curves["roc"]["thresholds"][0] is None
    assert curves["precision_recall"]["precision"][-1] == 1


def test_majority_predictions_and_constant_score_ranking():
    metrics, _ = evaluate_scores(np.array([0, 0, 0, 1]), np.zeros(4), np.zeros(4))
    assert metrics["precision"] is None
    assert metrics["f1"] == 0
    assert metrics["roc_auc"] == 0.5
    assert metrics["average_precision"] == 0.25


def test_prediction_table_uses_original_id_order_and_rejects_invalid_shape():
    frame = pd.DataFrame({"TransactionID": [71, 14], "TransactionDT": [2, 5], "isFraud": [0, 1]},
                         index=[19, 7])
    aligned = prediction_table(frame, {"example": (np.array([0.8, 0.2]), np.array([1, 0]))})
    assert aligned["TransactionID"].tolist() == [71, 14]
    assert aligned["example_score"].tolist() == [0.8, 0.2]
    assert aligned["example_decision"].tolist() == [1, 0]
    with pytest.raises(ValueError, match="does not match"):
        prediction_table(frame, {"example": (np.array([[0.8], [0.2]]), np.array([1, 0]))})
