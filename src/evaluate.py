"""Evaluate binary classifier decisions and continuous ranking scores."""

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

from .data_loader import ID, TARGET, TIME


def positive_scores(model, matrix) -> np.ndarray:
    """Read scores for class 1 using the estimator's actual class order."""
    index = np.flatnonzero(model.classes_ == 1).item()
    scores = model.predict_proba(matrix)[:, index]
    if scores.shape != (matrix.shape[0],) or not np.isfinite(scores).all():
        raise ValueError("The model did not return one finite score per transaction.")
    return scores


def select_threshold(validation_labels: np.ndarray, validation_scores: np.ndarray) -> float:
    """Maximize validation F1; ties favor the highest threshold (fewer alerts)."""
    if set(np.unique(validation_labels)) != {0, 1}:
        raise ValueError("Threshold selection needs both classes in validation.")
    precision, recall, thresholds = precision_recall_curve(validation_labels, validation_scores)
    total = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], total, out=np.zeros_like(total), where=total > 0)
    return float(thresholds[-1 - np.argmax(f1[::-1])])


def _ratio(top: int, bottom: int) -> float | None:
    return top / bottom if bottom else None


def evaluate_scores(labels: np.ndarray, scores: np.ndarray, decisions: np.ndarray) -> tuple[dict, dict]:
    """Return scalar metrics and curve coordinates. Undefined values are JSON null."""
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(labels, decisions, labels=[0, 1]).ravel())
    metrics = {
        "accuracy": _ratio(tp + tn, len(labels)),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "roc_auc": None,
        "average_precision": None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
    curves = {"roc": None, "precision_recall": None}
    if set(np.unique(labels)) == {0, 1}:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["average_precision"] = float(average_precision_score(labels, scores))
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        curves = {
            "roc": {"false_positive_rate": fpr.tolist(), "true_positive_rate": tpr.tolist(),
                    "thresholds": [float(t) if np.isfinite(t) else None for t in roc_thresholds]},
            "precision_recall": {"precision": precision.tolist(), "recall": recall.tolist(),
                                 "thresholds": pr_thresholds.tolist()},
        }
    return metrics, curves


def prediction_table(frame: pd.DataFrame, outputs: dict) -> pd.DataFrame:
    """Attach score and decision arrays positionally to the original transaction order."""
    result = frame[[ID, TIME, TARGET]].reset_index(drop=True).copy()
    if result[ID].isna().any() or not result[ID].is_unique:
        raise ValueError("Prediction rows need unique, nonmissing transaction IDs.")
    for name, (scores, decisions) in outputs.items():
        scores, decisions = np.asarray(scores), np.asarray(decisions)
        if scores.shape != (len(result),) or decisions.shape != (len(result),):
            raise ValueError(f"{name} prediction output does not match the transaction row count.")
        result[f"{name}_score"] = scores
        result[f"{name}_decision"] = decisions.astype(np.uint8)
    return result
