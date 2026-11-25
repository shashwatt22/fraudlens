"""Small, testable operations behind review and evaluation interactions."""

import re

import numpy as np
import pandas as pd

ID = "TransactionID"
TARGET = "isFraud"
TIME = "TransactionDT"
RANK_COLUMNS = {"Classifier score": "classifier_score", "Composite risk": "risk_score",
                "Anomaly percentile": "anomaly_percentile"}
MODEL_NAMES = {"dummy": "Majority dummy", "logistic_regression": "Logistic regression", "random_forest": "Random forest",
               "xgboost": "XGBoost with history", "xgboost_basic": "XGBoost basic", "mlp": "PyTorch MLP"}
CORE_MODELS = ("dummy", "logistic_regression", "random_forest", "xgboost", "mlp")


def parse_id(value: str) -> int | None:
    value = value.strip()
    if not re.fullmatch(r"[0-9]{1,10}", value):
        return None
    number = int(value)
    return number if 0 < number <= np.iinfo(np.int32).max else None


def review_rows(frame: pd.DataFrame, ranking: str, priorities: list[str], query: str, limit: int) -> pd.DataFrame:
    selected = frame
    if priorities:
        selected = selected.loc[selected["review_priority"].isin(priorities)]
    if query.strip():
        transaction_id = parse_id(query)
        selected = selected.loc[selected[ID].eq(transaction_id)]
    return selected.sort_values([RANK_COLUMNS[ranking], ID], ascending=[False, True]).head(limit)


def demo_cases(frame: pd.DataFrame, preferred: int | None = None) -> dict[str, int]:
    """Select the earliest row of each outcome and the strongest anomaly with ID tie-breaks."""
    ordered = frame.sort_values([TIME, ID])
    positive = ordered["classifier_decision"].astype(bool)
    fraud = ordered[TARGET].eq(1)
    masks = {"Correctly detected fraud": positive & fraud, "Correctly cleared legitimate": ~positive & ~fraud,
             "False positive": positive & ~fraud, "Missed fraud": ~positive & fraud,
             "Behavioral context": ordered["prior_mean_amount"].notna() & ordered["prior_txn_count"].gt(0)}
    examples = {label: int(ordered.loc[mask, ID].iloc[0]) for label, mask in masks.items() if mask.any()}
    if preferred is not None and ordered[ID].eq(preferred).any():
        examples["Behavioral context"] = preferred
    if len(frame):
        examples["High anomaly"] = int(frame.sort_values(["anomaly_score", ID], ascending=[False, True])[ID].iloc[0])
    return examples


def outcome(label: int, decision: int) -> str:
    return {(1, 1): "True positive", (0, 0): "True negative", (0, 1): "False positive", (1, 0): "False negative"}[
        (int(label), int(decision))
    ]


def threshold_metrics(labels, scores, threshold: float) -> dict:
    y = np.asarray(labels)
    s = np.asarray(scores)
    if (y.ndim != 1 or s.shape != y.shape or not y.size or not np.isin(y, (0, 1)).all()
            or not np.isfinite(s).all() or not 0 <= threshold <= 1):
        raise ValueError("Threshold evaluation requires aligned labels, finite scores, and a cutoff in [0, 1].")
    actual = y == 1
    predicted = s >= threshold
    tp, fp = int(np.sum(actual & predicted)), int(np.sum(~actual & predicted))
    fn, tn = int(np.sum(actual)) - tp, int(np.sum(~actual)) - fp

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {"confusion_matrix": [[tn, fp], [fn, tp]], "flagged": tp + fp,
            "accuracy": (tn + tp) / len(y), "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "specificity": ratio(tn, tn + fp), "f1": ratio(2 * tp, 2 * tp + fp + fn)}
