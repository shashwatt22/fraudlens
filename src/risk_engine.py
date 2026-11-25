"""Fixed review-priority scoring, historical context, and validation rankings."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .data_loader import AMOUNT, ID, IDENTITY_MISSING, TARGET, TIME
from .features import HISTORICAL_COLUMNS

WEIGHTS = (0.8, 0.2)
PRIORITY_CUTS = (40, 75)
PRIORITIES = ("Low", "Medium", "High")
REVIEW_PERCENTAGES = (1, 5, 10)
CONTEXT_COLUMNS = [AMOUNT, *HISTORICAL_COLUMNS, IDENTITY_MISSING]


def _bounded(values, maximum: float, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > maximum):
        raise ValueError(f"{name} must be finite and between 0 and {maximum:g}.")
    return values


def composite_risk(classifier_scores, anomaly_percentiles) -> np.ndarray:
    classifier = _bounded(classifier_scores, 1, "Classifier scores")
    anomaly = _bounded(anomaly_percentiles, 1, "Anomaly percentiles")
    if classifier.shape != anomaly.shape:
        raise ValueError("Classifier scores and anomaly percentiles must have matching shapes.")
    return 100 * (WEIGHTS[0] * classifier + WEIGHTS[1] * anomaly)


def review_priority(risk_scores) -> np.ndarray:
    scores = _bounded(risk_scores, 100, "Risk scores")
    return np.asarray(PRIORITIES)[np.searchsorted(PRIORITY_CUTS, scores, side="right")]


def require_alignment(reference: pd.DataFrame, other: pd.DataFrame, fields=(ID, TIME, TARGET)) -> None:
    if (reference[ID].isna().any() or not reference[ID].is_unique
            or any(not np.array_equal(reference[field], other[field]) for field in fields)):
        raise ValueError("Transaction IDs or partition row order differ between artifacts.")


def build_risk_table(frame: pd.DataFrame, classifier: pd.DataFrame, anomaly: pd.DataFrame,
                     threshold: float) -> pd.DataFrame:
    require_alignment(frame, classifier)
    require_alignment(frame, anomaly, fields=(ID,))
    scores = classifier["classifier_score"].to_numpy()
    frozen = classifier["classifier_decision"].to_numpy()
    if _bounded(threshold, 1, "Classification threshold").ndim != 0:
        raise ValueError("The frozen classification threshold must be a scalar.")
    if not np.array_equal(frozen, scores >= np.asarray(threshold, dtype=scores.dtype)):
        raise ValueError("The saved decisions do not match the frozen classification threshold.")
    raw_scores = anomaly["anomaly_score"].to_numpy()
    if not np.isfinite(raw_scores).all():
        raise ValueError("Anomaly scores must be finite.")
    percentiles = anomaly["anomaly_percentile"].to_numpy()
    risk = composite_risk(scores, percentiles)
    return frame[[ID, TIME, TARGET, *CONTEXT_COLUMNS]].assign(
        classifier_score=scores, classifier_decision=frozen, anomaly_score=raw_scores,
        anomaly_percentile=percentiles, risk_score=risk, review_priority=review_priority(risk),
    )


def assess_transaction(row: pd.Series, explanation: dict | None = None) -> dict:
    """Describe one risk-table row, leaving unknown quantities as null."""
    context_row = row[CONTEXT_COLUMNS].astype(object)
    context = context_row.where(context_row.notna(), None).to_dict()
    context["history_status"] = ("cold_start" if context["prior_txn_count"] == 0 else "prior_observations")
    context["identity_available"] = (None if context[IDENTITY_MISSING] is None else not bool(context[IDENTITY_MISSING]))
    return {ID: int(row[ID]), TIME: int(row[TIME]),
            "classifier_score": float(row["classifier_score"]), "classifier_decision": bool(row["classifier_decision"]),
            "anomaly_score": float(row["anomaly_score"]), "anomaly_percentile": float(row["anomaly_percentile"]),
            "risk_score": float(row["risk_score"]), "review_priority": str(row["review_priority"]),
            "behavioral_context": context,
            "shap": explanation if explanation is not None else {"status": "available_on_demand", ID: int(row[ID])}}


def validation_rankings(table: pd.DataFrame) -> dict:
    """Report AP, AUC, and alert-budget metrics on identical labeled validation rows."""
    ids = table[ID].to_numpy()
    labels = table[TARGET].to_numpy()
    if not table[ID].is_unique or table[ID].isna().any() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Review ranking needs unique transaction IDs and both validation classes.")
    results = {}
    for name, key in (("classifier", "classifier_score"), ("anomaly", "anomaly_score"), ("composite", "risk_score")):
        scores = table[key].to_numpy(dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("Review ranking needs finite scores.")
        ordered = np.lexsort((ids, -scores))
        positives = int(labels.sum())
        top = {}
        for percentage in REVIEW_PERCENTAGES:
            count = (len(labels) * percentage + 99) // 100
            caught = int(labels[ordered[:count]].sum())
            top[str(percentage)] = {"reviewed": count, "fraud_found": caught,
                                    "precision": caught / count, "recall": caught / positives}
        results[name] = {"average_precision": float(average_precision_score(labels, scores)),
                         "roc_auc": float(roc_auc_score(labels, scores)), "top_review": top}
    return results


def priority_distribution(table: pd.DataFrame) -> dict:
    counts = table["review_priority"].value_counts()
    return {name: {"count": int(counts.get(name, 0)), "fraction": float(counts.get(name, 0) / len(table))}
            for name in PRIORITIES}
