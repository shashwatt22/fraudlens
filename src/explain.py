"""Exact, sparse-input TreeSHAP for the saved XGBoost classifier."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix_csr
from scipy.special import expit
import shap
from xgboost import DMatrix

from .evaluate import positive_scores
from .features import CATEGORICAL_BASIC

METHOD = "shap.TreeExplainer, exact path-dependent TreeSHAP"
UNITS = "raw XGBoost margin (log-odds)"
MARGIN_ATOL = 2e-5
SCORE_ATOL = 1e-6
DISPLAY_NAMES = {
    "TransactionAmt": "Transaction amount", "log_amount": "Log transaction amount",
    "identity_missing": "Identity record missing", "prior_txn_count": "Prior transaction count",
    "prior_mean_amount": "Prior mean amount", "amount_to_prior_mean": "Amount relative to prior mean",
    "seconds_since_prior_txn": "Seconds since prior transaction", "prior_count_1h": "Prior one-hour count",
    "prior_count_24h": "Prior 24-hour count", "is_new_product": "Previously unseen product category",
    "ProductCD": "Product category", "card1": "Card code 1", "card2": "Card code 2",
    "card3": "Card code 3", "card4": "Card category 4", "card5": "Card code 5", "card6": "Card category 6",
    "addr1": "Address code 1", "addr2": "Address code 2", "P_emaildomain": "Purchaser email domain",
    "R_emaildomain": "Recipient email domain", "DeviceType": "Device type", "DeviceInfo": "Device information",
}


def feature_mapping(preprocessor, expected_names: list[str]) -> list[dict]:
    """Follow the fitted transformer's order, including each one-hot block."""
    names = preprocessor.get_feature_names_out().tolist()
    numeric = preprocessor.numeric_columns_
    groups = [*numeric, *numeric]
    for column, categories in zip(CATEGORICAL_BASIC, preprocessor.encoder_.categories_, strict=True):
        groups.extend([column] * len(categories))
    if names != expected_names or len(set(names)) != len(names) or len(groups) != len(names):
        raise ValueError("The fitted transformed feature order does not match the saved classifier schema.")
    return [{"name": name, "original_feature": group, "display_name": DISPLAY_NAMES[group]}
            for name, group in zip(names, groups, strict=True)]


@dataclass
class AttributionBatch:
    values: np.ndarray
    grouped: np.ndarray
    base: float
    margins: np.ndarray
    scores: np.ndarray
    checks: dict


class FraudExplainer:
    def __init__(self, model, preprocessor, expected_names: list[str], threads: int = 6):
        self.mapping = feature_mapping(preprocessor, expected_names)
        self.feature_names = expected_names
        self.group_names = list(dict.fromkeys(item["original_feature"] for item in self.mapping))
        self.group_indices = {group: [i for i, item in enumerate(self.mapping) if item["original_feature"] == group]
                              for group in self.group_names}
        self.model = model
        self.preprocessor = preprocessor
        self.booster = model.get_booster()
        if self.booster.num_features() != len(expected_names):
            raise ValueError("The saved model and preprocessor have different feature dimensions.")
        self.threads = threads
        self.tree = shap.TreeExplainer(model, model_output="raw", feature_perturbation="tree_path_dependent")

    def explain(self, frame: pd.DataFrame, saved_scores: np.ndarray) -> AttributionBatch:
        """Explain only requested rows, checking margins and saved CSR scores."""
        matrix = self.preprocessor.transform(frame)
        if not isspmatrix_csr(matrix) or matrix.dtype != np.float32:
            raise ValueError("TreeSHAP requires the deployed float32 CSR representation.")
        data = DMatrix(matrix, missing=self.model.missing, nthread=self.threads)
        values = np.asarray(self.tree.shap_values(data, approximate=False, check_additivity=True))
        base = float(np.asarray(self.tree.expected_value).item())
        margins = self.booster.predict(data, output_margin=True)
        scores = positive_scores(self.model, matrix)
        if values.shape != matrix.shape or not np.isfinite(values).all() or not np.isfinite(margins).all():
            raise ValueError("TreeSHAP returned invalid attributions or margins.")
        saved_scores = np.asarray(saved_scores)
        if saved_scores.shape != scores.shape or not np.isfinite(saved_scores).all():
            raise ValueError("Expected one saved classifier score per explanation row.")
        grouped = np.column_stack([values[:, indices].sum(axis=1, dtype=np.float64)
                                   for indices in self.group_indices.values()])
        totals = values.sum(axis=1, dtype=np.float64)
        checks = {
            "max_margin_error": float(np.max(np.abs(base + totals - margins))),
            "max_group_sum_error": float(np.max(np.abs(grouped.sum(axis=1) - totals))),
            "max_sigmoid_score_error": float(np.max(np.abs(expit(margins) - scores))),
            "max_saved_score_error": float(np.max(np.abs(scores - saved_scores))),
        }
        if (checks["max_margin_error"] > MARGIN_ATOL or checks["max_group_sum_error"] > MARGIN_ATOL
                or max(checks["max_sigmoid_score_error"], checks["max_saved_score_error"]) > SCORE_ATOL):
            raise ValueError(f"TreeSHAP failed additivity, grouping, or saved-score checks: {checks}")
        return AttributionBatch(values, grouped, base, margins, scores, checks)

    def local(self, frame: pd.DataFrame, saved_score: float) -> dict:
        if len(frame) != 1:
            raise ValueError("A local explanation requires exactly one transaction.")
        batch = self.explain(frame, np.asarray([saved_score]))
        groups = [{"feature": name, "display_name": DISPLAY_NAMES[name], "contribution": float(value)}
                  for name, value in zip(self.group_names, batch.grouped[0], strict=True)]
        return {"method": METHOD, "units": UNITS, "base_value": batch.base,
                "raw_margin": float(batch.margins[0]), "reconstructed_score": float(expit(batch.margins[0])),
                "grouped_contributions": sorted(groups, key=lambda item: -abs(item["contribution"])),
                "transformed_contributions": [{**item, "contribution": float(value)}
                                              for item, value in zip(self.mapping, batch.values[0], strict=True)],
                "checks": batch.checks}

    def global_importance(self, batch: AttributionBatch) -> dict:
        def ranked(items, values):
            importances = np.abs(values).mean(axis=0, dtype=np.float64)
            return sorted([{**item, "mean_absolute_shap": float(value)}
                           for item, value in zip(items, importances, strict=True)],
                          key=lambda item: -item["mean_absolute_shap"])

        return {"method": METHOD, "units": UNITS, "base_value": batch.base, "checks": batch.checks,
                "grouping": "Mean of absolute signed group sums across sampled rows.",
                "transformed": ranked(self.mapping, batch.values),
                "grouped": ranked([{"feature": name, "display_name": DISPLAY_NAMES[name]}
                                   for name in self.group_names], batch.grouped)}
