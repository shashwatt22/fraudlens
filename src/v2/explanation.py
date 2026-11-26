"""Verify native tree contributions for the frozen V2 model or one component."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
import xgboost as xgb

from src.data_loader import ID, TIME
from .challengers import BOOSTING, _encode_frequency
from .data import sha256, write_json
from .experiment import TRIALS
from .final import _cache_matrices, _views


def tree_contributions(name, model, payload, view, rows, threads):
    family = "xgb" if name in TRIALS else BOOSTING[name].family
    start, stop = rows
    features = list(view.features)
    if family == "xgb":
        matrix = xgb.DMatrix(view.read(start, stop), feature_names=features, feature_types=view.types,
                             nthread=threads, enable_categorical=True)
        contributions = model.predict(matrix, pred_contribs=True)
        margins = model.predict(matrix, output_margin=True)
        gain = model.get_score(importance_type="gain")
    elif family == "lgbm":
        matrix = view.read(start, stop)
        contributions = model.predict(matrix, pred_contrib=True, num_threads=threads)
        margins = model.predict(matrix, raw_score=True, num_threads=threads)
        gain = dict(zip(features, model.feature_importance(importance_type="gain")))
    else:
        from catboost import Pool

        matrix = _encode_frequency(view.data[start:stop, view.columns], payload["frequency_maps"])
        contributions = model.get_feature_importance(Pool(matrix), type="ShapValues", thread_count=threads)
        margins = model.predict(matrix, prediction_type="RawFormulaVal", thread_count=threads)
        gain = dict(zip(features, model.feature_importances_))
    margin_error = float(np.max(np.abs(contributions.sum(axis=1, dtype=np.float64) - margins)))
    if margin_error > 1e-4:
        raise ValueError(f"V2 explanation does not reconstruct the raw margin ({margin_error:g}).")
    return contributions, margins, margin_error, features, gain


class V2Explainer:
    def __init__(self, root, threads=6):
        self.root, self.threads = root, threads
        self.policy = json.loads((root / "decision_freeze.json").read_bytes())
        result = json.loads((root / "retrospective_results.json").read_bytes())
        if result["champion"] != self.policy["champion"]:
            raise ValueError("V2 results and frozen explanation model disagree.")
        self.champion = result["champion"]
        supported = set(TRIALS) | set(BOOSTING)
        supported.discard("v1")
        if self.champion in supported:
            self.name = self.champion
        else:
            blend = self.policy["blends"].get(self.champion, {})
            trees = {name: weight for name, weight in blend.items() if name in supported and weight > 0}
            if not trees:
                raise ValueError("The selected V2 scorer has no verified tree explanation.")
            self.name = max(trees, key=trees.get)
        details = self.policy["models"][self.name]
        path = root / details["file"]
        if sha256(path) != details["sha256"]:
            raise ValueError("V2 explanation model differs from the frozen model.")
        if self.name in TRIALS:
            self.model = xgb.Booster(params={"nthread": threads})
            self.model.load_model(path)
            self.payload = {}
        else:
            self.payload = joblib.load(path)
            self.model = self.payload["model"]
        self.matrices = _cache_matrices(root / "retrospective")
        self.metadata = pd.read_parquet(root / "retrospective" / "development_metadata.parquet", columns=[ID, TIME])
        self.view = _views(self.name, self.matrices, self.policy["train_end"], self.metadata[TIME].to_numpy())
        self.scores = pd.read_parquet(root / "retrospective" / "champion_scores.parquet", columns=[ID, "raw_score"])
        self.offset = len(self.metadata) - len(self.scores)
        if not np.array_equal(self.metadata[ID].iloc[self.offset:].to_numpy(), self.scores[ID].to_numpy()):
            raise ValueError("V2 explanation and frozen prediction rows do not align.")

    def explain(self, transaction_id):
        rows = np.flatnonzero(self.scores[ID].to_numpy() == transaction_id)
        if len(rows) != 1:
            raise ValueError("V2 explanation requires a single transaction in the retrospective holdout.")
        position = self.offset + rows.item()
        contributions, margins, error, features, gain = tree_contributions(
            self.name, self.model, self.payload, self.view, (position, position + 1), self.threads)
        score = float(expit(margins[0]))
        score_error = None if self.champion != self.name else abs(score - self.scores["raw_score"].iloc[rows.item()])
        if score_error is not None and score_error > 2e-6:
            raise ValueError(f"V2 raw score reconstruction changed ({score_error:g}).")
        items = sorted(zip(features, map(float, contributions[0, :-1]), strict=True), key=lambda item: abs(item[1]), reverse=True)
        importance = sorted(((name, float(value)) for name, value in gain.items()), key=lambda item: item[1], reverse=True)
        return {"model": self.name, "champion": self.champion, "component_only": self.champion != self.name,
                "base_value": float(contributions[0, -1]), "raw_margin": float(margins[0]), "score": score,
                "margin_error": error, "raw_score_error": None if score_error is None else float(score_error),
                "contributions": items[:12], "model_importance": importance[:20]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/v2"))
    args = parser.parse_args()
    explainer = V2Explainer(args.output)
    positions = np.linspace(0, len(explainer.scores) - 1, 5, dtype=int)
    reports = [explainer.explain(int(i)) for i in explainer.scores[ID].iloc[positions]]
    errors = [row["raw_score_error"] for row in reports if row["raw_score_error"] is not None]
    write_json(args.output / "explanation_audit.json", {"checked": len(reports), "model": explainer.name,
               "component_only": explainer.champion != explainer.name,
               "max_margin_error": max(report["margin_error"] for report in reports),
               "max_score_error": max(errors, default=None), "model_importance": reports[0]["model_importance"]})


if __name__ == "__main__":
    main()
