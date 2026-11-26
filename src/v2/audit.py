"""Check influential anonymized inputs with single-feature development probes."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import TARGET
from .data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/v2"))
    args = parser.parse_args()
    root = args.output
    manifest = json.loads((root / "features.json").read_bytes())
    model = joblib.load(root / "challengers" / "xgb_advanced_depth8" / "model3.joblib")["model"]
    gain = model.get_score(importance_type="gain")
    checked = sorted((name for name in gain if name in manifest["features"]), key=gain.get, reverse=True)[:10]
    matrix = np.load(root / "features.npy", mmap_mode="r")
    labels = pd.read_parquet(root / "development_metadata.parquet", columns=[TARGET])[TARGET].to_numpy()
    reports = []
    for name in checked:
        values = np.array(matrix[:, manifest["features"].index(name)], copy=True)[:, None]
        ap = []
        for fold in manifest["folds"]:
            end, stop = fold["train_end"], fold["validation_end"]
            probe = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42).fit(values[:end], labels[:end])
            scores = probe.predict_proba(values[end:stop])[:, 1]
            ap.append(float(average_precision_score(labels[end:stop], scores)))
        reports.append({"feature": name, "validation_ap": ap})
    write_json(root / "univariate_audit.json", reports)


if __name__ == "__main__":
    main()
