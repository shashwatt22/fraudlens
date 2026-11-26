"""Compare training-only anomaly, cluster and association configurations."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.association import Condition, MiningSettings, encode_basket, evaluate_rules, fit_conditions, mine_rules
from src.clustering import MAX_NEIGHBOR_LINKS, choose_configuration, cluster_metrics, sample_positions
from src.data_loader import TARGET

from .data import write_json
from .experiment import review_budgets

CORE = ("amount_log", "prior_txn_count", "amount_to_prior_mean", "seconds_since_prior_txn", "prior_count_24h")
ENRICHED = (*CORE, "card_address_anchor_prior_std", "card_address_anchor_prior_count_1h",
            "card_address_anchor_amount_ratio", "card_address_email_prior_count_24h", "card_address_amount_ratio")


def transform(values, train_end):
    compressed = np.log1p(np.maximum(values, 0))
    prep = SimpleImputer(strategy="median", keep_empty_features=True).fit(compressed[:train_end])
    output = np.column_stack([prep.transform(compressed), np.isnan(values).any(axis=1)])
    return StandardScaler().fit(output[:train_end]).transform(output)


def compare_anomaly(matrix, labels, end, threads):
    sample = sample_positions(end, 24_000)
    hold = labels[end:]
    results = []
    for size in (256, 768):
        model = IsolationForest(n_estimators=100, max_samples=size, n_jobs=threads, random_state=42)
        model.fit(matrix[sample])
        scores = -model.score_samples(matrix[end:])
        results.append({"sample_rows": len(sample), "tree_samples": size,
                        "average_precision": float(average_precision_score(hold, scores)),
                        "roc_auc": float(roc_auc_score(hold, scores)), "budgets": review_budgets(hold, scores)})
    return results


def compare_clusters(matrix, labels, end, threads):
    sample = sample_positions(end, 3_500)
    training = matrix[sample]
    tree = cKDTree(training)
    candidates, cluster_labels = [], []
    for radius in (.45, .65, .85, 1.05):
        links = int(tree.query_ball_point(training, radius, return_length=True, workers=threads).sum())
        for minimum in (10, 20):
            record = {"eps": radius, "min_samples": minimum, "neighbor_links": links}
            if links > MAX_NEIGHBOR_LINKS:
                candidates.append({**record, "status": "skipped"})
                cluster_labels.append(None)
                continue
            model = DBSCAN(eps=radius, min_samples=minimum, n_jobs=threads)
            predicted = model.fit_predict(training)
            candidates.append({**record, "status": "completed", **cluster_metrics(training, predicted)})
            cluster_labels.append(predicted)
    index = choose_configuration(candidates)
    selected, assigned = candidates[index], cluster_labels[index]
    resample = sample_positions(len(training), int(.9 * len(training)), seed=19)
    repeat = DBSCAN(eps=selected["eps"], min_samples=selected["min_samples"], n_jobs=threads).fit_predict(training[resample])
    stats = pd.DataFrame({"cluster": assigned, "fraud": labels[sample]}).groupby("cluster")["fraud"].agg(["size", "mean"])
    profile = [{"cluster": int(key), "rows": int(row["size"]), "fraud_rate": float(row["mean"])} for key, row in stats.iterrows()]
    return {"training_rows": len(sample), "profile": profile, "selected": selected, "candidates": candidates,
            "purged_10_percent_ari": float(adjusted_rand_score(assigned[resample], repeat))}


def compare_rules(frame, end):
    training, validation = frame.iloc[:end], frame.iloc[end:]
    base = fit_conditions(training)
    sample = training.iloc[sample_positions(len(training), 30_000)]
    extra = []
    for name, label in (("card_address_anchor_amount_ratio", "High amount vs. card/address history"),
                        ("card_address_email_prior_count_24h", "High 24h activity for card/address/email")):
        cut = float(training[name].dropna().quantile(.85))
        extra.append(Condition(name, label, name, "ge", cut, "Training p85", False))
    results = []
    for mode, conditions, settings in (("baseline", base, MiningSettings()),
                                        ("additional_past_features", (*base, *extra), MiningSettings(min_matches=200))):
        rules, itemsets = mine_rules(encode_basket(sample, conditions), settings)
        evaluated = evaluate_rules(rules, encode_basket(validation, conditions), conditions)
        results.append({"mode": mode, "settings": asdict(settings), "training_rows": len(sample), "itemsets": itemsets,
                        "rules": evaluated})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/v2"))
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    root = args.output
    cache = json.loads((root / "features.json").read_bytes())
    advanced = json.loads((root / "advanced.json").read_bytes())
    matrix = np.load(root / "features_advanced.npy", mmap_mode="r")
    frame = pd.read_parquet(root / "development_metadata.parquet", columns=[TARGET])
    index = dict(zip(advanced["features"], range(len(advanced["features"]))))
    data = pd.DataFrame({name: matrix[:, index[name]] for name in (*ENRICHED, "TransactionAmt", "is_new_product", "identity_missing")})
    data[TARGET] = frame[TARGET]
    released = {"feature_sha256": sha256((root / "advanced.json").read_bytes()).hexdigest(), "folds": []}
    with threadpool_limits(limits=args.threads):
        for number, fold in enumerate(cache["folds"], 1):
            tick = perf_counter()
            end, stop = fold["train_end"], fold["validation_end"]
            window = data.iloc[:stop]
            labels = window[TARGET].to_numpy(dtype=np.uint8)
            fold_report = {"fold": number, "families": {}, "rules": compare_rules(window, end)}
            for group, names in (("core", CORE), ("enriched", ENRICHED)):
                columns = transform(window[list(names)].to_numpy(dtype=np.float32), end)
                fold_report["families"][group] = {"anomaly": compare_anomaly(columns, labels, end, args.threads),
                                                  "clusters": compare_clusters(columns, labels, end, args.threads)}
            fold_report["runtime_seconds"] = perf_counter() - tick
            released["folds"].append(fold_report)
            write_json(root / "patterns.json", released)
            print(f"Pattern fold {number} completed in {fold_report['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
