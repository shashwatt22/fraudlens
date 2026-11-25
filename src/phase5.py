"""Build offline behavioral clustering and association artifacts from Phase 1."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .association import (MiningSettings, baseline, condition_definitions, encode_basket, evaluate_rules,
                          fit_conditions, mine_rules)
from .clustering import (EPS_VALUES, FEATURES, MAX_NEIGHBOR_LINKS, MIN_SAMPLES, SCALED_FEATURES, SEED,
                         SILHOUETTE_CAP, run_clustering, sample_positions)
from .data_loader import ID
from .phase3 import _write_table
from .train import THREADS, _write_json
from .training_data import load_partition, read_phase1_manifest

CLUSTER_LIMITS = {"quick": 5_000, "smoke": 5_000, "full": 12_000}
MINING_LIMIT = 30_000


def run_phase5(artifact_root: Path, mode: str, threads: int = THREADS) -> Path:
    started = perf_counter()
    mode_dir = artifact_root / mode
    source, digest = read_phase1_manifest(mode_dir)
    training = load_partition(mode_dir, source, "train")
    context = {"mode": mode, "processed_rows": source["processed_rows"], "seed": SEED}
    phase_dir = mode_dir / "phase5"
    runs_dir = phase_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / uuid4().hex
    files = {"clusters": "cluster_assignments.parquet", "profiles": "cluster_profiles.parquet",
             "clustering": "clustering.json", "cluster_preprocessor": "cluster_preprocessor.joblib", "pca": "pca.joblib",
             "mining_sample": "mining_training_ids.parquet", "association": "association_rules.json"}

    with TemporaryDirectory(prefix=".phase5-", dir=phase_dir) as temp, threadpool_limits(limits=threads):
        stage = Path(temp) / "run"
        stage.mkdir()
        clock = perf_counter()
        cluster = run_clustering(training, CLUSTER_LIMITS[mode], threads)
        cluster_seconds = perf_counter() - clock
        pca_variance = [float(value) if np.isfinite(value) else None for value in cluster.pca.explained_variance_ratio_]
        cluster_report = {**context, "partition": "train", "sample_rows": len(cluster.assignments),
                          "training_population_rows": len(training), "sample_limit": CLUSTER_LIMITS[mode],
                          "sampling": "Uniform without replacement by row position, then chronological order.",
                          "raw_features": list(FEATURES), "processed_features": list(SCALED_FEATURES),
                          "preprocessing": "Log1p for counts, ratio and gap; training-sample medians, incomplete-history flag and standard scaling.",
                          "clustering_space": "Standardized original input space; PCA is used for visualization only.",
                          "eps_grid": EPS_VALUES, "min_samples_grid": MIN_SAMPLES, "max_neighbor_links": MAX_NEIGHBOR_LINKS,
                          "silhouette_sample_cap": SILHOUETTE_CAP, "pca_explained_variance_ratio": pca_variance,
                          "selection": "Highest non-noise silhouette among candidates with at most 55% noise and 90% largest-cluster share among inliers; fallback to any valid silhouette, then least noise.",
                          "selected": cluster.selected, "comparisons": cluster.comparisons, "seconds": cluster_seconds}
        _write_table(stage / files["clusters"], cluster.assignments, mode, "train_sample", source["processed_rows"])
        _write_table(stage / files["profiles"], cluster.profiles, mode, "train_sample_profiles", source["processed_rows"])
        joblib.dump(cluster.preprocessor, stage / files["cluster_preprocessor"])
        joblib.dump(cluster.pca, stage / files["pca"])
        _write_json(stage / files["clustering"], cluster_report)
        print(f"DBSCAN: {len(cluster.assignments):,} sampled training rows, {cluster.selected['clusters']} clusters, "
              f"{cluster.selected['noise_fraction']:.1%} noise in {cluster_seconds:.2f}s.", flush=True)

        clock = perf_counter()
        conditions = fit_conditions(training)
        mining_sample = training.iloc[sample_positions(len(training), MINING_LIMIT)]
        basket = encode_basket(mining_sample, conditions)
        settings = MiningSettings()
        rules, itemsets = mine_rules(basket, settings)
        discovery_seconds = perf_counter() - clock

        validation = load_partition(mode_dir, source, "validation")
        if validation[ID].isin(training[ID]).any():
            raise ValueError("Training and validation share transaction IDs.")
        clock = perf_counter()
        validation_basket = encode_basket(validation, conditions)
        evaluated = evaluate_rules(rules, validation_basket, conditions)
        validation_seconds = perf_counter() - clock
        association_report = {**context, "discovery_partition": "train", "evaluation_partition": "validation",
                              "sampling": "Uniform without replacement; the validation partition is scored in full.",
                              "conditions_fitted_on_rows": len(training), "sample_limit": MINING_LIMIT,
                              "training_baseline": baseline(basket), "validation_baseline": baseline(validation_basket),
                              "settings": asdict(settings), "conditions": condition_definitions(conditions),
                              "frequent_itemsets": itemsets, "rules": evaluated,
                              "discovery_seconds": discovery_seconds, "validation_seconds": validation_seconds}
        _write_table(stage / files["mining_sample"], mining_sample[[ID]], mode, "train_sample", source["processed_rows"])
        _write_json(stage / files["association"], association_report)
        print(f"Apriori: {len(evaluated)} retained rules across {len(basket):,} training and {len(validation_basket):,} "
              f"validation rows, discovery {discovery_seconds:.2f}s, evaluation {validation_seconds:.2f}s.", flush=True)

        manifest = {**context, "phase": 5, "schema_version": 1, "threads": threads, "partitions": source["partitions"],
                    "used_partitions": ["train", "validation"], "phase1_manifest_sha256": digest,
                    "phase1_files": {name: source["files"][name] for name in ("train", "validation")},
                    "versions": {"python": platform.python_version(), **{name: version(name) for name in
                                 ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "mlxtend", "joblib")}},
                    "run_directory": run_dir.relative_to(phase_dir).as_posix(), "files": files,
                    "timings_seconds": {"clustering": cluster_seconds, "rule_discovery": discovery_seconds,
                                        "rule_validation": validation_seconds}, "runtime_seconds": perf_counter() - started,
                    "completed_stages": ["clustering", "pca", "cluster_profiles", "rule_discovery", "rule_validation"]}
        _write_json(stage / "manifest.json", manifest)
        stage.rename(run_dir)
        latest = Path(temp) / "manifest.json"
        _write_json(latest, manifest)
        latest.replace(phase_dir / "manifest.json")

    print(f"Phase 5 {mode}: {manifest['runtime_seconds']:.2f}s. Manifest: {(phase_dir / 'manifest.json').resolve()}", flush=True)
    return phase_dir / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(CLUSTER_LIMITS), default="quick")
    parser.add_argument("--threads", type=int, choices=range(1, 9), metavar="1-8", default=THREADS)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    try:
        run_phase5(args.artifact_root, args.mode, args.threads)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Phase 5 error: {error}\n")


if __name__ == "__main__":
    main()
