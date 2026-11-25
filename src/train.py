"""Train and evaluate the Phase 2 classical models from Phase 1 Parquet files."""

import argparse
from collections.abc import Generator
from contextlib import closing
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from .data_loader import ID, TARGET
from .evaluate import evaluate_scores, positive_scores, prediction_table, select_threshold
from .preprocessing import FraudPreprocessor
from .training_data import complete_run, load_partition, read_phase1_manifest

SEED = 42
THREADS = 6
CORE = ("dummy", "logistic_regression", "random_forest", "xgboost")
CORE_LEARNED = CORE[1:]
MODELS_BY_FEATURE_SET = {"behavioral": CORE, "basic": ("xgboost_basic",)}
ABLATION_METRICS = ("average_precision", "roc_auc", "precision", "recall", "f1")


def make_models(labels: np.ndarray, threads: int) -> dict:
    positives = np.count_nonzero(labels == 1)
    imbalance_ratio = (len(labels) - positives) / positives
    xgb_settings = dict(n_estimators=240, max_depth=5, learning_rate=0.075, min_child_weight=4,
                        subsample=0.85, colsample_bytree=0.8, max_bin=128, n_jobs=threads,
                        tree_method="hist", device="cpu", scale_pos_weight=imbalance_ratio,
                        objective="binary:logistic", eval_metric="logloss", random_state=SEED)
    return {
        "dummy": DummyClassifier(strategy="most_frequent", random_state=SEED),
        "logistic_regression": LogisticRegression(solver="lbfgs", max_iter=500, class_weight="balanced",
                                                 random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=120, max_depth=12, min_samples_leaf=4,
                                                class_weight="balanced", n_jobs=threads, random_state=SEED),
        "xgboost": XGBClassifier(**xgb_settings),
        "xgboost_basic": XGBClassifier(**xgb_settings),
    }


def champion_from_validation(validation_metrics: dict) -> str:
    """Choose among learned classifiers by AP, breaking ties in CORE_LEARNED order."""
    return max(CORE_LEARNED, key=lambda name: validation_metrics[name]["average_precision"])


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _classify(model, matrix, scores: np.ndarray, threshold: float | None) -> np.ndarray:
    return model.predict(matrix) if threshold is None else (scores >= threshold).astype(np.uint8)


def _matrix_bytes(matrix) -> int:
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


def _model_parameters(model) -> dict:
    # XGBoost uses NaN as its missing sentinel; JSON represents this default as null.
    return {key: None if isinstance(value, float) and not np.isfinite(value) else value
            for key, value in model.get_params().items()}


def _train_on_validation(stage: Path, train, validation, threads: int) -> tuple[dict, dict, dict, dict, dict]:
    labels = train[TARGET].to_numpy()
    validation_labels = validation[TARGET].to_numpy()
    models = make_models(labels, threads)
    model_info, feature_info, metrics, curves, outputs = {}, {}, {}, {}, {}
    for feature_set, names in MODELS_BY_FEATURE_SET.items():
        started = perf_counter()
        preprocessor = FraudPreprocessor(include_history=feature_set == "behavioral")
        train_matrix = preprocessor.fit_transform(train)
        validation_matrix = preprocessor.transform(validation)
        feature_info[feature_set] = {
            "raw_features": preprocessor.feature_names_in_.tolist(),
            "transformed_features": preprocessor.get_feature_names_out().tolist(),
            "raw_dimension": int(preprocessor.n_features_in_),
            "transformed_dimension": train_matrix.shape[1],
            "train_csr_bytes": _matrix_bytes(train_matrix),
            "validation_csr_bytes": _matrix_bytes(validation_matrix),
            "preprocess_seconds": perf_counter() - started,
        }
        joblib.dump(preprocessor, stage / f"preprocessor_{feature_set}.joblib")
        for name in names:
            model = models.pop(name)
            started = perf_counter()
            model.fit(train_matrix, labels)
            fit_seconds = perf_counter() - started
            inference_started = perf_counter()
            scores = positive_scores(model, validation_matrix)
            inference_seconds = perf_counter() - inference_started
            threshold = None if name == "dummy" else select_threshold(validation_labels, scores)
            decisions = _classify(model, validation_matrix, scores, threshold)
            metrics[name], curves[name] = evaluate_scores(validation_labels, scores, decisions)
            outputs[name] = (scores, decisions)
            model_info[name] = {"feature_set": feature_set, "configuration": _model_parameters(model),
                                "fit_seconds": fit_seconds, "threshold": threshold,
                                "validation_inference_seconds": inference_seconds}
            joblib.dump(model, stage / f"{name}.joblib", compress=3)
            print(f"{name}: fit {fit_seconds:.2f}s, validation AP {metrics[name]['average_precision']:.6f}",
                  flush=True)
            del model
        del train_matrix, validation_matrix
    return model_info, feature_info, metrics, curves, outputs


def _score_test(stage: Path, test, frozen: dict, model_info: dict) -> tuple[dict, dict, dict]:
    labels = test[TARGET].to_numpy()
    metrics, curves, outputs = {}, {}, {}
    for feature_set, names in MODELS_BY_FEATURE_SET.items():
        preprocessor = joblib.load(stage / f"preprocessor_{feature_set}.joblib")
        matrix = preprocessor.transform(test)
        for name in names:
            model = joblib.load(stage / f"{name}.joblib")
            inference_started = perf_counter()
            scores = positive_scores(model, matrix)
            model_info[name]["test_inference_seconds"] = perf_counter() - inference_started
            decisions = _classify(model, matrix, scores, frozen["thresholds"][name])
            metrics[name], curves[name] = evaluate_scores(labels, scores, decisions)
            outputs[name] = (scores, decisions)
            del model
        del matrix
    return metrics, curves, outputs


def _write_partition_outputs(stage: Path, mode: str, processed_rows: int, name: str,
                             frame, metrics: dict, curves: dict, outputs: dict) -> None:
    context = {"mode": mode, "partition": name, "rows": len(frame)}
    _write_json(stage / f"{name}_metrics.json", {**context, "models": metrics})
    _write_json(stage / f"{name}_curves.json", {**context, "models": curves})
    table = pa.Table.from_pandas(prediction_table(frame, outputs), preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update({b"fraudlens.mode": mode.encode(), b"fraudlens.partition": name.encode(),
                     b"fraudlens.processed_rows": str(processed_rows).encode(),
                     b"fraudlens.partition_rows": str(len(frame)).encode()})
    pq.write_table(table.replace_schema_metadata(metadata), stage / f"{name}_predictions.parquet")


def _ablation_results(validation_metrics: dict, test_metrics: dict, models: dict) -> dict:
    comparisons = {}
    for name, metrics in [("validation", validation_metrics), ("test", test_metrics)]:
        basic = {key: metrics["xgboost_basic"][key] for key in ABLATION_METRICS}
        history = {key: metrics["xgboost"][key] for key in ABLATION_METRICS}
        delta = {key: history[key] - basic[key] if history[key] is not None and basic[key] is not None else None
                 for key in ABLATION_METRICS}
        comparisons[name] = {"basic": basic, "behavioral": history, "delta_behavioral_minus_basic": delta}
    return {"basic_model": "xgboost_basic", "behavioral_model": "xgboost",
            "thresholds": {name: models[name]["threshold"] for name in ("xgboost_basic", "xgboost")},
            "fit_seconds": {name: models[name]["fit_seconds"] for name in ("xgboost_basic", "xgboost")},
            **comparisons}


def prepare_training(artifact_root: Path, mode: str, threads: int = THREADS) -> Generator[tuple[Path, dict], None, Path]:
    """Fit the classical models and pause before loading test."""
    started = perf_counter()
    mode_dir = artifact_root / mode
    source, source_hash = read_phase1_manifest(mode_dir)
    train = load_partition(mode_dir, source, "train")
    validation = load_partition(mode_dir, source, "validation")
    if validation[ID].isin(train[ID]).any():
        raise ValueError("Train and validation share transaction IDs.")
    row_count = source["processed_rows"]
    print(f"{mode}: {len(train):,} train and {len(validation):,} validation rows, {threads} CPU threads",
          flush=True)

    phase_dir = mode_dir / "phase2"
    runs_dir = phase_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / uuid4().hex
    with TemporaryDirectory(prefix=".phase2-", dir=phase_dir) as temp, threadpool_limits(limits=threads):
        stage = Path(temp) / "run"
        stage.mkdir()
        models, features, validation_metrics, validation_curves, validation_outputs = _train_on_validation(
            stage, train, validation, threads
        )
        _write_partition_outputs(stage, mode, row_count, "validation", validation,
                                 validation_metrics, validation_curves, validation_outputs)
        frozen = {
            "mode": mode, "processed_rows": row_count, "selection_partition": "validation",
            "champion": champion_from_validation(validation_metrics),
            "champion_rule": "Maximum validation average precision among learned core classifiers; ties favor "
                             "logistic regression, then random forest, then XGBoost.",
            "threshold_rule": "Maximum validation F1; ties favor the highest threshold; decisions use score >= threshold.",
            "thresholds": {name: info["threshold"] for name, info in models.items()},
            "dummy_rule": "Standard most-frequent-class predict; no tuned threshold.",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(stage / "selection.json", frozen)
        manifest = {
            "phase": 2, "schema_version": 1, "mode": mode, "processed_rows": row_count, "seed": SEED,
            "threads": threads, "device": "cpu", "test_scope": "full chronological holdout" if mode == "full"
                               else "development diagnostics",
            "versions": {"python": platform.python_version(), **{name: version(name) for name in
                         ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "xgboost", "joblib", "threadpoolctl")}},
            "phase1_manifest_sha256": source_hash, "phase1_partition_files": {
                name: source["files"][name] for name in ("train", "validation", "test")},
            "partitions": source["partitions"], "champion": frozen["champion"], "features": features,
            "models": models, "run_directory": run_dir.relative_to(phase_dir).as_posix(),
            "files": {
                "preprocessors": {name: f"preprocessor_{name}.joblib" for name in features},
                "classifiers": {name: f"{name}.joblib" for name in models},
                "selection": "selection.json", "ablation": "ablation.json",
                "validation_metrics": "validation_metrics.json", "test_metrics": "test_metrics.json",
                "validation_curves": "validation_curves.json", "test_curves": "test_curves.json",
                "validation_predictions": "validation_predictions.parquet", "test_predictions": "test_predictions.parquet",
            },
            "completed_stages": ["preprocessing", "training", "validation", "freeze_selection", "test", "ablation"],
        }
        prior_ids = pd.concat([train[ID], validation[ID]], ignore_index=True)
        del train, validation, validation_curves, validation_outputs
        wait_started = perf_counter()
        yield stage, manifest
        started += perf_counter() - wait_started
        print(f"Frozen champion: {frozen['champion']}; opening the {mode} test partition", flush=True)
        test = load_partition(mode_dir, source, "test")
        if test[ID].isin(prior_ids).any():
            raise ValueError("Test shares transaction IDs with train or validation.")
        test_metrics, test_curves, test_outputs = _score_test(stage, test, frozen, models)
        _write_partition_outputs(stage, mode, row_count, "test", test, test_metrics, test_curves, test_outputs)
        _write_json(stage / "ablation.json", {"mode": mode, "processed_rows": row_count,
                                             **_ablation_results(validation_metrics, test_metrics, models)})

        manifest["runtime_seconds"] = perf_counter() - started
        _write_json(stage / "manifest.json", manifest)
        stage.rename(run_dir)
        latest = Path(temp) / "manifest.json"
        _write_json(latest, manifest)
        latest.replace(phase_dir / "manifest.json")

    result = phase_dir / "manifest.json"
    print(f"Phase 2 {mode}: {manifest['runtime_seconds']:.2f}s, manifest {result.resolve()}")
    return result


def run_training(artifact_root: Path, mode: str, threads: int = THREADS) -> Path:
    with closing(prepare_training(artifact_root, mode, threads)) as run:
        next(run)
        return complete_run(run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full", "smoke"), default="quick")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=THREADS, metavar="1-8")
    args = parser.parse_args()
    try:
        run_training(args.artifact_root, args.mode, args.threads)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Phase 2 error: {error}\n")


if __name__ == "__main__":
    main()
