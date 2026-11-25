"""Run the matched syllabus and from-scratch experiments from saved Phase 1 data."""

import argparse
from collections.abc import Generator
from contextlib import closing
from dataclasses import asdict
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
from sklearn import config_context
from threadpoolctl import threadpool_limits
import torch

from .data_loader import ID, TARGET, TIME
from .evaluate import evaluate_scores, select_threshold
from .lab_benchmark import (BENCHMARK_MODELS, DENSE_MODELS, EVAL_LIMIT, IMBALANCE, KNN_BATCH, TRAIN_LIMIT,
                            ScratchSettings, build_models, classifier_scores, fit_network, fitted_parameters, stratified_positions)
from .phase3 import _write_table
from .preprocessing import FraudPreprocessor
from .scratch_checks import TorchReference, verify_scratch
from .scratch_neural import ScratchNetwork, positive_weight
from .train import SEED, THREADS, _write_json, _write_partition_outputs
from .training_data import complete_run, load_partition, read_phase1_manifest

NETWORKS = {"numpy_linear": 0, "numpy_relu32": 32, "torch_relu32_reference": 32}


def _sample(mode_dir, source, partition, limit):
    frame = load_partition(mode_dir, source, partition)
    return frame.iloc[stratified_positions(frame[TARGET].to_numpy(), limit)].reset_index(drop=True)


def _summary(frame, source_rows):
    return {"rows": len(frame), "source_rows": source_rows, "fraud_rows": int(frame[TARGET].sum()),
            "fraud_rate": float(frame[TARGET].mean()), "first_second": int(frame[TIME].iloc[0]),
            "last_second": int(frame[TIME].iloc[-1])}


def _dense(matrix):
    size = matrix.shape[0] * matrix.shape[1] * np.dtype(np.float32).itemsize
    if size > 64 * 2**20:
        raise ValueError(f"A dense benchmark matrix would need {size / 2**20:.1f} MiB. Reduce the sample cap.")
    return matrix.toarray().astype(np.float32, copy=False), size


def _predict(name, model, sparse, dense, threshold):
    if name in NETWORKS:
        scores = model.predict_scores(dense)
        if scores.shape != (len(dense),) or not np.isfinite(scores).all():
            raise ValueError(f"{name} returned invalid scores.")
        decisions = (scores >= threshold).astype(np.uint8) if threshold is not None else None
    else:
        matrix = dense if name in DENSE_MODELS else sparse
        scores = classifier_scores(name, model, matrix)
        decisions = model.predict(matrix) if name == "dummy" else None
        if threshold is not None:
            decisions = (scores >= threshold).astype(np.uint8)
    return scores, decisions


def _validation_result(name, model, sparse, dense, labels):
    start = perf_counter()
    scores, decisions = _predict(name, model, sparse, dense, None)
    elapsed = perf_counter() - start
    threshold = None if name == "dummy" else select_threshold(labels, scores)
    if threshold is not None:
        decisions = (scores >= threshold).astype(np.uint8)
    return threshold, elapsed, (scores, decisions)


def _evaluate(label, outputs):
    metrics, curves = {}, {}
    for name, (scores, decisions) in outputs.items():
        metrics[name], curves[name] = evaluate_scores(label, scores, decisions)
    return metrics, curves


def prepare_phase6(artifact_root: Path, mode: str, threads: int = THREADS,
                   settings: ScratchSettings = ScratchSettings()) -> Generator[Path, None, Path]:
    """Fit the matched reduced-data benchmarks and pause before sampling test."""
    started = perf_counter()
    checks = verify_scratch()
    print("Both NumPy architectures passed finite differences, PyTorch comparison, and fixture learning.", flush=True)
    torch.set_num_threads(threads)
    mode_dir = artifact_root / mode
    source, digest = read_phase1_manifest(mode_dir)
    phase_dir = mode_dir / "phase6"
    runs = phase_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    run_dir = runs / uuid4().hex
    processed = source["processed_rows"]
    scope = {"mode": mode, "processed_rows": processed, "seed": SEED}
    files = {"verification": "gradient_verification.json", "selection": "selection.json",
             "history": "network_history.json", "reference_comparison": "reference_comparison.json",
             "preprocessor": "preprocessor.joblib"}
    for name in ("train", "validation", "test"):
        files[f"{name}_sample"] = f"{name}_sample.parquet"
    for name in ("validation", "test"):
        for kind, ext in (("metrics", "json"), ("curves", "json"), ("predictions", "parquet")):
            files[f"{name}_{kind}"] = f"{name}_{kind}.{ext}"

    with TemporaryDirectory(prefix=".phase6-", dir=phase_dir) as temp, threadpool_limits(limits=threads), config_context(working_memory=64):
        stage = Path(temp) / "run"
        stage.mkdir()
        _write_json(stage / files["verification"], {**scope, **checks})
        train = _sample(mode_dir, source, "train", TRAIN_LIMIT)
        validation = _sample(mode_dir, source, "validation", EVAL_LIMIT)
        labels, val_labels = train[TARGET].to_numpy(), validation[TARGET].to_numpy()
        weight = positive_weight(labels)
        preprocess_start = perf_counter()
        preprocessor = FraudPreprocessor().fit(train)
        train_sparse, val_sparse = preprocessor.transform(train), preprocessor.transform(validation)
        train_dense, train_bytes = _dense(train_sparse)
        val_dense, val_bytes = _dense(val_sparse)
        preprocess_seconds = perf_counter() - preprocess_start
        joblib.dump(preprocessor, stage / files["preprocessor"])
        summaries = {name: _summary(frame, source["partitions"][name]["rows"])
                     for name, frame in (("train", train), ("validation", validation))}
        for name, frame in (("train", train), ("validation", validation)):
            _write_table(stage / files[f"{name}_sample"], frame[[ID, TIME, TARGET]], mode, f"{name}_sample", processed)

        models, metadata, validation_outputs, histories = {}, {}, {}, {}
        for name, model in build_models(labels, threads, settings.l2).items():
            mark = perf_counter()
            fit_matrix = train_dense if name in DENSE_MODELS else train_sparse
            fit_params = {"sample_weight": np.where(labels == 1, weight, 1.)} if name in ("adaboost", "sklearn_objective_reference") else {}
            model.fit(fit_matrix, labels, **fit_params)
            fit_seconds = perf_counter() - mark
            threshold, pred_seconds, output = _validation_result(name, model, val_sparse, val_dense, val_labels)
            path = f"{name}.joblib"
            joblib.dump(model, stage / path, compress=3)
            metadata[name] = {"fit_seconds": fit_seconds, "threshold": threshold, "validation_inference_seconds": pred_seconds,
                              "class": type(model).__name__, "configuration": fitted_parameters(model),
                              "imbalance": IMBALANCE[name], "file": path,
                              "score_domain": "raw margin" if name == "linear_svm" else "positive-class score"}
            models[name], validation_outputs[name] = model, output
            print(f"{name}: fit {fit_seconds:.2f}s, validation inference {pred_seconds:.2f}s", flush=True)

        train_float = labels.astype(np.float32)
        val_float = val_labels.astype(np.float32)
        for name, width in NETWORKS.items():
            model = ScratchNetwork(train_dense.shape[1], width, seed=settings.seed)
            if name == "torch_relu32_reference":
                model = TorchReference(model)
            history = fit_network(model, train_dense, train_float, val_dense, val_float, settings, name)
            threshold, pred_seconds, output = _validation_result(name, model, val_sparse, val_dense, val_labels)
            path = f"{name}.npz"
            np.savez(stage / path, **model.snapshot())
            histories[name] = history
            metadata[name] = {"fit_seconds": history["fit_seconds"], "threshold": threshold,
                              "validation_inference_seconds": pred_seconds, "file": path,
                              "configuration": {"input_dim": train_dense.shape[1], "hidden_width": width, **asdict(settings)},
                              "score_domain": "sigmoid score", "best_epoch": history["best_epoch"],
                              "parameter_count": history["parameter_count"], "engine": "PyTorch" if isinstance(model, TorchReference) else "NumPy"}
            models[name], validation_outputs[name] = model, output
        _write_json(stage / files["history"], {**scope, "models": histories})

        validation_metrics, validation_curves = _evaluate(val_labels, validation_outputs)
        _write_partition_outputs(stage, mode, processed, "validation", validation, validation_metrics, validation_curves, validation_outputs)
        thresholds = {name: info["threshold"] for name, info in metadata.items()}
        frozen = {**scope, "selected_on": "validation", "thresholds": thresholds,
                  "threshold_rule": "Maximum validation F1, highest threshold on ties, score >= threshold; dummy uses its majority rule.",
                  "checkpoint_rule": "Maximum validation AP, earliest tie",
                  "best_epochs": {name: value["best_epoch"] for name, value in histories.items()},
                  "configurations": {name: info["configuration"] for name, info in metadata.items()},
                  "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
        _write_json(stage / files["selection"], frozen)
        train_dim = train_sparse.shape[1]
        del train_dense, val_dense, train_sparse, val_sparse, train_float, val_float, validation_curves
        wait_started = perf_counter()
        yield stage
        started += perf_counter() - wait_started
        print("Validation checkpoint and threshold selection frozen. Opening test sample.", flush=True)

        test = _sample(mode_dir, source, "test", EVAL_LIMIT)
        if test[ID].isin(train[ID]).any() or test[ID].isin(validation[ID]).any() or validation[ID].isin(train[ID]).any():
            raise ValueError("Chronological benchmark samples must have disjoint transaction IDs.")
        summaries["test"] = _summary(test, source["partitions"]["test"]["rows"])
        _write_table(stage / files["test_sample"], test[[ID, TIME, TARGET]], mode, "test_sample", processed)
        test_sparse = preprocessor.transform(test)
        test_dense, test_bytes = _dense(test_sparse)
        outputs = {}
        for name, model in models.items():
            mark = perf_counter()
            outputs[name] = _predict(name, model, test_sparse, test_dense, thresholds[name])
            metadata[name]["test_inference_seconds"] = perf_counter() - mark
        test_metrics, test_curves = _evaluate(test[TARGET].to_numpy(), outputs)
        _write_partition_outputs(stage, mode, processed, "test", test, test_metrics, test_curves, outputs)

        comparisons = []
        for numpy_name, reference in (("numpy_linear", "sklearn_objective_reference"), ("numpy_relu32", "torch_relu32_reference")):
            record = {"numpy": numpy_name, "reference": reference, "partitions": {}}
            for name, observed, metrics in (("validation", validation_outputs, validation_metrics), ("test", outputs, test_metrics)):
                error = observed[numpy_name][0] - observed[reference][0]
                record["partitions"][name] = {
                    "maximum_absolute_score_difference": float(np.max(np.abs(error))),
                    "mean_absolute_score_difference": float(np.mean(np.abs(error))),
                    "numpy_ap": metrics[numpy_name]["average_precision"], "reference_ap": metrics[reference]["average_precision"],
                }
            comparisons.append(record)
        _write_json(stage / files["reference_comparison"], {**scope, "pairs": comparisons})
        manifest = {**scope, "phase": 6, "schema_version": 1, "threads": threads, "partitions": source["partitions"],
                    "phase1_manifest_sha256": digest, "source_files": {name: source["files"][name] for name in summaries},
                    "test_scope": "matched quick-test development subset" if mode == "quick" else "matched chronological test subset",
                    "samples": summaries, "sampling": "Stratified without replacement in each chronological split; restored source order; seed 42.",
                    "training_limit": TRAIN_LIMIT, "evaluation_limit": EVAL_LIMIT, "knn_inference_batch": KNN_BATCH,
                    "features": {"raw": preprocessor.feature_names_in_.tolist(), "transformed": preprocessor.get_feature_names_out().tolist(),
                                 "raw_dimension": int(preprocessor.n_features_in_), "transformed_dimension": train_dim,
                                 "dense_bytes": {"train": train_bytes, "validation": val_bytes, "test": test_bytes},
                                 "training_and_validation_preprocess_seconds": preprocess_seconds},
                    "benchmark_models": list(BENCHMARK_MODELS), "models": metadata, "positive_weight": weight,
                    "network_training": asdict(settings), "run_directory": run_dir.relative_to(phase_dir).as_posix(), "files": files,
                    "versions": {"python": platform.python_version(), **{name: version(name) for name in
                                 ("numpy", "pandas", "scikit-learn", "xgboost", "torch", "joblib", "pyarrow")}},
                    "runtime_seconds": perf_counter() - started}
        _write_json(stage / "manifest.json", manifest)
        stage.rename(run_dir)
        latest = Path(temp) / "manifest.json"
        _write_json(latest, manifest)
        latest.replace(phase_dir / "manifest.json")
    print(f"Phase 6 {mode}: {manifest['runtime_seconds']:.2f}s. Manifest: {(phase_dir / 'manifest.json').resolve()}", flush=True)
    return phase_dir / "manifest.json"


def run_phase6(artifact_root: Path, mode: str, threads: int = THREADS,
               settings: ScratchSettings = ScratchSettings()) -> Path:
    with closing(prepare_phase6(artifact_root, mode, threads, settings)) as run:
        next(run)
        return complete_run(run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full", "smoke"), default="quick")
    parser.add_argument("--threads", type=int, choices=range(1, 9), metavar="1-8", default=THREADS)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--verify-only", action="store_true", help="Run the small mathematical checks without reading data.")
    args = parser.parse_args()
    try:
        if args.verify_only:
            print(json.dumps(verify_scratch(), indent=2))
        else:
            run_phase6(args.artifact_root, args.mode, args.threads)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Phase 6 error: {error}\n")


if __name__ == "__main__":
    main()
