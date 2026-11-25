"""Train the Phase 2B MLP alongside the existing classical benchmark."""

import argparse
from collections.abc import Generator
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from .data_loader import ID, TARGET, TIME
from .evaluate import evaluate_scores, select_threshold
from .neural import (DROPOUT, HIDDEN_WIDTHS, FraudMLP, TrainingConfig, fit_model, load_checkpoint,
                     make_loader, score_batches)
from .train import THREADS, _write_json, _write_partition_outputs
from .training_data import PARTITIONS, complete_run, load_partition, read_phase1_manifest


def load_classical_preprocessor(mode_dir: Path, source: dict, source_hash: str,
                                draft: tuple[Path, dict] | None = None) -> tuple:
    """Reuse the fitted behavioral encoder for the exact same Phase 1 snapshot."""
    phase_dir = mode_dir / "phase2"
    if draft is None:
        manifest_path = phase_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing {manifest_path}. Run python -m src.train --mode {mode_dir.name} first.")
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        run_dir = phase_dir / manifest["run_directory"]
        digest = sha256(manifest_bytes).hexdigest()
    else:
        run_dir, manifest = draft
        digest = None
    if (manifest.get("phase"), manifest.get("schema_version"), manifest.get("mode")) != (2, 1, mode_dir.name):
        raise ValueError("An existing Phase 2 manifest for the same mode is required.")
    if (manifest["phase1_manifest_sha256"] != source_hash
            or manifest["partitions"] != source["partitions"]
            or manifest["phase1_partition_files"] != {name: source["files"][name] for name in PARTITIONS}):
        raise ValueError("The Phase 2 preprocessor was fitted against a different Phase 1 snapshot.")
    for package in ("scikit-learn", "numpy", "scipy", "pandas", "joblib"):
        if version(package) != manifest["versions"][package]:
            raise ValueError(f"The fitted Phase 2 preprocessor requires {package}=={manifest['versions'][package]}.")

    run_dir = run_dir.resolve()
    path = (run_dir / manifest["files"]["preprocessors"]["behavioral"]).resolve()
    if not path.is_relative_to(phase_dir.resolve()) or not path.is_file():
        raise FileNotFoundError(f"Missing the trusted local Phase 2 behavioral preprocessor: {path}")
    preprocessor = joblib.load(path)
    expected = manifest["features"]["behavioral"]
    names = preprocessor.get_feature_names_out().tolist()
    if (list(preprocessor.feature_names_in_) != expected["raw_features"]
            or names != expected["transformed_features"] or len(names) != expected["transformed_dimension"]):
        raise ValueError("The fitted preprocessor's feature order or dimension differs from its Phase 2 manifest.")
    reference = {"path": path.relative_to(mode_dir.resolve()).as_posix(),
                 "sha256": sha256(path.read_bytes()).hexdigest(), "feature_set": "behavioral",
                 "classical_run": run_dir.relative_to(mode_dir.resolve()).as_posix(),
                 "classical_manifest_sha256": digest,
                 "classical_champion": manifest["champion"],
                 "raw_features": expected["raw_features"], "transformed_features": names}
    return preprocessor, reference, run_dir, manifest


def _classical_metrics(run_dir: Path, manifest: dict, name: str, frame: pd.DataFrame) -> dict:
    """Verify row order against the existing predictions before comparing model metrics."""
    fields = [ID, TIME, TARGET]
    classical = pd.read_parquet(run_dir / manifest["files"][f"{name}_predictions"], columns=fields)
    if any(not np.array_equal(frame[field], classical[field]) for field in fields):
        raise ValueError(f"The neural and classical {name} rows do not align.")
    report = json.loads((run_dir / manifest["files"][f"{name}_metrics"]).read_bytes())
    if (report["mode"], report["partition"], report["rows"]) != (manifest["mode"], name, len(frame)):
        raise ValueError(f"The Phase 2 metrics metadata does not match the {name} rows.")
    return report["models"]


def _save_evaluation(stage: Path, source: dict, name: str, frame: pd.DataFrame, scores: np.ndarray,
                     threshold: float) -> dict:
    decisions = (scores >= threshold).astype(np.uint8)
    metrics, curves = evaluate_scores(frame[TARGET].to_numpy(), scores, decisions)
    _write_partition_outputs(stage, source["mode"], source["processed_rows"], name, frame,
                             {"mlp": metrics}, {"mlp": curves}, {"mlp": (scores, decisions)})
    return metrics


def prepare_training(artifact_root: Path, mode: str, threads: int = THREADS,
                     config: TrainingConfig = TrainingConfig(), *,
                     draft: tuple[Path, dict] | None = None) -> Generator[Path, None, Path]:
    """Select the neural checkpoint and threshold using validation, then pause."""
    started = perf_counter()
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    mode_dir = artifact_root / mode
    source, source_hash = read_phase1_manifest(mode_dir)
    preprocessor, reference, classical_dir, classical_manifest = load_classical_preprocessor(mode_dir, source, source_hash, draft)
    train = load_partition(mode_dir, source, "train")
    validation = load_partition(mode_dir, source, "validation")
    if validation[ID].isin(train[ID]).any():
        raise ValueError("Train and validation share transaction IDs.")
    input_dim = len(reference["transformed_features"])
    memory = {"dense_split_estimate_bytes": {name: part["rows"] * input_dim * 4
                                             for name, part in source["partitions"].items()},
              "maximum_dense_input_batch_bytes": config.batch_size * input_dim * 4}
    estimate = memory["dense_split_estimate_bytes"]["train"] / 2**20
    batch_size_mib = memory["maximum_dense_input_batch_bytes"] / 2**20
    print(f"{mode}: {len(train):,} train and {len(validation):,} validation rows, {input_dim} features, "
          f"{threads} CPU threads. Dense train estimate {estimate:.2f} MiB, "
          f"dense input per batch at most {batch_size_mib:.2f} MiB.", flush=True)

    phase_dir = mode_dir / "phase2b"
    runs = phase_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    run_dir = runs / uuid4().hex
    with TemporaryDirectory(prefix=".phase2b-", dir=phase_dir) as temp, threadpool_limits(limits=threads):
        stage = Path(temp) / "run"
        stage.mkdir()
        train_batches = make_loader(preprocessor.transform(train), train[TARGET].to_numpy(),
                                    config.batch_size, training=True, seed=config.seed)
        validation_batches = make_loader(preprocessor.transform(validation), validation[TARGET].to_numpy(),
                                         config.batch_size, training=False, seed=config.seed)
        model = FraudMLP(input_dim)
        training = fit_model(model, train_batches, validation_batches, config)
        _write_json(stage / "training_history.json", {"mode": mode, "processed_rows": source["processed_rows"],
                                                      "train_rows": len(train), "validation_rows": len(validation),
                                                      **training})

        architecture = {"name": "FraudMLP", "input_dim": input_dim, "hidden_widths": list(HIDDEN_WIDTHS),
                        "hidden_order": ["Linear", "BatchNorm1d", "ReLU", "Dropout"], "dropout": DROPOUT,
                        "output": "one raw logit per row", "trainable_parameters":
                        sum(p.numel() for p in model.parameters() if p.requires_grad)}
        checkpoint_path = stage / "best_model.pt"
        torch.save(model.state_dict(), checkpoint_path)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(training["positive_class_weight"], dtype=torch.float32))
        inference_started = perf_counter()
        _, val_scores = score_batches(model, validation_batches, criterion)
        validation_inference = perf_counter() - inference_started
        threshold = select_threshold(validation[TARGET].to_numpy(), val_scores)
        val_metrics = _save_evaluation(stage, source, "validation", validation, val_scores, threshold)
        val_classical = _classical_metrics(classical_dir, classical_manifest, "validation", validation)
        selection = {"mode": mode, "processed_rows": source["processed_rows"], "selected_on": "validation",
                     "best_epoch": training["best_epoch"], "checkpoint_rule": "maximum validation AP, earliest tie",
                     "best_validation_ap": training["best_validation_ap"], "threshold": threshold,
                     "threshold_rule": "maximum validation F1, highest threshold on ties, score >= threshold",
                     "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
        _write_json(stage / "selection.json", selection)
        prior_ids = pd.concat([train[ID], validation[ID]], ignore_index=True)
        del train, validation, train_batches, validation_batches, model
        wait_started = perf_counter()
        yield stage
        started += perf_counter() - wait_started
        if draft is not None:
            _, final_reference, classical_dir, classical_manifest = load_classical_preprocessor(mode_dir, source, source_hash)
            if final_reference["sha256"] != reference["sha256"]:
                raise ValueError("The finalized behavioral preprocessor differs from the fitted MLP input.")
            reference = final_reference
        print(f"Frozen checkpoint epoch {training['best_epoch']} and threshold {threshold:.8f}. Opening {mode} test.", flush=True)
        test = load_partition(mode_dir, source, "test")
        if test[ID].isin(prior_ids).any():
            raise ValueError("Test shares transaction IDs with train or validation.")
        test_batches = make_loader(preprocessor.transform(test), test[TARGET].to_numpy(),
                                   config.batch_size, training=False, seed=config.seed)
        model = load_checkpoint(checkpoint_path, input_dim)
        inference_started = perf_counter()
        _, test_scores = score_batches(model, test_batches, criterion)
        test_inference = perf_counter() - inference_started
        test_metrics = _save_evaluation(stage, source, "test", test, test_scores, threshold)
        test_classical = _classical_metrics(classical_dir, classical_manifest, "test", test)
        test_scope = "full chronological holdout" if mode == "full" else "development diagnostics"
        _write_json(stage / "comparison.json", {"mode": mode, "processed_rows": source["processed_rows"],
                                                "test_scope": test_scope, "classical_champion": reference["classical_champion"],
                                                "validation": {**val_classical, "mlp": val_metrics},
                                                "test": {**test_classical, "mlp": test_metrics}})

        manifest = {
            "phase": "2B", "schema_version": 1, "mode": mode, "processed_rows": source["processed_rows"],
            "partitions": source["partitions"], "test_scope": test_scope,
            "phase1_manifest_sha256": source_hash, "preprocessor": reference, "architecture": architecture,
            "training_config": {**asdict(config), "optimizer": "Adam", "loss": "weighted BCEWithLogitsLoss mean",
                                "positive_class_weight": training["positive_class_weight"]},
            "device": "cpu", "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "cpu": platform.processor(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "versions": {"python": platform.python_version(), **{name: version(name) for name in
                         ("torch", "numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "joblib", "threadpoolctl")}},
            "memory": memory, "selection": selection,
            "validation_inference_seconds": validation_inference, "test_inference_seconds": test_inference,
            "training_summary": {key: value for key, value in training.items() if key != "epochs"},
            "run_directory": run_dir.relative_to(phase_dir).as_posix(),
            "files": {"checkpoint": "best_model.pt", "history": "training_history.json",
                      "selection": "selection.json", "comparison": "comparison.json",
                      **{f"{name}_{kind}": f"{name}_{kind}.{ext}" for name in ("validation", "test")
                         for kind, ext in (("metrics", "json"), ("curves", "json"), ("predictions", "parquet"))}},
            "completed_stages": ["transform", "train", "restore", "validate", "freeze", "test", "compare"],
            "runtime_seconds": perf_counter() - started,
        }
        _write_json(stage / "manifest.json", manifest)
        stage.rename(run_dir)
        latest = Path(temp) / "manifest.json"
        _write_json(latest, manifest)
        latest.replace(phase_dir / "manifest.json")
    print(f"Phase 2B {mode}: {manifest['runtime_seconds']:.2f}s, "
          f"validation AP {val_metrics['average_precision']:.6f}, test AP {test_metrics['average_precision']:.6f}. "
          f"Manifest: {(phase_dir / 'manifest.json').resolve()}", flush=True)
    return phase_dir / "manifest.json"


def run_training(artifact_root: Path, mode: str, threads: int = THREADS,
                 config: TrainingConfig = TrainingConfig()) -> Path:
    with closing(prepare_training(artifact_root, mode, threads, config)) as run:
        next(run)
        return complete_run(run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full", "smoke"), default="quick")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=THREADS, metavar="1-8")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    torch.set_num_interop_threads(1)
    try:
        run_training(args.artifact_root, args.mode, args.threads)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Phase 2B error: {error}\n")


if __name__ == "__main__":
    main()
