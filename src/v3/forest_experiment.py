"""Fit the archived V2 Random Forest recipe on each full V3 training prefix."""

import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from src.evaluate import positive_scores
from src.lab_benchmark import stratified_positions
from src.v2.challengers import SUBSET, TRAIN_LIMIT, _estimator
from src.v2.data import sha256, write_json
from src.v2.experiment import _peak_memory, review_budgets

from .diagnostics import annotated_metrics, development_inputs
from .logistic_experiment import BUDGET_SECONDS, CUTOFF, MEMORY_LIMIT_BYTES, METRICS, THREADS, worker_peak
from .logistic_features import ChunkedDenseFeatures, ROW_BATCH


NAME = "random_forest_depth"
SCORE_BATCH = 2048
SCORE_ATOL = 2e-15


def training_matrix(transform, training):
    """Build one C-order float32 training matrix, reusing the V2 transform."""
    rows = len(training)
    columns = len(transform.selected)
    if rows * columns * np.dtype(np.float32).itemsize > 400_000_000:
        raise MemoryError("The transformed forest input exceeds its separate allocation budget.")
    result = np.empty((rows, columns), dtype=np.float32)
    for lo in range(0, rows, ROW_BATCH):
        hi = min(lo + ROW_BATCH, rows)
        result[lo:hi] = transform.transform(training[lo:hi])
    return result


def score_partition(model, transform, matrix):
    return np.concatenate([positive_scores(model, transform.transform(matrix[lo:lo + SCORE_BATCH]))
                           for lo in range(0, len(matrix), SCORE_BATCH)])


def check_control(root, advanced_hash, training, labels, validation, truth, fold, index, categorical):
    candidate = SUBSET[NAME]
    folder = root / "challengers" / NAME
    model_path = folder / f"model{index}.joblib"
    report = json.loads((folder / f"fold{index}.json").read_bytes())
    if (report["candidate"] != asdict(candidate) or report["threads"] != THREADS or report["fold"] != index
            or report["feature_sha256"] != advanced_hash or report["fit_rows"] != TRAIN_LIMIT
            or (report["train_end"], report["validation_end"]) != (fold.train_end, fold.validation_end)
            or report["threshold"] != CUTOFF):
        raise ValueError(f"Saved sampled control has incompatible provenance in fold {index}.")
    saved = np.load(folder / f"scores{index}.npy", allow_pickle=False)
    checkpoint = joblib.load(model_path)
    if saved.shape != truth.shape or not np.isfinite(saved).all() or checkpoint["train_rows"] != TRAIN_LIMIT:
        raise ValueError(f"Invalid sampled score or model population in fold {index}.")
    old = checkpoint["transform"]
    replay = score_partition(checkpoint["model"], old, validation)
    np.testing.assert_allclose(replay, saved, rtol=0, atol=SCORE_ATOL)
    scores = annotated_metrics(truth, saved, CUTOFF)
    for name in METRICS:
        np.testing.assert_allclose(scores[name], report["metrics"][name], atol=1e-12, rtol=0)

    positions = stratified_positions(labels, TRAIN_LIMIT)
    chunked = ChunkedDenseFeatures(candidate.top_k, categorical, False).fit(training[positions], labels[positions])
    np.testing.assert_array_equal(chunked.selected, old.selected)
    np.testing.assert_array_equal(chunked.imputer.statistics_, old.imputer.statistics_)
    original_probe = old.transform(validation[:SCORE_BATCH])
    adapted_probe = chunked.transform(validation[:SCORE_BATCH])
    np.testing.assert_allclose(adapted_probe, original_probe, rtol=0, atol=1e-6)
    return saved, scores, checkpoint["model"].get_params(), {
        "fit_rows": report["fit_rows"], "fit_seconds": report["fit_seconds"],
        "inference_seconds": report["inference_seconds"], "process_peak_working_set": report["process_peak_working_set"],
        "checkpoint_bytes": model_path.stat().st_size,
        "max_replayed_score_difference": float(np.max(np.abs(replay - saved))),
        "preprocessing_probe_max_difference": float(np.max(np.abs(adapted_probe - original_probe))),
    }


def run_fold(root, out, index):
    started = perf_counter()
    matrices, _, labels, folds = development_inputs(root)
    cache, advanced, _, matrix = matrices
    fold = folds[index - 1]
    start, stop = fold.train_end, fold.validation_end
    names = advanced["features"]
    if (start != (212_594, 283_459, 354_324)[index - 1] or len(names) != 497 or matrix.dtype != np.float32):
        raise ValueError("The run requires the verified V2 folds and its 497 cached float32 features.")
    training, y = matrix[:start], labels[:start]
    validation, truth = matrix[start:stop], labels[start:stop]
    categorical = [i for i, name in enumerate(names) if name in cache["categorical"]]
    folder = out / f"fold{index}"
    folder.mkdir(exist_ok=True)
    score_path = folder / "full_scores.npy"
    model_path = folder / "forest_full.joblib"
    record_path = folder / "result.json"
    provenance = {"fold": index, "train_end": start, "validation_end": stop,
                  "candidate": asdict(SUBSET[NAME]), "cutoff": CUTOFF, "threads": THREADS,
                  "input_digests": {name: sha256(root / name) for name in ("advanced.json", "development_metadata.parquet")},
                  "implementation_digests": {"forest_experiment.py": sha256(Path(__file__)),
                                              "logistic_features.py": sha256(Path(__file__).with_name("logistic_features.py"))},
                  "control_scores_sha256": sha256(root / "challengers" / NAME / f"scores{index}.npy")}
    if record_path.is_file():
        existing = json.loads(record_path.read_bytes())
        if (any(existing[key] != value for key, value in provenance.items())
                or sha256(score_path) != existing["full_scores_sha256"] or sha256(model_path) != existing["full_model_sha256"]):
            raise ValueError(f"Fold {index} checkpoint differs from its provenance.")
        print(f"Fold {index}: verified and reused the existing checkpoint.", flush=True)
        return

    print(f"Fold {index}: {len(training):,} training rows, {len(validation):,} validation rows.", flush=True)
    # The forest already parallelizes by tree. Limit native pools to avoid nested parallelism.
    with threadpool_limits(limits=1):
        control, sampled_metrics, archived_settings, archive = check_control(
            root, provenance["input_digests"]["advanced.json"], training, y, validation, truth, fold, index, categorical)
        candidate = SUBSET[NAME]
        model = _estimator(candidate, THREADS)
        if model.get_params() != archived_settings or archived_settings["n_estimators"] != 140:
            raise ValueError("Full-data Random Forest must use the archived classifier settings.")
        prep_start = perf_counter()
        transform = ChunkedDenseFeatures(candidate.top_k, categorical, False).fit(training, y)
        X = training_matrix(transform, training)
        prep_time = perf_counter() - prep_start
        input_bytes = X.nbytes
        print(f"Fold {index}: preprocessing {prep_time:.1f}s. Fitting 140 trees with six workers.", flush=True)
        fit_start = perf_counter()
        model.fit(X, y)
        fit_time = perf_counter() - fit_start
        if len(model.estimators_) != 140:
            raise ValueError("The full-data forest has an unexpected tree count.")
        tree_nodes = sum(tree.tree_.node_count for tree in model.estimators_)
        del X

        model_temp = folder / "forest_full.tmp.joblib"
        joblib.dump({"model": model, "transform": transform, "train_rows": start}, model_temp, compress=1)
        model_temp.replace(model_path)
        score_start = perf_counter()
        prediction = score_partition(model, transform, validation)
        scoring = perf_counter() - score_start
        del model, transform
        checkpoint = joblib.load(model_path)
        probe = score_partition(checkpoint["model"], checkpoint["transform"], validation[:SCORE_BATCH])
        np.testing.assert_allclose(probe, prediction[:len(probe)], rtol=0, atol=SCORE_ATOL)
        del checkpoint

    peak = _peak_memory()
    if peak is not None and peak >= MEMORY_LIMIT_BYTES:
        raise MemoryError("The worker exceeded the 7.5 GB memory guard.")
    score_temp = folder / "full_scores.tmp.npy"
    np.save(score_temp, prediction, allow_pickle=False)
    score_temp.replace(score_path)
    new_metrics = annotated_metrics(truth, prediction, CUTOFF)
    summary = {**provenance, "fit_rows": start, "validation_rows": stop - start, "fit_fraud_rows": int(y.sum()),
               "sampled": sampled_metrics, "full": new_metrics, "archived": archive,
               "review_budgets": {"sampled": review_budgets(truth, control), "full": review_budgets(truth, prediction)},
               "settings": archived_settings, "preprocessing_seconds": prep_time, "forest_fit_seconds": fit_time,
               "training_seconds": prep_time + fit_time, "inference_seconds": scoring,
               "worker_seconds": perf_counter() - started, "peak_working_set_bytes": peak,
               "input_matrix_bytes": input_bytes, "tree_nodes": tree_nodes, "checkpoint_bytes": model_path.stat().st_size,
               "full_scores_sha256": sha256(score_path), "full_model_sha256": sha256(model_path)}
    write_json(record_path, summary)
    print(f"Fold {index}: AP {sampled_metrics['average_precision']:.5f} to {new_metrics['average_precision']:.5f}, "
          f"forest fit {fit_time:.1f}s. Checkpoint saved.", flush=True)


def supervise(root, out, index, deadline):
    worker = mp.get_context("spawn").Process(target=run_fold, args=(root, out, index))
    worker.start()
    try:
        while worker.is_alive():
            remaining = deadline - perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"Fold {index} reached the 30-minute run budget.")
            if worker_peak(worker.pid) >= MEMORY_LIMIT_BYTES:
                raise MemoryError(f"Fold {index} reached the 7.5 GB working-set guard.")
            worker.join(timeout=min(remaining, 1.0))
        if worker.exitcode != 0:
            raise RuntimeError(f"Fold {index} worker exited with status {worker.exitcode}.")
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join()
        worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.development.resolve(), args.output.resolve()
    if out.is_relative_to(root) or "retrospective" in {part.lower() for part in root.parts}:
        raise ValueError("Use V2 development inputs and an independent output directory.")
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    status = None
    done = []
    try:
        for index in (1, 2, 3):
            supervise(root, out, index, started + BUDGET_SECONDS)
            done.append(index)
    except (TimeoutError, MemoryError, RuntimeError) as error:
        status = str(error)
        raise
    finally:
        write_json(out / "run_status.json", {"verified_folds": done, "budget_seconds": BUDGET_SECONDS,
                                            "worker_memory_guard_bytes": MEMORY_LIMIT_BYTES,
                                            "wall_seconds": perf_counter() - started, "constraint": status})


if __name__ == "__main__":
    main()
