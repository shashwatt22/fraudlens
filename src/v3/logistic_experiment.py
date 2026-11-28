"""Run the three chronological V3 full-data Logistic Regression fits."""

import argparse
import ctypes
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.evaluate import positive_scores
from src.lab_benchmark import stratified_positions
from src.v2.challengers import SUBSET, TRAIN_LIMIT, _estimator
from src.v2.data import sha256, write_json
from src.v2.experiment import _peak_memory, review_budgets
from src.v2.final import _score
from .diagnostics import annotated_metrics, development_inputs
from .logistic_features import ChunkedDenseFeatures, solver_matrix


NAME = "logistic_l2"
THREADS = 6
CUTOFF = .5
BUDGET_SECONDS = 30 * 60
MEMORY_LIMIT_BYTES = 7_500_000_000
METRICS = ("average_precision", "roc_auc", "precision", "recall", "f1", "accuracy")


def check_control(root, matrices, times, labels, fold, index):
    directory = root / "challengers" / NAME
    record = json.loads((directory / f"fold{index}.json").read_bytes())
    start, stop = fold.train_end, fold.validation_end
    candidate = SUBSET[NAME]
    if (record["candidate"] != asdict(candidate) or record["threads"] != THREADS or record["fold"] != index
            or record["feature_sha256"] != sha256(root / "advanced.json") or record["fit_rows"] != TRAIN_LIMIT
            or record["train_end"] != start or record["validation_end"] != stop or record["threshold"] != CUTOFF):
        raise ValueError(f"Unexpected sampled control manifest for fold {index}.")
    model_path = directory / f"model{index}.joblib"
    expected = np.load(directory / f"scores{index}.npy", allow_pickle=False)
    actual = _score(NAME, model_path, matrices, start, stop, start, times, THREADS)
    if expected.shape != (stop - start,) or not np.isfinite(expected).all() or not np.array_equal(expected, actual):
        raise ValueError(f"Saved control scores could not be reproduced exactly in fold {index}.")
    metrics = annotated_metrics(labels[start:stop], expected, CUTOFF)
    for metric in METRICS:
        np.testing.assert_allclose(metrics[metric], record["metrics"][metric], atol=1e-12)
    return joblib.load(model_path), record, metrics, expected


def check_transform_parity(old, training, labels, categorical, test_rows):
    """Compare new fit mechanics to the archived transform on its original 16k sample."""
    positions = stratified_positions(labels, TRAIN_LIMIT)
    transform = ChunkedDenseFeatures(SUBSET[NAME].top_k, categorical, True).fit(training[positions], labels[positions])
    np.testing.assert_array_equal(transform.selected, old.selected)
    np.testing.assert_array_equal(transform.imputer.statistics_, old.imputer.statistics_)
    np.testing.assert_allclose(transform.scaler.mean_, old.scaler.mean_, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(transform.scaler.var_, old.scaler.var_, atol=1e-10, rtol=1e-10)
    expected, actual = old.transform(test_rows), transform.transform(test_rows)
    if actual.shape != expected.shape:
        raise ValueError("The chunked transform changed the representation width on the original sample.")
    error = float(np.max(np.abs((actual - expected).data), initial=0.0))
    if error > 1e-6:
        raise ValueError(f"Chunked transform differs from the saved sampled transform by {error:.3g}.")
    return {"rows": len(positions), "columns": actual.shape[1], "probe_rows": len(test_rows), "maximum_difference": error}


def run_fold(root, out, index):
    wall = perf_counter()
    matrices, times, labels, folds = development_inputs(root)
    cache, advanced, _, source = matrices
    fold = folds[index - 1]
    start, stop = fold.train_end, fold.validation_end
    if start != (212_594, 283_459, 354_324)[index - 1] or len(advanced["features"]) != 497:
        raise ValueError("Expected the verified V2 fold boundaries and original feature matrix.")
    categorical = [i for i, name in enumerate(advanced["features"]) if name in cache["categorical"]]
    directory = out / f"fold{index}"
    directory.mkdir(exist_ok=True)
    score_path = directory / "full_scores.npy"
    model_path = directory / "logistic_full.joblib"
    record_path = directory / "result.json"
    provenance = {"fold": index, "train_end": start, "validation_end": stop, "candidate": asdict(SUBSET[NAME]),
                  "cutoff": CUTOFF, "threads": THREADS,
                  "input_digests": {name: sha256(root / name) for name in ("advanced.json", "development_metadata.parquet")},
                  "implementation_digests": {name: sha256(Path(__file__).with_name(name))
                                             for name in ("logistic_features.py", "logistic_experiment.py")},
                  "control_scores_sha256": sha256(root / "challengers" / NAME / f"scores{index}.npy")}
    if record_path.is_file():
        saved = json.loads(record_path.read_bytes())
        if (any(saved[key] != value for key, value in provenance.items())
                or sha256(score_path) != saved["full_scores_sha256"] or sha256(model_path) != saved["full_model_sha256"]):
            raise ValueError(f"Fold {index} files differ from the completed checkpoint.")
        print(f"Fold {index}: verified existing checkpoint.", flush=True)
        return

    training, y = source[:start], labels[:start]
    print(f"Fold {index}: {start:,} training rows, {stop - start:,} validation rows.", flush=True)
    with threadpool_limits(limits=THREADS):
        archived, record, control_metrics, control = check_control(root, matrices, times, labels, fold, index)
        model = _estimator(SUBSET[NAME], THREADS)
        if model.get_params() != archived["model"].get_params():
            raise ValueError("Full and sampled Logistic Regression settings differ.")
        parity = check_transform_parity(archived["transform"], training, y, categorical, source[start:start + 2048])
        prep_start = perf_counter()
        transform = ChunkedDenseFeatures(SUBSET[NAME].top_k, categorical, True).fit(training, y)
        prep_time = perf_counter() - prep_start
        csr_start = perf_counter()
        X = solver_matrix(transform, training)
        build_time = perf_counter() - csr_start
        matrix_shape, matrix_nnz = X.shape, X.nnz
        matrix_bytes = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
        print(f"Fold {index}: preprocessing {prep_time + build_time:.1f}s, CSR {matrix_bytes / (1 << 30):.2f} GiB. "
              "Fitting L-BFGS.", flush=True)
        solver_start = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, y)
        solver_time = perf_counter() - solver_start
        peak = _peak_memory()
        del X
        messages = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
        iterations = int(model.n_iter_.max())
        converged = not messages and iterations < model.max_iter
        elapsed = prep_time + build_time + solver_time
        if peak is not None and peak >= MEMORY_LIMIT_BYTES:
            raise MemoryError("The worker exceeded its peak working set budget.")
        temporary_model = directory / "logistic_full.tmp.joblib"
        joblib.dump({"model": model, "transform": transform, "train_rows": start}, temporary_model, compress=1)
        temporary_model.replace(model_path)
        reloaded = joblib.load(model_path)
        inference_start = perf_counter()
        scores = np.concatenate([positive_scores(model, transform.transform(source[lo:min(lo + 2048, stop)]))
                                 for lo in range(start, stop, 2048)])
        inference = perf_counter() - inference_start
        probe = reloaded["transform"].transform(source[start:start + 2048])
        np.testing.assert_array_equal(positive_scores(reloaded["model"], probe), scores[:probe.shape[0]])

    temporary_scores = directory / "full_scores.tmp.npy"
    np.save(temporary_scores, scores, allow_pickle=False)
    temporary_scores.replace(score_path)
    full_metrics = annotated_metrics(labels[start:stop], scores, CUTOFF)
    result = {**provenance, "fit_rows": start, "fit_fraud_rows": int(y.sum()), "validation_rows": stop - start,
              "sampled": control_metrics, "full": full_metrics,
              "review_budgets": {"sampled": review_budgets(labels[start:stop], control),
                                 "full": review_budgets(labels[start:stop], scores)},
              "settings": model.get_params(), "old_preprocessing_parity": parity, "converged": converged,
              "iterations": iterations, "convergence_warnings": messages,
              "sampled_iterations": int(archived["model"].n_iter_.max()), "sampled_fit_rows": record["fit_rows"],
              "sampled_archived_training_seconds": record["fit_seconds"],
              "sampled_archived_peak_bytes": record["process_peak_working_set"],
              "prep_fit_seconds": prep_time, "matrix_build_seconds": build_time, "solver_seconds": solver_time,
              "training_seconds": elapsed, "inference_seconds": inference, "worker_seconds": perf_counter() - wall,
              "peak_working_set_bytes": _peak_memory(), "csr_bytes": matrix_bytes,
              "csr_nnz": matrix_nnz, "csr_shape": matrix_shape,
              "full_scores_sha256": sha256(score_path), "full_model_sha256": sha256(model_path)}
    write_json(record_path, result)
    print(f"Fold {index}: sampled AP {control_metrics['average_precision']:.5f}, full AP {full_metrics['average_precision']:.5f}, "
          f"{iterations} solver iterations, converged={converged}, checkpoint saved.", flush=True)


def worker_peak(pid):
    if os.name != "nt":
        return 0
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x410, False, pid)
    if not handle:
        return 0
    try:
        counters = (ctypes.c_size_t * 10)()
        counters[0] = ctypes.sizeof(counters)
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle), ctypes.byref(counters), ctypes.sizeof(counters)):
            return int(counters[1])
        return 0
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


def supervise(root, out, index, deadline):
    worker = mp.get_context("spawn").Process(target=run_fold, args=(root, out, index))
    worker.start()
    try:
        while worker.is_alive():
            remaining = deadline - perf_counter()
            if remaining <= 0:
                raise TimeoutError("The 30-minute wall-clock experiment budget expired.")
            if worker_peak(worker.pid) >= MEMORY_LIMIT_BYTES:
                raise MemoryError("The worker reached the 7.5 GB safety limit.")
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
        raise ValueError("Keep the outputs separate and use the development-only V2 inputs.")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(THREADS)
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
