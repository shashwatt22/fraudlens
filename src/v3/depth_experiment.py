"""Compare depth 10 with the saved depth-8 XGBoost on the V2 development folds."""

import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import xgboost as xgb
from threadpoolctl import threadpool_limits

from src.v2.challengers import BOOSTING
from src.v2.data import CHUNK_ROWS, sha256, write_json
from src.v2.experiment import MatrixView, Trial, _peak_memory, _train_rich, review_budgets
from .behavior import anchor_keys
from .behavior_experiment import ROUNDS, THREADS, THRESHOLD, check_control, load_inputs, metrics_by_group
from .diagnostics import XGB, annotated_metrics, seen_groups
from .logistic_experiment import BUDGET_SECONDS, MEMORY_LIMIT_BYTES, worker_peak


def depth10_trial():
    control = BOOSTING[XGB]
    if control.options["max_depth"] != 8 or control.options["n_estimators"] != ROUNDS or control.weighted:
        raise ValueError("The archived depth-8 configuration changed.")
    return Trial("v3_depth10", "advanced", {**control.options, "max_depth": 10}, control.weighted)


def predict(model, view, start, stop):
    return np.concatenate([model.inplace_predict(view.read(i, min(i + CHUNK_ROWS, stop)), validate_features=False)
                           for i in range(start, stop, CHUNK_ROWS)])


def run_fold(root, out, index):
    started = perf_counter()
    root, out, matrices, times, labels, folds = load_inputs(root, out)
    fold = folds[index - 1]
    start, stop = fold.train_end, fold.validation_end
    cache, advanced, _, matrix = matrices
    names = advanced["features"]
    if len(names) != 497 or matrix.dtype != np.float32 or start != (212_594, 283_459, 354_324)[index - 1]:
        raise ValueError("Expected the unchanged V2 matrix and training folds.")
    trial = depth10_trial()
    directory = out / f"fold{index}"
    directory.mkdir(exist_ok=True)
    model_path = directory / "depth10.ubj"
    score_path = directory / "depth10_scores.npy"
    record_path = directory / "result.json"
    archived = root / "challengers" / XGB
    source_files = (root / "advanced.json", root / "development_metadata.parquet",
                    archived / f"fold{index}.json", archived / f"model{index}.joblib", archived / f"scores{index}.npy")
    provenance = {"fold": index, "fit_rows": start, "validation_rows": stop - start, "validation_end": stop,
                  "threads": THREADS, "cutoff": THRESHOLD, "challenger": asdict(trial),
                  "input_digests": {path.name: sha256(path) for path in source_files},
                  "runner_sha256": sha256(Path(__file__)), "xgboost_version": xgb.__version__}
    if record_path.exists():
        saved = json.loads(record_path.read_bytes())
        if (any(saved.get(key) != value for key, value in provenance.items())
                or saved["model_sha256"] != sha256(model_path) or saved["scores_sha256"] != sha256(score_path)):
            raise ValueError(f"Fold {index} differs from its completed checkpoint.")
        print(f"Fold {index}: verified existing checkpoint.", flush=True)
        return

    settings = {**cache, "features": names, "groups": {"advanced": names}}
    view = MatrixView(matrix, dict(zip(names, range(len(names)))), cache["categorical"]).fit(start)
    print(f"Fold {index}: checking depth-8 control on {stop - start:,} validation rows.", flush=True)
    with threadpool_limits(limits=THREADS):
        control, history = check_control(root, matrices, times, labels, fold, index)
        old = joblib.load(archived / f"model{index}.joblib")["model"]
        if old.num_boosted_rounds() != ROUNDS or old.feature_names != names:
            raise ValueError("The saved control model has unexpected rounds or inputs.")
        control_training = predict(old, view, 0, start)
        del old
        print(f"Fold {index}: fitting depth 10 on {start:,} rows.", flush=True)
        challenger, fit_seconds, inference_seconds, model = _train_rich(settings, matrix, trial, start, stop, labels, THREADS)
        if model.num_boosted_rounds() != ROUNDS or model.feature_names != names or not np.isfinite(challenger).all():
            raise ValueError("The challenger has unexpected rounds, inputs, or validation scores.")
        temporary = directory / "depth10.tmp.ubj"
        model.save_model(temporary)
        temporary.replace(model_path)
        temporary_scores = directory / "depth10_scores.tmp.npy"
        np.save(temporary_scores, challenger, allow_pickle=False)
        temporary_scores.replace(score_path)
        del model
        restored = xgb.Booster(params={"nthread": THREADS})
        restored.load_model(model_path)
        replay = predict(restored, view, start, stop)
        np.testing.assert_array_equal(replay, challenger)
        challenger_training = predict(restored, view, 0, start)
        if not np.isfinite(control_training).all() or not np.isfinite(challenger_training).all():
            raise ValueError("Expected finite predictions on the training rows.")
        del restored

    truth = labels[start:stop]
    keys = anchor_keys(matrix[:stop], names)
    groups = seen_groups(keys, np.arange(start), start, stop)
    peak = _peak_memory()
    if peak is not None and peak >= MEMORY_LIMIT_BYTES:
        raise MemoryError("Fold worker exceeded the 7.5 GB working-set guard.")
    record = {**provenance, "rounds": ROUNDS, "control": annotated_metrics(truth, control, THRESHOLD),
              "treatment": annotated_metrics(truth, challenger, THRESHOLD),
              "train": {"control": annotated_metrics(labels[:start], control_training, THRESHOLD),
                        "treatment": annotated_metrics(labels[:start], challenger_training, THRESHOLD)},
              "review_budgets": {"control": review_budgets(truth, control), "treatment": review_budgets(truth, challenger)},
              "entities": metrics_by_group(truth, control, challenger, groups),
              "archived_control": {"fit_seconds": history["fit_seconds"], "inference_seconds": history["inference_seconds"],
                                   "peak_working_set_bytes": history["process_peak_working_set"],
                                   "checkpoint_bytes": (archived / f"model{index}.joblib").stat().st_size},
              "challenger_resources": {"fit_seconds": fit_seconds, "inference_seconds": inference_seconds,
                                       "peak_working_set_bytes": peak, "checkpoint_bytes": model_path.stat().st_size},
              "worker_seconds": perf_counter() - started, "control_reload_max_difference": 0.0,
              "challenger_reload_max_difference": 0.0, "model_sha256": sha256(model_path), "scores_sha256": sha256(score_path)}
    write_json(record_path, record)
    print(f"Fold {index}: AP {record['control']['average_precision']:.5f} to {record['treatment']['average_precision']:.5f}; "
          f"depth-10 fit {fit_seconds:.1f}s. Checkpoint saved.", flush=True)


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
            worker.join(min(1.0, remaining))
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
        raise ValueError("Use V2 development inputs and a separate output directory.")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(THREADS)
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    verified, constraint = [], None
    try:
        for index in (1, 2, 3):
            supervise(root, out, index, started + BUDGET_SECONDS)
            verified.append(index)
    except (TimeoutError, MemoryError, RuntimeError) as error:
        constraint = str(error)
        raise
    finally:
        write_json(out / "run_status.json", {"verified_folds": verified, "constraint": constraint,
                                            "wall_seconds": perf_counter() - started, "budget_seconds": BUDGET_SECONDS,
                                            "memory_guard_bytes": MEMORY_LIMIT_BYTES})


if __name__ == "__main__":
    main()
