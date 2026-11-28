"""Compare one wider neural input representation with the saved Step 6A control."""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import torch

from src.v2.data import sha256, write_json
from src.v2.experiment import _peak_memory, review_budgets
from .behavior import anchor_keys
from .diagnostics import annotated_metrics, development_inputs, seen_groups
from .logistic_experiment import MEMORY_LIMIT_BYTES, worker_peak
from .torch_full import (BUDGET, CUTOFF, SCORE_ATOL, SEED, THREADS, fit_epochs, gradient_end,
                         network, save_atomic, score, score_source)
from .torch_rich_features import RichNeuralFeatures, input_inventory, transform_block, validate_features


def load_control(folder, boundaries, source_digests, external):
    record = json.loads((folder / "result.json").read_bytes())
    provenance = record["provenance"]
    if (any(provenance.get(key) != value for key, value in boundaries.items())
            or any(provenance["input_sha256"].get(key) != value for key, value in source_digests.items())
            or record["input_dim"] != 224 or provenance["seed"] != SEED or provenance["threads"] != THREADS
            or provenance["runner_sha256"] != sha256(Path(__file__).with_name("torch_full.py"))
            or any(sha256(folder / name) != digest for name, digest in record["artifact_sha256"].items())):
        raise ValueError("Step 6A artifacts or populations differ from the comparison protocol.")
    scores = np.load(folder / "external_scores.npy", allow_pickle=False)
    payload = torch.load(folder / "best_model.pt", weights_only=True, map_location="cpu")
    prep = joblib.load(folder / "preprocessor.joblib")
    model = network(payload["input_dim"])
    model.load_state_dict(payload["state_dict"])
    replay = score_source(model, external, prep)
    if not np.isfinite(scores).all():
        raise ValueError("The archived external scores are not finite.")
    np.testing.assert_allclose(replay, scores, rtol=0, atol=SCORE_ATOL)
    return record, scores, prep, float(np.max(np.abs(replay - scores)))


def run_fold(root, baseline, out, index):
    wall = perf_counter()
    matrices, times, labels, folds = development_inputs(root)
    cache, advanced, _, matrix = matrices
    names = advanced["features"]
    validate_features(names)
    if len(names) != 497 or matrix.dtype != np.float32:
        raise ValueError("Expected the fixed V2 feature matrix.")
    fold = folds[index - 1]
    start, stop = fold.train_end, fold.validation_end
    end = gradient_end(times[:start])
    boundaries = {"fold": index, "gradient_rows": end, "early_stopping_rows": start - end,
                  "train_end": start, "validation_end": stop}
    source_files = ("advanced.json", "development_metadata.parquet")
    source_digests = {name: sha256(root / name) for name in source_files}
    old_folder = baseline / f"fold{index}"
    folder = out / f"fold{index}"
    folder.mkdir(exist_ok=True)
    model_path, prep_path = folder / "best_model.pt", folder / "preprocessor.joblib"
    scores_path, record_path = folder / "external_scores.npy", folder / "result.json"
    provenance = {**boundaries, "source_sha256": source_digests, "threads": THREADS, "seed": SEED,
                  "torch_version": str(torch.__version__), "control_result_sha256": sha256(old_folder / "result.json"),
                  "implementation_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                                             ("torch_rich.py", "torch_rich_features.py", "torch_full.py")}}
    if record_path.exists():
        existing = json.loads(record_path.read_bytes())
        if (existing["provenance"] != provenance
                or any(sha256(folder / name) != digest for name, digest in existing["artifact_sha256"].items())):
            raise ValueError(f"Fold {index} completed checkpoint differs from its manifest.")
        print(f"Fold {index}: verified and reused.", flush=True)
        return

    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=THREADS):
        control, old_scores, old_prep, control_error = load_control(old_folder, boundaries, source_digests, matrix[start:stop])
        categories = [i for i, name in enumerate(names) if name in cache["categorical"]]
        prepare = perf_counter()
        prep = RichNeuralFeatures(categories).fit(matrix[:end])
        inventory = input_inventory(names, categories, old_prep, prep)
        width = len(prep.selected)
        training = transform_block(prep, matrix[:end])
        internal = transform_block(prep, matrix[end:start])
        prep_seconds = perf_counter() - prepare
        torch.manual_seed(SEED)
        model = network(width)
        print(f"Fold {index}: 224 to {width} inputs, {end:,} gradient rows, {start-end:,} early-stop rows; "
              f"preprocessing {prep_seconds:.1f}s.", flush=True)
        history, best = fit_epochs(model, training, labels[:end], internal, labels[end:start], folder / "epoch_progress.pt", provenance)
        save_atomic({"state_dict": model.state_dict(), "input_dim": width, "best_epoch": best}, model_path, torch.save)
        save_atomic(prep, prep_path, lambda payload, path: joblib.dump(payload, path, compress=1))
        inference = perf_counter()
        new_scores = score_source(model, matrix[start:stop], prep)
        inference_seconds = perf_counter() - inference
        train_scores, _ = score(model, training)
        del training, internal
        saved = torch.load(model_path, weights_only=True, map_location="cpu")
        restored = network(saved["input_dim"])
        restored.load_state_dict(saved["state_dict"])
        replay = score_source(restored, matrix[start:stop], joblib.load(prep_path))
        np.testing.assert_allclose(replay, new_scores, rtol=0, atol=SCORE_ATOL)
        if not (np.isfinite(new_scores).all() and np.isfinite(train_scores).all()):
            raise ValueError("Challenger scores must be finite.")

    truth = labels[start:stop]
    arms = {"control": old_scores, "challenger": new_scores}
    keys = anchor_keys(matrix[:stop], names)
    groups = seen_groups(keys, np.arange(end), start, stop)
    peak = _peak_memory()
    if peak is not None and peak >= MEMORY_LIMIT_BYTES:
        raise MemoryError("The worker reached its 7.5 GB working-set guard.")
    save_atomic(new_scores, scores_path, lambda values, path: np.save(path, values, allow_pickle=False))
    result = {"provenance": provenance, "input_inventory": inventory, "input_dim": width,
              "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "architecture": repr(model),
              "control_input_dim": control["input_dim"], "control_parameters": control["parameters"],
              "history": history, "best_epoch": best, "control_history": control["history"], "control_best_epoch": control["best_epoch"],
              "external": {arm: annotated_metrics(truth, scores, CUTOFF) for arm, scores in arms.items()},
              "training": {"control": control["train"]["full"], "challenger": annotated_metrics(labels[:end], train_scores, CUTOFF)},
              "review_budgets": {arm: review_budgets(truth, scores) for arm, scores in arms.items()},
              "entities": {name: {arm: annotated_metrics(truth[mask], scores[mask], CUTOFF) for arm, scores in arms.items()}
                           for name, mask in groups.items()},
              "prep_seconds": prep_seconds, "training_seconds": prep_seconds + sum(row["seconds"] for row in history),
              "inference_seconds": inference_seconds, "worker_seconds": perf_counter() - wall, "peak_working_set_bytes": peak,
              "checkpoint_bytes": model_path.stat().st_size + prep_path.stat().st_size,
              "control_resources": {name: control[name] for name in ("training_seconds", "inference_seconds", "peak_working_set_bytes", "checkpoint_bytes")},
              "control_replay_max_difference": control_error, "challenger_replay_max_difference": float(np.max(np.abs(replay - new_scores))),
              "artifact_sha256": {file.name: sha256(file) for file in (model_path, prep_path, scores_path)}}
    write_json(record_path, result)
    print(f"Fold {index}: external AP {result['external']['control']['average_precision']:.5f} to "
          f"{result['external']['challenger']['average_precision']:.5f}. Checkpoint saved.", flush=True)


def supervise(root, control, out, index, deadline):
    worker = mp.get_context("spawn").Process(target=run_fold, args=(root, control, out, index))
    worker.start()
    try:
        while worker.is_alive():
            remaining = deadline - perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"Fold {index} reached the 45-minute run budget.")
            if worker_peak(worker.pid) >= MEMORY_LIMIT_BYTES:
                raise MemoryError(f"Fold {index} reached the 7.5 GB worker guard.")
            worker.join(min(1., remaining))
        if worker.exitcode != 0:
            raise RuntimeError(f"Fold {index} exited with status {worker.exitcode}.")
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join()
        worker.close()


def experiment_args(parser):
    for flag in ("development", "control", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)


def paths(args):
    root, control, out = (getattr(args, field).resolve() for field in ("development", "control", "output"))
    if out.is_relative_to(root) or out.is_relative_to(control) or any(
            "retrospective" in path.parts for path in (root, control, out)):
        raise ValueError("Use development inputs and a separate challenger output directory.")
    return root, control, out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    experiment_args(parser)
    root, control, out = paths(parser.parse_args())
    out.mkdir(parents=True, exist_ok=True)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(THREADS)
    start, done, constraint = perf_counter(), [], None
    try:
        for index in (1, 2, 3):
            supervise(root, control, out, index, start + BUDGET)
            done.append(index)
    except (RuntimeError, TimeoutError, MemoryError) as exc:
        constraint = str(exc)
        raise
    finally:
        write_json(out / "run_status.json", {"verified_folds": done, "constraint": constraint, "wall_seconds": perf_counter() - start,
                                            "budget_seconds": BUDGET, "memory_guard_bytes": MEMORY_LIMIT_BYTES})


if __name__ == "__main__":
    main()
