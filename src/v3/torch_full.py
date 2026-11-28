"""Fit the original dense tabular MLP on full chronological gradient prefixes."""

import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from src.lab_benchmark import stratified_positions
from src.v2.challengers import SUBSET, TRAIN_LIMIT
from src.v2.data import sha256, write_json
from src.v2.experiment import _peak_memory, review_budgets
from .behavior import anchor_keys
from .diagnostics import annotated_metrics, development_inputs, seen_groups
from .forest_experiment import training_matrix
from .logistic_experiment import MEMORY_LIMIT_BYTES, worker_peak
from .logistic_features import ChunkedDenseFeatures


NAME = "torch_mlp_regularized"
CANDIDATE = SUBSET[NAME]
THREADS, SEED, EPOCHS, PATIENCE, BATCH = 6, 42, 20, 4, 256
CUTOFF, SCORE_BATCH, BUDGET = .5, 2048, 45 * 60
SCORE_ATOL = 1e-6


def gradient_end(times):
    """Match V2's 90% cut and move it past all rows tied at that timestamp."""
    end = int(np.searchsorted(times, times[int(.90 * len(times)) - 1], side="right"))
    if not 0 < end < len(times):
        raise ValueError("Expected non-empty gradient and early-stopping blocks.")
    return end


def network(width):
    layers = []
    for units in CANDIDATE.options["widths"]:
        layers.extend((nn.Linear(width, units), nn.BatchNorm1d(units), nn.GELU(), nn.Dropout(CANDIDATE.options["dropout"])))
        width = units
    return nn.Sequential(*layers, nn.Linear(width, 1))


def score(model, dense, labels=None):
    model.eval()
    outputs = []
    loss = 0.
    with torch.inference_mode():
        for lo in range(0, len(dense), SCORE_BATCH):
            hi = min(lo + SCORE_BATCH, len(dense))
            logits = model(torch.from_numpy(dense[lo:hi])).squeeze(1)
            outputs.append(torch.sigmoid(logits).numpy())
            if labels is not None:
                target = torch.from_numpy(labels[lo:hi].astype(np.float32))
                loss += nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="sum").item()
    return np.concatenate(outputs), loss / len(dense) if labels is not None else None


def score_source(model, matrix, transform):
    return np.concatenate([score(model, transform.transform(matrix[lo:lo + SCORE_BATCH]))[0]
                           for lo in range(0, len(matrix), SCORE_BATCH)])


def save_atomic(payload, path, writer):
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    writer(payload, temporary)
    temporary.replace(path)


def fit_epochs(model, training, y, internal, y_internal, progress, provenance):
    config = CANDIDATE.options
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1., dtype=torch.float32))
    X, y_tensor = torch.from_numpy(training), torch.from_numpy(y.astype(np.float32))
    rng = np.random.default_rng(SEED)
    history, best, best_epoch, best_state = [], -np.inf, 0, None
    if progress.exists():
        state = torch.load(progress, map_location="cpu", weights_only=True)
        if state["provenance"] != provenance:
            raise ValueError("Saved epoch progress differs from the current experiment.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        rng.bit_generator.state = json.loads(state["numpy_rng"])
        history, best, best_epoch, best_state = (state[key] for key in ("history", "best", "best_epoch", "best_state"))
        del state
        print(f"Resuming after epoch {len(history)}.", flush=True)

    for epoch in range(len(history) + 1, EPOCHS + 1):
        if epoch - 1 - best_epoch >= PATIENCE:
            break
        start = perf_counter()
        model.train()
        rate = optimizer.param_groups[0]["lr"]
        total = 0.
        for batch in np.array_split(rng.permutation(len(y)), int(np.ceil(len(y) / BATCH))):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X[batch]).squeeze(1), y_tensor[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        scheduler.step()
        predictions, internal_loss = score(model, internal, y_internal)
        ap = float(average_precision_score(y_internal, predictions))
        train_loss = total / len(y)
        if not np.isfinite([train_loss, internal_loss, ap]).all():
            raise ValueError("Non-finite neural training or internal metrics.")
        history.append({"epoch": epoch, "train_loss": train_loss, "internal_validation_loss": internal_loss,
                        "internal_validation_ap": ap, "learning_rate": rate, "seconds": perf_counter() - start})
        if ap > best:
            best, best_epoch = ap, epoch
            best_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
                     "numpy_rng": json.dumps(rng.bit_generator.state), "history": history, "best": best,
                     "best_epoch": best_epoch, "best_state": best_state, "provenance": provenance}, progress, torch.save)
        print(f"Epoch {epoch:2d}: train loss {train_loss:.5f}, internal loss {internal_loss:.5f}, "
              f"internal AP {ap:.5f}, best {best_epoch}, {history[-1]['seconds']:.1f}s.", flush=True)
    model.load_state_dict(best_state)
    return history, best_epoch


def check_control(folder, index, train_end, stop, root, matrix, labels, positions, categories):
    record = json.loads((folder / f"fold{index}.json").read_bytes())
    candidate = json.loads(json.dumps(asdict(CANDIDATE)))
    if (record["candidate"] != candidate or record["fit_rows"] != TRAIN_LIMIT or record["train_end"] != train_end
            or record["validation_end"] != stop or record["threads"] != THREADS or record["threshold"] != CUTOFF
            or record["feature_sha256"] != sha256(root / "advanced.json")):
        raise ValueError(f"Saved neural model metadata mismatch, fold {index}.")
    archived = joblib.load(folder / f"model{index}.joblib")
    if archived["train_rows"] != TRAIN_LIMIT:
        raise ValueError("The archived model has an unexpected gradient sample size.")
    expected = np.load(folder / f"scores{index}.npy", allow_pickle=False)
    actual = score_source(archived["model"], matrix[train_end:stop], archived["transform"])
    if expected.shape != (stop - train_end,) or not np.isfinite(expected).all():
        raise ValueError("Invalid archived validation score array.")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=SCORE_ATOL)
    probe = matrix[train_end:train_end + SCORE_BATCH]
    prep = ChunkedDenseFeatures(CANDIDATE.top_k, categories, False).fit(matrix[positions], labels[positions])
    original = archived["transform"]
    np.testing.assert_array_equal(prep.selected, original.selected)
    max_difference = float(np.max(np.abs(prep.transform(probe) - original.transform(probe))))
    if max_difference > SCORE_ATOL:
        raise ValueError("Chunked preprocessing changed the sampled representation.")
    return archived, record, expected, float(np.max(np.abs(actual - expected))), max_difference


def run_fold(root, out, index):
    started = perf_counter()
    matrices, times, labels, plan = development_inputs(root)
    cache, advanced, _, matrix = matrices
    names = advanced["features"]
    fold = plan[index - 1]
    start, stop = fold.train_end, fold.validation_end
    if len(names) != 497 or matrix.dtype != np.float32 or start != (212_594, 283_459, 354_324)[index - 1]:
        raise ValueError("Expected the unchanged V2 matrix and fold boundaries.")
    end = gradient_end(times[:start])
    archive = root / "challengers" / NAME
    inputs = (root / "advanced.json", root / "development_metadata.parquet",
              *(archive / name for name in (f"fold{index}.json", f"model{index}.joblib", f"scores{index}.npy")))
    provenance = {"fold": index, "gradient_rows": end, "early_stopping_rows": start - end,
                  "train_end": start, "validation_end": stop, "threads": THREADS, "seed": SEED,
                  "candidate": json.loads(json.dumps(asdict(CANDIDATE))), "torch_version": str(torch.__version__),
                  "runner_sha256": sha256(Path(__file__)), "input_sha256": {p.name: sha256(p) for p in inputs}}
    folder = out / f"fold{index}"
    folder.mkdir(exist_ok=True)
    model_path, prep_path = folder / "best_model.pt", folder / "preprocessor.joblib"
    scores_path, record_path = folder / "external_scores.npy", folder / "result.json"
    if record_path.exists():
        previous = json.loads(record_path.read_bytes())
        if (previous["provenance"] != provenance
                or any(sha256(folder / name) != digest for name, digest in previous["artifact_sha256"].items())):
            raise ValueError(f"Completed checkpoint failed verification in fold {index}.")
        print(f"Fold {index}: verified and reused completed checkpoint.", flush=True)
        return

    print(f"Fold {index}: {end:,} gradient rows, {start-end:,} early-stop rows, {stop-start:,} external rows.", flush=True)
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=THREADS):
        categories = [i for i, name in enumerate(names) if name in cache["categorical"]]
        sample = stratified_positions(labels[:end], TRAIN_LIMIT, seed=SEED)
        control, original, control_scores, replay_error, prep_error = check_control(
            archive, index, start, stop, root, matrix, labels, sample, categories)
        prep_time = perf_counter()
        prep = ChunkedDenseFeatures(CANDIDATE.top_k, categories, False).fit(matrix[:end], labels[:end])
        training = training_matrix(prep, matrix[:end])
        internal = training_matrix(prep, matrix[end:start])
        prep_seconds = perf_counter() - prep_time
        torch.manual_seed(SEED)
        model = network(training.shape[1])
        if repr(model) != repr(control["model"]):
            raise ValueError("The full-data architecture differs from the archived MLP.")
        print(f"Fold {index}: preprocessing {prep_seconds:.1f}s, starting full-data epochs.", flush=True)
        history, best = fit_epochs(model, training, labels[:end], internal, labels[end:start], folder / "epoch_progress.pt", provenance)
        training_seconds = prep_seconds + sum(row["seconds"] for row in history)
        save_atomic({"state_dict": model.state_dict(), "input_dim": training.shape[1], "best_epoch": best}, model_path, torch.save)
        save_atomic(prep, prep_path, lambda payload, path: joblib.dump(payload, path, compress=1))
        started_inference = perf_counter()
        full_scores = score_source(model, matrix[start:stop], prep)
        inference_seconds = perf_counter() - started_inference
        restored = torch.load(model_path, weights_only=True, map_location="cpu")
        reloaded = network(restored["input_dim"])
        reloaded.load_state_dict(restored["state_dict"])
        replay = score_source(reloaded, matrix[start:stop], joblib.load(prep_path))
        np.testing.assert_allclose(replay, full_scores, rtol=0, atol=SCORE_ATOL)
        control_train, _ = score(control["model"], control["transform"].transform(matrix[sample]))
        full_train, _ = score(model, training)
        target = labels[start:stop]
        keys = anchor_keys(matrix[:stop], names)
        # Also report common strata, since the two fits see different entity sets.
        full_groups = seen_groups(keys, np.arange(end), start, stop)
        sample_groups = seen_groups(keys, sample, start, stop)
        by_fit = {arm: {group: annotated_metrics(target[mask], preds[mask], CUTOFF) for group, mask in groups.items()}
                  for arm, preds, groups in (("sampled", control_scores, sample_groups), ("full", full_scores, full_groups))}
        common = {group: {arm: annotated_metrics(target[mask], preds[mask], CUTOFF)
                          for arm, preds in (("sampled", control_scores), ("full", full_scores))}
                  for group, mask in full_groups.items()}
        del training, internal

    peak = _peak_memory()
    if peak is not None and peak >= MEMORY_LIMIT_BYTES:
        raise MemoryError("The worker reached the 7.5 GB working-set guard.")
    save_atomic(full_scores, scores_path, lambda payload, path: np.save(path, payload, allow_pickle=False))
    result = {"provenance": provenance, "external": {"sampled": annotated_metrics(target, control_scores, CUTOFF),
                                                     "full": annotated_metrics(target, full_scores, CUTOFF)},
              "train": {"sampled": annotated_metrics(labels[sample], control_train, CUTOFF),
                        "full": annotated_metrics(labels[:end], full_train, CUTOFF)},
              "review_budgets": {"sampled": review_budgets(target, control_scores), "full": review_budgets(target, full_scores)},
              "entity_by_actual_fit": by_fit, "entity_common_full_fit": common, "input_dim": restored["input_dim"],
              "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "architecture": repr(model), "history": history, "best_epoch": best,
              "archived": {"gradient_rows": original["fit_rows"], "history": original["learning_history"],
                           "fit_seconds": original["fit_seconds"], "inference_seconds": original["inference_seconds"],
                           "peak_working_set_bytes": original["process_peak_working_set"],
                           "checkpoint_bytes": (archive / f"model{index}.joblib").stat().st_size},
              "prep_seconds": prep_seconds, "training_seconds": training_seconds, "inference_seconds": inference_seconds,
              "worker_seconds": perf_counter() - started, "peak_working_set_bytes": peak,
              "checkpoint_bytes": model_path.stat().st_size + prep_path.stat().st_size,
              "checkpoint_parts_bytes": {path.name: path.stat().st_size for path in (model_path, prep_path)},
              "reload_max_score_difference": float(np.max(np.abs(replay - full_scores))),
              "control_max_score_difference": replay_error, "control_preprocessing_max_difference": prep_error,
              "artifact_sha256": {path.name: sha256(path) for path in (model_path, prep_path, scores_path)}}
    write_json(record_path, result)
    print(f"Fold {index}: external AP {result['external']['sampled']['average_precision']:.5f} to "
          f"{result['external']['full']['average_precision']:.5f}; checkpoint saved.", flush=True)


def supervise(root, out, index, deadline):
    worker = mp.get_context("spawn").Process(target=run_fold, args=(root, out, index))
    worker.start()
    try:
        while worker.is_alive():
            remaining = deadline - perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"Fold {index} reached the 45-minute wall-clock budget.")
            if worker_peak(worker.pid) >= MEMORY_LIMIT_BYTES:
                raise MemoryError(f"Fold {index} reached the 7.5 GB working-set guard.")
            worker.join(min(1., remaining))
        if worker.exitcode != 0:
            raise RuntimeError(f"Fold {index} exited with status {worker.exitcode}.")
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join()
        worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.development.resolve(), args.output.resolve()
    if out.is_relative_to(root) or "retrospective" in {p.lower() for p in root.parts}:
        raise ValueError("Use the V2 development cache and a separate experiment directory.")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(THREADS)
    started = perf_counter()
    done, constraint = [], None
    try:
        for index in (1, 2, 3):
            supervise(root, out, index, started + BUDGET)
            done.append(index)
    except (RuntimeError, MemoryError, TimeoutError) as exc:
        constraint = str(exc)
        raise
    finally:
        write_json(out / "run_status.json", {"verified_folds": done, "constraint": constraint,
                                            "wall_seconds": perf_counter() - started,
                                            "budget_seconds": BUDGET, "memory_guard_bytes": MEMORY_LIMIT_BYTES})


if __name__ == "__main__":
    main()
