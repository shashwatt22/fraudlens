"""Refit the four selected V3 models before the reserved development policy block."""

import argparse
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
import torch
import xgboost as xgb

from src.train import make_models
from src.v2.challengers import SUBSET, _estimator
from src.v2.data import sha256, write_json
from src.v2.experiment import MatrixView, ViewIterator, _peak_memory
from src.v2.final import policy_boundaries
from .depth_experiment import depth10_trial
from .diagnostics import development_inputs
from .forest_experiment import training_matrix
from .logistic_experiment import MEMORY_LIMIT_BYTES, worker_peak
from .logistic_features import ChunkedDenseFeatures, solver_matrix
from .torch_full import BUDGET, SEED, THREADS, fit_epochs, gradient_end, network, save_atomic
from .torch_rich_features import validate_features


SELECTIONS = {
    "logistic": ("step3a-logistic", "logistic_full.joblib", "full_scores.npy", "full_model_sha256", "full_scores_sha256"),
    "forest": ("step4a-forest", "forest_full.joblib", "full_scores.npy", "full_model_sha256", "full_scores_sha256"),
    "xgboost": ("step5a-depth", "depth10.ubj", "depth10_scores.npy", "model_sha256", "scores_sha256"),
    "torch": ("step6a-torch", "best_model.pt", "external_scores.npy", None, None),
}
CANDIDATES = {"logistic": "logistic_l2", "forest": "random_forest_depth", "torch": "torch_mlp_regularized"}


def populations(times, plan):
    end = plan[-1].validation_end
    if not 0 < end < len(times) or times[end-1] >= times[end]:
        raise ValueError("The reserved policy boundary must be chronological and keep timestamp ties intact.")
    grad = gradient_end(times[:end])
    quarter, half = policy_boundaries(times[end:])
    return {"pre_policy": (0, end), "torch_gradient": (0, grad), "torch_internal": (grad, end),
            "policy": (end, len(times)), "earlier_policy": (end, end + half), "later_policy": (end + half, len(times)),
            "original_calibration_cut": end + quarter}


def safe_paths(root, prior, out):
    root, prior, out = (path.resolve() for path in (root, prior, out))
    if (out.is_relative_to(root) or out == prior or any("retrospective" in path.as_posix().lower().split("/") for path in (root, prior, out))
            or any(out.is_relative_to(prior / value[0]) or (prior / value[0]).is_relative_to(out) for value in SELECTIONS.values())):
        raise ValueError("Use the development cache and a separate output directory.")
    return root, prior, out


def frozen_settings(labels):
    recipes = {name: {"candidate": asdict(SUBSET[key]), "estimator": _estimator(SUBSET[key], THREADS).get_params()}
               for name, key in CANDIDATES.items() if name != "torch"}
    trial = depth10_trial()
    xgb_model = make_models(labels, THREADS)["xgboost"].set_params(**trial.overrides)
    params = xgb_model.get_xgb_params()
    params["scale_pos_weight"] = 1.0
    recipes["xgboost"] = {"trial": asdict(trial), "estimator": params, "rounds": xgb_model.get_params()["n_estimators"]}
    recipes["torch"] = {"candidate": asdict(SUBSET[CANDIDATES["torch"]]), "input_dim": 224, "seed": SEED,
                        "batch_size": 256, "epoch_cap": 20, "patience": 4, "loss": "unweighted BCEWithLogitsLoss",
                        "optimizer": "AdamW", "schedule": "CosineAnnealingLR(T_max=20)"}
    return json.loads(json.dumps(recipes, allow_nan=False))


def selected_artifacts(prior, recipes):
    """Verify the selected checkpoints, including the rejected trials' separation by path."""
    records = {}
    for name, (folder, model, scores, model_key, score_key) in SELECTIONS.items():
        path = prior / folder / "fold3"
        result = json.loads((path / "result.json").read_bytes())
        hashes = ({model: result[model_key], scores: result[score_key]} if model_key else result["artifact_sha256"])
        if any(sha256(path / file) != digest for file, digest in hashes.items()):
            raise ValueError(f"Selected {name} artifacts failed verification.")
        if name in ("logistic", "forest"):
            matches = result["settings"] == recipes[name]["estimator"] and result["candidate"] == recipes[name]["candidate"]
        elif name == "torch":
            matches = result["provenance"]["candidate"] == recipes[name]["candidate"] and result["input_dim"] == recipes[name]["input_dim"]
        else:
            matches = result["challenger"] == recipes[name]["trial"] and result["rounds"] == recipes[name]["rounds"]
        if not matches:
            raise ValueError(f"Selected {name} recipe differs from the archived trial.")
        records[name] = {"path": f"{folder}/fold3", "result_sha256": sha256(path / "result.json"), "artifacts": hashes}
    return records


def freeze(root, prior, out):
    matrices, times, labels, plan = development_inputs(root)
    raw, advanced = matrices[:2]
    validate_features(advanced["features"])
    ranges = populations(times, plan)
    end = ranges["pre_policy"][1]
    if len(advanced["features"]) != 497 or end != 425_188 or len(times) - end != raw["policy_rows"] or len(times) != 472_432:
        raise ValueError("The fit or policy population differs from the saved V2 protocol.")
    recipes = frozen_settings(labels[:end])
    payload = {"scope": "previously inspected development policy block", "threads": THREADS, "rows": len(times),
               "features": len(advanced["features"]), "recipes": recipes, "selected": selected_artifacts(prior, recipes),
               "fold3": asdict(plan[-1]), "original_policy_calibration_end": ranges.pop("original_calibration_cut"),
               "partitions": {name: {"start": lo, "end": hi, "rows": hi-lo, "first_second": int(times[lo]),
                                     "last_second": int(times[hi-1]), "fraud_rows": int(labels[lo:hi].sum())}
                              for name, (lo, hi) in ranges.items()},
               "inputs": {name: sha256(root / name) for name in ("features.json", "advanced.json", "development_metadata.parquet")},
               "training_code_sha256": sha256(Path(__file__))}
    manifest = out / "refit_plan.json"
    if manifest.exists():
        if payload != json.loads(manifest.read_bytes()):
            raise ValueError("The existing refit plan differs from the verified configuration.")
    else:
        write_json(manifest, payload)
    return payload


def fit_model(name, source, labels, categories, names, plan, folder, freeze_digest):
    if len(source) != plan["partitions"]["pre_policy"]["rows"] or len(labels) != len(source):
        raise ValueError("Training received rows outside the frozen pre-policy population.")
    recipe = plan["recipes"][name]
    if name == "xgboost":
        view = MatrixView(source, dict(zip(names, range(len(names)))), {names[i] for i in categories}).fit(len(source))
        matrix = xgb.QuantileDMatrix(ViewIterator(view, labels, 0, len(source)), max_bin=recipe["estimator"]["max_bin"],
                                   nthread=THREADS, enable_categorical=True)
        model = xgb.train(recipe["estimator"], matrix, num_boost_round=recipe["rounds"])
        if model.num_boosted_rounds() != recipe["rounds"]:
            raise ValueError("Unexpected boosting rounds.")
        return model, None, {"input_dim": len(names), "rounds": model.num_boosted_rounds()}

    end = plan["partitions"]["torch_gradient"]["end"] if name == "torch" else len(source)
    prep = ChunkedDenseFeatures(recipe["candidate"]["top_k"], categories, name == "logistic").fit(source[:end], labels[:end])
    if name == "torch":
        training, internal = training_matrix(prep, source[:end]), training_matrix(prep, source[end:])
        torch.manual_seed(SEED)
        model = network(training.shape[1])
        history, best = fit_epochs(model, training, labels[:end], internal, labels[end:], folder / "epoch_progress.pt",
                                   {"freeze_sha256": freeze_digest, "name": name})
        return {"state_dict": model.state_dict(), "input_dim": training.shape[1], "best_epoch": best}, prep, {
            "input_dim": training.shape[1], "best_epoch": best, "history": history,
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}

    model = _estimator(SUBSET[CANDIDATES[name]], THREADS)
    training = solver_matrix(prep, source) if name == "logistic" else training_matrix(prep, source)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(training, labels)
    info = {"input_dim": training.shape[1]}
    if name == "logistic":
        info["iterations"] = int(model.n_iter_.max())
        info["convergence_warnings"] = [str(item.message) for item in messages if issubclass(item.category, ConvergenceWarning)]
        info["converged"] = info["iterations"] < model.max_iter and not info["convergence_warnings"]
    else:
        info["trees"] = len(model.estimators_)
    return model, prep, info


def run_fit(root, out, name):
    manifest = out / "refit_plan.json"
    plan = json.loads(manifest.read_bytes())
    digest = sha256(manifest)
    folder = out / name
    folder.mkdir(exist_ok=True)
    record_path = folder / "fit.json"
    if record_path.exists():
        record = json.loads(record_path.read_bytes())
        if record["freeze_sha256"] != digest or any(sha256(folder / file) != checksum for file, checksum in record["artifacts"].items()):
            raise ValueError(f"Saved {name} checkpoint cannot be reused.")
        print(f"Verified and reused {name}.", flush=True)
        return
    matrices, _, labels, _ = development_inputs(root)
    cache, advanced, _, source = matrices
    names = advanced["features"]
    categories = [i for i, field in enumerate(names) if field in cache["categorical"]]
    end = plan["partitions"]["pre_policy"]["end"]
    start = perf_counter()
    print(f"Training {name} with the frozen configuration through row {end:,}.", flush=True)
    if name == "torch":
        torch.set_num_threads(THREADS)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1 if name == "forest" else THREADS):
        model, prep, extras = fit_model(name, source[:end], labels[:end], categories, names, plan, folder, digest)
    seconds = perf_counter() - start
    if name == "xgboost":
        file = folder / "model.ubj"
        save_atomic(model, file, lambda saved, path: saved.save_model(path))
        files = [file]
    else:
        file = folder / ("model.pt" if name == "torch" else "model.joblib")
        dump = lambda obj, path: joblib.dump(obj, path, compress=1)
        save_atomic(model, file, torch.save if name == "torch" else dump)
        preprocessor = folder / "preprocessor.joblib"
        save_atomic(prep, preprocessor, dump)
        files = [file, preprocessor]
    peak = _peak_memory()
    if peak is not None and peak >= MEMORY_LIMIT_BYTES:
        raise MemoryError("The fit exceeded its memory guard.")
    write_json(record_path, {"name": name, "freeze_sha256": digest, "model_file": file.name,
                             "artifacts": {p.name: sha256(p) for p in files}, "checkpoint_bytes": sum(p.stat().st_size for p in files),
                             "fit_seconds": seconds, "peak_working_set_bytes": peak, **extras})
    print(f"Checkpoint saved for {name} after {seconds:.1f}s.", flush=True)


def supervise(root, out, name, deadline):
    worker = mp.get_context("spawn").Process(target=run_fit, args=(root, out, name))
    worker.start()
    try:
        while worker.is_alive():
            remaining = deadline - perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"{name} reached the 45-minute run budget.")
            if worker_peak(worker.pid) >= MEMORY_LIMIT_BYTES:
                raise MemoryError(f"{name} reached the 7.5 GB worker guard.")
            worker.join(min(1., remaining))
        if worker.exitcode != 0:
            raise RuntimeError(f"{name} exited with status {worker.exitcode}.")
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join()
        worker.close()


def add_paths(parser):
    for name in ("development", "prior", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_paths(parser)
    args = parser.parse_args()
    root, prior, out = safe_paths(args.development, args.prior, args.output)
    out.mkdir(parents=True, exist_ok=True)
    freeze(root, prior, out)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(THREADS)
    start, done, constraint = perf_counter(), [], None
    try:
        for name in SELECTIONS:
            supervise(root, out, name, start + BUDGET)
            done.append(name)
    except (RuntimeError, TimeoutError, MemoryError) as exc:
        constraint = str(exc)
        raise
    finally:
        write_json(out / "run_status.json", {"completed": done, "constraint": constraint, "wall_seconds": perf_counter() - start,
                                            "budget_seconds": BUDGET, "memory_guard_bytes": MEMORY_LIMIT_BYTES})


if __name__ == "__main__":
    main()
