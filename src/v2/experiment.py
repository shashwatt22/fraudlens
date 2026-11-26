"""Run isolated, resumable V2 development experiments on the V1 pre-holdout rows."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data_loader import ID, TARGET, TIME
from src.evaluate import evaluate_scores, positive_scores
from src.phase1 import _resources
from src.preprocessing import FraudPreprocessor
from src.train import make_models
from src.training_data import load_partition, read_phase1_manifest

from .data import CHUNK_ROWS, build_cache, sha256, write_json


@dataclass(frozen=True)
class Trial:
    name: str
    features: str
    overrides: dict
    weighted: bool = True


TRIALS = {
    "rich": Trial("rich", "rich_raw", {}),
    "v1": Trial("v1", "v1", {}),
    "history": Trial("history", "rich_history", {}),
    "basic_native": Trial("basic_native", "basic_native", {}),
    "no_anonymous": Trial("no_anonymous", "no_anonymous", {}),
    "no_vesta": Trial("no_vesta", "no_vesta", {}),
    "no_identity": Trial("no_identity", "no_identity", {}),
    "unweighted": Trial("unweighted", "rich_history", {}, False),
    "xgb_depth8": Trial("xgb_depth8", "rich_history", {"max_depth": 8, "n_estimators": 350, "learning_rate": .05,
                                                      "reg_lambda": 4, "min_child_weight": 8}, False),
    "xgb_regularized": Trial("xgb_regularized", "rich_history", {"max_depth": 7, "n_estimators": 320, "learning_rate": .06,
                                                                  "reg_lambda": 8, "reg_alpha": 1, "subsample": .8}, True),
}


class MatrixView:
    def __init__(self, data, features, category):
        self.data, self.features = data, features
        self.columns = list(features.values())
        self.categories = [index for index, name in enumerate(features) if name in category]
        self.types = ["c" if name in category else "q" for name in features]
        self.max_bin_code = None

    def fit(self, end):
        self.max_bin_code = [self.data[:end, self.columns[i]][np.isfinite(self.data[:end, self.columns[i]])].max(initial=-1)
                             for i in self.categories]
        return self

    def read(self, start, end):
        batch = self.data[start:end, self.columns].copy(order="C")
        for i, cutoff in zip(self.categories, self.max_bin_code, strict=True):
            batch[batch[:, i] > cutoff, i] = np.nan
        return batch


class ViewIterator(xgb.DataIter):
    def __init__(self, view, labels, start, end):
        super().__init__(release_data=True)
        self.view, self.labels, self.start, self.end = view, labels, start, end
        self.reset()

    def next(self, input_data):
        if self.position >= self.end:
            return False
        end = min(self.end, self.position + CHUNK_ROWS)
        input_data(data=self.view.read(self.position, end), label=self.labels[self.position:end],
                   feature_names=list(self.view.features), feature_types=self.view.types)
        self.position = end
        return True

    def reset(self):
        self.position = self.start


def _peak_memory():
    import ctypes
    import os

    if os.name != "nt":
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if os.uname().sysname == "Linux" else 1))
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    buffer = (ctypes.c_size_t * 10)()
    buffer[0] = ctypes.sizeof(buffer)
    if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(kernel.GetCurrentProcess()),
                                                ctypes.byref(buffer), ctypes.sizeof(buffer)):
        return int(buffer[1])
    return None


def review_budgets(labels, scores):
    order = np.argsort(-scores, kind="stable")
    return {str(q): {"reviewed": n, "precision": float(labels[order[:n]].mean()),
                    "recall": float(labels[order[:n]].sum() / labels.sum())}
            for q in (1, 5, 10) for n in ((len(order) * q + 99) // 100,)}


def _v1_frame(source):
    manifest, _ = read_phase1_manifest(source)
    return pd.concat([load_partition(source, manifest, key) for key in ("train", "validation")], ignore_index=True)


def _train_v1(v1, end, stop, labels, threads):
    fit_start = perf_counter()
    prep = FraudPreprocessor()
    train = prep.fit_transform(v1.iloc[:end])
    model = make_models(labels[:end], threads)["xgboost"]
    model.fit(train, labels[:end])
    fit_seconds = perf_counter() - fit_start
    infer_start = perf_counter()
    scores = positive_scores(model, prep.transform(v1.iloc[end:stop]))
    return scores, fit_seconds, perf_counter() - infer_start, model.get_booster()


def _train_rich(cache, data, trial, end, stop, labels, threads):
    fit_start = perf_counter()
    view = MatrixView(data, {name: cache["features"].index(name) for name in cache["groups"][trial.features]},
                      cache["categorical"]).fit(end)
    settings = make_models(labels[:end], threads)["xgboost"]
    settings.set_params(**trial.overrides)
    params = settings.get_xgb_params()
    if not trial.weighted:
        params["scale_pos_weight"] = 1.0
    quantized = xgb.QuantileDMatrix(ViewIterator(view, labels, 0, end), max_bin=params["max_bin"],
                                   nthread=threads, enable_categorical=True)
    model = xgb.train(params, quantized, num_boost_round=settings.get_params()["n_estimators"])
    fit_seconds = perf_counter() - fit_start
    infer_start = perf_counter()
    scores = np.concatenate([model.inplace_predict(view.read(i, min(i + CHUNK_ROWS, stop)), validate_features=False)
                             for i in range(end, stop, CHUNK_ROWS)])
    return scores, fit_seconds, perf_counter() - infer_start, model


def run_trial(trial, output, cache, data, frame, v1_source, threads):
    folder = output / "trials" / trial.name
    folder.mkdir(parents=True, exist_ok=True)
    descriptor = {"trial": asdict(trial), "threads": threads, "cache_sha256": sha256(output / "features.json")}
    labels = frame[TARGET].to_numpy(dtype=np.uint8)
    v1 = None
    measurements = []
    for k, partition in enumerate(cache["folds"], 1):
        checkpoint = folder / f"fold{k}.json"
        end, stop = partition["train_end"], partition["validation_end"]
        if checkpoint.is_file():
            record = json.loads(checkpoint.read_bytes())
            if any(record[key] != value for key, value in descriptor.items()):
                raise ValueError(f"Checkpoint configuration differs for {trial.name} fold {k}.")
            if not (folder / f"scores{k}.npy").is_file() or not (folder / f"model{k}.ubj").is_file():
                raise ValueError(f"Checkpoint {trial.name} fold {k} is incomplete.")
        else:
            print(f"{trial.name} fold {k}: {end:,} training / {stop-end:,} validation", flush=True)
            if set(np.unique(labels[:end])) != {0, 1} or set(np.unique(labels[end:stop])) != {0, 1}:
                raise ValueError("Both classes must be represented within each fold.")
            if trial.features == "v1":
                if v1 is None:
                    v1 = _v1_frame(v1_source)
                    if not np.array_equal(v1[ID].to_numpy(), frame[ID].to_numpy()):
                        raise ValueError("V1 feature order differs from the development cache.")
                scores, fit, inference, model = _train_v1(v1, end, stop, labels, threads)
            else:
                scores, fit, inference, model = _train_rich(cache, data, trial, end, stop, labels, threads)
            metrics, _ = evaluate_scores(labels[end:stop], scores, scores >= .5)
            model.save_model(folder / f"model{k}.ubj")
            np.save(folder / f"scores{k}.npy", scores)
            record = {**descriptor, "fold": k, **partition, "validation_rows": stop-end,
                      "first_second": int(frame[TIME].iloc[end]), "last_second": int(frame[TIME].iloc[stop - 1]),
                      "fraud_rows": int(labels[end:stop].sum()), "threshold": .5, "metrics": metrics,
                      "budgets": review_budgets(labels[end:stop], scores), "fit_seconds": fit,
                      "inference_seconds": inference, "process_peak_working_set": _peak_memory()}
            write_json(checkpoint, record)
            print(f"{trial.name} fold {k}: AP {metrics['average_precision']:.5f}, AUC {metrics['roc_auc']:.5f}, {fit:.1f}s", flush=True)
        measurements.append(record)

    summary = {**descriptor, "folds": measurements, "fit_seconds": sum(f["fit_seconds"] for f in measurements),
               "metrics": {key: {"mean": float(np.mean(values)), "std": float(np.std(values))}
                           for key in ("average_precision", "roc_auc", "accuracy", "f1", "precision", "recall", "specificity")
                           if (values := [f["metrics"][key] for f in measurements]) and None not in values}}
    write_json(folder / "summary.json", summary)
    return summary


def _champions(output):
    root = output / "trials"
    baseline = root / "v1" / "summary.json"
    if not baseline.is_file():
        return
    def rank(trial, metric):
        return trial["metrics"][metric]["mean"]

    current = json.loads(baseline.read_bytes())
    accepted = ["v1"]
    trials = {"v1": current}
    for name in TRIALS:
        path = root / name / "summary.json"
        if name == "v1" or not path.is_file():
            continue
        entry = json.loads(path.read_bytes())
        trials[name] = entry
        fold_gains = sum(a["metrics"]["average_precision"] > b["metrics"]["average_precision"]
                         for a, b in zip(entry["folds"], current["folds"], strict=True))
        if (rank(entry, "average_precision") > rank(current, "average_precision") and fold_gains >= 2
                and rank(entry, "roc_auc") >= rank(current, "roc_auc") - .01):
            accepted.append(name)
            current = entry
    write_json(output / "champions.json", {"accepted": accepted, "champion": accepted[-1], "evaluated": list(trials)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True, help="V1 full artifact directory, read-only")
    parser.add_argument("--output", type=Path, default=Path("artifacts/v2"))
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--trials", nargs="+", choices=TRIALS, default=["rich", "v1"])
    args = parser.parse_args()
    output, source = args.output.resolve(), args.v1.resolve()
    if args.threads < 1 or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("Use a positive thread count and a separate V2 output path.")
    output.mkdir(parents=True, exist_ok=True)
    freeze = output / "protocol.json"
    protocol = Path(__file__).resolve().parents[2] / "docs" / "V2_PROTOCOL.md"
    digest = sha256(protocol)
    if freeze.exists():
        provenance = json.loads(freeze.read_bytes())
        if provenance["protocol_sha256"] != digest:
            raise ValueError("The frozen development protocol has changed.")
    else:
        resources = _resources(args.raw, output)
        provenance = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": digest,
                      "software": {name: version(name) for name in ("numpy", "pandas", "pyarrow", "xgboost", "scikit-learn")},
                      "resources": resources, "seed": 42, "test_opened": False}
        write_json(freeze, provenance)
    cache = build_cache(args.raw, source, output, provenance)
    print(f"Cached {cache['rows']:,} development rows and {len(cache['features'])} features; "
          f"{cache['policy_rows']:,} later rows reserved for validation policy design.", flush=True)
    data = np.load(output / "features.npy", mmap_mode="r")
    frame = pd.read_parquet(output / "development_metadata.parquet")
    for name in args.trials:
        summary = run_trial(TRIALS[name], output, cache, data, frame, source, args.threads)
        print(f"{name}: mean AP {summary['metrics']['average_precision']['mean']:.5f}", flush=True)
        _champions(output)


if __name__ == "__main__":
    main()
