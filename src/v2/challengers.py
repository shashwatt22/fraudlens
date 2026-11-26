"""Development-only boosting and bounded supervised challengers."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from src.data_loader import TARGET, TIME
from src.evaluate import evaluate_scores, positive_scores
from src.lab_benchmark import ScratchSettings, fit_network, stratified_positions
from src.scratch_neural import ScratchNetwork

from .advanced import build_advanced
from .data import CHUNK_ROWS, sha256, write_json
from .experiment import MatrixView, Trial, _peak_memory, _train_rich, review_budgets


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    options: dict = field(default_factory=dict)
    weighted: bool = False
    top_k: int = 96


BOOSTING = {
    name: Candidate(name, family, options, weighted)
    for name, family, options, weighted in (
        ("xgb_advanced", "xgb", {}, True),
        ("xgb_advanced_unweighted", "xgb", {}, False),
        ("xgb_advanced_depth8", "xgb", {"max_depth": 8, "n_estimators": 320, "learning_rate": .055, "reg_lambda": 4}, False),
        ("lgbm", "lgbm", {"num_leaves": 47, "learning_rate": .055, "rounds": 300, "min_data_in_leaf": 80}, False),
        ("lgbm_leafwise", "lgbm", {"num_leaves": 95, "learning_rate": .045, "rounds": 380,
                                  "min_data_in_leaf": 125, "lambda_l2": 8.0, "feature_fraction": .72}, True),
        ("catboost", "catboost", {"depth": 6, "iterations": 240, "learning_rate": .075}, False),
        ("catboost_regularized", "catboost", {"depth": 8, "iterations": 250, "learning_rate": .06, "l2_leaf_reg": 8}, True),
    )
}

SUBSET = {
    name: Candidate(name, family, options, weighted, k)
    for name, family, options, weighted, k in (
        ("dummy", "dummy", {}, False, 0),
        ("logistic", "logistic", {"C": 1.0}, True, 128),
        ("logistic_l2", "logistic", {"C": .08}, False, 192),
        ("gnb", "gnb", {"var_smoothing": 1e-9}, False, 32),
        ("gnb_smooth", "gnb", {"var_smoothing": .025}, True, 16),
        ("knn", "knn", {"n_neighbors": 15}, False, 24),
        ("knn_neighbors", "knn", {"n_neighbors": 45}, False, 12),
        ("svm", "svm", {"C": .4}, True, 96),
        ("svm_l2", "svm", {"C": .025}, False, 160),
        ("decision_tree", "tree", {"max_depth": 6, "min_samples_leaf": 30}, True, 96),
        ("decision_tree_depth", "tree", {"max_depth": 12, "min_samples_leaf": 15}, False, 160),
        ("random_forest", "forest", {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 4}, True, 128),
        ("random_forest_depth", "forest", {"n_estimators": 140, "max_depth": 22, "min_samples_leaf": 3}, False, 180),
        ("adaboost", "ada", {"depth": 1, "n_estimators": 80, "learning_rate": .5}, True, 96),
        ("adaboost_depth", "ada", {"depth": 2, "n_estimators": 100, "learning_rate": .16}, False, 128),
        ("numpy_linear", "numpy", {"hidden": 0, "learning_rate": .025}, True, 128),
        ("numpy_linear_l2", "numpy", {"hidden": 0, "learning_rate": .045, "l2": .004}, True, 192),
        ("numpy_relu", "numpy", {"hidden": 32, "learning_rate": .025}, True, 96),
        ("numpy_relu_wide", "numpy", {"hidden": 64, "learning_rate": .012, "l2": .003}, True, 192),
        ("torch_mlp", "torch", {"widths": (128, 64), "dropout": .25, "rate": .001}, True, 160),
        ("torch_mlp_regularized", "torch", {"widths": (192, 96, 48), "dropout": .18, "rate": .0007,
                                              "weight_decay": .002}, False, 224),
    )
}
TRAIN_LIMIT = 16_000


def class_weights(labels):
    negatives, positives = np.bincount(labels, minlength=2)
    return float(negatives / positives)


def _frequency_maps(matrix, categorical):
    result = {}
    for col in categorical:
        codes = matrix[:, col]
        counts = np.bincount(codes[np.isfinite(codes)].astype(np.int32))
        result[col] = np.log1p(counts).astype(np.float32)
    return result


def _encode_frequency(matrix, mappings):
    for column, values in mappings.items():
        codes = matrix[:, column]
        observed = np.isfinite(codes)
        known = observed & (codes < len(values))
        codes[known] = values[codes[known].astype(np.int32)]
        codes[observed & ~known] = 0
    return matrix


class DenseFeatures:
    """Fit frequency, supervised feature ranks, imputation and scale on training only."""

    def __init__(self, top_k, categorical, sparse_categories=False):
        self.top_k = top_k
        self.categorical = categorical
        self.sparse_categories = sparse_categories

    def _transform_raw(self, matrix):
        values = _encode_frequency(matrix.copy(), self.frequencies)
        return np.column_stack((self.imputer.transform(values), np.isnan(values).astype(np.float32)))

    def fit(self, matrix, labels):
        self.frequencies = _frequency_maps(matrix, self.categorical)
        encoded = _encode_frequency(matrix.copy(), self.frequencies)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(encoded)
        full = self._transform_raw(matrix)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with np.errstate(invalid="ignore", divide="ignore"):
                scores, _ = f_classif(full, labels)
        scores = np.nan_to_num(scores, nan=-np.inf)
        self.selected = np.argsort(-scores, kind="stable")[:self.top_k]
        self.scaler = StandardScaler().fit(full[:, self.selected])
        if self.sparse_categories:
            self.one_hot = OneHotEncoder(dtype=np.float32, min_frequency=7, max_categories=96,
                                         handle_unknown="infrequent_if_exist").fit(matrix[:, self.categorical])
        return self

    def transform(self, matrix):
        numeric = np.ascontiguousarray(self.scaler.transform(self._transform_raw(matrix)[:, self.selected]), dtype=np.float32)
        if self.sparse_categories:
            return hstack((csr_matrix(numeric), self.one_hot.transform(matrix[:, self.categorical])), format="csr")
        return numeric


def _estimator(candidate, threads):
    family, settings, weighted = candidate.family, candidate.options.copy(), candidate.weighted
    weight = "balanced" if weighted else None
    if family == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if family == "logistic":
        return LogisticRegression(**settings, class_weight=weight, max_iter=1000, solver="lbfgs")
    if family == "gnb":
        return GaussianNB(**settings)
    if family == "knn":
        return KNeighborsClassifier(**settings, algorithm="brute", weights="distance", n_jobs=threads)
    if family == "svm":
        return LinearSVC(**settings, class_weight=weight, dual=False, max_iter=3500, random_state=42)
    if family == "tree":
        return DecisionTreeClassifier(**settings, class_weight=weight, random_state=42)
    if family == "forest":
        return RandomForestClassifier(**settings, class_weight=weight, random_state=42, n_jobs=threads)
    if family == "ada":
        tree = DecisionTreeClassifier(max_depth=settings.pop("depth"), min_samples_leaf=8, random_state=42)
        return AdaBoostClassifier(estimator=tree, **settings, random_state=42)
    raise ValueError(f"Unknown model family: {family}")


def _torch_fit(matrix, labels, validation, validation_labels, candidate, threads):
    from copy import deepcopy

    from sklearn.metrics import average_precision_score
    import torch
    from torch import nn

    torch.manual_seed(42)
    torch.set_num_threads(threads)
    settings = candidate.options
    width = matrix.shape[1]
    layers = []
    for units in settings["widths"]:
        layers += [nn.Linear(width, units), nn.BatchNorm1d(units), nn.GELU(), nn.Dropout(settings["dropout"])]
        width = units
    model = nn.Sequential(*layers, nn.Linear(width, 1))
    weights = class_weights(labels) if candidate.weighted else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, dtype=torch.float32))
    optim = torch.optim.AdamW(model.parameters(), lr=settings["rate"], weight_decay=settings.get("weight_decay", 1e-4))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=20)
    inputs, targets = torch.from_numpy(matrix), torch.from_numpy(labels.astype(np.float32))
    validation_inputs = torch.from_numpy(validation)
    history, best, best_state, best_epoch = [], -np.inf, None, 0
    order = np.random.default_rng(42)
    for epoch in range(1, 21):
        model.train()
        running, total = 0., 0
        for batch in np.array_split(order.permutation(len(labels)), int(np.ceil(len(labels)/256))):
            optim.zero_grad(set_to_none=True)
            value = criterion(model(inputs[batch]).squeeze(1), targets[batch])
            value.backward()
            optim.step()
            running += value.item() * len(batch)
            total += len(batch)
        schedule.step()
        model.eval()
        with torch.inference_mode():
            scores = torch.sigmoid(model(validation_inputs).squeeze(1)).numpy()
        ap = float(average_precision_score(validation_labels, scores))
        history.append({"epoch": epoch, "train_loss": running / total, "internal_validation_ap": ap})
        if ap > best:
            best, best_state, best_epoch = ap, deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= 4:
            break
    model.load_state_dict(best_state)
    return model.eval(), {"epochs": history, "best_epoch": best_epoch}


def _subset_fit(candidate, view, labels, times, end, stop, threads):
    start = perf_counter()
    neural = candidate.family in {"numpy", "torch"}
    inner_end = int(np.searchsorted(times[:end], times[int(.90 * end) - 1], side="right")) if neural else end
    positions = stratified_positions(labels[:inner_end], TRAIN_LIMIT)
    # All outer validation rows have their natural prevalence. Sample only within training.
    train = view.data[np.ix_(positions, view.columns)]
    y = labels[positions]
    if candidate.family == "dummy":
        model = _estimator(candidate, threads).fit(np.zeros((len(y), 1)), y)
        scores = positive_scores(model, np.zeros((stop-end, 1)))
        return scores, perf_counter()-start, 0., {"model": model, "train_rows": len(y)}
    transform = DenseFeatures(candidate.top_k, view.categories, candidate.name in {"logistic_l2", "svm_l2"}).fit(train, y)
    encoded = transform.transform(train)
    payload = {"transform": transform, "train_rows": len(y)}
    if neural:
        validation = labels[inner_end:end]
        if len(np.unique(validation)) != 2:
            raise ValueError("Chronological neural early-stopping block needs both labels.")
        early_stop_data = transform.transform(view.data[inner_end:end, view.columns])
    if candidate.family == "numpy":
        settings = candidate.options.copy()
        model = ScratchNetwork(encoded.shape[1], settings.pop("hidden"))
        payload["history"] = fit_network(model, encoded, y.astype(np.float32), early_stop_data,
                                         validation.astype(np.float32), ScratchSettings(max_epochs=20, **settings), candidate.name)
    elif candidate.family == "torch":
        model, payload["history"] = _torch_fit(encoded, y, early_stop_data, validation, candidate, threads)
    else:
        model = _estimator(candidate, threads)
        options = {}
        if candidate.weighted and candidate.family in {"ada", "gnb"}:
            options["sample_weight"] = np.where(y == 1, class_weights(y), 1)
        model.fit(encoded, y, **options)
    fit = perf_counter()-start
    infer = perf_counter()
    outputs = []
    for low in range(end, stop, 2048):
        val = transform.transform(view.data[low:min(stop, low+2048), view.columns])
        if not neural:
            scores = model.decision_function(val) if candidate.family == "svm" else positive_scores(model, val)
        elif candidate.family == "numpy":
            scores = model.predict_scores(val)
        else:
            import torch
            with torch.inference_mode():
                scores = torch.sigmoid(model(torch.from_numpy(val)).squeeze(1)).numpy()
        outputs.append(scores)
    payload["model"] = model
    return np.concatenate(outputs), fit, perf_counter()-infer, payload


def _fit_booster(candidate, data, cache, view, labels, times, end, stop, threads):
    if candidate.family == "xgb":
        trial = Trial(candidate.name, "advanced", candidate.options, candidate.weighted)
        scores, fit, inference, model = _train_rich(cache, data, trial, end, stop, labels, threads)
        return scores, fit, inference, {"model": model, "train_rows": end}
    start = perf_counter()
    options = candidate.options.copy()
    inner_end = int(np.searchsorted(times[:end], times[int(.90 * end) - 1], side="right"))
    internal = labels[inner_end:end]
    if len(np.unique(internal)) != 2:
        raise ValueError("Chronological booster early-stopping block needs both labels.")
    view.fit(inner_end)
    if candidate.family == "lgbm":
        import lightgbm as lgb

        matrix = view.read(0, end)
        weight = class_weights(labels[:inner_end]) if candidate.weighted else 1.
        rounds = options.pop("rounds")
        params = {"objective": "binary", "metric": "average_precision", "verbosity": -1, "num_threads": threads, "seed": 42,
                  "scale_pos_weight": weight, "max_bin": 127, "deterministic": True, "force_col_wise": True,
                  "bagging_fraction": .85, "bagging_freq": 1, **options}
        training = lgb.Dataset(matrix[:inner_end], label=labels[:inner_end], categorical_feature=view.categories, free_raw_data=True)
        validation = training.create_valid(matrix[inner_end:], label=internal)
        model = lgb.train(params, training, valid_sets=[validation], num_boost_round=rounds,
                          callbacks=[lgb.early_stopping(35, first_metric_only=True, verbose=False)], keep_training_booster=False)

        def score(low, high):
            return model.predict(view.read(low, high), num_threads=threads)

        positions = range(inner_end)
        mappings = None
    else:
        from catboost import CatBoostClassifier

        positions = stratified_positions(labels[:inner_end], 100_000)
        matrix = view.data[np.ix_(positions, view.columns)]
        mappings = _frequency_maps(matrix, view.categories)
        matrix = _encode_frequency(matrix, mappings)
        model = CatBoostClassifier(**options, eval_metric="PRAUC", thread_count=threads, random_seed=42, verbose=False,
                                   allow_writing_files=False, rsm=.75, scale_pos_weight=(class_weights(labels[positions])
                                   if candidate.weighted else 1.), border_count=64)
        stopping_data = _encode_frequency(view.data[inner_end:end, view.columns], mappings)
        model.fit(matrix, labels[positions], eval_set=(stopping_data, internal), early_stopping_rounds=35, use_best_model=True)

        def score(low, high):
            return positive_scores(model, _encode_frequency(view.data[low:high, view.columns], mappings))

    del matrix
    fit = perf_counter() - start
    infer = perf_counter()
    scores = np.concatenate([score(i, min(i+CHUNK_ROWS, stop)) for i in range(end, stop, CHUNK_ROWS)])
    return scores, fit, perf_counter() - infer, {"model": model, "frequency_maps": mappings, "train_rows": len(positions)}


def run(candidate, data, cache, frame, output, threads):
    folder = output / "challengers" / candidate.name
    folder.mkdir(parents=True, exist_ok=True)
    # Round-trip JSON so that tuple-shaped settings compare correctly on resume.
    provenance = json.loads(json.dumps({"candidate": asdict(candidate), "feature_sha256": cache["advanced_sha256"],
                                        "threads": threads}))
    labels = frame[TARGET].to_numpy(dtype=np.uint8)
    times = frame[TIME].to_numpy(dtype=np.int64)
    metrics = []
    for k, fold in enumerate(cache["folds"], 1):
        end, stop = fold["train_end"], fold["validation_end"]
        config_path = folder / f"fold{k}.json"
        output_path = folder / f"scores{k}.npy"
        model_path = folder / f"model{k}.joblib"
        if config_path.is_file():
            record = json.loads(config_path.read_bytes())
            if any(record[field] != value for field, value in provenance.items()) or not output_path.is_file() or not model_path.is_file():
                raise ValueError(f"Conflicting or incomplete checkpoint for {candidate.name} fold {k}.")
        else:
            print(f"{candidate.name} fold {k}: {stop-end:,} validation rows", flush=True)
            view = MatrixView(data, dict(zip(cache["features"], range(len(cache["features"])))), cache["categorical"]).fit(end)
            if candidate.name in BOOSTING:
                scores, fit, infer, payload = _fit_booster(candidate, data, cache, view, labels, times, end, stop, threads)
            else:
                scores, fit, infer, payload = _subset_fit(candidate, view, labels, times, end, stop, threads)
            threshold = 0. if candidate.family == "svm" else .5
            evaluation, _ = evaluate_scores(labels[end:stop], scores, scores >= threshold)
            record = {**provenance, **fold, "fit_rows": payload["train_rows"], "fold": k, "threshold": threshold,
                      "fit_seconds": fit, "inference_seconds": infer, "process_peak_working_set": _peak_memory(),
                      "metrics": evaluation, "budgets": review_budgets(labels[end:stop], scores)}
            if "history" in payload:
                record["learning_history"] = payload["history"]
            joblib.dump(payload, model_path, compress=1)
            np.save(output_path, scores)
            write_json(config_path, record)
            print(f"{candidate.name} fold {k}: AP {evaluation['average_precision']:.5f}, fit {fit:.1f}s, score {infer:.1f}s",
                  flush=True)
        metrics.append(record)
    means = {name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
             for name, value in metrics[0]["metrics"].items() if value is not None and isinstance(value, (float, int))
             if None not in (values := [fold["metrics"][name] for fold in metrics])}
    write_json(folder / "summary.json", {**provenance, "folds": metrics, "metrics": means})


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/v2"))
    parser.add_argument("--threads", type=int, default=6)
    all_candidates = BOOSTING | SUBSET
    parser.add_argument("--candidates", nargs="+", choices=all_candidates, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    cache = json.loads((output / "features.json").read_bytes())
    frame = pd.read_parquet(output / "development_metadata.parquet")
    if frame[TIME].max() >= cache["holdout_start"] or sha256(args.raw / "train_transaction.csv") != cache["raw_sha256"]["train_transaction.csv"]:
        raise ValueError("Development cache chronology or raw transaction fingerprint changed.")
    if sha256(args.raw / "train_identity.csv") != cache["raw_sha256"]["train_identity.csv"]:
        raise ValueError("Raw identity fingerprint changed.")
    data = np.load(output / "features.npy", mmap_mode="r")
    advanced = build_advanced(output, data, cache, frame, args.raw)
    del data
    data = np.load(output / "features_advanced.npy", mmap_mode="r")
    cache["features"] = advanced["features"]
    cache["groups"]["advanced"] = advanced["features"]
    cache["advanced_sha256"] = sha256(output / "advanced.json")
    with threadpool_limits(limits=args.threads):
        for name in args.candidates:
            run(all_candidates[name], data, cache, frame, output, args.threads)


if __name__ == "__main__":
    # Run through the imported module so fitted transformers pickle by package name.
    from src.v2.challengers import main as run_cli

    run_cli()
