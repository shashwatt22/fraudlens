"""Matched sample preparation, supplementary estimators and checkpoint selection."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .evaluate import positive_scores
from .scratch_neural import positive_weight
from .train import CORE, SEED, _model_parameters, make_models

TRAIN_LIMIT, EVAL_LIMIT = 20_000, 4_000
DENSE_MODELS = {"gaussian_nb", "knn"}
KNN_BATCH = 512
BENCHMARK_MODELS = (*CORE, "gaussian_nb", "knn", "linear_svm", "adaboost")
IMBALANCE = {
    "dummy": "Native majority rule", "logistic_regression": "Balanced class weights", "random_forest": "Balanced class weights",
    "xgboost": "Training negative/positive ratio as scale_pos_weight", "gaussian_nb": "Empirical class priors; no resampling",
    "knn": "Inverse distance per neighbor; no class-wide weight", "linear_svm": "Balanced class weights; raw decision margins",
    "adaboost": "Initial sample weights use training negative/positive ratio for positives",
    "sklearn_objective_reference": "Same positive sample weight and L2 objective as NumPy linear, optimized by L-BFGS",
}


def stratified_positions(labels: np.ndarray, limit: int, seed: int = SEED) -> np.ndarray:
    """Sample without replacement inside one partition, then restore source order."""
    if limit < 2 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("A benchmark sample requires at least two rows and both classes.")
    if len(labels) <= limit:
        return np.arange(len(labels))
    positions, _ = train_test_split(np.arange(len(labels)), train_size=limit, stratify=labels, random_state=seed)
    positions.sort()
    if len(np.unique(labels[positions])) != 2:
        raise ValueError("The stratified sample needs both classes. Increase its limit.")
    return positions


@dataclass(frozen=True)
class ScratchSettings:
    batch_size: int = 512
    learning_rate: float = .03
    l2: float = 1e-4
    max_epochs: int = 30
    patience: int = 6
    seed: int = SEED


def build_models(labels: np.ndarray, threads: int, l2: float) -> dict:
    baseline = make_models(labels, threads)
    baseline.pop("xgboost_basic")
    return {**baseline,
            "gaussian_nb": GaussianNB(var_smoothing=1e-9),
            "knn": KNeighborsClassifier(n_neighbors=min(15, len(labels)), algorithm="brute", metric="euclidean",
                                        weights="distance", n_jobs=threads),
            "linear_svm": LinearSVC(C=1., dual=False, class_weight="balanced", max_iter=5_000, random_state=SEED),
            "adaboost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=SEED),
                                           n_estimators=80, learning_rate=.5, random_state=SEED),
            "sklearn_objective_reference": LogisticRegression(C=1 / (len(labels) * l2), solver="lbfgs", max_iter=1_000,
                                                             tol=1e-7, random_state=SEED)}


def fitted_parameters(model) -> dict:
    settings = _model_parameters(model)
    if "estimator" in settings:
        settings["estimator"] = {"type": type(model.estimator).__name__, "parameters": model.estimator.get_params()}
    return settings


def classifier_scores(name: str, model, matrix) -> np.ndarray:
    if name == "linear_svm":
        scores = model.decision_function(matrix)
    elif name == "knn":
        scores = np.concatenate([positive_scores(model, matrix[start:start + KNN_BATCH])
                                 for start in range(0, len(matrix), KNN_BATCH)])
    else:
        scores = positive_scores(model, matrix)
    if scores.shape != (matrix.shape[0],) or not np.isfinite(scores).all():
        raise ValueError(f"{name} did not return one finite score per row.")
    return scores


def fit_network(model, train, labels, validation, validation_labels, settings: ScratchSettings, name: str) -> dict:
    """Use the same batch order, objective and validation checkpoint rule for both engines."""
    rng = np.random.default_rng(settings.seed)
    weight = positive_weight(labels)
    history, best_epoch, best_ap, best = [], 0, -np.inf, None
    started = perf_counter()
    for epoch in range(1, settings.max_epochs + 1):
        epoch_start = perf_counter()
        order = rng.permutation(len(labels))
        for start in range(0, len(labels), settings.batch_size):
            batch = order[start:start + settings.batch_size]
            model.update(train[batch], labels[batch], weight, settings.l2, settings.learning_rate)
        loss = model.objective(train, labels, weight, settings.l2)
        val_loss = model.objective(validation, validation_labels, weight, settings.l2)
        scores = model.predict_scores(validation)
        if not np.isfinite([loss, val_loss]).all() or not np.isfinite(scores).all():
            raise ValueError(f"Nonfinite output while fitting {name} at epoch {epoch}.")
        ap = float(average_precision_score(validation_labels, scores))
        if ap > best_ap:
            best_epoch, best_ap, best = epoch, ap, model.snapshot()
        history.append({"epoch": epoch, "train_loss": float(loss), "validation_loss": float(val_loss),
                        "validation_ap": ap, "seconds": perf_counter() - epoch_start})
        if epoch - best_epoch >= settings.patience:
            break
    model.restore(best)
    print(f"{name}: epoch {best_epoch}/{len(history)}, validation AP {best_ap:.6f}", flush=True)
    return {"epochs": history, "epochs_completed": len(history), "best_epoch": best_epoch, "best_validation_ap": best_ap,
            "fit_seconds": perf_counter() - started, "parameter_count": model.parameter_count, "positive_weight": weight,
            "hidden_width": model.hidden_width, "stop_reason": "patience" if len(history) < settings.max_epochs else "epoch_limit"}
