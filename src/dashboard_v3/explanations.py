"""Read V3 behavioral inputs and verify exact native TreeSHAP against saved scores."""

from threading import Lock

import numpy as np
from scipy.special import expit
import streamlit as st

from src.data_loader import ID, TIME
from .store import THRESHOLD, DECISION_SHA, FIT_ROWS, MODEL_SHA, ROWS, SCORE, DataUnavailable, report, score_manifest, signature, verify

HISTORY = {
    "card_address_anchor_prior_count": "Prior transactions",
    "card_address_anchor_prior_count_1h": "Prior hour",
    "card_address_anchor_prior_count_24h": "Prior 24 hours",
    "card_address_anchor_prior_mean": "Prior mean amount",
    "card_address_anchor_prior_std": "Prior amount standard deviation",
    "card_address_anchor_prior_gap": "Seconds since prior transaction",
    "card_address_anchor_amount_deviation": "Amount above prior mean",
    "card_address_anchor_amount_ratio": "Amount / prior mean",
}
FEATURE_LABELS = {**HISTORY, "TransactionAmt": "Transaction amount", "ProductCD": "Product code",
                  "day_minus_D1": "D1 anchor day", "identity_missing": "Identity record missing"}


@st.cache_resource(show_spinner=False, max_entries=1)
def _features(path, version, names):
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (590_540, 497) or matrix.dtype != np.float32 or len(names) != 497:
        raise DataUnavailable("The frozen feature matrix has an unexpected shape or type.")
    return matrix


def inputs(paths):
    score = score_manifest(paths)
    folder = paths.retrospective / "covariates"
    advanced = report(folder / "advanced.json", score["feature_manifest_sha256"])
    cache = report(folder / "features.json", advanced["features_sha256"])
    path = folder / "features_advanced.npy"
    version = verify(path, score["matrix_sha256"])
    names = tuple(advanced["features"])
    return _features(path, version, names), names, tuple(cache["categorical"]), version


def history(paths, position):
    matrix, names, _, _ = inputs(paths)
    row = matrix[len(matrix) - ROWS + position]
    values = {label: float(row[names.index(field)]) for field, label in HISTORY.items()}
    missing = any(not np.isfinite(row[names.index(field)]) for field in ("card1", "addr1", "day_minus_D1"))
    return missing, values


class Explainer:
    """One native booster and the exact category treatment used in Step 7C."""

    def __init__(self, model_path, source, names, categorical, fit_rows):
        import xgboost as xgb
        from src.v2.experiment import MatrixView

        self.model = xgb.Booster(params={"nthread": 4})
        self.model.load_model(model_path)
        if self.model.feature_names != list(names) or self.model.num_boosted_rounds() != 320:
            raise DataUnavailable("The loaded checkpoint does not match frozen V3 model inputs.")
        self.view = MatrixView(source, dict(zip(names, range(len(names)))), categorical).fit(fit_rows)
        self.names = names
        self.offset = len(source) - ROWS
        self.lock = Lock()

    def score(self, start, stop):
        """Run native inference with the frozen training-prefix category boundaries."""
        data = self.view.read(self.offset + start, self.offset + stop)
        with self.lock:
            return self.model.inplace_predict(data, validate_features=False)

    def contributions(self, positions, reported):
        import xgboost as xgb

        data = np.concatenate([self.view.read(self.offset + pos, self.offset + pos + 1) for pos in positions])
        batch = xgb.DMatrix(data, feature_names=list(self.names), feature_types=self.view.types,
                            enable_categorical=True, nthread=4)
        with self.lock:
            values = self.model.predict(batch, pred_contribs=True)
            margin = self.model.predict(batch, output_margin=True)
        margin_errors = np.abs(values.sum(axis=1, dtype=np.float64) - margin)
        score_errors = np.abs(expit(margin.astype(np.float64)) - reported)
        if (not np.isfinite(values).all() or not np.isfinite(margin).all()
                or margin_errors.max() > 1e-4 or score_errors.max() > 2e-6):
            raise DataUnavailable("TreeSHAP contributions did not reconstruct the matching frozen score.")
        return values, margin, float(margin_errors.max()), float(score_errors.max())


@st.cache_resource(show_spinner=False, max_entries=1)
def _explainer(paths, model_path, versions, fit_rows):
    source, names, categorical, _ = inputs(paths)
    return Explainer(model_path, source, names, categorical, fit_rows)


def explainer(paths):
    policy = report(paths.policy / "decision_freeze.json", DECISION_SHA)
    model = report(paths.policy / "model_freeze.json", MODEL_SHA)["models"]["xgboost"]
    if (model["fit_rows"] != FIT_ROWS or policy["calibrator"]["method"] != "identity"
            or policy["thresholds"]["xgboost"]["selected_f1"] != THRESHOLD):
        raise DataUnavailable("The checkpoint requires the frozen V3 fit and identity calibration.")
    model_path = paths.refit / "xgboost" / "model.ubj"
    versions = (verify(model_path, model["artifact_sha256"]["xgboost/model.ubj"]), inputs(paths)[3])
    return _explainer(paths, model_path, versions, FIT_ROWS)


@st.cache_data(show_spinner=False, max_entries=24)
def _local(paths, version, position, expected):
    engine = explainer(paths)
    all_values, margins, margin_error, score_error = engine.contributions([position], np.array([expected]))
    values = all_values[0]
    indexes = np.argsort(-np.abs(values[:-1]), kind="stable")[:12]
    return {"names": [engine.names[i] for i in indexes], "values": values[indexes].tolist(),
            "base": float(values[-1]), "remaining": float(values[:-1].sum(dtype=np.float64) - values[indexes].sum(dtype=np.float64)),
            "margin": float(margins[0]), "margin_error": margin_error, "score_error": score_error}


def local(paths, position, expected):
    path = paths.refit / "xgboost" / "model.ubj"
    return _local(paths, (signature(path), inputs(paths)[3]), position, expected)


@st.cache_data(show_spinner=False, max_entries=2)
def _global(paths, versions, positions, scores):
    engine = explainer(paths)
    values, _, margin_error, score_error = engine.contributions(positions, np.array(scores))
    means = np.abs(values[:, :-1]).mean(axis=0)
    order = np.argsort(-means, kind="stable")[:14]
    return {"names": [engine.names[i] for i in order], "values": means[order].tolist(),
            "samples": len(positions), "margin_error": margin_error, "score_error": score_error}


def global_summary(paths, frame):
    positions = tuple(np.linspace(0, len(frame) - 1, 24, dtype=int).tolist())
    scores = tuple(frame[SCORE].iloc[list(positions)].astype(float))
    versions = (signature(paths.refit / "xgboost" / "model.ubj"), inputs(paths)[3])
    return _global(paths, versions, positions, scores)
