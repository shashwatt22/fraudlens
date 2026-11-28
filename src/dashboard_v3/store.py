"""Read verified offline artifacts without fitting or rewriting anything."""

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import AMOUNT, ID, PRODUCT, TARGET, TIME

THRESHOLD = 0.19381101429462433
ROWS, FIT_ROWS = 118_108, 425_188
MODELS = ("xgboost", "logistic", "forest", "torch")
NAMES = {
    "xgboost": "XGBoost · depth 10", "logistic": "Logistic Regression",
    "forest": "Random Forest", "torch": "PyTorch · 224-input MLP",
    "v1_xgboost": "V1 XGBoost", "v2_ensemble": "V2 weighted ensemble",
}
BUNDLE_SHA = "9812d81f3538f231bd607395ac9592f31c9d78d85f8eb8ba38be8ce1241dc7b9"
SCORE_SHA = "c9044671236453f20d0b874092483604db7f115e65be07537680128b34f4e2d4"
V1_SHA = "302e1755897bc4ad3ddda4baaa0f5651976202e67ed6e38034e1707a95616a6b"
MODEL_SHA = "a11e926442fabcf6f1166e61b4600b7d191d2d2b5dc3b495b875133336c5b988"
DECISION_SHA = "138b12eac5773e4bbd79eec064f4dc3d5cafc9386cda2742673f4b4463579edf"
SCORE, DECISION = "xgboost_score", "xgboost_decision"


class DataUnavailable(Exception):
    """A section cannot safely display the requested artifact."""


@dataclass(frozen=True)
class Paths:
    v3: Path
    legacy: Path
    v2: Path

    @classmethod
    def from_env(cls, project):
        return cls(
            Path(os.environ.get("FRAUDLENS_V3_ARTIFACTS", project / "artifacts" / "v3")).resolve(),
            Path(os.environ.get("FRAUDLENS_ARTIFACTS", r"E:\fraudlens\artifacts")).resolve(),
            Path(os.environ.get("FRAUDLENS_V2_ARTIFACTS", r"E:\fraudlens-v2\artifacts\v2")).resolve(),
        )

    @property
    def retrospective(self):
        return self.v3 / "step7c-retrospective"

    @property
    def refit(self):
        return self.v3 / "step7a-refit"

    @property
    def policy(self):
        return self.v3 / "step7b-policy"


def signature(path):
    try:
        stat = path.stat()
    except OSError as error:
        raise DataUnavailable(f"{path.name} is unavailable. Check the local artifact paths.") from error
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@st.cache_data(show_spinner=False, max_entries=28)
def _verify(path, digest, version):
    try:
        with path.open("rb") as file:
            actual = sha256()
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                actual.update(chunk)
    except OSError as error:
        raise DataUnavailable(f"{path.name} could not be read.") from error
    if actual.hexdigest() != digest:
        raise DataUnavailable(f"{path.name} failed its frozen artifact check.")


def verify(path, digest):
    version = signature(path)
    _verify(path, digest, version)
    return version


@st.cache_data(show_spinner=False, max_entries=30)
def _json(path, version):
    try:
        value = json.loads(path.read_bytes())
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except (OSError, ValueError) as error:
        raise DataUnavailable(f"{path.name} could not be read.") from error


def report(path, digest=None):
    return _json(path, verify(path, digest) if digest else signature(path))


def bundle(paths):
    pack = report(paths.retrospective / "dashboard.json", BUNDLE_SHA)
    if (pack["schema_version"] != 1 or pack["scope"] != "reused retrospective holdout"
            or pack["primary"] != "xgboost" or pack["calibration"] != "identity"
            or pack["population"]["rows"] != ROWS
            or pack["models"]["xgboost"]["frozen"]["threshold"] != THRESHOLD):
        raise DataUnavailable("The saved results do not match the V3 release.")
    return pack


def score_manifest(paths):
    return report(paths.retrospective / "score_manifest.json", SCORE_SHA)


def source_manifest(paths):
    return report(paths.legacy / "full" / "manifest.json", V1_SHA)


def source_partition(paths, partition):
    base = paths.legacy / "full"
    path = (base / source_manifest(paths)["files"][partition]).resolve()
    if not path.is_relative_to(base.resolve()):
        raise DataUnavailable("Invalid chronological partition path.")
    return path


def policy_plan(paths):
    return report(paths.refit / "refit_plan.json", score_manifest(paths)["frozen_hashes"]["refit_plan.json"])


@st.cache_data(show_spinner=False, max_entries=1)
def _queue(predictions, covariates, versions, expected_flagged):
    try:
        scores = pd.read_parquet(predictions, columns=[ID, TIME, SCORE, DECISION])
        fields = pd.read_parquet(covariates, columns=[ID, TIME, AMOUNT, PRODUCT, "identity_missing"])
    except (OSError, ValueError, KeyError) as error:
        raise DataUnavailable("Local prediction and transaction tables could not be read.") from error
    if (len(scores) != ROWS or not scores[ID].is_unique or not scores[[ID, TIME]].equals(fields[[ID, TIME]])
            or not scores[SCORE].between(0, 1).all()
            or not np.array_equal(scores[SCORE].to_numpy(dtype=np.float64) >= THRESHOLD, scores[DECISION])
            or scores[DECISION].sum() != expected_flagged):
        raise DataUnavailable("Transaction rows or frozen decisions failed the release check.")
    return scores.assign(**{col: fields[col].to_numpy() for col in (AMOUNT, PRODUCT, "identity_missing")})


def queue(paths, pack):
    predicted = paths.retrospective / "predictions.parquet"
    covariates = source_partition(paths, "test")
    versions = (verify(predicted, score_manifest(paths)["predictions_sha256"]), signature(covariates))
    return _queue(predicted, covariates, versions, pack["models"]["xgboost"]["frozen"]["reviewed"])


def transaction_id(text):
    query = text.strip()
    if not re.fullmatch(r"[0-9]{1,10}", query) or int(query) > np.iinfo(np.int32).max:
        raise ValueError("Enter a numeric transaction ID.")
    return int(query)


def filter_queue(frame, *, query="", decision="Flagged", product="All products", minimum=0.0, order="Highest risk"):
    keep = frame[SCORE].ge(minimum)
    if query.strip():
        value = transaction_id(query)
        return frame.loc[frame[ID].eq(value)]
    if decision != "All decisions":
        keep &= frame[DECISION].eq(decision == "Flagged")
    if product != "All products":
        keep &= frame[PRODUCT].eq(product)
    sort_column, ascending = {
        "Highest risk": (SCORE, False), "Largest amount": (AMOUNT, False),
        "Newest": (TIME, False), "Oldest": (TIME, True),
    }[order]
    return frame.loc[keep].sort_values([sort_column, ID], ascending=[ascending, True])


@st.cache_data(show_spinner=False, max_entries=8)
def _label(path, version, tx_id, timestamp):
    result = pd.read_parquet(path, columns=[TIME, TARGET], filters=[(ID, "=", tx_id)])
    if len(result) != 1 or result[TIME].iloc[0] != timestamp or result[TARGET].iloc[0] not in (0, 1):
        raise DataUnavailable("The observed research label could not be aligned.")
    return int(result[TARGET].iloc[0])


def observed_label(paths, tx_id, timestamp):
    path = source_partition(paths, "test")
    return _label(path, signature(path), tx_id, timestamp)


def threshold_preview(labels, scores, threshold):
    """Summarize a user-selected cutoff for the development sandbox only."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    flagged = scores >= threshold
    tp = int(np.count_nonzero(flagged & (labels == 1)))
    total = int(flagged.sum())
    return {"reviewed": total, "precision": tp / total if total else None,
            "recall": tp / labels.sum() if labels.sum() else None, "rows": len(labels)}


@st.cache_data(show_spinner=False, max_entries=2)
def _policy_verification(source, scores_path, versions, policy_start, policy_end, lo, hi):
    labels = pd.read_parquet(source, columns=[TARGET])[TARGET].to_numpy()[lo:hi]
    scores = np.load(scores_path, allow_pickle=False)
    if len(scores) != policy_end - policy_start or len(labels) != hi - lo:
        raise DataUnavailable("The saved policy verification data could not be aligned.")
    return labels, scores[lo - policy_start:hi - policy_start]


def development_preview(paths, cutoff):
    plan = policy_plan(paths)["partitions"]
    policy, later = plan["policy"], plan["later_policy"]
    source = paths.v2 / "development_metadata.parquet"
    model = report(paths.policy / "model_freeze.json", MODEL_SHA)
    scores = paths.refit / "policy_scores" / "xgboost.npy"
    versions = (verify(source, model["development_metadata_sha256"]),
                verify(scores, model["models"]["xgboost"]["score_sha256"]))
    labels, values = _policy_verification(source, scores, versions, policy["start"], policy["end"],
                                          later["start"], later["end"])
    return threshold_preview(labels, values, cutoff)
