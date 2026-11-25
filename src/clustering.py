"""Bounded, training-only DBSCAN and PCA visualization."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .data_loader import AMOUNT, ID, TARGET

SEED = 42
FEATURES = ("log_amount", "prior_txn_count", "amount_to_prior_mean", "seconds_since_prior_txn", "prior_count_24h")
SCALED_FEATURES = (FEATURES[0], *(f"log1p_{name}" for name in FEATURES[1:]), "incomplete_history")
EPS_VALUES = (0.45, 0.65, 0.85, 1.05)
MIN_SAMPLES = (10, 20)
MAX_NEIGHBOR_LINKS = 8_000_000
SILHOUETTE_CAP = 1_500


class ClusteringPreprocessor:
    """Compress skew, impute observed training medians and standardize each column."""

    @staticmethod
    def _matrix(frame: pd.DataFrame) -> np.ndarray:
        values = frame[list(FEATURES)].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        incomplete = np.isnan(values[:, 2:4]).any(axis=1)
        values[:, 1:] = np.log1p(values[:, 1:])
        return np.column_stack((values, incomplete))

    def fit(self, training_sample: pd.DataFrame):
        values = self._matrix(training_sample)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler_ = StandardScaler().fit(self.imputer_.fit_transform(values))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.scaler_.transform(self.imputer_.transform(self._matrix(frame)))


def sample_positions(rows: int, limit: int, seed: int = SEED) -> np.ndarray:
    """Choose by position without labels and retain original chronological order."""
    if limit < 1 or rows < 1:
        raise ValueError("Sampling requires positive row and limit counts.")
    return np.sort(np.random.default_rng(seed).choice(rows, size=min(rows, limit), replace=False))


def cluster_metrics(matrix: np.ndarray, labels: np.ndarray) -> dict:
    """Summarize DBSCAN labels. Silhouette is computed on sampled non-noise rows."""
    inliers = np.flatnonzero(labels != -1)
    counts = np.unique(labels[inliers], return_counts=True)[1]
    silhouette, assessed = None, 0
    if len(counts) >= 2:
        selected = inliers[sample_positions(len(inliers), SILHOUETTE_CAP)]
        selected_labels = labels[selected]
        if 2 <= np.unique(selected_labels).size < len(selected):
            silhouette = float(silhouette_score(matrix[selected], selected_labels))
            assessed = len(selected)
    return {"clusters": len(counts), "clustered": len(inliers), "noise": len(labels) - len(inliers),
            "noise_fraction": (len(labels) - len(inliers)) / len(labels),
            "largest_cluster_share": float(counts.max() / len(inliers)) if len(counts) else None,
            "silhouette": silhouette, "silhouette_rows": assessed}


def choose_configuration(reports: list[dict]) -> int:
    completed = [index for index, report in enumerate(reports) if report["status"] == "completed"]
    if not completed:
        raise ValueError("Every DBSCAN radius exceeded the neighborhood budget. Reduce the training sample before retrying.")
    valid = [index for index in completed if reports[index]["silhouette"] is not None]
    balanced = [index for index in valid if reports[index]["noise_fraction"] <= .55
                and reports[index]["largest_cluster_share"] <= .90]
    candidates = balanced or valid
    if candidates:
        return max(candidates, key=lambda index: (reports[index]["silhouette"], -reports[index]["noise_fraction"]))
    return min(completed, key=lambda index: reports[index]["noise_fraction"])


def describe_clusters(sample: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Summarize the selected groups using raw behavior and then observed fraud labels."""
    frame = sample.assign(cluster=labels, no_prior_history=sample["prior_txn_count"].eq(0))
    profiles = frame.groupby("cluster", sort=True).agg(
        rows=(ID, "size"), fraud_rows=(TARGET, "sum"), fraud_rate=(TARGET, "mean"),
        median_amount=(AMOUNT, "median"), median_prior_count=("prior_txn_count", "median"),
        median_amount_ratio=("amount_to_prior_mean", "median"), median_gap_seconds=("seconds_since_prior_txn", "median"),
        median_prior_24h=("prior_count_24h", "median"), cold_start_share=("no_prior_history", "mean"),
    )
    return profiles.reset_index()


@dataclass
class ClusterRun:
    assignments: pd.DataFrame
    profiles: pd.DataFrame
    comparisons: list[dict]
    selected: dict
    preprocessor: ClusteringPreprocessor
    pca: PCA


def run_clustering(training: pd.DataFrame, limit: int, threads: int) -> ClusterRun:
    selected_rows = sample_positions(len(training), limit)
    if len(selected_rows) < 2:
        raise ValueError("DBSCAN and two-dimensional PCA need at least two sampled training rows.")
    sample = training.iloc[selected_rows]
    preprocessor = ClusteringPreprocessor().fit(sample)
    matrix = preprocessor.transform(sample)
    tree = cKDTree(matrix)
    reports, predictions = [], []
    for eps in EPS_VALUES:
        links = int(tree.query_ball_point(matrix, eps, return_length=True, workers=threads).sum())
        for minimum in MIN_SAMPLES:
            report = {"eps": eps, "min_samples": minimum, "neighbor_links": links}
            if links > MAX_NEIGHBOR_LINKS:
                reports.append({**report, "status": "skipped_dense"})
                predictions.append(None)
                continue
            start = perf_counter()
            labels = DBSCAN(eps=eps, min_samples=minimum, algorithm="kd_tree", n_jobs=threads).fit_predict(matrix)
            reports.append({**report, "status": "completed", **cluster_metrics(matrix, labels),
                            "seconds": perf_counter() - start})
            predictions.append(labels)

    selected = choose_configuration(reports)
    labels = predictions[selected]
    pca = PCA(n_components=2, svd_solver="full")
    # A constant sample has finite projections and undefined explained variance ratios.
    with np.errstate(invalid="ignore"):
        points = pca.fit_transform(matrix)
    assignments = sample[[ID]].assign(cluster=labels, pca_1=points[:, 0], pca_2=points[:, 1]).reset_index(drop=True)
    return ClusterRun(assignments, describe_clusters(sample, labels), reports, reports[selected], preprocessor, pca)
