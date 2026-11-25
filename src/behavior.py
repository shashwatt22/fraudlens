"""Training-only Isolation Forest features and fixed anomaly percentiles."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from .features import HISTORICAL_COLUMNS

ANOMALY_FEATURES = ["log_amount", *HISTORICAL_COLUMNS]
MISSING_FLAGS = ["prior_mean_amount", "amount_to_prior_mean", "seconds_since_prior_txn", "is_new_product"]
FIT_ROW_CAP = 60_000
SEED = 42


class AnomalyPreprocessor:
    """Impute training medians and retain explicit missing-history flags."""

    def _numeric(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[ANOMALY_FEATURES].to_numpy(dtype=np.float32, na_value=np.nan)

    def fit(self, train: pd.DataFrame):
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(self._numeric(train))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        flags = frame[MISSING_FLAGS].isna().to_numpy(dtype=np.float32)
        return np.column_stack((self.imputer_.transform(self._numeric(frame)), flags))

    def get_feature_names_out(self) -> list[str]:
        return [*ANOMALY_FEATURES, *(f"{column}__missing" for column in MISSING_FLAGS)]


def fit_isolation_forest(train: pd.DataFrame, threads: int) -> tuple[AnomalyPreprocessor, IsolationForest, int]:
    preprocessor = AnomalyPreprocessor().fit(train)
    if len(train) > FIT_ROW_CAP:
        indices = np.random.default_rng(SEED).choice(len(train), size=FIT_ROW_CAP, replace=False)
        fit_rows = train.iloc[np.sort(indices)]
    else:
        fit_rows = train
    model = IsolationForest(n_estimators=100, max_samples=min(256, len(fit_rows)), contamination="auto",
                            random_state=SEED, n_jobs=threads).fit(preprocessor.transform(fit_rows))
    return preprocessor, model, len(fit_rows)


def anomaly_scores(model: IsolationForest, preprocessor: AnomalyPreprocessor,
                   frame: pd.DataFrame) -> np.ndarray:
    """Negate sklearn's normality score so larger values mean more unusual."""
    scores = -model.score_samples(preprocessor.transform(frame))
    if not np.isfinite(scores).all():
        raise ValueError("Isolation Forest returned a nonfinite anomaly score.")
    return scores


def anomaly_percentiles(sorted_reference: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Empirical midranks against a fixed training reference. Results are in [0, 1]."""
    reference = np.asarray(sorted_reference, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if (reference.ndim != 1 or not reference.size or not np.isfinite(reference).all()
            or np.any(reference[1:] < reference[:-1]) or not np.isfinite(scores).all()):
        raise ValueError("Percentiles need a nonempty, sorted, finite reference and finite scores.")
    left = np.searchsorted(reference, scores, side="left")
    right = np.searchsorted(reference, scores, side="right")
    return (left + right) / (2 * len(reference))
