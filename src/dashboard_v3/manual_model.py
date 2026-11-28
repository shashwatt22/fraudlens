"""Four-input model contract and exact XGBoost inference."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from threading import Lock

import numpy as np
import pandas as pd
from scipy.special import expit
import xgboost as xgb

MODEL = "FraudLens Manual v1 · XGBoost"
AMOUNT = "TransactionAmt"
FIELDS = ("card4", "card6", "P_emaildomain")
LABELS = ("Amount", "Card network", "Card type", "Email provider")
UNKNOWN, OTHER = "unknown", "other"
OPTIONS = (
    {"visa": "Visa", "mastercard": "Mastercard", "american express": "American Express",
     "discover": "Discover", UNKNOWN: "Unknown"},
    {"debit": "Debit", "credit": "Credit", "charge card": "Charge card", UNKNOWN: "Unknown"},
    {"google": "Google", "yahoo": "Yahoo", "microsoft": "Microsoft", "aol": "AOL",
     "apple": "Apple", OTHER: "Other", UNKNOWN: "Unknown"},
)
PROVIDERS = {
    "gmail.com": "google", "googlemail.com": "google",
    "yahoo.com": "yahoo", "yahoo.com.mx": "yahoo", "yahoo.fr": "yahoo", "yahoo.de": "yahoo",
    "yahoo.es": "yahoo", "yahoo.co.uk": "yahoo", "yahoo.co.jp": "yahoo", "ymail.com": "yahoo", "rocketmail.com": "yahoo",
    "hotmail.com": "microsoft", "hotmail.co.uk": "microsoft", "hotmail.fr": "microsoft",
    "hotmail.de": "microsoft", "hotmail.es": "microsoft", "outlook.com": "microsoft",
    "outlook.es": "microsoft", "live.com": "microsoft", "live.com.mx": "microsoft", "live.fr": "microsoft", "msn.com": "microsoft",
    "aol.com": "aol", "aim.com": "aol", "icloud.com": "apple", "me.com": "apple", "mac.com": "apple",
    "anonymous.com": UNKNOWN,
}
MIN_AMOUNT, MAX_AMOUNT = .01, 32_000.00
SCHEMA = 1


def normalize(raw):
    """Map benchmark labels to categories users can actually supply."""
    mapped = pd.DataFrame({AMOUNT: raw[AMOUNT].round(2).astype("float32")}, index=raw.index)
    for col, choices in zip(FIELDS[:2], OPTIONS[:2], strict=True):
        mapped[col] = raw[col].where(raw[col].isin(choices), UNKNOWN)
    email = raw[FIELDS[-1]]
    mapped[FIELDS[-1]] = email.map(PROVIDERS).fillna(OTHER).where(email.notna(), UNKNOWN)
    return mapped


def fit_categories(training, fields=FIELDS):
    return {col: sorted(training[col].unique().tolist()) for col in fields}


def encode(frame, categories, fields=FIELDS):
    """Reuse one-hot columns learned on training rows."""
    groups, names = [0], ["amount"]
    values = [frame[AMOUNT].to_numpy(dtype=np.float32)]
    for group, col in enumerate(fields, start=1):
        for i, category in enumerate(categories[col]):
            names.append(f"{col}_{i}")
            groups.append(group)
            values.append(frame[col].eq(category).to_numpy(dtype=np.float32))
    return np.column_stack(values), names, np.array(groups)


def validate(amount, network, funding, provider):
    if isinstance(amount, bool):
        raise ValueError("Enter a valid amount in USD.")
    try:
        value = float(amount)
    except (TypeError, ValueError) as error:
        raise ValueError("Enter a valid amount in USD.") from error
    if not math.isfinite(value) or not MIN_AMOUNT <= value <= MAX_AMOUNT:
        raise ValueError(f"Enter an amount between USD {MIN_AMOUNT:,.2f} and USD {MAX_AMOUNT:,.2f}.")
    if not math.isclose(value, round(value, 2), abs_tol=1e-10, rel_tol=0):
        raise ValueError("Enter the amount with at most two decimal places.")
    values = (network, funding, provider)
    if any(not isinstance(item, str) or item not in choices for item, choices in zip(values, OPTIONS, strict=True)):
        raise ValueError("Choose a supported value for each transaction field.")
    return {AMOUNT: round(value, 2), **dict(zip(FIELDS, values, strict=True))}


@dataclass(frozen=True)
class AssessmentScore:
    score: float
    threshold: float
    margin: float
    base: float
    impacts: tuple[float, ...]
    labels: tuple[str, ...] = LABELS

    @property
    def flagged(self):
        return self.score >= self.threshold


class AssessmentModel:
    def __init__(self, directory, contract=None):
        self.name = MODEL if contract is None else contract.MODEL
        self.fields = FIELDS if contract is None else contract.FIELDS
        self.labels = LABELS if contract is None else contract.LABELS
        self.validate = validate if contract is None else contract.validate
        self.manifest = json.loads((directory / "manifest.json").read_bytes())
        checkpoint = directory / "model.ubj"
        if (self.manifest["schema"] != SCHEMA or self.manifest["model"] != self.name
                or sha256(checkpoint.read_bytes()).hexdigest() != self.manifest["model_sha256"]):
            raise ValueError("The manual assessment artifacts failed verification.")
        self.threshold = float(self.manifest["threshold"])
        self.categories = self.manifest["categories"]
        if not 0 < self.threshold < 1 or set(self.categories) != set(self.fields):
            raise ValueError("The manual model policy is invalid.")
        self.booster = xgb.Booster(params={"nthread": 1})
        self.booster.load_model(checkpoint)
        self.lock = Lock()

    def assess(self, amount, network, funding, provider, *optional):
        form = self.validate(amount, network, funding, provider, *optional)
        values, names, groups = encode(pd.DataFrame([form]), self.categories, self.fields)
        row = xgb.DMatrix(values, feature_names=names, nthread=1)
        with self.lock:
            score = float(self.booster.predict(row)[0])
            margin = float(self.booster.predict(row, output_margin=True)[0])
            contributions = self.booster.predict(row, pred_contribs=True)[0]
        base = float(contributions[-1])
        impacts = tuple(np.bincount(groups, weights=contributions[:-1], minlength=len(self.labels)).tolist())
        if (not math.isfinite(score) or abs(base + sum(impacts) - margin) > 1e-4
                or abs(float(expit(margin)) - score) > 2e-6):
            raise ValueError("The manual model explanation did not reconstruct the risk score.")
        return AssessmentScore(score, self.threshold, margin, base, impacts, self.labels)


def signature(directory):
    return tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in (directory / "manifest.json", directory / "model.ubj"))
