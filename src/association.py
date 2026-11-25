"""Fit interpretable Boolean conditions and mine training-only fraud rules."""

from dataclasses import asdict, dataclass

from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

from .data_loader import AMOUNT, IDENTITY_MISSING, TARGET

FRAUD = "fraud"


@dataclass(frozen=True)
class MiningSettings:
    min_support: float = .001
    min_confidence: float = .05
    min_matches: int = 100
    min_fraud_matches: int = 30
    max_antecedents: int = 3


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    column: str
    operation: str
    cutoff: float | None
    basis: str
    history_required: bool = False


def fit_conditions(train: pd.DataFrame) -> tuple[Condition, ...]:
    """Freeze rules for numerical binning using the unmodified training partition."""
    has_history = train["prior_txn_count"].gt(0).fillna(False)

    def quantile(column: str, q: float, history=False) -> float | None:
        observed = (train.loc[has_history, column] if history else train[column]).dropna()
        return float(observed.quantile(q)) if len(observed) else None

    recent = quantile("prior_count_24h", .75, history=True)
    return (
        Condition("high_amount", "High amount", AMOUNT, "ge", quantile(AMOUNT, .90), "Training amount p90"),
        Condition("large_vs_history", "Large vs. prior mean", "amount_to_prior_mean", "ge",
                  quantile("amount_to_prior_mean", .90, history=True), "Observed training amount-ratio p90", True),
        Condition("recent_activity", "High 24-hour activity", "prior_count_24h", "ge",
                  max(1., recent) if recent is not None else None, "Training 24-hour count p75 with prior history, floor 1", True),
        Condition("new_product", "New product with history", "is_new_product", "eq", 1,
                  "Observed novelty flag with strictly earlier history", True),
        Condition("no_prior_history", "No prior history", "prior_txn_count", "eq", 0, "Zero strictly earlier proxy transactions"),
        Condition("identity_missing", "Identity row missing", IDENTITY_MISSING, "eq", 1, "Identity join found no match"),
        Condition("long_gap", "Long transaction gap", "seconds_since_prior_txn", "ge",
                  quantile("seconds_since_prior_txn", .75, history=True), "Observed training elapsed-gap p75", True),
    )


def encode_basket(frame: pd.DataFrame, conditions: tuple[Condition, ...]) -> pd.DataFrame:
    has_history = frame["prior_txn_count"].gt(0).fillna(False)
    columns = {}
    for condition in conditions:
        values = frame[condition.column]
        if condition.cutoff is None:
            matched = pd.Series(False, index=frame.index)
        else:
            matched = values.ge(condition.cutoff) if condition.operation == "ge" else values.eq(condition.cutoff)
            matched = matched.fillna(False)
        columns[condition.key] = (matched & has_history if condition.history_required else matched).astype(bool)
    return pd.DataFrame({**columns, FRAUD: frame[TARGET].eq(1).astype(bool)}, index=frame.index)


def baseline(basket: pd.DataFrame) -> dict:
    fraud = int(basket[FRAUD].sum())
    return {"rows": len(basket), "fraud_rows": fraud, "fraud_rate": fraud / len(basket) if len(basket) else None}


def rule_metrics(basket: pd.DataFrame, antecedents: tuple[str, ...]) -> dict:
    """Count the given rule on one partition, explicitly handling undefined ratios."""
    if not antecedents or FRAUD in antecedents:
        raise ValueError("A fraud rule needs at least one behavioral antecedent.")
    matches = basket[list(antecedents)].all(axis=1)
    count = int(matches.sum())
    joint = int((matches & basket[FRAUD]).sum())
    summary = baseline(basket)
    confidence = joint / count if count else None
    return {**summary, "matches": count, "fraud_matches": joint,
            "support": joint / len(basket) if len(basket) else None, "confidence": confidence,
            "lift": confidence / summary["fraud_rate"] if confidence is not None and summary["fraud_rate"] else None}


def mine_rules(basket: pd.DataFrame, settings: MiningSettings = MiningSettings()) -> tuple[list[dict], int]:
    """Use MLxtend on training data and retain only supported fraud-consequent rules."""
    if basket.empty or not basket[FRAUD].any():
        return [], 0
    itemsets = apriori(basket, min_support=settings.min_support, use_colnames=True, max_len=settings.max_antecedents + 1)
    if not any(FRAUD in items and len(items) > 1 for items in itemsets["itemsets"]):
        return [], len(itemsets)
    candidates = association_rules(itemsets, num_itemsets=len(basket), metric="confidence",
                                   min_threshold=settings.min_confidence, return_metrics=["support", "confidence", "lift"])
    rules = []
    for parts, consequent in zip(candidates["antecedents"], candidates["consequents"], strict=True):
        if consequent != {FRAUD}:
            continue
        antecedents = tuple(sorted(parts))
        stats = rule_metrics(basket, antecedents)
        if stats["matches"] >= settings.min_matches and stats["fraud_matches"] >= settings.min_fraud_matches:
            rules.append({"antecedents": antecedents, "consequent": FRAUD, "train": stats})
    return sorted(rules, key=lambda rule: (-rule["train"]["lift"], rule["antecedents"])), len(itemsets)


def evaluate_rules(rules: list[dict], validation: pd.DataFrame, conditions: tuple[Condition, ...]) -> list[dict]:
    labels = {condition.key: condition.label for condition in conditions}
    result = []
    for number, rule in enumerate(rules, 1):
        displayed = " + ".join(labels[name] for name in rule["antecedents"])
        metrics = rule_metrics(validation, rule["antecedents"])
        text = (f"Among validation rows meeting {displayed}, {metrics['fraud_matches']:,} of {metrics['matches']:,} "
                "were labeled fraud.")
        result.append({"rule_id": number, "label": displayed, **rule, "validation": metrics, "interpretation": text})
    return result


def condition_definitions(conditions: tuple[Condition, ...]) -> list[dict]:
    return [asdict(condition) for condition in conditions]
