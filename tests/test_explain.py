"""Exact SHAP checks using a tiny synthetic sparse XGBoost model."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from scipy.special import expit
from xgboost import DMatrix, XGBClassifier

from src.explain import FraudExplainer
from src.features import CATEGORICAL_BASIC, HISTORICAL_COLUMNS, NUMERIC_BASIC
from src.preprocessing import FraudPreprocessor


def test_tree_shap_sparse_scores_order_additivity_and_grouping(monkeypatch):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({name: rng.normal(size=64) for name in [*NUMERIC_BASIC, *HISTORICAL_COLUMNS]})
    frame[CATEGORICAL_BASIC] = rng.choice(["A", "B", "C"], size=(64, len(CATEGORICAL_BASIC)))
    labels = ((frame["TransactionAmt"] > 0) & frame["ProductCD"].eq("A")).astype(int)
    prep = FraudPreprocessor().fit(frame)
    matrix = prep.transform(frame)
    model = XGBClassifier(n_estimators=8, max_depth=3, n_jobs=1, random_state=42).fit(matrix, labels)
    names = prep.get_feature_names_out().tolist()
    with pytest.raises(ValueError, match="feature order"):
        FraudExplainer(model, prep, names[::-1], threads=1)
    explainer = FraudExplainer(model, prep, names, threads=1)
    assert explainer.mapping[names.index("ProductCD_value=A")]["original_feature"] == "ProductCD"
    assert explainer.mapping[names.index("is_new_product__missing")]["original_feature"] == "is_new_product"

    def forbid_dense(*args, **kwargs):
        raise AssertionError("The XGBoost explanation path must keep CSR sparse semantics.")

    monkeypatch.setattr(csr_matrix, "toarray", forbid_dense)
    saved = model.predict_proba(matrix)[:, 1]
    batch = explainer.explain(frame, saved)
    native = model.get_booster().predict(DMatrix(matrix, nthread=1), pred_contribs=True)
    np.testing.assert_allclose(batch.values, native[:, :-1], atol=1e-6)
    np.testing.assert_allclose(batch.base + batch.grouped.sum(axis=1), batch.margins, atol=2e-5)
    np.testing.assert_allclose(expit(batch.margins), saved, atol=1e-6)
    np.testing.assert_allclose(batch.grouped.sum(axis=1), batch.values.sum(axis=1), atol=1e-6)
    assert len(explainer.global_importance(batch)["transformed"]) == len(names)
    local = explainer.local(frame.iloc[[0]], float(saved[0]))
    assert sum(group["contribution"] for group in local["grouped_contributions"]) == pytest.approx(
        local["raw_margin"] - local["base_value"], abs=2e-5
    )
    with pytest.raises(ValueError, match="saved-score checks"):
        explainer.explain(frame, saved + 0.1)
