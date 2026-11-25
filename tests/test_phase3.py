"""Round-trip the Phase 3 artifacts against small synthetic Phase 1 and 2 runs."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import phase1, phase3, train
from src.behavior import anomaly_percentiles, anomaly_scores
from src.data_loader import TRANSACTION_DTYPES
from src.training_data import load_partition, read_phase1_manifest


def _synthetic_csvs(data):
    rows = []
    for index in range(30):
        row = dict.fromkeys(TRANSACTION_DTYPES, 1)
        row.update(TransactionID=100 + index, isFraud=int(index % 3 == 1), TransactionDT=index // 2 + 1,
                   TransactionAmt=float(index + 1), ProductCD="W" if index % 3 == 1 else "C")
        rows.append(row)
    pd.DataFrame(rows[::-1]).to_csv(data / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": [100], "DeviceType": ["desktop"], "DeviceInfo": ["test"]}).to_csv(
        data / "train_identity.csv", index=False
    )


def test_phase3_artifacts_reload_and_explain_arbitrary_held_out_rows(tmp_path, monkeypatch):
    data, artifacts = tmp_path / "data", tmp_path / "artifacts"
    data.mkdir()
    _synthetic_csvs(data)
    phase1.run_pipeline(data, artifacts, "quick")
    make_models = train.make_models

    def small_models(labels, threads):
        models = make_models(labels, threads)
        for name in ("random_forest", "xgboost", "xgboost_basic"):
            models[name].set_params(n_estimators=4, max_depth=2)
        return models

    monkeypatch.setattr(train, "make_models", small_models)
    monkeypatch.setattr(train, "champion_from_validation", lambda _: "xgboost")
    phase2_manifest = train.run_training(artifacts, "quick", threads=1)
    original = phase2_manifest.read_bytes()
    fit = phase3.fit_isolation_forest
    seen = []

    def fit_training_only(frame, threads):
        seen.extend(frame["TransactionID"].to_list())
        return fit(frame, threads)

    monkeypatch.setattr(phase3, "fit_isolation_forest", fit_training_only)
    path = phase3.run_phase3(artifacts, "quick", threads=1)
    manifest = json.loads(path.read_bytes())
    assert seen == list(range(100, 118)) and phase2_manifest.read_bytes() == original
    assert manifest["anomaly"]["reference_rows"] == 18
    assert manifest["shap"]["global_rows"] == 6
    run_dir = path.parent / manifest["run_directory"]
    model = joblib.load(run_dir / manifest["files"]["anomaly_model"])
    prep = joblib.load(run_dir / manifest["files"]["anomaly_preprocessor"])
    reference = np.load(run_dir / manifest["files"]["anomaly_reference"], allow_pickle=False)

    store = phase3.Phase3Store(artifacts, "quick", threads=1)
    mode_dir = artifacts / "quick"
    source, _ = read_phase1_manifest(mode_dir)
    for split in ("validation", "test"):
        table = store.read_risk(split)
        frame = load_partition(mode_dir, source, split)
        np.testing.assert_array_equal(table["TransactionID"], frame["TransactionID"])
        scores = anomaly_scores(model, prep, frame)
        np.testing.assert_array_equal(table["anomaly_score"], scores)
        np.testing.assert_array_equal(table["anomaly_percentile"], anomaly_percentiles(reference, scores))
        transaction_id = int(table["TransactionID"].iloc[-1])
        cached = store.assess(split, transaction_id, include_shap=False)
        assert cached["shap"]["status"] == "available_on_demand"
        actual = store.assess(split, transaction_id)
        assert actual["TransactionID"] == transaction_id
        assert actual["classifier_score"] == pytest.approx(actual["shap"]["reconstructed_score"], abs=1e-6)
        assert actual["shap"]["checks"]["max_margin_error"] < 2e-5
    assert len(pd.read_parquet(run_dir / manifest["files"]["global_shap_values"])) == 6
    with pytest.raises(ValueError, match="found 0"):
        store.assess("test", -1)
