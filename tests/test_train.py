"""Exercise the Phase 2 CLI backend and artifact reload with synthetic data."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import phase1, train
from src.data_loader import TRANSACTION_DTYPES
from src.evaluate import positive_scores


def _write_synthetic_data(directory):
    rows = []
    for index in range(30):
        row = dict.fromkeys(TRANSACTION_DTYPES, 1)
        row.update(TransactionID=100 + index, isFraud=int(index % 3 == 1),
                   TransactionDT=index // 2 + 1, TransactionAmt=float(1 + index),
                   ProductCD="W" if index < 18 else "C")
        rows.append(row)
    pd.DataFrame(rows[::-1]).to_csv(directory / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": [100], "DeviceType": ["desktop"], "DeviceInfo": ["test"]}).to_csv(
        directory / "train_identity.csv", index=False
    )


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_champion_uses_average_precision_and_declared_tie_order():
    metrics = {name: {"average_precision": 0.5} for name in train.CORE_LEARNED}
    assert train.champion_from_validation(metrics) == "logistic_regression"
    metrics["xgboost"]["average_precision"] = 0.6
    assert train.champion_from_validation(metrics) == "xgboost"


def test_pipeline_freezes_validation_selection_before_opening_test_and_reloads(tmp_path, monkeypatch):
    data, artifacts = tmp_path / "data", tmp_path / "artifacts"
    data.mkdir()
    _write_synthetic_data(data)
    phase1.run_pipeline(data, artifacts, "quick")

    make_models = train.make_models
    load_partition = train.load_partition
    select_threshold = train.select_threshold
    fit_preprocessor = train.FraudPreprocessor.fit
    events = []

    def small_models(labels, threads):
        models = make_models(labels, threads)
        for name in ("random_forest", "xgboost", "xgboost_basic"):
            models[name].set_params(n_estimators=4, max_depth=2)
        return models

    def track_read(mode_dir, manifest, name):
        events.append(name)
        if name == "test":
            selection_paths = list((mode_dir / "phase2").glob(".phase2-*/run/selection.json"))
            assert len(selection_paths) == 1
            assert _read_json(selection_paths[0])["selection_partition"] == "validation"
        return load_partition(mode_dir, manifest, name)

    def track_threshold(labels, scores):
        events.append("threshold")
        assert labels.tolist() == [0, 1, 0, 0, 1, 0]
        return select_threshold(labels, scores)

    def track_fit(self, frame, y=None):
        events.append("fit")
        assert frame["TransactionID"].tolist() == list(range(100, 118))
        return fit_preprocessor(self, frame, y)

    monkeypatch.setattr(train, "make_models", small_models)
    monkeypatch.setattr(train, "load_partition", track_read)
    monkeypatch.setattr(train, "select_threshold", track_threshold)
    monkeypatch.setattr(train.FraudPreprocessor, "fit", track_fit)

    path = train.run_training(artifacts, "quick", threads=1)
    manifest = _read_json(path)
    assert events == ["train", "validation", "fit", "threshold", "threshold", "threshold", "fit", "threshold", "test"]
    assert manifest["processed_rows"] == 30
    assert manifest["test_scope"] == "development diagnostics"
    assert manifest["models"]["xgboost"]["configuration"] == manifest["models"]["xgboost_basic"]["configuration"]
    assert manifest["models"]["xgboost"]["configuration"]["scale_pos_weight"] == 2
    assert manifest["features"]["behavioral"]["transformed_dimension"] == (
        manifest["features"]["basic"]["transformed_dimension"] + 14
    )

    run = path.parent / manifest["run_directory"]
    source_manifest = _read_json(artifacts / "quick" / "manifest.json")
    selection = _read_json(run / manifest["files"]["selection"])
    metrics = _read_json(run / manifest["files"]["test_metrics"])["models"]
    ablation = _read_json(run / manifest["files"]["ablation"])
    assert ablation["test"]["delta_behavioral_minus_basic"]["average_precision"] == pytest.approx(
        metrics["xgboost"]["average_precision"] - metrics["xgboost_basic"]["average_precision"]
    )

    for partition in ("validation", "test"):
        predictions = pd.read_parquet(run / manifest["files"][f"{partition}_predictions"])
        raw = load_partition(artifacts / "quick", source_manifest, partition)
        assert predictions["TransactionID"].tolist() == raw["TransactionID"].tolist()
        assert predictions["isFraud"].tolist() == raw["isFraud"].tolist()
        for name, model_info in manifest["models"].items():
            preprocessor = joblib.load(run / manifest["files"]["preprocessors"][model_info["feature_set"]])
            model = joblib.load(run / manifest["files"]["classifiers"][name])
            scores = positive_scores(model, preprocessor.transform(raw))
            assert scores.shape == (len(predictions),) and np.isfinite(scores).all()
            np.testing.assert_array_equal(predictions[f"{name}_score"], scores)
            expected = np.zeros(len(predictions)) if name == "dummy" else scores >= selection["thresholds"][name]
            np.testing.assert_array_equal(predictions[f"{name}_decision"], expected)
