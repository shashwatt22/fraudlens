"""Matched pipeline lineage, checkpoint freezing, and dashboard rendering."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import phase1, phase6
from src.dashboard.artifacts import DashboardRun
from src.data_loader import TRANSACTION_DTYPES
from src.lab_benchmark import ScratchSettings
from src.scratch_neural import ScratchNetwork
from src.training_data import load_partition, read_phase1_manifest

ROOT = Path(__file__).resolve().parents[1]
V1_ARTIFACTS = Path(os.environ.get("FRAUDLENS_ARTIFACTS", ROOT / "artifacts"))


def test_sample_fit_and_freeze_then_save_matched_results(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    for index in range(40):
        row = dict.fromkeys(TRANSACTION_DTYPES, 1)
        row.update(TransactionID=100 + index, isFraud=index % 2, TransactionDT=index // 2 + 1,
                   TransactionAmt=float(1 + index), ProductCD="A" if index % 2 else "B")
        rows.append(row)
    pd.DataFrame(rows).to_csv(data / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": [100], "DeviceType": ["desktop"], "DeviceInfo": ["fixture"]}).to_csv(
        data / "train_identity.csv", index=False
    )
    root = tmp_path / "artifacts"
    phase1.run_pipeline(data, root, "quick")
    real_build = phase6.build_models

    def small_models(y, threads, l2):
        models = real_build(y, threads, l2)
        for key in ("random_forest", "xgboost", "adaboost"):
            models[key].set_params(n_estimators=3)
        return models

    loaded, fitted_ids = [], []
    fit = phase6.FraudPreprocessor.fit

    def track_read(mode_dir, manifest, name):
        loaded.append(name)
        if name == "test":
            frozen, = (mode_dir / "phase6").glob(".phase6-*/run/selection.json")
            selection = json.loads(frozen.read_bytes())
            assert selection["selected_on"] == "validation" and len(selection["best_epochs"]) == 3
        return load_partition(mode_dir, manifest, name)

    def track_fit(preprocessor, frame):
        fitted_ids.extend(frame["TransactionID"].tolist())
        return fit(preprocessor, frame)

    monkeypatch.setattr(phase6, "build_models", small_models)
    monkeypatch.setattr(phase6, "load_partition", track_read)
    monkeypatch.setattr(phase6.FraudPreprocessor, "fit", track_fit)
    monkeypatch.setattr(phase6, "TRAIN_LIMIT", 16)
    monkeypatch.setattr(phase6, "EVAL_LIMIT", 6)
    path = phase6.run_phase6(root, "quick", 1, ScratchSettings(max_epochs=3, patience=2, batch_size=4))
    assert loaded == ["train", "validation", "test"]
    manifest = json.loads(path.read_bytes())
    run = path.parent / manifest["run_directory"]
    files = manifest["files"]
    assert len(fitted_ids) == 16
    assert fitted_ids == pd.read_parquet(run / files["train_sample"])["TransactionID"].tolist()
    preprocessor = joblib.load(run / files["preprocessor"])
    source, _ = read_phase1_manifest(root / "quick")
    for partition in ("validation", "test"):
        sample = pd.read_parquet(run / files[f"{partition}_sample"])
        pred = pd.read_parquet(run / files[f"{partition}_predictions"])
        np.testing.assert_array_equal(sample["TransactionID"], pred["TransactionID"])
        for key in manifest["models"]:
            assert pred[f"{key}_score"].notna().sum() == len(sample) == 6
        frame = load_partition(root / "quick", source, partition).set_index("TransactionID").loc[sample["TransactionID"]]
        dense = preprocessor.transform(frame).toarray()
        for key in ("numpy_linear", "numpy_relu32"):
            model = ScratchNetwork.load(run / manifest["models"][key]["file"])
            np.testing.assert_array_equal(pred[f"{key}_score"], model.predict_scores(dense))


def test_real_syllabus_views_load_both_matched_report_types():
    root = V1_ARTIFACTS
    if not (root / "quick" / "phase6" / "manifest.json").is_file():
        pytest.skip("Generate the Phase 6 quick artifacts for live dashboard integration tests.")
    run = DashboardRun(root, "quick")
    run.require_optional(6)
    manifest = run.manifests[6]
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40)
    app.session_state["dataset_mode"] = "quick"
    app.run()
    app.radio(key="navigation").set_value("Model Lab").run()
    assert not app.exception and not app.error and not app.warning
    assert any("K-nearest neighbors" in table.value["Model"].values for table in app.dataframe if "Model" in table.value)
    app.radio(key="syllabus_view").set_value("From-scratch networks").run()
    assert not app.exception and not app.error and not app.warning
    assert any(metric.value == f"{manifest['models']['numpy_relu32']['parameter_count']:,} parameters" for metric in app.metric)
    assert any("Max finite-difference error" in table.value.columns for table in app.dataframe)
