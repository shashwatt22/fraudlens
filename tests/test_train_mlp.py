"""Round-trip the full Phase 2B pipeline with small synthetic chronological files."""

from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src import neural, phase1, train, train_mlp
from src.data_loader import TRANSACTION_DTYPES
from src.training_data import load_partition, read_phase1_manifest


def _write_synthetic_data(data):
    rows = []
    for index in range(30):
        row = dict.fromkeys(TRANSACTION_DTYPES, 1)
        row.update(TransactionID=100 + index, isFraud=int(index % 3 == 1),
                   TransactionDT=index // 2 + 1, TransactionAmt=float(1 + index),
                   ProductCD="W" if index < 18 else "C")
        rows.append(row)
    pd.DataFrame(rows[::-1]).to_csv(data / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": [100], "DeviceType": ["desktop"], "DeviceInfo": ["test"]}).to_csv(
        data / "train_identity.csv", index=False
    )


def test_pipeline_reuses_preprocessor_freezes_selection_and_reloads(tmp_path, monkeypatch):
    data, artifacts = tmp_path / "data", tmp_path / "artifacts"
    data.mkdir()
    _write_synthetic_data(data)
    phase1.run_pipeline(data, artifacts, "quick")

    make_models = train.make_models

    def small_models(labels, threads):
        models = make_models(labels, threads)
        for name in ("random_forest", "xgboost", "xgboost_basic"):
            models[name].set_params(n_estimators=2, max_depth=2)
        return models

    monkeypatch.setattr(train, "make_models", small_models)
    classical = train.run_training(artifacts, "quick", threads=1)
    original_classical = classical.read_bytes()
    events = []

    def track_read(mode_dir, manifest, name):
        events.append(name)
        if name == "test":
            stage, = (mode_dir / "phase2b").glob(".phase2b-*/run")
            assert (stage / "best_model.pt").is_file()
            selection = json.loads((stage / "selection.json").read_bytes())
            assert selection["selected_on"] == "validation"
        return load_partition(mode_dir, manifest, name)

    monkeypatch.setattr(train_mlp, "load_partition", track_read)
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    config = neural.TrainingConfig(batch_size=4, max_epochs=3, patience=2)
    try:
        manifest_path = train_mlp.run_training(artifacts, "quick", threads=1, config=config)
        manifest = json.loads(manifest_path.read_bytes())
        assert events == ["train", "validation", "test"]
        assert classical.read_bytes() == original_classical
        assert manifest["training_config"]["positive_class_weight"] == 2

        mode_dir = artifacts / "quick"
        source, digest = read_phase1_manifest(mode_dir)
        preprocessor, reference, _, _ = train_mlp.load_classical_preprocessor(mode_dir, source, digest)
        with pytest.raises(ValueError, match="different Phase 1 snapshot"):
            train_mlp.load_classical_preprocessor(mode_dir, source, "different-digest")
        assert manifest["preprocessor"] == reference
        assert reference["sha256"] == sha256((mode_dir / reference["path"]).read_bytes()).hexdigest()
        input_dim = manifest["architecture"]["input_dim"]
        assert input_dim == len(preprocessor.get_feature_names_out())

        run = manifest_path.parent / manifest["run_directory"]
        model = neural.load_checkpoint(run / manifest["files"]["checkpoint"], input_dim)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))
        for split in ("validation", "test"):
            frame = load_partition(mode_dir, source, split)
            loader = neural.make_loader(preprocessor.transform(frame), frame["isFraud"].to_numpy(),
                                        config.batch_size, training=False)
            _, scores = neural.score_batches(model, loader, criterion)
            saved = pd.read_parquet(run / manifest["files"][f"{split}_predictions"])
            np.testing.assert_array_equal(saved["TransactionID"], frame["TransactionID"])
            np.testing.assert_array_equal(saved["isFraud"], frame["isFraud"])
            np.testing.assert_allclose(saved["mlp_score"], scores, atol=1e-7, rtol=1e-6)
            np.testing.assert_array_equal(saved["mlp_decision"], scores >= manifest["selection"]["threshold"])

        report = json.loads((run / manifest["files"]["comparison"]).read_bytes())
        assert report["test_scope"] == "development diagnostics"
        assert all("mlp" in report[split] and "xgboost" in report[split] for split in ("validation", "test"))
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)
