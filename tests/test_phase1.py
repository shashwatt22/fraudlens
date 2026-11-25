"""Round-trip the pipeline using tiny synthetic CSVs and real Parquet files."""

import json

import pandas as pd
import pyarrow.parquet as pq

from src import phase1
from src.data_loader import TRANSACTION_DTYPES


def test_quick_artifacts_keep_cross_partition_history_and_only_observed_categories(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    for index in range(12):
        row = dict.fromkeys(TRANSACTION_DTYPES, 1)
        row.update(TransactionID=100 + index, isFraud=index % 2, TransactionDT=index // 2 + 1,
                   TransactionAmt=10.0, ProductCD={7: "X", 9: "Z", 10: "Q"}.get(index, "W"))
        rows.append(row)
    pd.DataFrame(rows[::-1]).to_csv(data / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": [100], "DeviceType": ["desktop"], "DeviceInfo": ["example"]}).to_csv(
        data / "train_identity.csv", index=False
    )

    monkeypatch.setattr(phase1, "QUICK_ROWS", 9)
    manifest_path = phase1.run_pipeline(data, tmp_path / "artifacts", "quick")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["processed_rows"] == 10
    assert manifest["mode"] == "quick"

    paths = {name: manifest_path.parent / manifest["files"][name] for name in manifest["partitions"]}
    partitions = {name: pd.read_parquet(path) for name, path in paths.items()}
    train, validation, test = partitions.values()
    assert train["TransactionID"].tolist() == list(range(100, 106))
    assert validation["TransactionID"].tolist() == [106, 107]
    assert test["TransactionID"].tolist() == [108, 109]
    assert validation["prior_txn_count"].tolist() == [6, 6]
    assert test["prior_txn_count"].tolist() == [8, 8]
    assert train["ProductCD"].cat.categories.tolist() == ["W"]
    assert set(validation["ProductCD"].cat.categories) == {"W", "X"}
    assert pq.read_metadata(paths["test"]).metadata[b"fraudlens.processed_rows"] == b"10"

    eda = json.loads((manifest_path.parent / manifest["files"]["eda"]).read_text(encoding="utf-8"))
    assert eda["processed"]["identity_matches"] == 1
    assert eda["processed"]["missing"]["DeviceType"]["count"] == 9
    assert eda["sources"]["transactions"]["rows"] == 12
