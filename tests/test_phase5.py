"""Round-trip saved artifacts and exercise the finished live-view components."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import phase1, phase5
from src.dashboard.artifacts import ArtifactError, DashboardRun, generation_commands
from src.data_loader import TRANSACTION_DTYPES
from src.training_data import load_partition, read_phase1_manifest

ROOT = Path(__file__).resolve().parents[1]
V1_ARTIFACTS = Path(os.environ.get("FRAUDLENS_ARTIFACTS", ROOT / "artifacts"))


def test_phase5_round_trip_reads_only_training_and_validation(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    for index in range(40):
        row = dict.fromkeys(TRANSACTION_DTYPES, 1)
        row.update(TransactionID=100 + index, isFraud=int(index % 2 == 1), TransactionDT=index // 2 + 1,
                   TransactionAmt=float(index + 1), ProductCD="A")
        rows.append(row)
    pd.DataFrame(rows).to_csv(data / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": [100], "DeviceType": ["desktop"], "DeviceInfo": ["test"]}).to_csv(
        data / "train_identity.csv", index=False
    )
    root = tmp_path / "artifacts"
    phase1.run_pipeline(data, root, "quick")
    used = []

    def track(mode_dir, manifest, name):
        used.append(name)
        return load_partition(mode_dir, manifest, name)

    monkeypatch.setattr(phase5, "load_partition", track)
    path = phase5.run_phase5(root, "quick", 1)
    assert used == ["train", "validation"]
    source, digest = read_phase1_manifest(root / "quick")
    manifest = json.loads(path.read_bytes())
    directory = path.parent / manifest["run_directory"]
    assert manifest["phase1_manifest_sha256"] == digest
    files = manifest["files"]
    points = pd.read_parquet(directory / files["clusters"])
    train = load_partition(root / "quick", source, "train").set_index("TransactionID")
    prep = joblib.load(directory / files["cluster_preprocessor"])
    pca = joblib.load(directory / files["pca"])
    projection = pca.transform(prep.transform(train.loc[points["TransactionID"]]))
    np.testing.assert_allclose(projection, points[["pca_1", "pca_2"]], atol=1e-10)
    assert pd.read_parquet(directory / files["profiles"])["rows"].sum() == len(points)
    association = json.loads((directory / files["association"]).read_bytes())
    assert association["validation_baseline"]["rows"] == 8
    assert association["rules"] == []


def test_phase5_missing_or_incompatible_does_not_block_existing_pages(tmp_path):
    original = V1_ARTIFACTS / "quick"
    if not (original / "phase3" / "manifest.json").is_file():
        pytest.skip("Generate the Git-ignored quick artifacts to run manifest integration checks.")
    mode = tmp_path / "quick"
    for folder in ("", "phase2", "phase2b", "phase3"):
        destination = mode / folder / "manifest.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((original / folder / "manifest.json").read_bytes())
    run = DashboardRun(tmp_path, "quick")
    with pytest.raises(ArtifactError, match="Phase 5 quick manifest"):
        run.require_patterns()
    path = mode / "phase5" / "manifest.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"phase": 5, "schema_version": 1, "mode": "quick", "phase1_manifest_sha256": "wrong"}))
    run = DashboardRun(tmp_path, "quick")
    with pytest.raises(ArtifactError, match="different Phase 1 snapshot"):
        run.require_patterns()
    assert "src.phase5 --mode quick --threads 6" in generation_commands("quick", 5)


def test_real_artifacts_render_phase5_charts_profiles_and_rule_search():
    root = V1_ARTIFACTS
    if not (root / "quick" / "phase5" / "manifest.json").is_file():
        pytest.skip("Generate Phase 5 quick artifacts to run real dashboard checks.")
    run = DashboardRun(root, "quick")
    clusters = run.report(5, "clustering")
    rules = run.report(5, "association")["rules"]
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40)
    app.session_state["dataset_mode"] = "quick"
    app.run()
    app.radio(key="navigation").set_value("Behavioral Intelligence & Patterns").run()
    assert not app.exception and not app.error and not app.warning
    assert [tab.label for tab in app.tabs] == ["Anomaly Intelligence", "Behavioral Clustering", "Association Rules", "Global Explainability"]
    values = {metric.label: metric.value for metric in app.metric}
    assert values["Clusters"] == str(clusters["selected"]["clusters"])
    assert values["Fraud-associated rules"] == str(len(rules))
    assert any(item.value == "PCA cluster map" for item in app.subheader)
    assert any("Fraud rate" in table.value.columns for table in app.dataframe)
    if rules:
        assert any("Antecedent → fraud" in table.value.columns
                   and table.value["Antecedent → fraud"].eq(rules[0]["label"]).any() for table in app.dataframe)
        app.text_input(key="apriori_search").set_value("impossible synthetic filter").run()
        assert any("No mined rules match" in message.value for message in app.info)
