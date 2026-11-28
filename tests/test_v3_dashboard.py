"""Regression checks for the frozen, read-only V3 workbench."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.data_loader import AMOUNT, ID, PRODUCT, TIME
from src.dashboard_v3 import explanations, store

ROOT = Path(__file__).resolve().parents[1]


def test_queue_search_covers_all_decisions_and_sort_has_stable_tiebreak():
    rows = pd.DataFrame({ID: [9, 3, 7], TIME: [1, 2, 3], store.SCORE: [.35, .35, .1],
                         store.DECISION: [True, True, False], AMOUNT: [10, 20, 22], PRODUCT: ["W", "C", "W"]})
    assert store.filter_queue(rows)[ID].tolist() == [3, 9]
    assert store.filter_queue(rows, product="W", minimum=.2)[ID].tolist() == [9]
    assert store.filter_queue(rows, query="7", decision="Flagged", minimum=.99)[ID].tolist() == [7]
    assert store.filter_queue(rows, decision="All decisions", order="Largest amount")[ID].tolist() == [7, 3, 9]
    with pytest.raises(ValueError, match="numeric"):
        store.filter_queue(rows, query="hello")


def test_exploratory_threshold_handles_no_flags_and_inclusive_cutoff():
    assert store.threshold_preview([1, 0], [.4, .1], 1) == {
        "reviewed": 0, "precision": None, "recall": 0, "rows": 2,
    }
    assert store.threshold_preview([1, 0], [.4, .1], .4)["reviewed"] == 1


def test_missing_or_replaced_bundle_is_explained_in_app(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDLENS_V3_ARTIFACTS", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "archive" / "apps" / "app_research.py")).run()
    assert not app.exception and "unavailable" in app.warning[0].value
    file = tmp_path / "step7c-retrospective" / "dashboard.json"
    file.parent.mkdir()
    file.write_text("{}", encoding="utf-8")
    app.run()
    assert not app.exception and "failed its frozen artifact check" in app.warning[0].value


def test_real_release_counts_policy_and_exact_native_tree_shap():
    paths = store.Paths.from_env(ROOT)
    if not (paths.retrospective / "predictions.parquet").is_file():
        pytest.skip("Local V3 release artifacts are required.")
    pack = store.bundle(paths)
    frame = store.queue(paths, pack)
    assert len(frame) == 118_108
    assert pack["models"]["xgboost"]["average_precision"] == pytest.approx(.56989, abs=5e-6)
    assert store.DECISION in frame and frame[store.DECISION].sum() == 3_378
    assert "isFraud" not in frame
    position = int(frame[store.SCORE].idxmax())
    row = frame.iloc[position]
    assert store.observed_label(paths, int(row[ID]), int(row[TIME])) in (0, 1)
    _, history = explanations.history(paths, position)
    assert len(history) == 8
    local = explanations.local(paths, position, float(row[store.SCORE]))
    assert local["margin_error"] < 1e-4 and local["score_error"] < 2e-6
    assert local["base"] + local["remaining"] + sum(local["values"]) == pytest.approx(local["margin"], abs=1e-4)
    assert store.development_preview(paths, store.THRESHOLD)["reviewed"] == 575
