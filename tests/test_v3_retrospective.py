import numpy as np
import pandas as pd
import pytest

from src.data_loader import ID, TIME
from src.v3 import retrospective as retro
from src.v3.retrospective_report import chart_points


def test_holdout_ids_and_timestamps_must_match_in_source_order():
    meta = pd.DataFrame({ID: [1, 2], TIME: [100, 100]})
    retro.validate_alignment(meta, meta.assign(score=[.2, .4]))
    with pytest.raises(ValueError, match="source order"):
        retro.validate_alignment(meta, meta.iloc[::-1])


def test_label_gate_requires_scores_and_is_exclusive(tmp_path):
    scores = tmp_path / "score_manifest.json"
    with pytest.raises(ValueError, match="before opening labels"):
        retro.label_gate(tmp_path, scores)
    scores.write_text("{}", encoding="utf-8")
    retro.label_gate(tmp_path, scores)
    assert retro.read(tmp_path / "label_access.json")["score_manifest_sha256"] == retro.sha256(scores)
    with pytest.raises(FileExistsError):
        retro.label_gate(tmp_path, scores)


def test_frozen_and_default_decisions_use_the_same_original_ranking():
    labels = np.array([1, 0, 1, 0])
    predictions = np.array([.8, .3, .2, .1])
    result, curves = retro.model_metrics(labels, predictions, .2)
    assert result["at_0_5"]["confusion_matrix"] == [[2, 0], [1, 1]]
    assert result["frozen"]["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["frozen"]["review_rate"] == .75
    assert result["average_precision"] == pytest.approx(5 / 6)
    assert result["review_budgets"]["1"]["recall"] == .5
    assert set(curves) == {"roc", "precision_recall", "reliability"}


def test_rejects_modified_policy_before_reading_models(tmp_path):
    for name in ("decision_freeze.json", "model_freeze.json", "policy_metrics.json", "artifact_hashes.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest or metrics hash mismatch"):
        retro.frozen_inputs(tmp_path, tmp_path)


def test_dashboard_curve_is_bounded_and_contains_only_coordinates():
    full = {"recall": list(range(11)), "precision": list(range(10, -1, -1)), "thresholds": list(range(10))}
    assert chart_points(full, "recall", "precision", limit=3) == {
        "recall": [0, 5, 10], "precision": [10, 5, 0],
    }
