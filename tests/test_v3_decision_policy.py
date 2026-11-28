import numpy as np
import pytest

from src.v3 import decision_policy as policy


def test_verify_requires_a_freeze_before_loading_any_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "load_window", lambda *args: pytest.fail("Loaded scores before freeze"))
    with pytest.raises(ValueError, match="Freeze the early policy"):
        policy.verify(tmp_path, tmp_path, tmp_path)


def test_freeze_file_cannot_change(tmp_path):
    path = tmp_path / "decision.json"
    policy.freeze_file(path, {"cutoff": .5})
    policy.freeze_file(path, {"cutoff": .5})
    with pytest.raises(ValueError, match="Refusing"):
        policy.freeze_file(path, {"cutoff": .4})
    assert policy.read(path) == {"cutoff": .5}


def test_calibration_comparison_rejects_split_timestamp():
    with pytest.raises(ValueError, match="disjoint timestamp"):
        policy.compare_calibration(np.array([0, 1, 1, 2]), np.array([0, 1, 0, 1]), np.array([.1, .8, .3, .7]), 2)


def test_calibration_fits_only_first_internal_partition(monkeypatch):
    scores = np.array([.1, .8, .3, .7])
    labels = np.array([0, 1, 0, 1])
    original = policy.fit_calibrator

    def spy(method, fit_scores, fit_labels):
        np.testing.assert_array_equal(fit_scores, scores[:2])
        np.testing.assert_array_equal(fit_labels, labels[:2])
        return original(method, fit_scores, fit_labels)

    monkeypatch.setattr(policy, "fit_calibrator", spy)
    result = policy.compare_calibration(np.array([0, 1, 86400, 2*86400]), labels, scores, 2)
    assert result["fit_rows"] == 2 and result["comparison_rows"] == 2
    assert np.isfinite(result["methods"]["isotonic"]["log_loss"])


def test_rankings_ignore_probability_transform_and_workload_uses_frozen_threshold():
    labels = np.array([0, 1, 0, 1])
    scores = {"xgboost": np.array([.25, .55, .45, .8])}
    decision = {"calibrator": {"method": "isotonic", "x": [0, .4, 1], "y": [.1, .3, .3]},
                "thresholds": {"xgboost": {"selected_f1": .3}}}
    result = policy.window_metrics(labels, scores, decision)["xgboost"]
    assert result["ranking"]["ap"] == 1
    assert result["ranking"]["roc_auc"] == 1
    assert result["operating_points"]["selected_f1"]["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["operating_points"]["selected_f1"]["reviewed"] == 3
    assert result["operating_points"]["original_score_0.5"]["reviewed"] == 2
