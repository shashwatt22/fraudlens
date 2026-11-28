import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.v3.final_refit import fit_model, populations, safe_paths
from src.v3.final_refit_report import policy_metrics, score_all


def test_populations_keep_tied_policy_rows_and_separate_neural_early_stopping():
    times = np.r_[np.arange(24), [24, 25, 25, 26, 27, 28, 29, 30]]
    parts = populations(times, [SimpleNamespace(validation_end=24)])
    assert parts["torch_gradient"] == (0, 21)
    assert parts["torch_internal"] == (21, 24)
    assert parts["earlier_policy"] == (24, 28)
    assert parts["later_policy"] == (28, 32)
    assert parts["original_calibration_cut"] == 27
    with pytest.raises(ValueError, match="timestamp ties"):
        populations(times, [SimpleNamespace(validation_end=26)])


def test_fit_rejects_a_population_including_policy_rows(tmp_path):
    plan = {"partitions": {"pre_policy": {"rows": 20}}}
    with pytest.raises(ValueError, match="outside the frozen"):
        fit_model("logistic", np.ones((21, 1)), np.zeros(21), [], ["x"], plan, tmp_path, "digest")


def test_cannot_score_before_all_models_are_frozen(tmp_path):
    (tmp_path / "run_status.json").write_text(json.dumps({"completed": ["logistic"], "constraint": None}))
    with pytest.raises(ValueError, match="four model checkpoints"):
        score_all(tmp_path / "development", tmp_path / "prior", tmp_path)


def test_output_cannot_overwrite_selected_artifacts(tmp_path):
    prior = tmp_path / "prior"
    with pytest.raises(ValueError, match="separate output"):
        safe_paths(tmp_path / "development", prior, prior / "step6a-torch" / "fold3")


def test_policy_metrics_keep_fixed_cutoff_and_source_order_budget_ties():
    labels = np.array([0, 1, 1, 0])
    result = policy_metrics(labels, np.array([.9, .9, .5, .1]), {"seen": np.array([False, True, True, True])})
    assert result["overall"]["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["review_budgets"]["1"]["recall"] == 0
    assert result["entities"]["seen"]["rows"] == 3
