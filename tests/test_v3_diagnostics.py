import numpy as np

from src.v3.diagnostics import fit_population, metrics_at_cutoff, seen_groups


def test_diagnostic_turns_cutoff_into_decisions_including_ties():
    results = metrics_at_cutoff(np.array([0, 1, 0, 1]), np.array([.2, .5, .8, .9]), .5)
    assert results["confusion_matrix"] == [[1, 1], [0, 2]]
    assert results["recall"] == 1


def test_seen_groups_use_only_actual_fit_rows_and_separate_missing_keys():
    keys = np.array([1, 2, -1, 3, 1, 2, -1, 3, 4, 4])
    groups = seen_groups(keys, np.array([0, 2]), 4, 10)
    assert groups["seen_in_fit"].tolist() == [True, False, False, False, False, False]
    assert groups["unseen_in_fit"].tolist() == [False, True, False, True, True, True]
    assert groups["missing_key"].tolist() == [False, False, True, False, False, False]


def test_torch_early_stop_role_respects_a_tied_boundary():
    times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 11, 12])
    labels = np.array([0, 1] * 6)
    fitted, early_stop = fit_population("torch_mlp_regularized", 11, times, labels, 10)
    np.testing.assert_array_equal(fitted, np.arange(10))
    assert early_stop == 1
