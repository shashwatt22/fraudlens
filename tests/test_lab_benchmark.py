"""Sampling and classifier score contracts on small fixtures."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from src.evaluate import evaluate_scores, select_threshold
from src.lab_benchmark import classifier_scores, stratified_positions


def test_stratified_subsets_are_deterministic_and_keep_source_order():
    labels = np.array([0] * 180 + [1] * 20)
    selected = stratified_positions(labels, 50)
    np.testing.assert_array_equal(selected, stratified_positions(labels, 50))
    assert (np.diff(selected) > 0).all() and labels[selected].sum() == 5
    assert not np.array_equal(selected, stratified_positions(labels, 50, seed=5))


def test_svm_margin_ranking_and_negative_validation_cutoff():
    x = np.arange(-3., 3.).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LinearSVC(dual=False).fit(x, y)
    scores = classifier_scores("linear_svm", model, x * 10)
    assert scores.min() < -1 and scores.max() > 1
    np.testing.assert_array_equal(scores, model.decision_function(x * 10))
    margins = np.array([-3., -2., -1., -.5])
    target = np.array([0, 0, 1, 1])
    threshold = select_threshold(target, margins)
    assert threshold == -1.
    result, _ = evaluate_scores(target, margins, (margins >= threshold).astype(int))
    assert result["f1"] == 1 and result["average_precision"] == 1


def test_knn_batches_preserve_prediction_order_and_dimensions():
    rng = np.random.default_rng(42)
    fit = rng.normal(size=(40, 2)).astype(np.float32)
    query = rng.normal(size=(555, 2)).astype(np.float32)
    classifier = KNeighborsClassifier(n_neighbors=15, algorithm="brute", weights="distance", n_jobs=1).fit(fit, fit[:, 0] > 0)
    scores = classifier_scores("knn", classifier, query)
    assert scores.shape == (555,) and ((0 <= scores) & (scores <= 1)).all()
    np.testing.assert_allclose(scores, classifier.predict_proba(query)[:, 1])
