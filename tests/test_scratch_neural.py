"""Independent gradient, stability, learning and reload checks on synthetic data."""

import numpy as np
import pytest

from src.scratch_checks import verify_scratch
from src.scratch_neural import ScratchNetwork, positive_weight, sigmoid, weighted_bce


def test_sigmoid_and_weighted_bce_are_stable_at_extreme_logits():
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_array_equal(sigmoid(np.array([-1_000., 0., 1_000.])), [0., .5, 1.])
        assert weighted_bce(np.array([-1_000., 1_000.]), np.array([0., 1.]), 3.) == 0
        assert weighted_bce(np.array([-1_000., 1_000.]), np.array([1., 0.]), 3.) == 2_000
    assert positive_weight(np.array([1, 0, 0, 0])) == 3


def test_hand_worked_relu_forward_gradient_and_bias_updates():
    model = ScratchNetwork(2, 2, dtype=np.float64)
    model.restore({"W1": np.array([[.5, -.5], [.25, -.25]]), "b1": np.zeros(2),
                   "W2": np.array([[.4], [-.3]]), "b2": np.array([.1])})
    inputs, target = np.array([[1., 2.]]), np.array([1.])
    logits, hidden, preactivation = model.forward(inputs)
    np.testing.assert_allclose(logits, [.5])
    np.testing.assert_array_equal(hidden, [[1., 0.]])
    np.testing.assert_array_equal(preactivation, [[1., -1.]])
    dz = 3 * (1 / (1 + np.exp(-.5)) - 1)
    gradients = model.gradients(inputs, target, 3., .2)
    np.testing.assert_allclose(gradients["b2"], [dz])
    np.testing.assert_allclose(gradients["b1"], [.4 * dz, 0])
    np.testing.assert_allclose(gradients["W2"], [[dz + .08], [-.06]])
    np.testing.assert_allclose(gradients["W1"], [[.4 * dz + .1, -.1], [.8 * dz + .05, -.05]])
    model.update(inputs, target, 3., .2, .03)
    np.testing.assert_allclose(model.parameters["b2"], [.1 - .03 * dz])
    np.testing.assert_allclose(model.parameters["b1"], [-.03 * .4 * dz, 0])


@pytest.mark.parametrize("width, expected", [(0, 4), (32, 161)])
def test_parameter_shapes_initialization_and_reload(tmp_path, width, expected):
    inputs = np.array([[1, 0, 2], [-4, 2, 3]], dtype=np.float32)
    model = ScratchNetwork(3, width, seed=42)
    other = ScratchNetwork(3, width, seed=42)
    assert model.parameter_count == expected
    for key in model.parameters:
        np.testing.assert_array_equal(model.parameters[key], other.parameters[key])
    assert model.logits(inputs).shape == (2,)
    model.update(inputs, np.array([1., 0.], dtype=np.float32), 2., .01, .03)
    path = tmp_path / "weights.npz"
    np.savez(path, **model.snapshot())
    loaded = ScratchNetwork.load(path)
    np.testing.assert_array_equal(model.predict_scores(inputs), loaded.predict_scores(inputs))
    np.testing.assert_array_equal(model.predict(inputs, .6), loaded.predict(inputs, .6))


def test_finite_difference_autograd_and_learning_checks():
    report = verify_scratch()
    for key in ("numpy_linear", "numpy_relu32"):
        result = report["models"][key]
        assert result["max_gradient_absolute"] < 1e-8
        assert result["max_pytorch_absolute"] < 1e-12
        fixture = result["learning_fixture"]
        assert fixture["before"]["loss"] > fixture["after"]["loss"]
        assert fixture["before"]["accuracy"] < fixture["after"]["accuracy"] == 1
    assert set(report["models"]["numpy_relu32"]["finite_differences"]) == {"W1", "b1", "W2", "b2"}
