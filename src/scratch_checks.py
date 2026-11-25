"""Independent finite differences and a PyTorch autograd reference."""

import numpy as np
import torch
from torch import nn

from .scratch_neural import ScratchNetwork


class TorchReference:
    """Same architecture, initial weights and SGD as ScratchNetwork, using autograd."""

    def __init__(self, initial: ScratchNetwork):
        dtype = torch.float64 if initial.parameters["W2"].dtype == np.float64 else torch.float32
        self.input_dim, self.hidden_width = initial.input_dim, initial.hidden_width
        self.layers = {}
        if initial.hidden_width:
            self.layers["1"] = nn.Linear(self.input_dim, self.hidden_width, dtype=dtype)
        self.layers["2"] = nn.Linear(self.hidden_width or self.input_dim, 1, dtype=dtype)
        self.net = (nn.Sequential(self.layers["1"], nn.ReLU(), self.layers["2"]) if self.hidden_width else self.layers["2"])
        self.restore(initial.snapshot())
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=1.)

    def _loss(self, inputs, labels, pos_weight, l2):
        logits = self.net(torch.from_numpy(inputs)).flatten()
        weight = logits.new_tensor([pos_weight])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(labels), pos_weight=weight)
        return loss + .5 * l2 * sum(layer.weight.square().sum() for layer in self.layers.values())

    @torch.no_grad()
    def logits(self, inputs):
        return self.net(torch.from_numpy(inputs)).flatten().numpy()

    @torch.no_grad()
    def predict_scores(self, inputs):
        return torch.sigmoid(self.net(torch.from_numpy(inputs)).flatten()).numpy()

    @torch.no_grad()
    def objective(self, inputs, labels, pos_weight, l2):
        return float(self._loss(inputs, labels, pos_weight, l2))

    def gradients(self, inputs, labels, pos_weight, l2):
        self.optimizer.zero_grad(set_to_none=True)
        self._loss(inputs, labels, pos_weight, l2).backward()
        return {name: value for number, layer in self.layers.items()
                for name, value in ((f"W{number}", layer.weight.grad.detach().numpy().T.copy()),
                                    (f"b{number}", layer.bias.grad.detach().numpy().copy()))}

    def update(self, inputs, labels, pos_weight, l2, rate):
        self.optimizer.param_groups[0]["lr"] = rate
        self.optimizer.zero_grad(set_to_none=True)
        self._loss(inputs, labels, pos_weight, l2).backward()
        self.optimizer.step()

    @torch.no_grad()
    def snapshot(self):
        return {name: value for number, layer in self.layers.items()
                for name, value in ((f"W{number}", layer.weight.numpy().T.copy()), (f"b{number}", layer.bias.numpy().copy()))}

    @torch.no_grad()
    def restore(self, parameters):
        for number, layer in self.layers.items():
            layer.weight.copy_(torch.from_numpy(parameters[f"W{number}"].T))
            layer.bias.copy_(torch.from_numpy(parameters[f"b{number}"]))

    @property
    def parameter_count(self):
        return sum(value.numel() for value in self.net.parameters())


def finite_differences(model, inputs, labels, pos_weight, l2, step=1e-5) -> dict:
    analytical = model.gradients(inputs, labels, pos_weight, l2)
    errors = {}
    for name, parameter in model.parameters.items():
        numerical = np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + step
            plus = model.objective(inputs, labels, pos_weight, l2)
            parameter[index] = original - step
            minus = model.objective(inputs, labels, pos_weight, l2)
            parameter[index] = original
            numerical[index] = (plus - minus) / (2 * step)
        difference = np.abs(numerical - analytical[name])
        scale = np.maximum(1e-12, np.abs(numerical) + np.abs(analytical[name]))
        errors[name] = {"max_absolute": float(np.max(difference)), "max_relative": float(np.max(difference / scale))}
    return errors


def _max_error(left, right):
    return float(np.max(np.abs(left - right)))


def verify_scratch() -> dict:
    """Validate both exact gradients and learning before touching real data."""
    torch.set_num_threads(1)
    inputs = np.array([[.3, -.4], [.7, .2], [-.2, .8], [-.6, -.1]], dtype=np.float64)
    labels = np.array([0., 1., 0., 0.])
    pos_weight, l2, rate = 3., .07, .035
    report = {}
    for width, name in ((0, "numpy_linear"), (3, "numpy_relu32")):
        model = ScratchNetwork(2, width, dtype=np.float64)
        model.parameters["b2"][:] = .21
        if width:
            model.parameters["b1"][:] = [.13, -.19, .23]
            if np.min(np.abs(model.forward(inputs)[2])) < .001:
                raise ValueError("Gradient fixture is too close to a ReLU kink.")
        else:
            model.parameters["W2"][:] = [[.27], [-.43]]
        numeric = finite_differences(model, inputs, labels, pos_weight, l2)
        reference = TorchReference(model)
        gradients = model.gradients(inputs, labels, pos_weight, l2)
        autograd = reference.gradients(inputs, labels, pos_weight, l2)
        check = {"logits": _max_error(model.logits(inputs), reference.logits(inputs)),
                 "loss": abs(model.objective(inputs, labels, pos_weight, l2) - reference.objective(inputs, labels, pos_weight, l2)),
                 "gradients": {key: _max_error(value, autograd[key]) for key, value in gradients.items()}}
        model.update(inputs, labels, pos_weight, l2, rate)
        reference.update(inputs, labels, pos_weight, l2, rate)
        updated = reference.snapshot()
        check["one_update"] = {key: _max_error(value, updated[key]) for key, value in model.parameters.items()}
        finite_error = max(value["max_absolute"] for value in numeric.values())
        torch_error = max(check["logits"], check["loss"], *check["gradients"].values(), *check["one_update"].values())
        if finite_error >= 1e-8 or torch_error >= 1e-12:
            raise ValueError(f"Scratch {name} verification failed: finite difference {finite_error:g}, PyTorch {torch_error:g}.")
        report[name] = {"fixture_input_dim": 2, "fixture_hidden_width": width, "finite_differences": numeric,
                        "max_gradient_absolute": finite_error, "pytorch": check, "max_pytorch_absolute": torch_error}

    x = np.array([[-2., -1.], [-1., -.5], [-2., .5], [.9, .2], [1., -.6], [2., 1.]])
    y = np.array([0., 0., 0., 1., 1., 1.])
    for width, name in ((0, "numpy_linear"), (32, "numpy_relu32")):
        model = ScratchNetwork(2, width, seed=42, dtype=np.float64)
        before = {"loss": model.objective(x, y, 1., 0.), "accuracy": float(np.mean(model.predict(x, .5) == y))}
        for _ in range(100):
            model.update(x, y, 1., 0., .08)
        after = {"loss": model.objective(x, y, 1., 0.), "accuracy": float(np.mean(model.predict(x, .5) == y))}
        if not (after["loss"] < before["loss"] and after["accuracy"] > before["accuracy"]):
            raise ValueError(f"{name} failed to learn the synthetic fixture.")
        report[name]["learning_fixture"] = {"steps": 100, "before": before, "after": after}
    return {"dtype": "float64", "finite_difference_step": 1e-5, "positive_weight": pos_weight,
            "l2": l2, "gradient_absolute_tolerance": 1e-8, "pytorch_absolute_tolerance": 1e-12, "models": report}
