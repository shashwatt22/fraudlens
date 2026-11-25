"""Binary logistic regression and a one-hidden-layer network using NumPy only."""

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(-np.abs(logits))
    return np.where(logits >= 0, 1 / (1 + exp), exp / (1 + exp))


def weighted_bce(logits: np.ndarray, labels: np.ndarray, pos_weight: float) -> float:
    """Reduce by the number of observations, including after class weighting."""
    terms = (1 - labels) * np.logaddexp(0, logits) + pos_weight * labels * np.logaddexp(0, -logits)
    return float(np.mean(terms, dtype=np.float64))


def positive_weight(labels: np.ndarray) -> float:
    positives = np.count_nonzero(labels == 1)
    if not 0 < positives < len(labels):
        raise ValueError("Positive weighting requires both training classes.")
    return (len(labels) - positives) / positives


class ScratchNetwork:
    """A linear logit or a ReLU hidden layer followed by one linear logit."""

    def __init__(self, input_dim: int, hidden_width: int = 0, seed: int = 42, dtype=np.float32):
        self.input_dim = input_dim
        self.hidden_width = hidden_width
        rng = np.random.default_rng(seed)
        self.parameters = {}
        if hidden_width:
            self.parameters["W1"] = rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden_width)).astype(dtype)
            self.parameters["b1"] = np.zeros(hidden_width, dtype=dtype)
        output_width = hidden_width or input_dim
        self.parameters["W2"] = (rng.normal(0, np.sqrt(1 / output_width), (output_width, 1)).astype(dtype)
                                 if hidden_width else np.zeros((output_width, 1), dtype=dtype))
        self.parameters["b2"] = np.zeros(1, dtype=dtype)

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        p = self.parameters
        hidden_logits = inputs @ p["W1"] + p["b1"] if self.hidden_width else None
        hidden = np.maximum(hidden_logits, 0) if hidden_logits is not None else inputs
        return (hidden @ p["W2"] + p["b2"]).ravel(), hidden, hidden_logits

    def logits(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)[0]

    def predict_scores(self, inputs: np.ndarray) -> np.ndarray:
        return sigmoid(self.logits(inputs))

    def predict(self, inputs: np.ndarray, threshold: float) -> np.ndarray:
        return (self.predict_scores(inputs) >= threshold).astype(np.uint8)

    def objective(self, inputs: np.ndarray, labels: np.ndarray, pos_weight: float, l2: float) -> float:
        weights_squared = sum(np.sum(value ** 2, dtype=np.float64) for name, value in self.parameters.items()
                              if name.startswith("W"))
        return weighted_bce(self.logits(inputs), labels, pos_weight) + .5 * l2 * weights_squared

    def gradients(self, inputs: np.ndarray, labels: np.ndarray, pos_weight: float, l2: float) -> dict:
        p = self.parameters
        logits, hidden, hidden_logits = self.forward(inputs)
        residual = ((1 - labels) * sigmoid(logits) - pos_weight * labels * sigmoid(-logits)) / len(labels)
        dz = residual[:, None]
        gradients = {"W2": hidden.T @ dz + l2 * p["W2"], "b2": dz.sum(axis=0)}
        if hidden_logits is not None:
            dh = (dz @ p["W2"].T) * (hidden_logits > 0)
            gradients.update(W1=inputs.T @ dh + l2 * p["W1"], b1=dh.sum(axis=0))
        return gradients

    def update(self, inputs: np.ndarray, labels: np.ndarray, pos_weight: float, l2: float, rate: float) -> None:
        for name, gradient in self.gradients(inputs, labels, pos_weight, l2).items():
            self.parameters[name] -= rate * gradient

    def snapshot(self) -> dict[str, np.ndarray]:
        return {name: values.copy() for name, values in self.parameters.items()}

    def restore(self, parameters: dict[str, np.ndarray]) -> None:
        for name, value in self.parameters.items():
            value[...] = parameters[name]

    @property
    def parameter_count(self) -> int:
        return sum(value.size for value in self.parameters.values())

    @classmethod
    def load(cls, path):
        """Reload NumPy weights without executable pickle data."""
        with np.load(path, allow_pickle=False) as saved:
            parameters = dict(saved)
        input_dim = parameters["W1"].shape[0] if "W1" in parameters else parameters["W2"].shape[0]
        hidden_width = len(parameters["b1"]) if "b1" in parameters else 0
        model = cls(input_dim, hidden_width, dtype=parameters["W2"].dtype)
        model.restore(parameters)
        return model
