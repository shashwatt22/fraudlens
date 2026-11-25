"""CPU PyTorch model, batch loading, and validation-driven training."""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import average_precision_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

HIDDEN_WIDTHS = (128, 64)
DROPOUT = 0.25
SEED = 42


class FraudMLP(nn.Module):
    """Return one raw fraud logit for each transaction in a batch."""

    def __init__(self, input_dim: int):
        super().__init__()
        first, second = HIDDEN_WIDTHS
        self.layers = nn.Sequential(
            nn.Linear(input_dim, first), nn.BatchNorm1d(first), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(first, second), nn.BatchNorm1d(second), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(second, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs).squeeze(-1)


class TransactionDataset(Dataset):
    """Keep the source matrix sparse and densify only the requested batch."""

    def __init__(self, matrix: csr_matrix, labels: np.ndarray):
        self.matrix = csr_matrix(matrix, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        if self.labels.shape != (self.matrix.shape[0],):
            raise ValueError("Need one label per encoded transaction.")

    def __len__(self) -> int:
        return self.matrix.shape[0]

    def __getitem__(self, index: int) -> int:
        return index

    def collate(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.matrix[indices].toarray()), torch.from_numpy(self.labels[indices])


def make_loader(matrix: csr_matrix, labels: np.ndarray, batch_size: int, training: bool,
                seed: int = SEED) -> DataLoader:
    dataset = TransactionDataset(matrix, labels)
    if batch_size < 2 or len(dataset) < (2 if training else 1):
        raise ValueError("Use batches of at least two and provide enough rows for this partition.")
    # Drop only a final training singleton, which BatchNorm cannot use in training mode.
    drop_singleton = training and len(dataset) % batch_size == 1
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, drop_last=drop_singleton,
                      num_workers=0, collate_fn=dataset.collate, generator=torch.Generator().manual_seed(seed))


def positive_weight(training_labels: np.ndarray) -> float:
    if set(np.unique(training_labels)) != {0, 1}:
        raise ValueError("Class weighting requires both classes in the training partition.")
    positives = np.count_nonzero(training_labels == 1)
    return (len(training_labels) - positives) / positives


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    max_epochs: int = 30
    patience: int = 5
    seed: int = SEED


def _finite_loss(loss: torch.Tensor) -> float:
    value = loss.item()
    if not isfinite(value):
        raise ValueError("The weighted BCE loss became nonfinite.")
    return value


def train_epoch(model: FraudMLP, loader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer) -> float:
    model.train()
    loss_sum, rows = 0.0, 0
    for inputs, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), labels)
        loss_sum += _finite_loss(loss) * len(labels)
        loss.backward()
        optimizer.step()
        rows += len(labels)
    return loss_sum / rows


@torch.inference_mode()
def score_batches(model: FraudMLP, loader: DataLoader, criterion: nn.Module) -> tuple[float, np.ndarray]:
    model.eval()
    loss_sum, rows = 0.0, 0
    scores = []
    for inputs, labels in loader:
        logits = model(inputs)
        if not torch.isfinite(logits).all():
            raise ValueError("The model produced nonfinite logits.")
        loss_sum += _finite_loss(criterion(logits, labels)) * len(labels)
        scores.append(torch.sigmoid(logits).numpy())
        rows += len(labels)
    return loss_sum / rows, np.concatenate(scores)


def fit_model(model: FraudMLP, train: DataLoader, validation: DataLoader, config: TrainingConfig) -> dict:
    """Train with Adam and restore the earliest checkpoint attaining the best validation AP."""
    if config.max_epochs < 1 or config.patience < 1:
        raise ValueError("The epoch limit and patience must be positive.")
    pos_weight = positive_weight(train.dataset.labels)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    best_ap = -np.inf
    best_epoch, best_state = 0, None
    stop_reason = "max_epochs"
    started = perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        epoch_started = perf_counter()
        train_loss = train_epoch(model, train, criterion, optimizer)
        val_loss, scores = score_batches(model, validation, criterion)
        validation_ap = float(average_precision_score(validation.dataset.labels, scores))
        if validation_ap > best_ap:
            best_ap, best_epoch = validation_ap, epoch
            best_state = deepcopy(model.state_dict())  # Includes BatchNorm running statistics.

        elapsed = perf_counter() - epoch_started
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                        "validation_ap": validation_ap, "learning_rate": optimizer.param_groups[0]["lr"],
                        "elapsed_seconds": elapsed})
        print(f"Epoch {epoch:02d}  train {train_loss:.5f}  val {val_loss:.5f}  "
              f"val AP {validation_ap:.6f}  lr {config.learning_rate:g}  {elapsed:.2f}s", flush=True)
        if epoch - best_epoch >= config.patience:
            stop_reason = "validation_ap_patience"
            break

    model.load_state_dict(best_state)
    model.eval()
    return {"epochs": history, "epochs_completed": len(history), "best_epoch": best_epoch,
            "best_validation_ap": best_ap, "epochs_since_best": len(history) - best_epoch,
            "stop_reason": stop_reason, "training_seconds": perf_counter() - started,
            "positive_class_weight": pos_weight,
            "dropped_training_rows_per_epoch": int(train.drop_last)}


def load_checkpoint(path, input_dim: int) -> FraudMLP:
    """Reload a CPU checkpoint containing only model tensors."""
    model = FraudMLP(input_dim)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()
