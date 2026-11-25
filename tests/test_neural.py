"""Small synthetic checks for the MLP, its batches, and its training loop."""

import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import torch
from torch import nn

from src import neural


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("input_dim", [3, 331])
def test_dynamic_input_shape_finite_logits_and_weighted_loss(input_dim):
    model = neural.FraudMLP(input_dim)
    with torch.no_grad():
        model.layers[-1].weight.zero_()
        model.layers[-1].bias.fill_(-2.0)
    logits = model(torch.ones(4, input_dim))
    assert logits.shape == (4,)
    torch.testing.assert_close(logits, torch.full((4,), -2.0))
    scores = torch.sigmoid(logits)
    assert torch.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all()
    assert sum(p.numel() for p in model.parameters()) == 128 * input_dim + 8833

    labels = np.array([0, 0, 0, 1])
    weight = neural.positive_weight(labels)
    assert weight == 3
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    loss = criterion(logits, torch.tensor(labels, dtype=torch.float32))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    extreme_loss = criterion(torch.tensor([1000.0, -1000.0]), torch.tensor([0.0, 1.0]))
    assert extreme_loss.item() == pytest.approx(2000)


def test_class_weight_uses_training_counts_and_stays_fixed_for_validation():
    train_y = np.array([0, 1, 0, 0, 0])
    validation_y = np.array([0, 1, 1])
    weight = neural.positive_weight(train_y)
    assert weight == 4
    with pytest.raises(ValueError, match="both classes"):
        neural.positive_weight(np.zeros(4))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    actual = criterion(torch.zeros(3), torch.tensor(validation_y, dtype=torch.float32))
    assert actual.item() == pytest.approx(math.log(2) * (1 + 2 * weight) / 3)


def test_singleton_batches_train_eval_modes_and_validation_order():
    matrix = csr_matrix(np.arange(15, dtype=np.float32).reshape(5, 3))
    labels = np.array([1, 0, 1, 0, 0])
    train = neural.make_loader(matrix, labels, batch_size=2, training=True)
    val = neural.make_loader(matrix, labels, batch_size=2, training=False)
    assert train.dataset.matrix.format == "csr"
    assert [len(y) for _, y in train] == [2, 2]
    assert [len(y) for _, y in val] == [2, 2, 1]
    np.testing.assert_array_equal(torch.cat([y for _, y in val]).numpy(), labels)

    model = neural.FraudMLP(3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neural.positive_weight(labels)))
    optimizer = torch.optim.Adam(model.parameters())
    modes = []
    hook = model.register_forward_pre_hook(lambda m, _: modes.append((m.training, torch.is_grad_enabled())))
    assert math.isfinite(neural.train_epoch(model, train, criterion, optimizer))
    running_mean = model.layers[1].running_mean.clone()
    val_loss, scores = neural.score_batches(model, val, criterion)
    hook.remove()
    assert math.isfinite(val_loss) and scores.shape == (5,)
    assert modes == [(True, True)] * 2 + [(False, False)] * 3
    torch.testing.assert_close(model.layers[1].running_mean, running_mean)
    with torch.inference_mode():
        expected = torch.sigmoid(model(torch.from_numpy(matrix.toarray()))).numpy()
    np.testing.assert_allclose(scores, expected, atol=1e-7, rtol=1e-6)


def test_early_stopping_restores_best_weights_and_batchnorm_buffers(monkeypatch):
    labels = np.array([0, 1, 0, 0])
    matrix = csr_matrix(np.ones((4, 3), dtype=np.float32))
    train = neural.make_loader(matrix, labels, 2, training=True)
    val = neural.make_loader(matrix, labels, 2, training=False)
    model = neural.FraudMLP(3)
    epoch = 0

    def fake_train(model, *_):
        nonlocal epoch
        epoch += 1
        with torch.no_grad():
            model.layers[-1].bias.fill_(epoch)
            model.layers[1].running_mean.fill_(epoch)
        return 1.0

    measured_ap = iter([0.2, 0.4, 0.3, 0.4])
    monkeypatch.setattr(neural, "train_epoch", fake_train)
    monkeypatch.setattr(neural, "average_precision_score", lambda *_: next(measured_ap))
    result = neural.fit_model(model, train, val, neural.TrainingConfig(max_epochs=10, patience=2))
    assert result["epochs_completed"] == 4
    assert result["best_epoch"] == 2
    assert result["best_validation_ap"] == 0.4
    assert result["stop_reason"] == "validation_ap_patience"
    assert model.layers[-1].bias.item() == 2.0
    torch.testing.assert_close(model.layers[1].running_mean, torch.full((128,), 2.0))
    assert not model.training
