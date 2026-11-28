import numpy as np
import pytest
import torch

from src.v3 import torch_full


def test_chronological_cut_keeps_equal_timestamps_in_gradient_block():
    times = np.r_[np.arange(17), [17, 17, 19]]
    assert torch_full.gradient_end(times) == 19


def test_architecture_matches_original_and_outputs_a_single_logit():
    model = torch_full.network(224).eval()
    assert sum(p.numel() for p in model.parameters()) == 67105
    assert model(torch.zeros(2, 224)).shape == (2, 1)


def test_epoch_checkpoint_resumes_with_matching_weights(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(32, 8)).astype(np.float32)
    y = np.tile([0, 1], 16).astype(np.uint8)
    args = (X[:24], y[:24], X[24:], y[24:])
    monkeypatch.setattr(torch_full, "EPOCHS", 2)
    torch.manual_seed(42)
    original = torch_full.network(8)
    h1, best1 = torch_full.fit_epochs(original, *args, tmp_path / "original.pt", {})

    save = torch_full.save_atomic

    def stop_after_checkpoint(payload, path, writer):
        save(payload, path, writer)
        raise InterruptedError("Simulated interruption after epoch one")

    monkeypatch.setattr(torch_full, "save_atomic", stop_after_checkpoint)
    torch.manual_seed(42)
    interrupted = torch_full.network(8)
    with pytest.raises(InterruptedError):
        torch_full.fit_epochs(interrupted, *args, tmp_path / "resume.pt", {})

    monkeypatch.setattr(torch_full, "save_atomic", save)
    restored = torch_full.network(8)
    h2, best2 = torch_full.fit_epochs(restored, *args, tmp_path / "resume.pt", {})
    assert best1 == best2 and len(h1) == len(h2) == 2
    assert [{k: v for k, v in epoch.items() if k != "seconds"} for epoch in h1] == [
        {k: v for k, v in epoch.items() if k != "seconds"} for epoch in h2]
    np.testing.assert_array_equal(torch_full.score(original, X)[0], torch_full.score(restored, X)[0])
    assert all(np.isfinite(row["internal_validation_loss"]) for row in h1)
