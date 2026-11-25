"""Synthetic checks for the shared holdout release and artifact-mode fallback."""

import json

import pytest

from src import pipeline
from src.dashboard import artifacts
from src.training_data import TEST_SEALED, load_partition


def test_shared_gate_blocks_a_test_partition_read_before_io():
    token = TEST_SEALED.set(True)
    try:
        with pytest.raises(RuntimeError, match="Test labels are sealed"):
            load_partition(None, {}, "test")
    finally:
        TEST_SEALED.reset(token)


def test_full_coordinator_freezes_every_phase_before_first_test(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    data = tmp_path / "raw"
    data.mkdir()
    full = root / "full"
    release = full / "holdout_release.json"
    observed = []

    def freeze(data, root, mode_dir, threads):
        (mode_dir / "protocol_freeze.json").write_text("{}")
        return {}

    def prepare(name, returns_draft=False):
        def phase(root, mode, threads, **kwargs):
            assert TEST_SEALED.get() and not release.exists()
            observed.append(("validation", name))
            folder = full / name
            folder.mkdir()
            (folder / "selection.json").write_text("{}")
            yield (folder, {"champion": "xgboost"}) if returns_draft else folder
            assert not TEST_SEALED.get() and release.is_file()
            observed.append(("test", name))
        return phase

    def patterns(*args):
        assert TEST_SEALED.get() and not release.exists()
        observed.append(("validation", "patterns"))
        path = full / "patterns.json"
        path.write_text("{}")
        return path

    monkeypatch.setattr(pipeline, "_record_protocol", freeze)
    monkeypatch.setattr(pipeline, "_separation", lambda root: {"overlap_ids": 0})
    monkeypatch.setattr(pipeline, "read_phase1_manifest", lambda path: ({"processed_rows": 590_540}, "source"))
    monkeypatch.setattr(pipeline.phase1, "run_pipeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline.train, "prepare_training", prepare("core", returns_draft=True))
    monkeypatch.setattr(pipeline.train_mlp, "prepare_training", prepare("mlp"))
    monkeypatch.setattr(pipeline.phase3, "prepare_phase3", prepare("risk"))
    monkeypatch.setattr(pipeline.phase5, "run_phase5", patterns)
    monkeypatch.setattr(pipeline.phase6, "prepare_phase6", prepare("lab"))
    assert pipeline.run_pipeline(data, root, "full", threads=1).is_file()
    assert observed == [("validation", phase) for phase in ("core", "mlp", "risk", "patterns", "lab")] + [
        ("test", phase) for phase in ("core", "mlp", "risk", "lab")]
    assert json.loads(release.read_bytes())["frozen_artifacts"]["phase2"]["selection.json"]
    assert not TEST_SEALED.get()
    with pytest.raises(FileExistsError, match="already released"):
        pipeline.run_pipeline(data, root, "full", threads=1)


def test_mode_order_checks_full_artifact_compatibility(tmp_path, monkeypatch):
    for mode in ("quick", "full"):
        directory = tmp_path / mode
        directory.mkdir()
        (directory / "manifest.json").touch()

    class Full:
        def require_complete(self):
            if not (tmp_path / "complete").exists():
                raise artifacts.ArtifactError(6, "Missing subset benchmark")

    monkeypatch.setattr(artifacts, "load_run", lambda root, mode: Full())
    assert artifacts.available_modes(tmp_path) == ["quick", "full"]
    (tmp_path / "complete").touch()
    assert artifacts.available_modes(tmp_path) == ["full", "quick"]
