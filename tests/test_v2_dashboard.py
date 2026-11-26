"""V2 panels load only frozen results from the selected V1 lineage."""

from hashlib import sha256
import json
from types import SimpleNamespace

from src.dashboard.v2_results import compatible_root


def test_v2_panels_require_the_frozen_full_retrospective_lineage(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUDLENS_V2_ARTIFACTS", str(tmp_path))
    run = SimpleNamespace(mode="full", digests={1: "example-study"}, partitions={"test": {"rows": 2}})
    assert compatible_root(run, "test") is None
    decision = tmp_path / "decision_freeze.json"
    decision.write_text('{"champion":"tree"}', encoding="utf-8")
    documents = {
        "retrospective_results.json": {"rows": 2, "scope": "reused retrospective holdout", "champion": "tree"},
        "features.json": {"v1_manifest_sha256": "example-study"},
        "retrospective_access.json": {"holdout_rows": 2, "decision_freeze_sha256": sha256(decision.read_bytes()).hexdigest()},
    }
    for name, data in documents.items():
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    assert compatible_root(run, "validation") is None
    assert compatible_root(run, "test") == tmp_path
    run.digests[1] = "unrelated-study"
    assert compatible_root(run, "test") is None
