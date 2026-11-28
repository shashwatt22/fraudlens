"""Verify the public export against its build and exercise the standalone entrypoint."""

from hashlib import sha256
import json
from pathlib import Path
import re

import pytest
from streamlit.testing.v1 import AppTest

from scripts.build_public_results import build
from scripts.package_public_demo import FILES, package
from src.dashboard_v3.store import Paths

ROOT = Path(__file__).resolve().parents[1]


def test_public_package_is_explicit_and_only_contains_safe_assets(tmp_path):
    package(tmp_path)
    assert {file.relative_to(tmp_path).as_posix() for file in tmp_path.rglob("*") if file.is_file()} == set(FILES.values())
    with pytest.raises(ValueError, match="empty"):
        package(tmp_path)
    for file in tmp_path.rglob("*"):
        if file.is_file():
            assert not re.search(rb"(?i)[E]:[/\\]|FRAUDLENS_SMTP|TransactionID|isFraud|predictions.parquet|model.ubj",
                                 file.read_bytes()), file.name
    config = (tmp_path / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "enableStaticServing = false" in config and "address =" not in config
    payload = (tmp_path / "results.json").read_bytes()
    assert sha256(payload).hexdigest() in (tmp_path / "app.py").read_text(encoding="utf-8")
    data = json.loads(payload)
    assert all(len(model["curves"]["roc"]["true_positive_rate"]) <= 101 for model in data["models"].values())
    assert all(point["count"] >= 100 for model in data["models"].values() for point in model["curves"]["reliability"])
    assert data["models"]["xgboost"]["frozen"]["reviewed"] == 3378
    assert data["models"]["xgboost"]["frozen"]["threshold"] == pytest.approx(.1938110143)


def test_public_export_is_reproducible_from_verified_local_reports():
    paths = Paths.from_env(ROOT)
    if not (paths.retrospective / "dashboard.json").is_file():
        pytest.skip("Frozen local V3 reports are required for source parity.")
    result = build(paths.v3, paths.legacy / "full", paths.v2)
    assert result == json.loads((ROOT / "public_demo" / "results.json").read_bytes())


def test_public_ui_never_exposes_case_or_delivery_widgets(tmp_path, monkeypatch):
    package(tmp_path)
    monkeypatch.setenv("FRAUDLENS_PUBLIC_DEMO", "0")
    monkeypatch.setenv("FRAUDLENS_SMTP_ENABLED", "1")
    monkeypatch.setenv("FRAUDLENS_ARTIFACTS", str(tmp_path / "absent"))
    app = AppTest.from_file(str(tmp_path / "app.py")).run()
    assert not app.exception and not app.error
    assert app.radio(key="public_nav").options == ["Dashboard", "Check Transaction & Alerts"]
    app.radio(key="public_nav").set_value("Check Transaction & Alerts").run()
    assert not app.exception and not app.error
    assert not app.number_input and not app.text_input and not app.file_uploader
    assert not app.button and not app.checkbox
