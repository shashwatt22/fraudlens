"""Resolve and cache the matching, immutable artifacts used by the dashboard."""

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import streamlit as st

PHASES = (1, 2, "2B", 3)
OPTIONAL = (5, 6)
FOLDERS = {1: "", 2: "phase2", "2B": "phase2b", 3: "phase3", 5: "phase5", 6: "phase6"}
MODULES = {1: "phase1", 2: "train", "2B": "train_mlp", 3: "phase3", 5: "phase5", 6: "phase6"}
PARTITIONS = ("train", "validation", "test")


class ArtifactError(Exception):
    def __init__(self, phase, message):
        super().__init__(message)
        self.phase = phase


def generation_commands(mode: str, through=3) -> str:
    if mode == "full":
        return r".\.venv\Scripts\python.exe -m src.pipeline --mode full --threads 6"
    if through in OPTIONAL:
        return f".\\.venv\\Scripts\\python.exe -m src.{MODULES[through]} --mode {mode} --threads 6"
    end = PHASES.index(through) + 1
    return "\n".join(f".\\.venv\\Scripts\\python.exe -m src.{MODULES[phase]} --mode {mode}"
                     for phase in PHASES[:end])


def available_modes(root: Path) -> list[str]:
    modes = [mode for mode in ("quick", "smoke", "full") if (root / mode / "manifest.json").is_file()]
    if "full" in modes:
        try:
            load_run(root, "full").require_complete()
        except ArtifactError:
            pass
        else:
            modes.remove("full")
            modes.insert(0, "full")
    return modes or ["quick"]


@st.cache_data(max_entries=32, show_spinner=False)
def _json(path: Path) -> dict:
    return json.loads(path.read_bytes())


@st.cache_data(max_entries=28, show_spinner=False)
def _parquet(path: Path, columns: tuple | None = None, transaction_id: int | None = None) -> pd.DataFrame:
    filters = None if transaction_id is None else [("TransactionID", "=", transaction_id)]
    return pd.read_parquet(path, columns=columns, filters=filters)


class DashboardRun:
    """A single dataset mode with lineage checks across the four offline runs."""

    def __init__(self, root: Path, mode: str):
        self.root, self.mode = root, mode
        self.mode_dir = root / mode
        self.manifests, self.digests = {}, {}
        for phase in PHASES:
            path = self.mode_dir / FOLDERS[phase] / "manifest.json"
            try:
                data = path.read_bytes()
                manifest = json.loads(data)
                if (manifest.get("phase"), manifest.get("schema_version"), manifest.get("mode")) != (phase, 1, mode):
                    raise ValueError("unexpected phase, dataset mode, or schema version")
                self.manifests[phase] = manifest
                self.digests[phase] = sha256(data).hexdigest()
                if phase != 1 and (manifest["phase1_manifest_sha256"] != self.digests[1]
                                   or manifest["partitions"] != self.manifests[1]["partitions"]):
                    raise ValueError("this run refers to a different Phase 1 snapshot")
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                raise ArtifactError(phase, f"Could not read the Phase {phase} {mode} manifest at {path}. {error}") from error

        try:
            if self.manifests["2B"]["preprocessor"]["classical_manifest_sha256"] != self.digests[2]:
                raise ArtifactError("2B", "The neural run refers to a different Phase 2 snapshot.")
            if self.manifests[3]["classifier"]["phase2_manifest_sha256"] != self.digests[2]:
                raise ArtifactError(3, "The risk and SHAP run refers to a different Phase 2 snapshot.")
            classifier = self.manifests[3]["classifier"]
            if classifier["champion"] != self.manifests[2]["champion"] or classifier["threshold"] != (
                self.manifests[2]["models"][classifier["champion"]]["threshold"]
            ):
                raise ArtifactError(3, "The risk and SHAP run has a different deployed classifier or frozen threshold.")
        except (KeyError, TypeError) as error:
            raise ArtifactError(3, f"The model lineage metadata is incomplete. {error}") from error

        self.optional_errors = {phase: self._load_optional(phase) for phase in OPTIONAL}

    def _load_optional(self, phase) -> str | None:
        path = self.mode_dir / FOLDERS[phase] / "manifest.json"
        try:
            data = path.read_bytes()
            manifest = json.loads(data)
            if (manifest.get("phase"), manifest.get("schema_version"), manifest.get("mode")) != (phase, 1, self.mode):
                raise ValueError("unexpected phase, dataset mode, or schema version")
            if manifest["phase1_manifest_sha256"] != self.digests[1] or manifest["partitions"] != self.manifests[1]["partitions"]:
                raise ValueError("this run refers to a different Phase 1 snapshot")
            self.manifests[phase] = manifest
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            return f"Could not read the Phase {phase} {self.mode} manifest. {error}"
        return None

    def require_complete(self) -> None:
        """Check file presence and run lineage before offering full as the default."""
        for phase in OPTIONAL:
            self.require_optional(phase)
        for phase, manifest in self.manifests.items():
            files = manifest.get("files")
            if not isinstance(files, dict) or not files:
                raise ArtifactError(phase, f"Phase {phase} is missing its artifact index.")
            for key, value in files.items():
                if isinstance(value, dict):
                    for member in value:
                        self.path(phase, key, member)
                else:
                    self.path(phase, key)

    def require_patterns(self) -> None:
        self.require_optional(5)

    def require_optional(self, phase) -> None:
        error = self.optional_errors[phase]
        if error is not None:
            raise ArtifactError(phase, error)

    @property
    def partitions(self) -> dict:
        return self.manifests[1]["partitions"]

    def partition_name(self, partition: str) -> str:
        if partition == "test":
            return {"quick": "Quick test", "full": "Final test"}.get(self.mode, "Test")
        return partition.capitalize()

    def path(self, phase, key: str, member: str | None = None) -> Path:
        try:
            manifest = self.manifests[phase]
            base = self.mode_dir / FOLDERS[phase]
            if phase != 1:
                base /= manifest["run_directory"]
            relative = manifest["files"][key]
            path = (base / (relative if member is None else relative[member])).resolve()
            if not path.is_relative_to(base.resolve()) or not path.is_relative_to(self.mode_dir.resolve()) or not path.is_file():
                raise FileNotFoundError(path)
            return path
        except (OSError, KeyError, TypeError) as error:
            raise ArtifactError(phase, f"Phase {phase} file '{key}' is missing or invalid. {error}") from error

    def report(self, phase, key: str) -> dict:
        path = self.path(phase, key)
        try:
            report = _json(path)
            if report.get("mode") != self.mode:
                raise ValueError("dataset mode differs from the manifest")
            return report
        except (OSError, ValueError, TypeError, AttributeError) as error:
            raise ArtifactError(phase, f"Could not read {path.name}. {error}") from error

    def table(self, phase, key: str, columns=None, transaction_id: int | None = None) -> pd.DataFrame:
        path = self.path(phase, key)
        try:
            return _parquet(path, None if columns is None else tuple(columns), transaction_id)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ArtifactError(phase, f"Could not read {path.name}. {error}") from error

    def risk(self, partition: str) -> pd.DataFrame:
        table = self.table(3, f"{partition}_risk")
        expected = {"TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "classifier_score",
                    "classifier_decision", "anomaly_score", "anomaly_percentile", "risk_score", "review_priority",
                    "identity_missing", "prior_txn_count", "prior_mean_amount", "amount_to_prior_mean",
                    "seconds_since_prior_txn", "prior_count_1h", "prior_count_24h", "is_new_product"}
        if (expected - set(table.columns) or len(table) != self.partitions[partition]["rows"]
                or table["TransactionID"].isna().any() or not table["TransactionID"].is_unique):
            raise ArtifactError(3, f"The {self.partition_name(partition)} risk table is incomplete.")
        return table

    def metrics(self, partition: str) -> dict:
        classical = self.report(2, f"{partition}_metrics")
        neural = self.report("2B", f"{partition}_metrics")
        for phase, report in ((2, classical), ("2B", neural)):
            if (report.get("partition") != partition or report.get("rows") != self.partitions[partition]["rows"]
                    or not isinstance(report.get("models"), dict)):
                raise ArtifactError(phase, "Metrics do not match the selected chronological partition.")
        return {**classical["models"], **neural["models"]}


@st.cache_resource(max_entries=4, show_spinner=False)
def _dashboard(root: Path, mode: str, signature: tuple) -> DashboardRun:
    return DashboardRun(root, mode)


def load_run(root: Path, mode: str) -> DashboardRun:
    signature = []
    for phase in (*PHASES, *OPTIONAL):
        path = root / mode / FOLDERS[phase] / "manifest.json"
        try:
            stat = path.stat()
            signature.append((stat.st_mtime_ns, stat.st_size))
        except OSError:
            signature.append(None)
    return _dashboard(root, mode, tuple(signature))


@st.cache_resource(max_entries=4, show_spinner=False)
def _phase3_store(root: Path, mode: str, fingerprint: str):
    from src.phase3 import Phase3Store

    return Phase3Store(root, mode)


@st.cache_data(max_entries=24, show_spinner=False)
def explain_case(root: Path, mode: str, fingerprint: str, partition: str, transaction_id: int) -> dict:
    try:
        return _phase3_store(root, mode, fingerprint).assess(partition, transaction_id)["shap"]
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        raise ArtifactError(3, f"On-demand SHAP is unavailable for transaction {transaction_id}. {error}") from error
