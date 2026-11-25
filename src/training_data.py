"""Read the exact Phase 1 partition files and confirm their manifest metadata."""

from contextvars import ContextVar
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .data_loader import ID, TARGET, TIME
from .features import CATEGORICAL_BASIC, HISTORICAL_COLUMNS, METADATA_COLUMNS, NUMERIC_BASIC

PARTITIONS = ("train", "validation", "test")
TEST_SEALED = ContextVar("test_sealed", default=False)


def complete_run(run):
    """Finish a phase after its single validation freeze point."""
    try:
        next(run)
    except StopIteration as result:
        return result.value
    raise RuntimeError("The phase yielded more than once after validation.")


def read_phase1_manifest(mode_dir: Path) -> tuple[dict, str]:
    path = mode_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run python -m src.phase1 --mode {mode_dir.name} first.")
    contents = path.read_bytes()
    manifest = json.loads(contents)
    if (manifest.get("phase"), manifest.get("schema_version"), manifest.get("mode")) != (1, 1, mode_dir.name):
        raise ValueError(f"{path} must be a Phase 1 manifest for mode {mode_dir.name} with schema v1.")

    schema = manifest.get("feature_schema", {})
    expected = {"metadata": METADATA_COLUMNS, "label": TARGET, "basic_numeric": NUMERIC_BASIC,
                "basic_categorical": CATEGORICAL_BASIC, "historical_numeric": HISTORICAL_COLUMNS}
    if any(schema.get(key) != value for key, value in expected.items()):
        raise ValueError(f"The Phase 1 feature schema in {path} does not match the training inputs.")
    try:
        train, validation, test = (manifest["partitions"][name] for name in PARTITIONS)
        if sum(partition["rows"] for partition in (train, validation, test)) != manifest["processed_rows"]:
            raise ValueError("Phase 1 partition counts do not match the processed row total.")
        if not (train["last_second"] < validation["first_second"]
                <= validation["last_second"] < test["first_second"]):
            raise ValueError("Phase 1 partition ranges overlap or are out of chronological order.")
    except KeyError as error:
        raise ValueError(f"Missing Phase 1 partition metadata in {path}: {error}") from error
    return manifest, sha256(contents).hexdigest()


def load_partition(mode_dir: Path, manifest: dict, name: str) -> pd.DataFrame:
    """Load just the model inputs, targets and metadata needed for row alignment."""
    if name == "test" and TEST_SEALED.get():
        raise RuntimeError("Test labels are sealed until every validation decision is frozen.")
    relative_path = manifest.get("files", {}).get(name)
    if not relative_path:
        raise ValueError(f"Phase 1 manifest is missing the {name} Parquet path.")
    path = (mode_dir / relative_path).resolve()
    if not path.is_relative_to(mode_dir.resolve()) or not path.is_file():
        raise FileNotFoundError(f"Missing {name} Parquet artifact inside {mode_dir}: {relative_path}")

    parquet = pq.ParquetFile(path)
    expected = manifest["partitions"][name]
    tags = parquet.metadata.metadata or {}
    expected_tags = {
        b"fraudlens.mode": manifest["mode"].encode(),
        b"fraudlens.partition": name.encode(),
        b"fraudlens.processed_rows": str(manifest["processed_rows"]).encode(),
        b"fraudlens.partition_rows": str(expected["rows"]).encode(),
    }
    if parquet.metadata.num_rows != expected["rows"] or any(tags.get(key) != value for key, value in expected_tags.items()):
        raise ValueError(f"Parquet metadata for {name} does not match the Phase 1 manifest.")

    columns = [ID, TIME, TARGET, *NUMERIC_BASIC, *HISTORICAL_COLUMNS, *CATEGORICAL_BASIC]
    frame = parquet.read(columns=columns).to_pandas()
    if set(columns) - set(frame.columns):
        raise ValueError(f"The {name} Parquet file is missing required columns.")
    if frame[ID].isna().any() or not frame[ID].is_unique:
        raise ValueError(f"The {name} partition needs unique, nonmissing transaction IDs.")
    if set(frame[TARGET].unique()) != {0, 1} or frame[TARGET].sum() != expected["fraud_rows"]:
        raise ValueError(f"The {name} partition labels do not match the Phase 1 manifest.")
    times = frame[TIME]
    if (times.isna().any() or not times.is_monotonic_increasing
            or (times.iloc[0], times.iloc[-1]) != (expected["first_second"], expected["last_second"])):
        raise ValueError(f"The {name} partition chronology does not match the Phase 1 manifest.")
    return frame
