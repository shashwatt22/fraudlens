"""Build reusable Phase 1 data artifacts. Run with python -m src.phase1."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data_loader import load_inputs, merge_identity
from .eda import describe_data
from .features import add_behavioral_features, feature_schema
from .splits import chronological_prefix, chronological_split

QUICK_ROWS = 100_000
SMOKE_ROWS = 10_000
SEED = 42


def _resources(data_dir: Path, output_root: Path) -> dict:
    resources = {
        "logical_processors": os.cpu_count(),
        "data_disk_free_bytes": shutil.disk_usage(data_dir).free,
        "output_disk_free_bytes": shutil.disk_usage(output_root).free,
    }
    if os.name == "nt":
        # MEMORYSTATUSEX starts with two DWORD fields followed by seven uint64 fields.
        status = (ctypes.c_uint64 * 8)()
        status[0] = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            resources["physical_ram_bytes"] = int(status[1])
            resources["available_ram_bytes"] = int(status[2])
    return resources


def _write_json(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _publish_artifacts(mode_dir: Path, partitions: dict, eda: dict, manifest: dict,
                       started: float) -> Path:
    runs_dir = mode_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / uuid4().hex
    manifest["files"] = {name: f"{run_dir.relative_to(mode_dir).as_posix()}/{name}.parquet"
                         for name in partitions}
    manifest["files"]["eda"] = f"{run_dir.relative_to(mode_dir).as_posix()}/eda.json"

    with TemporaryDirectory(prefix=".phase1-", dir=mode_dir) as temp:
        stage = Path(temp) / "run"
        stage.mkdir()
        write_started = perf_counter()
        for name, partition in partitions.items():
            # Storage categories should only contain values observed in this partition.
            categories = partition.select_dtypes(include="category")
            stored = partition.assign(**{
                column: categories[column].cat.remove_unused_categories() for column in categories
            })
            table = pa.Table.from_pandas(stored, preserve_index=False)
            metadata = dict(table.schema.metadata or {})
            metadata.update({
                b"fraudlens.mode": manifest["mode"].encode(),
                b"fraudlens.partition": name.encode(),
                b"fraudlens.processed_rows": str(manifest["processed_rows"]).encode(),
                b"fraudlens.partition_rows": str(len(partition)).encode(),
            })
            pq.write_table(table.replace_schema_metadata(metadata), stage / f"{name}.parquet",
                           compression="snappy")

        _write_json(stage / "eda.json", eda)
        manifest["stage_seconds"]["write_parquet_and_eda"] = perf_counter() - write_started
        manifest["runtime_seconds"] = perf_counter() - started
        _write_json(stage / "manifest.json", manifest)
        stage.rename(run_dir)

        # Publish the manifest only after its complete, immutable run is available.
        latest = Path(temp) / "manifest.json"
        _write_json(latest, manifest)
        latest.replace(mode_dir / "manifest.json")
    return mode_dir / "manifest.json"


def run_pipeline(data_dir: Path, output_root: Path, mode: str, smoke_rows: int = SMOKE_ROWS, *,
                 seal_test: bool = False) -> Path:
    row_limits = {"quick": QUICK_ROWS, "full": None, "smoke": smoke_rows}
    limit = row_limits[mode]
    if limit is not None and limit < 1:
        raise ValueError("The requested row limit must be positive.")

    started = perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    resources = _resources(data_dir, output_root)
    print(f"Preflight resources: {resources}", flush=True)
    transactions, identities, sources = load_inputs(data_dir)
    inputs_loaded = perf_counter()

    selected = chronological_prefix(transactions, limit)
    del transactions
    merged = merge_identity(selected, identities)
    del selected, identities
    merged_at = perf_counter()

    processed = add_behavioral_features(merged)
    del merged
    partitions = chronological_split(processed)
    featurized = perf_counter()

    eda = {"mode": mode, "processed_rows": len(processed), **describe_data(processed, partitions, sources)}
    manifest = {
        "phase": 1,
        "schema_version": 1,
        "mode": mode,
        "seed": SEED,
        "requested_rows": limit,
        "processed_rows": len(processed),
        "source_rows": {name: info["rows"] for name, info in sources.items()},
        "resources_at_start": resources,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "pyarrow": pa.__version__},
        "partitions": eda["partitions"],
        "feature_schema": feature_schema(processed),
        "completed_stages": ["load_and_validate", "left_join", "historical_features",
                             "chronological_split", "eda", "data_artifacts"],
        "stage_seconds": {
            "load_and_validate": inputs_loaded - started,
            "chronological_selection_and_join": merged_at - inputs_loaded,
            "features_and_splits": featurized - merged_at,
            "eda": perf_counter() - featurized,
        },
    }
    output = _publish_artifacts(output_root / mode, partitions, eda, manifest, started)
    print(f"{mode}: {len(processed):,} transactions in {manifest['runtime_seconds']:.2f} seconds")
    for name, summary in eda["partitions"].items():
        fraud = "fraud labels sealed" if seal_test and name == "test" else f"{summary['fraud_rate']:.4%} fraud"
        print(f"{name}: {summary['rows']:,} rows, {fraud}, "
              f"seconds {summary['first_second']:,} to {summary['last_second']:,}")
    print(f"Manifest: {output.resolve()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full", "smoke"], default="quick")
    parser.add_argument("--smoke-rows", type=int, default=SMOKE_ROWS)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    try:
        run_pipeline(args.data_dir, args.output_dir, args.mode, args.smoke_rows)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Phase 1 error: {error}\n")


if __name__ == "__main__":
    main()
