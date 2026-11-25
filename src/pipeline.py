"""Run all offline FraudLens phases serially, with a shared full-holdout release."""

import argparse
from contextlib import ExitStack, closing
import ctypes
from datetime import datetime, timezone
from hashlib import file_digest
from importlib.metadata import version
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
import torch

from . import phase1, phase3, phase5, phase6, train, train_mlp
from .data_loader import ID, TIME
from .training_data import TEST_SEALED, complete_run, read_phase1_manifest

FULL_ROWS = 590_540
PROJECT = Path(__file__).resolve().parents[1]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def _inventory(directory):
    return {path.relative_to(directory).as_posix(): _hash(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _peak_working_set():
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                        ("peak_working_set", ctypes.c_size_t), ("other", ctypes.c_size_t * 7)]

        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        memory_info.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        memory_info.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if memory_info(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.peak_working_set
    return None


def _separation(root):
    """Prove holdout isolation from development using only IDs and elapsed seconds."""
    def metadata(mode, partition):
        directory = root / mode
        manifest, _ = read_phase1_manifest(directory)
        return pq.read_table(directory / manifest["files"][partition], columns=[ID, TIME]).to_pandas()

    quick = [metadata("quick", name) for name in ("train", "validation", "test")]
    final = metadata("full", "test")
    latest_quick = max(int(frame[TIME].max()) for frame in quick)
    earliest_final = int(final[TIME].min())
    overlap = sum(int(np.isin(final[ID], frame[ID]).sum()) for frame in quick)
    if overlap or latest_quick >= earliest_final:
        raise ValueError("Full final test is not disjoint from and strictly later than quick development data.")
    return {"quick_rows": sum(len(frame) for frame in quick), "final_rows": len(final),
            "latest_quick_second": latest_quick, "earliest_final_second": earliest_final,
            "overlap_ids": overlap, "test_columns_read": [ID, TIME]}


def _record_protocol(data, root, mode_dir, threads):
    protocol = PROJECT / "docs" / "EXPERIMENT_PROTOCOL.md"
    pins = PROJECT / "requirements.txt"
    quick_files = _inventory(root / "quick")
    if "manifest.json" not in quick_files:
        raise FileNotFoundError("Generate the quick development artifacts in this artifact root before the full run.")
    names = [line.split("==", 1)[0] for line in pins.read_text().splitlines() if "==" in line]
    record = {"frozen_at_utc": _now(), "mode": "full", "seed": train.SEED, "threads": threads,
              "python": platform.python_version(), "versions": {name: version(name) for name in names},
              "resources": phase1._resources(data, root), "protocol_sha256": _hash(protocol),
              "requirements_sha256": _hash(pins),
              "source_sha256": {path.name: _hash(path) for path in sorted((PROJECT / "src").glob("*.py"))},
              "raw_sources": {path.name: {"bytes": path.stat().st_size, "sha256": _hash(path)}
                              for path in (data / "train_transaction.csv", data / "train_identity.csv")},
              "prior_quick_artifacts": quick_files}
    train._write_json(mode_dir / "protocol_freeze.json", record)
    print("Final protocol and raw input hashes recorded.", flush=True)
    return quick_files


def run_pipeline(data: Path, root: Path, mode: str, threads: int = train.THREADS) -> Path:
    started = perf_counter()
    data, root = data.resolve(), root.resolve()
    mode_dir = root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    release = mode_dir / "holdout_release.json"
    if mode == "full" and release.exists():
        raise FileExistsError(f"The final holdout was already released in {mode_dir}. Preserve that run before repeating it.")

    phase = "protocol freeze"
    clock = perf_counter()
    times = {}

    def step(name):
        nonlocal phase, clock
        times[phase] = perf_counter() - clock
        phase, clock = name, perf_counter()
        print(f"\n{mode} · {name}", flush=True)

    token = TEST_SEALED.set(mode == "full")
    try:
        if mode == "full":
            quick_files = _record_protocol(data, root, mode_dir, threads)
        step("Phase 1 data preparation")
        phase1.run_pipeline(data, root, mode, seal_test=mode == "full")
        if mode == "quick":
            for name, run in (("Phase 2 core models", train.run_training),
                              ("Phase 2B PyTorch MLP", train_mlp.run_training),
                              ("Phase 3 risk and SHAP", phase3.run_phase3),
                              ("Phase 5 DBSCAN and Apriori", phase5.run_phase5),
                              ("Phase 6 supplementary and NumPy", phase6.run_phase6)):
                step(name)
                run(root, mode, threads)
        else:
            step("Holdout chronology and ID audit")
            source, source_hash = read_phase1_manifest(mode_dir)
            if source["processed_rows"] != FULL_ROWS:
                raise ValueError(f"Expected {FULL_ROWS:,} real source transactions for the final full run.")
            separation = _separation(root)
            print(f"Holdout audit: {separation}", flush=True)
            with ExitStack() as stack:
                stack.enter_context(threadpool_limits(limits=threads))
                step("Phase 2 fit and validation")
                core = stack.enter_context(closing(train.prepare_training(root, mode, threads)))
                draft = next(core)
                if draft[1]["champion"] != "xgboost":
                    raise ValueError("Validation chose a different classical classifier from the designated XGBoost system.")
                step("Phase 2B fit and validation")
                mlp = stack.enter_context(closing(train_mlp.prepare_training(root, mode, threads, draft=draft)))
                mlp_stage = next(mlp)
                step("Phase 3 anomaly fit and validation risk ranking")
                risk = stack.enter_context(closing(phase3.prepare_phase3(root, mode, threads, draft=draft)))
                risk_stage = next(risk)
                step("Phase 5 DBSCAN and Apriori fit and validation")
                patterns = phase5.run_phase5(root, mode, threads)
                step("Phase 6 supplementary and NumPy fit and validation")
                lab = stack.enter_context(closing(phase6.prepare_phase6(root, mode, threads)))
                lab_stage = next(lab)
                step("Release frozen final holdout")
                train._write_json(release, {
                    "released_at_utc": _now(), "mode": mode, "phase1_manifest_sha256": source_hash,
                    "protocol_freeze_sha256": _hash(mode_dir / "protocol_freeze.json"), "separation": separation,
                    "frozen_artifacts": {"phase2": _inventory(draft[0]), "phase2b": _inventory(mlp_stage),
                                         "phase3": _inventory(risk_stage), "phase6": _inventory(lab_stage),
                                         "phase5_manifest_sha256": _hash(patterns)}})
                TEST_SEALED.set(False)
                for name, run in (("Phase 2 final test", core), ("Phase 2B final test", mlp),
                                  ("Phase 3 final test", risk), ("Phase 6 matched final test subset", lab)):
                    step(name)
                    complete_run(run)
            step("Verify preserved quick files")
            if quick_files != _inventory(root / "quick"):
                raise ValueError("The quick artifact inventory changed during the full run.")
        step("Write run summary")
        path = mode_dir / "pipeline_summary.json"
        train._write_json(path, {"mode": mode, "finished_at_utc": _now(), "runtime_seconds": perf_counter() - started,
                                 "stage_seconds": times, "peak_working_set_bytes": _peak_working_set(),
                                 "resources_at_end": phase1._resources(data, root), "threads": threads,
                                 "quick_files_preserved": len(quick_files) if mode == "full" else None})
        return path
    except Exception as error:
        failed = "after holdout release" if release.exists() else "before holdout release"
        raise RuntimeError(f"{mode} pipeline failed in {phase} ({failed}): {error}") from error
    finally:
        TEST_SEALED.reset(token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--threads", type=int, default=train.THREADS)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_interop_threads(1)
    result = run_pipeline(args.data_dir, args.artifact_root, args.mode, args.threads)
    print(f"Completed {args.mode} pipeline: {result}", flush=True)


if __name__ == "__main__":
    main()
