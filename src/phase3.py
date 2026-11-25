"""Run anomaly scoring, exact TreeSHAP, and fixed review-priority assessment."""

import argparse
from collections.abc import Generator
from contextlib import closing
from functools import cached_property
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from .behavior import (ANOMALY_FEATURES, FIT_ROW_CAP, MISSING_FLAGS, SEED, anomaly_percentiles,
                       anomaly_scores, fit_isolation_forest)
from .data_loader import ID, TARGET, TIME
from .explain import MARGIN_ATOL, METHOD, SCORE_ATOL, UNITS, FraudExplainer
from .risk_engine import (PRIORITY_CUTS, REVIEW_PERCENTAGES, WEIGHTS, assess_transaction, build_risk_table,
                          priority_distribution, validation_rankings)
from .train import THREADS, _write_json
from .training_data import PARTITIONS, complete_run, load_partition, read_phase1_manifest

GLOBAL_ROWS = 2_000
EVALUATION = ("validation", "test")


def _local_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise FileNotFoundError(f"Missing local artifact inside {root}: {relative}")
    return path


def _phase2_reference(mode_dir: Path, source: dict, source_hash: str,
                      draft: tuple[Path, dict] | None = None) -> dict:
    phase2 = mode_dir / "phase2"
    if draft is None:
        manifest_bytes = _local_path(phase2, "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        run_dir = phase2 / manifest["run_directory"]
        digest = sha256(manifest_bytes).hexdigest()
    else:
        run_dir, manifest = draft
        digest = None
    if (manifest.get("phase"), manifest.get("schema_version"), manifest.get("mode"),
            manifest.get("champion")) != (2, 1, mode_dir.name, "xgboost"):
        raise ValueError("Phase 3 requires the saved Phase 2 XGBoost champion for the same mode.")
    if (source_hash != manifest["phase1_manifest_sha256"] or source["partitions"] != manifest["partitions"]
            or manifest["phase1_partition_files"] != {name: source["files"][name] for name in PARTITIONS}):
        raise ValueError("Phase 1 and Phase 2 refer to different data snapshots.")
    for package in ("xgboost", "scikit-learn", "numpy", "pandas", "scipy", "joblib"):
        if version(package) != manifest["versions"][package]:
            raise ValueError(f"The saved Phase 2 artifacts require {package}=={manifest['versions'][package]}.")

    files = manifest["files"]
    selection = json.loads(_local_path(run_dir, files["selection"]).read_bytes())
    model_info = manifest["models"]["xgboost"]
    threshold = selection["thresholds"]["xgboost"]
    if (selection["champion"] != "xgboost" or model_info["feature_set"] != "behavioral"
            or threshold != model_info["threshold"] or not 0 <= threshold <= 1):
        raise ValueError("The Phase 2 champion or its frozen threshold has inconsistent metadata.")

    def fingerprint(relative):
        path = _local_path(mode_dir, (run_dir / relative).relative_to(mode_dir).as_posix())
        return {"path": path.relative_to(mode_dir.resolve()).as_posix(), "sha256": sha256(path.read_bytes()).hexdigest()}

    features = manifest["features"]["behavioral"]
    return {"model": fingerprint(files["classifiers"]["xgboost"]),
            "preprocessor": fingerprint(files["preprocessors"]["behavioral"]),
            "predictions": {name: (run_dir / files[f"{name}_predictions"]).relative_to(mode_dir).as_posix()
                            for name in EVALUATION},
            "threshold": threshold, "champion": "xgboost", "phase2_manifest_sha256": digest,
            "raw_features": features["raw_features"], "transformed_features": features["transformed_features"]}


def _load_explainer(mode_dir: Path, reference: dict, threads: int) -> FraudExplainer:
    def load(name):
        info = reference[name]
        path = _local_path(mode_dir, info["path"])
        if sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Saved classifier {name} does not match the Phase 3 reference.")
        return joblib.load(path)

    model, preprocessor = load("model"), load("preprocessor")
    if preprocessor.feature_names_in_.tolist() != reference["raw_features"]:
        raise ValueError("The classifier's raw feature order has changed.")
    return FraudExplainer(model, preprocessor, reference["transformed_features"], threads)


def _write_table(path: Path, frame: pd.DataFrame, mode: str, partition: str, processed_rows: int) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update({b"fraudlens.mode": mode.encode(), b"fraudlens.partition": partition.encode(),
                     b"fraudlens.processed_rows": str(processed_rows).encode(),
                     b"fraudlens.partition_rows": str(len(frame)).encode()})
    pq.write_table(table.replace_schema_metadata(metadata), path)


def _anomaly_table(frame: pd.DataFrame, scores: np.ndarray, reference: np.ndarray) -> pd.DataFrame:
    return frame[[ID]].assign(anomaly_score=scores, anomaly_percentile=anomaly_percentiles(reference, scores))


def _global_shap(stage: Path, explainer: FraudExplainer, frame: pd.DataFrame, risk: pd.DataFrame,
                 mode: str, processed_rows: int) -> dict:
    size = min(GLOBAL_ROWS, len(frame))
    positions = np.sort(np.random.default_rng(SEED).choice(len(frame), size=size, replace=False))
    scores = risk["classifier_score"].to_numpy()[positions]
    sample = frame.iloc[positions]
    batch = explainer.explain(sample, scores)
    values = pd.DataFrame(batch.values, columns=explainer.feature_names)
    values.insert(0, ID, sample[ID].to_numpy())
    values = values.assign(base_value=batch.base, raw_margin=batch.margins, classifier_score=batch.scores)
    _write_table(stage / "global_shap_values.parquet", values, mode, "validation_sample", processed_rows)
    importance = {"mode": mode, "processed_rows": processed_rows, "partition": "validation", "rows": size,
                  "seed": SEED, **explainer.global_importance(batch)}
    _write_json(stage / "global_shap_importance.json", importance)
    _write_json(stage / "shap_feature_mapping.json", {"mode": mode, "processed_rows": processed_rows,
                                                      "features": explainer.mapping})
    return {"global_rows": size, "sample_partition": "validation", "base_value": batch.base, "checks": batch.checks}


class Phase3Store:
    """Read assessment artifacts and lazily explain any requested evaluation row."""

    def __init__(self, artifact_root: Path, mode: str, threads: int = THREADS):
        self.mode_dir = artifact_root / mode
        phase = self.mode_dir / "phase3"
        if not (phase / "manifest.json").is_file():
            raise FileNotFoundError(f"Missing {phase / 'manifest.json'}. Run python -m src.phase3 --mode {mode} first.")
        self.manifest = json.loads((phase / "manifest.json").read_bytes())
        if (self.manifest.get("phase"), self.manifest.get("schema_version"), self.manifest.get("mode")) != (3, 1, mode):
            raise ValueError("Expected a Phase 3 manifest for the requested mode.")
        self.run_dir = phase / self.manifest["run_directory"]
        self.threads = threads

    @cached_property
    def explainer(self) -> FraudExplainer:
        return _load_explainer(self.mode_dir, self.manifest["classifier"], self.threads)

    def read_risk(self, partition: str, transaction_id: int | None = None) -> pd.DataFrame:
        if partition not in EVALUATION:
            raise ValueError("Use the validation or test assessment partition.")
        path = _local_path(self.run_dir, self.manifest["files"][f"{partition}_risk"])
        filters = None if transaction_id is None else [(ID, "=", transaction_id)]
        return pd.read_parquet(path, filters=filters)

    def assess(self, partition: str, transaction_id: int, include_shap: bool = True) -> dict:
        rows = self.read_risk(partition, transaction_id)
        if len(rows) != 1:
            raise ValueError(f"Expected one transaction {transaction_id} in {partition}; found {len(rows)}.")
        row = rows.iloc[0]
        explanation = None
        if include_shap:
            path = _local_path(self.mode_dir, self.manifest["phase1_partitions"][partition])
            raw = pd.read_parquet(path, columns=[ID, TIME, *self.manifest["classifier"]["raw_features"]],
                                  filters=[(ID, "=", transaction_id)])
            if len(raw) != 1 or raw[TIME].iloc[0] != row[TIME]:
                raise ValueError("The Phase 1 row no longer matches the selected assessment.")
            explanation = self.explainer.local(raw, row["classifier_score"])
        return {"mode": self.manifest["mode"], "partition": partition, **assess_transaction(row, explanation)}


def prepare_phase3(artifact_root: Path, mode: str, threads: int = THREADS, *,
                   draft: tuple[Path, dict] | None = None) -> Generator[Path, None, Path]:
    """Fit anomaly detection and evaluate validation ranking before pausing."""
    started = perf_counter()
    mode_dir = artifact_root / mode
    source, source_hash = read_phase1_manifest(mode_dir)
    classifier = _phase2_reference(mode_dir, source, source_hash, draft)
    row_count = source["processed_rows"]
    phase_dir = mode_dir / "phase3"
    runs = phase_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    run_dir = runs / uuid4().hex
    timings = {}

    with TemporaryDirectory(prefix=".phase3-", dir=phase_dir) as temp, threadpool_limits(limits=threads):
        stage = Path(temp) / "run"
        stage.mkdir()
        train = load_partition(mode_dir, source, "train")
        fit_started = perf_counter()
        prep, model, fit_rows = fit_isolation_forest(train, threads)
        timings["anomaly_fit"] = perf_counter() - fit_started
        score_started = perf_counter()
        train_scores = anomaly_scores(model, prep, train)
        reference = np.sort(train_scores)
        train_anomaly = _anomaly_table(train, train_scores, reference)
        timings["anomaly_train_reference"] = perf_counter() - score_started
        _write_table(stage / "train_anomaly.parquet", train_anomaly, mode, "train", row_count)
        np.save(stage / "anomaly_reference.npy", reference, allow_pickle=False)
        joblib.dump(prep, stage / "anomaly_preprocessor.joblib")
        joblib.dump(model, stage / "isolation_forest.joblib", compress=3)
        print(f"{mode}: Isolation Forest fit on {fit_rows:,} train rows in {timings['anomaly_fit']:.2f}s; "
              f"reference scored on {len(train):,} training rows.", flush=True)

        explainer = _load_explainer(mode_dir, classifier, threads)
        distributions = {}
        prior_ids = train[ID].copy()
        del train, train_anomaly, train_scores
        for name in EVALUATION:
            if name == "test":
                wait_started = perf_counter()
                yield stage
                started += perf_counter() - wait_started
                if draft is not None:
                    final_classifier = _phase2_reference(mode_dir, source, source_hash)
                    if (any(final_classifier[key]["sha256"] != classifier[key]["sha256"]
                            for key in ("model", "preprocessor"))
                            or final_classifier["threshold"] != classifier["threshold"]):
                        raise ValueError("The finalized classifier differs from the validation classifier.")
                    classifier = final_classifier
            frame = load_partition(mode_dir, source, name)
            if frame[ID].isin(prior_ids).any():
                raise ValueError(f"Duplicate transaction IDs found across partitions before {name} scoring.")
            prior_ids = pd.concat([prior_ids, frame[ID]], ignore_index=True)
            saved_pred = pd.read_parquet(_local_path(mode_dir, classifier["predictions"][name]),
                                         columns=[ID, TIME, TARGET, "xgboost_score", "xgboost_decision"])
            saved_pred = saved_pred.rename(columns={"xgboost_score": "classifier_score", "xgboost_decision": "classifier_decision"})

            score_started = perf_counter()
            raw_scores = anomaly_scores(model, prep, frame)
            anomaly = _anomaly_table(frame, raw_scores, reference)
            timings[f"anomaly_{name}"] = perf_counter() - score_started
            risk = build_risk_table(frame, saved_pred, anomaly, classifier["threshold"])
            distributions[name] = priority_distribution(risk)
            _write_table(stage / f"{name}_anomaly.parquet", anomaly, mode, name, row_count)
            _write_table(stage / f"{name}_risk.parquet", risk, mode, name, row_count)

            explanation_started = perf_counter()
            if name == "validation":
                rankings = validation_rankings(risk)
                _write_json(stage / "validation_rankings.json", {"mode": mode, "processed_rows": row_count,
                            "partition": name, "rows": len(frame), "rankings": rankings,
                            "tie_rule": "Descending score, then ascending TransactionID; top counts round up."})
                explanation_started = perf_counter()
                shap_summary = _global_shap(stage, explainer, frame, risk, mode, row_count)
                timings["global_shap_and_save"] = perf_counter() - explanation_started
            else:
                _write_json(stage / "test_rankings.json", {"mode": mode, "processed_rows": row_count,
                            "partition": name, "rows": len(frame), "rankings": validation_rankings(risk),
                            "tie_rule": "Descending score, then ascending TransactionID; top counts round up."})
                # Select a representative trace by history availability and chronological order only.
                available = risk["prior_mean_amount"].notna() & risk["is_new_product"].notna()
                index = int(np.flatnonzero(available.to_numpy())[0]) if available.any() else 0
                row = risk.iloc[index]
                local = explainer.local(frame.iloc[[index]], row["classifier_score"])
                example = {"mode": mode, "partition": name, "processed_rows": row_count,
                           "selection_rule": "First chronological test row with a prior mean and known product novelty, else first row.",
                           "classifier_threshold": classifier["threshold"], **assess_transaction(row, local)}
                _write_json(stage / "example_assessment.json", example)
                timings["single_shap_and_save"] = perf_counter() - explanation_started
            del frame, saved_pred, anomaly, risk

        _write_json(stage / "priority_distributions.json", {"mode": mode, "processed_rows": row_count,
                                                           "partitions": distributions})
        manifest = {
            "phase": 3, "schema_version": 1, "mode": mode, "processed_rows": row_count, "seed": SEED, "threads": threads,
            "partitions": source["partitions"], "phase1_manifest_sha256": source_hash,
            "phase1_partitions": {name: source["files"][name] for name in PARTITIONS}, "classifier": classifier,
            "anomaly": {"numeric_features": ANOMALY_FEATURES, "missing_flags_for": MISSING_FLAGS,
                        "transformed_features": prep.get_feature_names_out(), "imputation": "training medians; zero if entirely missing",
                        "fit_rows": fit_rows, "fit_cap": FIT_ROW_CAP, "reference_rows": len(reference),
                        "reference_partition": "train", "settings": model.get_params(),
                        "score": "negative sklearn IsolationForest.score_samples; higher is more unusual",
                        "percentile": "(# strictly below + 0.5 * # equal) / number of training reference scores"},
            "shap": {"method": METHOD, "units": UNITS, "margin_atol": MARGIN_ATOL, "score_atol": SCORE_ATOL,
                     "example_checks": local["checks"], **shap_summary},
            "risk": {"classifier_weight": WEIGHTS[0], "anomaly_weight": WEIGHTS[1], "priority_cuts": PRIORITY_CUTS,
                     "review_percentages": REVIEW_PERCENTAGES},
            "test_scope": "full chronological holdout" if mode == "full" else "development outputs",
            "versions": {"python": platform.python_version(), **{name: version(name) for name in
                         ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "xgboost", "shap", "numba",
                          "llvmlite", "joblib", "threadpoolctl")}},
            "timings_seconds": timings, "runtime_seconds": perf_counter() - started,
            "run_directory": run_dir.relative_to(phase_dir).as_posix(),
            "files": {"anomaly_model": "isolation_forest.joblib", "anomaly_preprocessor": "anomaly_preprocessor.joblib",
                      "anomaly_reference": "anomaly_reference.npy", "train_anomaly": "train_anomaly.parquet",
                      "rankings": "validation_rankings.json", "test_rankings": "test_rankings.json", "priorities": "priority_distributions.json",
                      "global_shap_importance": "global_shap_importance.json", "global_shap_values": "global_shap_values.parquet",
                      "shap_mapping": "shap_feature_mapping.json", "example": "example_assessment.json",
                      **{f"{name}_{kind}": f"{name}_{kind}.parquet" for name in EVALUATION for kind in ("anomaly", "risk")}},
        }
        _write_json(stage / "manifest.json", manifest)
        stage.rename(run_dir)
        latest = Path(temp) / "manifest.json"
        _write_json(latest, manifest)
        latest.replace(phase_dir / "manifest.json")

    print(f"Phase 3 {mode}: {manifest['runtime_seconds']:.2f}s. Validation AP "
          f"classifier {rankings['classifier']['average_precision']:.6f}, "
          f"anomaly {rankings['anomaly']['average_precision']:.6f}, "
          f"composite {rankings['composite']['average_precision']:.6f}.", flush=True)
    return phase_dir / "manifest.json"


def run_phase3(artifact_root: Path, mode: str, threads: int = THREADS) -> Path:
    with closing(prepare_phase3(artifact_root, mode, threads)) as run:
        next(run)
        return complete_run(run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full", "smoke"), default="quick")
    parser.add_argument("--threads", type=int, choices=range(1, 9), metavar="1-8", default=THREADS)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    try:
        manifest = run_phase3(args.artifact_root, args.mode, args.threads)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Phase 3 error: {error}\n")
    print(f"Manifest: {manifest.resolve()}")


if __name__ == "__main__":
    main()
