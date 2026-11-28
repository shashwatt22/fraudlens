"""Fit only the V3 XGBoost treatment against the saved V2 development control."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from src.v2.challengers import BOOSTING
from src.v2.data import sha256, write_json
from src.v2.experiment import Trial, _peak_memory, _train_rich, review_budgets
from src.v2.final import _score
from .behavior import FEATURES, INPUTS, PREFIX, anchor_keys, behavioral_block
from .diagnostics import XGB, annotated_metrics, development_inputs, seen_groups


THREADS = 6
THRESHOLD = .5
ROUNDS = 320


class TreatmentMatrix:
    """Read the original matrix and the 13 new columns together one batch at a time."""

    def __init__(self, base, addition):
        if len(base) != len(addition):
            raise ValueError("Treatment rows must match the control matrix.")
        self.base = base
        self.addition = addition
        self.full_columns = list(range(base.shape[1] + addition.shape[1]))

    def __getitem__(self, index):
        rows, columns = index
        if isinstance(columns, int):
            # MatrixView.fit calls this only for the original categorical columns.
            return self.base[rows, columns]
        if columns != self.full_columns:
            raise ValueError("The controlled comparison requires every feature in the same order.")
        return np.concatenate((self.base[rows], self.addition[rows]), axis=1)


def check_control(root, matrices, times, labels, fold, index):
    directory = root / "challengers" / XGB
    record = json.loads((directory / f"fold{index}.json").read_bytes())
    candidate = BOOSTING[XGB]
    start, stop = fold.train_end, fold.validation_end
    if (record["candidate"] != asdict(candidate) or record["threads"] != THREADS
            or record["feature_sha256"] != sha256(root / "advanced.json") or record["fold"] != index
            or record["train_end"] != start or record["fit_rows"] != start
            or record["validation_end"] != stop or record["threshold"] != THRESHOLD
            or candidate.options["n_estimators"] != ROUNDS):
        raise ValueError(f"Saved control configuration does not match the experiment in fold {index}.")
    expected = np.load(directory / f"scores{index}.npy", allow_pickle=False)
    actual = _score(XGB, directory / f"model{index}.joblib", matrices, start, stop, start, times, THREADS)
    if (expected.shape != (stop - start,) or not np.isfinite(expected).all()
            or not np.array_equal(actual, expected)):
        raise ValueError(f"Could not reproduce the saved control scores exactly in fold {index}.")
    np.testing.assert_allclose(annotated_metrics(labels[start:stop], expected, THRESHOLD)["average_precision"],
                               record["metrics"]["average_precision"], atol=1e-12)
    return expected, record


def feature_manifest(root, out, matrices, times, stop):
    _, advanced, _, source = matrices
    saved = out / "feature_block.json"
    target = out / "behavior_features.npy"
    signatures = {name: sha256(root / name) for name in ("features.json", "advanced.json", "development_metadata.parquet")}
    signatures["behavior.py"] = sha256(Path(__file__).with_name("behavior.py"))
    expected = {"source_sha256": signatures, "features": list(FEATURES), "rows": stop}
    if saved.exists():
        record = json.loads(saved.read_bytes())
        if any(record[k] != value for k, value in expected.items()) or sha256(target) != record["cache_sha256"]:
            raise ValueError("Behavior cache or its inputs changed. Use a fresh output directory.")
        return record

    start = perf_counter()
    names = advanced["features"]
    if set(FEATURES) & set(names) or len(set(FEATURES)) != len(FEATURES):
        raise ValueError("The treatment would duplicate an existing feature.")
    base = source[:stop]
    keys = anchor_keys(base, names)
    measures = base[:, [names.index(name) for name in INPUTS]]
    extra = behavioral_block(times[:stop], keys, measures)
    temporary = target.with_name("behavior_features.tmp.npy")
    np.save(temporary, extra, allow_pickle=False)
    temporary.replace(target)
    record = {**expected, "cache_sha256": sha256(target), "seconds": perf_counter() - start,
              "process_peak_working_set_bytes": _peak_memory()}
    write_json(saved, record)
    return record


def metrics_by_group(labels, control, treatment, groups):
    return {name: {"control": annotated_metrics(labels[mask], control[mask], THRESHOLD),
                   "treatment": annotated_metrics(labels[mask], treatment[mask], THRESHOLD)}
            for name, mask in groups.items()}


def train_fold(root, out, matrices, times, labels, fold, index, feature):
    _, advanced, _, source = matrices
    start, stop = fold.train_end, fold.validation_end
    directory = out / f"fold{index}"
    directory.mkdir(exist_ok=True)
    scores_path = directory / "treatment_scores.npy"
    checkpoint = directory / "treatment.ubj"
    manifest = directory / "result.json"
    provenance = {"fold": index, "train_end": start, "validation_end": stop, "threads": THREADS,
                  "feature_manifest_sha256": sha256(out / "feature_block.json"),
                  "training_code_sha256": sha256(Path(__file__)),
                  "v2_training_code_sha256": sha256(Path(__file__).resolve().parents[1] / "v2" / "experiment.py"),
                  "control_scores_sha256": sha256(root / "challengers" / XGB / f"scores{index}.npy"),
                  "candidate": asdict(BOOSTING[XGB])}
    if manifest.exists():
        previous = json.loads(manifest.read_bytes())
        if (any(previous[k] != v for k, v in provenance.items())
                or previous["treatment_scores_sha256"] != sha256(scores_path)
                or previous["treatment_model_sha256"] != sha256(checkpoint)):
            raise ValueError(f"Fold {index} checkpoint differs from the recorded run.")
        print(f"Fold {index} already verified; reused its checkpoint.", flush=True)
        return previous

    wall = perf_counter()
    extra = np.load(out / "behavior_features.npy", mmap_mode="r", allow_pickle=False)
    if extra.dtype != np.float32 or extra.shape != (feature["rows"], len(FEATURES)):
        raise ValueError("Unexpected feature block shape or dtype.")
    cache, _, _, _ = matrices
    names = [*advanced["features"], *FEATURES]
    candidate = BOOSTING[XGB]
    trial = Trial("v3_behavior_7d", "advanced", candidate.options, candidate.weighted)
    combined = TreatmentMatrix(source[:len(extra)], extra)
    settings = {**cache, "features": names, "groups": {"advanced": names}}
    with threadpool_limits(limits=THREADS):
        control, control_record = check_control(root, matrices, times, labels, fold, index)
        scores, fit_seconds, score_seconds, model = _train_rich(settings, combined, trial, start, stop, labels, THREADS)
        peak = _peak_memory()
        if model.num_boosted_rounds() != ROUNDS or not np.isfinite(scores).all():
            raise ValueError("Unexpected treatment boosting rounds or scores.")
        # Save model and scores atomically before the aggregate calculations.
        temporary_model = directory / "treatment.tmp.ubj"
        model.save_model(temporary_model)
        temporary_model.replace(checkpoint)
        temporary_scores = directory / "treatment_scores.tmp.npy"
        np.save(temporary_scores, scores, allow_pickle=False)
        temporary_scores.replace(scores_path)
        split_counts = model.get_score(importance_type="weight")
        gains = model.get_score(importance_type="total_gain")
        importance = {kind: {name: float(values.get(name, 0.)) for name in FEATURES}
                      for kind, values in (("weight", split_counts), ("total_gain", gains))}
        total = sum(gains.values())
        importance["share_of_total_gain"] = sum(importance["total_gain"].values()) / total
        del model

    validation = labels[start:stop]
    keys = anchor_keys(source[:stop], advanced["features"])
    prior_count = source[start:stop, advanced["features"].index(f"{PREFIX}_prior_count")]
    groups = {**seen_groups(keys, np.arange(start), start, stop), "prior_history": prior_count > 0,
              "complete_key_without_history": prior_count == 0}
    report = {**provenance, "threshold": THRESHOLD, "rounds": ROUNDS, "fit_rows": start,
              "fit_seconds": fit_seconds, "inference_seconds": score_seconds, "wall_seconds": perf_counter() - wall,
              "treatment_fit_and_score_peak_working_set_bytes": peak, "worker_peak_working_set_bytes": _peak_memory(),
              "treatment_scores_sha256": sha256(scores_path), "treatment_model_sha256": sha256(checkpoint),
              "control_archived_fit_seconds": control_record["fit_seconds"], "control_verified_max_score_difference": 0.,
              "control": annotated_metrics(validation, control, THRESHOLD),
              "treatment": annotated_metrics(validation, scores, THRESHOLD),
              "review_budgets": {"control": review_budgets(validation, control), "treatment": review_budgets(validation, scores)},
              "groups": metrics_by_group(validation, control, scores, groups), "importance": importance}
    write_json(manifest, report)
    print(f"Fold {index}: AP {report['control']['average_precision']:.5f} → {report['treatment']['average_precision']:.5f}; "
          f"treatment fit {fit_seconds:.1f}s; checkpoint saved.", flush=True)
    return report


def load_inputs(development, output):
    development, output = development.resolve(), output.resolve()
    if output.is_relative_to(development) or "retrospective" in {part.lower() for part in development.parts}:
        raise ValueError("Keep outputs separate and use only the V2 development root.")
    matrices, times, labels, folds = development_inputs(development)
    output.mkdir(parents=True, exist_ok=True)
    return development, output, matrices, times, labels, folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), help="Omit to build only the read-only feature cache.")
    args = parser.parse_args()
    root, out, matrices, times, labels, folds = load_inputs(args.development, args.output)
    feature = feature_manifest(root, out, matrices, times, folds[-1].validation_end)
    if args.fold is not None:
        train_fold(root, out, matrices, times, labels, folds[args.fold - 1], args.fold, feature)
    else:
        print(f"Behavioral cache ready: {feature['rows']:,} rows, {len(FEATURES)} additional features.", flush=True)


if __name__ == "__main__":
    main()
