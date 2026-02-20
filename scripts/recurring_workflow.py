#!/usr/bin/env python3
"""Recurring model improvement workflow.

Workflow steps:
1) Run evaluation and collect misclassified clips.
2) Tag failures by metadata dimensions.
3) Merge curated examples into dataset metadata.
4) Retrain and compare on fixed benchmark.
5) Promote only if macro-F1 and worst-language recall both improve.

This script writes all outputs into versioned experiment folders:
`experiments/YYYY-MM-DD-run-name/`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import librosa
import numpy as np
import tensorflow as tf


@dataclass
class ClipRecord:
    path: str
    label: str
    language: str
    generator: str
    speaker: str
    artifact_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recurring model improvement workflow")
    parser.add_argument("--run-name", required=True, help="Human-friendly run name")
    parser.add_argument("--current-model", required=True, help="Model used for global/folder evaluation")
    parser.add_argument("--previous-model", required=True, help="Previous production model for benchmark comparison")
    parser.add_argument("--evaluation-manifest", required=True, help="CSV manifest for global/folder evaluation")
    parser.add_argument("--benchmark-manifest", required=True, help="CSV manifest for fixed benchmark")
    parser.add_argument("--dataset-manifest", required=True, help="CSV manifest for current training dataset")
    parser.add_argument("--curated-manifest", required=True, help="CSV manifest with curated examples to merge")
    parser.add_argument(
        "--retrain-command",
        default="",
        help=(
            "Optional shell command for retraining. Supports placeholders: "
            "{dataset_manifest}, {base_model}, {output_model}."
        ),
    )
    parser.add_argument("--output-root", default="experiments", help="Root folder for experiment logs")
    parser.add_argument(
        "--promote-model-path",
        default="model/model-promoted.h5",
        help="Where to copy promoted model when gate passes",
    )
    parser.add_argument("--sample-rate", type=int, default=None, help="Optional librosa sample rate")
    parser.add_argument("--n-mels", type=int, default=91, help="Mel bands for feature extraction")
    parser.add_argument("--max-time-steps", type=int, default=150, help="Spectrogram time axis length")
    return parser.parse_args()


def load_manifest(path: Path) -> List[ClipRecord]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        required = {"path", "label", "language", "generator", "speaker", "artifact_type"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest {path} missing required columns: {sorted(missing)}")

        rows: List[ClipRecord] = []
        for row in reader:
            rows.append(
                ClipRecord(
                    path=row["path"].strip(),
                    label=row["label"].strip().lower(),
                    language=row["language"].strip(),
                    generator=row["generator"].strip(),
                    speaker=row["speaker"].strip(),
                    artifact_type=row["artifact_type"].strip(),
                )
            )
        return rows


def preprocess_audio(file_path: str, n_mels: int, max_time_steps: int, sample_rate: int | None) -> np.ndarray:
    audio, _ = librosa.load(file_path, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=audio, n_mels=n_mels)
    mel = librosa.power_to_db(mel, ref=np.max)
    if mel.shape[1] < max_time_steps:
        mel = np.pad(mel, ((0, 0), (0, max_time_steps - mel.shape[1])), mode="constant")
    else:
        mel = mel[:, :max_time_steps]
    return mel


def predict_label(model: tf.keras.Model, mel: np.ndarray) -> Tuple[str, float, float]:
    batch = mel[np.newaxis, ...]
    pred = model.predict(batch, verbose=0)[0]
    ai_prob = float(pred[0])
    human_prob = float(pred[1])
    label = "ai" if ai_prob >= human_prob else "human"
    return label, ai_prob, human_prob


def evaluate_model(
    model: tf.keras.Model,
    records: Sequence[ClipRecord],
    n_mels: int,
    max_time_steps: int,
    sample_rate: int | None,
) -> Tuple[List[dict], List[dict]]:
    evaluated: List[dict] = []
    failures: List[dict] = []

    for rec in records:
        mel = preprocess_audio(rec.path, n_mels=n_mels, max_time_steps=max_time_steps, sample_rate=sample_rate)
        pred_label, ai_prob, human_prob = predict_label(model, mel)
        row = {
            "path": rec.path,
            "true_label": rec.label,
            "predicted_label": pred_label,
            "ai_probability": ai_prob,
            "human_probability": human_prob,
            "language": rec.language,
            "generator": rec.generator,
            "speaker": rec.speaker,
            "artifact_type": rec.artifact_type,
            "correct": pred_label == rec.label,
        }
        evaluated.append(row)
        if not row["correct"]:
            failures.append(row)

    return evaluated, failures


def group_failures(failures: Iterable[dict]) -> Dict[str, Dict[str, int]]:
    buckets = {
        "language": Counter(),
        "generator": Counter(),
        "speaker": Counter(),
        "artifact_type": Counter(),
    }
    for failure in failures:
        buckets["language"][failure["language"]] += 1
        buckets["generator"][failure["generator"]] += 1
        buckets["speaker"][failure["speaker"]] += 1
        buckets["artifact_type"][failure["artifact_type"]] += 1
    return {key: dict(counter) for key, counter in buckets.items()}


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def merge_curated_examples(dataset_rows: Sequence[ClipRecord], curated_rows: Sequence[ClipRecord]) -> List[ClipRecord]:
    existing = {(r.path, r.label) for r in dataset_rows}
    merged = list(dataset_rows)
    for row in curated_rows:
        key = (row.path, row.label)
        if key not in existing:
            merged.append(row)
            existing.add(key)
    return merged


def f1_for_positive_class(true_labels: Sequence[str], pred_labels: Sequence[str], positive: str) -> float:
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == positive and p != positive)
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1(true_labels: Sequence[str], pred_labels: Sequence[str]) -> float:
    labels = sorted(set(true_labels))
    if not labels:
        return 0.0
    return float(sum(f1_for_positive_class(true_labels, pred_labels, lab) for lab in labels) / len(labels))


def language_recalls(rows: Sequence[dict]) -> Dict[str, float]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["language"]].append(row)

    recalls: Dict[str, float] = {}
    for lang, lang_rows in grouped.items():
        total = len(lang_rows)
        correct = sum(1 for r in lang_rows if r["correct"])
        recalls[lang] = correct / total if total else 0.0
    return recalls


def benchmark_metrics(eval_rows: Sequence[dict]) -> dict:
    true_labels = [r["true_label"] for r in eval_rows]
    pred_labels = [r["predicted_label"] for r in eval_rows]
    macro = macro_f1(true_labels, pred_labels)
    recalls = language_recalls(eval_rows)
    worst_lang = min(recalls.values()) if recalls else 0.0
    return {
        "macro_f1": macro,
        "language_recall": recalls,
        "worst_language_recall": worst_lang,
        "size": len(eval_rows),
    }


def run_retraining(command_template: str, dataset_manifest: Path, base_model: Path, output_model: Path) -> dict:
    if not command_template:
        shutil.copy2(base_model, output_model)
        return {"executed": False, "command": "", "return_code": 0, "message": "No retrain command; copied base model."}

    command = command_template.format(
        dataset_manifest=str(dataset_manifest),
        base_model=str(base_model),
        output_model=str(output_model),
    )
    completed = subprocess.run(command, shell=True, text=True, capture_output=True)
    return {
        "executed": True,
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> None:
    args = parse_args()

    timestamp = dt.datetime.now().strftime("%Y-%m-%d")
    run_dir = Path(args.output_root) / f"{timestamp}-{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    evaluation_manifest = load_manifest(Path(args.evaluation_manifest))
    benchmark_manifest = load_manifest(Path(args.benchmark_manifest))
    dataset_manifest = load_manifest(Path(args.dataset_manifest))
    curated_manifest = load_manifest(Path(args.curated_manifest))

    current_model = tf.keras.models.load_model(args.current_model)
    previous_model = tf.keras.models.load_model(args.previous_model)

    global_eval_rows, failures = evaluate_model(
        current_model,
        evaluation_manifest,
        n_mels=args.n_mels,
        max_time_steps=args.max_time_steps,
        sample_rate=args.sample_rate,
    )

    write_csv(
        run_dir / "global_evaluation.csv",
        global_eval_rows,
        [
            "path",
            "true_label",
            "predicted_label",
            "ai_probability",
            "human_probability",
            "language",
            "generator",
            "speaker",
            "artifact_type",
            "correct",
        ],
    )
    write_csv(
        run_dir / "misclassified_clips.csv",
        failures,
        [
            "path",
            "true_label",
            "predicted_label",
            "ai_probability",
            "human_probability",
            "language",
            "generator",
            "speaker",
            "artifact_type",
            "correct",
        ],
    )

    failure_summary = group_failures(failures)
    with (run_dir / "failure_tags.json").open("w", encoding="utf-8") as fp:
        json.dump(failure_summary, fp, indent=2)

    merged_dataset = merge_curated_examples(dataset_manifest, curated_manifest)
    merged_manifest_path = run_dir / "dataset_with_curated.csv"
    write_csv(
        merged_manifest_path,
        [r.__dict__ for r in merged_dataset],
        ["path", "label", "language", "generator", "speaker", "artifact_type"],
    )

    candidate_model_path = run_dir / "candidate_model.h5"
    retrain_info = run_retraining(
        args.retrain_command,
        dataset_manifest=merged_manifest_path,
        base_model=Path(args.current_model),
        output_model=candidate_model_path,
    )
    if retrain_info.get("return_code", 1) != 0:
        raise RuntimeError(f"Retraining failed: {retrain_info}")

    candidate_model = tf.keras.models.load_model(candidate_model_path)

    prev_rows, _ = evaluate_model(
        previous_model,
        benchmark_manifest,
        n_mels=args.n_mels,
        max_time_steps=args.max_time_steps,
        sample_rate=args.sample_rate,
    )
    cand_rows, _ = evaluate_model(
        candidate_model,
        benchmark_manifest,
        n_mels=args.n_mels,
        max_time_steps=args.max_time_steps,
        sample_rate=args.sample_rate,
    )

    previous_metrics = benchmark_metrics(prev_rows)
    candidate_metrics = benchmark_metrics(cand_rows)

    promote = (
        candidate_metrics["macro_f1"] > previous_metrics["macro_f1"]
        and candidate_metrics["worst_language_recall"] > previous_metrics["worst_language_recall"]
    )

    promoted_path = None
    if promote:
        promoted_path = Path(args.promote_model_path)
        promoted_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_model_path, promoted_path)

    metrics_payload = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "global_evaluation_size": len(global_eval_rows),
        "misclassified_count": len(failures),
        "failure_tags": failure_summary,
        "curated_examples_added": len(merged_dataset) - len(dataset_manifest),
        "retraining": retrain_info,
        "benchmark": {
            "previous": previous_metrics,
            "candidate": candidate_metrics,
            "delta": {
                "macro_f1": candidate_metrics["macro_f1"] - previous_metrics["macro_f1"],
                "worst_language_recall": candidate_metrics["worst_language_recall"]
                - previous_metrics["worst_language_recall"],
            },
        },
        "promotion": {
            "promoted": promote,
            "promoted_model_path": str(promoted_path) if promoted_path else None,
            "gate": "macro_f1 and worst_language_recall must both improve",
        },
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)

    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
