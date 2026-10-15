from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid

from src.evaluation.benchmark_results import confidence_threshold_key


FORMAT_VERSION = 2
PUBLIC_METRIC_KEYS = (
    "match_rate",
    "miss_rate",
    "frames",
    "gt_boxes",
    "matched_boxes",
    "mean_iou",
    "median_iou",
    "mean_edge_error",
    "median_edge_error",
    "p90_edge_error",
    "mean_edge_error_px",
    "mean_center_error_px",
    "mean_center_error",
    "mean_size_error",
    "false_positives_per_frame",
    "inference_ms",
    "fps",
)


def _public_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in PUBLIC_METRIC_KEYS}


def build_evaluation_result(
    evaluation_id: str,
    dataset: dict[str, Any],
    models: dict[str, dict[str, str]],
    thresholds: list[float],
    match_iou_threshold: float,
    aggregate_metrics: dict[float, dict[str, dict[str, Any]]],
    per_sequence_metrics: dict[
        float,
        dict[str, dict[str, dict[str, Any]]],
    ],
    started_at: str,
    completed_at: str,
    evaluation_name: str | None = None,
    git_commit: str | None = None,
    split: dict[str, Any] | None = None,
    partition: str | None = None,
) -> dict[str, Any]:
    """Build the public result schema from already-computed benchmark metrics."""
    result_by_threshold = {
        confidence_threshold_key(threshold): {
            model_name: _public_metrics(metrics)
            for model_name, metrics in aggregate_metrics.get(threshold, {}).items()
        }
        for threshold in thresholds
    }
    sequence_by_threshold = {
        confidence_threshold_key(threshold): {
            model_name: {
                sequence_name: _public_metrics(metrics)
                for sequence_name, metrics in sequence_results.items()
            }
            for model_name, sequence_results in per_sequence_metrics.get(threshold, {}).items()
        }
        for threshold in thresholds
    }
    return {
        "format_version": FORMAT_VERSION,
        "evaluation_id": evaluation_id,
        "evaluation_name": evaluation_name or evaluation_id,
        "timestamp": completed_at,
        "git_commit": git_commit,
        "dataset": dataset,
        "split": split,
        "partition": partition,
        "models": models,
        "thresholds": thresholds,
        "match_iou_threshold": match_iou_threshold,
        "timing_semantics": {
            "fps_and_inference_ms": "quality-sweep inference timing at the detector's configured low inference floor; not operational-confidence deployment FPS",
            "operational_speed_tool": "src.evaluation.benchmark_detector_speed",
        },
        "results": result_by_threshold,
        "per_sequence": sequence_by_threshold,
        "started_at": started_at,
        "completed_at": completed_at,
    }


def save_evaluation_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(result, temporary_file, indent=2, allow_nan=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def default_evaluation_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"detector_{timestamp}_{uuid.uuid4().hex[:8]}"
