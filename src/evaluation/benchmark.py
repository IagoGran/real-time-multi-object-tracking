from dataclasses import dataclass
import logging
import statistics
import time
from pathlib import Path
from typing import Callable, Any

import cv2
import numpy as np
import torch

from src.detection.detector import Detector
from src.evaluation.detection_metrics import (
    Box,
    evaluate_bounding_box,
)
from src.evaluation.matching import match_boxes
from src.evaluation.mot17_dataset import Mot17Frame, Mot17Sequence
from src.evaluation.benchmark_results import (
    combination_completed,
    save_checkpoint,
    utc_timestamp,
)


LOGGER = logging.getLogger(__name__)
OBSERVATION_KEYS = (
    "iou",
    "edge_error_px",
    "edge_error",
    "center_error_px",
    "center_error",
    "size_error",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    signature: str
    factory: Callable[[], Detector]


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _frame_count(sequence: Mot17Sequence) -> int:
    return len(sequence.frames)


def _new_accumulator() -> dict[str, Any]:
    return {
        "frames": 0,
        "gt_boxes": 0,
        "matched_boxes": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "inference_seconds_total": 0.0,
        "observations": {key: [] for key in OBSERVATION_KEYS},
    }


def _add_metrics(
    accumulator: dict[str, Any],
    matching_result,
) -> None:
    for match in matching_result.matches:
        metrics = evaluate_bounding_box(
            match.predicted_box,
            match.ground_truth_box,
        )
        accumulator["matched_boxes"] += 1
        for key in OBSERVATION_KEYS:
            accumulator["observations"][key].append(
                getattr(metrics, key)
            )


def _summarize(accumulator: dict[str, Any]) -> dict[str, Any]:
    frames = accumulator["frames"]
    gt_boxes = accumulator["gt_boxes"]
    inference_seconds = accumulator["inference_seconds_total"]
    observations = accumulator["observations"]

    def mean(key: str) -> float:
        values = observations[key]
        return statistics.fmean(values) if values else 0.0

    def median(key: str) -> float:
        values = observations[key]
        return statistics.median(values) if values else 0.0

    edge_errors = observations["edge_error"]
    return {
        **accumulator,
        "match_rate": (
            accumulator["matched_boxes"] / gt_boxes if gt_boxes else 0.0
        ),
        "miss_rate": (
            accumulator["false_negatives"] / gt_boxes if gt_boxes else 0.0
        ),
        "false_positives_per_frame": (
            accumulator["false_positives"] / frames if frames else 0.0
        ),
        "mean_iou": mean("iou"),
        "median_iou": median("iou"),
        "mean_edge_error_px": mean("edge_error_px"),
        "mean_edge_error": mean("edge_error"),
        "median_edge_error": median("edge_error"),
        "p90_edge_error": (
            float(np.percentile(edge_errors, 90)) if edge_errors else 0.0
        ),
        "mean_center_error_px": mean("center_error_px"),
        "mean_center_error": mean("center_error"),
        "mean_size_error": mean("size_error"),
        "inference_ms": (
            inference_seconds / frames * 1000 if frames else 0.0
        ),
        "fps": frames / inference_seconds if inference_seconds else 0.0,
    }


def _log_result(label: str, metrics: dict[str, Any]) -> None:
    LOGGER.info(
        "%s | Match %.2f%% | IoU mean/median %.3f/%.3f | "
        "edge norm mean/median/P90 %.2f%%/%.2f%%/%.2f%% | "
        "edge px %.2f | center px/norm %.2f/%.2f%% | "
        "size %.2f%% | FP/frame %.2f | inference %.1f ms | %.2f FPS",
        label,
        metrics["match_rate"] * 100,
        metrics["mean_iou"],
        metrics["median_iou"],
        metrics["mean_edge_error"] * 100,
        metrics["median_edge_error"] * 100,
        metrics["p90_edge_error"] * 100,
        metrics["mean_edge_error_px"],
        metrics["mean_center_error_px"],
        metrics["mean_center_error"] * 100,
        metrics["mean_size_error"] * 100,
        metrics["false_positives_per_frame"],
        metrics["inference_ms"],
        metrics["fps"],
    )


def aggregate_sequence_results(
    sequence_results: list[dict[str, Any]],
) -> dict[str, Any]:
    accumulator = _new_accumulator()
    for result in sequence_results:
        accumulator["frames"] += result["frames"]
        accumulator["gt_boxes"] += result["gt_boxes"]
        accumulator["matched_boxes"] += result["matched_boxes"]
        accumulator["false_positives"] += result["false_positives"]
        accumulator["false_negatives"] += result["false_negatives"]
        accumulator["inference_seconds_total"] += result[
            "inference_seconds_total"
        ]
        for key in OBSERVATION_KEYS:
            accumulator["observations"][key].extend(
                result["observations"][key]
            )
    return _summarize(accumulator)


def _prediction_boxes(detections) -> list[Box]:
    return [
        (item.x1, item.y1, item.x2, item.y2)
        for item in detections
    ]


def _record_frame(
    detector: Detector,
    frame: np.ndarray,
    ground_truth_boxes: list[Box],
    confidence_threshold: float,
    match_iou_threshold: float,
    accumulator: dict[str, Any],
) -> None:
    _synchronize_cuda()
    inference_start = time.perf_counter()
    detections = detector.detect(
        frame,
        confidence_threshold=confidence_threshold,
    )
    _synchronize_cuda()
    accumulator["inference_seconds_total"] += (
        time.perf_counter() - inference_start
    )

    matching_result = match_boxes(
        _prediction_boxes(detections),
        ground_truth_boxes,
        iou_threshold=match_iou_threshold,
    )
    accumulator["gt_boxes"] += len(ground_truth_boxes)
    accumulator["false_positives"] += len(
        matching_result.unmatched_predictions
    )
    accumulator["false_negatives"] += len(
        matching_result.unmatched_ground_truth
    )
    _add_metrics(accumulator, matching_result)


def _read_frame(frame_ref: Mot17Frame) -> np.ndarray:
    frame = cv2.imread(str(frame_ref.path))
    if frame is None:
        raise RuntimeError(f"Failed to read frame: {frame_ref.path}")
    return frame


def format_duration(seconds: float) -> str:
    if seconds < 10:
        return f"{max(0.0, seconds):.1f}s"
    total_seconds = max(0, int(round(seconds)))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {remaining_seconds:02d}s"
    if minutes:
        return f"{minutes}m {remaining_seconds:02d}s"
    return f"{remaining_seconds}s"


def estimate_runtime(
    models: list[ModelSpec],
    sequences: list[Mot17Sequence],
    confidence_threshold: float,
    match_iou_threshold: float,
    sample_count: int,
) -> dict[str, float]:
    if sample_count <= 0:
        raise ValueError("sample_count must be greater than zero")

    estimates: dict[str, float] = {}
    for model in models:
        model_start = time.perf_counter()
        detector = model.factory()
        estimated_seconds = time.perf_counter() - model_start
        try:
            for sequence in sequences:
                warmup_start = time.perf_counter()
                warmup_frame = _read_frame(sequence.frames[0])
                detector.detect(
                    warmup_frame,
                    confidence_threshold=confidence_threshold,
                )
                _synchronize_cuda()
                estimated_seconds += time.perf_counter() - warmup_start

                sample_indexes = _sample_indexes(
                    len(sequence.frames),
                    sample_count,
                )
                sample_elapsed = 0.0
                sampled_frames = 0
                for index in sample_indexes:
                    frame_ref = sequence.frames[index]
                    start = time.perf_counter()
                    frame = _read_frame(frame_ref)
                    accumulator = _new_accumulator()
                    _record_frame(
                        detector,
                        frame,
                        sequence.ground_truth.get(frame_ref.number, []),
                        confidence_threshold,
                        match_iou_threshold,
                        accumulator,
                    )
                    sample_elapsed += time.perf_counter() - start
                    sampled_frames += 1

                if sampled_frames:
                    estimated_seconds += (
                        sample_elapsed
                        / sampled_frames
                        * _frame_count(sequence)
                    )
                LOGGER.info(
                    "Estimate sampling: %s / %s, %d samples at %dx%d",
                    model.name,
                    sequence.name,
                    sampled_frames,
                    warmup_frame.shape[1],
                    warmup_frame.shape[0],
                )
        finally:
            del detector
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        estimates[model.name] = estimated_seconds

    LOGGER.info("Estimated runtime:")
    for model in models:
        LOGGER.info(
            "  %-16s %s",
            model.name,
            format_duration(estimates[model.name]),
        )
    LOGGER.info("  %-16s %s", "Total", format_duration(sum(estimates.values())))
    return estimates


def _sample_indexes(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count == 1:
        return [0]
    count = min(sample_count, frame_count - 1)
    if count == 1:
        return [frame_count // 2]
    return [
        round(
            1 + position * (frame_count - 2) / (count - 1)
        )
        for position in range(count)
    ]


def _update_checkpoint_aggregates(checkpoint: dict[str, Any]) -> None:
    checkpoint["aggregates"] = {}
    for model_name, results in checkpoint["results"].items():
        checkpoint["aggregates"][model_name] = aggregate_sequence_results(
            list(results.values())
        )


def run_benchmark(
    models: list[ModelSpec],
    sequences: list[Mot17Sequence],
    confidence_threshold: float,
    match_iou_threshold: float,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    progress_every: int = 100,
) -> dict[str, dict[str, Any]]:
    if progress_every <= 0:
        raise ValueError("progress_every must be greater than zero")

    checkpoint["status"] = "running"
    checkpoint["completed_at"] = None
    save_checkpoint(checkpoint_path, checkpoint)

    total_combinations = len(models) * len(sequences)
    for model_index, model in enumerate(models, start=1):
        pending_sequences = [
            sequence
            for sequence in sequences
            if not combination_completed(
                checkpoint,
                model.name,
                sequence.name,
            )
        ]

        for sequence in sequences:
            state = (
                "completed -> skip"
                if combination_completed(
                    checkpoint, model.name, sequence.name
                )
                else "pending -> run"
            )
            LOGGER.info(
                "%s / %s  %s",
                model.name,
                sequence.name,
                state,
            )

        if not pending_sequences:
            continue

        detector = model.factory()
        try:
            model_results = checkpoint["results"].setdefault(
                model.name, {}
            )
            for sequence in pending_sequences:
                original_sequence_index = sequences.index(sequence) + 1
                combination_number = (
                    (model_index - 1) * len(sequences)
                    + original_sequence_index
                )
                LOGGER.info(
                    "Starting %s / %s (%d frames; combination %d/%d)",
                    model.name,
                    sequence.name,
                    len(sequence.frames),
                    combination_number,
                    total_combinations,
                )
                sequence_start = time.perf_counter()
                warmup_frame = _read_frame(sequence.frames[0])
                detector.detect(
                    warmup_frame,
                    confidence_threshold=confidence_threshold,
                )
                _synchronize_cuda()

                accumulator = _new_accumulator()
                for frame_position, frame_ref in enumerate(
                    sequence.frames, start=1
                ):
                    frame = _read_frame(frame_ref)
                    _record_frame(
                        detector,
                        frame,
                        sequence.ground_truth.get(frame_ref.number, []),
                        confidence_threshold,
                        match_iou_threshold,
                        accumulator,
                    )
                    accumulator["frames"] += 1

                    if (
                        frame_position % progress_every == 0
                        or frame_position == len(sequence.frames)
                    ):
                        elapsed = time.perf_counter() - sequence_start
                        eta = (
                            elapsed
                            / frame_position
                            * (len(sequence.frames) - frame_position)
                        )
                        LOGGER.info(
                            "%s | %s | %d/%d | elapsed %s | ETA %s",
                            model.name,
                            sequence.name,
                            frame_position,
                            len(sequence.frames),
                            format_duration(elapsed),
                            format_duration(eta),
                        )

                model_results[sequence.name] = _summarize(accumulator)
                _update_checkpoint_aggregates(checkpoint)
                save_checkpoint(checkpoint_path, checkpoint)

                metrics = model_results[sequence.name]
                LOGGER.info(
                    "Finished %s / %s in %s",
                    model.name,
                    sequence.name,
                    format_duration(time.perf_counter() - sequence_start),
                )
                _log_result(f"{model.name} / {sequence.name}", metrics)
        except Exception:
            LOGGER.exception("Failed while evaluating model %s", model.name)
            raise
        finally:
            del detector
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_completed = all(
        combination_completed(checkpoint, model.name, sequence.name)
        for model in models
        for sequence in sequences
    )
    checkpoint["status"] = "completed" if all_completed else "running"
    if all_completed:
        checkpoint["completed_at"] = utc_timestamp()
    _update_checkpoint_aggregates(checkpoint)
    save_checkpoint(checkpoint_path, checkpoint)

    LOGGER.info("Final benchmark results:")
    for model in models:
        aggregate = checkpoint["aggregates"].get(model.name)
        if aggregate is not None:
            _log_result(model.name, aggregate)
    return checkpoint["aggregates"]


def sequence_configuration(
    sequences: list[Mot17Sequence],
) -> list[dict[str, str | int]]:
    return [
        {
            "name": sequence.name,
            "path": str(sequence.path.resolve()),
            "frames": len(sequence.frames),
        }
        for sequence in sequences
    ]
