"""Measure detector inference at operational confidence thresholds (no GT/matching)."""
import argparse
import json
from pathlib import Path
import time

import cv2
import torch

from src.detection.yolo_detector import YoloDetector
from src.training.training_config import DEFAULT_SPLIT_CONFIG, load_training_config


def benchmark(model_spec: str, frames: list[Path], threshold: float, warmup: int = 5, device: str | None = None) -> dict[str, object]:
    if "=" not in model_spec:
        raise ValueError("--model must use NAME=WEIGHTS")
    name, weights = model_spec.split("=", 1)
    detector = YoloDetector(weights, ["person"], inference_confidence=threshold, device=device)
    images = []
    for path in frames:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Could not read frame {path}")
        images.append(image)
    if not images:
        raise ValueError("No benchmark frames selected")
    for image in images[:min(warmup, len(images))]:
        detector.predict(image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    for image in images:
        detector.predict(image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    milliseconds = elapsed * 1000 / len(images)
    return {"model": name, "weights": weights, "confidence_threshold": threshold, "frames": len(images), "warmup_frames": min(warmup, len(images)), "inference_ms_per_frame": milliseconds, "fps": len(images) / elapsed if elapsed else 0.0, "timing_scope": "detector.predict at operational confidence; excludes frame decoding; no matching"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="NAME=WEIGHTS; repeatable")
    parser.add_argument("--sequence", type=Path, action="append", default=[])
    parser.add_argument("--split-config", type=Path)
    parser.add_argument("--partition", choices=("train", "validation"))
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, action="append", required=True)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.max_frames <= 0 or args.warmup < 0 or any(not 0 <= c <= 1 for c in args.confidence_threshold):
        parser.error("max-frames must be positive, warmup nonnegative, and thresholds within [0, 1].")
    if bool(args.split_config) != bool(args.partition) or (args.split_config and args.sequence):
        parser.error("Use either --sequence or both --split-config and --partition.")
    sequence_dirs = args.sequence
    if args.split_config:
        config = load_training_config(args.split_config)
        names = config.train_sequences if args.partition == "train" else config.validation_sequences
        sequence_dirs = [config.dataset_root / name for name in names]
    if not sequence_dirs:
        parser.error("Provide sequence(s) or a split partition.")
    frames = [p for directory in sequence_dirs for p in sorted((directory / "img1").glob("*.jpg"))[:args.max_frames]]
    rows = [benchmark(model, frames, threshold, args.warmup, args.device) for model in args.model for threshold in args.confidence_threshold]
    print(json.dumps(rows, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
