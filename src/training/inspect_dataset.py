"""Audit MOT17 preparation retention and target-label accounting."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
from typing import Any

from src.training.mot17_training_dataset import PREPARATION_POLICY_VERSION, mot17_box_to_yolo
from src.training.training_config import DEFAULT_SPLIT_CONFIG, load_training_config


def audit_dataset(config_path: Path, dataset_root: Path | None = None) -> dict[str, Any]:
    config = load_training_config(config_path)
    if dataset_root is not None:
        config = type(config)(**{**config.__dict__, "dataset_root": dataset_root})
    selected = (*config.train_sequences, *config.validation_sequences)
    for name in selected:
        sequence_dir = config.dataset_root / name
        if not sequence_dir.is_dir() or not (sequence_dir / "img1").is_dir() or not (sequence_dir / "gt" / "gt.txt").is_file():
            raise FileNotFoundError(f"Missing sequence/images/GT for {name}: {sequence_dir}")

    import cv2

    report: dict[str, Any] = {
        "format_version": 2,
        "split_config": str(config_path.resolve()),
        "dataset_root": str(config.dataset_root.resolve()),
        "split": {"train": list(config.train_sequences), "validation": list(config.validation_sequences)},
        "preparation_policy_version": PREPARATION_POLICY_VERSION,
        "preparation_policy": "Retain all readable frames; convert only mark=1,class=1 annotations to pedestrian labels. Other annotations are excluded from positive labels, not used as ignore regions.",
        "ignore_region_loss": False,
        "sequences": {},
        "class_annotation_counts": {},
        "totals": {},
        "warnings": [],
    }
    class_counts_all: Counter[str] = Counter()
    totals_by_partition: dict[str, Counter[str]] = {"train": Counter(), "validation": Counter()}

    for partition, names in (("train", config.train_sequences), ("validation", config.validation_sequences)):
        for name in names:
            sequence_dir = config.dataset_root / name
            image_paths = sorted((sequence_dir / "img1").glob("*.jpg"))
            frame_numbers = {int(path.stem) for path in image_paths}
            image_dimensions: dict[int, tuple[int, int]] = {}
            resolutions: Counter[str] = Counter()
            for image_path in image_paths:
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                height, width = image.shape[:2]
                image_dimensions[int(image_path.stem)] = (width, height)
                resolutions[f"{width}x{height}"] += 1

            target_raw = target_valid = 0
            invalid_target = missing_frame_target = 0
            target_by_frame: Counter[int] = Counter()
            visibility: list[float] = []
            class_counts: Counter[str] = Counter()
            excluded_by_class: Counter[str] = Counter()
            frames_with_excluded: set[int] = set()
            with (sequence_dir / "gt" / "gt.txt").open(newline="", encoding="utf-8") as gt_file:
                for row in csv.reader(gt_file):
                    if len(row) < 9:
                        continue
                    frame = int(row[0])
                    mark, class_id = int(row[6]), int(row[7])
                    class_counts[str(class_id)] += 1
                    class_counts_all[str(class_id)] += 1
                    if mark != 1 or class_id != 1:
                        reason = f"class_{class_id}" if mark == 1 else f"mark_{mark}_class_{class_id}"
                        excluded_by_class[reason] += 1
                        if frame in frame_numbers:
                            frames_with_excluded.add(frame)
                        continue

                    target_raw += 1
                    if frame not in image_dimensions:
                        if frame not in frame_numbers:
                            missing_frame_target += 1
                        else:
                            invalid_target += 1  # image exists but cannot be decoded; preparation stops on this input
                        continue
                    x, y, width, height = map(float, row[2:6])
                    try:
                        mot17_box_to_yolo((x, y, x + width, y + height), *image_dimensions[frame])
                    except ValueError:
                        invalid_target += 1
                        continue
                    target_valid += 1
                    target_by_frame[frame] += 1
                    visibility.append(float(row[8]))

            retained_frames = len(image_dimensions)
            target_lost = target_raw - target_valid
            visibility_sorted = sorted(visibility)
            seq_report = {
                "partition": partition,
                "frames_original": len(image_paths),
                "frames_retained": retained_frames,
                "frames_retained_percent": 100.0 * retained_frames / len(image_paths) if image_paths else 0.0,
                "frames_discarded_for_non_target_annotations": 0,
                "frames_without_target_gt": retained_frames - sum(bool(target_by_frame[n]) for n in image_dimensions),
                "frames_with_target_gt": sum(bool(target_by_frame[n]) for n in image_dimensions),
                "frames_with_excluded_annotations_but_retained": len(frames_with_excluded & set(image_dimensions)),
                "unreadable_frames": len(image_paths) - retained_frames,
                "target_gt_annotations_raw": target_raw,
                "target_gt_annotations_valid": target_valid,
                "target_gt_annotations_lost": target_lost,
                "target_gt_annotation_loss_percent": 100.0 * target_lost / target_raw if target_raw else 0.0,
                "target_gt_annotations_lost_invalid_or_unreadable": invalid_target,
                "target_gt_annotations_lost_missing_image_frame": missing_frame_target,
                "excluded_annotations_by_class": dict(sorted(excluded_by_class.items())),
                "annotation_counts_by_mot_class": dict(sorted(class_counts.items(), key=lambda pair: int(pair[0]))),
                "resolutions": dict(resolutions),
                "visibility": {
                    "count": len(visibility_sorted),
                    "mean": statistics.fmean(visibility_sorted) if visibility_sorted else None,
                    "median": statistics.median(visibility_sorted) if visibility_sorted else None,
                    "min": min(visibility_sorted) if visibility_sorted else None,
                    "max": max(visibility_sorted) if visibility_sorted else None,
                    "bins": {
                        "<0.25": sum(v < 0.25 for v in visibility_sorted),
                        "0.25-0.50": sum(0.25 <= v < 0.5 for v in visibility_sorted),
                        "0.50-0.75": sum(0.5 <= v < 0.75 for v in visibility_sorted),
                        ">=0.75": sum(v >= 0.75 for v in visibility_sorted),
                    },
                },
            }
            report["sequences"][name] = seq_report
            partition_totals = totals_by_partition[partition]
            for key in ("frames_original", "frames_retained", "frames_with_target_gt", "frames_without_target_gt", "target_gt_annotations_raw", "target_gt_annotations_valid", "target_gt_annotations_lost", "unreadable_frames"):
                partition_totals[key] += seq_report[key]
            if seq_report["unreadable_frames"]:
                report["warnings"].append(f"{name}: {seq_report['unreadable_frames']} unreadable images; dataset preparation fails instead of silently dropping them.")
            if target_lost:
                report["warnings"].append(f"{name}: {target_lost} target annotations cannot become labels (invalid/unreadable boxes={invalid_target}, missing image frames={missing_frame_target}).")

    report["class_annotation_counts"] = dict(sorted(class_counts_all.items(), key=lambda pair: int(pair[0])))
    report["totals"] = {key: dict(value) for key, value in totals_by_partition.items()}
    return report


def assert_preparation_matches_audit(report: dict[str, Any], prepared: dict[str, Any]) -> None:
    """Fail if a supervised prepared manifest disagrees with source accounting."""
    from src.training.mot17_training_dataset import PREPARATION_POLICY_VERSION

    if prepared.get("preparation_policy_version") != PREPARATION_POLICY_VERSION:
        raise ValueError("Prepared dataset uses a different preparation policy version.")
    if prepared.get("mode") != "supervised":
        raise ValueError("Source GT label parity can only be checked for supervised preparation.")
    stats = prepared.get("stats", {})
    checks = (
        ("train", "frames_retained", "train_frames_retained"),
        ("validation", "frames_retained", "validation_frames_retained"),
        ("train", "target_gt_annotations_valid", "train_boxes"),
        ("validation", "target_gt_annotations_valid", "validation_boxes"),
    )
    mismatches = []
    for partition, audit_key, prepared_key in checks:
        expected = report["totals"][partition][audit_key]
        actual = stats.get(prepared_key)
        if actual != expected:
            mismatches.append(f"{partition} {prepared_key}: prepared={actual}, audit={expected}")
    if mismatches:
        raise ValueError("Prepared dataset does not match audit: " + "; ".join(mismatches))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, help="Override the dataset root from the split config.")
    parser.add_argument("--output", type=Path, default=Path("results/dataset_audit/mot17_training_split.json"))
    args = parser.parse_args(argv)
    report = audit_dataset(args.split_config, dataset_root=args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, row in report["sequences"].items():
        print(f"{name} [{row['partition']}]: frames {row['frames_retained']}/{row['frames_original']} retained; target GT {row['target_gt_annotations_valid']}/{row['target_gt_annotations_raw']} valid; frames with target={row['frames_with_target_gt']}; frames with excluded annotations retained={row['frames_with_excluded_annotations_but_retained']}; excluded classes={row['excluded_annotations_by_class']}")
    print(json.dumps(report["totals"], indent=2))
    print(f"Audit JSON: {args.output}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
