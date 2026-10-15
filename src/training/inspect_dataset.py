"""Audit the configured MOT17 split without changing it."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
from typing import Any

from src.training.training_config import DEFAULT_SPLIT_CONFIG, load_training_config


def audit_dataset(config_path: Path) -> dict[str, Any]:
    config = load_training_config(config_path)
    for name in (*config.train_sequences, *config.validation_sequences):
        sequence_dir = config.dataset_root / name
        if not sequence_dir.is_dir() or not (sequence_dir / "img1").is_dir() or not (sequence_dir / "gt" / "gt.txt").is_file():
            raise FileNotFoundError(f"Missing sequence/images/GT for {name}: {sequence_dir}")
    import cv2
    variants_by_scene: dict[str, set[str]] = {}
    for path in config.dataset_root.glob("MOT17-*"):
        parts = path.name.split("-")
        if len(parts) >= 3:
            variants_by_scene.setdefault("-".join(parts[:2]), set()).add(parts[-1])

    report: dict[str, Any] = {
        "format_version": 1,
        "split_config": str(config_path.resolve()),
        "dataset_root": str(config.dataset_root.resolve()),
        "split": {"train": list(config.train_sequences), "validation": list(config.validation_sequences)},
        "ignored_frame_policy": config.ignored_frame_policy,
        "sequences": {},
        "class_annotation_counts": {},
        "warnings": [],
        "policy_review": {"configured_policy": config.ignored_frame_policy, "changed_by_audit": False, "alert_threshold_percent": 10.0},
        "policy_review": {"configured_policy": config.ignored_frame_policy, "changed_by_audit": False, "alert_threshold_percent": 10.0},
    }
    total_train_frames = total_train_dropped = 0
    all_class_counts: Counter[str] = Counter()
    for partition, names in (("train", config.train_sequences), ("validation", config.validation_sequences)):
        for name in names:
            sequence_dir = config.dataset_root / name
            img_dir, gt_path = sequence_dir / "img1", sequence_dir / "gt" / "gt.txt"
            if not sequence_dir.is_dir() or not img_dir.is_dir() or not gt_path.is_file():
                raise FileNotFoundError(f"Missing sequence/images/GT for {name}: {sequence_dir}")
            scene = "-".join(name.split("-")[:2])
            variants = variants_by_scene.get(scene, set())
            frames = sorted(img_dir.glob("*.jpg"))
            frame_numbers = {int(p.stem) for p in frames}
            class_counts: Counter[str] = Counter()
            excluded_by_frame: dict[int, Counter[str]] = {}
            valid_ped_by_frame: Counter[int] = Counter()
            visibility: list[float] = []
            with gt_path.open(newline="", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if len(row) < 9:
                        continue
                    frame, mark, class_id = int(row[0]), int(row[6]), int(row[7])
                    class_counts[str(class_id)] += 1
                    all_class_counts[str(class_id)] += 1
                    width, height = float(row[4]), float(row[5])
                    if mark == 1 and class_id == 1 and width > 0 and height > 0:
                        valid_ped_by_frame[frame] += 1
                        visibility.append(float(row[8]))
                    else:
                        reason = f"class_{class_id}" if mark == 1 else f"mark_{mark}_class_{class_id}"
                        excluded_by_frame.setdefault(frame, Counter())[reason] += 1
            resolutions: Counter[str] = Counter()
            unreadable = 0
            for image_path in frames:
                image = cv2.imread(str(image_path))
                if image is None:
                    unreadable += 1
                    continue
                h, w = image.shape[:2]
                resolutions[f"{w}x{h}"] += 1
            dropped = frame_numbers & excluded_by_frame.keys()
            lost_pedestrians = sum(valid_ped_by_frame[n] for n in dropped)
            dropped_reasons: Counter[str] = Counter()
            dropped_annotation_reasons: Counter[str] = Counter()
            for frame in dropped:
                dropped_reasons.update(excluded_by_frame[frame].keys())
                dropped_annotation_reasons.update(excluded_by_frame[frame])
            count = len(frames)
            total_dropped = len(dropped) if partition == "train" else 0
            if partition == "train":
                total_train_frames += count
                total_train_dropped += total_dropped
            vis_sorted = sorted(visibility)
            visibility_stats = {
                "count": len(vis_sorted),
                "mean": statistics.fmean(vis_sorted) if vis_sorted else None,
                "median": statistics.median(vis_sorted) if vis_sorted else None,
                "min": min(vis_sorted) if vis_sorted else None,
                "max": max(vis_sorted) if vis_sorted else None,
                "bins": {
                    "<0.25": sum(v < 0.25 for v in vis_sorted),
                    "0.25-0.50": sum(0.25 <= v < 0.5 for v in vis_sorted),
                    "0.50-0.75": sum(0.5 <= v < 0.75 for v in vis_sorted),
                    ">=0.75": sum(v >= 0.75 for v in vis_sorted),
                },
            }
            report["sequences"][name] = {
                "partition": partition, "exists": True, "variants_found_for_scene": sorted(variants),
                "frames": count, "pedestrian_gt_valid": sum(valid_ped_by_frame.values()),
                "images_with_pedestrian_labels": sum(bool(valid_ped_by_frame[n]) for n in frame_numbers),
                "images_without_pedestrian_labels": sum(not valid_ped_by_frame[n] for n in frame_numbers),
                "resolutions": dict(resolutions), "unreadable_images": unreadable,
                "visibility": visibility_stats,
                "frames_dropped": len(dropped), "frames_dropped_percent": 100 * len(dropped) / count if count else 0.0,
                "frames_dropped_reasons": dict(sorted(dropped_reasons.items())),
                "excluded_annotations_on_dropped_frames_by_reason": dict(sorted(dropped_annotation_reasons.items())),
                "valid_pedestrian_gt_lost_on_dropped_frames": lost_pedestrians,
                "annotation_counts_by_mot_class": dict(sorted(class_counts.items(), key=lambda item: int(item[0]))),
            }
            selected_for_scene = [seq for seq in (*config.train_sequences, *config.validation_sequences) if "-".join(seq.split("-")[:2]) == scene]
            if len(selected_for_scene) != 1:
                report["warnings"].append(f"{scene} does not have exactly one selected variant: {selected_for_scene}")
    report["class_annotation_counts"] = dict(sorted(all_class_counts.items(), key=lambda item: int(item[0])))
    report["train_dropped_frames"] = {
        "count": total_train_dropped,
        "total_train_frames": total_train_frames,
        "percent_of_train_frames": 100 * total_train_dropped / total_train_frames if total_train_frames else 0.0,
    }
    if report["train_dropped_frames"]["percent_of_train_frames"] > 10:
        report["warnings"].append("More than 10% of train frames are dropped under the configured policy; review the policy before training.")
        report["policy_review"]["recommendation"] = (
            "Pause before full training and discuss: retain the current policy and accept the reduced sample, "
            "or design an explicit ignored-region-aware data/loss treatment. No policy or custom loss was changed."
        )
    else:
        report["policy_review"]["recommendation"] = "Dropped-frame share is at or below the 10% alert level; retaining the configured policy is reasonable for the initial run."
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("results/dataset_audit/mot17_training_split.json"))
    args = parser.parse_args(argv)
    report = audit_dataset(args.split_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, row in report["sequences"].items():
        print(f"{name} [{row['partition']}]: frames={row['frames']}, pedestrian GT={row['pedestrian_gt_valid']}, labeled/empty images={row['images_with_pedestrian_labels']}/{row['images_without_pedestrian_labels']}, dropped={row['frames_dropped']} ({row['frames_dropped_percent']:.2f}%), lost pedestrian GT={row['valid_pedestrian_gt_lost_on_dropped_frames']}")
        print(f"  resolutions={row['resolutions']}; visibility={row['visibility']}; dropped by reason={row['frames_dropped_reasons']}; MOT classes={row['annotation_counts_by_mot_class']}")
    print(f"Train frames dropped: {report['train_dropped_frames']['count']}/{report['train_dropped_frames']['total_train_frames']} ({report['train_dropped_frames']['percent_of_train_frames']:.2f}%)")
    print(f"Audit JSON: {args.output}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
