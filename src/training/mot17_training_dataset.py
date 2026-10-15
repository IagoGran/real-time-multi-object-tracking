from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
from typing import Iterable

from src.training.training_config import Mot17TrainingConfig


@dataclass(frozen=True)
class Mot17Annotations:
    boxes_by_frame: dict[int, list[tuple[float, float, float, float]]]
    frames_with_excluded_annotations: frozenset[int]


@dataclass(frozen=True)
class PreparedDataset:
    root: Path
    yaml_path: Path
    train_images: tuple[Path, ...]
    validation_images: tuple[Path, ...]
    stats: dict[str, int]


def read_mot17_annotations(gt_path: Path) -> Mot17Annotations:
    boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    excluded_frames: set[int] = set()
    with gt_path.open(newline="", encoding="utf-8") as gt_file:
        for row in csv.reader(gt_file):
            if len(row) < 9:
                continue
            frame_number = int(row[0])
            x, y, width, height = map(float, row[2:6])
            mark = int(row[6])
            class_id = int(row[7])
            if mark != 1 or class_id != 1 or width <= 0 or height <= 0:
                excluded_frames.add(frame_number)
                continue
            # MOT17 visibility is recorded in column 8. Evaluation currently
            # includes all marked pedestrian boxes, so training applies no new
            # visibility cutoff.
            boxes.setdefault(frame_number, []).append(
                (x, y, x + width, y + height)
            )
    return Mot17Annotations(boxes, frozenset(excluded_frames))


def mot17_box_to_yolo(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    class_id: int = 0,
) -> tuple[int, float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), float(image_width))
    x2 = min(max(x2, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    y2 = min(max(y2, 0.0), float(image_height))
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError("Bounding box has no area after clipping to the image.")
    center_x = (x1 + x2) / 2.0 / image_width
    center_y = (y1 + y2) / 2.0 / image_height
    return class_id, center_x, center_y, width / image_width, height / image_height


def _sequence_path(config: Mot17TrainingConfig, sequence_name: str) -> Path:
    path = config.dataset_root / sequence_name
    if not path.is_dir():
        raise FileNotFoundError(f"MOT17 sequence not found: {path}")
    return path


def _ensure_image_link(source_dir: Path, destination_dir: Path) -> None:
    if destination_dir.exists():
        return
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination_dir.symlink_to(source_dir.resolve(), target_is_directory=True)
    except OSError:
        # A hard link avoids copying multi-gigabyte image sequences. It works
        # when the source and prepared dataset are on the same volume.
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source in source_dir.glob("*.jpg"):
            target = destination_dir / source.name
            if not target.exists():
                os.link(source, target)


def _write_yolo_labels(
    destination: Path,
    boxes: Iterable[tuple[float, float, float, float]],
    image_size: tuple[int, int],
) -> int:
    width, height = image_size
    rows = []
    for box in boxes:
        try:
            label = mot17_box_to_yolo(box, width, height)
        except ValueError:
            continue
        rows.append("0 " + " ".join(f"{value:.8f}" for value in label[1:]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def prepare_mot17_dataset(
    config: Mot17TrainingConfig,
    output_dir: Path,
    mode: str = "supervised",
    pseudo_labels: dict[str, list[tuple[float, float, float, float]]] | None = None,
) -> PreparedDataset:
    """Stage one FRCNN variant per configured scene and write YOLO labels."""
    if mode not in {"supervised", "teacher"}:
        raise ValueError(f"Unsupported training mode: {mode}")
    if mode == "teacher" and pseudo_labels is None:
        raise ValueError("Teacher mode requires cached/generated pseudo-labels.")

    import cv2

    root = output_dir.resolve()
    train_entries: list[Path] = []
    validation_entries: list[Path] = []
    stats = {
        "train_frames": 0,
        "validation_frames": 0,
        "train_frames_skipped_for_ignored_annotations": 0,
        "validation_frames_skipped_for_ignored_annotations": 0,
        "train_boxes": 0,
        "validation_boxes": 0,
    }

    for split_name, sequence_names, entries in (
        ("train", config.train_sequences, train_entries),
        ("val", config.validation_sequences, validation_entries),
    ):
        stats_prefix = "train" if split_name == "train" else "validation"
        for sequence_name in sequence_names:
            sequence_dir = _sequence_path(config, sequence_name)
            image_source = sequence_dir / "img1"
            gt_path = sequence_dir / "gt" / "gt.txt"
            if not image_source.is_dir():
                raise FileNotFoundError(f"MOT17 images not found: {image_source}")
            if not gt_path.is_file():
                raise FileNotFoundError(f"MOT17 ground truth not found: {gt_path}")
            annotations = read_mot17_annotations(gt_path)
            staged_dir = root / "images" / split_name / sequence_name
            _ensure_image_link(image_source, staged_dir)

            for image_path in sorted(image_source.glob("*.jpg")):
                frame_number = int(image_path.stem)
                if split_name == "val" or mode == "supervised":
                    if frame_number in annotations.frames_with_excluded_annotations:
                        stats[f"{stats_prefix}_frames_skipped_for_ignored_annotations"] += 1
                        continue

                staged_image = staged_dir / image_path.name
                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError(f"Failed to read MOT17 image: {image_path}")
                image_height, image_width = image.shape[:2]
                label_path = (
                    root / "labels" / split_name / sequence_name
                    / f"{image_path.stem}.txt"
                )

                if split_name == "val" or mode == "supervised":
                    boxes = annotations.boxes_by_frame.get(frame_number, [])
                    box_count = _write_yolo_labels(
                        label_path, boxes, (image_width, image_height)
                    )
                else:
                    raw_boxes = (pseudo_labels or {}).get(str(image_path.resolve()), [])
                    box_count = _write_yolo_labels(
                        label_path, raw_boxes, (image_width, image_height)
                    )
                # Keep the staged /images/ path so Ultralytics resolves the
                # matching label under /labels/, even when images are symlinked.
                entries.append(staged_image.absolute())
                stats[f"{stats_prefix}_frames"] += 1
                stats[f"{stats_prefix}_boxes"] += box_count

    if not train_entries or not validation_entries:
        raise RuntimeError("The configured split produced an empty train or validation set.")
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.txt").write_text(
        "\n".join(path.as_posix() for path in train_entries) + "\n", encoding="utf-8"
    )
    (root / "val.txt").write_text(
        "\n".join(path.as_posix() for path in validation_entries) + "\n", encoding="utf-8"
    )
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        "path: " + root.as_posix() + "\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "names:\n"
        "  0: person\n",
        encoding="utf-8",
    )
    manifest = {
        "format_version": 1,
        "mode": mode,
        "train_sequences": list(config.train_sequences),
        "validation_sequences": list(config.validation_sequences),
        "ignored_frame_policy": config.ignored_frame_policy,
        "stats": stats,
    }
    (root / "dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return PreparedDataset(
        root=root,
        yaml_path=yaml_path,
        train_images=tuple(train_entries),
        validation_images=tuple(validation_entries),
        stats=stats,
    )
