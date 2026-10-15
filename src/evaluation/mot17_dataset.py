import csv
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.detection_metrics import Box


DEFAULT_TRAIN_PATH = Path("datasets/MOT17/train")
DEFAULT_SEQUENCE_PATH = DEFAULT_TRAIN_PATH / "MOT17-04-FRCNN"


@dataclass(frozen=True)
class Mot17Frame:
    number: int
    path: Path


@dataclass(frozen=True)
class Mot17Sequence:
    name: str
    path: Path
    frames: tuple[Mot17Frame, ...]
    ground_truth: dict[int, list[Box]]


def load_ground_truth(gt_path: Path) -> dict[int, list[Box]]:
    """Load only marked class-1 boxes; this is not MOT17 ignored-region handling."""
    ground_truth: dict[int, list[Box]] = {}

    with gt_path.open(newline="") as gt_file:
        for row in csv.reader(gt_file):
            if len(row) < 9:
                continue

            frame_number = int(row[0])
            x = float(row[2])
            y = float(row[3])
            width = float(row[4])
            height = float(row[5])
            mark = int(row[6])
            class_id = int(row[7])

            if mark != 1 or class_id != 1:
                continue

            ground_truth.setdefault(frame_number, []).append(
                (x, y, x + width, y + height)
            )

    return ground_truth


def discover_sequence_paths(
    sequence_path: Path = DEFAULT_SEQUENCE_PATH,
    all_train: bool = False,
    train_path: Path = DEFAULT_TRAIN_PATH,
) -> list[Path]:
    if not all_train:
        return [sequence_path]

    sequences = sorted(train_path.glob("MOT17-*-FRCNN"))
    if not sequences:
        raise RuntimeError(
            f"No MOT17 training sequences found in {train_path}"
        )
    return sequences


def load_sequence(
    sequence_path: Path,
    max_frames: int | None = None,
) -> Mot17Sequence:
    frames_path = sequence_path / "img1"
    gt_path = sequence_path / "gt" / "gt.txt"
    frame_paths = sorted(frames_path.glob("*.jpg"))

    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_path}")
    if not gt_path.exists():
        raise RuntimeError(f"Ground truth not found: {gt_path}")

    frames = tuple(
        Mot17Frame(
            number=int(frame_path.stem),
            path=frame_path,
        )
        for frame_path in frame_paths
    )
    return Mot17Sequence(
        name=sequence_path.name,
        path=sequence_path,
        frames=frames,
        ground_truth=load_ground_truth(gt_path),
    )


def load_sequences(
    sequence_path: Path = DEFAULT_SEQUENCE_PATH,
    all_train: bool = False,
    max_frames: int | None = None,
    train_path: Path = DEFAULT_TRAIN_PATH,
) -> list[Mot17Sequence]:
    paths = discover_sequence_paths(
        sequence_path=sequence_path,
        all_train=all_train,
        train_path=train_path,
    )
    return [
        load_sequence(path, max_frames=max_frames)
        for path in paths
    ]
