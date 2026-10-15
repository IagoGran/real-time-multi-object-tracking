import csv
import json
import sys
from types import SimpleNamespace

import pytest

from src.training.mot17_training_dataset import (
    PREPARATION_POLICY_VERSION,
    mot17_box_to_yolo,
    prepare_mot17_dataset,
)
from src.training.training_config import Mot17TrainingConfig, validate_sequence_split


def _config(root):
    return Mot17TrainingConfig(
        name="test",
        dataset_root=root,
        prepared_dataset_root=root / "prepared",
        variant="FRCNN",
        train_sequences=("MOT17-02-FRCNN",),
        validation_sequences=("MOT17-04-FRCNN",),
        preparation_policy="all_frames_target_class_only_v2",
    )


def _make_sequence(root, name, rows):
    sequence = root / name
    (sequence / "img1").mkdir(parents=True)
    (sequence / "gt").mkdir()
    (sequence / "img1" / "000001.jpg").touch()
    (sequence / "img1" / "000002.jpg").touch()
    with (sequence / "gt" / "gt.txt").open("w", newline="") as file:
        csv.writer(file).writerows(rows)


def test_split_rejects_sequence_overlap_and_duplicate_variants():
    with pytest.raises(ValueError, match="leakage"):
        validate_sequence_split(["MOT17-02-FRCNN"], ["MOT17-02-FRCNN"])
    with pytest.raises(ValueError, match="FRCNN"):
        validate_sequence_split(["MOT17-02-DPM"], ["MOT17-04-FRCNN"])


def test_mot17_box_converts_to_normalized_yolo_xywh():
    assert mot17_box_to_yolo((10, 20, 30, 60), 100, 100) == (0, 0.2, 0.4, 0.2, 0.4)


def test_preparation_keeps_all_frames_and_only_labels_valid_pedestrians(tmp_path, monkeypatch):
    train_root = tmp_path / "train"
    _make_sequence(
        train_root,
        "MOT17-02-FRCNN",
        [
            [1, 1, 10, 10, 20, 40, 1, 1, 0.15],
            [1, 2, 30, 10, 20, 40, 1, 2, 1.0],
            [1, 3, 60, 10, 20, 40, 1, 9, 1.0],
            [1, 4, 80, 10, 20, 40, 1, 11, 1.0],
            [2, 3, 15, 12, 20, 40, 1, 2, 0.0],
        ],
    )
    _make_sequence(
        train_root,
        "MOT17-04-FRCNN",
        [[1, 1, 10, 10, 20, 40, 1, 1, 0.2]],
    )
    # An alternate variant exists but is never selected by the explicit split.
    _make_sequence(train_root, "MOT17-02-DPM", [[1, 1, 1, 1, 1, 1, 1, 1, 1]])
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(imread=lambda _: SimpleNamespace(shape=(100, 100, 3))))

    prepared = prepare_mot17_dataset(_config(train_root), tmp_path / "out")

    assert len(prepared.train_images) == 2
    assert len(prepared.validation_images) == 2
    assert prepared.stats["train_frames_original"] == 2
    assert prepared.stats["train_frames_retained"] == 2
    assert prepared.stats["train_boxes"] == 1
    label = (tmp_path / "out" / "labels" / "train" / "MOT17-02-FRCNN" / "000002.txt")
    assert label.read_text(encoding="utf-8") == ""
    mixed_label = (tmp_path / "out" / "labels" / "train" / "MOT17-02-FRCNN" / "000001.txt")
    assert len(mixed_label.read_text(encoding="utf-8").splitlines()) == 1
    assert "class_" not in mixed_label.read_text(encoding="utf-8")
    assert json.loads((tmp_path / "out" / "dataset-manifest.json").read_text(encoding="utf-8"))["preparation_policy_version"] == PREPARATION_POLICY_VERSION
    assert all("DPM" not in str(path) for path in prepared.train_images + prepared.validation_images)


def test_teacher_dataset_does_not_mix_training_ground_truth(tmp_path, monkeypatch):
    train_root = tmp_path / "train"
    _make_sequence(
        train_root,
        "MOT17-02-FRCNN",
        [[1, 1, 10, 10, 20, 40, 1, 1, 1.0], [1, 2, 30, 10, 20, 40, 1, 2, 1.0]],
    )
    _make_sequence(
        train_root,
        "MOT17-04-FRCNN",
        [[1, 1, 10, 10, 20, 40, 1, 1, 1.0], [1, 2, 30, 10, 20, 40, 1, 2, 1.0], [2, 3, 40, 10, 20, 40, 1, 2, 1.0]],
    )
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(imread=lambda _: SimpleNamespace(shape=(100, 100, 3))))
    prepared = prepare_mot17_dataset(
        _config(train_root), tmp_path / "teacher-out", mode="teacher", pseudo_labels={}
    )
    train_label = tmp_path / "teacher-out" / "labels" / "train" / "MOT17-02-FRCNN" / "000001.txt"
    val_label = tmp_path / "teacher-out" / "labels" / "val" / "MOT17-04-FRCNN" / "000001.txt"
    assert train_label.read_text(encoding="utf-8") == ""
    assert (tmp_path / "teacher-out" / "labels" / "train" / "MOT17-02-FRCNN" / "000002.txt").is_file()
    assert val_label.read_text(encoding="utf-8").startswith("0 ")
    assert (tmp_path / "teacher-out" / "labels" / "val" / "MOT17-04-FRCNN" / "000002.txt").read_text(encoding="utf-8") == ""
    assert len(prepared.validation_images) == 2
    assert prepared.stats["train_boxes"] == 0
