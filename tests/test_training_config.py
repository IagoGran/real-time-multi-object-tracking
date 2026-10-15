import json

import pytest

from src.training.train_detector import _parse_args
from src.training.training_config import (
    DEFAULT_SPLIT_CONFIG,
    load_training_config,
    validate_sequence_split,
)


def test_default_split_is_explicit_and_has_no_sequence_overlap():
    config = load_training_config(DEFAULT_SPLIT_CONFIG)
    assert set(config.train_sequences).isdisjoint(config.validation_sequences)
    assert len(config.train_sequences) == 5
    assert len(config.validation_sequences) == 2
    assert all(sequence.endswith("-FRCNN") for sequence in config.train_sequences + config.validation_sequences)


def test_split_config_rejects_overlap(tmp_path):
    raw = {
        "format_version": 2,
        "name": "bad",
        "dataset_root": "data",
        "prepared_dataset_root": "prepared",
        "variant": "FRCNN",
        "train_sequences": ["MOT17-02-FRCNN"],
        "validation_sequences": ["MOT17-02-FRCNN"],
        "preparation_policy": "all_frames_target_class_only_v2",
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        load_training_config(path)


def test_split_rejects_two_variants_of_the_same_scene():
    with pytest.raises(ValueError, match="exactly one MOT17 variant per scene"):
        validate_sequence_split(
            ["MOT17-02-FRCNN", "MOT17-02-ALT-FRCNN"],
            ["MOT17-04-FRCNN"],
        )


def test_teacher_cli_records_confidence_and_requires_teacher_weights():
    with pytest.raises(SystemExit):
        _parse_args(["--mode", "teacher"])
    args = _parse_args(
        ["--mode", "teacher", "--teacher-weights", "teacher.pt", "--teacher-confidence", "0.42"]
    )
    assert args.teacher_weights == "teacher.pt"
    assert args.teacher_confidence == 0.42


def test_supervised_cli_has_configurable_training_defaults():
    args = _parse_args(["--mode", "supervised"])
    assert args.student_weights == "yolo26n.pt"
    assert args.epochs == 50
    assert args.batch == 16
    assert args.imgsz == 640
