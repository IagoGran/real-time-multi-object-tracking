import json

import pytest

from src.training.train_detector import (
    _parse_args,
    _training_parameters,
    _ultralytics_training_kwargs,
    _validate_freeze_compatibility,
    _verify_freeze_state,
    _verify_frozen_parameters_unchanged,
)
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
    assert args.multi_scale == 0.0
    assert args.freeze is None


def test_multi_scale_parsing_and_range_validation():
    args = _parse_args(["--multi-scale", "0.25"])
    assert args.multi_scale == 0.25
    with pytest.raises(SystemExit):
        _parse_args(["--multi-scale", "-0.1"])
    with pytest.raises(SystemExit):
        _parse_args(["--imgsz", "64", "--multi-scale", "0.5"])


def test_multi_scale_is_recorded_and_passed_to_ultralytics(tmp_path):
    args = _parse_args(["--multi-scale", "0.25", "--experiment-label", "960-ms"])
    data_yaml = tmp_path / "data.yaml"
    recorded = _training_parameters(args, data_yaml)
    propagated = _ultralytics_training_kwargs(args, data_yaml, tmp_path, "run-id")
    assert recorded["multi_scale"] == 0.25
    assert recorded["experiment_label"] == "960-ms"
    assert propagated["multi_scale"] == 0.25


def test_freeze_defaults_are_omitted_and_freeze_zero_is_propagated(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    default_args = _parse_args([])
    assert "freeze" not in _ultralytics_training_kwargs(default_args, data_yaml, tmp_path, "default")
    args = _parse_args(["--freeze", "0"])
    assert _training_parameters(args, data_yaml)["freeze"] == 0
    assert _ultralytics_training_kwargs(args, data_yaml, tmp_path, "unfrozen")["freeze"] == 0
    args = _parse_args(["--freeze", "11"])
    assert _training_parameters(args, data_yaml)["freeze"] == 11
    assert _ultralytics_training_kwargs(args, data_yaml, tmp_path, "frozen")["freeze"] == 11


def test_freeze_11_requires_eleven_declared_backbone_layers():
    class Model:
        model = [object() for _ in range(12)]
        yaml = {"backbone": [object() for _ in range(11)]}

    assert _validate_freeze_compatibility(Model(), 11)["backbone_layer_indices"] == list(range(11))
    Model.yaml = {"backbone": [object() for _ in range(10)]}
    with pytest.raises(ValueError, match="complete YOLO26n backbone"):
        _validate_freeze_compatibility(Model(), 11)
    with pytest.raises(ValueError, match="exceeds"):
        _validate_freeze_compatibility(Model(), 13)


def test_freeze_rejects_negative_values():
    with pytest.raises(SystemExit):
        _parse_args(["--freeze", "-1"])


def test_post_setup_freeze_verification_records_parameters_and_optimizer(tmp_path):
    import torch

    class Layer:
        def __init__(self, parameter=None):
            self.parameter = parameter

        def parameters(self):
            return [] if self.parameter is None else [self.parameter]

    frozen = torch.nn.Parameter(torch.ones(4), requires_grad=False)
    trainable = torch.nn.Parameter(torch.ones(6), requires_grad=True)

    class Model:
        model = [Layer(frozen), Layer(), Layer(trainable)]
        yaml = {"backbone": ["stem"]}

        def named_parameters(self):
            return [("model.0.weight", frozen), ("model.2.weight", trainable)]

    class Trainer:
        model = Model()
        optimizer = type("Optimizer", (), {"param_groups": [{"params": [frozen, trainable]}]})()

    manifest = {"training": {}}
    manifest_path = tmp_path / "experiment.json"
    trainer = Trainer()
    _verify_freeze_state(trainer, 1, manifest, manifest_path)
    _verify_frozen_parameters_unchanged(trainer, manifest, manifest_path)
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))["training"]["freeze_verification"]
    assert recorded["frozen_layer_indices"] == [0]
    assert recorded["parameters"] == {"total": 10, "frozen": 4, "trainable": 6}
    assert recorded["optimizer_includes_frozen_parameters"] == 1
    assert recorded["parameters_unchanged_after_epoch"] is True
    assert recorded["verified_epochs"] == 1
