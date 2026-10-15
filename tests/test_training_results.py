import json

from src.training.training_results import (
    make_experiment_id,
    new_experiment_manifest,
    save_experiment_manifest,
)


def test_experiment_name_and_supervised_manifest(tmp_path):
    experiment_id = make_experiment_id("supervised", "yolo26n.pt")
    manifest = new_experiment_manifest(
        experiment_id,
        "supervised",
        "yolo26n.pt",
        None,
        {"name": "MOT17", "train_sequences": ["02"], "validation_sequences": ["04"]},
        {"epochs": 2, "seed": 3},
    )
    path = tmp_path / experiment_id / "experiment.json"
    save_experiment_manifest(path, manifest)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert "yolo26n_supervised_" in experiment_id
    assert loaded["mode"] == "supervised"
    assert loaded["teacher"] is None
    assert loaded["training"]["seed"] == 3


def test_teacher_manifest_records_threshold_and_weights():
    manifest = new_experiment_manifest(
        "student_teacher_test",
        "teacher",
        "yolo26n.pt",
        {"weights": "yolo26m.pt", "confidence_threshold": 0.4},
        {"name": "MOT17"},
        {"epochs": 2},
    )
    assert manifest["teacher"]["weights"] == "yolo26m.pt"
    assert manifest["teacher"]["confidence_threshold"] == 0.4


def test_teacher_experiment_name_contains_teacher():
    assert "teacher_yolo26m" in make_experiment_id("teacher", "yolo26n.pt", "yolo26m.pt")
