import json

from src.training.training_results import (
    make_experiment_id,
    new_experiment_manifest,
    save_experiment_manifest,
)
from src.training.train_detector import _record_training_outcome


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


def test_experiment_ids_are_unique_and_directories_do_not_overwrite(tmp_path):
    first = make_experiment_id("supervised", "yolo26n.pt")
    second = make_experiment_id("supervised", "yolo26n.pt")
    assert first != second
    (tmp_path / first).mkdir()
    (tmp_path / second).mkdir(exist_ok=False)


def test_labeled_experiment_ids_keep_unique_suffix_and_do_not_overwrite(tmp_path):
    first = make_experiment_id("supervised", "yolo26n.pt", experiment_label="low-lr-640")
    second = make_experiment_id("supervised", "yolo26n.pt", experiment_label="low-lr-640")
    assert "low-lr-640" in first and first != second
    (tmp_path / first).mkdir()
    (tmp_path / second).mkdir(exist_ok=False)


def test_training_manifest_outcome_uses_best_map_and_detects_early_stop(tmp_path):
    history = tmp_path / "results.csv"
    history.write_text(
        "epoch,metrics/mAP50-95(B)\n1,0.3\n2,0.4\n3,0.35\n",
        encoding="utf-8",
    )
    manifest = {"training": {}}
    _record_training_outcome(manifest, history, requested_epochs=50)
    assert manifest["training"]["best_epoch"] == 2
    assert manifest["training"]["best_checkpoint_selection_metric"] == "metrics/mAP50-95(B)"
    assert manifest["training"]["best_checkpoint_selection_value"] == 0.4
    assert manifest["training"]["finish_reason"] == "early_stopping"
