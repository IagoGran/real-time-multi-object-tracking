import csv
import json
import sys
from types import SimpleNamespace

from src.training.inspect_dataset import audit_dataset, assert_preparation_matches_audit
from src.training.mot17_training_dataset import prepare_mot17_dataset
from src.training.training_config import load_training_config


def test_audit_distinguishes_retained_frames_target_gt_and_excluded_classes(tmp_path, monkeypatch):
    root = tmp_path / "MOT17" / "train"
    for sequence in ("MOT17-02-FRCNN", "MOT17-04-FRCNN"):
        (root / sequence / "img1").mkdir(parents=True)
        (root / sequence / "gt").mkdir()
        for frame in (1, 2):
            (root / sequence / "img1" / f"{frame:06d}.jpg").touch()
        rows = [
            [1, 1, 10, 10, 20, 40, 1, 1, 0.5],
            [1, 2, 30, 10, 20, 40, 1, 2, 1.0],
            [2, 3, 40, 10, 20, 40, 1, 2, 1.0],
        ]
        with (root / sequence / "gt" / "gt.txt").open("w", newline="") as f:
            csv.writer(f).writerows(rows)
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs" / "split.json"
    config.write_text(json.dumps({
        "format_version": 2, "name": "audit", "dataset_root": "MOT17/train",
        "prepared_dataset_root": "prepared", "variant": "FRCNN",
        "train_sequences": ["MOT17-02-FRCNN"], "validation_sequences": ["MOT17-04-FRCNN"],
        "preparation_policy": "all_frames_target_class_only_v2",
    }), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(imread=lambda _: SimpleNamespace(shape=(100, 200, 3))))

    report = audit_dataset(config)
    row = report["sequences"]["MOT17-02-FRCNN"]
    assert row["frames_original"] == row["frames_retained"] == 2
    assert row["frames_with_target_gt"] == 1
    assert row["frames_without_target_gt"] == 1
    assert row["target_gt_annotations_raw"] == row["target_gt_annotations_valid"] == 1
    assert row["target_gt_annotations_lost"] == 0
    assert row["frames_with_excluded_annotations_but_retained"] == 2
    assert row["excluded_annotations_by_class"] == {"class_2": 2}
    assert report["totals"]["train"]["frames_retained"] == 2
    prepared = prepare_mot17_dataset(load_training_config(config), tmp_path / "prepared")
    prepared_manifest = json.loads((prepared.root / "dataset-manifest.json").read_text(encoding="utf-8"))
    assert_preparation_matches_audit(report, prepared_manifest)
