import csv
import json
import sys
from types import SimpleNamespace

from src.training.inspect_dataset import audit_dataset


def test_audit_counts_dropped_frames_classes_and_lost_pedestrians(tmp_path, monkeypatch):
    root = tmp_path / "MOT17" / "train"
    for sequence in ("MOT17-02-FRCNN", "MOT17-04-FRCNN"):
        (root / sequence / "img1").mkdir(parents=True)
        (root / sequence / "gt").mkdir()
        for frame in (1, 2):
            (root / sequence / "img1" / f"{frame:06d}.jpg").touch()
        rows = [
            [1, 1, 10, 10, 20, 40, 1, 1, 0.5],
            [1, 2, 10, 10, 20, 40, 1, 2, 1.0],
            [2, 3, 10, 10, 20, 40, 1, 1, 0.8],
        ]
        with (root / sequence / "gt" / "gt.txt").open("w", newline="") as f:
            csv.writer(f).writerows(rows)
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs" / "split.json"
    config.write_text(json.dumps({
        "format_version": 1, "name": "audit", "dataset_root": "MOT17/train",
        "prepared_dataset_root": "prepared", "variant": "FRCNN",
        "train_sequences": ["MOT17-02-FRCNN"], "validation_sequences": ["MOT17-04-FRCNN"],
        "ignored_frame_policy": "skip_frames_with_excluded_annotations",
    }), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(imread=lambda _: SimpleNamespace(shape=(100, 200, 3))))
    report = audit_dataset(config)
    row = report["sequences"]["MOT17-02-FRCNN"]
    assert row["frames_dropped"] == 1
    assert row["frames_dropped_reasons"] == {"class_2": 1}
    assert row["valid_pedestrian_gt_lost_on_dropped_frames"] == 1
    assert row["annotation_counts_by_mot_class"] == {"1": 2, "2": 1}
    assert report["train_dropped_frames"]["percent_of_train_frames"] == 50.0
