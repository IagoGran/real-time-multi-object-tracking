import copy
import json

import pytest

from src.evaluation.benchmark_results import (
    build_benchmark_config,
    combination_completed,
    confidence_threshold_key,
    load_checkpoint,
    new_checkpoint,
    save_checkpoint,
)


def _config(confidence_thresholds=None, max_frames=5):
    return build_benchmark_config(
        models=[{"name": "YOLO", "signature": "yolo.pt"}],
        sequences=[
            {"name": "MOT17-02-FRCNN", "path": "/data/MOT17-02-FRCNN", "frames": 5}
        ],
        confidence_thresholds=confidence_thresholds or [0.1, 0.2],
        match_iou_threshold=0.5,
        max_frames=max_frames,
    )


def test_checkpoint_saves_and_loads_atomically(tmp_path):
    checkpoint_path = tmp_path / "nested" / "benchmark.json"
    checkpoint = new_checkpoint(_config())
    checkpoint["results"] = {
        "YOLO": {
            "MOT17-02-FRCNN": {
                "0.10": {"frames": 5},
                "0.20": {"frames": 5},
            }
        }
    }

    save_checkpoint(checkpoint_path, checkpoint)
    loaded = load_checkpoint(checkpoint_path, _config())

    assert loaded["results"] == checkpoint["results"]
    assert loaded["format_version"] == 2
    assert loaded["config"]["confidence_thresholds"] == [0.1, 0.2]
    assert confidence_threshold_key(0.1) == "0.10"
    assert confidence_threshold_key(0.305) == "0.305"
    assert combination_completed(
        loaded,
        "YOLO",
        "MOT17-02-FRCNN",
    )
    assert not list(checkpoint_path.parent.glob("*.tmp"))


@pytest.mark.parametrize(
    ("field", "requested_value"),
    [
        ("models", [{"name": "YOLO", "signature": "different.pt"}]),
        (
            "sequences",
            [{"name": "MOT17-04-FRCNN", "path": "/other/dataset", "frames": 5}],
        ),
        ("confidence_thresholds", [0.3]),
        ("match_iou_threshold", 0.6),
        ("max_frames", 6),
    ],
)
def test_resume_rejects_incompatible_benchmark_config(
    tmp_path,
    field,
    requested_value,
):
    checkpoint_path = tmp_path / "benchmark.json"
    save_checkpoint(checkpoint_path, new_checkpoint(_config()))
    requested_config = copy.deepcopy(_config())
    requested_config[field] = requested_value

    with pytest.raises(RuntimeError, match="configuration is incompatible"):
        load_checkpoint(checkpoint_path, requested_config)


def test_resume_rejects_unsupported_format_version(tmp_path):
    checkpoint_path = tmp_path / "benchmark.json"
    checkpoint_path.write_text(
        json.dumps({"format_version": 999, "config": _config(), "results": {}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="format version"):
        load_checkpoint(checkpoint_path, _config())


def test_version_one_checkpoint_is_explicitly_incompatible(tmp_path):
    checkpoint_path = tmp_path / "benchmark.json"
    checkpoint_path.write_text(json.dumps({"format_version": 1}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="version 1 is incompatible"):
        load_checkpoint(checkpoint_path, _config())


def test_combination_requires_all_configured_thresholds():
    checkpoint = new_checkpoint(_config())
    checkpoint["results"] = {"YOLO": {"MOT17-02-FRCNN": {"0.10": {}}}}
    assert not combination_completed(checkpoint, "YOLO", "MOT17-02-FRCNN")
    checkpoint["results"]["YOLO"]["MOT17-02-FRCNN"]["0.20"] = {}
    assert combination_completed(checkpoint, "YOLO", "MOT17-02-FRCNN")
