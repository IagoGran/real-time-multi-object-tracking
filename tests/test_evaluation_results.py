import json

from src.evaluation.evaluation_results import (
    PUBLIC_METRIC_KEYS,
    build_evaluation_result,
    save_evaluation_result,
)


def _metrics(value):
    return {key: value for key in PUBLIC_METRIC_KEYS}


def test_final_evaluation_json_supports_thresholds_and_hides_checkpoint_fields(tmp_path):
    result = build_evaluation_result(
        evaluation_id="exp-1",
        dataset={"name": "MOT17", "sequences": ["MOT17-04-FRCNN"]},
        models={"student": {"weights_path": "best.pt", "signature": "best.pt", "weights_sha256": "abc123"}},
        thresholds=[0.1, 0.3],
        match_iou_threshold=0.5,
        aggregate_metrics={0.1: {"student": _metrics(0.1)}, 0.3: {"student": _metrics(0.3)}},
        per_sequence_metrics={
            0.1: {"student": {"MOT17-04-FRCNN": _metrics(0.1)}},
            0.3: {"student": {"MOT17-04-FRCNN": _metrics(0.3)}},
        },
        started_at="start",
        completed_at="end",
        evaluation_name="final_comparison",
        git_commit="abc123",
        split={"name": "fixture"},
        partition="validation",
        inference_imgsz=640,
    )
    path = tmp_path / "evaluation.json"
    save_evaluation_result(path, result)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert set(loaded["results"]) == {"0.10", "0.30"}
    assert loaded["results"]["0.30"]["student"]["mean_iou"] == 0.3
    assert loaded["per_sequence"]["0.10"]["student"]["MOT17-04-FRCNN"]["fps"] == 0.1
    assert "observations" not in loaded["results"]["0.10"]["student"]
    assert loaded["evaluation_name"] == "final_comparison"
    assert loaded["inference_imgsz"] == 640
    assert loaded["yolo_inference_protocol"] == {"confidence_floor": 0.01, "nms": None, "iou": 0.7, "max_det": 300}
    assert loaded["partition"] == "validation"
    assert loaded["git_commit"] == "abc123"
    assert loaded["models"]["student"]["weights_sha256"] == "abc123"
    assert loaded["metric_protocol"]["ignored_regions_applied"] is False
    assert loaded["metric_protocol"]["equivalent_to_official_mot17_metrics"] is False
