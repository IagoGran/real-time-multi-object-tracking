import math

import numpy as np
import pytest

from src.detection.detector import Detector
from src.evaluation.benchmark import (
    ModelSpec,
    _update_checkpoint_aggregates,
    aggregate_sequence_results,
    run_benchmark,
)
from src.evaluation.benchmark_results import new_checkpoint
from src.evaluation.mot17_dataset import Mot17Frame, Mot17Sequence
from src.types import Detection


def _sequence_result(
    frames: int,
    gt_boxes: int,
    matched_boxes: int,
    false_positives: int,
    false_negatives: int,
    inference_seconds: float,
    ious: list[float],
    edge_errors: list[float] | None = None,
) -> dict:
    observations = {
        "iou": ious,
        "edge_error_px": [0.0] * len(ious),
        "edge_error": edge_errors or [0.0] * len(ious),
        "center_error_px": [0.0] * len(ious),
        "center_error": [0.0] * len(ious),
        "size_error": [0.0] * len(ious),
    }
    return {
        "frames": frames,
        "gt_boxes": gt_boxes,
        "matched_boxes": matched_boxes,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "inference_seconds_total": inference_seconds,
        "observations": observations,
    }


def test_aggregation_weights_counts_and_combines_original_observations():
    first = _sequence_result(1, 1, 1, 0, 0, 0.1, [1.0], [0.0])
    second = _sequence_result(
        9,
        9,
        9,
        9,
        0,
        0.9,
        [0.5] * 9,
        [0.5] * 8 + [1.0],
    )

    aggregate = aggregate_sequence_results([first, second])

    assert aggregate["match_rate"] == 1.0
    assert aggregate["mean_iou"] == 0.55
    assert aggregate["median_iou"] == 0.5
    assert math.isclose(aggregate["p90_edge_error"], 0.55)
    assert aggregate["false_positives_per_frame"] == 0.9
    assert aggregate["inference_ms"] == 100.0
    assert aggregate["fps"] == 10.0


def test_aggregates_are_separate_for_each_confidence_threshold():
    checkpoint = {
        "results": {
            "model": {
                "sequence-a": {
                    "0.10": _sequence_result(1, 1, 1, 0, 0, 0.1, [0.2]),
                    "0.20": _sequence_result(1, 1, 1, 0, 0, 0.1, [0.9]),
                },
                "sequence-b": {
                    "0.10": _sequence_result(3, 3, 3, 0, 0, 0.3, [0.4] * 3),
                    "0.20": _sequence_result(3, 3, 3, 0, 0, 0.3, [0.8] * 3),
                },
            }
        }
    }
    _update_checkpoint_aggregates(checkpoint, [0.1, 0.2])
    assert checkpoint["aggregates"]["0.10"]["model"]["mean_iou"] == pytest.approx(0.35)
    assert checkpoint["aggregates"]["0.20"]["model"]["mean_iou"] == pytest.approx(0.825)


def test_benchmark_runs_stub_detector_and_checkpoints_sequence(
    tmp_path,
    monkeypatch,
):
    import src.evaluation.benchmark as benchmark_module

    class StubDetector(Detector):
        def predict(self, frame):
            _ = frame
            return [Detection(0, 0, 10, 10, 0.9, 0, "person")]

    frame_path = tmp_path / "000007.jpg"
    frame_path.touch()
    sequence = Mot17Sequence(
        name="MOT17-TEST-FRCNN",
        path=tmp_path,
        frames=(Mot17Frame(7, frame_path),),
        ground_truth={7: [(0.0, 0.0, 10.0, 10.0)]},
    )
    monkeypatch.setattr(
        benchmark_module.cv2,
        "imread",
        lambda _: np.zeros((12, 12, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        benchmark_module.torch.cuda,
        "is_available",
        lambda: False,
    )
    checkpoint_path = tmp_path / "results.json"
    checkpoint = new_checkpoint(
        {
            "models": [{"name": "stub", "signature": "stub-v1"}],
            "sequences": [{"name": sequence.name, "path": str(tmp_path)}],
            "confidence_thresholds": [0.1, 0.2],
            "match_iou_threshold": 0.5,
            "max_frames": 1,
        }
    )

    result = run_benchmark(
        models=[
            ModelSpec(
                name="stub",
                signature="stub-v1",
                factory=StubDetector,
            )
        ],
        sequences=[sequence],
        confidence_thresholds=[0.1, 0.2],
        match_iou_threshold=0.5,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        progress_every=1,
    )

    sequence_result = checkpoint["results"]["stub"][sequence.name]
    assert checkpoint_path.exists()
    assert sequence_result["0.10"]["frames"] == 1
    assert sequence_result["0.10"]["matched_boxes"] == 1
    assert sequence_result["0.10"]["mean_iou"] == 1.0
    assert result[0.1]["stub"]["mean_iou"] == 1.0
    assert result[0.1]["stub"] == checkpoint["aggregates"]["0.10"]["stub"]
    assert result[0.1]["stub"]["inference_seconds_total"] == result[0.2]["stub"]["inference_seconds_total"]


@pytest.mark.parametrize("frame_count", [1, 10])
def test_sweep_runs_predict_once_per_frame_for_all_thresholds(tmp_path, monkeypatch, frame_count):
    import src.evaluation.benchmark as benchmark_module

    class CountingDetector(Detector):
        calls = 0

        def predict(self, frame):
            type(self).calls += 1
            _ = frame
            return [Detection(0, 0, 10, 10, 0.15, 0, "person"),
                    Detection(0, 0, 10, 10, 0.35, 0, "person"),
                    Detection(0, 0, 10, 10, 0.75, 0, "person")]

    frames = tuple(Mot17Frame(i, tmp_path / f"{i:06}.jpg") for i in range(frame_count))
    sequence = Mot17Sequence("test", tmp_path, frames, {})
    monkeypatch.setattr(benchmark_module.cv2, "imread", lambda _: np.zeros((12, 12, 3), dtype=np.uint8))
    monkeypatch.setattr(benchmark_module.torch.cuda, "is_available", lambda: False)
    thresholds = [0.1, 0.2, 0.4, 0.8, 0.5, 0.6]
    checkpoint = new_checkpoint({"confidence_thresholds": thresholds})

    result = run_benchmark(
        [ModelSpec("count", "fake", CountingDetector)], [sequence], thresholds,
        0.5, checkpoint, tmp_path / "sweep.json", progress_every=100,
    )

    assert CountingDetector.calls == frame_count
    assert [result[t]["count"]["false_positives"] for t in thresholds] == [
        count * frame_count for count in [3, 2, 1, 0, 1, 1]
    ]
