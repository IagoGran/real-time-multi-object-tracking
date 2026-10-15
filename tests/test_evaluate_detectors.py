import sys
from pathlib import Path

from src.evaluation.benchmark_results import build_benchmark_config
from src.evaluation.evaluate_detectors import (
    DEFAULT_CONFIDENCE_THRESHOLDS,
    _parse_args,
    confidence_thresholds,
    format_results_table,
    load_or_create_checkpoint,
    results_path_for_thresholds,
)


def test_confidence_thresholds_default_to_sweep(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate_detectors"])

    thresholds = confidence_thresholds(_parse_args())

    assert thresholds == [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    assert thresholds == DEFAULT_CONFIDENCE_THRESHOLDS


def test_confidence_threshold_argument_selects_single_threshold(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_detectors", "--confidence-threshold", "0.3"],
    )

    assert confidence_thresholds(_parse_args()) == [0.3]


def test_sweep_paths_include_formatted_threshold_and_single_keeps_base():
    base_path = Path("results/detector_benchmark.json")

    assert results_path_for_thresholds(base_path, [0.1, 0.2], 0.1) == Path(
        "results/detector_benchmark_conf_0p10.json"
    )
    assert results_path_for_thresholds(base_path, [0.3], 0.3) == base_path


def test_resume_creates_missing_threshold_checkpoint_and_reuses_existing(
    tmp_path,
):
    checkpoint_path = tmp_path / "detector_benchmark_conf_0p30.json"
    config = build_benchmark_config(
        models=[{"name": "stub", "signature": "stub"}],
        sequences=[{"name": "MOT17-test", "path": "/test", "frames": 1}],
        confidence_threshold=0.3,
        match_iou_threshold=0.5,
        max_frames=1,
    )

    created = load_or_create_checkpoint(checkpoint_path, config, resume=True)
    checkpoint_path.write_text(
        checkpoint_path.read_text(encoding="utf-8").replace(
            '"status": "pending"',
            '"status": "running"',
        ),
        encoding="utf-8",
    )
    loaded = load_or_create_checkpoint(checkpoint_path, config, resume=True)

    assert created["status"] == "pending"
    assert loaded["status"] == "running"


def test_results_table_formats_simulated_metrics():
    table = format_results_table(
        {
            0.1: {
                "YOLO26n": {
                    "match_rate": 0.6002,
                    "mean_iou": 0.790,
                    "mean_edge_error": 0.0597,
                    "p90_edge_error": 0.1174,
                    "mean_edge_error_px": 7.69,
                    "mean_center_error": 0.0431,
                    "mean_size_error": 0.0939,
                    "false_positives_per_frame": 4.41,
                    "inference_ms": 19.9,
                    "fps": 50.35,
                }
            }
        }
    )

    assert "Conf" in table and "Model" in table
    assert "0.10 YOLO26n" in table
    assert "60.02%" in table
    assert "0.790" in table
    assert "FP/fr" in table
    assert "P90%" in table
