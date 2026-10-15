import math

from src.evaluation.detection_metrics import (
    calculate_center_error,
    calculate_center_error_px,
    calculate_edge_error,
    calculate_edge_error_px,
    calculate_iou,
    calculate_size_error,
    evaluate_bounding_box,
)


def test_identical_boxes_have_perfect_iou_and_zero_errors():
    box = (1.0, 2.0, 11.0, 22.0)

    assert calculate_iou(box, box) == 1.0
    assert calculate_edge_error_px(box, box) == 0.0
    assert calculate_edge_error(box, box) == 0.0
    assert calculate_center_error_px(box, box) == 0.0
    assert calculate_center_error(box, box) == 0.0
    assert calculate_size_error(box, box) == 0.0


def test_edge_and_center_errors_use_documented_normalizations():
    predicted = (1.0, 2.0, 11.0, 22.0)
    ground_truth = (0.0, 0.0, 10.0, 20.0)

    assert calculate_edge_error_px(predicted, ground_truth) == 1.5
    assert math.isclose(calculate_edge_error(predicted, ground_truth), 0.1)
    assert math.isclose(
        calculate_center_error_px(predicted, ground_truth),
        math.sqrt(5),
    )
    assert math.isclose(calculate_center_error(predicted, ground_truth), 0.1)


def test_size_error_is_mean_relative_width_and_height_error():
    predicted = (0.0, 0.0, 12.0, 18.0)
    ground_truth = (0.0, 0.0, 10.0, 20.0)

    assert math.isclose(calculate_size_error(predicted, ground_truth), 0.15)


def test_evaluate_bounding_box_returns_all_metrics():
    metrics = evaluate_bounding_box((0, 0, 10, 10), (0, 0, 10, 10))

    assert metrics.iou == 1.0
    assert metrics.edge_error_px == 0.0
    assert metrics.edge_error == 0.0
    assert metrics.center_error_px == 0.0
    assert metrics.center_error == 0.0
    assert metrics.size_error == 0.0
