from src.evaluation.matching import match_boxes


def test_matching_is_one_to_one():
    predictions = [
        (0.0, 0.0, 10.0, 10.0),
        (0.0, 0.0, 10.0, 10.0),
    ]
    ground_truth = [
        (0.0, 0.0, 10.0, 10.0),
        (0.0, 0.0, 10.0, 10.0),
    ]

    result = match_boxes(predictions, ground_truth, iou_threshold=0.5)

    assert len(result.matches) == 2
    assert result.unmatched_predictions == []
    assert result.unmatched_ground_truth == []


def test_matching_reports_unmatched_predictions_and_ground_truth():
    predictions = [
        (0.0, 0.0, 10.0, 10.0),
        (30.0, 30.0, 40.0, 40.0),
    ]
    ground_truth = [
        (0.0, 0.0, 10.0, 10.0),
        (60.0, 60.0, 70.0, 70.0),
    ]

    result = match_boxes(predictions, ground_truth, iou_threshold=0.5)

    assert len(result.matches) == 1
    assert result.unmatched_predictions == [predictions[1]]
    assert result.unmatched_ground_truth == [ground_truth[1]]


def test_matching_threshold_is_inclusive():
    result = match_boxes(
        [(0.0, 0.0, 10.0, 10.0)],
        [(0.0, 0.0, 10.0, 10.0)],
        iou_threshold=1.0,
    )

    assert len(result.matches) == 1
