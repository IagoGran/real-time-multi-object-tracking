from src.evaluation.matching import match_boxes
from src.evaluation.mot17_dataset import load_ground_truth


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




def test_unmatched_prediction_overlapping_distractor_is_still_counted_as_fp(tmp_path):
    gt_path = tmp_path / "gt.txt"
    gt_path.write_text(
        "1,1,10,10,20,40,1,1,1.0\n"
        "1,2,50,10,20,40,1,2,1.0\n",
        encoding="utf-8",
    )
    target_gt = load_ground_truth(gt_path)[1]
    prediction_over_distractor = (50, 10, 70, 50)
    result = match_boxes([prediction_over_distractor], target_gt, iou_threshold=0.5)
    assert result.unmatched_predictions == [prediction_over_distractor]
