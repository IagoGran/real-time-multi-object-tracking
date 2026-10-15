from dataclasses import dataclass

from src.evaluation.detection_metrics import (
    Box,
    calculate_iou,
)


@dataclass(frozen=True)
class BoxMatch:
    predicted_box: Box
    ground_truth_box: Box
    iou: float


@dataclass
class MatchingResult:
    matches: list[BoxMatch]
    unmatched_predictions: list[Box]
    unmatched_ground_truth: list[Box]


def match_boxes(
    predicted_boxes: list[Box],
    ground_truth_boxes: list[Box],
    iou_threshold: float = 0.5,
) -> MatchingResult:
    """
    Match predicted boxes with GT boxes using IoU.

    Candidate pairs are sorted by IoU from best to worst.
    Each prediction and each GT box can be used only once.

    This is intentionally a simple greedy matcher for our
    detector benchmark. Official benchmark metrics should
    later use the appropriate official evaluator.
    """

    candidates: list[
        tuple[float, int, int]
    ] = []

    for predicted_index, predicted_box in enumerate(
        predicted_boxes
    ):
        for gt_index, gt_box in enumerate(
            ground_truth_boxes
        ):
            iou = calculate_iou(
                predicted_box,
                gt_box,
            )

            if iou >= iou_threshold:
                candidates.append(
                    (
                        iou,
                        predicted_index,
                        gt_index,
                    )
                )

    candidates.sort(
        key=lambda candidate: candidate[0],
        reverse=True,
    )

    matched_predictions: set[int] = set()
    matched_ground_truth: set[int] = set()

    matches: list[BoxMatch] = []

    for (
        iou,
        predicted_index,
        gt_index,
    ) in candidates:

        if (
            predicted_index
            in matched_predictions
        ):
            continue

        if (
            gt_index
            in matched_ground_truth
        ):
            continue

        matches.append(
            BoxMatch(
                predicted_box=predicted_boxes[
                    predicted_index
                ],
                ground_truth_box=ground_truth_boxes[
                    gt_index
                ],
                iou=iou,
            )
        )

        matched_predictions.add(
            predicted_index
        )

        matched_ground_truth.add(
            gt_index
        )

    unmatched_predictions = [
        predicted_box
        for index, predicted_box
        in enumerate(predicted_boxes)
        if index not in matched_predictions
    ]

    unmatched_ground_truth = [
        gt_box
        for index, gt_box
        in enumerate(ground_truth_boxes)
        if index not in matched_ground_truth
    ]

    return MatchingResult(
        matches=matches,
        unmatched_predictions=(
            unmatched_predictions
        ),
        unmatched_ground_truth=(
            unmatched_ground_truth
        ),
    )