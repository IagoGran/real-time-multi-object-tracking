from dataclasses import dataclass
from math import hypot


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class BoundingBoxMetrics:
    """Geometric quality metrics for one predicted/ground-truth box pair."""

    iou: float

    # Average absolute displacement of the four box edges, in pixels.
    edge_error_px: float

    # Same edge error normalized by GT width/height.
    edge_error: float

    # Distance between box centers.
    center_error_px: float

    # Center distance normalized by GT diagonal.
    center_error: float

    # Mean relative error in width and height.
    size_error: float


def _box_size(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box

    return (
        max(0.0, x2 - x1),
        max(0.0, y2 - y1),
    )


def calculate_iou(
    predicted_box: Box,
    ground_truth_box: Box,
) -> float:
    px1, py1, px2, py2 = predicted_box
    gx1, gy1, gx2, gy2 = ground_truth_box

    intersection_x1 = max(px1, gx1)
    intersection_y1 = max(py1, gy1)
    intersection_x2 = min(px2, gx2)
    intersection_y2 = min(py2, gy2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1,
    )
    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    predicted_width, predicted_height = _box_size(
        predicted_box
    )
    gt_width, gt_height = _box_size(
        ground_truth_box
    )

    predicted_area = (
        predicted_width
        * predicted_height
    )
    gt_area = (
        gt_width
        * gt_height
    )

    union_area = (
        predicted_area
        + gt_area
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def calculate_edge_error_px(
    predicted_box: Box,
    ground_truth_box: Box,
) -> float:
    """
    Mean absolute displacement of the four bounding-box edges.

    Returned in pixels.
    """

    px1, py1, px2, py2 = predicted_box
    gx1, gy1, gx2, gy2 = ground_truth_box

    return (
        abs(px1 - gx1)
        + abs(py1 - gy1)
        + abs(px2 - gx2)
        + abs(py2 - gy2)
    ) / 4.0


def calculate_edge_error(
    predicted_box: Box,
    ground_truth_box: Box,
) -> float:
    """
    Edge displacement normalized by GT dimensions.

    Horizontal edges are normalized by GT width.
    Vertical edges are normalized by GT height.

    A value of 0.05 means that, on average, the box edges
    differ from ground truth by roughly 5% of the relevant
    GT dimension.
    """

    px1, py1, px2, py2 = predicted_box
    gx1, gy1, gx2, gy2 = ground_truth_box

    gt_width, gt_height = _box_size(
        ground_truth_box
    )

    if gt_width <= 0 or gt_height <= 0:
        return 0.0

    left_error = (
        abs(px1 - gx1)
        / gt_width
    )

    right_error = (
        abs(px2 - gx2)
        / gt_width
    )

    top_error = (
        abs(py1 - gy1)
        / gt_height
    )

    bottom_error = (
        abs(py2 - gy2)
        / gt_height
    )

    return (
        left_error
        + right_error
        + top_error
        + bottom_error
    ) / 4.0


def calculate_center_error_px(
    predicted_box: Box,
    ground_truth_box: Box,
) -> float:
    px1, py1, px2, py2 = predicted_box
    gx1, gy1, gx2, gy2 = ground_truth_box

    predicted_center_x = (
        px1 + px2
    ) / 2.0
    predicted_center_y = (
        py1 + py2
    ) / 2.0

    gt_center_x = (
        gx1 + gx2
    ) / 2.0
    gt_center_y = (
        gy1 + gy2
    ) / 2.0

    return hypot(
        predicted_center_x - gt_center_x,
        predicted_center_y - gt_center_y,
    )


def calculate_center_error(
    predicted_box: Box,
    ground_truth_box: Box,
) -> float:
    """
    Distance between box centers normalized by GT diagonal.
    """

    gt_width, gt_height = _box_size(
        ground_truth_box
    )

    diagonal = hypot(
        gt_width,
        gt_height,
    )

    if diagonal <= 0:
        return 0.0

    return (
        calculate_center_error_px(
            predicted_box,
            ground_truth_box,
        )
        / diagonal
    )


def calculate_size_error(
    predicted_box: Box,
    ground_truth_box: Box,
) -> float:
    """
    Mean relative error in bounding-box width and height.
    """

    predicted_width, predicted_height = _box_size(
        predicted_box
    )

    gt_width, gt_height = _box_size(
        ground_truth_box
    )

    if gt_width <= 0 or gt_height <= 0:
        return 0.0

    width_error = (
        abs(
            predicted_width
            - gt_width
        )
        / gt_width
    )

    height_error = (
        abs(
            predicted_height
            - gt_height
        )
        / gt_height
    )

    return (
        width_error
        + height_error
    ) / 2.0


def evaluate_bounding_box(
    predicted_box: Box,
    ground_truth_box: Box,
) -> BoundingBoxMetrics:
    return BoundingBoxMetrics(
        iou=calculate_iou(
            predicted_box,
            ground_truth_box,
        ),
        edge_error_px=calculate_edge_error_px(
            predicted_box,
            ground_truth_box,
        ),
        edge_error=calculate_edge_error(
            predicted_box,
            ground_truth_box,
        ),
        center_error_px=calculate_center_error_px(
            predicted_box,
            ground_truth_box,
        ),
        center_error=calculate_center_error(
            predicted_box,
            ground_truth_box,
        ),
        size_error=calculate_size_error(
            predicted_box,
            ground_truth_box,
        ),
    )