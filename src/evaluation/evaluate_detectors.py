import argparse
from pathlib import Path
import logging
from typing import Callable

import torch

from src.detection.detector import Detector
from src.detection.faster_rcnn_detector import FasterRCNNDetector
from src.detection.yolo_detector import YoloDetector
from src.evaluation.benchmark import (
    ModelSpec,
    estimate_runtime,
    run_benchmark,
    sequence_configuration,
)
from src.evaluation.benchmark_results import (
    build_benchmark_config,
    load_checkpoint,
    new_checkpoint,
    save_checkpoint,
)
from src.evaluation.logging_config import configure_logging
from src.evaluation.mot17_dataset import (
    DEFAULT_SEQUENCE_PATH,
    DEFAULT_TRAIN_PATH,
    load_sequences,
)


CLASS_NAMES = ["person"]

DEFAULT_RESULTS_PATH = Path(
    "results/detector_benchmark.json"
)

DEFAULT_LOG_PATH = Path(
    "results/detector_benchmark.log"
)

DEFAULT_CONFIDENCE_THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
]

LOGGER = logging.getLogger(__name__)


MODELS_TO_COMPARE: list[
    tuple[
        str,
        str,
        Callable[[], Detector],
    ]
] = [
    (
        "YOLO26n",
        "yolo26n.pt",
        lambda: YoloDetector(
            "yolo26n.pt",
            CLASS_NAMES,
        ),
    ),
    (
        "YOLO26s",
        "yolo26s.pt",
        lambda: YoloDetector(
            "yolo26s.pt",
            CLASS_NAMES,
        ),
    ),
    (
        "YOLO26m",
        "yolo26m.pt",
        lambda: YoloDetector(
            "yolo26m.pt",
            CLASS_NAMES,
        ),
    ),
    (
        "Faster R-CNN",
        "torchvision:fasterrcnn_resnet50_fpn:DEFAULT",
        lambda: FasterRCNNDetector(
            CLASS_NAMES
        ),
    ),
]


def _models_to_compare() -> list[ModelSpec]:
    return [
        ModelSpec(
            name=name,
            signature=signature,
            factory=factory,
        )
        for (
            name,
            signature,
            factory,
        ) in MODELS_TO_COMPARE
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate detector localization "
            "quality on MOT17."
        )
    )

    parser.add_argument(
        "--sequence",
        type=Path,
        default=DEFAULT_SEQUENCE_PATH,
        help="Single MOT17 sequence to evaluate.",
    )

    parser.add_argument(
        "--all-train",
        action="store_true",
        help=(
            "Evaluate one FRCNN variant per "
            "MOT17 train scene."
        ),
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help=(
            "Run only this confidence threshold. "
            "When omitted, run the default "
            "threshold sweep."
        ),
    )

    parser.add_argument(
        "--match-iou-threshold",
        type=float,
        default=0.5,
        help=(
            "Minimum IoU for a prediction/"
            "ground-truth match."
        ),
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        help=(
            "Limit the number of frames "
            "per sequence."
        ),
    )

    parser.add_argument(
        "--skip-estimate",
        action="store_true",
        help=(
            "Skip the per-sequence runtime "
            "sampling phase."
        ),
    )

    parser.add_argument(
        "--estimate-samples",
        type=int,
        default=3,
        help=(
            "Frames sampled per sequence/model "
            "for runtime estimation."
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Log progress every N frames.",
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Checkpoint/result JSON path.",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Log file path.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from a compatible "
            "result JSON."
        ),
    )

    args = parser.parse_args()

    if (
        args.max_frames is not None
        and args.max_frames <= 0
    ):
        parser.error(
            "--max-frames must be greater than zero."
        )

    if (
        args.confidence_threshold is not None
        and not (
            0.0
            <= args.confidence_threshold
            <= 1.0
        )
    ):
        parser.error(
            "--confidence-threshold must be "
            "between 0 and 1."
        )

    if not (
        0.0
        <= args.match_iou_threshold
        <= 1.0
    ):
        parser.error(
            "--match-iou-threshold must be "
            "between 0 and 1."
        )

    if args.estimate_samples <= 0:
        parser.error(
            "--estimate-samples must be "
            "greater than zero."
        )

    if args.progress_every <= 0:
        parser.error(
            "--progress-every must be "
            "greater than zero."
        )

    if (
        args.all_train
        and args.sequence != DEFAULT_SEQUENCE_PATH
    ):
        parser.error(
            "--sequence cannot be combined "
            "with --all-train."
        )

    return args


def confidence_thresholds(
    args: argparse.Namespace,
) -> list[float]:

    if args.confidence_threshold is not None:
        return [
            args.confidence_threshold
        ]

    return (
        DEFAULT_CONFIDENCE_THRESHOLDS.copy()
    )


def _log_configuration(
    models: list[ModelSpec],
    sequences,
    thresholds: list[float],
    match_iou_threshold: float,
    max_frames: int | None,
) -> None:

    LOGGER.info(
        "Starting detector benchmark"
    )

    LOGGER.info(
        "Confidence thresholds: %s",
        ", ".join(
            f"{threshold:.2f}"
            for threshold in thresholds
        ),
    )

    LOGGER.info(
        "Matching IoU threshold: %.3f",
        match_iou_threshold,
    )

    LOGGER.info(
        "Maximum frames per sequence: %s",
        max_frames or "all",
    )

    LOGGER.info("Models:")

    for model in models:
        LOGGER.info(
            "  %s (%s)",
            model.name,
            model.signature,
        )

    LOGGER.info("Sequences:")

    for sequence in sequences:
        LOGGER.info(
            "  %s: %d frames",
            sequence.name,
            len(sequence.frames),
        )


def load_or_create_checkpoint(
    checkpoint_path: Path,
    config: dict[str, object],
    resume: bool,
) -> dict[str, object]:

    if resume and checkpoint_path.exists():
        return load_checkpoint(
            checkpoint_path,
            config,
        )

    checkpoint = new_checkpoint(
        config
    )

    save_checkpoint(
        checkpoint_path,
        checkpoint,
    )

    return checkpoint


def format_results_table(
    results_by_threshold: dict[
        float,
        dict[
            str,
            dict[str, object],
        ],
    ],
) -> str:

    width = 121

    lines = [
        "=" * width,
        (
            f"{'Conf':>6} "
            f"{'Model':<16} "
            f"{'Match%':>9} "
            f"{'IoU':>8} "
            f"{'Edge%':>9} "
            f"{'P90%':>8} "
            f"{'Edge px':>9} "
            f"{'Center%':>9} "
            f"{'Size%':>8} "
            f"{'FP/fr':>8} "
            f"{'ms':>9} "
            f"{'FPS':>9}"
        ),
        "=" * width,
    ]

    threshold_items = sorted(
        results_by_threshold.items()
    )

    for index, (
        threshold,
        model_results,
    ) in enumerate(
        threshold_items
    ):

        for (
            model_name,
            metrics,
        ) in model_results.items():

            lines.append(
                f"{threshold:>6.2f} "
                f"{model_name:<16} "
                f"{float(metrics['match_rate']) * 100:>8.2f}% "
                f"{float(metrics['mean_iou']):>8.3f} "
                f"{float(metrics['mean_edge_error']) * 100:>8.2f}% "
                f"{float(metrics['p90_edge_error']) * 100:>7.2f}% "
                f"{float(metrics['mean_edge_error_px']):>9.2f} "
                f"{float(metrics['mean_center_error']) * 100:>8.2f}% "
                f"{float(metrics['mean_size_error']) * 100:>7.2f}% "
                f"{float(metrics['false_positives_per_frame']):>8.2f} "
                f"{float(metrics['inference_ms']):>9.1f} "
                f"{float(metrics['fps']):>9.2f}"
            )

        if (
            index
            < len(threshold_items) - 1
        ):
            lines.append(
                "-" * width
            )

    lines.extend(
        [
            "=" * width,
            (
                "Match% = matched GT pedestrians; "
                "IoU = mean intersection over union"
            ),
            (
                "Edge% = mean normalized edge error; "
                "P90% = 90th percentile edge error"
            ),
            (
                "Center% = center error normalized "
                "by GT diagonal; "
                "Size% = relative size error"
            ),
            (
                "FP/fr = unmatched predictions "
                "per processed frame"
            ),
        ]
    )

    return "\n".join(
        lines
    )


def main() -> None:

    args = _parse_args()

    configure_logging(
        args.log_file
    )

    try:
        sequences = load_sequences(
            sequence_path=args.sequence,
            all_train=args.all_train,
            max_frames=args.max_frames,
            train_path=DEFAULT_TRAIN_PATH,
        )

        models = (
            _models_to_compare()
        )

        thresholds = (
            confidence_thresholds(
                args
            )
        )

        _log_configuration(
            models=models,
            sequences=sequences,
            thresholds=thresholds,
            match_iou_threshold=(
                args.match_iou_threshold
            ),
            max_frames=args.max_frames,
        )

        model_config = [
            {
                "name": model.name,
                "signature": model.signature,
            }
            for model in models
        ]

        # -----------------------------------------------------------
        # ONE benchmark configuration for the complete sweep.
        # -----------------------------------------------------------

        config = build_benchmark_config(
            models=model_config,
            sequences=(
                sequence_configuration(
                    sequences
                )
            ),
            confidence_thresholds=(
                thresholds
            ),
            match_iou_threshold=(
                args.match_iou_threshold
            ),
            max_frames=args.max_frames,
        )

        checkpoint = (
            load_or_create_checkpoint(
                checkpoint_path=(
                    args.results
                ),
                config=config,
                resume=args.resume,
            )
        )

        if args.resume:
            LOGGER.info(
                "Using checkpoint: %s",
                args.results,
            )

        # -----------------------------------------------------------
        # Runtime estimation.
        #
        # This estimates model inference ONCE,
        # not once for every confidence threshold.
        # -----------------------------------------------------------

        if not args.skip_estimate:
            estimate_runtime(
                models=models,
                sequences=sequences,
                confidence_thresholds=thresholds,
                match_iou_threshold=args.match_iou_threshold,
                sample_count=(
                    args.estimate_samples
                ),
            )

        # -----------------------------------------------------------
        # ONE inference pass.
        #
        # run_benchmark performs:
        #
        # frame
        #   -> detector.predict()
        #   -> predictions
        #       -> threshold 0.10
        #       -> threshold 0.20
        #       -> ...
        #
        # -----------------------------------------------------------

        results_by_threshold = (
            run_benchmark(
                models=models,
                sequences=sequences,
                confidence_thresholds=(
                    thresholds
                ),
                match_iou_threshold=(
                    args.match_iou_threshold
                ),
                checkpoint=checkpoint,
                checkpoint_path=(
                    args.results
                ),
                progress_every=(
                    args.progress_every
                ),
            )
        )

        print()
        print(
            format_results_table(
                results_by_threshold
            )
        )

    except Exception:
        LOGGER.exception(
            "Detector benchmark failed"
        )
        raise

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
