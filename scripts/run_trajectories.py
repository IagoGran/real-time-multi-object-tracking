# scripts/run_trajectories.py

from pathlib import Path
import configparser
import time

import cv2

from src.detection.yolo_detector import YoloDetector
from src.tracking.byte_tracker import ByteTracker
from src.tracking.trajectory_manager import TrajectoryManager
from src.visualization.frame_renderer import FrameRenderer


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

SEQUENCE_PATH = Path(
    "datasets/MOT17/train/MOT17-04-FRCNN"
)

MODEL_PATH = "yolo26m.pt"
CLASS_NAMES = ["person"]
CONFIDENCE_THRESHOLD = 0.1
MAX_TRAJECTORY_POINTS = 100


# -------------------------------------------------------------------
# Load sequence metadata
# -------------------------------------------------------------------

frames_path = SEQUENCE_PATH / "img1"
seqinfo_path = SEQUENCE_PATH / "seqinfo.ini"

frames = sorted(frames_path.glob("*.jpg"))

if not frames:
    raise RuntimeError(
        f"No frames found in {frames_path}"
    )

if not seqinfo_path.exists():
    raise RuntimeError(
        f"Sequence info not found: {seqinfo_path}"
    )

config = configparser.ConfigParser()
config.read(seqinfo_path)

source_fps = config.getint(
    "Sequence",
    "frameRate",
)

frame_duration_ms = 1000 / source_fps

width = config.getint(
    "Sequence",
    "imWidth",
)

height = config.getint(
    "Sequence",
    "imHeight",
)

print(f"Sequence: {SEQUENCE_PATH.name}")
print(f"Frames: {len(frames)}")
print(f"Resolution: {width}x{height}")
print(f"Source FPS: {source_fps}")


# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

detector = YoloDetector(
    MODEL_PATH,
    CLASS_NAMES,
)

tracker = ByteTracker()

trajectory_manager = TrajectoryManager(
    max_points=MAX_TRAJECTORY_POINTS,
)

renderer = FrameRenderer()


# -------------------------------------------------------------------
# Processing loop
# -------------------------------------------------------------------

try:
    for frame_index, frame_path in enumerate(
        frames,
        start=1,
    ):

        frame_start = time.perf_counter()

        frame = cv2.imread(str(frame_path))

        if frame is None:
            print(
                f"Failed to read frame: {frame_path}"
            )
            continue

        # -----------------------------------------------------------
        # Detection
        # -----------------------------------------------------------

        detections = detector.detect(
            frame,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )

        # -----------------------------------------------------------
        # Tracking
        # -----------------------------------------------------------

        tracks = tracker.update(
            detections,
            frame,
        )

        # -----------------------------------------------------------
        # Trajectories
        # -----------------------------------------------------------

        trajectory_manager.update(
            tracks,
            frame_index,
        )

        # -----------------------------------------------------------
        # Visualization
        # -----------------------------------------------------------

        for track in tracks:

            renderer.draw_track(
                frame,
                track,
            )

            trajectory = trajectory_manager.get(
                track.track_id
            )

            if trajectory is not None:
                renderer.draw_trajectory(
                    frame,
                    trajectory,
                )

        # -----------------------------------------------------------
        # Timing
        # -----------------------------------------------------------

        processing_time_ms = (
            time.perf_counter()
            - frame_start
        ) * 1000

        processing_fps = (
            1000 / processing_time_ms
            if processing_time_ms > 0
            else 0
        )

        # -----------------------------------------------------------
        # Status
        # -----------------------------------------------------------

        renderer.draw_status(
            frame,
            f"Frame: {frame_index}/{len(frames)} | "
            f"Detections: {len(detections)} | "
            f"Tracks: {len(tracks)} | "
            f"Trajectories: {len(trajectory_manager.trajectories)} | "
            f"Processing: {processing_fps:.1f} FPS | "
            f"Source: {source_fps} FPS",
        )

        cv2.imshow(
            "Multi-Object Tracking - Trajectories",
            frame,
        )

        # -----------------------------------------------------------
        # Match source playback speed when possible
        # -----------------------------------------------------------

        remaining_time_ms = (
            frame_duration_ms
            - processing_time_ms
        )

        delay = max(
            1,
            round(remaining_time_ms),
        )

        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

finally:
    cv2.destroyAllWindows()