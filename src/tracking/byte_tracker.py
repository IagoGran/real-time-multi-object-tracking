# src/tracking/byte_tracker.py

import torch

from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker as UltralyticsByteTracker
from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

from src.types import Detection, Track


class ByteTracker:

    def __init__(self):
        args = IterableSimpleNamespace(
            **YAML.load(ROOT / "cfg/trackers/bytetrack.yaml")
        )

        self.tracker = UltralyticsByteTracker(args)

    def update(
        self,
        detections: list[Detection],
        frame
    ) -> list[Track]:

        # Convert our Detection objects to the format expected by Ultralytics:
        # [x1, y1, x2, y2, confidence, class_id]
        data = torch.tensor(
            [
                [
                    detection.x1,
                    detection.y1,
                    detection.x2,
                    detection.y2,
                    detection.confidence,
                    detection.class_id,
                ]
                for detection in detections
            ],
            dtype=torch.float32,
        )

        # Ensure correct shape when there are no detections
        if len(detections) == 0:
            data = torch.empty((0, 6), dtype=torch.float32)

        boxes = Boxes(
            data,
            orig_shape=frame.shape[:2],
        )

        # ByteTrack maintains its internal state between calls
        tracked_objects = self.tracker.update(boxes, frame)

        tracks: list[Track] = []

        for tracked_object in tracked_objects:
            x1, y1, x2, y2 = tracked_object[:4]
            track_id = int(tracked_object[4])
            confidence = float(tracked_object[5])
            class_id = int(tracked_object[6])
            detection_index = int(tracked_object[7])

            # ByteTrack gives us the index of the original detection,
            # so we can recover our own metadata.
            class_name = detections[detection_index].class_name

            tracks.append(
                Track(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    track_id=track_id,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                )
            )

        return tracks