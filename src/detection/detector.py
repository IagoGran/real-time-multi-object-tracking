import numpy as np

from src.types import Detection


class Detector:
    """Shared detector API: inference plus reusable confidence filtering."""

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run model inference and return detections above its internal floor."""
        raise NotImplementedError

    @staticmethod
    def filter_by_confidence(
        detections: list[Detection],
        confidence_threshold: float,
    ) -> list[Detection]:
        return [
            detection
            for detection in detections
            if detection.confidence >= confidence_threshold
        ]

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> list[Detection]:
        return self.filter_by_confidence(
            self.predict(frame),
            confidence_threshold,
        )
