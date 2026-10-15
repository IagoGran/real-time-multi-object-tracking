import numpy as np

from src.detection.detector import Detector
from src.types import Detection


def test_detect_predicts_then_filters_inclusive_threshold():
    class StubDetector(Detector):
        def __init__(self):
            self.calls = 0

        def predict(self, frame):
            self.calls += 1
            return [
                Detection(0, 0, 1, 1, 0.49, 0, "person"),
                Detection(0, 0, 1, 1, 0.50, 0, "person"),
                Detection(0, 0, 1, 1, 0.90, 0, "person"),
            ]

    detector = StubDetector()
    filtered = detector.detect(np.zeros((2, 2, 3)), confidence_threshold=0.5)
    assert detector.calls == 1
    assert [item.confidence for item in filtered] == [0.5, 0.9]
    assert [item.confidence for item in detector.filter_by_confidence(filtered, 0.9)] == [0.9]
