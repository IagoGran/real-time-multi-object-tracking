from pathlib import Path
import math

import numpy as np
from ultralytics import YOLO

from src.detection.detector import Detector
from src.types import Detection


class YoloDetector(Detector):

    STRIDE = 32

    DEFAULT_INFERENCE_CONFIDENCE = 0.01

    def __init__(
        self,
        model_path: str | Path,
        class_names: list[str],
        inference_confidence: float = DEFAULT_INFERENCE_CONFIDENCE,
        device: str | int | None = None,
    ):
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.inference_confidence = inference_confidence
        self.device = device

    def predict(
        self,
        frame: np.ndarray,
    ) -> list[Detection]:
        """
        Run model inference once and return all relevant detections
        above the detector's minimum inference confidence.

        No benchmark/evaluation confidence threshold is applied here.
        """

        imgsz = self._get_inference_size(frame)

        prediction_options = {
            "imgsz": imgsz,
            "verbose": False,
            "conf": self.inference_confidence,
        }
        if self.device is not None:
            prediction_options["device"] = self.device
        results = self.model(frame, **prediction_options)

        detections: list[Detection] = []

        for result in results:
            for box, confidence, class_id in zip(
                result.boxes.xyxy,
                result.boxes.conf,
                result.boxes.cls,
            ):
                x1, y1, x2, y2 = map(
                    float,
                    box.tolist(),
                )

                confidence = float(confidence)
                class_id = int(class_id)
                class_name = self.model.names[class_id]

                if class_name not in self.class_names:
                    continue

                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name,
                    )
                )

        return detections

    def _get_inference_size(
        self,
        frame: np.ndarray,
    ) -> tuple[int, int]:

        height, width = frame.shape[:2]

        inference_height = (
            math.ceil(height / self.STRIDE)
            * self.STRIDE
        )

        inference_width = (
            math.ceil(width / self.STRIDE)
            * self.STRIDE
        )

        return (
            inference_height,
            inference_width,
        )
