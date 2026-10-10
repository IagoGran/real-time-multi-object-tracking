from pathlib import Path
import math

from ultralytics import YOLO

from src.types import Detection


class YoloDetector:

    STRIDE = 32

    def __init__(
        self,
        model_path: str | Path,
        class_names: list[str]
    ):
        self.model = YOLO(model_path)
        self.class_names = class_names

    def detect(
        self,
        frame,
        confidence_threshold: float = 0.5,
    ) -> list[Detection]:

        imgsz = self._get_inference_size(frame)

        results = self.model(
            frame,
            imgsz=imgsz,
            verbose=False,
        )

        detections = []

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

                if (
                    class_name in self.class_names
                    and confidence > confidence_threshold
                ):
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
        frame
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