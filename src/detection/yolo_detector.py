from pathlib import Path
from ultralytics import YOLO
from typing import List
from src.types import Detection

class YoloDetector:

    def __init__(self, model_path: str | Path, class_names: List[str]):
        self.model = YOLO(model_path)
        self.class_names = class_names

    def detect(self, frame, confidence_threshold: float = 0.5) -> list[Detection]:
        results = self.model(frame)
        detections = []
        for result in results:
            for box, confidence, class_id in zip(result.boxes.xyxy,
                                                 result.boxes.conf,
                                                 result.boxes.cls):
                x1, y1, x2, y2 = map(int, box.tolist())
                confidence = float(confidence)
                class_id = int(class_id)
                if self.model.names[class_id] in self.class_names and confidence > confidence_threshold:
                    detections.append(Detection(x1, y1, x2, y2,
                                                confidence,
                                                class_id,
                                                class_name=self.model.names[class_id]))
        return detections