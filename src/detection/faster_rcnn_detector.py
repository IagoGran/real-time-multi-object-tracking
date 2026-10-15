import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

from src.detection.detector import Detector
from src.types import Detection


class FasterRCNNDetector(Detector):

    def __init__(
        self,
        class_names: list[str] | None = None,
    ):
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights)
        self.model.eval()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.class_names = (
            ["person"] if class_names is None else class_names
        )
        self.categories = self.weights.meta["categories"]

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> list[Detection]:
        frame_tensor = torch.from_numpy(
            frame[:, :, ::-1].copy()
        ).permute(2, 0, 1).to(dtype=torch.float32)
        frame_tensor = frame_tensor.div_(255).to(self.device)

        with torch.inference_mode():
            result = self.model([frame_tensor])[0]

        detections = []

        for box, confidence, class_id in zip(
            result["boxes"].cpu().tolist(),
            result["scores"].cpu().tolist(),
            result["labels"].cpu().tolist(),
        ):
            class_id = int(class_id)
            class_name = self.categories[class_id]
            confidence = float(confidence)

            if (
                class_name in self.class_names
                and confidence > confidence_threshold
            ):
                x1, y1, x2, y2 = map(float, box)
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
