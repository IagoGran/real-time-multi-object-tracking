from typing import Protocol

import numpy as np

from src.types import Detection


class Detector(Protocol):

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> list[Detection]:
        ...
