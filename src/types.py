from collections import deque
from dataclasses import dataclass, field


@dataclass
class Detection:
    """
    Represents a single detection result from an object detection model.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


@dataclass
class Track:
    """
    Represents a tracked object in the current frame.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int
    confidence: float
    class_id: int
    class_name: str


@dataclass
class TrajectoryPoint:
    """
    Represents the position of a tracked object in a specific frame.
    """
    frame_index: int
    x: float
    y: float


@dataclass
class Trajectory:
    """
    Represents the trajectory of one tracked object across frames.
    """
    track_id: int
    points: deque[TrajectoryPoint] = field(default_factory=deque)
    
@dataclass(frozen=True)
class CountingLine:
    """
    Represents a line using normalized coordinates.

    Coordinates are expressed from 0.0 to 1.0 relative
    to the frame width and height.
    """
    start: tuple[float, float]
    end: tuple[float, float]

    def to_pixels(
        self,
        frame
    ) -> tuple[tuple[int, int], tuple[int, int]]:

        height, width = frame.shape[:2]

        start_x = int(self.start[0] * width)
        start_y = int(self.start[1] * height)

        end_x = int(self.end[0] * width)
        end_y = int(self.end[1] * height)

        return (
            (start_x, start_y),
            (end_x, end_y),
        )