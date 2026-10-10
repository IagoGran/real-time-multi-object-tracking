from collections import deque

from src.types import Track, Trajectory, TrajectoryPoint


class TrajectoryManager:

    def __init__(self, max_points: int = 100):
        self.max_points = max_points
        self.trajectories: dict[int, Trajectory] = {}

    def update(
        self,
        tracks: list[Track],
        frame_index: int
    ) -> None:

        for track in tracks:
            if track.track_id not in self.trajectories:
                self.trajectories[track.track_id] = Trajectory(
                    track_id=track.track_id,
                    points=deque(maxlen=self.max_points),
                )

            point = TrajectoryPoint(
                frame_index=frame_index,
                x=(track.x1 + track.x2) / 2,
                y=(track.y1 + track.y2) / 2,
            )

            self.trajectories[track.track_id].points.append(point)

    def get(self, track_id: int) -> Trajectory | None:
        return self.trajectories.get(track_id)