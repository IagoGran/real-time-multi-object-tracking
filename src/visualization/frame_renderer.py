import cv2

from src.types import Track, Trajectory, CountingLine


class FrameRenderer:

    @staticmethod
    def _draw_outlined_text(
        frame,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int],
        font_scale: float = 0.7,
        thickness: int = 2,
        outline_thickness: int = 5,
    ) -> None:

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Outline
        cv2.putText(
            frame,
            text,
            position,
            font,
            font_scale,
            (0, 0, 0),
            outline_thickness,
            cv2.LINE_AA,
        )

        # Foreground
        cv2.putText(
            frame,
            text,
            position,
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def draw_track(
        self,
        frame,
        track: Track
    ) -> None:

        x1, y1, x2, y2 = map(
            int,
            (track.x1, track.y1, track.x2, track.y2)
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        self._draw_outlined_text(
            frame,
            f"ID: {track.track_id}",
            (x1, max(y1 - 10, 20)),
            (0, 255, 0),
            font_scale=0.5,
            thickness=2,
            outline_thickness=4,
        )

    def draw_trajectory(
        self,
        frame,
        trajectory: Trajectory
    ) -> None:

        points = list(trajectory.points)

        for i in range(1, len(points)):
            previous = points[i - 1]
            current = points[i]

            cv2.line(
                frame,
                (int(previous.x), int(previous.y)),
                (int(current.x), int(current.y)),
                (0, 255, 255),
                2,
            )

    def draw_counting_line(
        self,
        frame,
        line: CountingLine,
    ) -> None:

        start, end = line.to_pixels(frame)

        cv2.line(
            frame,
            start,
            end,
            (0, 0, 255),
            3,
        )

        self._draw_outlined_text(
            frame,
            "COUNTING LINE",
            (start[0], max(start[1] - 10, 20)),
            (0, 0, 255),
            font_scale=0.6,
        )

    def draw_crossing_counts(
        self,
        frame,
        a_to_b: int,
        b_to_a: int,
    ) -> None:

        self._draw_outlined_text(
            frame,
            f"A -> B: {a_to_b}",
            (20, 70),
            (0, 255, 0),
            font_scale=0.8,
        )

        self._draw_outlined_text(
            frame,
            f"B -> A: {b_to_a}",
            (20, 110),
            (0, 165, 255),
            font_scale=0.8,
        )
    def draw_status(
        self,
        frame,
        text: str,
    ) -> None:

        self._draw_outlined_text(
            frame,
            text,
            (20, 30),
            (255, 255, 255),
            font_scale=0.65,
            thickness=2,
            outline_thickness=5,
        )