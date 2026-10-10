from src.types import Track


Point = tuple[float, float]


class LineCrossingCounter:
    """
    Counts tracked objects crossing a finite line segment.

    Each tracked object is represented by the center of its bounding box.

    The line divides the image into two sides:
        A -> negative side
        B -> positive side

    Crossing direction depends on the orientation from
    line_start to line_end.
    """

    def __init__(
        self,
        line_start: Point,
        line_end: Point,
        tolerance: float = 10.0,
        cooldown_frames: int = 10,
    ):
        if line_start == line_end:
            raise ValueError(
                "line_start and line_end cannot be identical."
            )

        self.line_start = line_start
        self.line_end = line_end

        self.tolerance = tolerance
        self.cooldown_frames = cooldown_frames

        self.a_to_b = 0
        self.b_to_a = 0

        # Last known stable side of each track.
        self._last_side: dict[int, int] = {}

        # Last position outside the tolerance area.
        self._last_position: dict[int, Point] = {}

        # Prevent repeated counting caused by small oscillations.
        self._last_crossing_frame: dict[int, int] = {}

    @property
    def total(self) -> int:
        return self.a_to_b + self.b_to_a

    def update(
        self,
        tracks: list[Track],
        frame_index: int,
    ) -> None:
        """
        Update crossing state using the tracks from the current frame.
        """

        for track in tracks:

            position = self._track_center(track)

            side = self._side(position)

            # If the object's center is very close to the line,
            # keep the previous stable state.
            if side == 0:
                continue

            previous_side = self._last_side.get(
                track.track_id
            )

            previous_position = self._last_position.get(
                track.track_id
            )

            if (
                previous_side is not None
                and previous_position is not None
                and previous_side != side
            ):
                if self._can_count_crossing(
                    track.track_id,
                    frame_index,
                ):
                    if self._segments_intersect(
                        previous_position,
                        position,
                        self.line_start,
                        self.line_end,
                    ):
                        if previous_side == -1 and side == 1:
                            self.a_to_b += 1

                        elif previous_side == 1 and side == -1:
                            self.b_to_a += 1

                        self._last_crossing_frame[
                            track.track_id
                        ] = frame_index

            self._last_side[track.track_id] = side
            self._last_position[track.track_id] = position

    def _can_count_crossing(
        self,
        track_id: int,
        frame_index: int,
    ) -> bool:
        """
        Check whether enough frames have passed since the last
        counted crossing for this track.
        """

        last_crossing = self._last_crossing_frame.get(
            track_id
        )

        if last_crossing is None:
            return True

        return (
            frame_index - last_crossing
            >= self.cooldown_frames
        )

    @staticmethod
    def _track_center(
        track: Track,
    ) -> Point:
        """
        Return the center of the track bounding box.
        """

        return (
            (track.x1 + track.x2) / 2,
            (track.y1 + track.y2) / 2,
        )

    def _side(
        self,
        point: Point,
    ) -> int:
        """
        Determine on which side of the line a point lies.

        Returns:
            -1: side A
             0: inside tolerance area
             1: side B
        """

        x1, y1 = self.line_start
        x2, y2 = self.line_end

        px, py = point

        cross_product = (
            (x2 - x1) * (py - y1)
            - (y2 - y1) * (px - x1)
        )

        line_length = (
            (x2 - x1) ** 2
            + (y2 - y1) ** 2
        ) ** 0.5

        distance = abs(cross_product) / line_length

        if distance <= self.tolerance:
            return 0

        return 1 if cross_product > 0 else -1

    @staticmethod
    def _segments_intersect(
        p1: Point,
        p2: Point,
        q1: Point,
        q2: Point,
    ) -> bool:
        """
        Check whether segment p1-p2 intersects segment q1-q2.
        """

        def orientation(
            a: Point,
            b: Point,
            c: Point,
        ) -> float:

            return (
                (b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0])
            )

        o1 = orientation(p1, p2, q1)
        o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1)
        o4 = orientation(q1, q2, p2)

        return (
            o1 * o2 <= 0
            and o3 * o4 <= 0
        )