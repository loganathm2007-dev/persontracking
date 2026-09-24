from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import cv2
from ultralytics import YOLO

from camera import CameraConfig, CameraManager


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CountingConfig:
    """Configuration for shopper entry/exit analytics."""

    model_path: str = "yolo11n.pt"

    # --------------------------------------------------------
    # Camera source
    #
    # 0 = laptop / USB webcam
    #
    # Later:
    # "rtsp://username:password@camera-ip/stream"
    # --------------------------------------------------------

    camera_source: int | str = 0

    camera_width: int = 1280
    camera_height: int = 720

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    person_class_id: int = 0

    confidence_threshold: float = 0.40
    iou_threshold: float = 0.50

    tracker_config: str = "bytetrack.yaml"

    # --------------------------------------------------------
    # Entry / exit line
    # --------------------------------------------------------

    line_y: int = 360

    # Dead zone around the counting line.
    crossing_threshold: int = 15

    # Prevent rapid duplicate crossings.
    crossing_cooldown_seconds: float = 1.0

    # Track history.
    history_length: int = 30

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    window_name: str = (
        "RetailEdge AI - Entry Exit Analytics"
    )


# ============================================================
# TRACK STATE
# ============================================================

@dataclass
class TrackState:
    """State maintained for each tracking ID."""

    positions: deque = field(
        default_factory=lambda: deque(maxlen=30)
    )

    # -1 = above counting line
    #  0 = dead zone / unknown
    # +1 = below counting line
    last_side: int = 0

    last_crossing_time: float = 0.0


# ============================================================
# ENTRY / EXIT COUNTER
# ============================================================

class EntryExitCounter:
    """
    Real-time shopper entry/exit analytics.

    Camera management is delegated to CameraManager.

    Pipeline:

        CameraManager
             ↓
           Frame
             ↓
        YOLO + ByteTrack
             ↓
         Tracking ID
             ↓
        Person centroid
             ↓
        Virtual line
             ↓
        Direction detection
             ↓
        Entry / Exit
    """

    def __init__(
        self,
        config: CountingConfig,
    ) -> None:

        self.config = config

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        self.model = YOLO(
            self.config.model_path
        )

        # ----------------------------------------------------
        # Camera manager
        # ----------------------------------------------------

        camera_config = CameraConfig(
            source=self.config.camera_source,
            width=self.config.camera_width,
            height=self.config.camera_height,
        )

        self.camera = CameraManager(
            camera_config
        )

        # ----------------------------------------------------
        # Tracking state
        # ----------------------------------------------------

        self.tracks: dict[
            int,
            TrackState,
        ] = {}

        # ----------------------------------------------------
        # Business metrics
        # ----------------------------------------------------

        self.entry_count: int = 0
        self.exit_count: int = 0

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        self.previous_time = (
            time.perf_counter()
        )

    # ========================================================
    # FPS
    # ========================================================

    def calculate_fps(self) -> float:
        """Calculate approximate processing FPS."""

        current_time = (
            time.perf_counter()
        )

        elapsed = (
            current_time
            - self.previous_time
        )

        self.previous_time = (
            current_time
        )

        if elapsed <= 0:
            return 0.0

        return 1.0 / elapsed

    # ========================================================
    # YOLO + BYTETRACK
    # ========================================================

    def track_people(
        self,
        frame,
    ):
        """Run YOLO person detection and ByteTrack."""

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.config.tracker_config,
            classes=[
                self.config.person_class_id
            ],
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            verbose=False,
        )

        return results[0]

    # ========================================================
    # TRACK STATE
    # ========================================================

    def get_track_state(
        self,
        track_id: int,
    ) -> TrackState:
        """Get or create state for a tracking ID."""

        if track_id not in self.tracks:

            self.tracks[track_id] = TrackState(
                positions=deque(
                    maxlen=self.config.history_length
                )
            )

        return self.tracks[track_id]

    # ========================================================
    # LINE SIDE
    # ========================================================

    def get_line_side(
        self,
        center_y: int,
    ) -> int:
        """
        Determine which side of the counting line
        the person is currently on.

        Returns:

            -1 → above line
             0 → dead zone
            +1 → below line
        """

        line_y = self.config.line_y
        threshold = self.config.crossing_threshold

        if center_y < (
            line_y - threshold
        ):

            return -1

        if center_y > (
            line_y + threshold
        ):

            return 1

        return 0

    # ========================================================
    # CROSSING DETECTION
    # ========================================================

    def update_track(
        self,
        track_id: int,
        center_x: int,
        center_y: int,
    ) -> None:
        """
        Update tracking state and detect crossings.

        TOP → BOTTOM = ENTRY

        BOTTOM → TOP = EXIT
        """

        track = self.get_track_state(
            track_id
        )

        # ----------------------------------------------------
        # Store centroid history.
        # ----------------------------------------------------

        track.positions.append(
            (
                center_x,
                center_y,
            )
        )

        # ----------------------------------------------------
        # Determine line side.
        # ----------------------------------------------------

        current_side = (
            self.get_line_side(
                center_y
            )
        )

        # Ignore dead-zone positions.
        if current_side == 0:

            return

        # ----------------------------------------------------
        # First valid observation.
        # ----------------------------------------------------

        if track.last_side == 0:

            track.last_side = (
                current_side
            )

            return

        # ----------------------------------------------------
        # No side change.
        # ----------------------------------------------------

        if current_side == (
            track.last_side
        ):

            return

        # ----------------------------------------------------
        # Crossing cooldown.
        # ----------------------------------------------------

        current_time = time.monotonic()

        time_since_last_crossing = (
            current_time
            - track.last_crossing_time
        )

        if (
            time_since_last_crossing
            < self.config.crossing_cooldown_seconds
        ):

            return

        # ----------------------------------------------------
        # TOP → BOTTOM = ENTRY
        # ----------------------------------------------------

        if (
            track.last_side == -1
            and current_side == 1
        ):

            self.entry_count += 1

            print(
                f"[ENTRY] "
                f"Person ID={track_id} | "
                f"Total entries="
                f"{self.entry_count}"
            )

        # ----------------------------------------------------
        # BOTTOM → TOP = EXIT
        # ----------------------------------------------------

        elif (
            track.last_side == 1
            and current_side == -1
        ):

            self.exit_count += 1

            print(
                f"[EXIT] "
                f"Person ID={track_id} | "
                f"Total exits="
                f"{self.exit_count}"
            )

        # ----------------------------------------------------
        # Update state.
        # ----------------------------------------------------

        track.last_side = (
            current_side
        )

        track.last_crossing_time = (
            current_time
        )

    # ========================================================
    # DRAW COUNTING LINE
    # ========================================================

    def draw_counting_line(
        self,
        frame,
    ) -> None:
        """Draw the entry/exit counting line."""

        height, width = (
            frame.shape[:2]
        )

        # Main line.
        cv2.line(
            frame,
            (
                0,
                self.config.line_y,
            ),
            (
                width,
                self.config.line_y,
            ),
            (255, 255, 0),
            3,
        )

        # Label.
        cv2.putText(
            frame,
            "ENTRY / EXIT LINE",
            (
                20,
                self.config.line_y - 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Entry direction.
        cv2.putText(
            frame,
            "ENTRY ↓",
            (
                20,
                self.config.line_y + 40,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Exit direction.
        cv2.putText(
            frame,
            "EXIT ↑",
            (
                20,
                self.config.line_y - 55,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # ========================================================
    # DRAW PEOPLE
    # ========================================================

    def draw_people(
        self,
        frame,
        result,
    ) -> int:
        """
        Draw tracked people and update crossing state.

        Returns:
            Number of currently tracked people.
        """

        if (
            result.boxes is None
            or result.boxes.id is None
        ):

            return 0

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .tolist()
        )

        tracked_count = len(
            track_ids
        )

        for (
            box,
            track_id,
            confidence,
        ) in zip(
            boxes,
            track_ids,
            confidences,
        ):

            x1, y1, x2, y2 = map(
                int,
                box,
            )

            # ------------------------------------------------
            # Centroid.
            # ------------------------------------------------

            center_x = (
                x1 + x2
            ) // 2

            center_y = (
                y1 + y2
            ) // 2

            # ------------------------------------------------
            # Update crossing logic.
            # ------------------------------------------------

            self.update_track(
                track_id,
                center_x,
                center_y,
            )

            # ------------------------------------------------
            # Bounding box.
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (
                    x1,
                    y1,
                ),
                (
                    x2,
                    y2,
                ),
                (0, 255, 0),
                2,
            )

            # ------------------------------------------------
            # Centroid.
            # ------------------------------------------------

            cv2.circle(
                frame,
                (
                    center_x,
                    center_y,
                ),
                5,
                (0, 0, 255),
                -1,
            )

            # ------------------------------------------------
            # Tracking label.
            # ------------------------------------------------

            label = (
                f"ID {track_id} | "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        y1 - 10,
                        25,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return tracked_count

    # ========================================================
    # DASHBOARD
    # ========================================================

    def draw_dashboard(
        self,
        frame,
        tracked_count: int,
        fps: float,
    ) -> None:
        """Draw entry/exit analytics."""

        inside_count = max(
            0,
            self.entry_count
            - self.exit_count,
        )

        cv2.rectangle(
            frame,
            (
                10,
                10,
            ),
            (
                390,
                155,
            ),
            (0, 0, 0),
            -1,
        )

        # Current tracked people.
        cv2.putText(
            frame,
            f"Tracked: {tracked_count}",
            (
                20,
                40,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Entries.
        cv2.putText(
            frame,
            f"Entries: {self.entry_count}",
            (
                20,
                70,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Exits.
        cv2.putText(
            frame,
            f"Exits: {self.exit_count}",
            (
                20,
                100,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # Estimated current occupancy.
        cv2.putText(
            frame,
            f"Inside: {inside_count}",
            (
                20,
                130,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # FPS.
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (
                270,
                40,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # ========================================================
    # MAIN LOOP
    # ========================================================

    def run(self) -> None:
        """Run real-time entry/exit analytics."""

        self.camera.open()

        print()
        print("=" * 60)
        print(
            "RetailEdge AI - "
            "Shopper Entry/Exit Analytics"
        )
        print("=" * 60)
        print(
            f"Model:       "
            f"{self.config.model_path}"
        )
        print(
            "Tracker:     ByteTrack"
        )
        print(
            f"Camera:      "
            f"{self.config.camera_source}"
        )
        print(
            "Target:      Person"
        )
        print(
            f"Confidence:  "
            f"{self.config.confidence_threshold}"
        )
        print(
            f"Line Y:      "
            f"{self.config.line_y}"
        )
        print()
        print(
            "TOP -> BOTTOM = ENTRY"
        )
        print(
            "BOTTOM -> TOP = EXIT"
        )
        print()
        print(
            "Press Q to quit."
        )
        print("=" * 60)
        print()

        try:

            while True:

                # ------------------------------------------------
                # Get frame from CameraManager.
                # ------------------------------------------------

                success, frame = (
                    self.camera.read()
                )

                if not success:

                    print(
                        "Warning: failed to "
                        "read camera frame."
                    )

                    continue

                # ------------------------------------------------
                # YOLO + ByteTrack.
                # ------------------------------------------------

                result = (
                    self.track_people(
                        frame
                    )
                )

                # ------------------------------------------------
                # Track people and count crossings.
                # ------------------------------------------------

                tracked_count = (
                    self.draw_people(
                        frame,
                        result,
                    )
                )

                # ------------------------------------------------
                # Counting line.
                # ------------------------------------------------

                self.draw_counting_line(
                    frame
                )

                # ------------------------------------------------
                # FPS.
                # ------------------------------------------------

                fps = (
                    self.calculate_fps()
                )

                # ------------------------------------------------
                # Dashboard.
                # ------------------------------------------------

                self.draw_dashboard(
                    frame,
                    tracked_count,
                    fps,
                )

                # ------------------------------------------------
                # Display.
                # ------------------------------------------------

                cv2.imshow(
                    self.config.window_name,
                    frame,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("q"):

                    break

        except KeyboardInterrupt:

            print()
            print(
                "Counting interrupted."
            )

        finally:

            self.shutdown()

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(self) -> None:
        """Release camera and GUI resources."""

        self.camera.release()

        cv2.destroyAllWindows()

        print()
        print(
            "RetailEdge counting stopped."
        )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main() -> None:
    """Application entry point."""

    config = CountingConfig()

    counter = EntryExitCounter(
        config
    )

    counter.run()


if __name__ == "__main__":
    main()