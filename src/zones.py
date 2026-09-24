from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
from ultralytics import YOLO

from camera import CameraConfig, CameraManager


# ============================================================
# ZONE CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class Zone:
    """Rectangular region representing a store zone."""

    name: str

    x1: int
    y1: int
    x2: int
    y2: int

    def contains(
        self,
        x: int,
        y: int,
    ) -> bool:
        """Return True if a point is inside this zone."""

        return (
            self.x1 <= x <= self.x2
            and self.y1 <= y <= self.y2
        )


@dataclass(frozen=True)
class ZoneConfig:
    """Configuration for shopper zone analytics."""

    model_path: str = "yolo11n.pt"

    # --------------------------------------------------------
    # Camera
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
    # Display
    # --------------------------------------------------------

    window_name: str = (
        "RetailEdge AI - Zone Analytics"
    )


# ============================================================
# ZONE ANALYZER
# ============================================================

class ZoneAnalyzer:
    """
    Real-time shopper zone analytics.

    Capabilities:

        - Person detection
        - Persistent tracking IDs
        - Zone classification
        - Current occupancy
        - Unique zone visitors
        - Zone transitions
        - Per-zone dwell time

    Pipeline:

        CameraManager
             ↓
           Frame
             ↓
        YOLO + ByteTrack
             ↓
         Person ID
             ↓
        Person centroid
             ↓
        Zone classification
             ↓
        Occupancy + Visitor History + Dwell Time
    """

    def __init__(
        self,
        config: ZoneConfig,
    ) -> None:

        self.config = config

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        self.model = YOLO(
            self.config.model_path
        )

        # ----------------------------------------------------
        # Camera Manager
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
        # FPS
        # ----------------------------------------------------

        self.previous_time = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # Store Zones
        # ----------------------------------------------------

        self.zones = [
            Zone(
                name="Zone A",
                x1=0,
                y1=0,
                x2=640,
                y2=360,
            ),

            Zone(
                name="Zone B",
                x1=0,
                y1=360,
                x2=640,
                y2=720,
            ),

            Zone(
                name="Zone C",
                x1=640,
                y1=0,
                x2=1280,
                y2=720,
            ),
        ]

        # ----------------------------------------------------
        # Current occupancy.
        # ----------------------------------------------------

        self.current_occupancy: dict[
            str,
            int,
        ] = {
            zone.name: 0
            for zone in self.zones
        }

        # ----------------------------------------------------
        # Unique visitors per zone.
        #
        # Example:
        #
        # Zone A → {1, 4, 7}
        # ----------------------------------------------------

        self.zone_visitors: dict[
            str,
            set[int],
        ] = {
            zone.name: set()
            for zone in self.zones
        }

        # ----------------------------------------------------
        # Current zone of every active track.
        #
        # ID 10 → Zone A
        # ID 11 → Zone B
        # ----------------------------------------------------

        self.person_current_zone: dict[
            int,
            Optional[str],
        ] = {}

        # ----------------------------------------------------
        # Zone entry timestamp for each person.
        #
        # Example:
        #
        # (12, "Zone A") → timestamp
        #
        # This tells us when ID 12 entered Zone A.
        # ----------------------------------------------------

        self.zone_entry_time: dict[
            tuple[int, str],
            float,
        ] = {}

        # ----------------------------------------------------
        # Accumulated dwell time.
        #
        # Example:
        #
        # Zone A → {
        #     12: 18.5,
        #     17: 42.1
        # }
        #
        # Values represent completed dwell sessions.
        # ----------------------------------------------------

        self.person_zone_dwell: dict[
            tuple[int, str],
            float,
        ] = {}

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
        """Detect and track people."""

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
    # FIND ZONE
    # ========================================================

    def get_zone(
        self,
        center_x: int,
        center_y: int,
    ) -> Optional[Zone]:
        """Return the zone containing a centroid."""

        for zone in self.zones:

            if zone.contains(
                center_x,
                center_y,
            ):
                return zone

        return None

    # ========================================================
    # ENTER ZONE
    # ========================================================

    def enter_zone(
        self,
        track_id: int,
        zone_name: str,
        current_time: float,
    ) -> None:
        """Register a shopper entering a zone."""

        key = (
            track_id,
            zone_name,
        )

        # Only create an entry timestamp if
        # one doesn't already exist.
        if key not in self.zone_entry_time:

            self.zone_entry_time[
                key
            ] = current_time

    # ========================================================
    # EXIT ZONE
    # ========================================================

    def exit_zone(
        self,
        track_id: int,
        zone_name: str,
        current_time: float,
    ) -> None:
        """
        Register a shopper leaving a zone.

        The elapsed time is added to the person's
        accumulated dwell time for that zone.
        """

        key = (
            track_id,
            zone_name,
        )

        entry_time = (
            self.zone_entry_time.pop(
                key,
                None,
            )
        )

        if entry_time is None:

            return

        dwell_seconds = max(
            0.0,
            current_time - entry_time,
        )

        self.person_zone_dwell[
            key
        ] = (
            self.person_zone_dwell.get(
                key,
                0.0,
            )
            + dwell_seconds
        )

    # ========================================================
    # UPDATE ZONE STATE
    # ========================================================

    def update_zone_state(
        self,
        track_id: int,
        zone: Optional[Zone],
        current_time: float,
    ) -> None:
        """
        Update a person's current zone.

        Handles:

            Zone A → Zone B
            Zone A → Outside
            Outside → Zone A
            Zone A → Zone A
        """

        previous_zone_name = (
            self.person_current_zone.get(
                track_id
            )
        )

        current_zone_name = (
            zone.name
            if zone is not None
            else None
        )

        # ----------------------------------------------------
        # No zone change.
        # ----------------------------------------------------

        if (
            previous_zone_name
            == current_zone_name
        ):

            return

        # ----------------------------------------------------
        # Leaving previous zone.
        # ----------------------------------------------------

        if previous_zone_name is not None:

            self.exit_zone(
                track_id,
                previous_zone_name,
                current_time,
            )

        # ----------------------------------------------------
        # Entering new zone.
        # ----------------------------------------------------

        if current_zone_name is not None:

            self.enter_zone(
                track_id,
                current_zone_name,
                current_time,
            )

            # Register unique visitor.
            self.zone_visitors[
                current_zone_name
            ].add(track_id)

        # ----------------------------------------------------
        # Store current state.
        # ----------------------------------------------------

        self.person_current_zone[
            track_id
        ] = current_zone_name

        # ----------------------------------------------------
        # Log zone transition.
        # ----------------------------------------------------

        if (
            previous_zone_name is not None
            and current_zone_name is not None
        ):

            print(
                f"[ZONE] "
                f"Person ID={track_id}: "
                f"{previous_zone_name} -> "
                f"{current_zone_name}"
            )

        elif current_zone_name is not None:

            print(
                f"[ZONE] "
                f"Person ID={track_id}: "
                f"entered {current_zone_name}"
            )

        elif previous_zone_name is not None:

            print(
                f"[ZONE] "
                f"Person ID={track_id}: "
                f"left {previous_zone_name}"
            )

    # ========================================================
    # CLEAN LOST TRACKS
    # ========================================================

    def cleanup_lost_tracks(
        self,
        active_track_ids: set[int],
        current_time: float,
    ) -> None:
        """
        Handle people whose tracking IDs disappeared.

        Their current zone is closed so that accumulated
        dwell time is not lost.
        """

        tracked_ids = set(
            self.person_current_zone.keys()
        )

        lost_ids = (
            tracked_ids
            - active_track_ids
        )

        for track_id in lost_ids:

            previous_zone_name = (
                self.person_current_zone.pop(
                    track_id
                )
            )

            if previous_zone_name is not None:

                self.exit_zone(
                    track_id,
                    previous_zone_name,
                    current_time,
                )

    # ========================================================
    # CURRENT DWELL TIME
    # ========================================================

    def get_current_dwell(
        self,
        track_id: int,
        zone_name: str,
        current_time: float,
    ) -> float:
        """
        Return total dwell time for a person in a zone,
        including the currently active session.
        """

        key = (
            track_id,
            zone_name,
        )

        accumulated = (
            self.person_zone_dwell.get(
                key,
                0.0,
            )
        )

        entry_time = (
            self.zone_entry_time.get(
                key
            )
        )

        if entry_time is None:

            return accumulated

        active_session = max(
            0.0,
            current_time - entry_time,
        )

        return (
            accumulated
            + active_session
        )

    # ========================================================
    # ZONE AVERAGE DWELL
    # ========================================================

    def get_zone_average_dwell(
        self,
        zone_name: str,
        current_time: float,
    ) -> float:
        """
        Calculate average dwell time among shoppers
        who have visited the zone during this session.
        """

        visitor_ids = (
            self.zone_visitors[
                zone_name
            ]
        )

        if not visitor_ids:

            return 0.0

        total_dwell = 0.0

        for track_id in visitor_ids:

            total_dwell += (
                self.get_current_dwell(
                    track_id,
                    zone_name,
                    current_time,
                )
            )

        return (
            total_dwell
            / len(visitor_ids)
        )

    # ========================================================
    # DRAW ZONES
    # ========================================================

    def draw_zones(
        self,
        frame,
    ) -> None:
        """Draw configured store zones."""

        for zone in self.zones:

            cv2.rectangle(
                frame,
                (
                    zone.x1,
                    zone.y1,
                ),
                (
                    zone.x2,
                    zone.y2,
                ),
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                zone.name,
                (
                    zone.x1 + 10,
                    zone.y1 + 30,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
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
        current_time: float,
    ) -> dict[str, int]:
        """
        Draw tracked people and update:

            - occupancy
            - visitors
            - zone transitions
            - dwell time
        """

        zone_counts = {
            zone.name: 0
            for zone in self.zones
        }

        active_track_ids: set[int] = set()

        # ----------------------------------------------------
        # No tracking IDs.
        # ----------------------------------------------------

        if (
            result.boxes is None
            or result.boxes.id is None
        ):

            self.current_occupancy = (
                zone_counts
            )

            self.cleanup_lost_tracks(
                active_track_ids,
                current_time,
            )

            return zone_counts

        # ----------------------------------------------------
        # Extract tracking information.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Process every tracked person.
        # ----------------------------------------------------

        for (
            box,
            track_id,
            confidence,
        ) in zip(
            boxes,
            track_ids,
            confidences,
        ):

            active_track_ids.add(
                track_id
            )

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
            # Find current zone.
            # ------------------------------------------------

            zone = self.get_zone(
                center_x,
                center_y,
            )

            if zone is not None:

                zone_counts[
                    zone.name
                ] += 1

                zone_label = zone.name

            else:

                zone_label = "Outside"

            # ------------------------------------------------
            # Update zone state.
            # ------------------------------------------------

            self.update_zone_state(
                track_id,
                zone,
                current_time,
            )

            # ------------------------------------------------
            # Calculate current dwell.
            # ------------------------------------------------

            if zone is not None:

                dwell_seconds = (
                    self.get_current_dwell(
                        track_id,
                        zone.name,
                        current_time,
                    )
                )

                dwell_label = (
                    f"{dwell_seconds:.1f}s"
                )

            else:

                dwell_label = ""

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
                f"{zone_label}"
            )

            if dwell_label:

                label += (
                    f" | {dwell_label}"
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
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # Clean lost tracking IDs.
        # ----------------------------------------------------

        self.cleanup_lost_tracks(
            active_track_ids,
            current_time,
        )

        self.current_occupancy = (
            zone_counts
        )

        return zone_counts

    # ========================================================
    # DASHBOARD
    # ========================================================

    def draw_dashboard(
        self,
        frame,
        zone_counts: dict[str, int],
        fps: float,
        current_time: float,
    ) -> None:
        """Draw zone analytics dashboard."""

        cv2.rectangle(
            frame,
            (
                10,
                10,
            ),
            (
                500,
                205,
            ),
            (0, 0, 0),
            -1,
        )

        y_position = 40

        for (
            zone_name,
            current_count,
        ) in zone_counts.items():

            visitor_count = len(
                self.zone_visitors[
                    zone_name
                ]
            )

            average_dwell = (
                self.get_zone_average_dwell(
                    zone_name,
                    current_time,
                )
            )

            text = (
                f"{zone_name}: "
                f"{current_count} now | "
                f"{visitor_count} visitors | "
                f"Avg {average_dwell:.1f}s"
            )

            cv2.putText(
                frame,
                text,
                (
                    20,
                    y_position,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            y_position += 40

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (
                400,
                190,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # ========================================================
    # MAIN LOOP
    # ========================================================

    def run(self) -> None:
        """Run real-time zone analytics."""

        self.camera.open()

        print()
        print("=" * 70)
        print(
            "RetailEdge AI - Shopper Zone Analytics"
        )
        print("=" * 70)
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
        print()
        print(
            "Analytics:"
        )
        print(
            "  Current occupancy"
        )
        print(
            "  Unique zone visitors"
        )
        print(
            "  Zone transitions"
        )
        print(
            "  Per-shopper dwell time"
        )
        print(
            "  Average zone dwell time"
        )
        print()
        print(
            "Press Q to quit."
        )
        print("=" * 70)
        print()

        try:

            while True:

                # ------------------------------------------------
                # Read frame.
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

                current_time = (
                    time.monotonic()
                )

                # ------------------------------------------------
                # YOLO + ByteTrack.
                # ------------------------------------------------

                result = (
                    self.track_people(
                        frame
                    )
                )

                # ------------------------------------------------
                # Zone analytics.
                # ------------------------------------------------

                zone_counts = (
                    self.draw_people(
                        frame,
                        result,
                        current_time,
                    )
                )

                # ------------------------------------------------
                # Draw zones.
                # ------------------------------------------------

                self.draw_zones(
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
                    zone_counts,
                    fps,
                    current_time,
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
                "Zone analytics interrupted."
            )

        finally:

            self.shutdown()

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(self) -> None:
        """Release camera and GUI resources."""

        # ----------------------------------------------------
        # Close active dwell sessions.
        # ----------------------------------------------------

        current_time = (
            time.monotonic()
        )

        for (
            track_id,
            zone_name,
        ) in list(
            self.person_current_zone.items()
        ):

            if zone_name is not None:

                self.exit_zone(
                    track_id,
                    zone_name,
                    current_time,
                )

        # ----------------------------------------------------
        # Release camera.
        # ----------------------------------------------------

        self.camera.release()

        cv2.destroyAllWindows()

        print()
        print(
            "Camera released."
        )

        print(
            "RetailEdge zone analytics stopped."
        )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main() -> None:
    """Application entry point."""

    config = ZoneConfig()

    analyzer = ZoneAnalyzer(
        config
    )

    analyzer.run()


if __name__ == "__main__":
    main()