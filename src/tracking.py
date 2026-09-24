from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
from ultralytics import YOLO

from camera import CameraConfig, CameraManager


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TrackingConfig:
    """Configuration for shopper tracking."""

    model_path: str = "yolo11n.pt"

    # Camera source:
    #
    # 0 = laptop / USB webcam
    #
    # Later:
    # "rtsp://username:password@camera-ip/stream"
    #
    camera_source: int | str = 0

    camera_width: int = 1280
    camera_height: int = 720

    # COCO class 0 = person.
    person_class_id: int = 0

    confidence_threshold: float = 0.40
    iou_threshold: float = 0.50

    tracker_config: str = "bytetrack.yaml"

    window_name: str = (
        "RetailEdge AI - Person Tracking"
    )


# ============================================================
# SHOPPER TRACKER
# ============================================================

class ShopperTracker:
    """
    Real-time shopper tracking using YOLO + ByteTrack.

    Responsibilities:

        - Receive frames from CameraManager
        - Detect people
        - Maintain tracking IDs
        - Calculate centroids
        - Draw tracking information
        - Measure FPS

    Camera management is completely separated.
    """

    def __init__(
        self,
        config: TrackingConfig,
    ) -> None:

        self.config = config

        # ----------------------------------------------------
        # Load YOLO model once.
        # ----------------------------------------------------

        self.model = YOLO(
            self.config.model_path
        )

        # ----------------------------------------------------
        # Camera manager.
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
        # FPS.
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
        """
        Run YOLO person detection and ByteTrack.

        persist=True allows the tracker to maintain
        identities across consecutive frames.
        """

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
    # DRAW TRACKS
    # ========================================================

    def draw_tracks(
        self,
        frame,
        result,
    ) -> int:
        """
        Draw tracked people.

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
            # Person centroid.
            # ------------------------------------------------

            center_x = (
                x1 + x2
            ) // 2

            center_y = (
                y1 + y2
            ) // 2

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
                f"Person ID: {track_id} | "
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
    # STATUS
    # ========================================================

    def draw_status(
        self,
        frame,
        tracked_count: int,
        fps: float,
    ) -> None:
        """Draw tracking status."""

        cv2.rectangle(
            frame,
            (
                10,
                10,
            ),
            (
                350,
                100,
            ),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            (
                f"Tracked people: "
                f"{tracked_count}"
            ),
            (
                20,
                40,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (
                20,
                72,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # ========================================================
    # MAIN LOOP
    # ========================================================

    def run(self) -> None:
        """Run the real-time tracking application."""

        self.camera.open()

        print()
        print("=" * 60)
        print(
            "RetailEdge AI - Shopper Tracking"
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
        print()
        print(
            "Press Q to quit."
        )
        print("=" * 60)
        print()

        try:

            while True:

                # ------------------------------------------------
                # Get frame through CameraManager.
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
                # Draw tracking results.
                # ------------------------------------------------

                tracked_count = (
                    self.draw_tracks(
                        frame,
                        result,
                    )
                )

                # ------------------------------------------------
                # FPS.
                # ------------------------------------------------

                fps = (
                    self.calculate_fps()
                )

                # ------------------------------------------------
                # Status.
                # ------------------------------------------------

                self.draw_status(
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
                "Tracking interrupted."
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
            "RetailEdge tracking stopped."
        )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main() -> None:
    """Application entry point."""

    config = TrackingConfig()

    tracker = ShopperTracker(
        config
    )

    tracker.run()


if __name__ == "__main__":
    main()