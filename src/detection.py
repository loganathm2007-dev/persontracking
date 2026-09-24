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
class DetectionConfig:
    """Configuration for the person-detection pipeline."""

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

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    window_name: str = (
        "RetailEdge AI - Person Detection"
    )


# ============================================================
# PERSON DETECTOR
# ============================================================

class PersonDetector:
    """
    Real-time person detection using YOLO.

    Responsibilities:

        - YOLO model management
        - Person detection
        - Bounding-box rendering
        - FPS measurement
        - Camera-independent frame processing

    Camera access is handled by CameraManager.
    """

    def __init__(
        self,
        config: DetectionConfig,
    ) -> None:

        self.config = config

        # ----------------------------------------------------
        # Load YOLO once.
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
    # DETECTION
    # ========================================================

    def detect(self, frame):
        """
        Run YOLO inference on one frame.

        Returns:
            YOLO result object.
        """

        results = self.model(
            source=frame,
            classes=[
                self.config.person_class_id
            ],
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            verbose=False,
        )

        return results[0]

    # ========================================================
    # DRAW DETECTIONS
    # ========================================================

    def draw_detections(
        self,
        frame,
        result,
    ) -> int:
        """
        Draw detected people.

        Returns:
            Number of detected people.
        """

        if result.boxes is None:

            return 0

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .tolist()
        )

        detected_people = len(
            boxes
        )

        for (
            box,
            confidence,
        ) in zip(
            boxes,
            confidences,
        ):

            x1, y1, x2, y2 = map(
                int,
                box,
            )

            # Bounding box.
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

            # Label.
            label = (
                f"Person | "
                f"{confidence:.2f}"
            )

            label_y = max(
                y1 - 10,
                25,
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    label_y,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return detected_people

    # ========================================================
    # STATUS OVERLAY
    # ========================================================

    def draw_status(
        self,
        frame,
        detected_people: int,
        fps: float,
    ) -> None:
        """Render detection status."""

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
                f"People detected: "
                f"{detected_people}"
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
        """Run real-time person detection."""

        self.camera.open()

        print()
        print("=" * 60)
        print(
            "RetailEdge AI - Person Detection"
        )
        print("=" * 60)
        print(
            f"Model:       "
            f"{self.config.model_path}"
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
                # Read frame through CameraManager.
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
                # YOLO inference.
                # ------------------------------------------------

                result = self.detect(
                    frame
                )

                # ------------------------------------------------
                # Draw detections.
                # ------------------------------------------------

                detected_people = (
                    self.draw_detections(
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
                    detected_people,
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
                "Detection interrupted."
            )

        finally:

            self.shutdown()

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(self) -> None:
        """Release all resources."""

        self.camera.release()

        cv2.destroyAllWindows()

        print()
        print(
            "RetailEdge detection stopped."
        )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main() -> None:
    """Application entry point."""

    config = DetectionConfig()

    detector = PersonDetector(
        config
    )

    detector.run()


if __name__ == "__main__":
    main()