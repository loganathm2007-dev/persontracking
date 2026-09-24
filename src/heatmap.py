from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from camera import CameraConfig, CameraManager


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class HeatmapConfig:
    """Configuration for shopper movement heatmap."""

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
    # Heatmap
    # --------------------------------------------------------

    # How quickly old activity fades.
    decay: float = 0.995

    # Gaussian blur applied to the accumulated heatmap.
    blur_kernel_size: int = 51

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    window_name: str = (
        "RetailEdge AI - Shopper Heatmap"
    )


# ============================================================
# SHOPPER HEATMAP
# ============================================================

class ShopperHeatmap:
    """
    Real-time shopper movement heatmap.

    Pipeline:

        CameraManager
             ↓
           Frame
             ↓
        YOLO + ByteTrack
             ↓
         Person centroid
             ↓
        Heatmap accumulation
             ↓
        Spatial activity visualization
    """

    def __init__(
        self,
        config: HeatmapConfig,
    ) -> None:

        self.config = config

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        self.model = YOLO(
            self.config.model_path
        )

        # ----------------------------------------------------
        # Camera
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
        # Heatmap matrix.
        #
        # Created after we know the actual frame size.
        # ----------------------------------------------------

        self.heatmap: np.ndarray | None = None

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
    # INITIALIZE HEATMAP
    # ========================================================

    def initialize_heatmap(
        self,
        frame,
    ) -> None:
        """Create heatmap matrix based on frame size."""

        height, width = (
            frame.shape[:2]
        )

        self.heatmap = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.float32,
        )

    # ========================================================
    # TRACK PEOPLE
    # ========================================================

    def track_people(
        self,
        frame,
    ):
        """Run YOLO + ByteTrack."""

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
    # UPDATE HEATMAP
    # ========================================================

    def update_heatmap(
        self,
        result,
    ) -> int:
        """
        Add current shopper positions to the heatmap.

        Returns:
            Number of tracked people.
        """

        if self.heatmap is None:

            return 0

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

        tracked_count = len(
            boxes
        )

        # ----------------------------------------------------
        # Fade older activity.
        # ----------------------------------------------------

        self.heatmap *= (
            self.config.decay
        )

        # ----------------------------------------------------
        # Add current shopper positions.
        # ----------------------------------------------------

        for box in boxes:

            x1, y1, x2, y2 = map(
                int,
                box,
            )

            center_x = (
                x1 + x2
            ) // 2

            center_y = (
                y1 + y2
            ) // 2

            # Make sure coordinates are inside frame.
            center_x = max(
                0,
                min(
                    center_x,
                    self.heatmap.shape[1] - 1,
                ),
            )

            center_y = max(
                0,
                min(
                    center_y,
                    self.heatmap.shape[0] - 1,
                ),
            )

            # Increase activity at the shopper location.
            self.heatmap[
                center_y,
                center_x,
            ] += 1.0

        return tracked_count

    # ========================================================
    # CREATE HEATMAP IMAGE
    # ========================================================

    def create_heatmap_overlay(
        self,
        frame,
    ):
        """
        Convert the accumulated heatmap into
        a visual overlay.
        """

        if self.heatmap is None:

            return frame

        # ----------------------------------------------------
        # Blur activity.
        # ----------------------------------------------------

        kernel = (
            self.config.blur_kernel_size
        )

        # Ensure odd kernel size.
        if kernel % 2 == 0:

            kernel += 1

        blurred = cv2.GaussianBlur(
            self.heatmap,
            (
                kernel,
                kernel,
            ),
            0,
        )

        # ----------------------------------------------------
        # Normalize to 0-255.
        # ----------------------------------------------------

        normalized = cv2.normalize(
            blurred,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )

        normalized = (
            normalized.astype(
                np.uint8
            )
        )

        # ----------------------------------------------------
        # Apply OpenCV heatmap color mapping.
        # ----------------------------------------------------

        colored_heatmap = cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_JET,
        )

        # ----------------------------------------------------
        # Overlay on original frame.
        # ----------------------------------------------------

        overlay = cv2.addWeighted(
            frame,
            0.60,
            colored_heatmap,
            0.40,
            0,
        )

        return overlay

    # ========================================================
    # DRAW PEOPLE
    # ========================================================

    def draw_people(
        self,
        frame,
        result,
    ) -> int:
        """Draw current tracked shoppers."""

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
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return len(
            track_ids
        )

    # ========================================================
    # DASHBOARD
    # ========================================================

    def draw_dashboard(
        self,
        frame,
        tracked_count: int,
        fps: float,
    ) -> None:
        """Draw heatmap status."""

        cv2.rectangle(
            frame,
            (
                10,
                10,
            ),
            (
                330,
                100,
            ),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            (
                f"Tracked: "
                f"{tracked_count}"
            ),
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

        cv2.putText(
            frame,
            (
                f"FPS: "
                f"{fps:.1f}"
            ),
            (
                20,
                72,
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
        """Run real-time shopper heatmap."""

        self.camera.open()

        print()
        print("=" * 60)
        print(
            "RetailEdge AI - Shopper Heatmap"
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
        print()
        print(
            "Heatmap is built from shopper centroid"
        )
        print(
            "positions over time."
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

                # ------------------------------------------------
                # Initialize heatmap after first frame.
                # ------------------------------------------------

                if self.heatmap is None:

                    self.initialize_heatmap(
                        frame
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
                # Update heatmap.
                # ------------------------------------------------

                tracked_count = (
                    self.update_heatmap(
                        result
                    )
                )

                # ------------------------------------------------
                # Create heatmap visualization.
                # ------------------------------------------------

                display_frame = (
                    self.create_heatmap_overlay(
                        frame
                    )
                )

                # ------------------------------------------------
                # Draw people.
                # ------------------------------------------------

                self.draw_people(
                    display_frame,
                    result,
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
                    display_frame,
                    tracked_count,
                    fps,
                )

                # ------------------------------------------------
                # Display.
                # ------------------------------------------------

                cv2.imshow(
                    self.config.window_name,
                    display_frame,
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
                "Heatmap interrupted."
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
            "RetailEdge heatmap stopped."
        )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main() -> None:
    """Application entry point."""

    config = HeatmapConfig()

    heatmap = ShopperHeatmap(
        config
    )

    heatmap.run()


if __name__ == "__main__":
    main()