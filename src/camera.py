from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2


@dataclass(frozen=True)
class CameraConfig:
    """
    Configuration for a video source.

    source can be:
        0 = laptop / USB webcam
        "video.mp4" = recorded video
        "rtsp://..." = IP / CCTV camera

    flip_horizontal:
        True  -> mirror the video horizontally
        False -> keep the original camera orientation
    """

    source: int | str = 0

    width: int = 1280
    height: int = 720

    buffer_size: int = 1

    use_directshow: bool = True

    flip_horizontal: bool = True


class CameraManager:
    """
    Unified camera interface for RetailEdge AI.

    Supports:
        - Laptop webcam
        - USB webcam
        - Video files
        - RTSP CCTV/IP cameras

    The AI modules should receive frames from this class
    instead of accessing OpenCV cameras directly.
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config

        self.capture: Optional[cv2.VideoCapture] = None

        self.is_open: bool = False

    def open(self) -> None:
        """Open the configured video source."""

        source = self.config.source

        # -------------------------------------------------
        # Open webcam using DirectShow on Windows
        # -------------------------------------------------
        if isinstance(source, int):

            if self.config.use_directshow:
                self.capture = cv2.VideoCapture(
                    source,
                    cv2.CAP_DSHOW,
                )
            else:
                self.capture = cv2.VideoCapture(source)

        # -------------------------------------------------
        # Open video file / RTSP / other stream
        # -------------------------------------------------
        else:
            self.capture = cv2.VideoCapture(source)

        # -------------------------------------------------
        # Validate camera
        # -------------------------------------------------
        if self.capture is None or not self.capture.isOpened():

            raise RuntimeError(
                f"Unable to open camera source: {source}"
            )

        # -------------------------------------------------
        # Configure resolution
        # -------------------------------------------------
        self.capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.config.width,
        )

        self.capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.config.height,
        )

        # -------------------------------------------------
        # Reduce frame buffering
        # -------------------------------------------------
        self.capture.set(
            cv2.CAP_PROP_BUFFERSIZE,
            self.config.buffer_size,
        )

        self.is_open = True

        print(f"Camera source opened: {source}")

        print(
            f"Resolution: "
            f"{self.config.width}x{self.config.height}"
        )

        print(
            f"Horizontal flip: "
            f"{self.config.flip_horizontal}"
        )

    def read(self):
        """
        Read one frame from the camera.

        Returns:
            success: bool
            frame: OpenCV image
        """

        if self.capture is None or not self.is_open:

            raise RuntimeError(
                "Camera is not open."
            )

        success, frame = self.capture.read()

        if not success or frame is None:
            return False, None

        # -------------------------------------------------
        # Optional horizontal flip
        # -------------------------------------------------
        if self.config.flip_horizontal:

            frame = cv2.flip(
                frame,
                1,
            )

        return True, frame

    def is_available(self) -> bool:
        """Check whether the camera is currently available."""

        return (
            self.capture is not None
            and self.capture.isOpened()
            and self.is_open
        )

    def release(self) -> None:
        """Release the camera."""

        if self.capture is not None:

            self.capture.release()

            self.capture = None

        self.is_open = False

        print("Camera source released.")

    def __enter__(self):
        self.open()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:

        self.release()


def main() -> None:

    # -----------------------------------------------------
    # Laptop webcam configuration
    # -----------------------------------------------------
    config = CameraConfig(
        source=0,

        width=1280,
        height=720,

        use_directshow=True,

        # Mirror laptop webcam
        flip_horizontal=True,
    )

    camera = CameraManager(config)

    camera.open()

    print()
    print("=" * 55)
    print("RetailEdge AI - Camera Manager Test")
    print("=" * 55)

    print("Source:", config.source)

    print(
        "Resolution:",
        f"{config.width}x{config.height}",
    )

    print(
        "Horizontal Flip:",
        config.flip_horizontal,
    )

    print()
    print("Press Q to quit.")
    print("=" * 55)

    try:

        while True:

            success, frame = camera.read()

            if not success:

                print(
                    "Warning: failed to read "
                    "camera frame."
                )

                break

            cv2.imshow(
                "RetailEdge AI - Camera Test",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        camera.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()