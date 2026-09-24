from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from camera import CameraConfig, CameraManager


# ============================================================
# ZONE CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class ZoneConfig:
    """
    Retail analytics zone.

    Coordinates are normalized:

        (0.0, 0.0) -> top-left
        (1.0, 1.0) -> bottom-right

    Using normalized coordinates means the same zone
    configuration can work across different resolutions.
    """

    name: str

    polygon: tuple[tuple[float, float], ...]


# ============================================================
# ANALYTICS CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class AnalyticsConfig:
    """
    Central configuration for the RetailEdge AI
    shopper intelligence pipeline.
    """

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    model_path: str = "yolo11n.pt"

    person_class_id: int = 0

    confidence_threshold: float = 0.40

    iou_threshold: float = 0.50

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    tracker_config: str = (
        "src/botsort_retail.yaml"
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    camera_source: int | str = 0

    camera_width: int = 1280

    camera_height: int = 720

    flip_horizontal: bool = True

    # --------------------------------------------------------
    # Track state
    # --------------------------------------------------------

    trajectory_length: int = 90

    track_timeout_seconds: float = 2.5

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    heatmap_decay: float = 0.995

    heatmap_blur_kernel: int = 51

    heatmap_strength: float = 0.45

    heatmap_point_radius: int = 18

    # --------------------------------------------------------
    # Behavioral analytics
    # --------------------------------------------------------

    long_dwell_seconds: float = 30.0

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    window_name: str = (
        "RetailEdge AI - Unified Shopper Analytics"
    )


# ============================================================
# TRACK STATE
# ============================================================


@dataclass
class TrackState:
    """
    Runtime state for one anonymous tracked shopper.

    No face recognition or personally identifying
    information is stored.
    """

    track_id: int

    first_seen: float

    last_seen: float

    last_position: tuple[int, int]

    trajectory: deque[tuple[int, int]] = field(
        default_factory=deque
    )

    current_zone: Optional[str] = None

    zone_entered_at: Optional[float] = None

    total_dwell_by_zone: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )

    zone_visit_count: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )


# ============================================================
# SHOPPER ANALYTICS ENGINE
# ============================================================


class ShopperAnalyticsEngine:
    """
    Unified retail computer-vision pipeline.

    Architecture:

        Camera
           ↓
        CameraManager
           ↓
        YOLO
           ↓
        BoT-SORT
           ↓
        Anonymous Tracks
           ↓
        ┌──────────┬──────────┬──────────┐
        │          │          │          │
     Occupancy   Zones     Trajectory  Heatmap
        │          │          │
        │        Dwell        │
        │          │          │
        └──────────┴──────────┘
                   ↓
           Shopper Intelligence

    YOLO + tracking executes once per frame.
    All analytics consume the same track objects.
    """

    def __init__(
        self,
        config: AnalyticsConfig,
        zones: list[ZoneConfig],
    ) -> None:

        self.config = config

        self.zones = zones

        # ----------------------------------------------------
        # Load YOLO model once
        # ----------------------------------------------------

        print(
            f"Loading YOLO model: "
            f"{self.config.model_path}"
        )

        self.model = YOLO(
            self.config.model_path
        )

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        self.camera = CameraManager(
            CameraConfig(
                source=self.config.camera_source,
                width=self.config.camera_width,
                height=self.config.camera_height,
                use_directshow=True,
                flip_horizontal=self.config.flip_horizontal,
            )
        )

        # ----------------------------------------------------
        # Track state
        # ----------------------------------------------------

        self.tracks: dict[int, TrackState] = {}

        self.seen_track_ids: set[int] = set()

        # ----------------------------------------------------
        # Zone state
        # ----------------------------------------------------

        self.zone_occupancy: dict[str, int] = {
            zone.name: 0
            for zone in self.zones
        }

        self.zone_visitors: dict[str, set[int]] = {
            zone.name: set()
            for zone in self.zones
        }

        self.zone_transitions: list[dict] = []

        # ----------------------------------------------------
        # Behavioral events
        # ----------------------------------------------------

        self.long_dwell_events: list[dict] = []

        # ----------------------------------------------------
        # Heatmap
        # ----------------------------------------------------

        self.heatmap: Optional[np.ndarray] = None

        # ----------------------------------------------------
        # Performance
        # ----------------------------------------------------

        self.frame_count = 0

        self.fps = 0.0

        self.inference_ms = 0.0

        self._fps_frame_count = 0

        self._fps_start_time = (
            time.perf_counter()
        )

    # ========================================================
    # FPS
    # ========================================================

    def update_fps(self) -> None:

        self._fps_frame_count += 1

        now = time.perf_counter()

        elapsed = (
            now
            - self._fps_start_time
        )

        if elapsed >= 1.0:

            self.fps = (
                self._fps_frame_count
                / elapsed
            )

            self._fps_frame_count = 0

            self._fps_start_time = now

    # ========================================================
    # HEATMAP INITIALIZATION
    # ========================================================

    def initialize_heatmap(
        self,
        frame: np.ndarray,
    ) -> None:

        height, width = frame.shape[:2]

        self.heatmap = np.zeros(
            (height, width),
            dtype=np.float32,
        )

    # ========================================================
    # HEATMAP FRAME DECAY
    # ========================================================

    def decay_heatmap(self) -> None:
        """
        Apply temporal decay exactly once per frame.

        This is important.

        We do NOT decay once per detected person.
        """

        if self.heatmap is None:
            return

        self.heatmap *= (
            self.config.heatmap_decay
        )

    # ========================================================
    # HEATMAP UPDATE
    # ========================================================

    def add_heatmap_point(
        self,
        position: tuple[int, int],
    ) -> None:
        """
        Add movement energy for one tracked shopper.

        Decay is deliberately NOT performed here.
        """

        if self.heatmap is None:
            return

        x, y = position

        height, width = (
            self.heatmap.shape
        )

        if not (
            0 <= x < width
            and 0 <= y < height
        ):
            return

        cv2.circle(
            self.heatmap,
            (x, y),
            self.config.heatmap_point_radius,
            1.0,
            -1,
        )

    # ========================================================
    # HEATMAP VISUALIZATION
    # ========================================================

    def create_heatmap_overlay(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:

        if self.heatmap is None:
            return frame

        heatmap = self.heatmap.copy()

        kernel = (
            self.config.heatmap_blur_kernel
        )

        if kernel % 2 == 0:
            kernel += 1

        heatmap = cv2.GaussianBlur(
            heatmap,
            (kernel, kernel),
            0,
        )

        maximum = float(
            np.max(heatmap)
        )

        if maximum <= 0.0:
            return frame

        normalized = (
            heatmap
            / maximum
            * 255.0
        )

        normalized = np.clip(
            normalized,
            0,
            255,
        ).astype(np.uint8)

        color_heatmap = cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_JET,
        )

        return cv2.addWeighted(
            frame,
            1.0
            - self.config.heatmap_strength,
            color_heatmap,
            self.config.heatmap_strength,
            0,
        )

    # ========================================================
    # YOLO + BOT-SORT
    # ========================================================

    def run_tracking(
        self,
        frame: np.ndarray,
    ):
        """
        Execute exactly one detection + tracking pass.
        """

        start_time = (
            time.perf_counter()
        )

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.config.tracker_config,
            classes=[
                self.config.person_class_id
            ],
            conf=(
                self.config
                .confidence_threshold
            ),
            iou=(
                self.config
                .iou_threshold
            ),
            verbose=False,
        )

        end_time = (
            time.perf_counter()
        )

        self.inference_ms = (
            end_time
            - start_time
        ) * 1000.0

        return results

    # ========================================================
    # NORMALIZED POLYGON
    # ========================================================

    def normalized_polygon(
        self,
        zone: ZoneConfig,
        width: int,
        height: int,
    ) -> np.ndarray:

        points = []

        for x, y in zone.polygon:

            pixel_x = int(
                x * width
            )

            pixel_y = int(
                y * height
            )

            points.append(
                [
                    pixel_x,
                    pixel_y,
                ]
            )

        return np.array(
            points,
            dtype=np.int32,
        )

    # ========================================================
    # FIND ZONE
    # ========================================================

    def find_zone(
        self,
        position: tuple[int, int],
        frame_shape: tuple[int, ...],
    ) -> Optional[str]:

        height, width = (
            frame_shape[:2]
        )

        x, y = position

        for zone in self.zones:

            polygon = (
                self.normalized_polygon(
                    zone,
                    width,
                    height,
                )
            )

            inside = cv2.pointPolygonTest(
                polygon,
                (
                    float(x),
                    float(y),
                ),
                False,
            )

            if inside >= 0:

                return zone.name

        return None

    # ========================================================
    # EXTRACT TRACKS
    # ========================================================

    def extract_tracks(
        self,
        results,
        frame_shape: tuple[int, ...],
    ) -> list[dict]:

        detections: list[dict] = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        boxes = result.boxes

        if boxes.id is None:
            return detections

        xyxy = (
            boxes.xyxy
            .cpu()
            .numpy()
        )

        confidences = (
            boxes.conf
            .cpu()
            .numpy()
        )

        track_ids = (
            boxes.id
            .int()
            .cpu()
            .tolist()
        )

        for (
            box,
            confidence,
            track_id,
        ) in zip(
            xyxy,
            confidences,
            track_ids,
        ):

            x1, y1, x2, y2 = map(
                int,
                box,
            )

            center_x = int(
                (x1 + x2)
                / 2
            )

            center_y = int(
                (y1 + y2)
                / 2
            )

            center = (
                center_x,
                center_y,
            )

            detections.append(
                {
                    "track_id": int(
                        track_id
                    ),
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    "center": center,
                    "confidence": float(
                        confidence
                    ),
                    "zone": self.find_zone(
                        center,
                        frame_shape,
                    ),
                }
            )

        return detections

    # ========================================================
    # UPDATE TRACK STATE
    # ========================================================

    def update_track_state(
        self,
        detection: dict,
        now: float,
    ) -> None:

        track_id = (
            detection["track_id"]
        )

        position = (
            detection["center"]
        )

        zone = (
            detection["zone"]
        )

        # ----------------------------------------------------
        # NEW TRACK
        # ----------------------------------------------------

        if track_id not in self.tracks:

            state = TrackState(
                track_id=track_id,
                first_seen=now,
                last_seen=now,
                last_position=position,
            )

            state.trajectory = deque(
                maxlen=(
                    self.config
                    .trajectory_length
                )
            )

            state.trajectory.append(
                position
            )

            state.current_zone = zone

            if zone is not None:

                state.zone_entered_at = (
                    now
                )

                state.zone_visit_count[
                    zone
                ] += 1

                self.zone_visitors[
                    zone
                ].add(
                    track_id
                )

            self.tracks[
                track_id
            ] = state

            self.seen_track_ids.add(
                track_id
            )

            return

        # ----------------------------------------------------
        # EXISTING TRACK
        # ----------------------------------------------------

        state = self.tracks[
            track_id
        ]

        previous_zone = (
            state.current_zone
        )

        state.last_seen = now

        state.last_position = position

        state.trajectory.append(
            position
        )

        # ----------------------------------------------------
        # ZONE TRANSITION
        # ----------------------------------------------------

        if (
            zone
            != previous_zone
        ):

            self.handle_zone_transition(
                state=state,
                previous_zone=(
                    previous_zone
                ),
                new_zone=zone,
                now=now,
            )

        state.current_zone = zone

        # ----------------------------------------------------
        # LONG DWELL
        # ----------------------------------------------------

        if (
            zone is not None
            and state.zone_entered_at
            is not None
        ):

            dwell = (
                now
                - state.zone_entered_at
            )

            if (
                dwell
                >= self.config
                .long_dwell_seconds
            ):

                self.register_long_dwell(
                    track_id=track_id,
                    zone=zone,
                    dwell_seconds=dwell,
                    timestamp=now,
                )

    # ========================================================
    # LONG DWELL EVENT
    # ========================================================

    def register_long_dwell(
        self,
        track_id: int,
        zone: str,
        dwell_seconds: float,
        timestamp: float,
    ) -> None:

        for event in (
            self.long_dwell_events
        ):

            if (
                event["track_id"]
                == track_id
                and event["zone"]
                == zone
            ):

                return

        self.long_dwell_events.append(
            {
                "timestamp": timestamp,
                "track_id": track_id,
                "zone": zone,
                "dwell_seconds": (
                    dwell_seconds
                ),
            }
        )

    # ========================================================
    # ZONE TRANSITION
    # ========================================================

    def handle_zone_transition(
        self,
        state: TrackState,
        previous_zone: Optional[str],
        new_zone: Optional[str],
        now: float,
    ) -> None:

        track_id = (
            state.track_id
        )

        # ----------------------------------------------------
        # CLOSE PREVIOUS ZONE
        # ----------------------------------------------------

        if (
            previous_zone is not None
            and state.zone_entered_at
            is not None
        ):

            dwell = (
                now
                - state.zone_entered_at
            )

            state.total_dwell_by_zone[
                previous_zone
            ] += max(
                0.0,
                dwell,
            )

        # ----------------------------------------------------
        # ENTER NEW ZONE
        # ----------------------------------------------------

        if new_zone is not None:

            state.zone_entered_at = (
                now
            )

            state.zone_visit_count[
                new_zone
            ] += 1

            self.zone_visitors[
                new_zone
            ].add(
                track_id
            )

        else:

            state.zone_entered_at = (
                None
            )

        # ----------------------------------------------------
        # EVENT
        # ----------------------------------------------------

        self.zone_transitions.append(
            {
                "timestamp": now,
                "track_id": track_id,
                "from_zone": (
                    previous_zone
                ),
                "to_zone": new_zone,
            }
        )

    # ========================================================
    # LOST TRACK CLEANUP
    # ========================================================

    def cleanup_lost_tracks(
        self,
        active_ids: set[int],
        now: float,
    ) -> None:

        timeout = (
            self.config
            .track_timeout_seconds
        )

        for (
            track_id,
            state,
        ) in list(
            self.tracks.items()
        ):

            if (
                track_id
                in active_ids
            ):
                continue

            if (
                now
                - state.last_seen
                < timeout
            ):
                continue

            # Close active dwell session.
            if (
                state.current_zone
                is not None
                and state.zone_entered_at
                is not None
            ):

                dwell = (
                    state.last_seen
                    - state.zone_entered_at
                )

                state.total_dwell_by_zone[
                    state.current_zone
                ] += max(
                    0.0,
                    dwell,
                )

            del self.tracks[
                track_id
            ]

    # ========================================================
    # ZONE OCCUPANCY
    # ========================================================

    def calculate_zone_occupancy(
        self,
    ) -> None:

        occupancy = {
            zone.name: 0
            for zone in self.zones
        }

        for state in (
            self.tracks.values()
        ):

            if (
                state.current_zone
                is not None
            ):

                occupancy[
                    state.current_zone
                ] += 1

        self.zone_occupancy = (
            occupancy
        )

    # ========================================================
    # CURRENT OCCUPANCY
    # ========================================================

    def current_occupancy(
        self,
    ) -> int:

        return len(
            self.tracks
        )

    # ========================================================
    # UNIQUE VISITORS
    # ========================================================

    def unique_visitors(
        self,
    ) -> int:

        return len(
            self.seen_track_ids
        )

    # ========================================================
    # CURRENT DWELL
    # ========================================================

    def get_current_dwell(
        self,
        state: TrackState,
        now: float,
    ) -> float:

        if (
            state.current_zone
            is None
            or state.zone_entered_at
            is None
        ):

            return 0.0

        return max(
            0.0,
            now
            - state.zone_entered_at,
        )

    # ========================================================
    # ZONE DWELL SUMMARY
    # ========================================================

    def get_zone_dwell_summary(
        self,
        now: float,
    ) -> dict[str, float]:

        totals = {
            zone.name: 0.0
            for zone in self.zones
        }

        for state in (
            self.tracks.values()
        ):

            for (
                zone,
                dwell,
            ) in (
                state
                .total_dwell_by_zone
                .items()
            ):

                totals[zone] += dwell

            if (
                state.current_zone
                is not None
                and state.zone_entered_at
                is not None
            ):

                totals[
                    state.current_zone
                ] += (
                    self.get_current_dwell(
                        state,
                        now,
                    )
                )

        return totals

    # ========================================================
    # DRAW ZONES
    # ========================================================

    def draw_zones(
        self,
        frame: np.ndarray,
    ) -> None:

        height, width = (
            frame.shape[:2]
        )

        for zone in self.zones:

            polygon = (
                self.normalized_polygon(
                    zone,
                    width,
                    height,
                )
            )

            cv2.polylines(
                frame,
                [polygon],
                isClosed=True,
                color=(255, 180, 0),
                thickness=2,
            )

            x, y = (
                polygon[0]
            )

            occupancy = (
                self.zone_occupancy[
                    zone.name
                ]
            )

            label = (
                f"{zone.name}: "
                f"{occupancy}"
            )

            cv2.putText(
                frame,
                label,
                (
                    x + 8,
                    y + 25,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 180, 0),
                2,
            )

    # ========================================================
    # DRAW TRACKS
    # ========================================================

    def draw_tracks(
        self,
        frame: np.ndarray,
        detections: list[dict],
        now: float,
    ) -> None:

        for detection in detections:

            track_id = (
                detection[
                    "track_id"
                ]
            )

            x1, y1, x2, y2 = (
                detection["bbox"]
            )

            center_x, center_y = (
                detection["center"]
            )

            confidence = (
                detection[
                    "confidence"
                ]
            )

            zone = (
                detection["zone"]
            )

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # ------------------------------------------------
            # Centroid
            # ------------------------------------------------

            cv2.circle(
                frame,
                (
                    center_x,
                    center_y,
                ),
                5,
                (0, 255, 255),
                -1,
            )

            # ------------------------------------------------
            # Trajectory
            # ------------------------------------------------

            state = self.tracks.get(
                track_id
            )

            if state is not None:

                points = list(
                    state.trajectory
                )

                for index in range(
                    1,
                    len(points),
                ):

                    cv2.line(
                        frame,
                        points[
                            index - 1
                        ],
                        points[index],
                        (255, 0, 255),
                        2,
                    )

            # ------------------------------------------------
            # Dwell
            # ------------------------------------------------

            dwell = 0.0

            if state is not None:

                dwell = (
                    self.get_current_dwell(
                        state,
                        now,
                    )
                )

            label = (
                f"ID {track_id} "
                f"{confidence:.2f}"
            )

            if zone is not None:

                label += (
                    f" | {zone}"
                )

            if dwell > 0:

                label += (
                    f" | {dwell:.1f}s"
                )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        25,
                        y1 - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

    # ========================================================
    # DASHBOARD
    # ========================================================

    def draw_dashboard(
        self,
        frame: np.ndarray,
    ) -> None:

        height, width = (
            frame.shape[:2]
        )

        dashboard_height = 145

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (
                width,
                dashboard_height,
            ),
            (20, 20, 20),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.70,
            frame,
            0.30,
            0,
            frame,
        )

        occupancy = (
            self.current_occupancy()
        )

        unique = (
            self.unique_visitors()
        )

        lines = [
            (
                f"FPS: {self.fps:.1f}"
                f"   Inference: "
                f"{self.inference_ms:.1f} ms"
            ),
            (
                f"Current Shoppers: "
                f"{occupancy}"
            ),
            (
                f"Unique Visitors: "
                f"{unique}"
            ),
        ]

        y = 30

        for text in lines:

            cv2.putText(
                frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            y += 30

        zone_text = " | ".join(
            (
                f"{name}: {count}"
            )
            for name, count
            in self.zone_occupancy.items()
        )

        cv2.putText(
            frame,
            zone_text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            2,
        )

    # ========================================================
    # PROCESS ONE FRAME
    # ========================================================

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:

        now = time.monotonic()

        # ----------------------------------------------------
        # 1. Decay heatmap ONCE per frame
        # ----------------------------------------------------

        self.decay_heatmap()

        # ----------------------------------------------------
        # 2. ONE YOLO + BoT-SORT inference
        # ----------------------------------------------------

        results = self.run_tracking(
            frame
        )

        # ----------------------------------------------------
        # 3. Extract tracks
        # ----------------------------------------------------

        detections = (
            self.extract_tracks(
                results,
                frame.shape,
            )
        )

        active_ids = {
            detection[
                "track_id"
            ]
            for detection in detections
        }

        # ----------------------------------------------------
        # 4. Update track state
        # ----------------------------------------------------

        for detection in detections:

            self.update_track_state(
                detection,
                now,
            )

            self.add_heatmap_point(
                detection[
                    "center"
                ]
            )

        # ----------------------------------------------------
        # 5. Remove stale tracks
        # ----------------------------------------------------

        self.cleanup_lost_tracks(
            active_ids,
            now,
        )

        # ----------------------------------------------------
        # 6. Calculate zone occupancy
        # ----------------------------------------------------

        self.calculate_zone_occupancy()

        # ----------------------------------------------------
        # 7. Create heatmap overlay
        # ----------------------------------------------------

        output = (
            self.create_heatmap_overlay(
                frame
            )
        )

        # ----------------------------------------------------
        # 8. Draw zones
        # ----------------------------------------------------

        self.draw_zones(
            output
        )

        # ----------------------------------------------------
        # 9. Draw people / trajectories
        # ----------------------------------------------------

        self.draw_tracks(
            output,
            detections,
            now,
        )

        # ----------------------------------------------------
        # 10. Draw dashboard
        # ----------------------------------------------------

        self.draw_dashboard(
            output
        )

        return output

    # ========================================================
    # MAIN LOOP
    # ========================================================

    def run(self) -> None:

        self.camera.open()

        print()
        print("=" * 70)
        print(
            "RetailEdge AI - Unified Shopper Analytics"
        )
        print("=" * 70)

        print(
            f"Model: "
            f"{self.config.model_path}"
        )

        print(
            f"Tracker: "
            f"{self.config.tracker_config}"
        )

        print()
        print(
            "Analytics modules:"
        )

        print(
            "  [OK] Person detection"
        )

        print(
            "  [OK] BoT-SORT tracking"
        )

        print(
            "  [OK] Occupancy"
        )

        print(
            "  [OK] Zone analytics"
        )

        print(
            "  [OK] Dwell time"
        )

        print(
            "  [OK] Trajectory"
        )

        print(
            "  [OK] Movement heatmap"
        )

        print(
            "  [OK] Long-dwell events"
        )

        print()
        print(
            "Press Q to quit."
        )

        print("=" * 70)

        try:

            while True:

                success, frame = (
                    self.camera.read()
                )

                if not success:

                    print(
                        "Warning: failed to "
                        "read camera frame."
                    )

                    break

                self.frame_count += 1

                # Initialize heatmap after the first
                # successfully captured frame.
                if (
                    self.heatmap
                    is None
                ):

                    self.initialize_heatmap(
                        frame
                    )

                output = (
                    self.process_frame(
                        frame
                    )
                )

                self.update_fps()

                cv2.imshow(
                    self.config.window_name,
                    output,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("q"):

                    break

        finally:

            self.shutdown()

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(self) -> None:

        self.camera.release()

        cv2.destroyAllWindows()

        print()
        print("=" * 70)
        print(
            "Unified Analytics Shutdown"
        )
        print("=" * 70)

        print(
            f"Frames processed: "
            f"{self.frame_count}"
        )

        print(
            f"Unique visitors: "
            f"{len(self.seen_track_ids)}"
        )

        print(
            f"Current occupancy: "
            f"{self.current_occupancy()}"
        )

        print(
            f"Final FPS: "
            f"{self.fps:.2f}"
        )

        print(
            f"Final inference latency: "
            f"{self.inference_ms:.2f} ms"
        )

        print(
            f"Zone transitions: "
            f"{len(self.zone_transitions)}"
        )

        print(
            f"Long-dwell events: "
            f"{len(self.long_dwell_events)}"
        )

        print("=" * 70)


# ============================================================
# DEFAULT RETAIL ZONES
# ============================================================


def create_default_zones() -> list[ZoneConfig]:
    """
    Temporary demonstration zones.

    These MUST eventually be calibrated against the
    actual supermarket CCTV camera view.
    """

    return [
        ZoneConfig(
            name="Zone A",
            polygon=(
                (0.00, 0.00),
                (0.50, 0.00),
                (0.50, 0.50),
                (0.00, 0.50),
            ),
        ),

        ZoneConfig(
            name="Zone B",
            polygon=(
                (0.00, 0.50),
                (0.50, 0.50),
                (0.50, 1.00),
                (0.00, 1.00),
            ),
        ),

        ZoneConfig(
            name="Zone C",
            polygon=(
                (0.50, 0.00),
                (1.00, 0.00),
                (1.00, 1.00),
                (0.50, 1.00),
            ),
        ),
    ]


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================


def main() -> None:

    config = AnalyticsConfig(

        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        model_path="yolo11n.pt",

        person_class_id=0,

        confidence_threshold=0.40,

        iou_threshold=0.50,

        # ----------------------------------------------------
        # Custom Retail BoT-SORT
        # ----------------------------------------------------

        tracker_config=(
            "src/botsort_retail.yaml"
        ),

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        camera_source=0,

        camera_width=1280,

        camera_height=720,

        flip_horizontal=True,

        # ----------------------------------------------------
        # Tracking
        # ----------------------------------------------------

        trajectory_length=90,

        track_timeout_seconds=2.5,

        # ----------------------------------------------------
        # Heatmap
        # ----------------------------------------------------

        heatmap_decay=0.995,

        heatmap_blur_kernel=51,

        heatmap_strength=0.45,

        heatmap_point_radius=18,

        # ----------------------------------------------------
        # Behavior
        # ----------------------------------------------------

        long_dwell_seconds=30.0,
    )

    zones = (
        create_default_zones()
    )

    engine = ShopperAnalyticsEngine(
        config=config,
        zones=zones,
    )

    engine.run()


if __name__ == "__main__":
    main()