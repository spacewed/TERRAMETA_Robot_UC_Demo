#!/usr/bin/env python3
"""Robot-local RealSense viewer and AI runner with no frame transmission.

Run this on the robot when the cameras, OpenGL viewer, YOLO workers, pose
worker, and VLM GUI features should all live in one process:

    python3 On_Robot/local_ai_viewer.py

The file reuses On_Robot/sender.py for RealSense capture/recovery and
On_Receiver/receiver.py for rendering plus AI workers. It intentionally does
not create TCP camera senders or UDP latency probes.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = REPO_ROOT / "On_Robot"
RECEIVER_DIR = REPO_ROOT / "On_Receiver"
for module_dir in (ROBOT_DIR, RECEIVER_DIR):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import receiver as viewer
import sender as robot_sender


class LocalCameraCaptureWorker(threading.Thread):
    """Capture one RealSense camera directly into the local viewer frame source."""

    def __init__(
        self,
        camera_id: int,
        camera: robot_sender.RealSenseCamera,
        frame_source: "LocalFrameSource",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"LocalCameraCapture-{camera_id}", daemon=True)
        self._camera_id = camera_id
        self._camera = camera
        self._frame_source = frame_source
        self._stop_event = stop_event
        self._sequence = 0

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    self._camera.start()
                    captured = self._camera.capture()
                except Exception as exc:
                    self._frame_source.set_camera_connected(self._camera_id, False)
                    logging.warning(
                        "Camera %d (%s) local capture failed: %s",
                        self._camera_id,
                        self._camera.serial,
                        exc,
                    )
                    if (
                        robot_sender.RECOVER_ON_CAPTURE_FAILURE
                        and self._camera.can_recover_now()
                    ):
                        self._camera.recover()
                    self._stop_event.wait(robot_sender.RECONNECT_AFTER_ERROR_DELAY)
                    continue

                self._sequence += 1
                frame = viewer.CameraFrame(
                    camera_id=self._camera_id,
                    camera_serial=self._camera.serial,
                    sequence=self._sequence,
                    capture_received_time_ns=int(captured["capture_received_time_ns"]),
                    depth_frame_number=int(captured["depth_frame_number"]),
                    color_frame_number=int(captured["color_frame_number"]),
                    depth_device_timestamp_ms=float(
                        captured["depth_device_timestamp_ms"]
                    ),
                    color_device_timestamp_ms=float(
                        captured["color_device_timestamp_ms"]
                    ),
                    depth=captured["depth"],
                    color=captured["color"],
                )
                byte_count = int(frame.depth.nbytes + frame.color.nbytes)
                self._frame_source.store_camera_frame(frame, byte_count)
        finally:
            self._frame_source.set_camera_connected(self._camera_id, False)
            self._camera.stop()


class LocalFrameSource(threading.Thread):
    """FrameReceiver-compatible source backed by local RealSense capture."""

    def __init__(
        self,
        cameras: Sequence[robot_sender.RealSenseCamera],
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="LocalFrameSource", daemon=True)
        self._cameras = list(cameras)
        self._stop_event = stop_event
        state_count = max(len(self._cameras), len(viewer.CAMERA_ROTATIONS))
        self._camera_states = [
            viewer.ReceiverCameraState() for _ in range(state_count)
        ]
        self._frame_lock = threading.Lock()
        self._camera_fps_last_sample_s = time.time()
        self._latency_metrics = viewer.LatencyJitterMetrics()
        self._workers: List[LocalCameraCaptureWorker] = [
            LocalCameraCaptureWorker(camera_id, camera, self, stop_event)
            for camera_id, camera in enumerate(self._cameras)
        ]

    def run(self) -> None:
        serials = tuple(camera.serial for camera in self._cameras)
        if not serials:
            logging.error("No local RealSense camera serials are configured")
            self._stop_event.set()
            return

        if robot_sender.STARTUP_HARD_RESET:
            robot_sender.hardware_reset_all(
                serials,
                reset_delay=robot_sender.STARTUP_RESET_DELAY,
            )

        try:
            visible_serials = robot_sender.query_connected_serials(
                robot_sender.rs.context()
            )
        except Exception as exc:
            logging.warning("Unable to query local cameras at startup: %s", exc)
            visible_serials = set()
        missing_serials = set(serials) - visible_serials
        if missing_serials:
            logging.warning(
                "Configured local cameras missing at startup; workers will retry: %s",
                sorted(missing_serials),
            )

        for worker in self._workers:
            worker.start()

        try:
            while not self._stop_event.wait(robot_sender.CAMERA_METRICS_LOG_INTERVAL):
                self._log_camera_status()
        finally:
            logging.info("Local frame source shutting down")
            self._stop_event.set()
            for worker in self._workers:
                worker.join(timeout=robot_sender.FRAME_TIMEOUT_MS / 1000.0 + 2.0)
                if worker.is_alive():
                    logging.warning("%s did not stop cleanly", worker.name)
            if robot_sender.STARTUP_HARD_RESET:
                robot_sender.hardware_reset_all(serials, reset_delay=2.0)

    def store_camera_frame(self, frame: viewer.CameraFrame, byte_count: int) -> None:
        now_s = time.time()
        with self._frame_lock:
            if not 0 <= frame.camera_id < len(self._camera_states):
                return
            state = self._camera_states[frame.camera_id]
            if state.pending is not None:
                state.replaced_latest += 1
            state.pending = frame
            state.latest = frame
            state.serial = frame.camera_serial
            state.last_frame_time_s = now_s
            state.receive_count += 1
            state.bytes_received += byte_count
            state.connections = 1

    def set_camera_connected(self, camera_id: int, connected: bool) -> None:
        with self._frame_lock:
            if 0 <= camera_id < len(self._camera_states):
                self._camera_states[camera_id].connections = 1 if connected else 0

    def pop_pending_frames(self) -> Dict[int, viewer.CameraFrame]:
        with self._frame_lock:
            pending = {
                camera_id: state.pending
                for camera_id, state in enumerate(self._camera_states)
                if state.pending is not None
            }
            for state in self._camera_states:
                state.pending = None
        return pending  # type: ignore[return-value]

    def snapshot_latest_frames(
        self,
        camera_ids: Optional[Tuple[int, ...]] = None,
    ) -> Dict[int, viewer.CameraFrame]:
        allowed = set(camera_ids) if camera_ids is not None else None
        with self._frame_lock:
            return {
                camera_id: state.latest
                for camera_id, state in enumerate(self._camera_states)
                if state.latest is not None
                and (allowed is None or camera_id in allowed)
            }  # type: ignore[return-value]

    def active_camera_ids(self) -> Tuple[int, ...]:
        with self._frame_lock:
            return tuple(
                camera_id
                for camera_id, state in enumerate(self._camera_states)
                if state.connections > 0 or state.latest is not None
            )

    def stop(self) -> None:
        self._stop_event.set()

    def is_connected(self) -> bool:
        with self._frame_lock:
            return any(state.connections > 0 for state in self._camera_states)

    def latency_summary(self) -> Dict[str, Optional[float]]:
        return self._latency_metrics.summary()

    def record_latency_probe(self, round_trip_time_ns: int, receiver_time_ns: int) -> None:
        return None

    def write_metrics_result_if_due(self, now_s: Optional[float] = None) -> None:
        return None

    def take_average_camera_fps(self, now_s: float) -> float:
        elapsed_s = max(now_s - self._camera_fps_last_sample_s, 1e-9)
        with self._frame_lock:
            camera_counts = [
                state.receive_count
                for state in self._camera_states
                if state.connections > 0 or state.receive_count > 0
            ]
            for state in self._camera_states:
                state.receive_count = 0
                state.bytes_received = 0
                state.replaced_latest = 0
        self._camera_fps_last_sample_s = now_s
        if not camera_counts:
            return 0.0
        return (float(sum(camera_counts)) / len(camera_counts)) / elapsed_s

    def _log_camera_status(self) -> None:
        active_ids = self.active_camera_ids()
        logging.info(
            "Local camera source active cameras: %s",
            ", ".join(str(camera_id) for camera_id in active_ids) or "none",
        )


class LocalPointCloudApp(viewer.PointCloudApp):
    """PointCloudApp variant with local-source window title and metrics."""

    def _update_window_caption(self) -> None:
        mode = (
            "RGB"
            if self._visualization_mode == viewer.VISUALIZATION_RGB
            else "Heatmap"
        )
        det_status = "ALL" if self._object_detection_enabled else (
            "HUMAN" if self._detection_enabled else "OFF"
        )
        pose_status = "ON" if self._pose_enabled else "OFF"
        vlm_status = "ON" if self._vlm_enabled else "OFF"
        scene_desc_status = "ON" if self._scene_description_enabled else "OFF"
        viewer.pygame.display.set_caption(
            "TerraMeta Robot Local AI Viewer | "
            f"{mode} | Detection {det_status} | Pose {pose_status} | "
            f"VLM {vlm_status} | Scene {scene_desc_status}"
        )

    def _log_metrics(self) -> None:
        now = time.time()
        if now - self._last_metrics_time < viewer.BITRATE_LOG_INTERVAL:
            return

        capture_fps = self._receiver.take_average_camera_fps(now)
        active_ids = ()
        if hasattr(self._receiver, "active_camera_ids"):
            active_ids = self._receiver.active_camera_ids()
        logging.info(
            "Local capture FPS %.2f | Cameras %s",
            capture_fps,
            ", ".join(str(camera_id) for camera_id in active_ids) or "none",
        )
        self._last_metrics_time = now


def configure_logging() -> None:
    viewer.configure_logging()


def main() -> None:
    configure_logging()
    stop_event = threading.Event()
    metrics = viewer.Metrics()
    cameras = [
        robot_sender.RealSenseCamera(serial)
        for serial in robot_sender.CAMERA_SERIALS
    ]
    frame_source = LocalFrameSource(cameras, stop_event)
    frame_source.start()

    try:
        app = LocalPointCloudApp(frame_source, metrics, stop_event)
        app.run()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, shutting down local viewer")
        frame_source.stop()
    except Exception:
        logging.exception("Local AI viewer crashed unexpectedly")
        frame_source.stop()
    finally:
        frame_source.stop()
        frame_source.join(timeout=10.0)
        if frame_source.is_alive():
            logging.warning("Local frame source did not shut down cleanly")
        for camera in cameras:
            with contextlib.suppress(Exception):
                camera.stop()


if __name__ == "__main__":
    main()
