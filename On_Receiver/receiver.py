#!/usr/bin/env python3
import collections
import contextlib
import csv
import ctypes
import logging
import os
import pickle
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import moderngl
import numpy as np
import pygame
from pygame.locals import (
    DOUBLEBUF,
    OPENGL,
    K_a,
    K_d,
    K_e,
    K_ESCAPE,
    K_h,
    K_l,
    K_LSHIFT,
    K_o,
    K_p,
    K_q,
    K_s,
    K_t,
    K_v,
    K_w,
)

from yolo_detector import YoloSegDetector
from yolo_pose_detector import YoloPosePersonDetector, PoseDetection2D
from vlm_detector import OpenAIVLMObjectDetector, VLMDetection2D, VLMDetectorError
from vlm_scene_describer import SceneDescription, SceneDescriptionWorker
from depth_to_3d import (
    PersonBBox3D,
    build_camera_intrinsics_from_fov,
    fit_bbox_only,
    fit_person_bbox,
)
from pose_to_3d import PersonPose3D, fit_pose_bbox
from bbox_renderer import BBoxRenderer

SERVER_HOST = ""
SERVER_PORT = 9999
SOCKET_RCVBUF = 2**24
SOCKET_BACKLOG = 8
MESSAGE_HEADER_STRUCT = "<L"  # payload length
MESSAGE_HEADER_SIZE = struct.calcsize(MESSAGE_HEADER_STRUCT)
BUFFERED_FRAME_MAGIC = b"TMF5"
BUFFERED_FRAME_HEADER_STRUCT = "<4sIH"  # magic, pickle length, buffer count
BUFFERED_FRAME_HEADER_SIZE = struct.calcsize(BUFFERED_FRAME_HEADER_STRUCT)
BUFFERED_FRAME_BUFFER_LENGTH_STRUCT = "<Q"
BUFFERED_FRAME_BUFFER_LENGTH_SIZE = struct.calcsize(BUFFERED_FRAME_BUFFER_LENGTH_STRUCT)
MAX_BUFFERED_FRAME_BUFFERS = 8
LATENCY_PROBE_PORT = 10000
# Probe v1 requests carry a sender monotonic timestamp plus the previous echo
# RTT. Probe v2 adds sequence and CLOCK_TAI timestamps for PTP-backed one-way
# latency while preserving v1 compatibility.
LATENCY_PROBE_MAGIC = b"TMLP"
LATENCY_PROBE_VERSION = 2
LATENCY_PROBE_REQUEST_STRUCT = "<QQ"
LATENCY_PROBE_REQUEST_SIZE = struct.calcsize(LATENCY_PROBE_REQUEST_STRUCT)
LATENCY_PROBE_ECHO_STRUCT = "<Q"
LATENCY_PROBE_REQUEST_V2_STRUCT = "<4sHHQQQQ"
LATENCY_PROBE_REQUEST_V2_SIZE = struct.calcsize(LATENCY_PROBE_REQUEST_V2_STRUCT)
LATENCY_PROBE_ECHO_V2_STRUCT = "<4sHHQQQQ"
LATENCY_PROBE_RECV_SIZE = 128
# Linux CPU isolation for the UDP probe. -1 reserves the highest allowed CPU;
# set to a specific CPU id to choose it, or None to disable affinity.
LATENCY_PROBE_CPU_CORE = -1
RESERVE_LATENCY_PROBE_CPU = True
LATENCY_PROBE_DSCP = int(os.environ.get("TM_PROBE_DSCP", "46"))
LATENCY_PROBE_SOCKET_PRIORITY = int(os.environ.get("TM_PROBE_PRIORITY", "6"))
LATENCY_PROBE_BIND_DEVICE = os.environ.get("TM_PROBE_BIND_DEVICE") or os.environ.get("TM_NET_IFACE")
LATENCY_PROBE_RT_PRIORITY = int(os.environ.get("TM_PROBE_RT_PRIORITY", "80"))
LATENCY_PROBE_MLOCK = os.environ.get("TM_PROBE_MLOCK", "1").lower() not in {"0", "false", "no"}
LATENCY_PROBE_BUSY_POLL_US = int(os.environ.get("TM_PROBE_BUSY_POLL_US", "50"))
LATENCY_PROBE_SOCKET_BUFFER = int(os.environ.get("TM_PROBE_SOCKET_BUFFER", str(1 << 20)))
LATENCY_PROBE_ONE_WAY_MAX_NS = int(os.environ.get("TM_PROBE_ONE_WAY_MAX_NS", "100000000"))
LATENCY_PROBE_ONE_WAY_RTT_FACTOR = float(os.environ.get("TM_PROBE_ONE_WAY_RTT_FACTOR", "2.0"))
LATENCY_PROBE_ONE_WAY_RTT_MARGIN_NS = int(
    os.environ.get("TM_PROBE_ONE_WAY_RTT_MARGIN_NS", "200000")
)
ACCEPT_TIMEOUT = 1.0
ACCEPT_RETRY_DELAY = 1.0
RECV_TIMEOUT = 5.0
BITRATE_LOG_INTERVAL = 1.0
METRICS_RESULT_INTERVAL = 30.0
METRICS_RESULTS_PATH = Path(__file__).with_name("receiver_30s_metrics.csv")
PAYLOAD_VERSION = 3
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

SO_BUSY_POLL = getattr(socket, "SO_BUSY_POLL", 46)
SO_PREFER_BUSY_POLL = getattr(socket, "SO_PREFER_BUSY_POLL", 69)
SO_INCOMING_CPU = getattr(socket, "SO_INCOMING_CPU", 49)
MCL_CURRENT = 1
MCL_FUTURE = 2

WINDOW_SIZE = (1600, 900)
TARGET_FPS = 60
MOVEMENT_SPEED = 3.0  # units per second
MOUSE_SENSITIVITY = 0.002
ZOOM_STEP = 0.15
MIN_ZOOM = 0.05
MAX_ZOOM = 5.0

# Approximate RealSense D4xx stereo depth FOV.
# We derive effective intrinsics from these angles after downsampling so
# adjacent camera views overlap more like the real hardware.
DEPTH_HORIZONTAL_FOV_DEG = 90.0
DEPTH_VERTICAL_FOV_DEG = 60.0
MAX_DEPTH_METERS = 10.0
COLOR_SCALE_MAX_DEPTH_METERS = 9.0
POINT_SIZE_PIXELS = 1.0
VISUALIZATION_HEATMAP = 0
VISUALIZATION_RGB = 1
CONTROL_PANEL_MARGIN_PX = 16
CONTROL_PANEL_WIDTH_PX = 308
CONTROL_PANEL_TOP_PX = 16
CONTROL_PANEL_LEFT_PX = WINDOW_SIZE[0] - CONTROL_PANEL_WIDTH_PX - CONTROL_PANEL_MARGIN_PX
CONTROL_PANEL_PADDING_PX = 10
CONTROL_PANEL_TITLE_HEIGHT_PX = 24
CONTROL_PANEL_TITLE_GAP_PX = 6
CONTROL_PANEL_ROW_HEIGHT_PX = 30
CONTROL_PANEL_ROW_GAP_PX = 4
CONTROL_PANEL_FONT_SIZE_PX = 18
CONTROL_PANEL_TITLE_FONT_SIZE_PX = 22
CONTROL_PANEL_SWITCH_WIDTH_PX = 68
CONTROL_PANEL_SWITCH_HEIGHT_PX = 20

# Robot-centered extrinsics:
#   0: front/back swapped, currently facing rearward
#   1: left D455, 20 cm left of center, facing left
#   2: right D455, 20 cm right of center, facing right
#   3: front/back swapped, currently facing forward
CAMERA_ROTATIONS = (180.0, 90.0, -90.0, 0.0)
CAMERA_OFFSETS = (
    np.array([0.0, 0.0, -0.05], dtype=np.float32),
    np.array([-0.05, -0.09, 0.0], dtype=np.float32),
    np.array([0.05, -0.09, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.05], dtype=np.float32),
)
CAMERA_MAX_DEPTHS = (MAX_DEPTH_METERS, MAX_DEPTH_METERS, MAX_DEPTH_METERS, 3.0)
# Human detection settings
DETECTION_ENABLED = False          # Default off, toggle with 'H' key
OBJECT_DETECTION_ENABLED = False   # Default off, toggle with 'O' key
DETECTION_MODEL_PATH = "yolo11n-seg.pt"
DETECTION_CONF_THRESHOLD = 0.5
DETECTION_IOU_THRESHOLD = 0.45
DETECTION_IMGSZ = 640
DETECTION_DEVICE = "cuda:0"         # GPU-only by default; use "cpu" only if explicitly wanted
DETECTION_HALF_PRECISION = True
DETECTION_BATCH_SIZE = 1            # Keep GPU peak memory low beside vLLM
DETECTION_MAX_DET = 20
DETECTION_RETINA_MASKS = False      # Full-res masks cost much more VRAM
DETECTION_CAMERAS = tuple(range(len(CAMERA_ROTATIONS)))  # Skip missing cameras per frame
DETECTION_EROSION_KERNEL = 3
DETECTION_SUBSAMPLE_STEP = 2
DETECTION_OUTLIER_THRESHOLD_M = 0.6
DETECTION_EXTENT_TRIM_PERCENTILE = 1.0
DETECTION_MIN_POINTS = 10
DETECTION_DEPTH_WORKERS = max(1, min(16, (os.cpu_count() or 2) - 1))
DETECTION_WORKER_JOIN_TIMEOUT_S = 1.0
PERSON_DETECTION_CLASS_IDS = (0,)
ALL_DETECTION_CLASS_IDS = None
BBOX_COLOR = (0.0, 1.0, 0.5, 1.0)  # Green-cyan wireframe

# Pose estimation settings
POSE_ENABLED = False              # Default off, toggle with 'P' key
POSE_MODEL_PATH = "yolo11n-pose.pt"
POSE_CONF_THRESHOLD = 0.45
POSE_IOU_THRESHOLD = 0.45
POSE_IMGSZ = 640
POSE_DEVICE = "cuda:0"             # GPU-only by default; use "cpu" only if explicitly wanted
POSE_HALF_PRECISION = True
POSE_BATCH_SIZE = 1
POSE_MAX_DET = 10
POSE_CAMERAS = tuple(range(len(CAMERA_ROTATIONS)))
POSE_KEYPOINT_CONF_THRESHOLD = 0.5
POSE_MIN_VALID_KEYPOINTS = 4
POSE_MIN_VALID_CONNECTIONS = 2
POSE_DEPTH_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
POSE_WORKER_JOIN_TIMEOUT_S = 1.0
SKELETON_COLOR = (1.0, 0.9, 0.2, 1.0)  # Bright yellow skeleton bones + keypoints

# Low-rate semantic VLM object localization settings.
VLM_ENABLED = False                 # Default off, toggle with 'V' key
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "EMPTY")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen3.6-35b-a3b")
VLM_REQUEST_RATE_HZ = 2.0           # Total across all camera requests
VLM_REQUEST_TIMEOUT_S = 5.0
VLM_MAX_TOKENS = 32
VLM_JPEG_QUALITY = 90
VLM_IMAGE_MAX_SIDE_PX = 640
VLM_MAX_OBJECTS = 1
VLM_CONF_THRESHOLD = 0.5
VLM_MIN_BOX_SIDE_PX = 8.0
VLM_CAMERAS = tuple(range(len(CAMERA_ROTATIONS)))
VLM_REQUEST_WORKERS = max(1, min(2, len(VLM_CAMERAS)))
VLM_STALE_TTL_S = 30.0
VLM_EROSION_KERNEL = 5
VLM_SUBSAMPLE_STEP = 3
VLM_OUTLIER_THRESHOLD_M = 0.45
VLM_EXTENT_TRIM_PERCENTILE = 3.0
VLM_MIN_POINTS = 20
VLM_DEPTH_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
VLM_WORKER_JOIN_TIMEOUT_S = 1.0

# VLM scene description settings (natural language, not bounding boxes).
SCENE_DESCRIPTION_ENABLED = False     # Default off, toggle with 'L' key
SCENE_DESCRIPTION_REFRESH_INTERVAL_S = 10.0  # Refresh every 10 seconds
SCENE_DESCRIPTION_MAX_TOKENS = 256    # Richer 2-4 sentence summaries
SCENE_DESCRIPTION_JPEG_QUALITY = 82
SCENE_DESCRIPTION_IMAGE_MAX_SIDE_PX = 512
SCENE_DESCRIPTION_CAMERAS = tuple(range(len(CAMERA_ROTATIONS)))
SCENE_DESCRIPTION_WORKERS = max(1, min(4, len(SCENE_DESCRIPTION_CAMERAS)))
SCENE_DESCRIPTION_PANEL_MARGIN_PX = 12
SCENE_DESCRIPTION_PANEL_GAP_PX = 8
SCENE_DESCRIPTION_PANEL_HEIGHT_PX = 116
SCENE_DESCRIPTION_PANEL_PADDING_PX = 8
SCENE_DESCRIPTION_FONT_SIZE_PX = 18
SCENE_DESCRIPTION_TITLE_FONT_SIZE_PX = 20

@dataclass
class CameraFrame:
    """One depth/color packet from one camera stream."""

    camera_id: int
    camera_serial: str
    sequence: int
    capture_received_time_ns: int
    depth_frame_number: int
    color_frame_number: int
    depth_device_timestamp_ms: float
    color_device_timestamp_ms: float
    depth: np.ndarray
    color: np.ndarray


@dataclass
class ReceiverCameraState:
    pending: Optional[CameraFrame] = None
    latest: Optional[CameraFrame] = None
    serial: str = ""
    last_frame_time_s: Optional[float] = None
    receive_count: int = 0
    bytes_received: int = 0
    replaced_latest: int = 0
    connections: int = 0


def round_trip_latency_estimate_s(round_trip_time_ns: int) -> Optional[float]:
    if round_trip_time_ns <= 0:
        return None
    return round_trip_time_ns / 2_000_000_000.0


def one_way_latency_estimate_s(sender_tai_ns: int, receiver_tai_ns: int) -> Optional[float]:
    if sender_tai_ns <= 0 or receiver_tai_ns <= 0:
        return None
    latency_ns = receiver_tai_ns - sender_tai_ns
    if latency_ns < 0 or latency_ns > LATENCY_PROBE_ONE_WAY_MAX_NS:
        return None
    return latency_ns / 1_000_000_000.0


def bounded_one_way_latency_estimate_s(
    sender_tai_ns: int,
    receiver_tai_ns: int,
    round_trip_time_ns: int,
) -> Optional[float]:
    latency_s = one_way_latency_estimate_s(sender_tai_ns, receiver_tai_ns)
    if latency_s is None or round_trip_time_ns <= 0:
        return None

    latency_ns = int(latency_s * 1_000_000_000.0)
    max_latency_ns = int(round_trip_time_ns * LATENCY_PROBE_ONE_WAY_RTT_FACTOR)
    max_latency_ns += LATENCY_PROBE_ONE_WAY_RTT_MARGIN_NS
    if latency_ns > max_latency_ns:
        return None
    return latency_s


def select_latency_probe_cpu(
    available_cpu_ids: Tuple[int, ...],
    configured_cpu: Optional[int],
) -> Optional[int]:
    """Choose the CPU id reserved for latency probes."""
    if configured_cpu is None:
        return None
    cpu_ids = tuple(sorted({int(cpu_id) for cpu_id in available_cpu_ids}))
    if not cpu_ids:
        return None
    if int(configured_cpu) < 0:
        return cpu_ids[-1]
    if int(configured_cpu) in cpu_ids:
        return int(configured_cpu)
    return None


def reserve_latency_probe_cpu() -> Optional[int]:
    """Keep future app worker threads off the CPU used by the UDP probe."""
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return None
    try:
        app_cpu_ids = set(os.sched_getaffinity(0))
    except OSError as exc:
        logging.warning("Unable to read receiver CPU affinity for latency probe: %s", exc)
        return None

    probe_cpu = select_latency_probe_cpu(tuple(app_cpu_ids), LATENCY_PROBE_CPU_CORE)
    if probe_cpu is None:
        if LATENCY_PROBE_CPU_CORE is not None:
            logging.warning(
                "Receiver latency probe CPU %s is not in allowed CPU set %s",
                LATENCY_PROBE_CPU_CORE,
                sorted(app_cpu_ids),
            )
        return None
    if not RESERVE_LATENCY_PROBE_CPU or len(app_cpu_ids) < 2:
        return probe_cpu

    try:
        os.sched_setaffinity(0, app_cpu_ids - {probe_cpu})
        logging.info("Receiver reserving CPU %d for latency probes", probe_cpu)
    except OSError as exc:
        logging.warning(
            "Unable to reserve receiver latency probe CPU %d: %s",
            probe_cpu,
            exc,
        )
        return None
    return probe_cpu


def pin_current_thread_to_cpu(cpu_core: Optional[int], name: str) -> None:
    """Pin the calling Linux thread to one CPU when affinity is enabled."""
    if cpu_core is None or not hasattr(os, "sched_setaffinity"):
        return
    try:
        os.sched_setaffinity(0, {cpu_core})
    except OSError as exc:
        logging.warning("Unable to pin %s to CPU %d: %s", name, cpu_core, exc)


def configure_realtime_thread(priority: int, name: str) -> None:
    """Best-effort SCHED_FIFO for the latency probe thread/process."""
    if priority <= 0 or not hasattr(os, "sched_setscheduler"):
        return
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
        logging.info("%s using SCHED_FIFO priority %d", name, priority)
    except OSError as exc:
        logging.warning("Unable to set %s SCHED_FIFO priority %d: %s", name, priority, exc)


def lock_process_memory(name: str) -> None:
    """Best-effort mlockall to avoid page-fault spikes in the probe loop."""
    if not LATENCY_PROBE_MLOCK:
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.mlockall(MCL_CURRENT | MCL_FUTURE) != 0:
            errno = ctypes.get_errno()
            logging.warning("Unable to mlockall for %s: errno %d", name, errno)
        else:
            logging.info("%s locked current/future memory", name)
    except Exception as exc:
        logging.warning("Unable to mlockall for %s: %s", name, exc)


def clock_tai_ns() -> int:
    if hasattr(time, "CLOCK_TAI"):
        try:
            return time.clock_gettime_ns(time.CLOCK_TAI)
        except OSError:
            return 0
    return 0


def configure_latency_probe_socket(
    sock: socket.socket,
    name: str,
    incoming_cpu: Optional[int] = None,
) -> None:
    """Apply app-level QoS and low-latency socket options for the UDP probe."""
    tos = max(0, min(63, LATENCY_PROBE_DSCP)) << 2
    options = (
        (socket.IPPROTO_IP, socket.IP_TOS, tos, "IP_TOS"),
        (socket.SOL_SOCKET, socket.SO_PRIORITY, LATENCY_PROBE_SOCKET_PRIORITY, "SO_PRIORITY"),
        (socket.SOL_SOCKET, socket.SO_SNDBUF, LATENCY_PROBE_SOCKET_BUFFER, "SO_SNDBUF"),
        (socket.SOL_SOCKET, socket.SO_RCVBUF, LATENCY_PROBE_SOCKET_BUFFER, "SO_RCVBUF"),
    )
    for level, option, value, label in options:
        with contextlib.suppress(OSError):
            sock.setsockopt(level, option, value)
            logging.debug("%s set %s=%s", name, label, value)

    if LATENCY_PROBE_BIND_DEVICE:
        try:
            sock.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_BINDTODEVICE,
                LATENCY_PROBE_BIND_DEVICE.encode() + b"\0",
            )
            logging.info("%s bound to %s", name, LATENCY_PROBE_BIND_DEVICE)
        except OSError as exc:
            logging.warning("%s unable to bind to %s: %s", name, LATENCY_PROBE_BIND_DEVICE, exc)

    if LATENCY_PROBE_BUSY_POLL_US > 0:
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.SOL_SOCKET, SO_BUSY_POLL, LATENCY_PROBE_BUSY_POLL_US)
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.SOL_SOCKET, SO_PREFER_BUSY_POLL, 1)
    if incoming_cpu is not None:
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.SOL_SOCKET, SO_INCOMING_CPU, int(incoming_cpu))


class Metrics:
    """Thread-safe throughput metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: "collections.deque[Tuple[float, int]]" = collections.deque()
        self._total_bytes = 0

    def add_bytes(self, count: int) -> None:
        now = time.time()
        with self._lock:
            self._samples.append((now, count))
            self._total_bytes += count
            self._expire_locked(now)

    def bitrate_mbps(self) -> float:
        now = time.time()
        with self._lock:
            self._expire_locked(now)
            total = self._total_bytes
            if not self._samples:
                return 0.0
            earliest = self._samples[0][0]
        window = max(now - earliest, 1e-9)
        return (total * 8) / (window * 1024 * 1024)

    def _expire_locked(self, now: float) -> None:
        cutoff = now - 3.0
        while self._samples and self._samples[0][0] < cutoff:
            _, count = self._samples.popleft()
            self._total_bytes -= count


class LatencyJitterMetrics:
    """Thread-safe UDP echo latency-estimate and jitter metrics."""

    def __init__(self, window_s: float = 3.0) -> None:
        self._lock = threading.Lock()
        self._window_s = float(window_s)
        self._samples: Deque[Tuple[float, float, float, Optional[float], Optional[float]]] = collections.deque()
        self._prev_latency_s: Optional[float] = None
        self._prev_one_way_latency_s: Optional[float] = None

    def add_sample(
        self,
        *,
        round_trip_time_ns: int,
        receiver_time_ns: Optional[int] = None,
        sender_tai_ns: int = 0,
        receiver_tai_ns: int = 0,
    ) -> None:
        if receiver_time_ns is None:
            receiver_time_ns = time.time_ns()

        latency_s = round_trip_latency_estimate_s(round_trip_time_ns)
        if latency_s is None:
            return

        receiver_time_s = receiver_time_ns / 1_000_000_000.0
        jitter_sample_s = 0.0
        one_way_latency_s = bounded_one_way_latency_estimate_s(
            sender_tai_ns,
            receiver_tai_ns,
            round_trip_time_ns,
        )
        one_way_jitter_sample_s: Optional[float] = None

        with self._lock:
            if self._prev_latency_s is not None:
                jitter_sample_s = abs(latency_s - self._prev_latency_s)
            self._prev_latency_s = latency_s

            if one_way_latency_s is not None:
                if self._prev_one_way_latency_s is not None:
                    one_way_jitter_sample_s = abs(
                        one_way_latency_s - self._prev_one_way_latency_s
                    )
                self._prev_one_way_latency_s = one_way_latency_s

            self._samples.append(
                (
                    receiver_time_s,
                    latency_s,
                    jitter_sample_s,
                    one_way_latency_s,
                    one_way_jitter_sample_s,
                )
            )
            self._expire_locked(receiver_time_s)

    def summary(self) -> Dict[str, Optional[float]]:
        """Returns metrics in milliseconds where applicable."""
        now = time.time()
        with self._lock:
            self._expire_locked(now)
            if not self._samples:
                return {
                    "count": 0,
                    "latency_mean_ms": None,
                    "latency_p95_ms": None,
                    "latency_p99_ms": None,
                    "latency_last_ms": None,
                    "jitter_mean_ms": None,
                    "jitter_last_ms": None,
                    "jitter_p95_ms": None,
                    "jitter_p99_ms": None,
                    "jitter_std_ms": None,
                    "one_way_count": 0,
                    "one_way_latency_mean_ms": None,
                    "one_way_latency_p95_ms": None,
                    "one_way_latency_p99_ms": None,
                    "one_way_jitter_mean_ms": None,
                    "one_way_jitter_p95_ms": None,
                    "one_way_jitter_p99_ms": None,
                }
            latencies = np.array([s[1] for s in self._samples], dtype=np.float64)
            jitter_samples = np.array([s[2] for s in self._samples], dtype=np.float64)
            one_way_latencies = np.array(
                [s[3] for s in self._samples if s[3] is not None],
                dtype=np.float64,
            )
            one_way_jitter_samples = np.array(
                [s[4] for s in self._samples if s[4] is not None],
                dtype=np.float64,
            )
            last = float(latencies[-1])
            mean = float(np.mean(latencies))
            p95 = float(np.percentile(latencies, 95))
            p99 = float(np.percentile(latencies, 99))
            jitter_mean = float(np.mean(jitter_samples))
            jitter_last = float(jitter_samples[-1])
            jitter_p95 = float(np.percentile(jitter_samples, 95))
            jitter_p99 = float(np.percentile(jitter_samples, 99))
            jitter_std = float(np.std(jitter_samples))
            one_way_summary = self._summary_array_ms(one_way_latencies)
            one_way_jitter_summary = self._summary_array_ms(one_way_jitter_samples)

        return {
            "count": int(len(latencies)),
            "latency_mean_ms": mean * 1000.0,
            "latency_p95_ms": p95 * 1000.0,
            "latency_p99_ms": p99 * 1000.0,
            "latency_last_ms": last * 1000.0,
            "jitter_mean_ms": jitter_mean * 1000.0,
            "jitter_last_ms": jitter_last * 1000.0,
            "jitter_p95_ms": jitter_p95 * 1000.0,
            "jitter_p99_ms": jitter_p99 * 1000.0,
            "jitter_std_ms": jitter_std * 1000.0,
            "one_way_count": int(len(one_way_latencies)),
            "one_way_latency_mean_ms": one_way_summary[0],
            "one_way_latency_p95_ms": one_way_summary[1],
            "one_way_latency_p99_ms": one_way_summary[2],
            "one_way_jitter_mean_ms": one_way_jitter_summary[0],
            "one_way_jitter_p95_ms": one_way_jitter_summary[1],
            "one_way_jitter_p99_ms": one_way_jitter_summary[2],
        }

    def _expire_locked(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    @staticmethod
    def _summary_array_ms(samples_s: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if len(samples_s) == 0:
            return None, None, None
        return (
            float(np.mean(samples_s)) * 1000.0,
            float(np.percentile(samples_s, 95)) * 1000.0,
            float(np.percentile(samples_s, 99)) * 1000.0,
        )


class AverageMetricsCsvRecorder:
    """Writes fixed-window receiver network timing results."""

    _FIELDS = (
        "window_start_utc",
        "window_end_utc",
        "duration_s",
        "bytes_received",
        "average_bitrate_mbps",
        "latency_samples",
        "average_network_latency_ms",
        "p95_network_latency_ms",
        "p99_network_latency_ms",
        "jitter_samples",
        "average_network_jitter_ms",
        "p95_network_jitter_ms",
        "p99_network_jitter_ms",
        "one_way_latency_samples",
        "average_one_way_latency_ms",
        "p95_one_way_latency_ms",
        "p99_one_way_latency_ms",
        "one_way_jitter_samples",
        "average_one_way_jitter_ms",
        "p95_one_way_jitter_ms",
        "p99_one_way_jitter_ms",
    )

    def __init__(self, path: Path, interval_s: float) -> None:
        self._path = path
        self._interval_s = float(interval_s)
        self._lock = threading.Lock()
        self._window_start_s: Optional[float] = None
        self._bytes_received = 0
        self._latency_samples_s: list[float] = []
        self._jitter_samples_s: list[float] = []
        self._one_way_latency_samples_s: list[float] = []
        self._one_way_jitter_samples_s: list[float] = []
        self._prev_latency_s: Optional[float] = None
        self._prev_one_way_latency_s: Optional[float] = None
        self._wrote_schema_header = False

    def add_frame_bytes(self, *, byte_count: int, receiver_time_ns: int) -> None:
        receiver_time_s = receiver_time_ns / 1_000_000_000.0

        with self._lock:
            if self._window_start_s is None:
                self._window_start_s = receiver_time_s
            self._write_due_results_locked(receiver_time_s)
            self._bytes_received += byte_count

    def add_latency_probe(
        self,
        *,
        round_trip_time_ns: int,
        receiver_time_ns: int,
        sender_tai_ns: int = 0,
        receiver_tai_ns: int = 0,
    ) -> None:
        receiver_time_s = receiver_time_ns / 1_000_000_000.0
        latency_s = round_trip_latency_estimate_s(round_trip_time_ns)
        one_way_latency_s = bounded_one_way_latency_estimate_s(
            sender_tai_ns,
            receiver_tai_ns,
            round_trip_time_ns,
        )

        with self._lock:
            if self._window_start_s is None:
                self._window_start_s = receiver_time_s
            self._write_due_results_locked(receiver_time_s)

            if latency_s is None:
                return

            self._latency_samples_s.append(latency_s)
            if self._prev_latency_s is not None:
                self._jitter_samples_s.append(abs(latency_s - self._prev_latency_s))
            self._prev_latency_s = latency_s
            if one_way_latency_s is not None:
                self._one_way_latency_samples_s.append(one_way_latency_s)
                if self._prev_one_way_latency_s is not None:
                    self._one_way_jitter_samples_s.append(
                        abs(one_way_latency_s - self._prev_one_way_latency_s)
                    )
                self._prev_one_way_latency_s = one_way_latency_s

    def write_due_results(self, now_s: Optional[float] = None) -> None:
        if now_s is None:
            now_s = time.time()
        with self._lock:
            self._write_due_results_locked(now_s)

    def _write_due_results_locked(self, now_s: float) -> None:
        while self._window_start_s is not None and now_s >= self._window_start_s + self._interval_s:
            window_end_s = self._window_start_s + self._interval_s
            if self._bytes_received or self._latency_samples_s:
                self._append_result_locked(window_end_s)
            self._window_start_s = window_end_s
            self._bytes_received = 0
            self._latency_samples_s = []
            self._jitter_samples_s = []
            self._one_way_latency_samples_s = []
            self._one_way_jitter_samples_s = []

    def _append_result_locked(self, window_end_s: float) -> None:
        if self._window_start_s is None:
            return

        mean_latency_ms, p95_latency_ms, p99_latency_ms = self._summary_ms(self._latency_samples_s)
        mean_jitter_ms, p95_jitter_ms, p99_jitter_ms = self._summary_ms(self._jitter_samples_s)
        mean_one_way_ms, p95_one_way_ms, p99_one_way_ms = self._summary_ms(
            self._one_way_latency_samples_s
        )
        mean_one_way_jitter_ms, p95_one_way_jitter_ms, p99_one_way_jitter_ms = self._summary_ms(
            self._one_way_jitter_samples_s
        )

        row = {
            "window_start_utc": self._format_utc(self._window_start_s),
            "window_end_utc": self._format_utc(window_end_s),
            "duration_s": f"{self._interval_s:.1f}",
            "bytes_received": str(self._bytes_received),
            "average_bitrate_mbps": f"{(self._bytes_received * 8) / (self._interval_s * 1024 * 1024):.3f}",
            "latency_samples": str(len(self._latency_samples_s)),
            "average_network_latency_ms": mean_latency_ms,
            "p95_network_latency_ms": p95_latency_ms,
            "p99_network_latency_ms": p99_latency_ms,
            "jitter_samples": str(len(self._jitter_samples_s)),
            "average_network_jitter_ms": mean_jitter_ms,
            "p95_network_jitter_ms": p95_jitter_ms,
            "p99_network_jitter_ms": p99_jitter_ms,
            "one_way_latency_samples": str(len(self._one_way_latency_samples_s)),
            "average_one_way_latency_ms": mean_one_way_ms,
            "p95_one_way_latency_ms": p95_one_way_ms,
            "p99_one_way_latency_ms": p99_one_way_ms,
            "one_way_jitter_samples": str(len(self._one_way_jitter_samples_s)),
            "average_one_way_jitter_ms": mean_one_way_jitter_ms,
            "p95_one_way_jitter_ms": p95_one_way_jitter_ms,
            "p99_one_way_jitter_ms": p99_one_way_jitter_ms,
        }

        try:
            needs_header = self._needs_header()
            with self._path.open("a", newline="", encoding="utf-8") as result_file:
                writer = csv.DictWriter(result_file, fieldnames=self._FIELDS)
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)
        except OSError as exc:
            logging.warning("Unable to write receiver metrics result to %s: %s", self._path, exc)
            return

    @staticmethod
    def _format_utc(timestamp_s: float) -> str:
        return datetime.fromtimestamp(timestamp_s, timezone.utc).isoformat(timespec="milliseconds")

    def _needs_header(self) -> bool:
        if not self._path.exists() or self._path.stat().st_size == 0:
            return True
        try:
            with self._path.open("r", newline="", encoding="utf-8") as result_file:
                first_line = result_file.readline().strip()
        except OSError:
            return True
        if first_line == ",".join(self._FIELDS):
            return False
        if self._wrote_schema_header:
            return False
        self._wrote_schema_header = True
        return True

    @staticmethod
    def _summary_ms(samples_s: list[float]) -> Tuple[str, str, str]:
        if not samples_s:
            return "", "", ""

        samples_ms = np.array(samples_s, dtype=np.float64) * 1000.0
        return (
            f"{float(np.mean(samples_ms)):.3f}",
            f"{float(np.percentile(samples_ms, 95)):.3f}",
            f"{float(np.percentile(samples_ms, 99)):.3f}",
        )


class FrameReceiver(threading.Thread):
    """Accepts independent per-camera TCP streams and retains latest frames."""

    def __init__(self, host: str, port: int, metrics: Metrics, stop_event: threading.Event) -> None:
        super().__init__(name="FrameReceiver", daemon=True)
        self._host = host
        self._port = port
        self._metrics = metrics
        self._stop_event = stop_event
        self._frame_lock = threading.Lock()
        self._camera_states = [ReceiverCameraState() for _ in CAMERA_ROTATIONS]
        self._server_socket: Optional[socket.socket] = None
        self._client_sockets: set[socket.socket] = set()
        self._client_threads: set[threading.Thread] = set()
        self._socket_lock = threading.Lock()
        self._camera_fps_last_sample_s = time.time()

        self._latency_metrics = LatencyJitterMetrics(window_s=3.0)
        self._metrics_recorder = AverageMetricsCsvRecorder(
            METRICS_RESULTS_PATH,
            METRICS_RESULT_INTERVAL,
        )

    def run(self) -> None:
        logging.info("Starting receiver on %s:%s", self._host or "0.0.0.0", self._port)
        try:
            self._open_server()
            while not self._stop_event.is_set():
                conn = self._accept_client()
                if conn is None:
                    continue
                peer = self._peer_name(conn)
                handler = threading.Thread(
                    target=self._receive_connection,
                    args=(conn, peer),
                    name=f"FrameStream-{peer}",
                    daemon=True,
                )
                with self._socket_lock:
                    self._client_sockets.add(conn)
                    self._client_threads.add(handler)
                logging.debug("Frame stream connected from %s", peer)
                handler.start()
        except Exception:
            logging.exception("Frame receiver encountered an unexpected error")
        finally:
            self._close_sockets()
            self._join_client_threads()
            logging.info("Frame receiver stopped")

    def _open_server(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_RCVBUF)
        server.bind((self._host, self._port))
        server.listen(SOCKET_BACKLOG)
        server.settimeout(ACCEPT_TIMEOUT)
        with self._socket_lock:
            self._server_socket = server

    def _accept_client(self) -> Optional[socket.socket]:
        with self._socket_lock:
            server = self._server_socket
        if server is None:
            return None
        try:
            conn, _addr = server.accept()
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_RCVBUF)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            conn.settimeout(RECV_TIMEOUT)
            return conn
        except socket.timeout:
            return None
        except OSError as exc:
            if not self._stop_event.is_set():
                logging.warning("Accept failed: %s", exc)
                time.sleep(ACCEPT_RETRY_DELAY)
            return None

    def _receive_connection(self, conn: socket.socket, peer: str) -> None:
        camera_id: Optional[int] = None
        try:
            while not self._stop_event.is_set():
                header = self._recv_exact(conn, MESSAGE_HEADER_SIZE, peer)
                if not header:
                    break
                header_recv_time_ns = time.time_ns()
                (message_length,) = struct.unpack(MESSAGE_HEADER_STRUCT, header)
                if message_length <= 0:
                    logging.warning("Invalid payload length %d from %s", message_length, peer)
                    break
                payload = self._recv_exact(conn, message_length, peer)
                if payload is None:
                    break

                byte_count = len(header) + len(payload)
                self._metrics.add_bytes(byte_count)
                self._metrics_recorder.add_frame_bytes(
                    byte_count=byte_count,
                    receiver_time_ns=header_recv_time_ns,
                )

                try:
                    frame = self._extract_camera_packet(
                        self._deserialize_camera_payload(payload)
                    )
                except ValueError as exc:
                    logging.warning("Rejected frame payload from %s: %s", peer, exc)
                    continue
                except Exception:
                    logging.exception("Failed to deserialize frame payload from %s", peer)
                    continue

                if camera_id is None:
                    camera_id = frame.camera_id
                    self._change_camera_connections(camera_id, 1)
                    logging.debug(
                        "Frame stream %s identified as camera %d (%s)",
                        peer,
                        camera_id,
                        frame.camera_serial,
                    )
                elif camera_id != frame.camera_id:
                    logging.warning(
                        "Frame stream %s changed camera id from %d to %d; frame skipped",
                        peer,
                        camera_id,
                        frame.camera_id,
                    )
                    continue

                self._store_camera_frame(frame, byte_count)
        finally:
            if camera_id is not None:
                self._change_camera_connections(camera_id, -1)
            logging.debug("Frame stream disconnected from %s", peer)
            with contextlib.suppress(Exception):
                conn.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(Exception):
                conn.close()
            with self._socket_lock:
                self._client_sockets.discard(conn)
                self._client_threads.discard(threading.current_thread())

    @staticmethod
    def _extract_camera_packet(raw: Any) -> CameraFrame:
        if not isinstance(raw, dict):
            raise ValueError(f"expected dict payload, got {type(raw)}")
        if raw.get("version") != PAYLOAD_VERSION:
            raise ValueError(f"unsupported payload version {raw.get('version')}")

        camera_id = raw.get("camera_id")
        if not isinstance(camera_id, int) or not 0 <= camera_id < len(CAMERA_ROTATIONS):
            raise ValueError(f"invalid camera id {camera_id}")

        camera_serial = raw.get("camera_serial")
        if not isinstance(camera_serial, str) or not camera_serial:
            raise ValueError("packet missing camera serial")

        int_fields = (
            "sequence",
            "capture_received_time_ns",
            "depth_frame_number",
            "color_frame_number",
        )
        for field in int_fields:
            if not isinstance(raw.get(field), int):
                raise ValueError(f"packet missing integer '{field}'")

        timestamp_fields = ("depth_device_timestamp_ms", "color_device_timestamp_ms")
        for field in timestamp_fields:
            if not isinstance(raw.get(field), (int, float)):
                raise ValueError(f"packet missing numeric '{field}'")

        depth = raw.get("depth")
        color = raw.get("color")
        if (
            not isinstance(depth, np.ndarray)
            or depth.dtype != np.uint16
            or depth.shape != (FRAME_HEIGHT, FRAME_WIDTH)
        ):
            raise ValueError(
                f"invalid depth array for camera {camera_id}: "
                f"{getattr(depth, 'shape', None)} {getattr(depth, 'dtype', None)}"
            )
        if (
            not isinstance(color, np.ndarray)
            or color.dtype != np.uint8
            or color.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3)
        ):
            raise ValueError(
                f"invalid color array for camera {camera_id}: "
                f"{getattr(color, 'shape', None)} {getattr(color, 'dtype', None)}"
            )

        return CameraFrame(
            camera_id=camera_id,
            camera_serial=camera_serial,
            sequence=raw["sequence"],
            capture_received_time_ns=raw["capture_received_time_ns"],
            depth_frame_number=raw["depth_frame_number"],
            color_frame_number=raw["color_frame_number"],
            depth_device_timestamp_ms=float(raw["depth_device_timestamp_ms"]),
            color_device_timestamp_ms=float(raw["color_device_timestamp_ms"]),
            depth=depth,
            color=color,
        )

    @staticmethod
    def _deserialize_camera_payload(payload: bytes | bytearray) -> Any:
        if not payload.startswith(BUFFERED_FRAME_MAGIC):
            return pickle.loads(payload)
        if len(payload) < BUFFERED_FRAME_HEADER_SIZE:
            raise ValueError("buffered payload header is truncated")

        _magic, pickle_length, buffer_count = struct.unpack_from(
            BUFFERED_FRAME_HEADER_STRUCT,
            payload,
        )
        if buffer_count > MAX_BUFFERED_FRAME_BUFFERS:
            raise ValueError(f"buffered payload has too many buffers: {buffer_count}")

        lengths_offset = BUFFERED_FRAME_HEADER_SIZE
        lengths_end = lengths_offset + buffer_count * BUFFERED_FRAME_BUFFER_LENGTH_SIZE
        pickle_end = lengths_end + pickle_length
        if len(payload) < pickle_end:
            raise ValueError("buffered payload pickle data is truncated")

        buffer_lengths = [
            struct.unpack_from(
                BUFFERED_FRAME_BUFFER_LENGTH_STRUCT,
                payload,
                lengths_offset + buffer_idx * BUFFERED_FRAME_BUFFER_LENGTH_SIZE,
            )[0]
            for buffer_idx in range(buffer_count)
        ]
        payload_view = memoryview(payload)
        pickle_payload = payload_view[lengths_end:pickle_end]
        pickle_buffers = []
        buffer_offset = pickle_end
        for buffer_length in buffer_lengths:
            buffer_end = buffer_offset + buffer_length
            if len(payload) < buffer_end:
                raise ValueError("buffered payload array data is truncated")
            pickle_buffers.append(payload_view[buffer_offset:buffer_end])
            buffer_offset = buffer_end
        if buffer_offset != len(payload):
            raise ValueError("buffered payload has trailing bytes")

        return pickle.loads(pickle_payload, buffers=pickle_buffers)

    def _recv_exact(self, conn: socket.socket, size: int, peer: str) -> Optional[bytearray]:
        buffer = bytearray(size)
        view = memoryview(buffer)
        received = 0
        while received < size and not self._stop_event.is_set():
            try:
                chunk = conn.recv_into(view[received:])
            except socket.timeout:
                logging.warning("Timed out waiting for data from %s", peer)
                return None
            except OSError as exc:
                if not self._stop_event.is_set():
                    logging.warning("Socket error from %s: %s", peer, exc)
                return None
            if chunk == 0:
                return None
            received += chunk
        if received < size:
            return None
        return buffer

    def _store_camera_frame(self, frame: CameraFrame, byte_count: int) -> None:
        now_s = time.time()
        with self._frame_lock:
            state = self._camera_states[frame.camera_id]
            if state.pending is not None:
                state.replaced_latest += 1
            state.pending = frame
            state.latest = frame
            state.serial = frame.camera_serial
            state.last_frame_time_s = now_s
            state.receive_count += 1
            state.bytes_received += byte_count

    def pop_pending_frames(self) -> Dict[int, CameraFrame]:
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
    ) -> Dict[int, CameraFrame]:
        allowed = set(camera_ids) if camera_ids is not None else None
        with self._frame_lock:
            return {
                camera_id: state.latest
                for camera_id, state in enumerate(self._camera_states)
                if state.latest is not None
                and (allowed is None or camera_id in allowed)
            }  # type: ignore[return-value]

    def stop(self) -> None:
        """Request shutdown and interrupt blocking accept/recv calls."""
        self._stop_event.set()
        self._close_sockets()

    def is_connected(self) -> bool:
        with self._frame_lock:
            return any(state.connections > 0 for state in self._camera_states)

    def latency_summary(self) -> Dict[str, Optional[float]]:
        return self._latency_metrics.summary()

    def record_latency_probe(
        self,
        round_trip_time_ns: int,
        receiver_time_ns: int,
        sender_tai_ns: int = 0,
        receiver_tai_ns: int = 0,
    ) -> None:
        self._latency_metrics.add_sample(
            round_trip_time_ns=round_trip_time_ns,
            receiver_time_ns=receiver_time_ns,
            sender_tai_ns=sender_tai_ns,
            receiver_tai_ns=receiver_tai_ns,
        )
        self._metrics_recorder.add_latency_probe(
            round_trip_time_ns=round_trip_time_ns,
            receiver_time_ns=receiver_time_ns,
            sender_tai_ns=sender_tai_ns,
            receiver_tai_ns=receiver_tai_ns,
        )

    def write_metrics_result_if_due(self, now_s: Optional[float] = None) -> None:
        self._metrics_recorder.write_due_results(now_s)

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
        return float(np.mean(camera_counts)) / elapsed_s

    def _change_camera_connections(self, camera_id: int, delta: int) -> None:
        with self._frame_lock:
            state = self._camera_states[camera_id]
            state.connections = max(0, state.connections + delta)

    def _close_sockets(self) -> None:
        with self._socket_lock:
            server = self._server_socket
            clients = list(self._client_sockets)
            self._server_socket = None
        for conn in clients:
            with contextlib.suppress(Exception):
                conn.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(Exception):
                conn.close()
        if server:
            with contextlib.suppress(Exception):
                server.close()

    def _join_client_threads(self) -> None:
        with self._socket_lock:
            threads = list(self._client_threads)
        for thread in threads:
            thread.join(timeout=RECV_TIMEOUT + 1.0)

    @staticmethod
    def _peer_name(conn: socket.socket) -> str:
        try:
            host, port = conn.getpeername()
            return f"{host}:{port}"
        except OSError:
            return "unknown"


class LatencyProbeReceiver(threading.Thread):
    """Echoes UDP probes and records sender-measured RTT results."""

    def __init__(
        self,
        host: str,
        port: int,
        frame_receiver: FrameReceiver,
        stop_event: threading.Event,
        cpu_core: Optional[int] = None,
    ) -> None:
        super().__init__(name="LatencyProbeReceiver", daemon=True)
        self._host = host
        self._port = port
        self._frame_receiver = frame_receiver
        self._stop_event = stop_event
        self._cpu_core = cpu_core
        self._socket_lock = threading.Lock()
        self._socket: Optional[socket.socket] = None

    def run(self) -> None:
        pin_current_thread_to_cpu(self._cpu_core, self.name)
        configure_realtime_thread(LATENCY_PROBE_RT_PRIORITY, self.name)
        lock_process_memory(self.name)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            configure_latency_probe_socket(sock, self.name, self._cpu_core)
            sock.bind((self._host, self._port))
            sock.settimeout(ACCEPT_TIMEOUT)
            with self._socket_lock:
                self._socket = sock

            logging.info(
                "Listening for latency probes on UDP %s:%s",
                self._host or "0.0.0.0",
                self._port,
            )
            while not self._stop_event.is_set():
                try:
                    packet, _peer = sock.recvfrom(LATENCY_PROBE_RECV_SIZE)
                    receiver_time_ns = time.time_ns()
                except socket.timeout:
                    continue
                except OSError as exc:
                    if self._stop_event.is_set():
                        break
                    logging.warning("Latency probe socket error: %s", exc)
                    break

                if len(packet) == LATENCY_PROBE_REQUEST_V2_SIZE and packet.startswith(LATENCY_PROBE_MAGIC):
                    (
                        _magic,
                        version,
                        _flags,
                        sequence,
                        sender_time_ns,
                        sender_tai_ns,
                        round_trip_time_ns,
                    ) = struct.unpack(LATENCY_PROBE_REQUEST_V2_STRUCT, packet)
                    if version != LATENCY_PROBE_VERSION:
                        continue
                    receiver_tai_ns = clock_tai_ns()
                    with contextlib.suppress(OSError):
                        sock.sendto(
                            struct.pack(
                                LATENCY_PROBE_ECHO_V2_STRUCT,
                                LATENCY_PROBE_MAGIC,
                                LATENCY_PROBE_VERSION,
                                0,
                                sequence,
                                sender_time_ns,
                                sender_tai_ns,
                                receiver_tai_ns,
                            ),
                            _peer,
                        )
                    if round_trip_time_ns > 0:
                        self._frame_receiver.record_latency_probe(
                            round_trip_time_ns=round_trip_time_ns,
                            receiver_time_ns=receiver_time_ns,
                            sender_tai_ns=sender_tai_ns,
                            receiver_tai_ns=receiver_tai_ns,
                        )
                    continue

                if len(packet) != LATENCY_PROBE_REQUEST_SIZE:
                    continue
                sender_time_ns, round_trip_time_ns = struct.unpack(
                    LATENCY_PROBE_REQUEST_STRUCT,
                    packet,
                )
                with contextlib.suppress(OSError):
                    sock.sendto(
                        struct.pack(LATENCY_PROBE_ECHO_STRUCT, sender_time_ns),
                        _peer,
                    )
                if round_trip_time_ns > 0:
                    self._frame_receiver.record_latency_probe(
                        round_trip_time_ns=round_trip_time_ns,
                        receiver_time_ns=receiver_time_ns,
                    )
        except OSError as exc:
            if not self._stop_event.is_set():
                logging.error("Unable to receive latency probes on UDP %s: %s", self._port, exc)
        finally:
            self.stop()
            with contextlib.suppress(Exception):
                sock.close()
            logging.info("Latency probe receiver stopped")

    def stop(self) -> None:
        with self._socket_lock:
            sock = self._socket
            self._socket = None
        if sock is not None:
            with contextlib.suppress(Exception):
                sock.close()


def build_yaw_rotation_matrix(rotation_deg: float) -> np.ndarray:
    rotation_rad = np.radians(rotation_deg)
    cos_theta = np.cos(rotation_rad)
    sin_theta = np.sin(rotation_rad)
    return np.array(
        [
            [cos_theta, 0.0, -sin_theta],
            [0.0, 1.0, 0.0],
            [sin_theta, 0.0, cos_theta],
        ],
        dtype=np.float32,
    )


def _split_vlm_camera_groups(
    camera_ids: Tuple[int, ...],
    worker_count: int,
) -> List[Tuple[int, ...]]:
    """Shard cameras so vLLM can batch independent low-latency requests."""
    group_count = max(1, min(int(worker_count), len(camera_ids) or 1))
    groups = [tuple(camera_ids[index::group_count]) for index in range(group_count)]
    return [group for group in groups if group]


class DetectionWorker(threading.Thread):
    """Runs YOLO and depth-to-3D fitting on changed per-camera snapshots."""

    def __init__(
        self,
        receiver: FrameReceiver,
        stop_event: threading.Event,
        detection_cameras: Tuple[int, ...],
        model_path: str,
        conf_threshold: float,
        iou_threshold: float,
        imgsz: int,
        device: str,
        half_precision: bool,
        batch_size: int,
        max_det: int,
        retina_masks: bool,
        erosion_kernel: int,
        subsample_step: int,
        outlier_threshold_m: float,
        min_points: int,
        camera_rotations: list,
        camera_offsets: tuple,
        camera_max_depths: tuple,
        horizontal_fov_deg: float,
        vertical_fov_deg: float,
        class_ids: Optional[Tuple[int, ...]] = PERSON_DETECTION_CLASS_IDS,
        initially_enabled: bool = False,
    ) -> None:
        super().__init__(name="DetectionWorker", daemon=True)
        self._receiver = receiver
        self._app_stop_event = stop_event
        self._worker_stop_event = threading.Event()
        self._enabled_event = threading.Event()
        if initially_enabled:
            self._enabled_event.set()
        self._detection_cameras = detection_cameras

        # Detection parameters
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._imgsz = imgsz
        self._device = device
        self._half_precision = half_precision
        self._batch_size = batch_size
        self._max_det = max_det
        self._retina_masks = retina_masks
        self._erosion_kernel = erosion_kernel
        self._subsample_step = subsample_step
        self._outlier_threshold_m = outlier_threshold_m
        self._min_points = min_points

        # Camera parameters for back-projection
        self._camera_rotations = camera_rotations
        self._camera_offsets = camera_offsets
        self._camera_max_depths = camera_max_depths
        self._horizontal_fov_deg = horizontal_fov_deg
        self._vertical_fov_deg = vertical_fov_deg

        # Shared state
        self._lock = threading.Lock()
        self._detections_by_camera: Dict[int, List[PersonBBox3D]] = {
            camera_id: [] for camera_id in detection_cameras
        }
        self._detector: Optional[YoloSegDetector] = None
        self._intrinsics_cache: Dict[int, dict] = {}
        self._class_ids = class_ids
        self._mode_generation = 0

    def get_detections(self) -> List[PersonBBox3D]:
        """Thread-safe getter for latest detections."""
        with self._lock:
            detections: List[PersonBBox3D] = []
            for camera_id in sorted(self._detections_by_camera):
                detections.extend(self._detections_by_camera[camera_id])
            return detections

    def request_stop(self) -> None:
        """Stop after any in-flight model call returns."""
        self._worker_stop_event.set()

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            self._enabled_event.set()
            return
        self._enabled_event.clear()
        self.clear_detections()

    def set_class_ids(self, class_ids: Optional[Tuple[int, ...]]) -> None:
        """Switch class filtering on the resident YOLO model."""
        normalized_class_ids = (
            tuple(sorted({int(class_id) for class_id in class_ids}))
            if class_ids is not None
            else None
        )
        with self._lock:
            if normalized_class_ids == self._class_ids:
                return
            self._class_ids = normalized_class_ids
            self._mode_generation += 1
            for camera_id in self._detections_by_camera:
                self._detections_by_camera[camera_id] = []
        detector = self._detector
        if detector is not None:
            detector.set_class_ids(normalized_class_ids)

    def is_enabled(self) -> bool:
        return self._enabled_event.is_set()

    def clear_detections(self) -> None:
        with self._lock:
            for camera_id in self._detections_by_camera:
                self._detections_by_camera[camera_id] = []

    def _should_stop(self) -> bool:
        return self._worker_stop_event.is_set() or self._app_stop_event.is_set()

    def _wait(self, timeout_s: float) -> None:
        self._worker_stop_event.wait(timeout_s)

    def _get_intrinsics(self, width: int, height: int) -> dict:
        """Get or compute intrinsics for given resolution."""
        key = (width, height)
        if key not in self._intrinsics_cache:
            self._intrinsics_cache[key] = build_camera_intrinsics_from_fov(
                width, height,
                self._horizontal_fov_deg,
                self._vertical_fov_deg,
            )
        return self._intrinsics_cache[key]

    def _fit_detection_bbox(
        self,
        depth_image: np.ndarray,
        det,
        intrinsics: dict,
    ) -> Optional[PersonBBox3D]:
        cam_idx = det.camera_id
        return fit_person_bbox(
            depth_image=depth_image,
            mask=det.mask,
            camera_id=cam_idx,
            fx=intrinsics["fx"],
            fy=intrinsics["fy"],
            cx=intrinsics["cx"],
            cy=intrinsics["cy"],
            camera_rotation_matrix=self._camera_rotations[cam_idx],
            camera_offset=self._camera_offsets[cam_idx],
            bbox_xyxy=det.bbox_xyxy,
            max_depth_m=self._camera_max_depths[cam_idx],
            erosion_kernel=self._erosion_kernel,
            subsample_step=self._subsample_step,
            outlier_threshold_m=self._outlier_threshold_m,
            extent_trim_percentile=DETECTION_EXTENT_TRIM_PERCENTILE,
            min_points=self._min_points,
            confidence=det.confidence,
            label=det.label,
            timestamp=det.frame_timestamp,
            class_id=det.class_id,
        )

    def _fit_bboxes(
        self,
        executor: ThreadPoolExecutor,
        fit_jobs: List[Tuple[np.ndarray, Any, dict]],
    ) -> List[PersonBBox3D]:
        if len(fit_jobs) == 1:
            depth_image, det, intrinsics = fit_jobs[0]
            box = self._fit_detection_bbox(depth_image, det, intrinsics)
            return [box] if box is not None else []

        futures = [
            executor.submit(self._fit_detection_bbox, depth_image, det, intrinsics)
            for depth_image, det, intrinsics in fit_jobs
        ]
        boxes: List[PersonBBox3D] = []
        for future in futures:
            try:
                box = future.result()
            except Exception:
                logging.exception("3D bounding box fit failed")
                continue
            if box is not None:
                boxes.append(box)
        return boxes

    def _replace_camera_detections(
        self,
        camera_ids: Tuple[int, ...],
        boxes: List[PersonBBox3D],
    ) -> None:
        boxes_by_camera: Dict[int, List[PersonBBox3D]] = {
            camera_id: [] for camera_id in camera_ids
        }
        for box in boxes:
            if box.camera_id in boxes_by_camera:
                boxes_by_camera[box.camera_id].append(box)

        with self._lock:
            # Empty lists intentionally clear only cameras processed for this
            # snapshot; stale cameras keep their last boxes.
            self._detections_by_camera.update(boxes_by_camera)

    def run(self) -> None:
        logging.info("Detection worker started")
        try:
            self._detector = YoloSegDetector(
                model_path=self._model_path,
                conf_threshold=self._conf_threshold,
                iou_threshold=self._iou_threshold,
                imgsz=self._imgsz,
                device=self._device,
                half_precision=self._half_precision,
                class_ids=self._class_ids,
                batch_size=self._batch_size,
                max_det=self._max_det,
                retina_masks=self._retina_masks,
            )
            self._detector.prepare(
                warmup_batch_size=self._batch_size
            )
        except Exception as exc:
            logging.error("Failed to prepare YOLO detector: %s", exc)
            return

        last_frame_tokens: Dict[int, Tuple[str, int, int]] = {}
        last_mode_generation = -1
        with ThreadPoolExecutor(
            max_workers=DETECTION_DEPTH_WORKERS,
            thread_name_prefix="DepthBBoxFit",
        ) as depth_fit_executor:
            while not self._should_stop():
                if not self.is_enabled():
                    self._wait(0.05)
                    continue

                with self._lock:
                    mode_generation = self._mode_generation
                if mode_generation != last_mode_generation:
                    last_frame_tokens.clear()
                    last_mode_generation = mode_generation

                latest_frames = self._receiver.snapshot_latest_frames(
                    self._detection_cameras
                )
                changed_frames: Dict[int, CameraFrame] = {}
                for cam_idx, frame in latest_frames.items():
                    frame_token = (
                        frame.camera_serial,
                        frame.sequence,
                        frame.capture_received_time_ns,
                    )
                    if last_frame_tokens.get(cam_idx) == frame_token:
                        continue
                    changed_frames[cam_idx] = frame
                    last_frame_tokens[cam_idx] = frame_token

                if not changed_frames:
                    self._wait(0.01)
                    continue

                camera_images: List[Tuple[int, np.ndarray]] = []
                depth_images: Dict[int, np.ndarray] = {}
                intrinsics_by_camera: Dict[int, dict] = {}
                for cam_idx, frame in changed_frames.items():
                    depth_image = frame.depth
                    camera_images.append((cam_idx, frame.color))
                    depth_images[cam_idx] = depth_image
                    h, w = depth_image.shape[:2]
                    intrinsics_by_camera[cam_idx] = self._get_intrinsics(w, h)

                if not camera_images:
                    self._wait(0.005)
                    continue

                try:
                    detections_2d = self._detector.detect_batch(camera_images)
                except Exception:
                    logging.exception("Batched detection failed")
                    detections_2d = []

                if self._should_stop():
                    break
                if not self.is_enabled():
                    continue

                fit_jobs = [
                    (
                        depth_images[det.camera_id],
                        det,
                        intrinsics_by_camera[det.camera_id],
                    )
                    for det in detections_2d
                    if det.camera_id in depth_images
                ]

                all_bboxes = (
                    self._fit_bboxes(depth_fit_executor, fit_jobs) if fit_jobs else []
                )
                if not self.is_enabled():
                    continue
                self._replace_camera_detections(tuple(changed_frames), all_bboxes)

                # Wait for a new frame without delaying an explicit stop request.
                self._wait(0.005)


class PoseDetectionWorker(threading.Thread):
    """Runs YOLO pose estimation and depth-to-3D keypoint fitting on changed per-camera snapshots."""

    def __init__(
        self,
        receiver: FrameReceiver,
        stop_event: threading.Event,
        detection_cameras: Tuple[int, ...],
        model_path: str,
        conf_threshold: float,
        iou_threshold: float,
        imgsz: int,
        device: str,
        half_precision: bool,
        batch_size: int,
        max_det: int,
        keypoint_conf_threshold: float,
        min_valid_keypoints: int,
        min_valid_connections: int,
        camera_rotations: list,
        camera_offsets: tuple,
        camera_max_depths: tuple,
        horizontal_fov_deg: float,
        vertical_fov_deg: float,
        initially_enabled: bool = False,
    ) -> None:
        super().__init__(name="PoseDetectionWorker", daemon=True)
        self._receiver = receiver
        self._app_stop_event = stop_event
        self._worker_stop_event = threading.Event()
        self._enabled_event = threading.Event()
        if initially_enabled:
            self._enabled_event.set()
        self._detection_cameras = detection_cameras

        # Detection parameters
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._imgsz = imgsz
        self._device = device
        self._half_precision = half_precision
        self._batch_size = batch_size
        self._max_det = max_det
        self._keypoint_conf_threshold = keypoint_conf_threshold
        self._min_valid_keypoints = min_valid_keypoints
        self._min_valid_connections = min_valid_connections

        # Camera parameters for back-projection
        self._camera_rotations = camera_rotations
        self._camera_offsets = camera_offsets
        self._camera_max_depths = camera_max_depths
        self._horizontal_fov_deg = horizontal_fov_deg
        self._vertical_fov_deg = vertical_fov_deg

        # Shared state
        self._lock = threading.Lock()
        self._poses_by_camera: Dict[int, List[PersonPose3D]] = {
            camera_id: [] for camera_id in detection_cameras
        }
        self._detector: Optional[YoloPosePersonDetector] = None
        self._intrinsics_cache: Dict[int, dict] = {}

        # Temporal smoothing (exponential moving average)
        self._smoothing_alpha = 0.4  # Higher = more responsive, lower = smoother
        self._prev_poses_by_camera: Dict[int, List[PersonPose3D]] = {}

    def get_poses(self) -> List[PersonPose3D]:
        """Thread-safe getter for latest pose detections."""
        with self._lock:
            poses: List[PersonPose3D] = []
            for camera_id in sorted(self._poses_by_camera):
                poses.extend(self._poses_by_camera[camera_id])
            return poses

    def request_stop(self) -> None:
        """Stop after any in-flight model call returns."""
        self._worker_stop_event.set()

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            self._enabled_event.set()
            return
        self._enabled_event.clear()
        self.clear_poses()

    def is_enabled(self) -> bool:
        return self._enabled_event.is_set()

    def clear_poses(self) -> None:
        with self._lock:
            for camera_id in self._poses_by_camera:
                self._poses_by_camera[camera_id] = []

    def _should_stop(self) -> bool:
        return self._worker_stop_event.is_set() or self._app_stop_event.is_set()

    def _wait(self, timeout_s: float) -> None:
        self._worker_stop_event.wait(timeout_s)

    def _get_intrinsics(self, width: int, height: int) -> dict:
        """Get or compute intrinsics for given resolution."""
        key = (width, height)
        if key not in self._intrinsics_cache:
            self._intrinsics_cache[key] = build_camera_intrinsics_from_fov(
                width, height,
                self._horizontal_fov_deg,
                self._vertical_fov_deg,
            )
        return self._intrinsics_cache[key]

    def _fit_pose_bbox(
        self,
        depth_image: np.ndarray,
        det,
        intrinsics: dict,
    ) -> Optional[PersonPose3D]:
        cam_idx = det.camera_id
        return fit_pose_bbox(
            depth_image=depth_image,
            keypoints_xy=det.keypoints_xy,
            keypoints_confidence=det.keypoints_confidence,
            camera_id=cam_idx,
            fx=intrinsics["fx"],
            fy=intrinsics["fy"],
            cx=intrinsics["cx"],
            cy=intrinsics["cy"],
            camera_rotation_matrix=self._camera_rotations[cam_idx],
            camera_offset=self._camera_offsets[cam_idx],
            max_depth_m=self._camera_max_depths[cam_idx],
            confidence_threshold=self._keypoint_conf_threshold,
            min_valid_keypoints=self._min_valid_keypoints,
            min_valid_connections=self._min_valid_connections,
            confidence=det.confidence,
            label=det.label,
            timestamp=det.frame_timestamp,
        )

    def _fit_bboxes(
        self,
        executor: ThreadPoolExecutor,
        fit_jobs: List[Tuple[np.ndarray, Any, dict]],
    ) -> List[PersonPose3D]:
        if len(fit_jobs) == 1:
            depth_image, det, intrinsics = fit_jobs[0]
            box = self._fit_pose_bbox(depth_image, det, intrinsics)
            return [box] if box is not None else []

        futures = [
            executor.submit(self._fit_pose_bbox, depth_image, det, intrinsics)
            for depth_image, det, intrinsics in fit_jobs
        ]
        boxes: List[PersonPose3D] = []
        for future in futures:
            try:
                box = future.result()
            except Exception:
                logging.exception("3D pose bounding box fit failed")
                continue
            if box is not None:
                boxes.append(box)
        return boxes

    def _replace_camera_detections(
        self,
        camera_ids: Tuple[int, ...],
        poses: List[PersonPose3D],
    ) -> None:
        poses_by_camera: Dict[int, List[PersonPose3D]] = {
            camera_id: [] for camera_id in camera_ids
        }
        for pose in poses:
            if pose.camera_id in poses_by_camera:
                poses_by_camera[pose.camera_id].append(pose)

        with self._lock:
            self._poses_by_camera.update(poses_by_camera)

    def run(self) -> None:
        logging.info("Pose detection worker started")
        try:
            self._detector = YoloPosePersonDetector(
                model_path=self._model_path,
                conf_threshold=self._conf_threshold,
                iou_threshold=self._iou_threshold,
                imgsz=self._imgsz,
                device=self._device,
                half_precision=self._half_precision,
                target_class_id=0,  # COCO person class
                batch_size=self._batch_size,
                max_det=self._max_det,
            )
            self._detector.prepare(
                warmup_batch_size=self._batch_size
            )
        except Exception as exc:
            logging.error("Failed to prepare YOLO pose detector: %s", exc)
            return

        last_frame_tokens: Dict[int, Tuple[str, int, int]] = {}
        with ThreadPoolExecutor(
            max_workers=POSE_DEPTH_WORKERS,
            thread_name_prefix="DepthPoseFit",
        ) as depth_fit_executor:
            while not self._should_stop():
                if not self.is_enabled():
                    self._wait(0.05)
                    continue

                latest_frames = self._receiver.snapshot_latest_frames(
                    self._detection_cameras
                )
                changed_frames: Dict[int, CameraFrame] = {}
                for cam_idx, frame in latest_frames.items():
                    frame_token = (
                        frame.camera_serial,
                        frame.sequence,
                        frame.capture_received_time_ns,
                    )
                    if last_frame_tokens.get(cam_idx) == frame_token:
                        continue
                    changed_frames[cam_idx] = frame
                    last_frame_tokens[cam_idx] = frame_token

                if not changed_frames:
                    self._wait(0.01)
                    continue

                camera_images: List[Tuple[int, np.ndarray]] = []
                depth_images: Dict[int, np.ndarray] = {}
                intrinsics_by_camera: Dict[int, dict] = {}
                for cam_idx, frame in changed_frames.items():
                    depth_image = frame.depth
                    camera_images.append((cam_idx, frame.color))
                    depth_images[cam_idx] = depth_image
                    h, w = depth_image.shape[:2]
                    intrinsics_by_camera[cam_idx] = self._get_intrinsics(w, h)

                if not camera_images:
                    self._wait(0.005)
                    continue

                try:
                    detections_2d = self._detector.detect_batch(camera_images)
                except Exception:
                    logging.exception("Batched pose detection failed")
                    detections_2d = []

                if self._should_stop():
                    break
                if not self.is_enabled():
                    continue

                fit_jobs = [
                    (
                        depth_images[det.camera_id],
                        det,
                        intrinsics_by_camera[det.camera_id],
                    )
                    for det in detections_2d
                    if det.camera_id in depth_images
                ]

                all_poses = (
                    self._fit_bboxes(depth_fit_executor, fit_jobs) if fit_jobs else []
                )
                if not self.is_enabled():
                    continue

                self._replace_camera_detections(tuple(changed_frames), all_poses)

                # Wait for a new frame without delaying an explicit stop request.
                self._wait(0.005)


class VLMDetectionWorker(threading.Thread):
    """Runs slow semantic VLM localization without blocking the render loop."""

    def __init__(
        self,
        receiver: FrameReceiver,
        stop_event: threading.Event,
        detection_cameras: Tuple[int, ...],
        base_url: str,
        api_key: str,
        model: str,
        request_rate_hz: float,
        timeout_s: float,
        max_tokens: int,
        jpeg_quality: int,
        image_max_side_px: int,
        max_objects: int,
        conf_threshold: float,
        min_box_side_px: float,
        stale_ttl_s: float,
        erosion_kernel: int,
        subsample_step: int,
        outlier_threshold_m: float,
        min_points: int,
        camera_rotations: list,
        camera_offsets: tuple,
        camera_max_depths: tuple,
        horizontal_fov_deg: float,
        vertical_fov_deg: float,
        initially_enabled: bool = False,
    ) -> None:
        super().__init__(name="VLMDetectionWorker", daemon=True)
        self._receiver = receiver
        self._app_stop_event = stop_event
        self._worker_stop_event = threading.Event()
        self._enabled_event = threading.Event()
        if initially_enabled:
            self._enabled_event.set()
        self._detection_cameras = detection_cameras

        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._request_interval_s = 1.0 / max(0.05, float(request_rate_hz))
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens
        self._jpeg_quality = jpeg_quality
        self._image_max_side_px = image_max_side_px
        self._max_objects = max_objects
        self._conf_threshold = conf_threshold
        self._min_box_side_px = min_box_side_px
        self._stale_ttl_s = max(0.0, float(stale_ttl_s))

        self._erosion_kernel = erosion_kernel
        self._subsample_step = subsample_step
        self._outlier_threshold_m = outlier_threshold_m
        self._min_points = min_points

        self._camera_rotations = camera_rotations
        self._camera_offsets = camera_offsets
        self._camera_max_depths = camera_max_depths
        self._horizontal_fov_deg = horizontal_fov_deg
        self._vertical_fov_deg = vertical_fov_deg

        self._lock = threading.Lock()
        self._detections_by_camera: Dict[int, List[PersonBBox3D]] = {
            camera_id: [] for camera_id in detection_cameras
        }
        self._last_success_monotonic: Dict[int, float] = {}
        self._intrinsics_cache: Dict[int, dict] = {}
        self._detector: Optional[OpenAIVLMObjectDetector] = None

    def get_detections(self) -> List[PersonBBox3D]:
        """Return non-stale VLM 3D boxes from all cameras."""
        with self._lock:
            self._expire_stale_locked(time.monotonic())
            detections: List[PersonBBox3D] = []
            for camera_id in sorted(self._detections_by_camera):
                detections.extend(self._detections_by_camera[camera_id])
            return detections

    def request_stop(self) -> None:
        self._worker_stop_event.set()

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            self._enabled_event.set()
            return
        self._enabled_event.clear()
        self.clear_detections()

    def is_enabled(self) -> bool:
        return self._enabled_event.is_set()

    def clear_detections(self) -> None:
        with self._lock:
            for camera_id in self._detections_by_camera:
                self._detections_by_camera[camera_id] = []
            self._last_success_monotonic.clear()

    def _should_stop(self) -> bool:
        return self._worker_stop_event.is_set() or self._app_stop_event.is_set()

    def _wait(self, timeout_s: float) -> None:
        self._worker_stop_event.wait(timeout_s)

    @staticmethod
    def _frame_token(frame: CameraFrame) -> Tuple[str, int, int]:
        return (
            frame.camera_serial,
            frame.sequence,
            frame.capture_received_time_ns,
        )

    def _select_next_changed_frame(
        self,
        latest_frames: Dict[int, CameraFrame],
        last_frame_tokens: Dict[int, Tuple[str, int, int]],
        cursor: int,
    ) -> Tuple[Optional[CameraFrame], int]:
        if not self._detection_cameras:
            return None, 0

        for offset in range(len(self._detection_cameras)):
            index = (cursor + offset) % len(self._detection_cameras)
            camera_id = self._detection_cameras[index]
            frame = latest_frames.get(camera_id)
            if frame is None:
                continue
            if last_frame_tokens.get(camera_id) == self._frame_token(frame):
                continue
            return frame, (index + 1) % len(self._detection_cameras)
        return None, (cursor + 1) % len(self._detection_cameras)

    def _get_intrinsics(self, width: int, height: int) -> dict:
        key = (width, height)
        if key not in self._intrinsics_cache:
            self._intrinsics_cache[key] = build_camera_intrinsics_from_fov(
                width,
                height,
                self._horizontal_fov_deg,
                self._vertical_fov_deg,
            )
        return self._intrinsics_cache[key]

    def _fit_detection_bbox(
        self,
        depth_image: np.ndarray,
        det: VLMDetection2D,
        intrinsics: dict,
    ) -> Optional[PersonBBox3D]:
        camera_id = det.camera_id
        return fit_bbox_only(
            depth_image=depth_image,
            bbox_xyxy=det.bbox_xyxy,
            camera_id=camera_id,
            fx=intrinsics["fx"],
            fy=intrinsics["fy"],
            cx=intrinsics["cx"],
            cy=intrinsics["cy"],
            camera_rotation_matrix=self._camera_rotations[camera_id],
            camera_offset=self._camera_offsets[camera_id],
            max_depth_m=self._camera_max_depths[camera_id],
            erosion_kernel=self._erosion_kernel,
            subsample_step=self._subsample_step,
            outlier_threshold_m=self._outlier_threshold_m,
            extent_trim_percentile=VLM_EXTENT_TRIM_PERCENTILE,
            min_points=self._min_points,
            confidence=det.confidence,
            label=det.label,
            timestamp=det.frame_timestamp,
            source="vlm",
        )

    def _fit_bboxes(
        self,
        executor: ThreadPoolExecutor,
        depth_image: np.ndarray,
        detections: List[VLMDetection2D],
        intrinsics: dict,
    ) -> List[PersonBBox3D]:
        if len(detections) == 1:
            box = self._fit_detection_bbox(depth_image, detections[0], intrinsics)
            return [box] if box is not None else []

        futures = [
            executor.submit(self._fit_detection_bbox, depth_image, det, intrinsics)
            for det in detections
        ]
        boxes: List[PersonBBox3D] = []
        for future in futures:
            try:
                box = future.result()
            except Exception:
                logging.exception("VLM 3D bounding box fit failed")
                continue
            if box is not None:
                boxes.append(box)
        return boxes

    def _replace_camera_detections(
        self,
        camera_id: int,
        boxes: List[PersonBBox3D],
        *,
        success_monotonic_s: Optional[float] = None,
    ) -> None:
        with self._lock:
            if camera_id not in self._detections_by_camera:
                return
            self._detections_by_camera[camera_id] = list(boxes)
            self._last_success_monotonic[camera_id] = (
                time.monotonic()
                if success_monotonic_s is None
                else float(success_monotonic_s)
            )

    def _expire_stale_locked(self, now_s: float) -> None:
        if self._stale_ttl_s <= 0.0:
            return
        for camera_id, last_success_s in tuple(self._last_success_monotonic.items()):
            if now_s - last_success_s <= self._stale_ttl_s:
                continue
            self._detections_by_camera[camera_id] = []
            del self._last_success_monotonic[camera_id]

    def run(self) -> None:
        logging.info("VLM detection worker started for %s", self._model)
        try:
            self._detector = OpenAIVLMObjectDetector(
                base_url=self._base_url,
                api_key=self._api_key,
                model=self._model,
                timeout_s=self._timeout_s,
                max_tokens=self._max_tokens,
                jpeg_quality=self._jpeg_quality,
                image_max_side_px=self._image_max_side_px,
                max_objects=self._max_objects,
                conf_threshold=self._conf_threshold,
                min_box_side_px=self._min_box_side_px,
            )
        except Exception as exc:
            logging.error("Failed to configure VLM detector: %s", exc)
            return

        last_frame_tokens: Dict[int, Tuple[str, int, int]] = {}
        camera_cursor = 0
        next_request_s = 0.0
        last_failure_log_s = 0.0

        with ThreadPoolExecutor(
            max_workers=VLM_DEPTH_WORKERS,
            thread_name_prefix="VLMDepthBBoxFit",
        ) as depth_fit_executor:
            while not self._should_stop():
                if not self.is_enabled():
                    self._wait(0.05)
                    continue

                now_s = time.monotonic()
                if now_s < next_request_s:
                    self._wait(min(0.05, next_request_s - now_s))
                    continue

                latest_frames = self._receiver.snapshot_latest_frames(
                    self._detection_cameras
                )
                frame, camera_cursor = self._select_next_changed_frame(
                    latest_frames,
                    last_frame_tokens,
                    camera_cursor,
                )
                if frame is None:
                    self._wait(0.05)
                    continue

                frame_token = self._frame_token(frame)
                last_frame_tokens[frame.camera_id] = frame_token
                request_started_s = time.monotonic()
                next_request_s = request_started_s + self._request_interval_s
                try:
                    detections_2d = self._detector.detect(
                        frame.color,
                        camera_id=frame.camera_id,
                        frame_token=frame_token,
                    )
                except VLMDetectorError as exc:
                    now_s = time.monotonic()
                    if now_s - last_failure_log_s >= 5.0:
                        logging.warning(
                            "VLM camera %d localization failed: %s",
                            frame.camera_id,
                            exc,
                        )
                        last_failure_log_s = now_s
                    else:
                        logging.debug(
                            "VLM camera %d localization failed: %s",
                            frame.camera_id,
                            exc,
                        )
                    continue
                except Exception as exc:
                    now_s = time.monotonic()
                    if now_s - last_failure_log_s >= 5.0:
                        logging.exception(
                            "Unexpected VLM camera %d localization failure: %s",
                            frame.camera_id,
                            exc,
                        )
                        last_failure_log_s = now_s
                    else:
                        logging.debug(
                            "Unexpected VLM camera %d localization failure: %s",
                            frame.camera_id,
                            exc,
                        )
                    continue

                if self._should_stop():
                    break
                if not self.is_enabled():
                    continue

                height, width = frame.depth.shape[:2]
                intrinsics = self._get_intrinsics(width, height)
                boxes = self._fit_bboxes(
                    depth_fit_executor,
                    frame.depth,
                    detections_2d,
                    intrinsics,
                )
                if not self.is_enabled():
                    continue

                self._replace_camera_detections(frame.camera_id, boxes)


SCENE_OVERLAY_VERTEX_SHADER = """
#version 330
in vec2 in_position;
in vec2 in_uv;
out vec2 frag_uv;
void main() {
    frag_uv = in_uv;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

SCENE_OVERLAY_FRAGMENT_SHADER = """
#version 330
uniform sampler2D overlay_tex;
in vec2 frag_uv;
out vec4 color;
void main() {
    color = texture(overlay_tex, frag_uv);
}
"""


@dataclass
class _SceneDescriptionPanel:
    texture: moderngl.Texture
    size_px: Tuple[int, int]
    signature: Tuple[int, int, str, bool, Optional[str]]


@dataclass
class _ControlPanelTexture:
    texture: moderngl.Texture
    size_px: Tuple[int, int]
    signature: Tuple[int, int, Tuple[Tuple[str, str, str, bool], ...]]


def _control_panel_height(row_count: int) -> int:
    if row_count <= 0:
        return (2 * CONTROL_PANEL_PADDING_PX) + CONTROL_PANEL_TITLE_HEIGHT_PX
    return (
        (2 * CONTROL_PANEL_PADDING_PX)
        + CONTROL_PANEL_TITLE_HEIGHT_PX
        + CONTROL_PANEL_TITLE_GAP_PX
        + (row_count * CONTROL_PANEL_ROW_HEIGHT_PX)
        + ((row_count - 1) * CONTROL_PANEL_ROW_GAP_PX)
    )


def _control_panel_rect(row_count: int) -> Tuple[int, int, int, int]:
    return (
        CONTROL_PANEL_LEFT_PX,
        CONTROL_PANEL_TOP_PX,
        CONTROL_PANEL_WIDTH_PX,
        _control_panel_height(row_count),
    )


def _control_panel_row_rects(row_keys: Tuple[str, ...]) -> Dict[str, Tuple[int, int, int, int]]:
    left, top, width, _height = _control_panel_rect(len(row_keys))
    row_top = (
        top
        + CONTROL_PANEL_PADDING_PX
        + CONTROL_PANEL_TITLE_HEIGHT_PX
        + CONTROL_PANEL_TITLE_GAP_PX
    )
    row_left = left + CONTROL_PANEL_PADDING_PX
    row_width = width - (2 * CONTROL_PANEL_PADDING_PX)
    rects: Dict[str, Tuple[int, int, int, int]] = {}
    for index, key in enumerate(row_keys):
        y = row_top + index * (CONTROL_PANEL_ROW_HEIGHT_PX + CONTROL_PANEL_ROW_GAP_PX)
        rects[key] = (row_left, y, row_width, CONTROL_PANEL_ROW_HEIGHT_PX)
    return rects


class ControlPanelOverlay:
    """OpenGL text panel for viewer feature toggles."""

    def __init__(self, ctx: moderngl.Context, window_size: Tuple[int, int]) -> None:
        self._ctx = ctx
        self._window_size = window_size
        self._program = self._ctx.program(
            vertex_shader=SCENE_OVERLAY_VERTEX_SHADER,
            fragment_shader=SCENE_OVERLAY_FRAGMENT_SHADER,
        )
        self._texture_uniform = self._program["overlay_tex"]
        self._texture_uniform.value = 3
        self._vbo = self._ctx.buffer(reserve=6 * 4 * 4)
        self._vao = self._ctx.vertex_array(
            self._program,
            [(self._vbo, "2f 2f", "in_position", "in_uv")],
        )
        pygame.font.init()
        self._font = pygame.font.Font(None, CONTROL_PANEL_FONT_SIZE_PX)
        self._title_font = pygame.font.Font(None, CONTROL_PANEL_TITLE_FONT_SIZE_PX)
        self._panel: Optional[_ControlPanelTexture] = None

    def release(self) -> None:
        if self._panel is not None:
            with contextlib.suppress(Exception):
                self._panel.texture.release()
            self._panel = None
        for gl_obj in (self._vao, self._vbo, self._program):
            with contextlib.suppress(Exception):
                gl_obj.release()

    def render(self, rows: List[Tuple[str, str, str, bool]]) -> None:
        rows_tuple = tuple(rows)
        if not rows_tuple:
            return
        rect = _control_panel_rect(len(rows_tuple))
        panel = self._get_or_build_panel(rows_tuple, rect[2], rect[3])

        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        try:
            panel.texture.use(location=3)
            self._vbo.write(self._quad_vertices(rect).tobytes())
            self._vao.render(moderngl.TRIANGLES, vertices=6)
        finally:
            self._ctx.disable(moderngl.BLEND)
            self._ctx.enable(moderngl.DEPTH_TEST)

    def _get_or_build_panel(
        self,
        rows: Tuple[Tuple[str, str, str, bool], ...],
        width: int,
        height: int,
    ) -> _ControlPanelTexture:
        signature = (width, height, rows)
        if self._panel is not None and self._panel.signature == signature:
            return self._panel
        if self._panel is not None:
            with contextlib.suppress(Exception):
                self._panel.texture.release()
        texture = self._build_panel_texture(rows, width, height)
        self._panel = _ControlPanelTexture(texture, (width, height), signature)
        return self._panel

    def _build_panel_texture(
        self,
        rows: Tuple[Tuple[str, str, str, bool], ...],
        width: int,
        height: int,
    ) -> moderngl.Texture:
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface.fill((6, 10, 12, 218))
        pygame.draw.rect(surface, (92, 220, 205, 210), surface.get_rect(), width=1, border_radius=4)

        padding = CONTROL_PANEL_PADDING_PX
        title_surface = self._title_font.render("Controls", True, (250, 252, 246))
        title_y = padding + max(0, (CONTROL_PANEL_TITLE_HEIGHT_PX - title_surface.get_height()) // 2)
        surface.blit(title_surface, (padding, title_y))

        row_rects = _control_panel_row_rects(tuple(row[0] for row in rows))
        panel_left = CONTROL_PANEL_LEFT_PX
        panel_top = CONTROL_PANEL_TOP_PX
        for key, label, value, enabled in rows:
            row_left, row_top, row_width, row_height = row_rects[key]
            local_rect = pygame.Rect(
                row_left - panel_left,
                row_top - panel_top,
                row_width,
                row_height,
            )
            row_color = (18, 25, 28, 190) if enabled else (18, 22, 24, 160)
            pygame.draw.rect(surface, row_color, local_rect, border_radius=4)

            switch_rect = pygame.Rect(
                local_rect.right - CONTROL_PANEL_SWITCH_WIDTH_PX - 7,
                local_rect.top + (row_height - CONTROL_PANEL_SWITCH_HEIGHT_PX) // 2,
                CONTROL_PANEL_SWITCH_WIDTH_PX,
                CONTROL_PANEL_SWITCH_HEIGHT_PX,
            )
            label_max_width = max(20, switch_rect.left - local_rect.left - 12)
            label_text = self._ellipsize(label, label_max_width, self._font)
            label_surface = self._font.render(label_text, True, (220, 238, 232))
            label_y = local_rect.top + (row_height - label_surface.get_height()) // 2
            surface.blit(label_surface, (local_rect.left + 9, label_y))

            track_color = (42, 174, 122, 235) if enabled else (72, 82, 88, 230)
            pygame.draw.rect(surface, track_color, switch_rect, border_radius=CONTROL_PANEL_SWITCH_HEIGHT_PX // 2)
            knob_radius = (CONTROL_PANEL_SWITCH_HEIGHT_PX - 4) // 2
            knob_x = switch_rect.right - knob_radius - 2 if enabled else switch_rect.left + knob_radius + 2
            pygame.draw.circle(
                surface,
                (246, 248, 241),
                (knob_x, switch_rect.centery),
                knob_radius,
            )

            switch_text = value.upper()
            switch_surface = self._font.render(switch_text, True, (7, 12, 14))
            if enabled:
                text_x = switch_rect.left + 8
            else:
                text_x = switch_rect.right - switch_surface.get_width() - 8
            text_y = switch_rect.top + (switch_rect.height - switch_surface.get_height()) // 2
            surface.blit(switch_surface, (text_x, text_y))

        texture = self._ctx.texture(
            (width, height),
            4,
            pygame.image.tobytes(surface, "RGBA", True),
            alignment=1,
        )
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        return texture

    @staticmethod
    def _ellipsize(text: str, max_width: int, font: pygame.font.Font) -> str:
        suffix = "..."
        if font.size(text)[0] <= max_width:
            return text
        trimmed = text
        while trimmed and font.size(trimmed + suffix)[0] > max_width:
            trimmed = trimmed[:-1]
        return (trimmed + suffix) if trimmed else suffix

    def _quad_vertices(self, rect: Tuple[int, int, int, int]) -> np.ndarray:
        left, top, width, height = rect
        viewport_w, viewport_h = self._window_size
        right = left + width
        bottom = top + height
        x1 = (2.0 * left / viewport_w) - 1.0
        x2 = (2.0 * right / viewport_w) - 1.0
        y1 = 1.0 - (2.0 * top / viewport_h)
        y2 = 1.0 - (2.0 * bottom / viewport_h)
        return np.array(
            [
                [x1, y1, 0.0, 1.0],
                [x2, y1, 1.0, 1.0],
                [x2, y2, 1.0, 0.0],
                [x1, y1, 0.0, 1.0],
                [x2, y2, 1.0, 0.0],
                [x1, y2, 0.0, 0.0],
            ],
            dtype=np.float32,
        )


class SceneDescriptionOverlay:
    """OpenGL text overlay for streaming per-camera scene descriptions."""

    def __init__(
        self,
        ctx: moderngl.Context,
        window_size: Tuple[int, int],
        camera_ids: Tuple[int, ...],
    ) -> None:
        self._ctx = ctx
        self._window_size = window_size
        self._camera_ids = tuple(camera_ids)
        self._program = self._ctx.program(
            vertex_shader=SCENE_OVERLAY_VERTEX_SHADER,
            fragment_shader=SCENE_OVERLAY_FRAGMENT_SHADER,
        )
        self._texture_uniform = self._program["overlay_tex"]
        self._texture_uniform.value = 2
        self._vbo = self._ctx.buffer(reserve=6 * 4 * 4)
        self._vao = self._ctx.vertex_array(
            self._program,
            [(self._vbo, "2f 2f", "in_position", "in_uv")],
        )
        pygame.font.init()
        self._font = pygame.font.Font(None, SCENE_DESCRIPTION_FONT_SIZE_PX)
        self._title_font = pygame.font.Font(None, SCENE_DESCRIPTION_TITLE_FONT_SIZE_PX)
        self._panels: Dict[int, _SceneDescriptionPanel] = {}

    def release(self) -> None:
        for panel in self._panels.values():
            with contextlib.suppress(Exception):
                panel.texture.release()
        self._panels.clear()
        for gl_obj in (self._vao, self._vbo, self._program):
            with contextlib.suppress(Exception):
                gl_obj.release()

    def render(self, descriptions: Dict[int, SceneDescription]) -> None:
        camera_ids = self._camera_ids or tuple(sorted(descriptions))
        if not camera_ids:
            return

        rects = self._panel_rects(camera_ids)
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        try:
            for camera_id in camera_ids:
                rect = rects[camera_id]
                panel = self._get_or_build_panel(camera_id, descriptions.get(camera_id), rect)
                panel.texture.use(location=2)
                self._vbo.write(self._quad_vertices(rect).tobytes())
                self._vao.render(moderngl.TRIANGLES, vertices=6)
        finally:
            self._ctx.disable(moderngl.BLEND)
            self._ctx.enable(moderngl.DEPTH_TEST)

    def _panel_rects(self, camera_ids: Tuple[int, ...]) -> Dict[int, Tuple[int, int, int, int]]:
        viewport_w, viewport_h = self._window_size
        count = len(camera_ids)
        cols = 2 if viewport_w >= 900 and count > 1 else 1
        rows = (count + cols - 1) // cols
        margin = SCENE_DESCRIPTION_PANEL_MARGIN_PX
        gap = SCENE_DESCRIPTION_PANEL_GAP_PX
        panel_w = max(
            200,
            (viewport_w - (2 * margin) - ((cols - 1) * gap)) // cols,
        )
        panel_h = SCENE_DESCRIPTION_PANEL_HEIGHT_PX
        total_h = (rows * panel_h) + ((rows - 1) * gap)
        top_start = max(margin, viewport_h - margin - total_h)

        rects: Dict[int, Tuple[int, int, int, int]] = {}
        for index, camera_id in enumerate(camera_ids):
            row = index // cols
            col = index % cols
            left = margin + col * (panel_w + gap)
            top = top_start + row * (panel_h + gap)
            rects[camera_id] = (left, top, panel_w, panel_h)
        return rects

    def _get_or_build_panel(
        self,
        camera_id: int,
        description: Optional[SceneDescription],
        rect: Tuple[int, int, int, int],
    ) -> _SceneDescriptionPanel:
        _left, _top, width, height = rect
        signature = self._panel_signature(camera_id, description, width, height)
        panel = self._panels.get(camera_id)
        if panel is not None and panel.signature == signature:
            return panel

        if panel is not None:
            with contextlib.suppress(Exception):
                panel.texture.release()
        texture = self._build_panel_texture(camera_id, description, width, height)
        panel = _SceneDescriptionPanel(texture, (width, height), signature)
        self._panels[camera_id] = panel
        return panel

    @staticmethod
    def _panel_signature(
        camera_id: int,
        description: Optional[SceneDescription],
        width: int,
        height: int,
    ) -> Tuple[int, int, str, bool, Optional[str]]:
        if description is None:
            return (width, height, f"camera-{camera_id}:waiting", False, None)
        return (
            width,
            height,
            description.description,
            bool(description.is_final),
            description.error,
        )

    def _build_panel_texture(
        self,
        camera_id: int,
        description: Optional[SceneDescription],
        width: int,
        height: int,
    ) -> moderngl.Texture:
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface.fill((5, 9, 12, 210))
        border_color = (252, 196, 61, 230)
        if description is not None and description.error:
            border_color = (255, 90, 80, 235)
        elif description is not None and not description.is_final:
            border_color = (80, 205, 255, 235)
        pygame.draw.rect(surface, border_color, surface.get_rect(), width=1, border_radius=3)

        padding = SCENE_DESCRIPTION_PANEL_PADDING_PX
        title = f"Cam {camera_id}"
        if description is None:
            status = "waiting"
        elif description.error:
            status = "error"
        elif description.is_final:
            status = f"{description.latency_ms:.0f} ms"
        else:
            status = "streaming"
        title_surface = self._title_font.render(f"{title}  {status}", True, (250, 252, 246))
        surface.blit(title_surface, (padding, padding))

        body = "Waiting for frame..."
        if description is not None:
            body = description.description.strip() or "Starting scene stream..."
            if not description.is_final and not description.error:
                body = f"{body} |"
        max_text_w = max(20, width - (2 * padding))
        line_height = self._font.get_linesize()
        y = padding + title_surface.get_height() + 4
        max_lines = max(1, (height - y - padding) // line_height)
        lines = self._wrap_text(body, max_text_w)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = self._ellipsize(lines[-1], max_text_w)
        for line in lines:
            text_surface = self._font.render(line, True, (216, 240, 232))
            surface.blit(text_surface, (padding, y))
            y += line_height

        texture = self._ctx.texture(
            (width, height),
            4,
            pygame.image.tobytes(surface, "RGBA", True),
            alignment=1,
        )
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        return texture

    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        lines: List[str] = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}" if current else word
            if self._font.size(candidate)[0] <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = word
            while self._font.size(current)[0] > max_width and len(current) > 1:
                split_at = max(1, len(current) - 1)
                while split_at > 1 and self._font.size(current[:split_at])[0] > max_width:
                    split_at -= 1
                lines.append(current[:split_at])
                current = current[split_at:]
        if current:
            lines.append(current)
        return lines or [""]

    def _ellipsize(self, text: str, max_width: int) -> str:
        suffix = "..."
        if self._font.size(text)[0] <= max_width:
            return text
        trimmed = text
        while trimmed and self._font.size(trimmed + suffix)[0] > max_width:
            trimmed = trimmed[:-1]
        return (trimmed + suffix) if trimmed else suffix

    def _quad_vertices(self, rect: Tuple[int, int, int, int]) -> np.ndarray:
        left, top, width, height = rect
        viewport_w, viewport_h = self._window_size
        right = left + width
        bottom = top + height
        x1 = (2.0 * left / viewport_w) - 1.0
        x2 = (2.0 * right / viewport_w) - 1.0
        y1 = 1.0 - (2.0 * top / viewport_h)
        y2 = 1.0 - (2.0 * bottom / viewport_h)
        return np.array(
            [
                [x1, y1, 0.0, 1.0],
                [x2, y1, 1.0, 1.0],
                [x2, y2, 1.0, 0.0],
                [x1, y1, 0.0, 1.0],
                [x2, y2, 1.0, 0.0],
                [x1, y2, 0.0, 0.0],
            ],
            dtype=np.float32,
        )


class PointCloudApp:
    """Main rendering loop for the point cloud viewer."""

    def __init__(self, receiver: FrameReceiver, metrics: Metrics, stop_event: threading.Event) -> None:
        self._receiver = receiver
        self._metrics = metrics
        self._stop_event = stop_event
        self._clock = pygame.time.Clock()
        self._last_metrics_time = time.time()
        self._camera_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        self._camera_rot = np.array([0.0, 0.0], dtype=np.float32)
        self._camera_zoom = 1.0

        self._ctx: Optional[moderngl.Context] = None
        self._program: Optional[moderngl.Program] = None
        self._view_uniform = None
        self._projection_uniform = None
        self._depth_texture_uniform = None
        self._color_texture_uniform = None
        self._camera_rotation_uniform = None
        self._camera_offset_uniform = None
        self._fx_uniform = None
        self._fy_uniform = None
        self._cx_uniform = None
        self._cy_uniform = None
        self._max_depth_uniform = None
        self._color_scale_uniform = None
        self._point_size_uniform = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._depth_textures: list[moderngl.Texture] = []
        self._color_textures: list[moderngl.Texture] = []
        self._frame_resolution: Optional[Tuple[int, int]] = None
        self._vertex_count = 0
        self._active_camera_ids: set[int] = set()
        self._last_render_frames: Dict[int, CameraFrame] = {}
        self._camera_rotation_matrices = [build_yaw_rotation_matrix(rotation) for rotation in CAMERA_ROTATIONS]
        self._visualization_mode = VISUALIZATION_HEATMAP

        # Segmentation detection state
        self._detection_enabled = DETECTION_ENABLED
        self._object_detection_enabled = OBJECT_DETECTION_ENABLED
        self._bbox_renderer: Optional[BBoxRenderer] = None
        self._detection_worker: Optional[DetectionWorker] = None
        self._detection_lock = threading.Lock()

        # Pose estimation state
        self._pose_enabled = POSE_ENABLED
        self._pose_worker: Optional[PoseDetectionWorker] = None

        # Low-rate VLM semantic object localization state
        self._vlm_enabled = VLM_ENABLED
        self._vlm_workers: List[VLMDetectionWorker] = []

        # VLM scene description state (natural language descriptions)
        self._scene_description_enabled = SCENE_DESCRIPTION_ENABLED
        self._scene_description_worker: Optional[SceneDescriptionWorker] = None
        self._scene_description_overlay: Optional[SceneDescriptionOverlay] = None
        self._control_panel_overlay: Optional[ControlPanelOverlay] = None

        self._initialize_pygame()
        self._initialize_opengl()
        self._start_detection_worker()
        self._start_pose_worker()
        self._start_vlm_workers()
        self._start_scene_description_worker()

    def run(self) -> None:
        logging.info("Entering render loop")
        try:
            while not self._stop_event.is_set():
                dt = self._clock.tick(TARGET_FPS) / 1000.0
                self._process_events()
                self._update_camera(dt)
                pending_frames = self._receiver.pop_pending_frames()
                for frame in pending_frames.values():
                    self._upload_camera_frame(frame)
                if self._vao is None and self._last_render_frames:
                    self._restore_render_frames()
                self._update_matrices()
                self._render_frame()
                self._log_metrics()
        except Exception:
            logging.exception("Fatal error in render loop")
            self._request_shutdown()
        finally:
            self._stop_detection_worker()
            self._stop_pose_worker()
            self._stop_vlm_workers()
            self._cleanup()

    def _initialize_pygame(self) -> None:
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)
        self._update_window_caption()

    def _initialize_opengl(self) -> None:
        self._ctx = moderngl.create_context()
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        vertex_shader = """
        #version 330
        in vec2 in_pixel;
        out vec3 fragColor;
        uniform mat4 view;
        uniform mat4 projection;
        uniform usampler2D depth_tex;
        uniform sampler2D color_tex;
        uniform mat3 camera_rotation;
        uniform vec3 camera_offset;
        uniform float fx;
        uniform float fy;
        uniform float cx;
        uniform float cy;
        uniform float max_depth;
        uniform float color_scale_max_depth;
        uniform float point_size;
        uniform int visualization_mode;

        vec3 jet(float t) {
            float r = clamp(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0);
            float g = clamp(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0);
            float b = clamp(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0);
            return vec3(r, g, b);
        }

        void main() {
            ivec2 pixel = ivec2(in_pixel);
            uint depth_mm = texelFetch(depth_tex, pixel, 0).r;
            float depth_m = float(depth_mm) * 0.001;
            if (depth_mm == 0u || depth_m >= max_depth) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                gl_PointSize = 0.0;
                fragColor = vec3(0.0);
                return;
            }

            float x = (in_pixel.x - cx) * depth_m / fx;
            float y = -(in_pixel.y - cy) * depth_m / fy;
            vec3 world_position = camera_rotation * vec3(x, y, depth_m) + camera_offset;

            gl_Position = projection * view * vec4(world_position, 1.0);
            gl_PointSize = point_size;
            if (visualization_mode == 1) {
                fragColor = texelFetch(color_tex, pixel, 0).rgb;
            } else {
                float color_t = 1.0 - clamp(depth_m / color_scale_max_depth, 0.0, 1.0);
                fragColor = jet(color_t);
            }
        }
        """
        fragment_shader = """
        #version 330
        in vec3 fragColor;
        out vec4 color;
        void main() {
            color = vec4(fragColor, 1.0);
        }
        """
        self._program = self._ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        self._view_uniform = self._program["view"]
        self._projection_uniform = self._program["projection"]
        self._depth_texture_uniform = self._program["depth_tex"]
        self._color_texture_uniform = self._program["color_tex"]
        self._camera_rotation_uniform = self._program["camera_rotation"]
        self._camera_offset_uniform = self._program["camera_offset"]
        self._fx_uniform = self._program["fx"]
        self._fy_uniform = self._program["fy"]
        self._cx_uniform = self._program["cx"]
        self._cy_uniform = self._program["cy"]
        self._max_depth_uniform = self._program["max_depth"]
        self._color_scale_uniform = self._program["color_scale_max_depth"]
        self._point_size_uniform = self._program["point_size"]
        self._depth_texture_uniform.value = 0
        self._color_texture_uniform.value = 1
        self._color_scale_uniform.value = COLOR_SCALE_MAX_DEPTH_METERS
        self._point_size_uniform.value = POINT_SIZE_PIXELS
        self._program["visualization_mode"].value = self._visualization_mode

    def _process_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._request_shutdown()
            elif event.type == pygame.KEYDOWN and event.key == K_ESCAPE:
                self._request_shutdown()
            elif event.type == pygame.KEYDOWN and event.key == K_t:
                self._toggle_visualization_mode()
            elif event.type == pygame.KEYDOWN and event.key == K_h:
                self._toggle_detection()
            elif event.type == pygame.KEYDOWN and event.key == K_o:
                self._toggle_object_detection()
            elif event.type == pygame.KEYDOWN and event.key == K_p:
                self._toggle_pose()
            elif event.type == pygame.KEYDOWN and event.key == K_v:
                self._toggle_vlm()
            elif event.type == pygame.KEYDOWN and event.key == K_l:
                self._toggle_scene_description()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and self._handle_control_panel_click(event.pos):
                    continue
                elif event.button == 4:
                    self._camera_zoom *= 1.0 + ZOOM_STEP
                elif event.button == 5:
                    self._camera_zoom *= 1.0 - ZOOM_STEP
                self._camera_zoom = float(np.clip(self._camera_zoom, MIN_ZOOM, MAX_ZOOM))

    def _request_shutdown(self) -> None:
        self._stop_event.set()
        self._receiver.stop()

    def _update_camera(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        shift_multiplier = 2.0 if keys[K_LSHIFT] else 1.0
        step = MOVEMENT_SPEED * dt * shift_multiplier
        if step > 0:
            cos_pitch = np.cos(self._camera_rot[0])
            sin_pitch = np.sin(self._camera_rot[0])
            cos_yaw = np.cos(self._camera_rot[1])
            sin_yaw = np.sin(self._camera_rot[1])
            forward = -np.array(
                [sin_yaw * cos_pitch, -sin_pitch, cos_yaw * cos_pitch],
                dtype=np.float32,
            )
            right = np.array([cos_yaw, 0.0, -sin_yaw], dtype=np.float32)
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            if keys[K_w]:
                self._camera_pos += forward * step
            if keys[K_s]:
                self._camera_pos -= forward * step
            if keys[K_a]:
                self._camera_pos -= right * step
            if keys[K_d]:
                self._camera_pos += right * step
            if keys[K_q]:
                self._camera_pos -= up * step
            if keys[K_e]:
                self._camera_pos += up * step

        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        if pygame.mouse.get_pressed()[0]:
            self._camera_rot[1] += mouse_dx * MOUSE_SENSITIVITY
            self._camera_rot[0] -= mouse_dy * MOUSE_SENSITIVITY
            self._camera_rot[0] = float(np.clip(self._camera_rot[0], -np.pi / 2 + 0.01, np.pi / 2 - 0.01))

    def _upload_camera_frame(self, frame: CameraFrame) -> None:
        try:
            camera_index = frame.camera_id
            if not 0 <= camera_index < len(CAMERA_ROTATIONS):
                return
            self._ensure_gpu_resources(frame.depth)
            if camera_index >= len(self._depth_textures):
                return
            depth_image = np.ascontiguousarray(frame.depth, dtype=np.uint16)
            color_image = np.ascontiguousarray(frame.color, dtype=np.uint8)
            # ModernGL accepts buffer-protocol data. A memoryview avoids making
            # another Python bytes copy for every 1280x720 texture upload.
            self._depth_textures[camera_index].write(memoryview(depth_image))
            self._color_textures[camera_index].write(memoryview(color_image))
            self._active_camera_ids.add(camera_index)
            self._last_render_frames[camera_index] = frame
        except Exception:
            logging.exception("Failed to upload camera %d frame to GPU", frame.camera_id)
            return

    def _restore_render_frames(self) -> None:
        for camera_index in sorted(self._last_render_frames):
            self._upload_camera_frame(self._last_render_frames[camera_index])

    def _ensure_gpu_resources(self, depth_image: np.ndarray) -> None:
        if self._ctx is None or self._program is None:
            return

        frame_height, frame_width = depth_image.shape[:2]
        frame_resolution = (frame_width, frame_height)

        if (
            self._vao is not None
            and self._frame_resolution == frame_resolution
            and len(self._depth_textures) == len(CAMERA_ROTATIONS)
        ):
            return

        self._release_buffers()

        pixel_grid = np.stack(
            np.meshgrid(
                np.arange(frame_width, dtype=np.float32),
                np.arange(frame_height, dtype=np.float32),
                indexing="xy",
            ),
            axis=-1,
        ).reshape(-1, 2)

        self._vbo = self._ctx.buffer(pixel_grid.tobytes())
        self._vao = self._ctx.vertex_array(self._program, [(self._vbo, "2f", "in_pixel")])
        self._vertex_count = frame_width * frame_height
        self._frame_resolution = frame_resolution

        fx = frame_width / (2.0 * np.tan(np.radians(DEPTH_HORIZONTAL_FOV_DEG) / 2.0))
        fy = frame_height / (2.0 * np.tan(np.radians(DEPTH_VERTICAL_FOV_DEG) / 2.0))
        cx = (frame_width - 1) / 2.0
        cy = (frame_height - 1) / 2.0
        self._fx_uniform.value = fx
        self._fy_uniform.value = fy
        self._cx_uniform.value = cx
        self._cy_uniform.value = cy

        for _ in CAMERA_ROTATIONS:
            depth_texture = self._ctx.texture((frame_width, frame_height), 1, dtype="u2")
            depth_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            depth_texture.repeat_x = False
            depth_texture.repeat_y = False
            self._depth_textures.append(depth_texture)

            color_texture = self._ctx.texture((frame_width, frame_height), 3, dtype="f1", alignment=1)
            color_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            color_texture.repeat_x = False
            color_texture.repeat_y = False
            self._color_textures.append(color_texture)

    def _update_matrices(self) -> None:
        if self._program is None:
            return
        view_matrix = self._compute_view_matrix()
        projection_matrix = self._compute_projection_matrix()
        self._view_uniform.write(view_matrix.astype("f4").tobytes())
        self._projection_uniform.write(projection_matrix.astype("f4").tobytes())

    def _compute_view_matrix(self) -> np.ndarray:
        cos_pitch = np.cos(self._camera_rot[0])
        sin_pitch = np.sin(self._camera_rot[0])
        cos_yaw = np.cos(self._camera_rot[1])
        sin_yaw = np.sin(self._camera_rot[1])

        xaxis = np.array([cos_yaw, 0.0, -sin_yaw], dtype=np.float32)
        yaxis = np.array([sin_yaw * sin_pitch, cos_pitch, cos_yaw * sin_pitch], dtype=np.float32)
        zaxis = np.array([sin_yaw * cos_pitch, -sin_pitch, cos_pitch * cos_yaw], dtype=np.float32)

        translation = np.array(
            [
                -np.dot(xaxis, self._camera_pos),
                -np.dot(yaxis, self._camera_pos),
                -np.dot(zaxis, self._camera_pos),
            ],
            dtype=np.float32,
        )

        return np.array(
            [
                [xaxis[0], yaxis[0], zaxis[0], 0.0],
                [xaxis[1], yaxis[1], zaxis[1], 0.0],
                [xaxis[2], yaxis[2], zaxis[2], 0.0],
                [translation[0], translation[1], translation[2], 1.0],
            ],
            dtype=np.float32,
        )

    def _compute_projection_matrix(self) -> np.ndarray:
        fov_scale = 1.0 / np.tan(np.radians(45.0 / 2.0))
        aspect_ratio = WINDOW_SIZE[0] / WINDOW_SIZE[1]
        near, far = 0.05, 500.0
        projection = np.array(
            [
                [fov_scale / aspect_ratio, 0.0, 0.0, 0.0],
                [0.0, fov_scale, 0.0, 0.0],
                [0.0, 0.0, -(far + near) / (far - near), -1.0],
                [0.0, 0.0, -(2.0 * far * near) / (far - near), 0.0],
            ],
            dtype=np.float32,
        )
        return (projection * self._camera_zoom).astype(np.float32)

    def _render_frame(self) -> None:
        if self._ctx is None:
            return
        self._ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        if self._vao is not None:
            self._program["visualization_mode"].value = self._visualization_mode
            for camera_index in sorted(self._active_camera_ids):
                if camera_index >= len(self._depth_textures):
                    continue
                self._depth_textures[camera_index].use(location=0)
                self._color_textures[camera_index].use(location=1)
                self._camera_rotation_uniform.write(self._camera_rotation_matrices[camera_index].astype("f4").tobytes())
                self._camera_offset_uniform.value = tuple(float(v) for v in CAMERA_OFFSETS[camera_index])
                self._max_depth_uniform.value = CAMERA_MAX_DEPTHS[camera_index]
                self._vao.render(moderngl.POINTS, vertices=self._vertex_count)
        view_matrix = self._compute_view_matrix()
        projection_matrix = self._compute_projection_matrix()

        # Render active 3D bounding boxes in one shared update.
        boxes: List[PersonBBox3D] = []
        if self._segmentation_detection_enabled() and self._detection_worker is not None:
            boxes.extend(self._detection_worker.get_detections())
        if self._vlm_enabled:
            for worker in self._vlm_workers:
                boxes.extend(worker.get_detections())
        if self._bbox_renderer is not None:
            self._bbox_renderer.update_boxes(boxes)
            if boxes:
                self._bbox_renderer.render(view_matrix, projection_matrix)
        # Render 3D skeleton lines if pose estimation is enabled
        if self._pose_enabled and self._bbox_renderer is not None and self._pose_worker is not None:
            poses = self._pose_worker.get_poses()
            self._bbox_renderer.update_skeletons(poses)
            if poses:
                self._bbox_renderer.render_skeletons(view_matrix, projection_matrix)
        self._render_control_panel()
        self._render_scene_description()
        pygame.display.flip()

    def _log_metrics(self) -> None:
        now = time.time()
        self._receiver.write_metrics_result_if_due(now)
        if now - self._last_metrics_time < BITRATE_LOG_INTERVAL:
            return

        render_fps = self._receiver.take_average_camera_fps(now)
        bitrate = self._metrics.bitrate_mbps()
        lat = self._receiver.latency_summary()

        if lat["count"] and lat["latency_last_ms"] is not None:
            lat_str = (
                f"RTT ms avg {lat['latency_mean_ms']:.3f} "
                f"p95 {lat['latency_p95_ms']:.3f} "
                f"p99 {lat['latency_p99_ms']:.3f}"
            )
            jit_str = (
                f"Jitter ms avg {lat['jitter_mean_ms']:.3f} "
                f"p95 {lat['jitter_p95_ms']:.3f} "
                f"p99 {lat['jitter_p99_ms']:.3f}"
            )
        else:
            lat_str = "RTT ms n/a"
            jit_str = "Jitter ms n/a"

        logging.info(
            "FPS %.2f | %.2f Mbps | %s | %s | Conn %s",
            render_fps,
            bitrate,
            lat_str,
            jit_str,
            self._receiver.is_connected(),
        )
        self._last_metrics_time = now

    def _cleanup(self) -> None:
        self._stop_scene_description_worker()
        if self._scene_description_overlay is not None:
            with contextlib.suppress(Exception):
                self._scene_description_overlay.release()
            self._scene_description_overlay = None
        if self._control_panel_overlay is not None:
            with contextlib.suppress(Exception):
                self._control_panel_overlay.release()
            self._control_panel_overlay = None
        self._release_buffers()
        if self._bbox_renderer is not None:
            with contextlib.suppress(Exception):
                self._bbox_renderer.release()
            self._bbox_renderer = None
        if self._program is not None:
            with contextlib.suppress(Exception):
                self._program.release()
            self._program = None
        if self._ctx is not None:
            with contextlib.suppress(Exception):
                self._ctx.release()
            self._ctx = None
        pygame.quit()

    def _release_buffers(self) -> None:
        for texture in self._depth_textures:
            with contextlib.suppress(Exception):
                texture.release()
        self._depth_textures = []
        for texture in self._color_textures:
            with contextlib.suppress(Exception):
                texture.release()
        self._color_textures = []

        for buffer_obj in (self._vao, self._vbo):
            if buffer_obj is not None:
                with contextlib.suppress(Exception):
                    buffer_obj.release()
        self._vao = None
        self._vbo = None
        self._frame_resolution = None
        self._vertex_count = 0
        self._active_camera_ids.clear()

    def _toggle_visualization_mode(self) -> None:
        if self._visualization_mode == VISUALIZATION_HEATMAP:
            self._visualization_mode = VISUALIZATION_RGB
        else:
            self._visualization_mode = VISUALIZATION_HEATMAP
        self._update_window_caption()

    def _update_window_caption(self) -> None:
        mode = "RGB" if self._visualization_mode == VISUALIZATION_RGB else "Heatmap"
        det_status = "ALL" if self._object_detection_enabled else (
            "HUMAN" if self._detection_enabled else "OFF"
        )
        pose_status = "ON" if self._pose_enabled else "OFF"
        vlm_status = "ON" if self._vlm_enabled else "OFF"
        scene_desc_status = "ON" if self._scene_description_enabled else "OFF"
        pygame.display.set_caption(
            f"TerraMeta Point Cloud Viewer | {mode} | Detection {det_status} | Pose {pose_status} | VLM {vlm_status} | Scene {scene_desc_status}"
        )

    def _control_panel_rows(self) -> List[Tuple[str, str, str, bool]]:
        return [
            (
                "mode",
                "View color",
                "RGB" if self._visualization_mode == VISUALIZATION_RGB else "Depth",
                self._visualization_mode == VISUALIZATION_RGB,
            ),
            (
                "human",
                "Human boxes",
                "ON" if self._detection_enabled else "OFF",
                self._detection_enabled,
            ),
            (
                "objects",
                "All objects",
                "ON" if self._object_detection_enabled else "OFF",
                self._object_detection_enabled,
            ),
            (
                "pose",
                "Pose points",
                "ON" if self._pose_enabled else "OFF",
                self._pose_enabled,
            ),
            (
                "vlm",
                "VLM boxes",
                "ON" if self._vlm_enabled else "OFF",
                self._vlm_enabled,
            ),
            (
                "scene",
                "Scene text",
                "ON" if self._scene_description_enabled else "OFF",
                self._scene_description_enabled,
            ),
        ]

    def _control_panel_row_rects(self) -> Dict[str, Tuple[int, int, int, int]]:
        row_keys = tuple(row[0] for row in self._control_panel_rows())
        return _control_panel_row_rects(row_keys)

    def _handle_control_panel_click(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        for key, (left, top, width, height) in self._control_panel_row_rects().items():
            if not (left <= x <= left + width and top <= y <= top + height):
                continue
            actions = {
                "mode": self._toggle_visualization_mode,
                "human": self._toggle_detection,
                "objects": self._toggle_object_detection,
                "pose": self._toggle_pose,
                "vlm": self._toggle_vlm,
                "scene": self._toggle_scene_description,
            }
            actions[key]()
            return True
        return False

    def _render_control_panel(self) -> None:
        if self._ctx is None:
            return
        if self._control_panel_overlay is None:
            self._control_panel_overlay = ControlPanelOverlay(self._ctx, WINDOW_SIZE)
        self._control_panel_overlay.render(self._control_panel_rows())

    def _segmentation_detection_enabled(self) -> bool:
        return self._detection_enabled or self._object_detection_enabled

    def _segmentation_class_ids(self) -> Optional[Tuple[int, ...]]:
        if self._object_detection_enabled:
            return ALL_DETECTION_CLASS_IDS
        return PERSON_DETECTION_CLASS_IDS


    def _start_detection_worker(self) -> None:
        """Start and warm the resident background detection worker."""
        if self._detection_worker is not None and self._detection_worker.is_alive():
            self._sync_detection_worker_mode()
            return
        if self._bbox_renderer is None and self._ctx is not None:
            self._bbox_renderer = BBoxRenderer(self._ctx, WINDOW_SIZE)
        try:
            self._detection_worker = DetectionWorker(
                receiver=self._receiver,
                stop_event=self._stop_event,
                detection_cameras=DETECTION_CAMERAS,
                model_path=DETECTION_MODEL_PATH,
                conf_threshold=DETECTION_CONF_THRESHOLD,
                iou_threshold=DETECTION_IOU_THRESHOLD,
                imgsz=DETECTION_IMGSZ,
                device=DETECTION_DEVICE,
                half_precision=DETECTION_HALF_PRECISION,
                batch_size=DETECTION_BATCH_SIZE,
                max_det=DETECTION_MAX_DET,
                retina_masks=DETECTION_RETINA_MASKS,
                erosion_kernel=DETECTION_EROSION_KERNEL,
                subsample_step=DETECTION_SUBSAMPLE_STEP,
                outlier_threshold_m=DETECTION_OUTLIER_THRESHOLD_M,
                min_points=DETECTION_MIN_POINTS,
                camera_rotations=[build_yaw_rotation_matrix(r) for r in CAMERA_ROTATIONS],
                camera_offsets=CAMERA_OFFSETS,
                camera_max_depths=CAMERA_MAX_DEPTHS,
                horizontal_fov_deg=DEPTH_HORIZONTAL_FOV_DEG,
                vertical_fov_deg=DEPTH_VERTICAL_FOV_DEG,
                class_ids=self._segmentation_class_ids(),
                initially_enabled=self._segmentation_detection_enabled(),
            )
            self._detection_worker.start()
            logging.info("Detection model preload started")
        except Exception as exc:
            logging.error("Failed to start detection worker: %s", exc)
            self._detection_worker = None

    def _start_pose_worker(self) -> None:
        """Start and warm the resident background pose estimation worker."""
        if self._pose_worker is not None and self._pose_worker.is_alive():
            self._pose_worker.set_enabled(self._pose_enabled)
            return
        if self._bbox_renderer is None and self._ctx is not None:
            self._bbox_renderer = BBoxRenderer(self._ctx, WINDOW_SIZE)
        try:
            self._pose_worker = PoseDetectionWorker(
                receiver=self._receiver,
                stop_event=self._stop_event,
                detection_cameras=POSE_CAMERAS,
                model_path=POSE_MODEL_PATH,
                conf_threshold=POSE_CONF_THRESHOLD,
                iou_threshold=POSE_IOU_THRESHOLD,
                imgsz=POSE_IMGSZ,
                device=POSE_DEVICE,
                half_precision=POSE_HALF_PRECISION,
                batch_size=POSE_BATCH_SIZE,
                max_det=POSE_MAX_DET,
                keypoint_conf_threshold=POSE_KEYPOINT_CONF_THRESHOLD,
                min_valid_keypoints=POSE_MIN_VALID_KEYPOINTS,
                min_valid_connections=POSE_MIN_VALID_CONNECTIONS,
                camera_rotations=[build_yaw_rotation_matrix(r) for r in CAMERA_ROTATIONS],
                camera_offsets=CAMERA_OFFSETS,
                camera_max_depths=CAMERA_MAX_DEPTHS,
                horizontal_fov_deg=DEPTH_HORIZONTAL_FOV_DEG,
                vertical_fov_deg=DEPTH_VERTICAL_FOV_DEG,
                initially_enabled=self._pose_enabled,
            )
            self._pose_worker.start()
            logging.info("Pose estimation model preload started")
        except Exception as exc:
            logging.error("Failed to start pose worker: %s", exc)
            self._pose_worker = None

    def _start_vlm_workers(self) -> None:
        """Start sharded low-rate VLM semantic localization workers."""
        live_workers = [worker for worker in self._vlm_workers if worker.is_alive()]
        if live_workers:
            self._vlm_workers = live_workers
            for worker in self._vlm_workers:
                worker.set_enabled(self._vlm_enabled)
            return
        if self._bbox_renderer is None and self._ctx is not None:
            self._bbox_renderer = BBoxRenderer(self._ctx, WINDOW_SIZE)
        camera_groups = _split_vlm_camera_groups(VLM_CAMERAS, VLM_REQUEST_WORKERS)
        request_rate_hz = VLM_REQUEST_RATE_HZ / max(1, len(camera_groups))
        self._vlm_workers = []
        for cameras in camera_groups:
            try:
                worker = VLMDetectionWorker(
                    receiver=self._receiver,
                    stop_event=self._stop_event,
                    detection_cameras=cameras,
                    base_url=VLM_BASE_URL,
                    api_key=VLM_API_KEY,
                    model=VLM_MODEL,
                    request_rate_hz=request_rate_hz,
                    timeout_s=VLM_REQUEST_TIMEOUT_S,
                    max_tokens=VLM_MAX_TOKENS,
                    jpeg_quality=VLM_JPEG_QUALITY,
                    image_max_side_px=VLM_IMAGE_MAX_SIDE_PX,
                    max_objects=VLM_MAX_OBJECTS,
                    conf_threshold=VLM_CONF_THRESHOLD,
                    min_box_side_px=VLM_MIN_BOX_SIDE_PX,
                    stale_ttl_s=VLM_STALE_TTL_S,
                    erosion_kernel=VLM_EROSION_KERNEL,
                    subsample_step=VLM_SUBSAMPLE_STEP,
                    outlier_threshold_m=VLM_OUTLIER_THRESHOLD_M,
                    min_points=VLM_MIN_POINTS,
                    camera_rotations=[build_yaw_rotation_matrix(r) for r in CAMERA_ROTATIONS],
                    camera_offsets=CAMERA_OFFSETS,
                    camera_max_depths=CAMERA_MAX_DEPTHS,
                    horizontal_fov_deg=DEPTH_HORIZONTAL_FOV_DEG,
                    vertical_fov_deg=DEPTH_VERTICAL_FOV_DEG,
                    initially_enabled=self._vlm_enabled,
                )
                worker.start()
                self._vlm_workers.append(worker)
            except Exception as exc:
                logging.error("Failed to start VLM worker for cameras %s: %s", cameras, exc)
        if self._vlm_workers:
            logging.info(
                "VLM semantic localization workers ready: %s",
                camera_groups,
            )

    def _stop_detection_worker(self) -> None:
        """Stop the detection worker thread."""
        worker = self._detection_worker
        if worker is None:
            return
        worker.request_stop()
        worker.join(timeout=DETECTION_WORKER_JOIN_TIMEOUT_S)
        if worker.is_alive():
            logging.warning(
                "Detection worker did not stop within %.1f seconds; continuing shutdown",
                DETECTION_WORKER_JOIN_TIMEOUT_S,
            )
        self._detection_worker = None

    def _stop_pose_worker(self) -> None:
        """Stop the pose estimation worker thread."""
        worker = self._pose_worker
        if worker is None:
            return
        worker.request_stop()
        worker.join(timeout=POSE_WORKER_JOIN_TIMEOUT_S)
        if worker.is_alive():
            logging.warning(
                "Pose worker did not stop within %.1f seconds; continuing shutdown",
                POSE_WORKER_JOIN_TIMEOUT_S,
            )
        self._pose_worker = None

    def _stop_vlm_workers(self) -> None:
        """Stop the VLM semantic localization worker threads."""
        for worker in self._vlm_workers:
            worker.request_stop()
        for worker in self._vlm_workers:
            worker.join(timeout=VLM_WORKER_JOIN_TIMEOUT_S)
            if worker.is_alive():
                logging.warning(
                    "VLM worker for cameras %s did not stop within %.1f seconds; continuing shutdown",
                    worker._detection_cameras,
                    VLM_WORKER_JOIN_TIMEOUT_S,
                )
        self._vlm_workers = []

    def _toggle_detection(self) -> None:
        """Toggle human detection on/off."""
        self._detection_enabled = not self._detection_enabled
        if self._detection_enabled:
            self._object_detection_enabled = False
        self._sync_detection_worker_mode()
        self._update_window_caption()
        logging.info("Human detection %s", "enabled" if self._detection_enabled else "disabled")

    def _toggle_object_detection(self) -> None:
        """Toggle all-class YOLO segmentation detection on/off."""
        self._object_detection_enabled = not self._object_detection_enabled
        if self._object_detection_enabled:
            self._detection_enabled = False
        self._sync_detection_worker_mode()
        self._update_window_caption()
        logging.info(
            "All-object detection %s",
            "enabled" if self._object_detection_enabled else "disabled",
        )

    def _sync_detection_worker_mode(self) -> None:
        if self._detection_worker is None or not self._detection_worker.is_alive():
            self._start_detection_worker()
        if self._detection_worker is not None:
            self._detection_worker.set_class_ids(self._segmentation_class_ids())
            self._detection_worker.set_enabled(self._segmentation_detection_enabled())

    def _toggle_pose(self) -> None:
        """Toggle pose estimation on/off."""
        self._pose_enabled = not self._pose_enabled
        if self._pose_worker is None or not self._pose_worker.is_alive():
            self._start_pose_worker()
        if self._pose_worker is not None:
            self._pose_worker.set_enabled(self._pose_enabled)
        self._update_window_caption()
        logging.info("Pose estimation %s", "enabled" if self._pose_enabled else "disabled")

    def _toggle_vlm(self) -> None:
        """Toggle VLM semantic object localization on/off."""
        self._vlm_enabled = not self._vlm_enabled
        if not any(worker.is_alive() for worker in self._vlm_workers):
            self._start_vlm_workers()
        for worker in self._vlm_workers:
            worker.set_enabled(self._vlm_enabled)
        self._update_window_caption()
        logging.info("VLM object localization %s", "enabled" if self._vlm_enabled else "disabled")

    def _render_scene_description(self) -> None:
        """Render VLM scene descriptions as text overlay."""
        if not self._scene_description_enabled or self._scene_description_worker is None:
            return
        if self._ctx is None:
            return
        if self._scene_description_overlay is None:
            self._scene_description_overlay = SceneDescriptionOverlay(
                self._ctx,
                WINDOW_SIZE,
                SCENE_DESCRIPTION_CAMERAS,
            )
        descriptions = self._scene_description_worker.get_descriptions()
        self._scene_description_overlay.render(descriptions)

    def _start_scene_description_worker(self) -> None:
        """Start the scene description worker."""
        if self._scene_description_worker is not None and self._scene_description_worker.is_alive():
            self._scene_description_worker.set_enabled(self._scene_description_enabled)
            return
        
        try:
            self._scene_description_worker = SceneDescriptionWorker(
                receiver=self._receiver,
                stop_event=self._stop_event,
                description_cameras=SCENE_DESCRIPTION_CAMERAS,
                base_url=VLM_BASE_URL,
                api_key=VLM_API_KEY,
                model=VLM_MODEL,
                refresh_interval_s=SCENE_DESCRIPTION_REFRESH_INTERVAL_S,
                timeout_s=VLM_REQUEST_TIMEOUT_S,
                max_tokens=SCENE_DESCRIPTION_MAX_TOKENS,
                jpeg_quality=SCENE_DESCRIPTION_JPEG_QUALITY,
                image_max_side_px=SCENE_DESCRIPTION_IMAGE_MAX_SIDE_PX,
                max_workers=SCENE_DESCRIPTION_WORKERS,
                initially_enabled=self._scene_description_enabled,
            )
            self._scene_description_worker.start()
        except Exception as exc:
            logging.error("Failed to start scene description worker: %s", exc)
            self._scene_description_worker = None

    def _stop_scene_description_worker(self) -> None:
        """Stop the scene description worker."""
        if self._scene_description_worker is None:
            return
        self._scene_description_worker.request_stop()
        self._scene_description_worker.join(timeout=2.0)
        if self._scene_description_worker.is_alive():
            logging.warning("Scene description worker did not stop cleanly")
        self._scene_description_worker = None

    def _toggle_scene_description(self) -> None:
        """Toggle scene description on/off."""
        self._scene_description_enabled = not self._scene_description_enabled
        if not any(worker.is_alive() for worker in [self._scene_description_worker] if worker):
            self._start_scene_description_worker()
        if self._scene_description_worker is not None:
            self._scene_description_worker.set_enabled(self._scene_description_enabled)
        self._update_window_caption()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for logger_name in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def main() -> None:
    configure_logging()
    probe_cpu_core = reserve_latency_probe_cpu()
    stop_event = threading.Event()
    metrics = Metrics()
    receiver = FrameReceiver(SERVER_HOST, SERVER_PORT, metrics, stop_event)
    receiver.start()
    probe_receiver = LatencyProbeReceiver(
        SERVER_HOST,
        LATENCY_PROBE_PORT,
        receiver,
        stop_event,
        cpu_core=probe_cpu_core,
    )
    probe_receiver.start()

    app = PointCloudApp(receiver, metrics, stop_event)

    try:
        app.run()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, shutting down")
        receiver.stop()
    except Exception:
        logging.exception("Viewer crashed unexpectedly")
        receiver.stop()
    finally:
        receiver.stop()
        probe_receiver.stop()
        probe_receiver.join(timeout=5)
        if probe_receiver.is_alive():
            logging.warning("Latency probe receiver thread did not shut down cleanly")
        receiver.join(timeout=5)
        if receiver.is_alive():
            logging.warning("Receiver thread did not shut down cleanly")


if __name__ == "__main__":
    main()
